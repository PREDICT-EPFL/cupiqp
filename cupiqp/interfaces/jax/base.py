"""Machinery shared by the JAX solvers.

Read top to bottom:

1. ``JaxSolverBase`` is the base of the public solvers. It holds the dtype and
   the settings. A subclass implements ``solve()``, which checks its own
   arguments and hands the arrays to ``JaxSolverBase._solve()``, plus three
   small methods that say how its backend differs: the rank of each input,
   and how the inputs become the core ``setup()`` / ``update()`` arguments.
2. ``_solve()`` turns a call into a *signature*: everything that is static
   under ``jax.jit`` (solver class, dtype, forward settings, the inputs
   given with their shapes and batch layout, the backend structure such as
   a sparsity pattern). Each signature gets one ``CompiledSolve`` object, cached
   for the process.
3. ``CompiledSolve`` owns one XLA custom call (``wp.jax_callable``) whose
   callback runs one core solver, and one ``jax.jit``-ed
   function with a ``vmap`` rule that turns a mapped call into one batched
   call.

The custom call is registered as side-effect free, so XLA may drop,
duplicate or reorder calls. That is sound because every call loads all of
its inputs with ``update()``, so its result depends on nothing else;
settings that would break this are rejected.

Stream order: the callback runs with the XLA stream current. The core
solver's ``update()`` and ``solve()`` make its own stream wait for the XLA
stream, the results are copied into the XLA output buffers on the solver
stream, and the XLA stream then waits for it. The interface never waits on
the host.
"""

from __future__ import annotations

import gc
import inspect
import threading
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import jax
import numpy as np
import warp as wp
from jax.custom_batching import custom_vmap

from ...settings import Settings
from .types import QPSolution

#: Solution blocks, with the input that makes each present and its width.
SOLUTION_BLOCKS = {
    "x": (None, "n"),
    "y": ("b", "p"),
    "z_l": ("h_l", "m"), "z_u": ("h_u", "m"), "z_bl": ("x_l", "n"), "z_bu": ("x_u", "n"),
    "s_l": ("h_l", "m"), "s_u": ("h_u", "m"), "s_bl": ("x_l", "n"), "s_bu": ("x_u", "n"),
}
DIAGNOSTICS = ("status", "iterations", "primal_res", "dual_res")

#: Settings that shape a forward solve: they are part of the signature.
FORWARD_SETTINGS = (
    "dtype", "rho_init", "delta_init", "eps_abs", "eps_rel", "check_duality_gap",
    "eps_duality_gap_abs", "eps_duality_gap_rel", "infeasibility_threshold",
    "reg_lower_limit", "reg_finetune_lower_limit", "reg_finetune_primal_update_threshold",
    "reg_finetune_dual_update_threshold", "max_iter", "max_factor_retires",
    "preconditioner_scale_cost", "preconditioner_iter", "tau",
    "iterative_refinement_always_enabled", "iterative_refinement_eps_abs",
    "iterative_refinement_eps_rel", "iterative_refinement_max_iter",
    "iterative_refinement_min_improvement_rate",
    "iterative_refinement_static_regularization_eps",
    "iterative_refinement_static_regularization_rel",
    "use_deterministic_mode_for_cudss", "enable_cuda_graph",
)
#: Settings the forward solve does not read (backward only, timings). They
#: are not part of the signature; ``enable_grad`` is always off here.
IGNORED_SETTINGS = (
    "enable_grad", "gradient_smoothing", "gradient_smoothing_mu", "gradient_smoothing_tol",
    "gradient_smoothing_max_iter", "gradient_on_failure", "compute_timings",
)
#: Settings that must keep their default because they would make a call
#: depend on earlier calls (reused scaling) or have a side effect (printing).
REJECTED_SETTINGS = {
    "preconditioner_reuse_on_update": "it reuses the scaling of an earlier call, so a "
                                      "result would depend on which problems were solved before",
    "verbose": "printing is a side effect, and XLA may skip or repeat a solve whose "
               "results are not used",
}


def settings_snapshot(settings: Settings) -> tuple:
    """The forward settings as a hashable tuple; rejects unsupported ones."""
    for name, reason in REJECTED_SETTINGS.items():
        if getattr(settings, name):
            raise ValueError(
                f"settings.{name} = True is not supported by the JAX interface: {reason}."
            )
    return tuple((name, getattr(settings, name)) for name in FORWARD_SETTINGS)


def check_constraint_blocks(has_A: bool, has_b: bool, has_G: bool, has_h_l: bool, has_h_u: bool) -> None:
    """The constraint blocks a call gives are consistent. Each backend calls
    this from ``solve()``, where it knows which of its inputs are which."""
    if has_A != has_b:
        raise ValueError("A and b must be given together.")
    if has_G and not (has_h_l or has_h_u):
        raise ValueError("G requires at least one of h_l and h_u.")
    if not has_G and (has_h_l or has_h_u):
        raise ValueError("h_l and h_u require G.")


# ---------------------------------------------------------------------------
# 1. The solver base class
# ---------------------------------------------------------------------------


class JaxSolverBase:
    """Base of the JAX solvers: the floating dtype and the settings.

    A subclass sets ``_core_class`` (the core solver it runs), implements
    ``solve()`` and ``_input_rank()``, and overrides ``_setup_arguments()``
    / ``_update_arguments()`` when its core takes the inputs in another form
    than the FFI passes them.
    """

    _core_class: type

    def __init__(self, dtype: Any = jax.numpy.float64, settings: Settings | None = None) -> None:
        if np.dtype(dtype) == np.float32:
            self._dtype = wp.float32
        elif np.dtype(dtype) == np.float64:
            self._dtype = wp.float64
        else:
            raise TypeError(f"dtype must be float32 or float64; got {dtype!r}.")
        if self._dtype is wp.float64 and not jax.config.jax_enable_x64:
            raise RuntimeError(
                f"{type(self).__name__} with dtype float64 requires 64-bit JAX arrays: "
                "call jax.config.update('jax_enable_x64', True) before creating any "
                "array, or use dtype=jnp.float32."
            )
        self.settings = Settings.for_dtype(self._dtype) if settings is None else settings

    @property
    def dtype(self) -> np.dtype:
        """The floating dtype of every argument and result."""
        return np.dtype(wp.dtype_to_numpy(self._dtype))

    @property
    def settings(self) -> Settings:
        """Solver configuration (a ``Settings`` dataclass), read at every
        call of ``solve()``. Change its fields before a solve, e.g.
        ``solver.settings.max_iter = 50``; a function compiled by ``jax.jit``
        keeps the settings it was traced with."""
        return self._settings

    @settings.setter
    def settings(self, value: Settings) -> None:
        if not isinstance(value, Settings):
            raise TypeError(f"settings must be a cupiqp.Settings; got {type(value).__name__}.")
        if value.dtype is not self._dtype:
            raise ValueError(
                f"settings.dtype is {value.dtype.__name__} but the solver dtype is "
                f"{self._dtype.__name__}; build the settings with Settings.for_dtype()."
            )
        self._settings = value

    def __repr__(self) -> str:
        return f"{type(self).__name__}(dtype={self.dtype})"

    # -- what a backend defines ------------------------------------------

    def _input_rank(self, name: str) -> int:
        """Rank of the input ``name`` for one problem (a batch axis adds one)."""
        raise NotImplementedError

    def _setup_arguments(self, inputs: Mapping[str, wp.array]) -> dict[str, Any]:
        """The core ``setup()`` arguments for the FFI inputs."""
        return dict(inputs)

    def _update_arguments(self, inputs: Mapping[str, wp.array]) -> dict[str, Any]:
        """The core ``update()`` arguments for the FFI inputs."""
        return dict(inputs)

    def _solve_from_ffi_inputs(self, inputs: dict[str, Any]) -> QPSolution:
        """``solve()`` called with the FFI inputs as keyword arguments. It
        exists only for the ``vmap`` rule, which re-enters ``solve()`` with
        batched arrays; a backend whose ``solve()`` takes the inputs in
        another form (multistage pairs) overrides it to rebuild that form."""
        return self.solve(**inputs)

    # -- the shared solve path -------------------------------------------

    def _solve(
        self,
        given: dict[str, Any],
        n: int,
        p: int,
        m: int,
        matrix_shapes: Mapping[str, tuple[int, ...]],
        structure: Any = None,
    ) -> QPSolution:
        """Check the arrays of a call, find or build its compiled function,
        and run it.

        ``given`` maps the inputs the user passed, in the backend's order,
        to arrays; ``matrix_shapes`` is the single-problem shape of each
        matrix input; ``structure`` is a hashable description of what the
        solver fixed at construction (a sparsity pattern), or ``None``. The
        backend has already checked that its constraint blocks are
        consistent (:func:`check_constraint_blocks`).
        """
        if n < 1:
            raise ValueError("the problem must have at least one variable.")

        expected = dict(matrix_shapes)
        expected.update({"c": (n,), "b": (p,), "h_u": (m,), "h_l": (m,), "x_u": (n,), "x_l": (n,)})
        dtype = np.dtype(wp.dtype_to_numpy(self._dtype))
        layout = []                       # (name, single-problem shape, batched) per input
        batch_sizes = set()
        for name, value in given.items():
            shape = getattr(value, "shape", None)
            if shape is None or getattr(value, "dtype", None) is None:
                raise TypeError(
                    f"{name} must be an array (a JAX array, or a NumPy array that JAX "
                    f"transfers to the device); got {type(value).__name__}."
                )
            if np.dtype(value.dtype) != dtype:
                raise TypeError(
                    f"{name} has dtype {np.dtype(value.dtype)}; this solver uses {dtype}. "
                    f"The solver never casts: convert {name} with .astype() first."
                )
            shape = tuple(int(s) for s in shape)
            rank = self._input_rank(name)
            if len(shape) == rank + 2:
                raise ValueError(
                    f"{name} has {len(shape)} dimensions: it carries two batch axes. "
                    "Nested batching (jax.vmap over an argument that already has a "
                    "batch axis) is not supported; flatten the batch axes first."
                )
            if len(shape) not in (rank, rank + 1):
                kind = {1: "vector", 2: "matrix", 3: "block array"}[rank]
                raise ValueError(
                    f"{name} must be a {kind}: {rank}-D for one problem, or {rank + 1}-D "
                    f"with a leading batch axis; got {len(shape)}-D."
                )
            batched = len(shape) == rank + 1
            if batched:
                batch_sizes.add(shape[0])
            if shape[int(batched):] != expected[name]:
                want = ("(B, " if batched else "(") + ", ".join(str(s) for s in expected[name]) + ")"
                raise ValueError(
                    f"{name} has shape {shape}; expected {want} with n={n}, p={p}, m={m}."
                )
            layout.append((name, shape[int(batched):], batched))
        if len(batch_sizes) > 1:
            raise ValueError(f"the batched arguments disagree on the batch size ({sorted(batch_sizes)}).")
        batch_size = batch_sizes.pop() if batch_sizes else 1
        if batch_size < 1:
            raise ValueError("the batch size must be at least 1.")

        signature = (type(self), self._dtype, settings_snapshot(self._settings),
                     structure, tuple(layout), batch_size)
        compiled = CACHE.get(signature)
        if compiled is None:
            compiled = CACHE[signature] = CompiledSolve(self, signature, n, p, m)
        return compiled.function(*given.values())


# ---------------------------------------------------------------------------
# 2. One compiled solve per signature
# ---------------------------------------------------------------------------

CACHE: dict[tuple, "CompiledSolve"] = {}


@dataclass
class Workspace:
    """A core solver with the stream it borrows and the event that orders
    the XLA stream after it."""

    stream: wp.Stream
    event: wp.Event
    solver: Any
    is_setup: bool = False


class CompiledSolve:
    """The compiled solve of one signature: its XLA custom call, the core
    solver behind it, and the jitted JAX function.

    It keeps the public solver instance that first produced the signature,
    for its class and its backend methods (``_core_class``, ``_input_rank``,
    ``_setup_arguments``, ...). Those depend only on data fixed at that
    instance's construction, which the signature identifies, so equivalent
    instances share this object safely; the instance's mutable settings are
    never read here, the signature carries their snapshot.
    """

    def __init__(self, solver: JaxSolverBase, signature: tuple, n: int, p: int, m: int) -> None:
        _, dtype, settings, _, layout, batch_size = signature
        self.solver = solver
        self.dtype = dtype
        self.settings = settings
        self.inputs = [(name, rank_shape, batched) for name, rank_shape, batched in layout]
        self.batch_size = batch_size
        self.unbatched = not any(batched for _, _, batched in layout)
        given = {name for name, _, _ in layout}
        self.outputs = {}                 # name -> batched shape
        for name, (gate, dim) in SOLUTION_BLOCKS.items():
            width = {"n": n, "p": p, "m": m}[dim] if gate is None or gate in given else 0
            self.outputs[name] = (batch_size, width)
        for name in DIAGNOSTICS:
            self.outputs[name] = (batch_size,)
        self.workspace: Workspace | None = None  # created by the first call
        self.execution_lock = threading.Lock()   # one call at a time on the mutable workspace
        self.function = self._build_function()

    # -- the XLA custom call ----------------------------------------------

    def _build_callable(self):
        """Register the custom call: a Python callback with one annotated
        parameter per input and per output, as ``wp.jax_callable`` needs."""
        annotations = {}
        for name, shape, batched in self.inputs:
            annotations[name] = wp.array(dtype=self.dtype, ndim=len(shape) + int(batched))
        for name, shape in self.outputs.items():
            dtype = wp.int32 if name in ("status", "iterations") else self.dtype
            annotations[name] = wp.array(dtype=dtype, ndim=len(shape))
        num_inputs = len(self.inputs)
        names = list(annotations)

        def callback(*arrays):
            self._run(dict(zip(names[:num_inputs], arrays[:num_inputs])),
                      dict(zip(names[num_inputs:], arrays[num_inputs:])))

        callback.__signature__ = inspect.Signature(
            [inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=annotations[name])
             for name in names],
            return_annotation=None,
        )
        callback.__annotations__ = {**annotations, "return": None}
        callback.__name__ = f"cupiqp_{type(self.solver).__name__}"
        callback.__module__ = __name__
        # XLA graph capture stays off: the solver runs on its own stream,
        # which a capture of the XLA stream would not record, and its first
        # call allocates and captures its own CUDA graphs during setup().
        return wp.jax_callable(
            callback, num_outputs=len(self.outputs), graph_mode=wp.JaxCallableGraphMode.NONE,
            vmap_method=None, output_dims=self.outputs, has_side_effect=False,
        )

    def _run(self, inputs: dict[str, wp.array], outputs: dict[str, wp.array]) -> None:
        """The callback body: solve ``inputs`` and write ``outputs``, both
        non-owning views of XLA buffers that are valid during the call."""
        with self.execution_lock:
            xla_stream = wp.get_stream()
            if self.workspace is None:
                stream = wp.Stream(xla_stream.device)
                core = self.solver._core_class(dtype=self.dtype, stream=stream)
                values = dict(self.settings)
                core.settings = Settings.for_dtype(values.pop("dtype"))
                for name, value in values.items():
                    setattr(core.settings, name, value)
                self.workspace = Workspace(stream, wp.Event(xla_stream.device), core)
            workspace = self.workspace
            core = workspace.solver
            # The first call also goes through update(), so that every call
            # takes the same path: a solve right after setup() differs from
            # one after update() by rounding in the core.
            if not workspace.is_setup:
                try:
                    core.setup(**self.solver._setup_arguments(inputs))
                except Exception:
                    self.workspace = None
                    close_workspace(workspace)
                    raise
                workspace.is_setup = True
            core.update(**self.solver._update_arguments(inputs))
            core.solve()
            result = core.result
            with wp.ScopedStream(workspace.stream, sync_enter=False):
                for name, destination in outputs.items():
                    if name == "status":
                        source = result.info.status
                    elif name == "iterations":
                        source = result.info.iter
                    elif name in ("primal_res", "dual_res"):
                        source = getattr(result.info, name)
                    else:
                        source = getattr(result, name)
                    wp.copy(destination, source)
            # XLA continues only once the outputs are written.
            xla_stream.wait_stream(workspace.stream, event=workspace.event)

    # -- the jitted function and its vmap rule -----------------------------

    def _build_function(self) -> Callable[..., QPSolution]:
        ffi = self._build_callable()
        names = list(self.outputs)

        @custom_vmap
        def solve(*arrays):
            values = dict(zip(names, ffi(*arrays)))
            if self.unbatched:
                values = {name: value[0] for name, value in values.items()}
            return QPSolution(**values)

        @solve.def_vmap
        def solve_vmap(axis_size, in_batched, *arrays):
            # JAX moves each mapped axis to the front. Mapped inputs now carry
            # a batch axis, the others are shared: that is a batched call of
            # the same solver, so re-enter solve() with these arrays.
            if not any(in_batched):
                return solve(*arrays), QPSolution(*([False] * len(QPSolution._fields)))
            if not self.unbatched:
                raise NotImplementedError(
                    "jax.vmap over a solve() call whose arguments already carry a "
                    "batch axis (nested batching) is not supported; pass the batch "
                    "to a single call instead."
                )
            # Re-enter through the public solve(): it rebuilds the signature
            # from the batched shapes. _solve_from_ffi_inputs() only maps the
            # FFI input names back to the solve() arguments.
            solution = self.solver._solve_from_ffi_inputs(dict(zip((name for name, _, _ in self.inputs), arrays)))
            return solution, QPSolution(*([True] * len(QPSolution._fields)))

        return jax.jit(solve)


def close_workspace(workspace: Workspace) -> None:
    """Wait for the solver stream, then free the core solver before the
    stream it borrows (its library handles and graphs are bound to it)."""
    wp.synchronize_stream(workspace.stream)
    workspace.solver = None
    gc.collect()


def release_solver_memory() -> None:
    """Wait for pending solves, then release every core solver the JAX
    interface keeps, and its GPU memory. Compiled functions stay valid;
    their next call creates its solver again (a slow first call)."""
    for compiled in CACHE.values():
        with compiled.execution_lock:
            if compiled.workspace is not None:
                close_workspace(compiled.workspace)
                compiled.workspace = None


def cache_info() -> dict[str, int]:
    """``{"signatures": ..., "live_solvers": ...}``: the problem signatures
    seen in this process (each keeps a compiled function and an XLA custom
    call until the process exits) and the core solvers currently holding
    GPU memory."""
    return {
        "signatures": len(CACHE),
        "live_solvers": sum(compiled.workspace is not None for compiled in CACHE.values()),
    }
