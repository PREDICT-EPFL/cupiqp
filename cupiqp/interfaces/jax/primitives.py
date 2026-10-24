"""FFI callbacks of the JAX interface: one core solver behind one pure XLA
custom call per problem signature.

A *signature* is everything XLA fixes at trace time: the solver dtype, the
problem dimensions, which data blocks are passed and which of them carry a
batch axis, the batch size, and the solver settings. For every distinct
signature this module registers one ``warp.jax_callable`` whose Python
callback owns one core solver per CUDA device. The first call of a
signature sets the solver up; every later call loads all of its inputs with
``update()`` and solves again, so the result of a call depends on nothing
but its inputs and XLA may freely drop, duplicate or reorder calls
(``has_side_effect=False``).

Stream ordering: XLA runs the callback with its own stream current. The core
solver keeps its own stream, waits for the XLA stream on entry (this is what
the solver's public methods already do), copies its results into the
XLA-owned output buffers on its stream, and the XLA stream then waits for it.
Every read of an XLA input and every write of an XLA output happens between
these two waits, so XLA may reuse the buffers as soon as the call completes
on its stream. No host synchronization is added by the interface.
"""

from __future__ import annotations

import gc
import threading
from dataclasses import dataclass, fields as dataclass_fields
from typing import Any, Final, Mapping

import warp as wp
from warp.jax_experimental.ffi import JaxCallableGraphMode, jax_callable

from ...dense.dense_solver import DenseSolver
from ...settings import Settings


#: The nine data arguments of a dense QP, in the order of ``DenseSolver.setup()``.
DATA_FIELDS: Final[tuple[str, ...]] = ("P", "c", "A", "b", "G", "h_u", "h_l", "x_u", "x_l")
#: Rank of each data argument for a single problem (a batch axis adds one).
DATA_NDIM: Final[dict[str, int]] = {
    "P": 2, "c": 1, "A": 2, "b": 1, "G": 2, "h_u": 1, "h_l": 1, "x_u": 1, "x_l": 1,
}
#: The primal, dual and slack variables of a solution, in the public order.
RESULT_FIELDS: Final[tuple[str, ...]] = (
    "x", "y", "z_l", "z_u", "z_bl", "z_bu", "s_l", "s_u", "s_bl", "s_bu"
)
#: The per-problem diagnostics exported with every solution.
INFO_FIELDS: Final[tuple[str, ...]] = ("status", "iterations", "primal_res", "dual_res")
#: Which data argument makes each result block present, and the block width.
_RESULT_SOURCE: Final[dict[str, tuple[str | None, str]]] = {
    "x": (None, "n"),
    "y": ("A", "p"),
    "z_l": ("h_l", "m"), "s_l": ("h_l", "m"),
    "z_u": ("h_u", "m"), "s_u": ("h_u", "m"),
    "z_bl": ("x_l", "n"), "s_bl": ("x_l", "n"),
    "z_bu": ("x_u", "n"), "s_bu": ("x_u", "n"),
}


def settings_key(settings: Settings) -> tuple:
    """Hashable snapshot of ``settings``: a tuple of ``(field, value)`` pairs.

    The snapshot is part of the signature, so a solver created from it uses
    exactly these values. ``enable_grad`` is always recorded as ``False``:
    the JAX operators do not differentiate yet, and gradient storage would
    only cost memory.
    """
    items = []
    for field in dataclass_fields(Settings):
        value = getattr(settings, field.name)
        if field.name == "enable_grad":
            value = False
        items.append((field.name, value))
    return tuple(items)


def settings_from_key(key: tuple) -> Settings:
    """The ``Settings`` recorded by :func:`settings_key`."""
    return Settings(**dict(key))


@dataclass(frozen=True)
class DenseSignature:
    """Everything that is static about one dense QP call under ``jax.jit``.

    ``fields`` lists the data arguments that are passed, in the order of
    ``DATA_FIELDS``; ``batched`` says for each of them whether it carries a
    leading batch axis. ``batch_size`` is ``1`` when nothing is batched, in
    which case the call is *unbatched* and its outputs have no batch axis.
    """

    dtype: Any                     # wp.float32 or wp.float64
    n: int
    m: int
    p: int
    fields: tuple[str, ...]
    batched: tuple[bool, ...]
    batch_size: int
    settings: tuple

    @property
    def unbatched(self) -> bool:
        return not any(self.batched)

    def has(self, name: str) -> bool:
        return name in self.fields

    def width(self, name: str) -> int:
        """Width of the result block ``name``: zero when its block is absent."""
        source, dim = _RESULT_SOURCE[name]
        if source is not None and not self.has(source):
            return 0
        return getattr(self, dim)

    def output_shapes(self) -> dict[str, tuple[int, ...]]:
        """Batched shape of every output of the FFI call: the present result
        blocks (absent ones are not part of the call) and the info fields."""
        B = self.batch_size
        shapes = {}
        for name in RESULT_FIELDS:
            if self.width(name) > 0:
                shapes[name] = (B, self.width(name))
        for name in INFO_FIELDS:
            shapes[name] = (B,)
        return shapes

    def output_dtype(self, name: str):
        return wp.int32 if name in ("status", "iterations") else self.dtype


class _DeviceSolver:
    """A core solver with its own stream, on one CUDA device."""

    def __init__(self, signature: DenseSignature, device: Any) -> None:
        self.device = device
        # The solver borrows this stream; keep it alive as long as the solver.
        self.stream = wp.Stream(device)
        self.exit_event = wp.Event(device)
        self.solver = DenseSolver(dtype=signature.dtype, stream=self.stream)
        self.solver.settings = settings_from_key(signature.settings)
        self.is_setup = False

    def close(self) -> None:
        """Release the solver before its stream (library handles and graphs
        are bound to the stream)."""
        self.solver = None
        gc.collect()
        self.stream = None
        self.exit_event = None


class _Workspace:
    """The solvers behind one signature, one per device, and the lock that
    serializes their use."""

    def __init__(self, signature: DenseSignature) -> None:
        self.signature = signature
        self.lock = threading.Lock()
        self._solvers: dict[int, _DeviceSolver] = {}

    def run(self, inputs: Mapping[str, wp.array], outputs: Mapping[str, wp.array]) -> None:
        """Solve the problems in ``inputs`` and write the results to
        ``outputs``. Called by the FFI callback with Warp's current stream set
        to the XLA stream; ``inputs`` and ``outputs`` are non-owning views of
        XLA buffers that are valid only for the duration of the call."""
        with self.lock:
            xla_stream = wp.get_stream()
            entry = self._solvers.get(xla_stream.device.ordinal)
            if entry is None:
                entry = _DeviceSolver(self.signature, xla_stream.device)
                self._solvers[xla_stream.device.ordinal] = entry
            solver = entry.solver
            # setup()/update() make the solver stream wait for the current
            # (XLA) stream, then copy the inputs into solver-owned storage.
            # The first call also goes through update() so that every solve
            # takes the same path: a solve right after setup() can differ
            # from one after update() by rounding, and a pure function must
            # give the same answer on every call.
            if not entry.is_setup:
                try:
                    solver.setup(**inputs)
                except Exception:
                    # A half-initialized solver cannot be set up again:
                    # drop it so that the next call starts from scratch.
                    del self._solvers[xla_stream.device.ordinal]
                    entry.close()
                    raise
                entry.is_setup = True
            solver.update(**inputs)
            solver.solve()
            result = solver.result
            with wp.ScopedStream(entry.stream, sync_enter=False):
                for name, destination in outputs.items():
                    wp.copy(destination, _result_array(result, name))
            # The XLA stream resumes only after the outputs are written.
            xla_stream.wait_stream(entry.stream, event=entry.exit_event)

    @property
    def has_solvers(self) -> bool:
        return bool(self._solvers)

    def close(self) -> None:
        with self.lock:
            for entry in self._solvers.values():
                entry.close()
            self._solvers.clear()


def _result_array(result, name: str) -> wp.array:
    """The ``(B, ...)`` device array of the output ``name`` in a core result."""
    if name == "status":
        return result.info.status
    if name == "iterations":
        return result.info.iter
    if name in ("primal_res", "dual_res"):
        return getattr(result.info, name)
    return getattr(result, name)


def _make_callback(signature: DenseSignature, workspace: _Workspace):
    """A Python function with one named, Warp-annotated parameter per input
    and per output of the FFI call, as ``warp.jax_callable`` requires."""
    in_names = signature.fields
    out_shapes = signature.output_shapes()
    out_names = tuple(out_shapes)
    num_inputs = len(in_names)

    annotations = {}
    for name, batched in zip(in_names, signature.batched):
        annotations[name] = wp.array(dtype=signature.dtype, ndim=DATA_NDIM[name] + int(batched))
    for name, shape in out_shapes.items():
        annotations[name] = wp.array(dtype=signature.output_dtype(name), ndim=len(shape))
    annotations["return"] = None

    def run(arrays):
        workspace.run(
            dict(zip(in_names, arrays[:num_inputs])),
            dict(zip(out_names, arrays[num_inputs:])),
        )

    parameters = ", ".join(in_names + out_names)
    source = f"def cupiqp_dense_qp({parameters}):\n    run(({parameters},))\n"
    namespace = {"run": run}
    exec(source, namespace)
    callback = namespace["cupiqp_dense_qp"]
    callback.__annotations__ = annotations
    callback.__module__ = __name__
    return callback


class DensePrimitive:
    """The XLA custom call of one dense signature.

    Calling it with the present data arguments (JAX arrays, in the order of
    ``signature.fields``) returns the raw outputs of the FFI call, in the
    order of ``signature.output_shapes()``, every one with a batch axis.
    """

    def __init__(self, signature: DenseSignature) -> None:
        self.signature = signature
        self.workspace = _Workspace(signature)
        self.output_names = tuple(signature.output_shapes())
        # Graph capture is off: the interior-point loop decides on the host
        # every iteration whether to continue.
        self._callable = jax_callable(
            _make_callback(signature, self.workspace),
            num_outputs=len(self.output_names),
            graph_mode=JaxCallableGraphMode.NONE,
            vmap_method=None,
            output_dims=dict(signature.output_shapes()),
            has_side_effect=False,
        )

    def __call__(self, *arrays) -> tuple:
        return tuple(self._callable(*arrays))


_CACHE_LOCK = threading.Lock()
_PRIMITIVES: dict[DenseSignature, DensePrimitive] = {}


def get_dense_primitive(signature: DenseSignature) -> DensePrimitive:
    """The primitive of ``signature``, created on first use.

    Primitives live for the whole process, like XLA's compiled functions
    that refer to them; what :func:`clear_solvers` releases are the core
    solvers behind them, which the next call creates again."""
    with _CACHE_LOCK:
        primitive = _PRIMITIVES.get(signature)
        if primitive is None:
            primitive = DensePrimitive(signature)
            _PRIMITIVES[signature] = primitive
        return primitive


def clear_solvers() -> None:
    """Release every core solver the primitives keep, and its GPU memory.
    Compiled JAX functions stay valid: their next call creates a new solver."""
    with _CACHE_LOCK:
        for primitive in _PRIMITIVES.values():
            primitive.workspace.close()


def solver_count() -> int:
    """Number of signatures that currently have a live core solver."""
    with _CACHE_LOCK:
        return sum(1 for primitive in _PRIMITIVES.values() if primitive.workspace.has_solvers)
