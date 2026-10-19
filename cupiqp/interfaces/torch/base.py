"""Backend-independent machinery of the PyTorch solver layers.

A solver layer is a ``torch.nn.Module`` that owns one core cuPIQP solver.
``setup()`` fixes the problem structure and allocates every buffer the layer
will ever need; ``forward()`` loads new values, solves, and returns the
primal solution; gradients reach every tensor passed to ``forward()``
through implicit differentiation of the KKT system. Neither ``forward()``
nor the backward pass allocates GPU memory.

This module holds what every backend shares: the core solver and its
settings, the ordering between the caller's Torch stream and the solver
stream, the backward state a differentiable call can lease (the current
factorization, or one of the preallocated saved states), the core calls,
and the presentation of results. The lifetime of a lease for one autograd node
belongs to ``QPSolveFunction`` in ``autograd.py``. Subclasses add only the
handling of their matrix format through a few protected hooks.
"""

from __future__ import annotations

import contextlib
import gc
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

import torch
import warp as wp

from ...data import Data
from ...results import Result
from ...settings import Settings
from ...solver import SolverBase
from .autograd import QPSolveFunction
from .types import StructureMismatchError, TorchSolution
from .utils import (
    DATA_FIELDS,
    MATRIX_FIELDS,
    SOLUTION_FIELDS,
    torch_to_warp_dtype,
    validate_dense_tensor,
)


@dataclass(frozen=True, slots=True)
class _BackwardLease:
    """Backward state reserved by one differentiable forward call.

    ``state_index`` is ``None`` when the backward will reuse the
    factorization left by the forward solve (``max_pending_solves == 1``);
    otherwise it selects the saved state that must be restored and
    refactorized. ``generation`` identifies the call that holds the lease,
    so that a late release of an old call cannot free state a newer call
    reuses.
    """

    generation: int
    state_index: int | None = None

    @property
    def uses_saved_state(self) -> bool:
        return self.state_index is not None


@dataclass(frozen=True, slots=True)
class _SavedBackwardState:
    """Warp views of the buffers of one state-pool slot: what one
    differentiable call saved after its solve. The ten solution fields have
    the shapes of the core result's fields, ``(B, 0)`` for an absent block;
    ``rho``, ``delta``, ``status`` and ``iterations`` are ``(B,)``."""

    x: wp.array
    y: wp.array
    z_u: wp.array
    z_l: wp.array
    z_bu: wp.array
    z_bl: wp.array
    s_u: wp.array
    s_l: wp.array
    s_bu: wp.array
    s_bl: wp.array
    rho: wp.array
    delta: wp.array
    status: wp.array
    iterations: wp.array


class _BackwardStatePool:
    """The state pool of ``max_pending_solves > 1``: saved backward states
    owned by the Torch layer, and their leases (acquisition and
    generation-checked release).

    Each field has one stacked Torch allocation ``(capacity, B, ...)``, made
    here; the Warp views of every slot, for the core copies, are built here
    too, so nothing is allocated or converted per call."""

    def __init__(
        self, core_result: Result, capacity: int, dtype: torch.dtype, device: torch.device
    ) -> None:
        batch_size = core_result.info.batch_size
        self._backing = {
            name: torch.zeros((capacity, *getattr(core_result, name).shape),
                              dtype=dtype, device=device)
            for name in SOLUTION_FIELDS
        }
        for name in ("rho", "delta"):
            self._backing[name] = torch.zeros((capacity, batch_size), dtype=dtype, device=device)
        for name in ("status", "iterations"):
            self._backing[name] = torch.zeros((capacity, batch_size), dtype=torch.int32,
                                              device=device)
        self._states = tuple(
            _SavedBackwardState(**{
                name: wp.from_torch(backing[index]) for name, backing in self._backing.items()
            })
            for index in range(capacity)
        )
        self._active_generations = [None] * capacity
        self._next_generation = 0

    def acquire(self) -> _BackwardLease:
        index = next((i for i, g in enumerate(self._active_generations) if g is None), None)
        if index is None:
            raise RuntimeError(
                f"{len(self._active_generations)} solves are already pending backward, "
                "the max_pending_solves of setup(): run backward() on the losses of "
                "earlier calls (or drop their outputs) before the next differentiable "
                "call, or set up the layer with a larger max_pending_solves."
            )
        self._next_generation += 1
        self._active_generations[index] = self._next_generation
        return _BackwardLease(generation=self._next_generation, state_index=index)

    def release(self, lease: _BackwardLease) -> None:
        if self._active_generations[lease.state_index] == lease.generation:
            self._active_generations[lease.state_index] = None

    def state_for(self, lease: _BackwardLease) -> _SavedBackwardState:
        return self._states[lease.state_index]

    def output_views(self, lease: _BackwardLease, names: Sequence[str]) -> tuple[torch.Tensor, ...]:
        """New Torch view objects (no allocation) of the named buffers of the
        leased slot. Fresh objects, so the pool never holds an autograd output."""
        return tuple(self._backing[name][lease.state_index] for name in names)

    def has_active_leases(self) -> bool:
        return any(g is not None for g in self._active_generations)


class SolverLayerBase(torch.nn.Module, ABC):
    """Common base of the differentiable solver layers.

    Lifecycle::

        layer = DenseSolverLayer()
        layer.settings.max_iter = 50
        layer.setup(P, c, A=A, b=b)      # fixes the structure, allocates
        x = layer(c=c_k, b=b_k)          # updates values, solves: x is (B, n)
        loss(x).backward()               # gradients for c_k and b_k

    ``setup()`` and ``forward()`` take the same argument names as the core
    solver. The layer always solves a batch of ``B`` problems, ``B`` being
    read from the batched arguments of ``setup()`` (1 if all are shared).
    Every argument may be shared (single-problem shape, one value for all
    problems) or batched (leading batch axis), independently of the others.
    Results always carry the batch axis. Gradients flow to the tensors
    passed to ``forward()``; a shared tensor receives the gradient summed
    over the batch.

    Memory: all GPU buffers are allocated by ``setup()``. A call and its
    backward allocate nothing, so the tensors they return are views of
    layer-owned memory that the next call or backward overwrites. Clone a
    result to keep it. Pass contiguous tensors: a non-contiguous input is
    made contiguous by Torch, which allocates.

    Pending solves (``setup(..., max_pending_solves=..., per_call_fields=...)``):
    a differentiable call is *pending* from its forward until its backward
    has run, or until its output graph is dropped. ``max_pending_solves``
    is how many may be pending at once, and it also decides how the
    backward works:

    * ``max_pending_solves=1`` (default): the backward reuses the KKT
      factorization left by the forward solve, which is the fastest
      backward. While a call is pending, the layer refuses every further
      call, even one under ``torch.no_grad()``, since any solve would
      overwrite that factorization. Run ``backward()`` on the loss first,
      drop the output, or use separate layer instances.
    * ``max_pending_solves=T > 1``: for example the ``T`` steps of a
      rollout with one ``backward()`` at the end. Every differentiable call
      saves its solution (about the size of one solution) into one of
      ``T`` states preallocated at setup and returns views of that state,
      so each pending call keeps its own output. Every backward restores
      the inputs of its call, refactorizes the KKT system at the saved
      point and differentiates there, also for the most recent call.
      ``setup()`` then requires ``per_call_fields``, the names of the
      arguments every call passes (for example ``("c", "h_u")``): every
      call, also one under ``torch.no_grad()``, must pass exactly those,
      and the other data keep their setup values. Inputs must be
      contiguous. Calls that need no gradient save no state, do not count
      against the limit, and may run while other calls are pending.

    Neither a call nor its backward waits for the GPU on the host, and
    ``layer.result`` holds device tensors only. Problems whose solve did not
    converge are differentiated anyway, or get zero gradients, according to
    ``settings.gradient_on_failure``; ``layer.result.status`` says which
    problems failed.

    Settings that the core solver fixes at ``setup()`` (``enable_grad``,
    ``gradient_smoothing`` and the preconditioner settings, see
    :class:`cupiqp.DenseSolver`) raise at the next call if changed; the
    other settings may change between calls.
    """

    #: The core solver class this layer wraps; set by the subclass.
    _core_solver_cls: type[SolverBase]

    def __init__(
        self,
        dtype: torch.dtype = torch.float64,
        stream: torch.cuda.Stream | None = None,
        device: torch.device | str | None = None
    ) -> None:
        super().__init__()
        torch_to_warp_dtype(dtype)               # raises TypeError for any other dtype
        if stream is not None and not isinstance(stream, torch.cuda.Stream):
            raise TypeError(
                "stream must be a torch.cuda.Stream or None; got "
                f"{type(stream).__name__}."
            )
        # The device comes from, in this order: the explicit argument, the
        # stream, Torch's current CUDA device. A stream from another device
        # cannot order the solver, so the two must agree.
        resolved = None
        if device is not None:
            resolved = torch.device(device)
            if resolved.type != "cuda":
                raise ValueError(f"device must be a CUDA device; got {resolved}.")
            if resolved.index is None:
                resolved = torch.device("cuda", torch.cuda.current_device())
        if stream is not None:
            if resolved is not None and stream.device != resolved:
                raise ValueError(
                    f"stream is on {stream.device} but device is {resolved}; "
                    "they must agree."
                )
            resolved = stream.device
        if resolved is None:
            resolved = torch.device("cuda", torch.cuda.current_device())
        self._dtype = dtype
        self._torch_stream = stream
        self._device = resolved
        self._create_core_solver()
        self._clear_setup_state()

    def _clear_setup_state(self) -> None:
        """Forget the problem set up (not the core solver or the settings)."""
        self._is_setup = False
        self._configured_data_fields = frozenset()
        self._unbatched_input_shapes = {}
        self._solution_fields = ()
        self._shared_input_grad_buffers = {}
        self._result = None
        self._active_factorization_generation = None
        self._next_factorization_generation = 0
        self._max_pending_solves = 1
        self._backward_state_pool = None
        self._per_call_fields = None

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def settings(self) -> Settings:
        """Configuration of the core solver. Change fields before
        ``setup()`` or between calls, e.g. ``layer.settings.max_iter = 50``.
        ``enable_grad`` is on by default for a layer."""
        return self._core_solver.settings

    @settings.setter
    def settings(self, value: Settings) -> None:
        if not isinstance(value, Settings):
            raise TypeError("settings must be a cupiqp.Settings instance.")
        if value.dtype is not torch_to_warp_dtype(self._dtype):
            raise ValueError(
                f"settings dtype {value.dtype!r} does not match the layer "
                f"dtype {self._dtype!r}."
            )
        self._core_solver.settings = value

    @property
    def dtype(self) -> torch.dtype:
        """Floating-point dtype of all QP data and results."""
        return self._dtype

    @property
    def device(self) -> torch.device:
        """CUDA device the layer solves on."""
        return self._device

    @property
    def is_setup(self) -> bool:
        """Whether ``setup()`` has fixed the problem structure."""
        return self._is_setup

    @property
    def batch_size(self) -> int:
        """Number of problems solved per call, fixed by ``setup()``."""
        if not self._is_setup:
            raise RuntimeError("The layer is not set up. Call setup() first.")
        return self._core_solver.data.batch_size

    @property
    def max_pending_solves(self) -> int:
        """How many differentiable calls may await their backward at once,
        fixed by ``setup()``; see the class documentation."""
        return self._max_pending_solves

    @property
    def result(self) -> TorchSolution | None:
        """Full solution of the latest call (duals, slacks, status) as a
        :class:`TorchSolution`, or ``None`` before the first call."""
        return self._result

    def reset(self) -> None:
        """Discard the problem and its GPU memory; ``setup()`` may then be
        called again, with a new structure. The settings are kept.

        Refuses while a differentiable call is pending backward, since its
        later ``backward()`` would run on the new solver: finish that
        backward first, or drop the output so that its autograd graph is
        freed.
        """
        # The previous result holds views of the solve's outputs and, through
        # their grad_fn, the autograd node that holds the lease. Drop it so
        # that a graph the user has already abandoned is freed now.
        self._result = None
        if self._has_active_leases():
            raise RuntimeError(
                "reset() with a differentiable solve pending backward: call "
                "backward() on its loss first, or delete the output tensors "
                "(and any loss built from them) to abandon it."
            )
        settings = self._core_solver.settings
        self._create_core_solver()
        self._core_solver.settings = settings
        self._clear_setup_state()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(
        self,
        P: torch.Tensor,
        c: torch.Tensor,
        A: torch.Tensor | None = None,
        b: torch.Tensor | None = None,
        G: torch.Tensor | None = None,
        h_u: torch.Tensor | None = None,
        h_l: torch.Tensor | None = None,
        x_u: torch.Tensor | None = None,
        x_l: torch.Tensor | None = None,
        max_pending_solves: int = 1,
        per_call_fields: Sequence[str] | None = None
    ) -> None:
        """Fix the problem structure from these tensors, load their values and
        allocate every GPU buffer of the layer.

        Same arguments as the core solver's ``setup()``: ``P`` and ``c`` are
        required; ``A, b`` (equalities), ``G, h_u, h_l`` (inequalities) and
        ``x_u, x_l`` (bounds) are optional. Each tensor is of the layer dtype,
        on the layer device, and either shared (single-problem shape) or
        batched (leading batch axis). The matrix format depends on the layer:
        dense CUDA tensors for :class:`DenseSolverLayer`, CUDA sparse CSR
        tensors for :class:`SparseSolverLayer`. Does not solve; call the
        layer afterwards. Gradients are not tracked for ``setup()`` tensors,
        only for those passed to the layer call.

        ``max_pending_solves`` is how many differentiable calls may await
        their backward at once: 1 reuses the forward factorization in the
        backward; more saves every call's solution and refactorizes in its
        backward, and then requires ``per_call_fields``, the names of the
        arguments every call passes. See the class documentation. Call
        ``setup()`` once per layer; ``reset()`` allows a new one.
        """
        inputs = dict(P=P, c=c, A=A, b=b, G=G, h_u=h_u, h_l=h_l, x_u=x_u, x_l=x_l)
        if self._is_setup:
            raise RuntimeError(
                "setup() may only be called once; call reset() before setting "
                "up a different problem."
            )
        if inputs["P"] is None or inputs["c"] is None:
            raise ValueError("Both 'P' and 'c' are required.")
        if (isinstance(max_pending_solves, bool) or not isinstance(max_pending_solves, int)
                or max_pending_solves < 1):
            raise ValueError(
                f"max_pending_solves must be a positive integer; got {max_pending_solves!r}."
            )
        per_call = None
        if max_pending_solves == 1:
            if per_call_fields is not None:
                raise ValueError(
                    "per_call_fields is only used with max_pending_solves > 1, where "
                    "each backward restores the data of its call; with "
                    "max_pending_solves=1 leave it as None."
                )
        else:
            if not self.settings.enable_grad:
                raise ValueError("max_pending_solves > 1 requires settings.enable_grad.")
            if per_call_fields is None:
                raise ValueError(
                    "max_pending_solves > 1 requires per_call_fields: the names "
                    "of the arguments every call passes, e.g. ('c', 'h_u')."
                )
            per_call = frozenset(per_call_fields)
            unknown = sorted(n for n in per_call if n not in DATA_FIELDS)
            absent = sorted(n for n in per_call if n in DATA_FIELDS and inputs[n] is None)
            if unknown or absent:
                raise ValueError(
                    f"per_call_fields must name arguments given to setup(); unknown "
                    f"{unknown}, not given {absent}."
                )
        core_setup_args = self._prepare_setup_inputs(inputs)
        with self._ordered_solver_stream():
            self._core_solver.setup(**core_setup_args)

        data = self._core_solver.data
        self._configured_data_fields = frozenset(
            name for name in DATA_FIELDS if inputs[name] is not None
        )
        vector_shapes = {
            "c": (data.n,), "b": (data.p,), "h_u": (data.m,), "h_l": (data.m,),
            "x_u": (data.n,), "x_l": (data.n,),
        }
        self._unbatched_input_shapes = {
            name: (tuple(self._matrix_unbatched_shape(name)) if name in MATRIX_FIELDS
                   else vector_shapes[name])
            for name in self._configured_data_fields
        }
        # Solution fields that exist for this structure. Only these are handed
        # out (absent fields are None), so no layer-owned tensor is ever an
        # autograd output that would pin a graph.
        result = self._core_solver.result
        self._solution_fields = tuple(
            name for name in SOLUTION_FIELDS if getattr(result, name).size > 0
        )
        # Everything a call or backward hands out is allocated here, once.
        if self.settings.enable_grad:
            self._shared_input_grad_buffers = {
                name: torch.empty(shape, dtype=self._dtype, device=self._device)
                for name, shape in self._unbatched_input_shapes.items()
            }
        self._max_pending_solves = max_pending_solves
        if max_pending_solves > 1:
            self._backward_state_pool = _BackwardStatePool(
                self._core_solver.result, max_pending_solves, self._dtype, self._device
            )
            self._per_call_fields = per_call
        self._is_setup = True

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        P: torch.Tensor | None = None,
        c: torch.Tensor | None = None,
        A: torch.Tensor | None = None,
        b: torch.Tensor | None = None,
        G: torch.Tensor | None = None,
        h_u: torch.Tensor | None = None,
        h_l: torch.Tensor | None = None,
        x_u: torch.Tensor | None = None,
        x_l: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Load the given values, solve, and return the primal solution ``x``
        of shape ``(B, n)``.

        Arguments left as ``None`` keep their current values. Each tensor is
        either shared, in the single-problem shape fixed by ``setup()``, or
        batched with the leading batch axis. Gradients flow to every tensor
        passed here that requires them. The returned tensor is a view of the
        solver's result buffer, overwritten by the next call; ``layer.result``
        holds the duals, slacks and the per-problem status.
        """
        # Drop the previous result first: it may hold the last reference to an
        # abandoned autograd graph, whose finalizer then releases its lease.
        self._result = None
        if not self._is_setup:
            raise RuntimeError("The layer is not set up. Call setup() first.")
        if self._backward_state_pool is None and self._active_factorization_generation is not None:
            raise RuntimeError(
                "This layer has a differentiable solve pending backward, and "
                "with max_pending_solves=1 that backward needs the current "
                "factorization, which any further solve would overwrite. Call "
                "backward() on its loss first, delete the output to abandon it, "
                "use another layer instance, or set up the layer with "
                "max_pending_solves > 1."
            )

        forward_inputs = (P, c, A, b, G, h_u, h_l, x_u, x_l)
        input_is_shared = self._validate_call_inputs(forward_inputs)

        requires_autograd = torch.is_grad_enabled() and any(
            tensor is not None and tensor.requires_grad for tensor in forward_inputs
        )
        if requires_autograd and not self.settings.enable_grad:
            raise RuntimeError(
                "An input requires gradients but settings.enable_grad is "
                "False. Enable it before setup(), or call the layer under "
                "torch.no_grad()."
            )
        if requires_autograd:
            all_outputs = QPSolveFunction.apply(self, input_is_shared, *forward_inputs)
        else:
            all_outputs = self._run_forward_solve(forward_inputs, backward_lease=None)

        solution_count = len(self._solution_fields)
        status, iterations = all_outputs[solution_count:]
        self._result = TorchSolution(
            dict(zip(self._solution_fields, all_outputs[:solution_count])),
            status,
            iterations,
            self._core_solver.result.info,
        )
        return self._result.x

    # ------------------------------------------------------------------
    # Backend hooks
    # ------------------------------------------------------------------

    @abstractmethod
    def _prepare_setup_inputs(self, inputs: dict[str, torch.Tensor | None]) -> dict[str, Any]:
        """Validate the ``setup()`` tensors of this backend and return the
        keyword arguments of the core solver's ``setup()``.

        ``inputs`` maps every data field to its tensor, or ``None`` if not
        given. Raise directly, naming the field, on an invalid input
        (``validate_cuda_tensor`` and ``validate_dense_tensor`` in ``utils``
        check device, dtype and layout); detach the tensors and make them
        contiguous or convert them to the core's format (for example CSR
        triples)."""

    @abstractmethod
    def _matrix_unbatched_shape(self, name: str) -> tuple[int, ...]:
        """Shape without the batch axis of the tensor ``forward()`` takes for
        this matrix, read from the set-up core data."""

    @abstractmethod
    def _matrix_data_gradient_array(self, name: str, data_gradients: Data) -> wp.array:
        """``(B, ...)`` Warp gradient array of this matrix in the core's
        gradient data (the backend's ``Data`` subclass), laid out like the
        tensor ``forward()`` takes."""

    # ------------------------------------------------------------------
    # Core execution (called by forward() and by QPSolveFunction)
    # ------------------------------------------------------------------

    def _run_forward_solve(
        self,
        forward_inputs: tuple[torch.Tensor | None, ...],
        backward_lease: _BackwardLease | None
    ) -> tuple[torch.Tensor, ...]:
        """Update the core with the given tensors, solve, and return
        ``(*solution_views, status, iterations)``: views of the core result,
        or of the leased saved state after saving the solve into it."""
        data_updates = {
            name: tensor.detach().contiguous()
            for name, tensor in zip(DATA_FIELDS, forward_inputs) if tensor is not None
        }
        saved_state = None
        if backward_lease is not None and backward_lease.uses_saved_state:
            saved_state = self._backward_state_pool.state_for(backward_lease)
        with self._ordered_solver_stream():
            if data_updates:
                self._core_solver.update(**data_updates)
            self._core_solver.solve()
            if saved_state is not None:
                self._save_backward_state(saved_state)
        # Fresh view objects every call (no device allocation): a tensor the
        # layer keeps must never be an autograd output, or it would pin the
        # graph of an abandoned call and its saved state forever.
        if saved_state is not None:
            return self._backward_state_pool.output_views(
                backward_lease, (*self._solution_fields, "status", "iterations")
            )
        result = self._core_solver.result
        solution_views = tuple(wp.to_torch(getattr(result, name)) for name in self._solution_fields)
        return (*solution_views, wp.to_torch(result.info.status), wp.to_torch(result.info.iter))

    def _save_backward_state(self, saved_state: _SavedBackwardState) -> None:
        """Copy the final state of the latest solve from the core result into
        a slot of the state pool: device-to-device copies on the solver
        stream, after the solve, without synchronization."""
        result = self._core_solver.result
        with wp.ScopedStream(self._warp_stream, sync_enter=False):
            for name in SOLUTION_FIELDS:
                source = getattr(result, name)
                if source.size > 0:
                    wp.copy(getattr(saved_state, name), source)
            wp.copy(saved_state.rho, result.info.rho)
            wp.copy(saved_state.delta, result.info.delta)
            wp.copy(saved_state.status, result.info.status)
            wp.copy(saved_state.iterations, result.info.iter)

    def _compute_input_gradients(
        self,
        *,
        solution_cotangents: tuple[torch.Tensor | None, ...],
        input_needs_grad: tuple[bool, ...],
        input_is_shared: tuple[bool | None, ...],
        backward_lease: _BackwardLease,
        restored_data: dict[str, torch.Tensor] | None
    ) -> tuple[torch.Tensor | None, ...]:
        """Map the cotangents of the solution views to gradients of the
        ``forward()`` inputs, one per data field (``None`` where no gradient
        is needed).

        ``solution_cotangents`` follow ``_solution_fields``;
        ``input_needs_grad`` and ``input_is_shared`` follow ``DATA_FIELDS``.
        A saved-state lease requires ``restored_data``, the tensors that call
        passed, which are loaded into the core before refactorizing; a
        factorization lease requires ``restored_data`` to be ``None``."""
        if backward_lease.uses_saved_state != (restored_data is not None):
            raise RuntimeError(
                "restored_data must be given for a saved-state lease and only then."
            )
        core_cotangents = {
            "grad_" + name: cotangent.detach().contiguous()
            for name, cotangent in zip(self._solution_fields, solution_cotangents)
            if cotangent is not None
        }
        with self._ordered_solver_stream():
            if backward_lease.uses_saved_state:
                # The data of that call; later calls may have changed it.
                # Always refactorized, also for the most recent call.
                self._core_solver.update(**restored_data)
                saved_state = self._backward_state_pool.state_for(backward_lease)
                data_gradients = self._core_solver.backward_from_state(
                    **{name: getattr(saved_state, name) for name in SOLUTION_FIELDS},
                    rho=saved_state.rho,
                    delta=saved_state.delta,
                    status=saved_state.status,
                    **core_cotangents,
                )
            else:
                data_gradients = self._core_solver.backward(**core_cotangents)
        # Past the ordered block the Torch stream has waited for the solver,
        # so the reductions below may read the core gradient buffers.
        input_gradients = []
        for name, needed, is_shared in zip(DATA_FIELDS, input_needs_grad, input_is_shared):
            if not needed:
                input_gradients.append(None)
                continue
            if name in MATRIX_FIELDS:
                batched_gradient_array = self._matrix_data_gradient_array(name, data_gradients)
            else:
                batched_gradient_array = getattr(data_gradients, name)
            input_gradient = wp.to_torch(batched_gradient_array)
            if is_shared:
                # A shared input served every problem: sum over the batch
                # into the buffer allocated at setup.
                input_gradient = torch.sum(
                    input_gradient, dim=0, out=self._shared_input_grad_buffers[name]
                )
            input_gradients.append(input_gradient)
        return tuple(input_gradients)

    # ------------------------------------------------------------------
    # Leases on the backward state
    # ------------------------------------------------------------------

    def _acquire_backward_lease(self) -> _BackwardLease:
        """Reserve the backward state for one differentiable call: a free
        saved state if ``max_pending_solves > 1``, else the current
        factorization."""
        if self._backward_state_pool is not None:
            return self._backward_state_pool.acquire()
        self._next_factorization_generation += 1
        self._active_factorization_generation = self._next_factorization_generation
        return _BackwardLease(generation=self._active_factorization_generation)

    def _release_backward_lease(self, lease: _BackwardLease) -> None:
        """Free the state of ``lease``; a stale lease (already released, or
        superseded by a newer call) is ignored."""
        if lease.uses_saved_state:
            if self._backward_state_pool is not None:
                self._backward_state_pool.release(lease)
        elif self._active_factorization_generation == lease.generation:
            self._active_factorization_generation = None

    def _has_active_leases(self) -> bool:
        if self._backward_state_pool is not None:
            return self._backward_state_pool.has_active_leases()
        return self._active_factorization_generation is not None

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_call_inputs(
        self, forward_inputs: tuple[torch.Tensor | None, ...]
    ) -> tuple[bool | None, ...]:
        """Validate the tensors of one layer call against the structure of
        ``setup()`` and return, per data field, whether the tensor is shared
        (single-problem shape), batched, or not passed (``None``).

        With ``max_pending_solves > 1`` every call must pass exactly the
        ``per_call_fields`` of ``setup()``, so that each backward restores
        well-defined data, as contiguous tensors, so that nothing allocates.
        """
        multi_pending = self._backward_state_pool is not None
        if multi_pending:
            passed_fields = frozenset(
                name for name, tensor in zip(DATA_FIELDS, forward_inputs) if tensor is not None
            )
            if passed_fields != self._per_call_fields:
                raise ValueError(
                    "With max_pending_solves > 1 every call must pass exactly the "
                    f"per_call_fields of setup(), {sorted(self._per_call_fields)}; got "
                    f"{sorted(passed_fields)}."
                )

        batch_size = self._core_solver.data.batch_size
        input_is_shared = []
        for name, tensor in zip(DATA_FIELDS, forward_inputs):
            if tensor is None:
                input_is_shared.append(None)
                continue
            validate_dense_tensor(name, tensor, self._device, self._dtype)
            if name not in self._configured_data_fields:
                raise StructureMismatchError(
                    f"'{name}' was not given at setup(); adding a block requires "
                    "reset() and a new setup()."
                )
            unbatched_shape = self._unbatched_input_shapes[name]
            batched_shape = (batch_size, *unbatched_shape)
            actual_shape = tuple(tensor.shape)
            if actual_shape == unbatched_shape:
                input_is_shared.append(True)
            elif actual_shape == batched_shape:
                input_is_shared.append(False)
            else:
                raise StructureMismatchError(
                    f"'{name}' must have shape {unbatched_shape} or {batched_shape} as "
                    f"fixed by setup(); got {actual_shape}. Call reset() and setup() to "
                    "change the structure."
                )
            if multi_pending and not tensor.is_contiguous():
                raise ValueError(
                    f"'{name}' is not contiguous; max_pending_solves > 1 "
                    "requires contiguous inputs (call .contiguous() first)."
                )
        return tuple(input_is_shared)

    # ------------------------------------------------------------------
    # Core solver and streams
    # ------------------------------------------------------------------

    def _create_core_solver(self) -> None:
        """Create the core solver on the layer's device, with its own Warp
        stream, and the Torch view of that stream used for ordering."""
        # The previous solver borrows the previous stream and must be torn
        # down before that stream goes away: drop it and collect any cycle
        # that still holds it, before a new stream replaces the old one.
        if getattr(self, "_core_solver", None) is not None:
            self._core_solver = None
            gc.collect()
        wp.init()                         # idempotent; needed before device lookup
        self._warp_stream = wp.Stream(wp.device_from_torch(self._device))
        self._core_solver = self._core_solver_cls(
            dtype=torch_to_warp_dtype(self._dtype), stream=self._warp_stream
        )
        self._core_solver.settings.enable_grad = True
        self._core_stream_view = wp.stream_to_torch(self._warp_stream)

    @contextlib.contextmanager
    def _ordered_solver_stream(self) -> Iterator[None]:
        """Order one block of core calls with the Torch stream (the one given
        to the constructor, else Torch's current stream): the solver stream
        waits for the Torch stream's prior work, and the Torch stream waits
        for the solver afterwards. Torch-side conversions (``contiguous``)
        must be issued before entering; Torch ops inside run on the Torch
        stream and are ordered after the solver by the exit wait only, so
        keep them to reads of solver buffers that follow a solver call."""
        torch_stream = self._torch_stream or torch.cuda.current_stream(self._device)
        view = self._core_stream_view
        view.wait_stream(torch_stream)
        try:
            yield
        finally:
            torch_stream.wait_stream(view)

    def extra_repr(self) -> str:
        if not self._is_setup:
            return f"dtype={self._dtype}, not set up"
        data = self._core_solver.data
        return (f"dtype={self._dtype}, B={data.batch_size}, n={data.n}, "
                f"p={data.p}, m={data.m}")
