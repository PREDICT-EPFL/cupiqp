"""Backend-independent machinery of the PyTorch solver layers.

A solver layer is a ``torch.nn.Module`` that owns one core cuPIQP solver.
``setup()`` fixes the problem structure and allocates every buffer the layer
will ever need; ``forward()`` loads new values, solves, and returns the
primal solution; gradients reach every tensor passed to ``forward()``
through implicit differentiation of the KKT system. Neither ``forward()``
nor the backward pass allocates GPU memory.

This module holds what every backend shares: the core solver and its
settings, the ordering between the caller's Torch stream and the solver
stream, the autograd bridge, the one-outstanding-backward lease, and the
presentation of results. Subclasses add only the handling of their matrix
format through a few protected hooks.
"""

from __future__ import annotations

import contextlib
from abc import ABC, abstractmethod
from typing import Optional

import torch
import warp as wp

from ...settings import Settings
from ...solver import SolverBase
from .autograd import _SolveFunction
from .types import StructureMismatchError, TorchSolution


SLOTS = ("P", "c", "A", "b", "G", "h_u", "h_l", "x_u", "x_l")
MATRIX_SLOTS = ("P", "A", "G")

_TORCH_TO_WARP = {torch.float32: wp.float32, torch.float64: wp.float64}
_RESULT_FIELDS = ("x", "y", "z_u", "z_l", "z_bu", "z_bl", "s_u", "s_l", "s_bu", "s_bl")


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

    One differentiable call may be outstanding at a time: run ``backward()``
    on its loss before the next differentiable call, or use separate layer
    instances. Dropping the output without a backward (so that its autograd
    graph is freed) also ends the outstanding call. Calls under
    ``torch.no_grad()`` or with inputs that do not require gradients are
    not limited.

    Neither a call nor its backward waits for the GPU on the host, and
    ``layer.result`` holds device tensors only. Problems whose solve did not
    converge are differentiated anyway, or get zero gradients, according to
    ``settings.gradient_on_failure``; ``layer.result.status`` says which
    problems failed.

    ``settings.enable_grad`` and ``settings.gradient_smoothing`` are fixed
    by ``setup()``; the other settings may change between calls.
    """

    _core_solver_cls: type[SolverBase]

    def __init__(
        self,
        dtype: torch.dtype = torch.float64,
        stream: Optional[torch.cuda.Stream] = None,
        device: Optional[torch.device | str] = None
    ):
        super().__init__()
        if dtype not in _TORCH_TO_WARP:
            raise TypeError(
                f"dtype must be torch.float32 or torch.float64; got {dtype!r}."
            )
        if stream is not None and not isinstance(stream, torch.cuda.Stream):
            raise TypeError(
                "stream must be a torch.cuda.Stream or None; got "
                f"{type(stream).__name__}."
            )
        # The device comes from, in this order: the explicit argument, the
        # stream, Torch's current CUDA device. A stream from another device
        # cannot order the solver, so the two must agree.
        if device is not None:
            device = torch.device(device)
            if device.type != "cuda":
                raise ValueError(f"device must be a CUDA device; got {device}.")
            if device.index is None:
                device = torch.device("cuda", torch.cuda.current_device())
        if stream is not None:
            if device is not None and stream.device != device:
                raise ValueError(
                    f"stream is on {stream.device} but device is {device}; "
                    "they must agree."
                )
            device = stream.device
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self._dtype = dtype
        self._torch_stream = stream
        self._device: torch.device = device
        self._core_solver: SolverBase = None
        self._core_stream_view: torch.cuda.Stream = None
        self.create_core_solver()
        self._clear_problem_state()

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
        if value.dtype is not _TORCH_TO_WARP[self._dtype]:
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
    def result(self) -> Optional[TorchSolution]:
        """Full solution of the latest call (duals, slacks, status) as a
        :class:`TorchSolution`, or ``None`` before the first call."""
        return self._result

    def reset(self) -> None:
        """Discard the problem and its GPU memory; ``setup()`` may then be
        called again, with a new structure. The settings are kept.

        Refuses while a differentiable call is outstanding, since its later
        ``backward()`` would run on the new solver: finish that backward
        first, or drop the output so that its autograd graph is freed.
        """
        # The previous result holds views of the solve's outputs and, through
        # their grad_fn, the autograd node that holds the lease. Drop it so
        # that a graph the user has already abandoned is freed now.
        self._result = None
        if self._lease_token is not None:
            raise RuntimeError(
                "reset() with a differentiable solve outstanding: call "
                "backward() on its loss first, or delete the output tensors "
                "(and any loss built from them) to abandon it."
            )
        settings = self._core_solver.settings
        self.create_core_solver()
        self._core_solver.settings = settings
        self._clear_problem_state()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        P=None,
        c=None,
        A=None,
        b=None,
        G=None,
        h_u=None,
        h_l=None,
        x_u=None,
        x_l=None
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
        if not self._is_setup:
            raise RuntimeError("The layer is not set up. Call setup() first.")
        self._result = None          # see reset(): frees an abandoned graph
        if self._lease_token is not None:
            raise RuntimeError(
                "This layer has a differentiable solve outstanding. Call "
                "backward() on its loss before the next call, use another "
                "layer instance, or delete the output to abandon it."
            )
        self._check_structural_settings()
        given = (P, c, A, b, G, h_u, h_l, x_u, x_l)
        shared = tuple(
            None if tensor is None else self._check_forward_tensor(name, tensor)
            for name, tensor in zip(SLOTS, given)
        )

        differentiable = torch.is_grad_enabled() and any(
            t is not None and t.requires_grad for t in given
        )
        if differentiable and not self.settings.enable_grad:
            raise RuntimeError(
                "An input requires gradients but settings.enable_grad was "
                "False at setup(). Enable it before setup(), or call the "
                "layer under torch.no_grad()."
            )
        if differentiable:
            outputs = _SolveFunction.apply(self, shared, *given)
        else:
            outputs = self._solve_core(given)
        self._result = TorchSolution(
            dict(zip(self._result_fields, outputs)),
            self._status_view,
            self._iterations_view,
            self._core_solver.result.info,
        )
        return self._result.x

    # ------------------------------------------------------------------
    # Backend hooks (matrix slots only; vectors are handled here)
    # ------------------------------------------------------------------

    @abstractmethod
    def _check_setup_matrix(self, name: str, tensor) -> None:
        """Validate a matrix given to ``setup()``."""

    @abstractmethod
    def _setup_matrix_to_core(self, name: str, tensor):
        """Convert a validated ``setup()`` matrix to the core argument."""

    @abstractmethod
    def _matrix_single_shape(self, name: str) -> tuple:
        """Single-problem shape of the tensor ``forward()`` takes for this
        matrix, read from the set-up core data."""

    @abstractmethod
    def _matrix_gradient_array(self, name: str, gradients) -> wp.array:
        """``(B, ...)`` Warp gradient array of this matrix in the core's
        gradient data, laid out like the tensor ``forward()`` takes."""

    # ------------------------------------------------------------------
    # Setup (called by the subclass' public setup())
    # ------------------------------------------------------------------

    def _setup(self, inputs: dict) -> None:
        if self._is_setup:
            raise RuntimeError(
                "setup() may only be called once; call reset() before setting "
                "up a different problem."
            )
        if inputs["P"] is None or inputs["c"] is None:
            raise ValueError("Both 'P' and 'c' are required.")
        core_solver_setup_args = {}
        for name in SLOTS:
            tensor = inputs[name]
            if tensor is None:
                core_solver_setup_args[name] = None
            elif name in MATRIX_SLOTS:
                self._check_setup_matrix(name, tensor)
                core_solver_setup_args[name] = self._setup_matrix_to_core(name, tensor)
            else:
                self._check_vector(name, tensor)
                if tensor.dim() not in (1, 2):
                    raise ValueError(
                        f"'{name}' must be 1-D or 2-D (batched); got shape "
                        f"{tuple(tensor.shape)}."
                    )
                core_solver_setup_args[name] = tensor.detach().contiguous()
        with self._ordered():
            self._core_solver.setup(**core_solver_setup_args)

        data = self._core_solver.data
        B = data.batch_size
        self._present = frozenset(name for name in SLOTS if inputs[name] is not None)
        vector_shapes = {
            "c": (data.n,), "b": (data.p,), "h_u": (data.m,), "h_l": (data.m,),
            "x_u": (data.n,), "x_l": (data.n,),
        }
        self._single_shapes = {
            name: (tuple(self._matrix_single_shape(name)) if name in MATRIX_SLOTS
                   else vector_shapes[name])
            for name in self._present
        }
        # Result fields that exist for this structure. Only these are handed
        # out (absent fields are None), so no layer-owned tensor is ever an
        # autograd output that would pin a graph.
        result = self._core_solver.result
        self._result_fields = tuple(
            name for name in _RESULT_FIELDS if getattr(result, name).size > 0
        )
        # Everything a call or backward hands out is allocated here, once.
        self._status_view = wp.to_torch(result.info.status)
        self._iterations_view = wp.to_torch(result.info.iter)
        if self.settings.enable_grad:
            self._shared_grad_buffers = {
                name: torch.empty(shape, dtype=self._dtype, device=self._device)
                for name, shape in self._single_shapes.items()
            }
        self._structural_settings = self._current_structural_settings()
        self._is_setup = True

    # ------------------------------------------------------------------
    # Core execution
    # ------------------------------------------------------------------

    def _solve_core(self, given: tuple) -> tuple:
        """Update the core with the given tensors, solve, return result views."""
        updates = {
            name: tensor.detach().contiguous()
            for name, tensor in zip(SLOTS, given) if tensor is not None
        }
        with self._ordered():
            if updates:
                self._core_solver.update(**updates)
            self._core_solver.solve()
        return self._result_views()

    def _backward_core(self, grad_outputs, has_inputs, needs_grad, shared) -> tuple:
        """Map upstream cotangents to gradients of the ``forward()`` tensors."""
        upstream = {
            "grad_" + name: grad.detach().contiguous()
            for name, grad in zip(self._result_fields, grad_outputs)
            if grad is not None
        }
        with self._ordered():
            gradients = self._core_solver.backward(**upstream)
        # Past the ordered block the Torch stream has waited for the solver,
        # so the reductions below may read the core gradient buffers.
        out = []
        for name, present, needed, is_shared in zip(SLOTS, has_inputs, needs_grad, shared):
            if not (present and needed):
                out.append(None)
                continue
            if name in MATRIX_SLOTS:
                array = self._matrix_gradient_array(name, gradients)
            else:
                array = getattr(gradients, name)
            gradient = wp.to_torch(array)
            if is_shared:
                # A shared input served every problem: sum over the batch
                # into the buffer allocated at setup.
                gradient = torch.sum(gradient, dim=0, out=self._shared_grad_buffers[name])
            out.append(gradient)
        return tuple(out)

    def _result_views(self) -> tuple:
        """Zero-copy Torch views of the present result fields."""
        result = self._core_solver.result
        return tuple(wp.to_torch(getattr(result, name)) for name in self._result_fields)

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _check_tensor(self, name: str, tensor) -> None:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"'{name}' must be a torch.Tensor; got {type(tensor).__name__}.")
        if tensor.device.type != "cuda" or (
            tensor.device.index is not None
            and tensor.device.index != self._device.index
        ):
            raise ValueError(
                f"'{name}' must be on {self._device}; got {tensor.device}."
            )
        if tensor.dtype != self._dtype:
            raise ValueError(
                f"'{name}' has dtype {tensor.dtype}; this layer uses {self._dtype}."
            )

    def _check_vector(self, name: str, tensor) -> None:
        self._check_tensor(name, tensor)
        if tensor.layout != torch.strided:
            raise TypeError(f"'{name}' must be a dense (strided) tensor.")

    def _check_forward_tensor(self, name: str, tensor) -> bool:
        """Validate a ``forward()`` tensor against the set-up structure and
        return whether it is shared (single-problem shape)."""
        self._check_vector(name, tensor)
        if name not in self._present:
            raise StructureMismatchError(
                f"'{name}' was not given at setup(); adding a block requires "
                "reset() and a new setup()."
            )
        single = self._single_shapes[name]
        batched = (self._core_solver.data.batch_size,) + single
        shape = tuple(tensor.shape)
        if shape == single:
            return True
        if shape == batched:
            return False
        raise StructureMismatchError(
            f"'{name}' must have shape {single} or {batched} as fixed by "
            f"setup(); got {shape}. Call reset() and setup() to change the "
            "structure."
        )

    # ------------------------------------------------------------------
    # Streams, lease, misc
    # ------------------------------------------------------------------

    def create_core_solver(self) -> None:
        """Create the core solver on the layer's device, with its own Warp
        stream, and the Torch view of that stream used for ordering."""
        self._core_solver = None          # release the previous one first
        self._warp_stream = wp.Stream(wp.device_from_torch(self._device))
        self._core_solver = self._core_solver_cls(
            dtype=_TORCH_TO_WARP[self._dtype], stream=self._warp_stream
        )
        self._core_solver.settings.enable_grad = True
        self._core_stream_view = wp.stream_to_torch(self._warp_stream)

    _STRUCTURAL_SETTINGS = ("enable_grad", "gradient_smoothing")

    def _current_structural_settings(self) -> tuple:
        return tuple(getattr(self.settings, name) for name in self._STRUCTURAL_SETTINGS)

    def _check_structural_settings(self) -> None:
        """Settings that size buffers or fix the differentiation mode are
        frozen by ``setup()`` in the core; reject a later change here rather
        than failing inside the solve or backward."""
        current = self._current_structural_settings()
        if current != self._structural_settings:
            changed = [name for name, a, b in zip(self._STRUCTURAL_SETTINGS,
                                                  self._structural_settings, current)
                       if a != b]
            raise RuntimeError(
                f"settings {changed} changed after setup(); they are fixed at "
                "setup(). Call reset() and setup() again to change them."
            )

    def _clear_problem_state(self) -> None:
        self._is_setup = False
        self._present = frozenset()
        self._single_shapes = {}
        self._result_fields = ()
        self._status_view = None
        self._iterations_view = None
        self._shared_grad_buffers = {}
        self._result = None
        self._structural_settings = None
        self._lease_token = None
        self._lease_counter = 0

    @contextlib.contextmanager
    def _ordered(self):
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

    def _acquire_lease(self) -> int:
        """Mark a differentiable solve outstanding; returns its token."""
        self._lease_counter += 1
        self._lease_token = self._lease_counter
        return self._lease_token

    def _release_lease(self, token: int) -> None:
        """End the outstanding solve identified by ``token``; a stale token
        (from a solve already released or superseded) is ignored."""
        if self._lease_token == token:
            self._lease_token = None

    def extra_repr(self) -> str:
        if not self._is_setup:
            return f"dtype={self._dtype}, not set up"
        data = self._core_solver.data
        return (f"dtype={self._dtype}, B={data.batch_size}, n={data.n}, "
                f"p={data.p}, m={data.m}")
