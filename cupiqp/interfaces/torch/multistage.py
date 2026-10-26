"""Differentiable multistage QP layer for PyTorch."""

from __future__ import annotations

from typing import Any

import torch
import warp as wp

from ...data import Data
from ...multistage.multistage_solver import MultistageSolver
from .base import SolverLayerBase
from .utils import MATRIX_FIELDS, validate_dense_tensor

# Forward field of every block: (matrix argument of the core, 0 for the
# diagonal blocks or 1 for the off-diagonal ones).
_BLOCK_OF_FIELD = {
    "P_diag": ("P", 0), "P_offdiag": ("P", 1),
    "A_diag": ("A", 0), "A_offdiag": ("A", 1),
    "G_diag": ("G", 0), "G_offdiag": ("G", 1),
}


class MultistageSolverLayer(SolverLayerBase):
    """Differentiable QP layer backed by ``cupiqp.MultistageSolver``.

    Solves the same QP as :class:`DenseSolverLayer` when ``P`` is block
    tridiagonal and ``A`` and ``G`` are block lower bidiagonal, as in optimal
    control over ``N`` stages of ``d`` variables (``n = N*d``). ``setup()``
    takes the arguments of ``cupiqp.MultistageSolver.setup``: each matrix is
    a ``(diag, offdiag)`` pair of CUDA tensors, and an ``offdiag`` given as
    ``None`` means that matrix has no coupling blocks for the lifetime of
    the layer::

        layer = cupiqp.torch.MultistageSolverLayer()
        layer.setup(P=(P_diag, None), c=c, A=(A_diag, A_offdiag), b=b,
                    x_l=x_l, x_u=x_u)
        x = layer(c=c_k, A_offdiag=dyn_k)   # x is (B, N*d)
        x.sum().backward()                   # gradients for c_k and dyn_k

    Every call takes one tensor per block, so each block can be passed,
    shared or batched, and differentiated on its own: ``P_diag (N, d, d)``,
    ``P_offdiag (N-1, d, d)``, ``A_diag``, ``A_offdiag``, ``G_diag`` and
    ``G_offdiag`` ``(N, r, d)``, each optionally with a leading batch axis.
    Vectors are flat, ``(k,)`` or ``(B, k)``, like the solution; a vector in
    block layout ``(N, d)`` is passed as ``c.reshape(-1)``, a view. A block
    absent at ``setup()`` cannot be passed later. ``per_call_fields`` names
    these call arguments, e.g. ``("c", "A_offdiag")``. See
    :class:`cupiqp.MultistageSolver` for the block layout and
    :class:`SolverLayerBase` for ``max_pending_solves``.

    Parameters
    ----------
    dtype : torch.dtype, default: torch.float64
        ``torch.float32`` or ``torch.float64``, used for all data and results.
    stream : torch.cuda.Stream, optional
        Torch stream to order the solver with. By default the layer orders
        itself with Torch's current stream at every call.
    device : torch.device or str, optional
        CUDA device to solve on. Defaults to the stream's device, else
        Torch's current CUDA device. All tensors must live on it.
    """

    _core_solver_cls = MultistageSolver
    _FORWARD_FIELDS = ("P_diag", "P_offdiag", "c", "A_diag", "A_offdiag", "b",
                       "G_diag", "G_offdiag", "h_u", "h_l", "x_u", "x_l")
    _MATRIX_FORWARD_FIELDS = frozenset(_BLOCK_OF_FIELD)

    def forward(
        self,
        P_diag: torch.Tensor | None = None,
        P_offdiag: torch.Tensor | None = None,
        c: torch.Tensor | None = None,
        A_diag: torch.Tensor | None = None,
        A_offdiag: torch.Tensor | None = None,
        b: torch.Tensor | None = None,
        G_diag: torch.Tensor | None = None,
        G_offdiag: torch.Tensor | None = None,
        h_u: torch.Tensor | None = None,
        h_l: torch.Tensor | None = None,
        x_u: torch.Tensor | None = None,
        x_l: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Load the given blocks and vectors, solve, and return the primal
        solution ``x`` of shape ``(B, N*d)``.

        Arguments left as ``None`` keep their current values. Each tensor is
        shared, in the single-problem shape, or batched with a leading batch
        axis. Gradients flow to every tensor passed here that requires them.
        The returned tensor is overwritten by the next call (see
        :class:`SolverLayerBase`); ``layer.result`` holds the duals, slacks
        and the per-problem status.
        """
        return self._forward_impl((P_diag, P_offdiag, c, A_diag, A_offdiag, b,
                                   G_diag, G_offdiag, h_u, h_l, x_u, x_l))

    def _prepare_setup_inputs(self, inputs: dict[str, Any]) -> dict[str, Any]:
        core_setup_args = {}
        for name, value in inputs.items():
            if value is None:
                core_setup_args[name] = None
            elif name in MATRIX_FIELDS:
                if not (isinstance(value, (tuple, list)) and len(value) == 2):
                    raise TypeError(
                        f"'{name}' must be a (diag, offdiag) pair of tensors at setup(); "
                        f"got {type(value).__name__}."
                    )
                if value[0] is None:
                    raise ValueError(
                        f"'{name}' needs its diagonal blocks; only the offdiag may be None."
                    )
                blocks = []
                for part, tensor in zip(("diag", "offdiag"), value):
                    if tensor is None:
                        blocks.append(None)
                        continue
                    field = f"{name}_{part}"
                    validate_dense_tensor(field, tensor, self._device, self._dtype)
                    if tensor.dim() not in (3, 4):
                        raise ValueError(
                            f"'{field}' must be 3-D or 4-D (batched); got shape "
                            f"{tuple(tensor.shape)}."
                        )
                    blocks.append(tensor.detach().contiguous())
                core_setup_args[name] = tuple(blocks)
            else:
                validate_dense_tensor(name, value, self._device, self._dtype)
                if value.dim() not in (1, 2, 3):
                    raise ValueError(
                        f"'{name}' must be flat (k,) or in block layout, optionally "
                        f"batched; got shape {tuple(value.shape)}."
                    )
                core_setup_args[name] = value.detach().contiguous()
        return core_setup_args

    def _forward_fields_given(self, inputs: dict[str, Any]) -> frozenset[str]:
        given = set()
        for name, value in inputs.items():
            if value is None:
                continue
            if name in MATRIX_FIELDS:
                given.add(f"{name}_diag")
                if isinstance(value, (tuple, list)) and len(value) == 2 and value[1] is not None:
                    given.add(f"{name}_offdiag")
            else:
                given.add(name)
        return frozenset(given)

    def _core_update_args(self, field_tensors: dict[str, torch.Tensor]) -> dict[str, Any]:
        # A block not passed stays None in its pair: the core update() keeps
        # its current value.
        core_args = {}
        pairs = {}
        for name, tensor in field_tensors.items():
            if name in _BLOCK_OF_FIELD:
                matrix, index = _BLOCK_OF_FIELD[name]
                pairs.setdefault(matrix, [None, None])[index] = tensor
            else:
                core_args[name] = tensor
        for matrix, (diag, offdiag) in pairs.items():
            core_args[matrix] = (diag, offdiag)
        return core_args

    def _matrix_unbatched_shape(self, name: str) -> tuple[int, ...]:
        return tuple(getattr(self._core_solver.data, name).shape[1:])

    def _matrix_data_gradient_array(self, name: str, data_gradients: Data) -> wp.array:
        return getattr(data_gradients, name)
