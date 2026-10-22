"""Differentiable sparse QP layer for PyTorch."""

from __future__ import annotations

from typing import Any

import torch
import warp as wp

from ...data import Data
from ...sparse.sparse_solver import SparseSolver
from .base import SolverLayerBase
from .utils import MATRIX_FIELDS, validate_cuda_tensor, validate_dense_tensor


class SparseSolverLayer(SolverLayerBase):
    """Differentiable QP layer backed by ``cupiqp.SparseSolver``.

    Solves the same QP as :class:`DenseSolverLayer` with sparse ``P``,
    ``A`` and ``G``. The sparsity patterns are part of the structure: they
    are given once, as CUDA sparse CSR tensors, to ``setup()``, and every
    later call takes only the stored values of each matrix, in the order of
    the CSR pattern::

        layer = cupiqp.torch.SparseSolverLayer()
        layer.setup(P_csr, c, A=A_csr, b=b)
        x = layer(P=P_csr.values(), c=c_k)      # values only
        x.sum().backward()                      # gradients for both

    ``setup()`` takes ``P``, ``A`` and ``G`` as CUDA sparse CSR tensors of
    the layer dtype, 2-D or 3-D with one shared pattern; store the full
    ``P`` (both triangles). The vectors are dense tensors, shared or
    batched. Values at every call are ``(nnz,)`` or ``(B, nnz)`` tensors,
    and their gradients have the same shape. A 3-D CSR tensor at ``setup()``
    must store the same pattern for every problem; the first one is used
    and the others are not checked against it. Changing a pattern requires
    ``reset()`` and a new ``setup()``. See :class:`SolverLayerBase` for
    ``max_pending_solves`` (several solves awaiting one backward, as in a
    rollout).

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

    _core_solver_cls = SparseSolver

    def _prepare_setup_inputs(self, inputs: dict[str, torch.Tensor | None]) -> dict[str, Any]:
        core_setup_args = {}
        for name, tensor in inputs.items():
            if tensor is None:
                core_setup_args[name] = None
            elif name in MATRIX_FIELDS:
                validate_cuda_tensor(name, tensor, self._device, self._dtype)
                if tensor.layout != torch.sparse_csr:
                    raise TypeError(
                        f"'{name}' must be a CUDA sparse CSR tensor at setup(); got "
                        f"layout {tensor.layout}."
                    )
                if tensor.dim() not in (2, 3):
                    raise ValueError(
                        f"'{name}' must be 2-D or 3-D (batched); got shape "
                        f"{tuple(tensor.shape)}."
                    )
                if name == "P" and tensor.shape[-1] != tensor.shape[-2]:
                    raise ValueError(f"'P' must be square; got shape {tuple(tensor.shape)}.")
                # A batched CSR tensor stores one pattern per problem; the
                # first one is used (all must be equal, see the class docs).
                tensor = tensor.detach()
                crow, col = tensor.crow_indices(), tensor.col_indices()
                if tensor.dim() == 3:
                    crow, col = crow[0], col[0]
                core_setup_args[name] = (crow.contiguous(), col.contiguous(),
                                         tensor.values().contiguous())
            else:
                validate_dense_tensor(name, tensor, self._device, self._dtype)
                if tensor.dim() not in (1, 2):
                    raise ValueError(
                        f"'{name}' must be 1-D or 2-D (batched); got shape "
                        f"{tuple(tensor.shape)}."
                    )
                core_setup_args[name] = tensor.detach().contiguous()
        return core_setup_args

    def _matrix_unbatched_shape(self, name: str) -> tuple[int, ...]:
        return (getattr(self._core_solver.data, name).nnz,)

    def _matrix_data_gradient_array(self, name: str, data_gradients: Data) -> wp.array:
        return getattr(data_gradients, name).data
