"""Differentiable sparse QP layer for PyTorch."""

from __future__ import annotations

import torch
import warp as wp

from ...sparse.sparse_solver import SparseSolver
from .base import SolverLayerBase


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

    Values are ``(nnz,)`` or ``(B, nnz)`` tensors, and their gradients have
    the same shape. A 3-D CSR tensor at ``setup()`` must store the same
    pattern for every problem; the first one is used and the others are not
    checked against it. Changing a pattern requires ``reset()`` and a new
    ``setup()``.

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
        x_l: torch.Tensor | None = None
    ) -> None:
        """Fix the problem structure, including the CSR patterns of ``P``,
        ``A`` and ``G``, and load the values.

        ``P``, ``A`` and ``G`` are CUDA sparse CSR tensors of the layer dtype,
        2-D or 3-D with one shared pattern. Store the full ``P`` (both
        triangles). The vectors are dense tensors, shared or batched. Does
        not solve; call the layer afterwards.
        """
        self._setup(dict(P=P, c=c, A=A, b=b, G=G, h_u=h_u, h_l=h_l, x_u=x_u, x_l=x_l))

    def _check_setup_matrix(self, name: str, tensor) -> None:
        self._check_tensor(name, tensor)
        if tensor.layout != torch.sparse_csr:
            raise TypeError(
                f"'{name}' must be a CUDA sparse CSR tensor at setup(); got "
                f"layout {tensor.layout}."
            )
        if tensor.dim() not in (2, 3):
            raise ValueError(
                f"'{name}' must be 2-D or 3-D (batched); got shape {tuple(tensor.shape)}."
            )
        if name == "P" and tensor.shape[-1] != tensor.shape[-2]:
            raise ValueError(f"'P' must be square; got shape {tuple(tensor.shape)}.")

    def _setup_matrix_to_core(self, name: str, tensor):
        tensor = tensor.detach()
        crow, col = tensor.crow_indices(), tensor.col_indices()
        if tensor.dim() == 3:
            crow, col = crow[0], col[0]
        return (crow.contiguous(), col.contiguous(), tensor.values().contiguous())

    def _matrix_single_shape(self, name: str) -> tuple:
        return (getattr(self._core_solver.data, name).nnz,)

    def _matrix_gradient_array(self, name: str, gradients) -> wp.array:
        return getattr(gradients, name).data
