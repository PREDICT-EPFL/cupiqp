"""Differentiable dense QP layer for PyTorch."""

from __future__ import annotations

import torch
import warp as wp

from ...dense.dense_solver import DenseSolver
from .base import SolverLayerBase


class DenseSolverLayer(SolverLayerBase):
    """Differentiable QP layer backed by ``cupiqp.DenseSolver``.

    Solves, for one problem or a batch,

        minimize    0.5 x^T P x + c^T x
        subject to  A x = b
                    h_l <= G x <= h_u
                    x_l <= x <= x_u

    with dense CUDA tensors. ``setup()`` fixes the dimensions and which
    blocks exist and allocates all GPU memory; every later call updates the
    values it is given, solves, and returns ``x``::

        layer = cupiqp.torch.DenseSolverLayer()
        layer.setup(P, c, x_l=x_l, x_u=x_u)
        x = layer(c=c_k)                    # solve with a new linear cost
        x.sum().backward()                  # gradient for c_k

    Each matrix is ``(rows, cols)`` or ``(B, rows, cols)`` and each vector
    ``(k,)`` or ``(B, k)``; the batch size is read from the batched
    arguments at ``setup()``. See :class:`cupiqp.DenseSolver` for the
    meaning of every argument and for ``+/-inf`` bounds.

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

    _core_solver_cls = DenseSolver

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
        """Fix the problem structure from these tensors and load their values.

        Same arguments as ``cupiqp.DenseSolver.setup``: dense CUDA tensors of
        the layer dtype, each shared or batched. Does not solve; call the
        layer afterwards. Gradients are not tracked for ``setup()`` tensors,
        only for those passed to the layer call.
        """
        self._setup(dict(P=P, c=c, A=A, b=b, G=G, h_u=h_u, h_l=h_l, x_u=x_u, x_l=x_l))

    def _check_setup_matrix(self, name: str, tensor) -> None:
        self._check_vector(name, tensor)
        if tensor.dim() not in (2, 3):
            raise ValueError(
                f"'{name}' must be 2-D or 3-D (batched); got shape {tuple(tensor.shape)}."
            )
        if name == "P" and tensor.shape[-1] != tensor.shape[-2]:
            raise ValueError(f"'P' must be square; got shape {tuple(tensor.shape)}.")

    def _setup_matrix_to_core(self, name: str, tensor):
        return tensor.detach().contiguous()

    def _matrix_single_shape(self, name: str) -> tuple:
        data = self._core_solver.data
        rows = {"P": data.n, "A": data.p, "G": data.m}[name]
        return (rows, data.n)

    def _matrix_gradient_array(self, name: str, gradients) -> wp.array:
        return getattr(gradients, name)
