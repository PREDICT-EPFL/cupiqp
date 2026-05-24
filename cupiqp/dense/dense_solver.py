import cupy as cp
import warp as wp

from ..results import Variables
from typing import Literal

from ..settings import Settings
from ..solver import SolverBase
from ..utils import is_cuda_array
from .dense_data import DenseData
from .dense_preconditioner import DenseRuizEquilibration
from .dense_solver_kernels import create_dense_data_gradients_kernel


def _check_dense(name: str, m) -> None:
    """Validate that ``m`` is a GPU dense array (skip if ``None``).

    Used for **all** P / c / A / b / G / h_* / x_* inputs in
    :class:`DenseSolver` — every ``setup`` argument that's not ``None``
    must be a GPU array exposing the CUDA Array Interface protocol.
    See :func:`cupiqp.utils.is_cuda_array`.
    """
    if m is None:
        return
    if not is_cuda_array(m):
        raise TypeError(
            f"DenseSolver requires {name} to be a GPU dense array "
            f"(any object exposing __cuda_array_interface__: "
            f"cupy.ndarray, dense CUDA torch.Tensor, JAX CUDA array, etc.); "
            f"got {type(m).__name__}."
        )


class DenseSolver(SolverBase):
    """Concrete :class:`SolverBase` subclass for the **dense Cholesky**
    KKT backend.

    ``DenseSolver`` is the type-strict, user-facing entry point for
    solving QPs whose problem data are **GPU-resident dense arrays**.
    It rejects non-dense ``P`` / ``A`` / ``G`` inputs (and any non-GPU
    vectors) with a clear, actionable :class:`TypeError`.

    Accepts any object that exposes the
    `__cuda_array_interface__ <https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html>`_
    protocol — this unifies cupy, dense CUDA :class:`torch.Tensor`, JAX
    CUDA arrays, numba CUDA device arrays, etc. behind a single check.

    cupiqp is GPU-only. CPU-only arrays (:class:`numpy.ndarray`, CPU
    torch tensors, CPU JAX arrays) are **rejected** rather than silently
    copied onto the GPU. Convert to a CUDA array explicitly first if
    you have CPU data::

        P_cuda = cupy.asarray(P_numpy)                  # cupy
        P_cuda = torch.tensor(P_numpy, device="cuda")   # torch
        s.setup(P=P_cuda, ...)

    Examples
    --------
    >>> import cupy as cp
    >>> from cupiqp import DenseSolver
    >>> P = cp.eye(4); c = cp.zeros(4)
    >>> s = DenseSolver()
    >>> s.setup(P=P, c=c)
    >>> s.solve()
    """

    def __init__(self, dtype: Literal["float32", "float64"] = "float64"):
        super().__init__(dtype=dtype)
        self._settings.kkt_solver = "dense_cholesky"

    @SolverBase.settings.setter
    def settings(self, value: Settings) -> None:
        # TODO: here we have to set the kkt solver back. That's pretty ugly. Should be improved in the future
        value.kkt_solver = "dense_cholesky"
        self._settings = value


    def _init_data(self, P, c, A, b, G, h_u, h_l, x_u, x_l):
        # Every non-None input must be a GPU dense array — matrices and
        # vectors alike. cupiqp does not silently do H2D copies.
        _check_dense("P", P)
        _check_dense("c", c)
        _check_dense("A", A)
        _check_dense("b", b)
        _check_dense("G", G)
        _check_dense("h_u", h_u)
        _check_dense("h_l", h_l)
        _check_dense("x_u", x_u)
        _check_dense("x_l", x_l)
        data = DenseData(dtype=self.settings.dtype, device=self.settings.device)
        data.init(P, c, A, b, G, h_u, h_l, x_u, x_l)
        return data

    def _init_preconditioner(self):
        return DenseRuizEquilibration(
            self._data.batch_size, self._data.n, self._data.p, self._data.m,
            self._data.idx_xl, self._data.idx_xu,
            self._data.idx_hl, self._data.idx_hu,
            use_warp_tile_kernels=True,
            dtype=self._data.dtype,
        )

    def setup(self, P, c, A=None, b=None, G=None,
              h_u=None, h_l=None, x_u=None, x_l=None):
        super().setup(P, c, A, b, G, h_u, h_l, x_u, x_l)
        if self.settings.enable_grad:
            d = self._data
            B = d.batch_size
            dtype = d.dtype
            self._dense_data_gradients_kernel = create_dense_data_gradients_kernel(
                d.n, d.p, d.m, dtype=dtype)
            self._grad_data = DenseData(dtype=dtype, device=self.settings.device)
            self._grad_data.init(
                P=cp.zeros((B, d.n, d.n), dtype=dtype),
                c=cp.zeros((B, d.n), dtype=dtype),
                A=cp.zeros((B, d.p, d.n), dtype=dtype) if d.p > 0 else None,
                b=cp.zeros((B, d.p), dtype=dtype) if d.p > 0 else None,
                G=cp.zeros((B, d.m, d.n), dtype=dtype) if d.m > 0 else None,
                h_u=cp.zeros((B, d.m), dtype=dtype) if d.num_hu > 0 else None,
                h_l=cp.zeros((B, d.m), dtype=dtype) if d.num_hl > 0 else None,
                x_u=cp.zeros((B, d.n), dtype=dtype) if d.num_xu > 0 else None,
                x_l=cp.zeros((B, d.n), dtype=dtype) if d.num_xl > 0 else None,
            )

    def _compute_data_gradients(self, adjoint_vector: Variables) -> DenseData:
        """Populate ``self._grad_data`` in place and return it.

        The returned instance is the same on every call; its buffers are
        overwritten by the next backward. Copy fields if you need to keep
        them across calls.
        """
        data = self._data
        grad_data = self._grad_data
        B = data.batch_size
        total = (data.n * data.n + data.p * data.n + data.m * data.n
                 + data.p + data.m + data.n)
        if total > 0:
            wp.launch(
                kernel=self._dense_data_gradients_kernel,
                dim=(B, total),
                inputs=[
                    adjoint_vector.x, adjoint_vector.y,
                    self._lam_zu_full, self._lam_zl_full,
                    self._lam_zbu_full,
                    self._zu_full, self._zl_full,
                    self._result.x, self._result.y,
                    grad_data._P, grad_data._A, grad_data._G,
                    grad_data._b, grad_data._h_u, grad_data._x_u,
                ],
                device="cuda",
                stream=wp.Stream(cuda_stream=cp.cuda.get_current_stream().ptr),
            )

        # Vector grads that are aliases of solver-internal buffers — copy in.
        grad_data._c[:] = adjoint_vector.x
        if data.num_hl > 0:
            grad_data._h_l[:] = self._lam_zl_full
        if data.num_xl > 0:
            grad_data._x_l[:] = self._lam_zbl_full

        return grad_data
