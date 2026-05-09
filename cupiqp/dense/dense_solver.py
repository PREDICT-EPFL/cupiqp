from ..solver import SolverBase
from ..utils import is_cuda_array
from .dense_data import DenseData
from .dense_preconditioner import DenseRuizEquilibration


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

    def __init__(self):
        super().__init__()
        self.settings.kkt_solver = "dense_cholesky"

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
        return DenseData(P, c, A, b, G, h_u, h_l, x_u, x_l)

    def _init_preconditioner(self):
        return DenseRuizEquilibration(
            self._data.batch_size, self._data.n, self._data.p, self._data.m,
            self._data.idx_xl, self._data.idx_xu,
            self._data.idx_hl, self._data.idx_hu,
            use_warp_tile_kernels=True,
        )
