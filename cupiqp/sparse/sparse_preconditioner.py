import cupy as cp
from cupyx.scipy.sparse import csr_matrix, linalg as sparse_la

from ..data import Data
from ..preconditioner import RuizEquilibration


def _squeeze(arr: cp.ndarray) -> cp.ndarray:
    """Squeeze leading batch dim for B=1: (1, k) → (k,). No-op if already 1D."""
    return arr[0] if arr.ndim == 2 and arr.shape[0] == 1 else arr


class SparseRuizEquilibration(RuizEquilibration):
    """Ruiz equilibration for sparse (CSR) matrix backends.

    NOTE: Sparse backend currently supports B=1 only. Scaling vectors from
    the batched base class are squeezed to 1D before applying to CSR matrices.
    """

    def eval_P_row_inf_norms(self, P: csr_matrix, out: cp.ndarray):
        _squeeze(out)[:] = sparse_la.norm(P, ord=cp.inf, axis=1)

    def eval_A_row_inf_norms(self, A: csr_matrix, out: cp.ndarray):
        _squeeze(out)[:] = sparse_la.norm(A, ord=cp.inf, axis=1)

    def eval_A_col_inf_norms(self, A: csr_matrix, out: cp.ndarray):
        _squeeze(out)[:] = sparse_la.norm(A, ord=cp.inf, axis=0)

    def eval_G_row_inf_norms(self, G: csr_matrix, out: cp.ndarray):
        _squeeze(out)[:] = sparse_la.norm(G, ord=cp.inf, axis=1)

    def eval_G_col_inf_norms(self, G: csr_matrix, out: cp.ndarray):
        _squeeze(out)[:] = sparse_la.norm(G, ord=cp.inf, axis=0)

    def _scale_matrices(self, data: Data,
                        d_x: cp.ndarray, d_y: cp.ndarray, d_z: cp.ndarray):
        dx, dy, dz = _squeeze(d_x), _squeeze(d_y), _squeeze(d_z)
        self._csr_row_scale(data._P, dx)
        self._csr_col_scale(data._P, dx)
        data._c *= d_x  # (B, n) *= (B, n) works

        if self.p > 0:
            self._csr_row_scale(data._A, dy)
            self._csr_col_scale(data._A, dx)
        if self.m > 0:
            self._csr_row_scale(data._G, dz)
            self._csr_col_scale(data._G, dx)

    def _apply_cost_scaling(self, data: Data):
        P_norms = self._csr_utri_symmetric_col_inf_norms(data._P)
        gamma = float(cp.mean(P_norms))
        gamma = self._limit_scaling_scalar(gamma)
        gamma = max(gamma, float(cp.max(cp.abs(data._c))))
        gamma = self._limit_scaling_scalar(gamma)
        gamma = 1.0 / gamma
        data._P.data *= gamma
        data._c *= gamma
        self.c_scaling *= gamma

    def _unscale_matrices(self, data: Data,
                          d_x_inv: cp.ndarray, d_y_inv: cp.ndarray, d_z_inv: cp.ndarray):
        c_inv = float(self._c_scaling_inv[0])
        dxi, dyi, dzi = _squeeze(d_x_inv), _squeeze(d_y_inv), _squeeze(d_z_inv)

        data._P.data *= c_inv
        self._csr_row_scale(data._P, dxi)
        self._csr_col_scale(data._P, dxi)
        data._c *= c_inv * d_x_inv

        if self.p > 0:
            self._csr_row_scale(data._A, dyi)
            self._csr_col_scale(data._A, dxi)
        if self.m > 0:
            self._csr_row_scale(data._G, dzi)
            self._csr_col_scale(data._G, dxi)

    def _apply_stored_scaling(self, data: Data,
                              d_x: cp.ndarray, d_y: cp.ndarray, d_z: cp.ndarray):
        c = float(self.c_scaling[0])
        dx, dy, dz = _squeeze(d_x), _squeeze(d_y), _squeeze(d_z)

        data._P.data *= c
        self._csr_row_scale(data._P, dx)
        self._csr_col_scale(data._P, dx)
        data._c *= c * d_x

        if self.p > 0:
            self._csr_row_scale(data._A, dy)
            self._csr_col_scale(data._A, dx)
        if self.m > 0:
            self._csr_row_scale(data._G, dz)
            self._csr_col_scale(data._G, dx)

    # ------------------------------------------------------------------
    # CSR helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _csr_row_scale(M, d):
        """Scale rows of CSR matrix M by 1D vector d."""
        if M.shape[0] == 0 or M.nnz == 0:
            return
        nz_indices = cp.arange(M.nnz, dtype=cp.int32)
        row_indices = cp.searchsorted(M.indptr[1:], nz_indices, side='right')
        M.data *= d[row_indices]

    @staticmethod
    def _csr_col_scale(M, d):
        """Scale columns of CSR matrix M by 1D vector d."""
        if M.nnz == 0:
            return
        M.data *= d[M.indices]

    @staticmethod
    def _csr_row_inf_norms(M):
        if M.shape[0] == 0 or M.nnz == 0:
            return cp.zeros(M.shape[0], dtype=cp.float64)
        return cp.asarray(abs(M).max(axis=1).toarray()).ravel()

    @staticmethod
    def _csr_col_inf_norms(M):
        if M.shape[1] == 0 or M.nnz == 0:
            return cp.zeros(M.shape[1], dtype=cp.float64)
        return cp.asarray(abs(M).max(axis=0).toarray()).ravel()

    @classmethod
    def _csr_utri_symmetric_col_inf_norms(cls, P):
        if P.nnz == 0:
            return cp.zeros(P.shape[0], dtype=cp.float64)
        return cp.maximum(cls._csr_row_inf_norms(P), cls._csr_col_inf_norms(P))
