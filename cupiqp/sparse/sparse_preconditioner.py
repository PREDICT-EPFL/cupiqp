import torch
import cupy as cp
from cupyx.scipy.sparse import csr_matrix, linalg as sparse_la

from ..data import Data
from ..preconditioner import RuizEquilibration


class SparseRuizEquilibration(RuizEquilibration):
    """Ruiz equilibration for sparse (CSR) matrix backends."""

    def eval_P_row_inf_norms(self, P: csr_matrix, out: torch.Tensor):
        out[:] = torch.as_tensor(sparse_la.norm(P, ord=cp.inf, axis=1), device='cuda')

    def eval_A_row_inf_norms(self, A: csr_matrix, out: torch.Tensor):
        out[:] = torch.as_tensor(sparse_la.norm(A, ord=cp.inf, axis=1), device='cuda')

    def eval_A_col_inf_norms(self, A: csr_matrix, out: torch.Tensor):
        out[:] = torch.as_tensor(sparse_la.norm(A, ord=cp.inf, axis=0), device='cuda')

    def eval_G_row_inf_norms(self, G: csr_matrix, out: torch.Tensor):
        out[:] = torch.as_tensor(sparse_la.norm(G, ord=cp.inf, axis=1), device='cuda')

    def eval_G_col_inf_norms(self, G: csr_matrix, out: torch.Tensor):
        out[:] = torch.as_tensor(sparse_la.norm(G, ord=cp.inf, axis=0), device='cuda')

    def _scale_matrices(self, data: Data,
                        d_x: torch.Tensor, d_y: torch.Tensor, d_z: torch.Tensor):
        self._csr_row_scale(data._P, d_x)
        self._csr_col_scale(data._P, d_x)
        data._c *= d_x

        if self.p > 0:
            self._csr_row_scale(data._A, d_y)
            self._csr_col_scale(data._A, d_x)
        if self.m > 0:
            self._csr_row_scale(data._G, d_z)
            self._csr_col_scale(data._G, d_x)

    def _apply_cost_scaling(self, data: Data):
        P_norms = self._csr_utri_symmetric_col_inf_norms(data._P)
        gamma = float(cp.mean(P_norms))
        gamma = self._limit_scaling_scalar(gamma)
        gamma = max(gamma, float(torch.max(torch.abs(data._c))))
        gamma = self._limit_scaling_scalar(gamma)
        gamma = 1.0 / gamma
        data._P.data *= gamma
        data._c *= gamma
        self.c_scaling *= gamma

    def _unscale_matrices(self, data: Data,
                          d_x_inv: torch.Tensor, d_y_inv: torch.Tensor, d_z_inv: torch.Tensor):
        c_inv = float(self._c_scaling_inv)

        data._P.data *= c_inv
        self._csr_row_scale(data._P, d_x_inv)
        self._csr_col_scale(data._P, d_x_inv)
        data._c *= c_inv * d_x_inv

        if self.p > 0:
            self._csr_row_scale(data._A, d_y_inv)
            self._csr_col_scale(data._A, d_x_inv)
        if self.m > 0:
            self._csr_row_scale(data._G, d_z_inv)
            self._csr_col_scale(data._G, d_x_inv)

    def _apply_stored_scaling(self, data: Data,
                              d_x: torch.Tensor, d_y: torch.Tensor, d_z: torch.Tensor):
        c = float(self.c_scaling)

        data._P.data *= c
        self._csr_row_scale(data._P, d_x)
        self._csr_col_scale(data._P, d_x)
        data._c *= c * d_x

        if self.p > 0:
            self._csr_row_scale(data._A, d_y)
            self._csr_col_scale(data._A, d_x)
        if self.m > 0:
            self._csr_row_scale(data._G, d_z)
            self._csr_col_scale(data._G, d_x)


    # ------------------------------------------------------------------
    # CSR helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _csr_row_scale(M, d):
        """Scale rows of CSR matrix M by vector d. M is CuPy CSR, d may be torch or CuPy."""
        if M.shape[0] == 0 or M.nnz == 0:
            return
        d_cp = cp.asarray(d) if not isinstance(d, cp.ndarray) else d
        nz_indices = cp.arange(M.nnz, dtype=cp.int32)
        row_indices = cp.searchsorted(M.indptr[1:], nz_indices, side='right')
        M.data *= d_cp[row_indices]

    @staticmethod
    def _csr_col_scale(M, d):
        """Scale columns of CSR matrix M by vector d. M is CuPy CSR, d may be torch or CuPy."""
        if M.nnz == 0:
            return
        d_cp = cp.asarray(d) if not isinstance(d, cp.ndarray) else d
        M.data *= d_cp[M.indices]

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