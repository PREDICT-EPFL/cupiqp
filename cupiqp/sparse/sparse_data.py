from typing import Optional, Union
import torch
import cupy as cp
from cupyx.scipy.sparse import csr_matrix, isspmatrix_csr


from ..data import Data
from ..typedef import PIQP_INF


class SparseData(Data):
    """
    Sparse data structure for optimization problems.
    """
    def __init__(self,
                 P: csr_matrix,
                 c: cp.ndarray,
                 A: Optional[csr_matrix] = None,
                 b: Optional[cp.ndarray] = None,
                 G: Optional[csr_matrix] = None,
                 h_u: Optional[cp.ndarray] = None,
                 h_l: Optional[cp.ndarray] = None,
                 x_u: Optional[cp.ndarray] = None,
                 x_l: Optional[cp.ndarray] = None):
        super().__init__(P, c, A, b, G, h_u, h_l, x_u, x_l)

    def _as_float64_mat(self, M: Union[csr_matrix, None]) -> csr_matrix:
        return csr_matrix(M, dtype=cp.float64) if M is not None else csr_matrix((0, self.n), dtype=cp.float64)

    @staticmethod
    def _check_same_sparsity(old: csr_matrix, new: csr_matrix):
        """Raise ValueError if sparsity pattern (indptr, indices) changed.

        NOTE: cp.array_equal causes D2H synchronisation.  Pass check=False
        to the set_* methods in the hot path to skip this validation.
        """
        if not isspmatrix_csr(new):
            raise ValueError(f"Expected csr_matrix, got {type(new)}")
        if new.shape != old.shape:
            raise ValueError(f"Shape changed: expected {old.shape}, got {new.shape}")
        if new.nnz != old.nnz:
            raise ValueError(f"Nnz changed: expected {old.nnz}, got {new.nnz}")
        if not cp.array_equal(new.indptr, old.indptr):
            raise ValueError("Sparsity pattern changed (indptr mismatch)")
        if not cp.array_equal(new.indices, old.indices):
            raise ValueError("Sparsity pattern changed (indices mismatch)")
        
    @staticmethod
    def _check_same_dimension(old: cp.ndarray, new: cp.ndarray):
        """Raise ValueError if dimension changed.  Check is optional (default: True)."""
        if new.shape != old.shape:
            raise ValueError(
                f"Dimension changed: expected {old.shape}, got {new.shape}"
            )
        
    def disable_inf_constraints(self):
        """
        For inequalities like -inf < g'x < +inf, set g to 0 and upper/lower bound to +1/-1.
        Override base class: self._G is a CuPy CSR matrix, so we use cp.zeros for row zeroing,
        while self._h_l and self._h_u are torch tensors from the parent class.
        """
        for i in range(self.m):
            if self._h_l[i] <= -PIQP_INF and self._h_u[i] >= PIQP_INF:
                self._G[i, :] = cp.zeros(self.n)
                self._h_l[i] = -1.
                self._h_u[i] = 1.

    def extract_P_diag(self, diag_P: torch.Tensor):
        diag_P[:] = torch.as_tensor(self._P.diagonal(), device='cuda')

    def set_P(self, value: csr_matrix, check: bool = True):
        if check:
            self._check_same_sparsity(self._P, value)
        self._P.data[:] = value.data

    def set_c(self, value: cp.ndarray, check: bool = True):
        if check:
            self._check_same_dimension(self._c, value)
        self._c[:] = value

    def set_A(self, value: csr_matrix, check: bool = True):
        if check:
            self._check_same_sparsity(self._A, value)
        self._A.data[:] = value.data

    def set_b(self, value: cp.ndarray, check: bool = True):
        if check:
            self._check_same_dimension(self._b, value)
        self._b[:] = value

    def set_G(self, value: csr_matrix, check: bool = True):
        if check:
            self._check_same_sparsity(self._G, value)
        self._G.data[:] = value.data

    def set_h_l(self, value: cp.ndarray, check: bool = True):
        if check:
            self._check_same_dimension(self._h_l, value)
        self._h_l[:] = value

    def set_h_u(self, value: cp.ndarray, check: bool = True):
        if check:
            self._check_same_dimension(self._h_u, value)
        self._h_u[:] = value

    def set_x_l(self, value: cp.ndarray, check: bool = True):
        if check:
            self._check_same_dimension(self._x_l, value)
        self._x_l[:] = value

    def set_x_u(self, value: cp.ndarray, check: bool = True):
        if check:
            self._check_same_dimension(self._x_u, value)
        self._x_u[:] = value
