from typing import Optional, Union, List
import cupy as cp
from cupyx.scipy.sparse import csr_matrix, isspmatrix_csr

from ..data import Data
from ..typedef import PIQP_INF


class SparseData(Data):
    """Sparse data structure for batched QP problems.

    Matrices (P, A, G) are stored as lists of B CSR matrices — one per
    problem in the batch.  All matrices in a list must share the same
    sparsity pattern (same indptr and indices arrays).

    Dense vectors (c, b, h_l, h_u, x_l, x_u) carry a leading batch
    dimension ``(B, k)``, matching the dense backend convention.

    For a single problem (``B = 1``), pass a plain CSR matrix and 1-D
    vectors — they are automatically wrapped.
    """

    def __init__(
        self,
        P: Union[csr_matrix, List[csr_matrix]],
        c: cp.ndarray,
        A: Optional[Union[csr_matrix, List[csr_matrix]]] = None,
        b: Optional[cp.ndarray] = None,
        G: Optional[Union[csr_matrix, List[csr_matrix]]] = None,
        h_u: Optional[cp.ndarray] = None,
        h_l: Optional[cp.ndarray] = None,
        x_u: Optional[cp.ndarray] = None,
        x_l: Optional[cp.ndarray] = None,
    ):
        # -- P -----------------------------------------------------------
        P_list = self._to_csr_list(P, "P")
        B = len(P_list)
        n = P_list[0].shape[0]
        if P_list[0].shape[0] != P_list[0].shape[1]:
            raise ValueError("P must be a square matrix.")
        for i in range(1, B):
            self._check_same_sparsity(P_list[0], P_list[i])

        self._batch_size = B
        self._n = n
        self._P = P_list

        # -- c -----------------------------------------------------------
        c = self._to_batched_vec(c, B, n, "c")
        self._c = c

        # -- A, b --------------------------------------------------------
        if A is not None and b is not None:
            A_list = self._to_csr_list(A, "A")
            if len(A_list) != B:
                raise ValueError(f"A batch size ({len(A_list)}) != P batch size ({B})")
            for i in range(1, B):
                self._check_same_sparsity(A_list[0], A_list[i])
            p = A_list[0].shape[0]
            self._A = A_list
            self._b = self._to_batched_vec(b, B, p, "b")
        else:
            self._A = [csr_matrix((0, n), dtype=cp.float64) for _ in range(B)]
            self._b = cp.zeros((B, 0), dtype=cp.float64)

        # -- G, h_u, h_l ------------------------------------------------
        if G is not None:
            G_list = self._to_csr_list(G, "G")
            if len(G_list) != B:
                raise ValueError(f"G batch size ({len(G_list)}) != P batch size ({B})")
            for i in range(1, B):
                self._check_same_sparsity(G_list[0], G_list[i])
            self._G = G_list
        else:
            self._G = [csr_matrix((0, n), dtype=cp.float64) for _ in range(B)]

        m = self._G[0].shape[0]
        self._h_u = self._to_batched_vec(h_u, B, m, "h_u") if h_u is not None else cp.zeros((B, 0), dtype=cp.float64)
        self._h_l = self._to_batched_vec(h_l, B, m, "h_l") if h_l is not None else cp.zeros((B, 0), dtype=cp.float64)
        self._x_u = self._to_batched_vec(x_u, B, n, "x_u") if x_u is not None else cp.zeros((B, 0), dtype=cp.float64)
        self._x_l = self._to_batched_vec(x_l, B, n, "x_l") if x_l is not None else cp.zeros((B, 0), dtype=cp.float64)

        # For B > 1, pack each matrix type's data into a contiguous buffer
        # so that SparseMatVecProduct can build a block-diagonal descriptor
        # that sees in-place modifications automatically.
        if B > 1:
            self._P_packed = self._pack_data(self._P)
            if self.p > 0:
                self._A_packed = self._pack_data(self._A)
            if self.m > 0:
                self._G_packed = self._pack_data(self._G)

        self._finalize()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def p(self) -> int:
        return self._A[0].shape[0]

    @property
    def m(self) -> int:
        return self._G[0].shape[0]

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_csr_list(
        mat: Union[csr_matrix, List[csr_matrix]], name: str
    ) -> list:
        """Normalise a single CSR or list of CSRs to ``list[csr_matrix]``."""
        if isinstance(mat, (list, tuple)):
            if len(mat) == 0:
                raise ValueError(f"{name} cannot be an empty list.")
            return [csr_matrix(m, dtype=cp.float64) for m in mat]
        return [csr_matrix(mat, dtype=cp.float64)]

    @staticmethod
    def _to_batched_vec(
        v: cp.ndarray, B: int, k: int, name: str
    ) -> cp.ndarray:
        """Ensure *v* has shape ``(B, k)``."""
        v = cp.asarray(v, dtype=cp.float64)
        if v.ndim == 1:
            v = v.reshape(1, -1)
        if v.shape[0] == 1 and B > 1:
            v = cp.broadcast_to(v, (B, v.shape[1])).copy()
        if v.shape[0] != B:
            raise ValueError(
                f"{name} batch size ({v.shape[0]}) != expected ({B})"
            )
        return v

    @staticmethod
    def _pack_data(mat_list: list) -> cp.ndarray:
        """Pack B CSR matrices' data arrays into one contiguous buffer.

        Each matrix's ``.data`` attribute is reassigned to a view of the
        returned buffer so that later in-place modifications are visible
        to any cuSPARSE descriptor created from the packed pointer.
        """
        B = len(mat_list)
        nnz = mat_list[0].nnz
        packed = cp.empty(B * nnz, dtype=cp.float64)
        for b in range(B):
            view = packed[b * nnz : (b + 1) * nnz]
            view[:] = mat_list[b].data
            mat_list[b].data = view
        return packed

    @staticmethod
    def _as_float64_mat(M: Union[csr_matrix, None]) -> csr_matrix:
        if M is not None:
            return csr_matrix(M, dtype=cp.float64)
        return csr_matrix((0, 0), dtype=cp.float64)

    # ------------------------------------------------------------------
    # Overrides for batched CSR storage
    # ------------------------------------------------------------------

    def extract_P_diag(self, diag_P: cp.ndarray):
        """Extract diagonal of each P into *diag_P* — shape ``(B, n)``."""
        for b in range(self._batch_size):
            diag_P[b] = self._P[b].diagonal()

    def _disable_inf_constraints(self):
        """Zero out G rows where both h_l and h_u are infinite."""
        m = self.m
        if m == 0:
            return
        # Bound structure is consistent across batch — check batch 0
        free = (self._h_l[0] <= -PIQP_INF) & (self._h_u[0] >= PIQP_INF)
        if not bool(cp.any(free)):
            return
        free_idx = cp.where(free)[0]
        for b in range(self._batch_size):
            for i in free_idx.get():
                self._G[b][i, :] = 0.0
        self._h_l[:, free] = -1.0
        self._h_u[:, free] = 1.0

    # ------------------------------------------------------------------
    # Sparsity-pattern validation
    # ------------------------------------------------------------------

    @staticmethod
    def _check_same_sparsity(old: csr_matrix, new: csr_matrix):
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

    # ------------------------------------------------------------------
    # In-place setters
    # ------------------------------------------------------------------

    def set_P(self, value, check: bool = True):
        vals = self._to_csr_list(value, "P") if not isinstance(value, list) else value
        for b in range(self._batch_size):
            if check:
                self._check_same_sparsity(self._P[b], vals[b])
            self._P[b].data[:] = vals[b].data

    def set_c(self, value: cp.ndarray, check: bool = True):
        if check and value.shape != self._c.shape:
            raise ValueError(f"c shape mismatch: expected {self._c.shape}, got {value.shape}")
        self._c[:] = value

    def set_A(self, value, check: bool = True):
        vals = self._to_csr_list(value, "A") if not isinstance(value, list) else value
        for b in range(self._batch_size):
            if check:
                self._check_same_sparsity(self._A[b], vals[b])
            self._A[b].data[:] = vals[b].data

    def set_b(self, value: cp.ndarray, check: bool = True):
        if check and value.shape != self._b.shape:
            raise ValueError(f"b shape mismatch: expected {self._b.shape}, got {value.shape}")
        self._b[:] = value

    def set_G(self, value, check: bool = True):
        vals = self._to_csr_list(value, "G") if not isinstance(value, list) else value
        for b in range(self._batch_size):
            if check:
                self._check_same_sparsity(self._G[b], vals[b])
            self._G[b].data[:] = vals[b].data

    def set_h_l(self, value: cp.ndarray, check: bool = True):
        if check and value.shape != self._h_l.shape:
            raise ValueError(f"h_l shape mismatch: expected {self._h_l.shape}, got {value.shape}")
        self._h_l[:] = value

    def set_h_u(self, value: cp.ndarray, check: bool = True):
        if check and value.shape != self._h_u.shape:
            raise ValueError(f"h_u shape mismatch: expected {self._h_u.shape}, got {value.shape}")
        self._h_u[:] = value

    def set_x_l(self, value: cp.ndarray, check: bool = True):
        if check and value.shape != self._x_l.shape:
            raise ValueError(f"x_l shape mismatch: expected {self._x_l.shape}, got {value.shape}")
        self._x_l[:] = value

    def set_x_u(self, value: cp.ndarray, check: bool = True):
        if check and value.shape != self._x_u.shape:
            raise ValueError(f"x_u shape mismatch: expected {self._x_u.shape}, got {value.shape}")
        self._x_u[:] = value
