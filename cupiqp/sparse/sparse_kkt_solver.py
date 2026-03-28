from typing import Optional
import numpy as np
import cupy as cp
from cupyx.scipy.sparse import csr_matrix, diags, bmat
import nvtx

from ..kkt_solver import KKTSolverBase
from .sparse_data import SparseData
from .sparse_matvec import SparseMatVecProduct
from .sparse_direct_solver import CudssSparseDirectSolver


class SparseKKTSolver(KKTSolverBase):
    """
    Sparse KKT solver with LDLT factorization.
    """
    def __init__(self, data: SparseData, use_deterministic_mode: bool = False):
        super().__init__()
        self._kkt_mat = self._initialize_kkt_csr(data.P, data.A, data.G)
        n, p, m = data.n, data.p, data.m

        # pre-compute diagonal indices for efficient in-place updates
        self._diag_x_indices = cp.empty(n, dtype=cp.int32)
        self._diag_y_indices = cp.empty(p, dtype=cp.int32)
        self._diag_z_indices = cp.empty(m, dtype=cp.int32)
        self._find_diagonal_indices()

        # pre-compute block-to-KKT index maps for update_data()
        self._P_indices = self._build_block_index_map(data.P, self._kkt_mat, 0, 0)
        if p > 0:
            self._A_indices = self._build_block_index_map(data.A, self._kkt_mat, n, 0)
            self._AT_indices = self._build_transpose_index_map(data.A, self._kkt_mat, 0, n)
        if m > 0:
            self._G_indices = self._build_block_index_map(data.G, self._kkt_mat, n + p, 0)
            self._GT_indices = self._build_transpose_index_map(data.G, self._kkt_mat, 0, n + p)

        # setup spmv operator for evaluating P, A, G matvecs
        self._spmv_P = SparseMatVecProduct(data.P, transa=False)
        self._spmv_A = SparseMatVecProduct(data.A, transa=False)
        self._spmv_AT = SparseMatVecProduct(data.A, transa=True)
        self._spmv_G = SparseMatVecProduct(data.G, transa=False)
        self._spmv_GT = SparseMatVecProduct(data.G, transa=True)

        # setup direct solver for KKT factorization and solves
        self._lin_sys_solver = CudssSparseDirectSolver(self._kkt_mat, use_deterministic_mode=use_deterministic_mode)
        plan_success = self._lin_sys_solver.plan()  # symbolic factorization and reordering
        if not plan_success:
            raise RuntimeError("Sparse direct solver planning failed.")

        self._stream_cp = cp.cuda.get_current_stream()

    def __del__(self):
        self._lin_sys_solver.__del__()

    @staticmethod
    def _build_block_index_map(block: csr_matrix, kkt: csr_matrix,
                               row_offset: int, col_offset: int) -> cp.ndarray:
        """Map each non-zero of *block* to its position in *kkt*.data.

        ``block[i, j]`` sits at ``kkt[row_offset + i, col_offset + j]``.
        Returns a cupy int32 array of length ``block.nnz``.
        """
        b_indptr = block.indptr.get()
        b_indices = block.indices.get()
        k_indptr = kkt.indptr.get()
        k_indices = kkt.indices.get()

        idx_map = np.empty(block.nnz, dtype=np.int32)

        for i in range(block.shape[0]):
            kkt_row = row_offset + i
            kkt_start = k_indptr[kkt_row]
            kkt_end = k_indptr[kkt_row + 1]
            kkt_cols = k_indices[kkt_start:kkt_end]

            for k in range(b_indptr[i], b_indptr[i + 1]):
                target_col = col_offset + b_indices[k]
                local_pos = np.searchsorted(kkt_cols, target_col)
                idx_map[k] = kkt_start + local_pos

        return cp.asarray(idx_map)

    @staticmethod
    def _build_transpose_index_map(block: csr_matrix, kkt: csr_matrix,
                                   row_offset: int, col_offset: int) -> cp.ndarray:
        """Map each non-zero of *block* to the transposed position in *kkt*.data.

        ``block[i, j]`` (value ``block.data[k]``) corresponds to
        ``kkt[row_offset + j, col_offset + i]``  (the A^T / G^T entry).
        Returns a cupy int32 array of length ``block.nnz``.
        """
        b_indptr = block.indptr.get()
        b_indices = block.indices.get()
        k_indptr = kkt.indptr.get()
        k_indices = kkt.indices.get()

        idx_map = np.empty(block.nnz, dtype=np.int32)

        for i in range(block.shape[0]):
            for k in range(b_indptr[i], b_indptr[i + 1]):
                j = b_indices[k]
                # transposed: kkt[row_offset + j, col_offset + i]
                kkt_row = row_offset + j
                kkt_start = k_indptr[kkt_row]
                kkt_end = k_indptr[kkt_row + 1]
                kkt_cols = k_indices[kkt_start:kkt_end]
                local_pos = np.searchsorted(kkt_cols, col_offset + i)
                idx_map[k] = kkt_start + local_pos

        return cp.asarray(idx_map)

    def update_data(self, data: SparseData, update_P: bool, update_A: bool, update_G: bool) -> None:
        """Update the sparse KKT matrix when P, A, or G values change.

        Uses precomputed index maps to scatter new values into the
        correct positions of ``_kkt_mat.data`` without rebuilding the
        full matrix.
        """
        if update_P:
            self._kkt_mat.data[self._P_indices] = data.P.data
        if update_A:
            self._kkt_mat.data[self._A_indices] = data.A.data
            self._kkt_mat.data[self._AT_indices] = data.A.data
        if update_G:
            self._kkt_mat.data[self._G_indices] = data.G.data
            self._kkt_mat.data[self._GT_indices] = data.G.data

    @staticmethod
    def _initialize_kkt_csr(P: csr_matrix, A: Optional[csr_matrix] = None, G: Optional[csr_matrix] = None) -> csr_matrix:
        """
        Initialize the KKT matrix based on the sparsity of P, A, G.

        This builds a CSR matrix with a fixed sparsity pattern suitable for repeated
        numeric refactorizations. We intentionally insert identity diagonals into each
        diagonal block so later updates can use setdiag() without changing structure.
        """
        P = P.tocsr()
        n = P.shape[0]

        p = 0 if A is None else int(A.shape[0])
        m = 0 if G is None else int(G.shape[0])

        # Sparse diagonal placeholders (avoid cp.diag / cp.eye which create dense matrices)
        # Do P+In make sure the diagonal entries are non-zero 
        # P is p.s.d so P's diagonal are all non-negative, adding I will not change non-zeros entries to zero
        In = diags(cp.ones(n, dtype=cp.float64), 0, shape=(n, n), format="csr")
        Ip = diags(cp.ones(p, dtype=cp.float64), 0, shape=(p, p), format="csr") if p else None
        Im = diags(cp.ones(m, dtype=cp.float64), 0, shape=(m, m), format="csr") if m else None
        kkt = bmat([
                [P+In, A.T,  G.T],
                [A,    Ip,   None],
                [G,    None, Im],
            ], format="csr", dtype=cp.float64
            )
        return kkt
    
    def _find_diagonal_indices(self) -> None:
        """
        Find the positions of diagonal elements in the CSR data array.
        Returns a 1D array of indices into mat.data where diagonal elements are located.
        """
        dim = self._kkt_mat.shape[0]
        diag_idx = cp.empty(dim, dtype=cp.int32)
        for i in range(dim):
            row_start = int(self._kkt_mat.indptr[i])
            row_end = int(self._kkt_mat.indptr[i + 1])
            # find where column == i in this row
            for j in range(row_start, row_end):
                if int(self._kkt_mat.indices[j]) == i:
                    diag_idx[i] = j
                    break

        n, p, m = self._diag_x_indices.size, self._diag_y_indices.size, self._diag_z_indices.size
        self._diag_x_indices = diag_idx[:n]
        self._diag_y_indices = diag_idx[n : n+p]
        self._diag_z_indices = diag_idx[n+p : n+p+m]
    
    @nvtx.annotate("SparseKKTSolver::update_kkt")
    def update_kkt(self, data: SparseData, delta: cp.ndarray, x_reg: cp.ndarray, z_reg: cp.ndarray) -> None:
        self._kkt_mat.data[self._diag_x_indices] = data.P.diagonal()
        self._kkt_mat.data[self._diag_x_indices] += x_reg        
        self._kkt_mat.data[self._diag_y_indices] = -delta
        self._kkt_mat.data[self._diag_z_indices] = -z_reg
    
    @nvtx.annotate("SparseKKTSolver::factor")
    def factor(self) -> bool:
        return self._lin_sys_solver.factor()

    @nvtx.annotate("SparseKKTSolver::solve")
    def solve(self, data: SparseData, rhs_x: cp.ndarray, rhs_y: cp.ndarray, rhs_z: cp.ndarray, delta_x: cp.ndarray, delta_y: cp.ndarray, delta_z: cp.ndarray):
        """
        Solve the KKT system using the factorized KKT matrix.
        """
        # ! cp.cuda.runtime.memcpyAsync has lower launch overhead than multiple small cp.copyto() calls
        # self._rhs <= [rhs_x, rhs_y, rhs_z]
        cp.cuda.runtime.memcpyAsync(self._lin_sys_solver.rhs.data.ptr, rhs_x.data.ptr, data.n * 8, 1, self._stream_cp.ptr)
        cp.cuda.runtime.memcpyAsync(self._lin_sys_solver.rhs.data.ptr + data.n * 8, rhs_y.data.ptr, data.p * 8, 1, self._stream_cp.ptr)
        cp.cuda.runtime.memcpyAsync(self._lin_sys_solver.rhs.data.ptr + (data.n+data.p) * 8, rhs_z.data.ptr, data.m * 8, 1, self._stream_cp.ptr)

        self._lin_sys_solver.solve()

        # [delta_x, delta_y, delta_z] <= self._sol
        cp.cuda.runtime.memcpyAsync(delta_x.data.ptr, self._lin_sys_solver.sol.data.ptr, data.n * 8, 1, self._stream_cp.ptr)
        cp.cuda.runtime.memcpyAsync(delta_y.data.ptr, self._lin_sys_solver.sol.data.ptr + data.n * 8, data.p * 8, 1, self._stream_cp.ptr)
        cp.cuda.runtime.memcpyAsync(delta_z.data.ptr, self._lin_sys_solver.sol.data.ptr + (data.n+data.p) * 8, data.m * 8, 1, self._stream_cp.ptr)

    @nvtx.annotate("SparseKKTSolver::eval_P_x")
    def eval_P_x(self, data: SparseData, alpha: float, x: cp.ndarray, z: cp.ndarray):
        self._spmv_P(x, z, alpha=alpha, beta=0.0, stream_ptr=cp.cuda.get_current_stream().ptr)
    
    @nvtx.annotate("SparseKKTSolver::eval_A_xn")
    def eval_A_xn(self, data: SparseData, alpha_n: float, xn: cp.ndarray, zn: cp.ndarray):
        self._spmv_A(xn, zn, alpha=alpha_n, beta=0.0, stream_ptr=cp.cuda.get_current_stream().ptr)

    @nvtx.annotate("SparseKKTSolver::eval_AT_xt")
    def eval_AT_xt(self, data: SparseData, alpha_t: float, xt: cp.ndarray, zt: cp.ndarray):
        self._spmv_AT(xt, zt, alpha=alpha_t, beta=0.0, stream_ptr=cp.cuda.get_current_stream().ptr)

    @nvtx.annotate("SparseKKTSolver::eval_G_xn")
    def eval_G_xn(self, data: SparseData, alpha_n: float, xn: cp.ndarray, zn: cp.ndarray):
        self._spmv_G(xn, zn, alpha=alpha_n, beta=0.0, stream_ptr=cp.cuda.get_current_stream().ptr)

    @nvtx.annotate("SparseKKTSolver::eval_GT_xt")
    def eval_GT_xt(self, data: SparseData, alpha_t: float, xt: cp.ndarray, zt: cp.ndarray):
        self._spmv_GT(xt, zt, alpha=alpha_t, beta=0.0, stream_ptr=cp.cuda.get_current_stream().ptr)

