from typing import Optional
import numpy as np
import cupy as cp
from cupyx.scipy.sparse import csr_matrix, diags, bmat
import nvtx

from ..kkt_solver import KKTSolverBase
from .batched_csr import BatchedCsrMatrix
from .sparse_data import SparseData
from .sparse_matvec import SingleSparseMatVecProduct, BatchedSparseMatVecProduct
from .sparse_direct_solver import CudssSparseDirectSolver


class SparseKKTSolver(KKTSolverBase):
    """Sparse KKT solver with LDLT factorization — batched.

    Manages B independent KKT systems sharing the same sparsity pattern.
    All B KKT matrices' data are packed into a contiguous ``(B, kkt_nnz)``
    buffer so that diagonal / block updates are vectorized (no Python loop).
    """

    def __init__(self, data: SparseData, use_deterministic_mode: bool = False):
        super().__init__()
        B = data.batch_size
        self._batch_size = B
        n, p, m = data.n, data.p, data.m

        # -- Build KKT structure from the first problem's sparsity ------
        P0, A0, G0 = data.P[0], data.A[0], data.G[0]
        kkt_template = self._initialize_kkt_csr(P0, A0, G0)

        # -- Pack all KKT matrices in the batch into a BatchedCsrMatrix,
        # which owns a contiguous (B, kkt_nnz) values buffer and shares a
        # single indices/indptr pair across batches. The initial values are
        # broadcast from the template.
        init_data = cp.empty((B, kkt_template.nnz), dtype=cp.float64)
        init_data[:] = kkt_template.data
        self._kkt_mats = BatchedCsrMatrix(
            batch_size=B,
            indices=kkt_template.indices,
            indptr=kkt_template.indptr,
            data=init_data,
        )

        # -- Diagonal indices (shared — same structure) -----------------
        single_kkt_diag_idx = self._find_csr_diag_indices(self._kkt_mats[0])
        self._diag_x_indices = single_kkt_diag_idx[:n]
        self._diag_y_indices = single_kkt_diag_idx[n:n+p]
        self._diag_z_indices = single_kkt_diag_idx[n+p:n+p+m]

        # -- P-diagonal CSR indices (for vectorized P diag extraction) --
        # P may have zero diagonal entries that are not stored in the CSR.
        # We precompute integer index arrays for the existing entries so that
        # update_kkt can gather without boolean masks (CUDA-graph safe).
        P0_indptr = P0.indptr.get()
        P0_indices = P0.indices.get()
        P_diag_csr_idx = []   # CSR data-array positions of existing diagonals
        P_diag_var_idx = []   # variable indices (0..n-1) that have a diagonal entry
        for i in range(n):
            for k in range(P0_indptr[i], P0_indptr[i + 1]):
                if P0_indices[k] == i:
                    P_diag_csr_idx.append(k)
                    P_diag_var_idx.append(i)
                    break
        self._P_diag_csr_idx = cp.asarray(P_diag_csr_idx, dtype=cp.int32)
        self._P_diag_var_idx = cp.asarray(P_diag_var_idx, dtype=cp.int32)

        # -- Block-to-KKT index maps (shared) --------------------------
        kkt0 = self._kkt_mats[0]
        self._P_indices = self._build_block_index_map(P0, kkt0, 0, 0)
        if p > 0:
            self._A_indices = self._build_block_index_map(A0, kkt0, n, 0)
            self._AT_indices = self._build_transpose_index_map(A0, kkt0, 0, n)
        if m > 0:
            self._G_indices = self._build_block_index_map(G0, kkt0, n + p, 0)
            self._GT_indices = self._build_transpose_index_map(G0, kkt0, 0, n + p)

        # -- Scatter initial P, A, G values (vectorized) ---------------
        self._scatter_data(data, update_P=True, update_A=(p > 0), update_G=(m > 0))

        # -- SpMV operators: single-matrix path for B=1 (no block-diagonal
        # overhead), batched block-diagonal path for B>1. data.P/A/G are
        # already BatchedCsrMatrix instances (SparseData normalizes every
        # accepted input form to BatchedCsrMatrix). --------------------
        if B == 1:
            self._spmv_P = SingleSparseMatVecProduct(data.P[0], transa=False)
            if p > 0:
                self._spmv_A = SingleSparseMatVecProduct(data.A[0], transa=False)
                self._spmv_AT = SingleSparseMatVecProduct(data.A[0], transa=True)
            if m > 0:
                self._spmv_G = SingleSparseMatVecProduct(data.G[0], transa=False)
                self._spmv_GT = SingleSparseMatVecProduct(data.G[0], transa=True)
        else:
            self._spmv_P = BatchedSparseMatVecProduct(data.P, transa=False)
            if p > 0:
                self._spmv_A = BatchedSparseMatVecProduct(data.A, transa=False)
                self._spmv_AT = BatchedSparseMatVecProduct(data.A, transa=True)
            if m > 0:
                self._spmv_G = BatchedSparseMatVecProduct(data.G, transa=False)
                self._spmv_GT = BatchedSparseMatVecProduct(data.G, transa=True)

        # Direct solver (CudssSparseDirectSolver handles B==1 and B>1
        # internally via the BatchedCsrMatrix)
        self._lin_sys_solver = CudssSparseDirectSolver(
            self._kkt_mats, use_deterministic_mode=use_deterministic_mode,
        )
        if not self._lin_sys_solver.plan(cuda_stream=cp.cuda.get_current_stream().ptr):
            raise RuntimeError("Sparse direct solver planning failed.")

        # Workspace for P diagonal extraction
        self._P_diag = cp.empty((B, n), dtype=cp.float64)

    def __del__(self):
        solver = getattr(self, "_lin_sys_solver", None)
        if solver is not None:
            solver.__del__()

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

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
        #
        # NOTE: cupyx's CSR addition sorts the operand's indices/data buffers
        # *in place* as a side effect (csrgeam pre-condition). ``data.P[0]``
        # is a view that shares its ``indices`` array with the backing
        # BatchedCsrMatrix, so we must operate on copies here to avoid
        # corrupting the shared structure.
        P = P.copy()
        A = A.copy() if p else A
        G = G.copy() if m else G
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

    @staticmethod
    def _find_csr_diag_indices(mat: csr_matrix) -> cp.ndarray:
        """Find positions of diagonal entries within a CSR matrix's data array.

        Returns a cupy int32 array of length ``min(rows, cols)``.
        """
        assert isinstance(mat, csr_matrix)
        assert mat.shape[0] == mat.shape[1], "The provided csr_matrix is not square. Got shape: {mat.shape}"
        indptr = mat.indptr.get()
        indices = mat.indices.get()
        n = mat.shape[0]
        diag_idx = cp.empty(n, dtype=cp.int32)
        for i in range(n):
            for k in range(indptr[i], indptr[i + 1]):
                if indices[k] == i:
                    diag_idx[i] = k
                    break
        return diag_idx

    def _scatter_data(self, data: SparseData, update_P: bool, update_A: bool, update_G: bool) -> None:
        """Vectorized scatter of P / A / G values into the (B, kkt_nnz) buffer."""
        if update_P:
            self._kkt_mats.data[:, self._P_indices] = data._P.data
        if update_A:
            self._kkt_mats.data[:, self._A_indices] = data._A.data
            self._kkt_mats.data[:, self._AT_indices] = data._A.data
        if update_G:
            self._kkt_mats.data[:, self._G_indices] = data._G.data
            self._kkt_mats.data[:, self._GT_indices] = data._G.data

    def update_data(self, data: SparseData, update_P: bool, update_A: bool, update_G: bool) -> None:
        """Update the sparse KKT matrices when P, A, or G values change.

        Uses precomputed index maps to scatter new values into the
        (B, kkt_nnz) buffer without rebuilding the matrices.
        """
        self._scatter_data(data, update_P, update_A, update_G)

    @nvtx.annotate("SparseKKTSolver::update_kkt")
    def update_kkt(self, data: SparseData, delta: cp.ndarray, x_reg: cp.ndarray, z_reg: cp.ndarray) -> None:
        """Update diagonal blocks of all B KKT matrices (vectorized).

        Parameters: delta (B,), x_reg (B, n), z_reg (B, m).
        """
        # Extract P diagonals: (B, n) — vectorized via packed P data.
        # Only read entries that actually exist in the CSR; missing diagonals stay 0.
        self._P_diag[:] = 0.0
        self._P_diag[:, self._P_diag_var_idx] = data._P.data[:, self._P_diag_csr_idx]

        # Scatter into (B, kkt_nnz): 3 kernel launches total
        self._kkt_mats.data[:, self._diag_x_indices] = self._P_diag + x_reg
        self._kkt_mats.data[:, self._diag_y_indices] = -delta[:, None]
        self._kkt_mats.data[:, self._diag_z_indices] = -z_reg

    @nvtx.annotate("SparseKKTSolver::factor")
    def factor(self) -> bool:
        return self._lin_sys_solver.factor(cuda_stream=cp.cuda.get_current_stream().ptr)

    @nvtx.annotate("SparseKKTSolver::solve")
    def solve(self, data: SparseData, rhs_x: cp.ndarray, rhs_y: cp.ndarray, rhs_z: cp.ndarray,
              delta_x: cp.ndarray, delta_y: cp.ndarray, delta_z: cp.ndarray):
        """Solve the KKT system for every problem in the batch.

        `rhs_*` and `delta_*` should have shape (B, *) where B is the batch size.
        """
        stream_ptr = cp.cuda.get_current_stream().ptr
        n, p, m = data.n, data.p, data.m
        B = self._batch_size
        dim = n + p + m  # cuDSS rhs/sol row length
        rhs_ptr = self._lin_sys_solver.rhs.data.ptr

        # Assemble [rhs_x | rhs_y | rhs_z] into solver's (B, dim) rhs buffer.
        # Each source may have a different row stride (non-contiguous view),
        # so we use memcpy2DAsync to scatter each block into the right columns.
        cp.cuda.runtime.memcpy2DAsync(
            rhs_ptr, dim * 8,
            rhs_x.data.ptr, rhs_x.strides[0],
            n * 8, B, 3, stream_ptr)
        if p > 0:
            cp.cuda.runtime.memcpy2DAsync(
                rhs_ptr + n * 8, dim * 8,
                rhs_y.data.ptr, rhs_y.strides[0],
                p * 8, B, 3, stream_ptr)
        if m > 0:
            cp.cuda.runtime.memcpy2DAsync(
                rhs_ptr + (n + p) * 8, dim * 8,
                rhs_z.data.ptr, rhs_z.strides[0],
                m * 8, B, 3, stream_ptr)

        self._lin_sys_solver.solve(cuda_stream=stream_ptr)

        # Disassemble solver's (B, dim) sol buffer into [delta_x, delta_y, delta_z].
        sol_ptr = self._lin_sys_solver.sol.data.ptr
        cp.cuda.runtime.memcpy2DAsync(
            delta_x.data.ptr, delta_x.strides[0],
            sol_ptr, dim * 8,
            n * 8, B, 3, stream_ptr)
        if p > 0:
            cp.cuda.runtime.memcpy2DAsync(
                delta_y.data.ptr, delta_y.strides[0],
                sol_ptr + n * 8, dim * 8,
                p * 8, B, 3, stream_ptr)
        if m > 0:
            cp.cuda.runtime.memcpy2DAsync(
                delta_z.data.ptr, delta_z.strides[0],
                sol_ptr + (n + p) * 8, dim * 8,
                m * 8, B, 3, stream_ptr)

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
