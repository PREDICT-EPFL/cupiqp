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

        # -- Pack all KKT matrices in the batch into contiguous (B, kkt_nnz) buffer ----
        self._kkt_data = cp.empty((B, kkt_template.nnz), dtype=cp.float64)
        self._kkt_mats = []
        for b in range(B):
            row = self._kkt_data[b]
            row[:] = kkt_template.data
            kkt = csr_matrix(
                (row, kkt_template.indices.copy(), kkt_template.indptr.copy()),
                shape=kkt_template.shape,
                copy=False,
            )
            self._kkt_mats.append(kkt)

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

        # -- SpMV operators (block-diagonal for B>1, single for B=1) ----
        self._spmv_P = SparseMatVecProduct(data.P, transa=False)
        self._spmv_A = SparseMatVecProduct(data.A, transa=False)
        self._spmv_AT = SparseMatVecProduct(data.A, transa=True)
        self._spmv_G = SparseMatVecProduct(data.G, transa=False)
        self._spmv_GT = SparseMatVecProduct(data.G, transa=True)

        # -- Direct solver (nvmath explicit batching for B>1) -----------
        kkt_input = self._kkt_mats if B > 1 else self._kkt_mats[0]
        self._lin_sys_solver = CudssSparseDirectSolver(kkt_input, use_deterministic_mode=use_deterministic_mode)
        if not self._lin_sys_solver.plan(cuda_stream=cp.cuda.get_current_stream().ptr):
            raise RuntimeError("Sparse direct solver planning failed.")

        # -- Workspace for P diagonal extraction ------------------------
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

    # ------------------------------------------------------------------
    # Data update (vectorized over batch)
    # ------------------------------------------------------------------

    def _get_packed_2d(self, data: SparseData, attr: str, mat_list_attr: str) -> cp.ndarray:
        """Return a (B, nnz) view of packed data, or build it on the fly for B=1."""
        if hasattr(data, attr):
            return getattr(data, attr).reshape(self._batch_size, -1)
        # B = 1 fallback
        mat_list = getattr(data, mat_list_attr)
        return mat_list[0].data.reshape(1, -1)

    def _scatter_data(self, data: SparseData, update_P: bool, update_A: bool, update_G: bool) -> None:
        """Vectorized scatter of P / A / G values into the (B, kkt_nnz) buffer."""
        if update_P:
            P_2d = self._get_packed_2d(data, '_P_packed', '_P')
            self._kkt_data[:, self._P_indices] = P_2d
        if update_A:
            A_2d = self._get_packed_2d(data, '_A_packed', '_A')
            self._kkt_data[:, self._A_indices] = A_2d
            self._kkt_data[:, self._AT_indices] = A_2d
        if update_G:
            G_2d = self._get_packed_2d(data, '_G_packed', '_G')
            self._kkt_data[:, self._G_indices] = G_2d
            self._kkt_data[:, self._GT_indices] = G_2d

    def update_data(self, data: SparseData, update_P: bool, update_A: bool, update_G: bool) -> None:
        """Update the sparse KKT matrices when P, A, or G values change.

        Uses precomputed index maps to scatter new values into the
        (B, kkt_nnz) buffer without rebuilding the matrices.
        """
        self._scatter_data(data, update_P, update_A, update_G)

    # ------------------------------------------------------------------
    # KKT update / factor / solve
    # ------------------------------------------------------------------

    @nvtx.annotate("SparseKKTSolver::update_kkt")
    def update_kkt(self, data: SparseData, delta: cp.ndarray, x_reg: cp.ndarray, z_reg: cp.ndarray) -> None:
        """Update diagonal blocks of all B KKT matrices (vectorized).

        Parameters: delta (B,), x_reg (B, n), z_reg (B, m).
        """
        # Extract P diagonals: (B, n) — vectorized via packed P data
        # Only read entries that actually exist in the CSR; missing diagonals stay 0.
        P_2d = self._get_packed_2d(data, '_P_packed', '_P')
        self._P_diag[:] = 0.0
        self._P_diag[:, self._P_diag_var_idx] = P_2d[:, self._P_diag_csr_idx]

        # Scatter into (B, kkt_nnz): 3 kernel launches total
        self._kkt_data[:, self._diag_x_indices] = self._P_diag + x_reg
        self._kkt_data[:, self._diag_y_indices] = -delta[:, None]
        self._kkt_data[:, self._diag_z_indices] = -z_reg

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

        for b in range(self._batch_size):
            solver_rhs = self._lin_sys_solver.rhs[b]
            # Assemble [rhs_x, rhs_y, rhs_z] into solver's contiguous rhs buffer
            cp.cuda.runtime.memcpyAsync(solver_rhs.data.ptr, rhs_x[b].data.ptr, n * 8, 1, stream_ptr)
            if p > 0:
                cp.cuda.runtime.memcpyAsync(solver_rhs.data.ptr + n * 8, rhs_y[b].data.ptr, p * 8, 1, stream_ptr)
            if m > 0:
                cp.cuda.runtime.memcpyAsync(solver_rhs.data.ptr + (n+p) * 8, rhs_z[b].data.ptr, m * 8, 1, stream_ptr)

        self._lin_sys_solver.solve(cuda_stream=stream_ptr)

        for b in range(self._batch_size):
            solver_sol = self._lin_sys_solver.sol[b]
            # Disassemble solver's solution into [delta_x, delta_y, delta_z]
            cp.cuda.runtime.memcpyAsync(delta_x[b].data.ptr, solver_sol.data.ptr, n * 8, 1, stream_ptr)
            if p > 0:
                cp.cuda.runtime.memcpyAsync(delta_y[b].data.ptr, solver_sol.data.ptr + n * 8, p * 8, 1, stream_ptr)
            if m > 0:
                cp.cuda.runtime.memcpyAsync(delta_z[b].data.ptr, solver_sol.data.ptr + (n+p) * 8, m * 8, 1, stream_ptr)

    # ------------------------------------------------------------------
    # Sparse matrix-vector products (single kernel for all B)
    # ------------------------------------------------------------------
    # The block-diagonal cuSPARSE SpMV treats x and y as flat contiguous
    # vectors [v_0 | v_1 | ... | v_{B-1}].  Variables views (e.g.
    # result.x) are non-contiguous column slices of a larger buffer, so
    # we must ensure contiguity before calling cuSPARSE.

    def _spmv_safe(self, spmv, x, z, alpha, beta):
        stream_ptr = cp.cuda.get_current_stream().ptr
        x_c = cp.ascontiguousarray(x) if not x.flags['C_CONTIGUOUS'] else x
        z_needs_copy = not z.flags['C_CONTIGUOUS']
        z_c = cp.empty(z.shape, dtype=z.dtype) if z_needs_copy else z
        spmv(x_c, z_c, alpha=alpha, beta=beta, stream_ptr=stream_ptr)
        if z_needs_copy:
            z[:] = z_c

    @nvtx.annotate("SparseKKTSolver::eval_P_x")
    def eval_P_x(self, data: SparseData, alpha: float, x: cp.ndarray, z: cp.ndarray):
        self._spmv_safe(self._spmv_P, x, z, alpha, 0.0)

    @nvtx.annotate("SparseKKTSolver::eval_A_xn")
    def eval_A_xn(self, data: SparseData, alpha_n: float, xn: cp.ndarray, zn: cp.ndarray):
        self._spmv_safe(self._spmv_A, xn, zn, alpha_n, 0.0)

    @nvtx.annotate("SparseKKTSolver::eval_AT_xt")
    def eval_AT_xt(self, data: SparseData, alpha_t: float, xt: cp.ndarray, zt: cp.ndarray):
        self._spmv_safe(self._spmv_AT, xt, zt, alpha_t, 0.0)

    @nvtx.annotate("SparseKKTSolver::eval_G_xn")
    def eval_G_xn(self, data: SparseData, alpha_n: float, xn: cp.ndarray, zn: cp.ndarray):
        self._spmv_safe(self._spmv_G, xn, zn, alpha_n, 0.0)

    @nvtx.annotate("SparseKKTSolver::eval_GT_xt")
    def eval_GT_xt(self, data: SparseData, alpha_t: float, xt: cp.ndarray, zt: cp.ndarray):
        self._spmv_safe(self._spmv_GT, xt, zt, alpha_t, 0.0)
