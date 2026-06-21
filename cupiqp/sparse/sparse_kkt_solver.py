from typing import Optional
import cupy as cp
import warp as wp
from cupyx.scipy.sparse import csr_matrix, diags, bmat
import nvtx

from ..kkt_solver import KKTSolverBase
from .batched_csr import UniformBatchedCsrMatrix
from .sparse_data import SparseData
from .sparse_matvec import SingleSparseMatVecProduct, BatchedSparseMatVecProduct
from .sparse_direct_solver import CudssSparseDirectSolver
from .csr_helpers import csr_diag_indices, csr_row_indices, csr_subblock_indices
from .sparse_kkt_solver_kernels import create_update_kkt_diag_kernel, create_scatter_masked_G_kernel


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
        self._dtype = data.dtype

        # -- Build KKT structure from the first problem's sparsity ------
        P0, A0, G0 = data.P[0], data.A[0], data.G[0]
        kkt_template = self._initialize_kkt_csr(P0, A0, G0, dtype=self._dtype)

        # -- Pack all KKT matrices in the batch into a UniformBatchedCsrMatrix,
        # which owns a contiguous (B, kkt_nnz) values buffer and shares a
        # single indices/indptr pair across batches. The initial values are
        # broadcast from the template.
        init_data = cp.empty((B, kkt_template.nnz), dtype=self._dtype)
        init_data[:] = kkt_template.data
        self._kkt_mats = UniformBatchedCsrMatrix(
            batch_size=B,
            indices=kkt_template.indices,
            indptr=kkt_template.indptr,
            data=init_data,
            dtype=self._dtype,
        )

        # -- Diagonal indices (shared — same structure) -----------------
        single_kkt_diag_idx = csr_diag_indices(self._kkt_mats[0])
        self._diag_x_indices = single_kkt_diag_idx[:n]
        self._diag_y_indices = single_kkt_diag_idx[n:n+p]
        self._diag_z_indices = single_kkt_diag_idx[n+p:n+p+m]

        self._update_kkt_diag_kernel = create_update_kkt_diag_kernel(n, p, m, dtype=self._dtype)
        self._scatter_masked_G_kernel = create_scatter_masked_G_kernel(dtype=self._dtype) if m > 0 else None

        # -- P-diagonal CSR indices (for vectorized P diag extraction) --
        # P may have zero diagonal entries that are not stored in the CSR.
        # An entry at position k is on the diagonal iff its row == its col;
        # we keep only those positions so update_kkt can gather without any
        # boolean-mask step (CUDA-graph safe).

        # For example, 
        # P = 
        # [5  0  8
        #  0  0  2
        #  0  1  4]
        # P.indices = [0 2 2 1 2]  (col indices)
        # P.indptr = [0 2 3 5]
        # P.data = [5 8 2 1 4]
        # We can compute P's row indices: [0 0 1 2 2]
        # (row_indices == col_indices) is [True False False False True], 
        # leading to self._indices_of_Pdata_containing_nonzero_diag_entry = [0, 4], meaning the 0th and 4th entries in P.data are diagonal elements,
        # and their row/col indices are P.indice[self._indices_of_Pdata_containing_nonzero_diag_entry] = [0 2], meaning these 2 diagonal elements are on the 0th and 2nd rows/cols
        P0_rows, P0_cols = csr_row_indices(P0), P0.indices
        self._indices_of_Pdata_containing_nonzero_diag_entry = cp.where(P0_rows == P0_cols)[0]
        self._cols_of_P_containing_nonzero_diag_entry = P0.indices[self._indices_of_Pdata_containing_nonzero_diag_entry]
        self._P_diag = cp.zeros((B, n), dtype=self._dtype)
        self._refresh_P_diag_buffer(data._P)

        # -- Block-to-KKT index maps (shared) --------------------------
        kkt0 = self._kkt_mats[0]
        self._P_indices = csr_subblock_indices(P0, kkt0, 0, 0)
        self._A_indices = csr_subblock_indices(A0, kkt0, n, 0) if p > 0 else None
        self._G_indices = csr_subblock_indices(G0, kkt0, n + p, 0) if m > 0 else None
        # Row index of each G non-zero, used to zero the G coupling of inactive
        # inequality rows (both bounds infinite) when scattering into the KKT.
        # Cast to int32 so the warp scatter kernel can index with it directly.
        self._G_row_idx = csr_row_indices(G0).astype(cp.int32) if m > 0 else None

        # -- Scatter initial P, A, G values (vectorized) ---------------
        self._scatter_P_A_G(data, update_P=True, update_A=(p > 0), update_G=(m > 0))

        # -- SpMV operators: single-matrix path for B=1 (no block-diagonal
        # overhead), batched block-diagonal path for B>1. data.P/A/G are
        # already UniformBatchedCsrMatrix instances (SparseData normalizes every
        # accepted input form to UniformBatchedCsrMatrix). ---------------
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
        # internally via the UniformBatchedCsrMatrix)
        self._lin_sys_solver = CudssSparseDirectSolver(
            self._kkt_mats, use_deterministic_mode=use_deterministic_mode,
        )
        if not self._lin_sys_solver.plan(cuda_stream=cp.cuda.get_current_stream().ptr):
            raise RuntimeError("Sparse direct solver planning failed.")

    def _refresh_P_diag_buffer(self, P: UniformBatchedCsrMatrix):
        self._P_diag[:, self._cols_of_P_containing_nonzero_diag_entry] = P.data[:, self._indices_of_Pdata_containing_nonzero_diag_entry]

    def __del__(self):
        solver = getattr(self, "_lin_sys_solver", None)
        if solver is not None:
            solver.__del__()

    @staticmethod
    def _initialize_kkt_csr(
        P: csr_matrix,
        A: Optional[csr_matrix] = None,
        G: Optional[csr_matrix] = None,
        dtype=cp.float64,
    ) -> csr_matrix:
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
        # UniformBatchedCsrMatrix, so we must operate on copies here to avoid
        # corrupting the shared structure.
        P = P.copy()
        A = A.copy() if p else A
        G = G.copy() if m else G
        In = diags(cp.ones(n, dtype=dtype), 0, shape=(n, n), format="csr")
        Ip = diags(cp.ones(p, dtype=dtype), 0, shape=(p, p), format="csr") if p else None
        Im = diags(cp.ones(m, dtype=dtype), 0, shape=(m, m), format="csr") if m else None
        # only store lower triangular part (but the full P is still stored)
        # TODO: store the lower triangular part of P only
        kkt = bmat([
                [P+In, None, None],
                [A,    Ip,   None],
                [G,    None, Im],
            ], format="csr", dtype=dtype
            )
        return kkt

    def _scatter_P_A_G(self, data: SparseData, update_P: bool, update_A: bool, update_G: bool) -> None:
        """Vectorized scatter of P / A / G values into the (B, kkt_nnz) buffer."""
        if update_P:
            self._kkt_mats.data[:, self._P_indices] = data._P.data
        if update_A and self._A_indices is not None:
            self._kkt_mats.data[:, self._A_indices] = data._A.data
        if update_G and self._G_indices is not None:
            # Scatter G into the KKT data buffer, zeroing the contribution of
            # inactive inequality rows (both bounds infinite) in place.
            wp.launch(
                kernel=self._scatter_masked_G_kernel,
                dim=(self._batch_size, self._G_indices.shape[0]),
                inputs=[
                    data._G.data, data.active_G_row,
                    self._G_row_idx, self._G_indices,
                    self._kkt_mats.data,
                ],
                device="cuda",
                stream=wp.Stream(cuda_stream=cp.cuda.get_current_stream().ptr),
            )

    def update_data(self, data: SparseData, update_P: bool, update_A: bool, update_G: bool) -> None:
        """Update the sparse KKT matrices when P, A, or G values change.

        Uses precomputed index maps to scatter new values into the
        (B, kkt_nnz) buffer without rebuilding the matrices.
        """
        self._scatter_P_A_G(data, update_P, update_A, update_G)
        if update_P:
            # refresh P_diag if P is updated
            self._refresh_P_diag_buffer(data._P)

    @nvtx.annotate("SparseKKTSolver::update_kkt")
    def update_kkt(self, data: SparseData, delta: cp.ndarray, x_reg: cp.ndarray, z_reg: cp.ndarray, z_reg_inv: cp.ndarray) -> None:
        """Update diagonal blocks of all batched KKT matrices."""
        wp.launch(
            kernel=self._update_kkt_diag_kernel,
            dim=(self._batch_size, data.n + data.p + data.m),
            inputs=[
                self._P_diag, x_reg, delta, z_reg,
                self._diag_x_indices, self._diag_y_indices, self._diag_z_indices,
                self._kkt_mats.data,
            ],
            device="cuda",
            stream=wp.Stream(cuda_stream=cp.cuda.get_current_stream().ptr),
        )

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
        itemsize = self._lin_sys_solver.rhs.itemsize  # dtype-dependent (4 for f32, 8 for f64)
        rhs_ptr = self._lin_sys_solver.rhs.data.ptr

        # Assemble [rhs_x | rhs_y | rhs_z] into solver's (B, dim) rhs buffer.
        # Each source may have a different row stride (non-contiguous view),
        # so we use memcpy2DAsync to scatter each block into the right columns.
        cp.cuda.runtime.memcpy2DAsync(
            rhs_ptr, dim * itemsize,
            rhs_x.data.ptr, rhs_x.strides[0],
            n * itemsize, B, 3, stream_ptr)
        if p > 0:
            cp.cuda.runtime.memcpy2DAsync(
                rhs_ptr + n * itemsize, dim * itemsize,
                rhs_y.data.ptr, rhs_y.strides[0],
                p * itemsize, B, 3, stream_ptr)
        if m > 0:
            cp.cuda.runtime.memcpy2DAsync(
                rhs_ptr + (n + p) * itemsize, dim * itemsize,
                rhs_z.data.ptr, rhs_z.strides[0],
                m * itemsize, B, 3, stream_ptr)

        self._lin_sys_solver.solve(cuda_stream=stream_ptr)

        # Disassemble solver's (B, dim) sol buffer into [delta_x, delta_y, delta_z].
        sol_ptr = self._lin_sys_solver.sol.data.ptr
        cp.cuda.runtime.memcpy2DAsync(
            delta_x.data.ptr, delta_x.strides[0],
            sol_ptr, dim * itemsize,
            n * itemsize, B, 3, stream_ptr)
        if p > 0:
            cp.cuda.runtime.memcpy2DAsync(
                delta_y.data.ptr, delta_y.strides[0],
                sol_ptr + n * itemsize, dim * itemsize,
                p * itemsize, B, 3, stream_ptr)
        if m > 0:
            cp.cuda.runtime.memcpy2DAsync(
                delta_z.data.ptr, delta_z.strides[0],
                sol_ptr + (n + p) * itemsize, dim * itemsize,
                m * itemsize, B, 3, stream_ptr)

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
