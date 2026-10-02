from typing import Optional
import cupy as cp
import warp as wp

from cupyx.scipy.sparse import csr_matrix, diags, bmat
import nvtx

from ..kkt_solver import KKTSolverBase
from ..utils import column_slice
from .batched_csr import UniformBatchedCsrMatrix, to_wp_int32
from .sparse_data import SparseData
from .sparse_matvec import SparseMatVecProduct
from .sparse_direct_solver import CudssSparseDirectSolver
from .csr_helpers import csr_diag_indices, csr_row_indices, csr_subblock_indices
from .sparse_kkt_solver_kernels import (
    create_update_kkt_diag_kernel,
    create_scatter_values_kernel,
    create_gather_scatter_values_kernel,
    create_scatter_masked_G_kernel
)


class SparseKKTSolver(KKTSolverBase):
    """Sparse KKT solver with LDLT factorization - batched.

    Manages B independent KKT systems sharing the same sparsity pattern.
    All B KKT matrices' data are packed into a contiguous ``(B, kkt_nnz)``
    Warp buffer so that diagonal / block updates are single kernel launches.
    The KKT pattern and the index maps are built once at setup with cupy;
    everything that runs afterwards is Warp kernels, cuSPARSE and cuDSS on
    Warp storage.
    """

    def __init__(self, data: SparseData, use_deterministic_mode: bool = False):
        B = data.batch_size
        super().__init__(B, data.dtype, data.device)
        self._batch_size = B
        n, p, m = data.n, data.p, data.m
        self._dtype = data.dtype
        self._device = device = data.device

        # -- Build KKT structure from the first problem's sparsity (cupy views) --
        P0, A0, G0 = data.P[0], data.A[0], data.G[0]
        kkt_template = self._initialize_kkt_csr(P0, A0, G0, dtype=wp.dtype_to_numpy(data.dtype))

        # -- Pack all KKT matrices in the batch into one UniformBatchedCsrMatrix
        # (contiguous (B, kkt_nnz) values, shared indices/indptr), initialized
        # from the template values.
        self._kkt_mats = UniformBatchedCsrMatrix(
            batch_size=B,
            indptr=kkt_template.indptr,
            indices=kkt_template.indices,
            data=kkt_template.data,
            shape=kkt_template.shape,
            dtype=data.dtype,
            device=device
        )
        kkt0 = self._kkt_mats[0]

        # -- Diagonal indices (shared - same structure) -----------------
        single_kkt_diag_idx = csr_diag_indices(kkt0)
        self._diag_x_indices = to_wp_int32(single_kkt_diag_idx[:n], device)
        self._diag_y_indices = to_wp_int32(single_kkt_diag_idx[n:n + p], device)
        self._diag_z_indices = to_wp_int32(single_kkt_diag_idx[n + p:n + p + m], device)

        self._update_kkt_diag_kernel = create_update_kkt_diag_kernel(n, p, m, dtype=self._dtype)
        self._scatter_values_kernel = create_scatter_values_kernel(dtype=self._dtype)
        self._gather_scatter_values_kernel = create_gather_scatter_values_kernel(dtype=self._dtype)
        self._scatter_masked_G_kernel = create_scatter_masked_G_kernel(dtype=self._dtype) if m > 0 else None

        # -- P-diagonal CSR indices (for vectorized P diag extraction) --
        # P may have zero diagonal entries that are not stored in the CSR.
        # An entry at position k is on the diagonal iff its row == its col;
        # we keep only those positions so update_kkt can gather without any
        # boolean-mask step (CUDA-graph safe).
        #
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
        # so the 0th and 4th entries of P.data are diagonal elements, and they
        # belong to rows/cols P.indices[[0, 4]] = [0, 2].
        P0_rows, P0_cols = csr_row_indices(P0), P0.indices
        diag_positions = cp.where(P0_rows == P0_cols)[0]
        self._P_diag_src_cols = to_wp_int32(diag_positions, device)           # positions in P.data
        self._P_diag_dst_cols = to_wp_int32(P0_cols[diag_positions], device)  # variable index
        self._P_diag = wp.zeros((B, n), dtype=self._dtype, device=device)
        self._refresh_P_diag_buffer(data.P)

        # -- Block-to-KKT index maps (shared) --------------------------
        self._P_indices = to_wp_int32(csr_subblock_indices(P0, kkt0, 0, 0), device)
        self._A_indices = to_wp_int32(csr_subblock_indices(A0, kkt0, n, 0), device) if p > 0 else None
        self._G_indices = to_wp_int32(csr_subblock_indices(G0, kkt0, n + p, 0), device) if m > 0 else None
        # Row index of each G non-zero, used to zero the G coupling of inactive
        # inequality rows (both bounds infinite) when scattering into the KKT.
        self._G_row_idx = data.G.row_indices if m > 0 else None

        # -- Scatter initial P, A, G values ------------------------------
        self._scatter_P_A_G(data, update_P=True, update_A=(p > 0), update_G=(m > 0))

        # -- SpMV operators (one block-diagonal SpMV per call; B = 1 uses the
        # matrix pattern as is) ------------------------------------------
        self._spmv_P = SparseMatVecProduct(data.P, transa=False)
        if p > 0:
            self._spmv_A = SparseMatVecProduct(data.A, transa=False)
            self._spmv_AT = SparseMatVecProduct(data.A, transa=True)
        if m > 0:
            self._spmv_G = SparseMatVecProduct(data.G, transa=False)
            self._spmv_GT = SparseMatVecProduct(data.G, transa=True)

        # Direct solver (cuDSS uniform batching handles B == 1 and B > 1)
        self._lin_sys_solver = CudssSparseDirectSolver(
            self._kkt_mats, use_deterministic_mode=use_deterministic_mode
        )
        if not self._lin_sys_solver.plan(cuda_stream=wp.get_stream("cuda").cuda_stream):
            raise RuntimeError("Sparse direct solver planning failed.")

        self._factor_status = self._lin_sys_solver.factor_status

    def _refresh_P_diag_buffer(self, P: UniformBatchedCsrMatrix) -> None:
        """P_diag[b, col] = P.data[b, pos] for every stored diagonal entry."""
        num = int(self._P_diag_src_cols.shape[0])
        if num > 0:
            wp.launch(
                kernel=self._gather_scatter_values_kernel,
                dim=(self._batch_size, num),
                inputs=[P.data, self._P_diag_src_cols, self._P_diag_dst_cols, self._P_diag],
                device=self._device
            )

    def __del__(self):
        solver = getattr(self, "_lin_sys_solver", None)
        if solver is not None:
            solver.__del__()

    @staticmethod
    def _initialize_kkt_csr(
        P: csr_matrix,
        A: Optional[csr_matrix] = None,
        G: Optional[csr_matrix] = None,
        dtype=cp.float64
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
        # *in place* as a side effect (csrgeam pre-condition). The inputs are
        # views of the solver's Warp buffers, so we operate on copies here to
        # avoid corrupting the shared structure.
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
        """Scatter P / A / G values into the (B, kkt_nnz) buffer."""
        B = self._batch_size
        if update_P and data.P.nnz > 0:
            wp.launch(
                kernel=self._scatter_values_kernel,
                dim=(B, data.P.nnz),
                inputs=[data.P.data, self._P_indices, self._kkt_mats.data],
                device=self._device
            )
        if update_A and self._A_indices is not None and data.A.nnz > 0:
            wp.launch(
                kernel=self._scatter_values_kernel,
                dim=(B, data.A.nnz),
                inputs=[data.A.data, self._A_indices, self._kkt_mats.data],
                device=self._device
            )
        if update_G and self._G_indices is not None and data.G.nnz > 0:
            # Scatter G into the KKT data buffer, zeroing the contribution of
            # inactive inequality rows (both bounds infinite) in place.
            wp.launch(
                kernel=self._scatter_masked_G_kernel,
                dim=(B, data.G.nnz),
                inputs=[
                    data.G.data, data.active_G_row,
                    self._G_row_idx, self._G_indices,
                    self._kkt_mats.data,
                ],
                device=self._device
            )

    def update_data(self, data: SparseData, update_P: bool, update_A: bool, update_G: bool) -> None:
        """Update the sparse KKT matrices when P, A, or G values change.

        Uses precomputed index maps to scatter new values into the
        (B, kkt_nnz) buffer without rebuilding the matrices.
        """
        self._scatter_P_A_G(data, update_P, update_A, update_G)
        if update_P:
            # refresh P_diag if P is updated
            self._refresh_P_diag_buffer(data.P)

    @nvtx.annotate("SparseKKTSolver::update_kkt")
    def update_kkt(self, data: SparseData, delta: wp.array, x_reg: wp.array, z_reg: wp.array, z_reg_inv: wp.array) -> None:
        """Update diagonal blocks of all batched KKT matrices."""
        wp.launch(
            kernel=self._update_kkt_diag_kernel,
            dim=(self._batch_size, data.n + data.p + data.m),
            inputs=[
                self._P_diag, x_reg, delta, z_reg,
                self._diag_x_indices, self._diag_y_indices, self._diag_z_indices,
                self._kkt_mats.data,
            ],
            device=self._device
        )

    @property
    def supports_conditional_capture(self) -> bool:
        # The cuDSS solve phase copies from pageable host memory, which a
        # conditional graph node rejects (see CudssSparseDirectSolver.solve).
        return False

    @nvtx.annotate("SparseKKTSolver::factor")
    def factor(self) -> None:
        self._lin_sys_solver.factor(cuda_stream=wp.get_stream("cuda").cuda_stream)

    @nvtx.annotate("SparseKKTSolver::solve")
    def solve(self, data: SparseData, rhs_x: wp.array, rhs_y: wp.array, rhs_z: wp.array,
              delta_x: wp.array, delta_y: wp.array, delta_z: wp.array) -> None:
        """Solve the KKT system for every problem in the batch.

        ``rhs_*`` and ``delta_*`` have shape ``(B, k)`` (row-strided views are fine).
        """
        n, p, m = data.n, data.p, data.m
        rhs, sol = self._lin_sys_solver.rhs, self._lin_sys_solver.sol

        # TODO: merge these 3 kernels
        # Assemble [rhs_x | rhs_y | rhs_z] into the solver's (B, dim) rhs buffer.
        wp.copy(column_slice(rhs, 0, n), rhs_x)
        if p > 0:
            wp.copy(column_slice(rhs, n, n + p), rhs_y)
        if m > 0:
            wp.copy(column_slice(rhs, n + p, n + p + m), rhs_z)

        self._lin_sys_solver.solve(cuda_stream=wp.get_stream("cuda").cuda_stream)

        # TODO: merge these 3 kernels
        # Disassemble the solver's (B, dim) sol buffer into [delta_x, delta_y, delta_z].
        wp.copy(delta_x, column_slice(sol, 0, n))
        if p > 0:
            wp.copy(delta_y, column_slice(sol, n, n + p))
        if m > 0:
            wp.copy(delta_z, column_slice(sol, n + p, n + p + m))
        self._write_solve_status(delta_x, delta_y, delta_z)

    @nvtx.annotate("SparseKKTSolver::eval_P_x")
    def eval_P_x(self, data: SparseData, alpha: float, x: wp.array, z: wp.array):
        self._spmv_P(x, z, alpha=alpha, beta=0.0)

    @nvtx.annotate("SparseKKTSolver::eval_A_xn")
    def eval_A_xn(self, data: SparseData, alpha_n: float, xn: wp.array, zn: wp.array):
        self._spmv_A(xn, zn, alpha=alpha_n, beta=0.0)

    @nvtx.annotate("SparseKKTSolver::eval_AT_xt")
    def eval_AT_xt(self, data: SparseData, alpha_t: float, xt: wp.array, zt: wp.array):
        self._spmv_AT(xt, zt, alpha=alpha_t, beta=0.0)

    @nvtx.annotate("SparseKKTSolver::eval_G_xn")
    def eval_G_xn(self, data: SparseData, alpha_n: float, xn: wp.array, zn: wp.array):
        self._spmv_G(xn, zn, alpha=alpha_n, beta=0.0)

    @nvtx.annotate("SparseKKTSolver::eval_GT_xt")
    def eval_GT_xt(self, data: SparseData, alpha_t: float, xt: wp.array, zt: wp.array):
        self._spmv_GT(xt, zt, alpha=alpha_t, beta=0.0)
