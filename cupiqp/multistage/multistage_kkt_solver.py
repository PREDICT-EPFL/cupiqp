import cupy as cp
import warp as wp
import nvtx

from socu.block_tridiag_solver import (
    create_cholesky_factor_launch,
    create_cholesky_solve_launch,
    calculate_off_diag_storage_len,
)

from ..kkt_solver import KKTSolverBase
from ..utils import to_warp_dtype
from .multistage_data import MultistageData
from .multistage_utils_kernels import (
    create_block_bidiag_gemv_n_kernel,
    create_block_bidiag_gemv_t_kernel,
    create_block_tridiag_gemv_kernel,
    create_block_syrk_kernel,
)
from .multistage_kkt_solver_kernels import create_update_kkt_kernel


class MultistageKKTSolver(KKTSolverBase):
    """Multistage KKT solver with block-tridiagonal Cholesky factorization, batched."""
    def __init__(self, data: MultistageData):
        super().__init__()
        B = data.batch_size
        N = data.num_blocks
        d = data.block_size

        self._batch_size = B
        self._block_size = d
        self.num_stages = N
        dtype = data.dtype
        self._wp_dtype = to_warp_dtype(dtype)

        self._delta_inv = cp.zeros(B, dtype=dtype)
        self._z_reg_inv = cp.zeros((B, data.m), dtype=dtype)

        # ---- Block-tridiag KKT storage (always 4-D) ----
        self._kkt_diag_blocks = wp.zeros((B, N, d, d), dtype=self._wp_dtype, device="cuda")
        self._kkt_offdiag_blocks = wp.zeros(
            (B, calculate_off_diag_storage_len(N), d, d), dtype=self._wp_dtype, device="cuda"
        )
        self._kkt_rhs = wp.zeros((B, N, d, 1), dtype=self._wp_dtype, device="cuda")

        # CuPy aliases of the Warp arrays — letting us call .fill(0) during
        # CUDA-graph capture without going through ``from_dlpack`` (whose
        # ``__dlpack__`` may trigger ops that invalidate the capture).
        self._kkt_diag_blocks_cp = cp.from_dlpack(self._kkt_diag_blocks)
        self._kkt_offdiag_blocks_cp = cp.from_dlpack(self._kkt_offdiag_blocks)
        # For the rhs scratch: the (B, n) flat view that the IPM passes us
        # is just a reshape away from (B, N, d, 1); cache the cupy view.
        self._kkt_rhs_cp = cp.from_dlpack(self._kkt_rhs)  # (B, N, d, 1)
        self._kkt_rhs_flat_cp = self._kkt_rhs_cp.reshape(B, data.n)

        self._cholesky_factor_launch = create_cholesky_factor_launch(
            self._kkt_diag_blocks, self._kkt_offdiag_blocks,
            device="cuda", dtype=self._wp_dtype, use_cuda_graph=True,
        )
        self._cholesky_solve_launch = create_cholesky_solve_launch(
            self._kkt_diag_blocks, self._kkt_offdiag_blocks, self._kkt_rhs,
            device="cuda", dtype=self._wp_dtype, use_cuda_graph=True,
        )

        # Workspace for matvec-then-scale steps in solve().
        self._work_n = cp.empty((B, data.n), dtype=dtype)

        # Precompute A^T A as block-tridiagonal (A is fixed across iterations).
        if data.p > 0:
            self._AtA_diag = wp.zeros((B, N, d, d), dtype=self._wp_dtype, device="cuda")
            self._AtA_offdiag = wp.zeros((B, N - 1, d, d), dtype=self._wp_dtype, device="cuda")
            self._eval_AT_A_kernel = create_block_syrk_kernel(N, data._A.rows_of_blocks, d, dtype=dtype)
            wp.launch(
                kernel=self._eval_AT_A_kernel,
                dim=(B, N, d, d),
                inputs=[
                    self._wp_dtype(1.0), data._A.D, data._A.E,
                    self._wp_dtype(0.0), self._AtA_diag, self._AtA_offdiag,
                ],
            )
        else:
            self._AtA_diag = wp.zeros((B, 0, 0, 0), dtype=self._wp_dtype, device="cuda")
            self._AtA_offdiag = wp.zeros((B, 0, 0, 0), dtype=self._wp_dtype, device="cuda")

        # G placeholders for the fused kernel when m == 0; same elision logic.
        if data.m > 0:
            self._kkt_G_D = data._G.D
            self._kkt_G_E = data._G.E
            rows_of_G_for_kkt = data._G.rows_of_blocks
        else:
            self._kkt_G_D = wp.zeros((B, 0, 0, 0), dtype=self._wp_dtype, device="cuda")
            self._kkt_G_E = wp.zeros((B, 0, 0, 0), dtype=self._wp_dtype, device="cuda")
            rows_of_G_for_kkt = 1

        self._update_kkt_kernel = create_update_kkt_kernel(
            num_blocks=N, block_size=d,
            p=data.p, m=data.m, rows_of_G=rows_of_G_for_kkt,
            dtype=dtype)

        # ---- matvec kernels ----
        self._eval_P_x_kernel = create_block_tridiag_gemv_kernel(N, d, dtype=dtype)

        if data.p > 0:
            self._eval_A_xn_kernel = create_block_bidiag_gemv_n_kernel(N, data._A.rows_of_blocks, d, dtype=dtype)
            self._eval_AT_xt_kernel = create_block_bidiag_gemv_t_kernel(N, data._A.rows_of_blocks, d, dtype=dtype)

        if data.m > 0:
            self._eval_G_xn_kernel = create_block_bidiag_gemv_n_kernel(N, data._G.rows_of_blocks, d, dtype=dtype)
            self._eval_GT_xt_kernel = create_block_bidiag_gemv_t_kernel(N, data._G.rows_of_blocks, d, dtype=dtype)

    def update_data(self, data: MultistageData, update_P: bool, update_A: bool, update_G: bool):
        if update_A and data.p > 0:
            stream = wp.Stream(cuda_stream=cp.cuda.get_current_stream().ptr)
            wp.launch(
                kernel=self._eval_AT_A_kernel,
                dim=(self._batch_size, self.num_stages, self._block_size, self._block_size),
                inputs=[
                    self._wp_dtype(1.0), data._A.D, data._A.E,
                    self._wp_dtype(0.0), self._AtA_diag, self._AtA_offdiag,
                ],
                stream=stream,
            )

    @nvtx.annotate("MultistageKKTSolver::update_kkt")
    def update_kkt(self, data: MultistageData, delta: cp.ndarray, x_reg: cp.ndarray, z_reg: cp.ndarray) -> None:
        """KKT[b] = P[b] + diag(x_reg[b]) + (1/delta[b])*A[b]^T A[b] + G[b]^T diag(z_reg_inv[b]) G[b]"""
        stream = wp.Stream(cuda_stream=cp.cuda.get_current_stream().ptr)

        B = self._batch_size
        N = self.num_stages
        d = self._block_size

        self._kkt_offdiag_blocks_cp.fill(0)

        # Launch covers k in [0, N+1) so threads at k = N can write the
        # final block of z_reg_inv; KKT writes are guarded by k < N (and
        # k < N-1 for the off-diagonal).
        wp.launch(
            kernel=self._update_kkt_kernel,
            dim=(B, N + 1, d, d),
            inputs=[
                data._P.D,
                data._P.E,
                x_reg,
                self._AtA_diag, self._AtA_offdiag,
                delta,
                self._kkt_G_D, self._kkt_G_E,
                z_reg,
                self._kkt_diag_blocks,
                self._kkt_offdiag_blocks,
                self._delta_inv,
                self._z_reg_inv,
            ],
            stream=stream,
        )

    @nvtx.annotate("MultistageKKTSolver::factor")
    def factor(self) -> bool:
        self._cholesky_factor_launch()
        # socu doesn't expose per-batch info; fall back to a NaN scan over
        # the whole 4-D buffer (one bool for the whole batch — same convention
        # as the dense backend's batched factor).
        diag_has_nan = cp.isnan(self._kkt_diag_blocks_cp).any()
        offdiag_has_nan = cp.isnan(self._kkt_offdiag_blocks_cp).any()
        return not bool(diag_has_nan or offdiag_has_nan)

    @nvtx.annotate("MultistageKKTSolver::solve")
    def solve(self, data: MultistageData,
              rhs_x: cp.ndarray, rhs_y: cp.ndarray, rhs_z: cp.ndarray,
              delta_x: cp.ndarray, delta_y: cp.ndarray, delta_z: cp.ndarray):
        B = self._batch_size
        N = self.num_stages
        d = self._block_size

        # delta_x = rhs_x + (1/delta)*A^T*rhs_y + G^T * z_reg_inv * rhs_z
        delta_x[:] = rhs_x

        if data.p > 0:
            # _work_n = A^T rhs_y; delta_x += (1/delta) * _work_n
            # (the gemv kernel takes scalar alpha so we apply the per-batch
            # 1/delta scaling outside via cupy broadcast.)
            wp.launch(
                kernel=self._eval_AT_xt_kernel,
                dim=(B, N, d),
                inputs=[
                    self._wp_dtype(1.0),
                    data._A.D, data._A.E,
                    rhs_y,
                    self._wp_dtype(0.0),
                    self._work_n,
                ],
            )
            delta_x += self._delta_inv[:, None] * self._work_n

        if data.m > 0:
            # delta_x += G^T * (z_reg_inv * rhs_z); reuse delta_z as scratch.
            cp.multiply(self._z_reg_inv, rhs_z, out=delta_z)
            wp.launch(
                kernel=self._eval_GT_xt_kernel, dim=(B, N, d),
                inputs=[
                    self._wp_dtype(1.0),
                    data._G.D, data._G.E,
                    delta_z,
                    self._wp_dtype(1.0),
                    delta_x,
                ],
            )

        # Stage rhs into the (B, N, d, 1) socu buffer (zero-copy reshape view).
        self._kkt_rhs_flat_cp[:] = delta_x
        self._cholesky_solve_launch()
        delta_x[:] = self._kkt_rhs_flat_cp

        # delta_y = (A * delta_x - rhs_y) / delta
        if data.p > 0:
            wp.launch(
                kernel=self._eval_A_xn_kernel,
                dim=(B, N + 1, data._A.rows_of_blocks),
                inputs=[
                    self._wp_dtype(1.0),
                    data._A.D, data._A.E,
                    delta_x,
                    self._wp_dtype(0.0),
                    delta_y,
                ],
            )
            delta_y -= rhs_y
            delta_y *= self._delta_inv[:, None]

        # delta_z = z_reg_inv * (G * delta_x - rhs_z)
        if data.m > 0:
            wp.launch(
                kernel=self._eval_G_xn_kernel,
                dim=(B, N + 1, data._G.rows_of_blocks),
                inputs=[
                    self._wp_dtype(1.0),
                    data._G.D, data._G.E,
                    delta_x,
                    self._wp_dtype(0.0),
                    delta_z,
                ],
            )
            delta_z -= rhs_z
            delta_z *= self._z_reg_inv

    @nvtx.annotate("MultistageKKTSolver::eval_P_x")
    def eval_P_x(self, data: MultistageData, alpha: float, x: cp.ndarray, z: cp.ndarray):
        stream_wp = wp.Stream(cuda_stream=cp.cuda.get_current_stream().ptr)
        wp.launch(
            self._eval_P_x_kernel,
            dim=(self._batch_size, self.num_stages, self._block_size),
            inputs=[
                self._wp_dtype(alpha),
                data._P.D,
                data._P.E,
                x,
                self._wp_dtype(0.0),
                z,
            ],
            stream=stream_wp,
        )

    @nvtx.annotate("MultistageKKTSolver::eval_A_xn")
    def eval_A_xn(self, data: MultistageData, alpha_n: float, xn: cp.ndarray, zn: cp.ndarray):
        stream_wp = wp.Stream(cuda_stream=cp.cuda.get_current_stream().ptr)
        wp.launch(
            self._eval_A_xn_kernel,
            dim=(self._batch_size, self.num_stages + 1, data._A.rows_of_blocks),
            inputs=[
                self._wp_dtype(alpha_n),
                data._A.D, data._A.E,
                xn,
                self._wp_dtype(0.0),
                zn,
            ],
            stream=stream_wp,
        )

    @nvtx.annotate("MultistageKKTSolver::eval_AT_xt")
    def eval_AT_xt(self, data: MultistageData, alpha_t: float, xt: cp.ndarray, zt: cp.ndarray):
        stream_wp = wp.Stream(cuda_stream=cp.cuda.get_current_stream().ptr)
        wp.launch(
            self._eval_AT_xt_kernel,
            dim=(self._batch_size, self.num_stages, self._block_size),
            inputs=[
                self._wp_dtype(alpha_t),
                data._A.D, data._A.E,
                xt,
                self._wp_dtype(0.0),
                zt,
            ],
            stream=stream_wp,
        )

    @nvtx.annotate("MultistageKKTSolver::eval_G_xn")
    def eval_G_xn(self, data: MultistageData, alpha_n: float, xn: cp.ndarray, zn: cp.ndarray):
        stream_wp = wp.Stream(cuda_stream=cp.cuda.get_current_stream().ptr)
        wp.launch(
            self._eval_G_xn_kernel,
            dim=(self._batch_size, self.num_stages + 1, data._G.rows_of_blocks),
            inputs=[
                self._wp_dtype(alpha_n),
                data._G.D, data._G.E,
                xn,
                self._wp_dtype(0.0),
                zn,
            ],
            stream=stream_wp,
        )

    @nvtx.annotate("MultistageKKTSolver::eval_GT_xt")
    def eval_GT_xt(self, data: MultistageData, alpha_t: float, xt: cp.ndarray, zt: cp.ndarray):
        stream_wp = wp.Stream(cuda_stream=cp.cuda.get_current_stream().ptr)
        wp.launch(
            self._eval_GT_xt_kernel,
            dim=(self._batch_size, self.num_stages, self._block_size),
            inputs=[
                self._wp_dtype(alpha_t),
                data._G.D, data._G.E,
                xt,
                self._wp_dtype(0.0),
                zt,
            ],
            stream=stream_wp,
        )
