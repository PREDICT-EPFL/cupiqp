import warp as wp
import nvtx

from socu.block_tridiag_solver import (
    create_cholesky_factor_launch,
    create_cholesky_solve_launch,
    calculate_off_diag_storage_len
)

from ..kkt_solver import KKTSolverBase
from .multistage_data import MultistageData
from .multistage_utils_kernels import (
    create_block_bidiag_gemv_n_kernel,
    create_block_bidiag_gemv_t_kernel,
    create_block_tridiag_gemv_kernel,
    create_block_syrk_kernel
)
from .multistage_kkt_solver_kernels import (
    create_update_kkt_kernel,
    create_add_scaled_rows_kernel,
    create_sub_scale_rows_kernel,
    sub_mul,
    create_has_nan_1d_kernel,
)


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
        self._dtype = dtype
        self._device = data.device

        self._delta_inv = wp.zeros((B,), dtype=self._dtype, device=self._device)
        self._z_reg_inv = wp.zeros((B, data.m), dtype=self._dtype, device=self._device)

        # ---- Block-tridiag KKT storage (always 4-D) ----
        self._kkt_diag_blocks = wp.zeros((B, N, d, d), dtype=self._dtype, device=self._device)
        self._kkt_offdiag_blocks = wp.zeros((B, calculate_off_diag_storage_len(N), d, d), dtype=self._dtype, device=self._device)
        self._kkt_rhs = wp.zeros((B, N, d, 1), dtype=self._dtype, device=self._device)
        # The (B, n) flat view of the rhs buffer the IPM vectors are copied to/from.
        self._kkt_rhs_flat = self._kkt_rhs.reshape((B, data.n))

        self._cholesky_factor_launch = create_cholesky_factor_launch(
            self._kkt_diag_blocks, self._kkt_offdiag_blocks,
            device="cuda", dtype=self._dtype, use_cuda_graph=True
        )
        self._cholesky_solve_launch = create_cholesky_solve_launch(
            self._kkt_diag_blocks, self._kkt_offdiag_blocks, self._kkt_rhs,
            device="cuda", dtype=self._dtype, use_cuda_graph=True
        )

        # Workspace for matvec-then-scale steps in solve().
        self._work_n = wp.zeros((B, data.n), dtype=self._dtype, device=self._device)
        # NaN scan of the factor (one host read per factorization).
        self._nan_flag = wp.zeros(1, dtype=wp.int32, device=self._device)
        self._nan_flag_host = wp.zeros(1, dtype=wp.int32, device="cpu", pinned=True)
        self._add_scaled_rows_kernel = create_add_scaled_rows_kernel(dtype)
        self._sub_scale_rows_kernel = create_sub_scale_rows_kernel(dtype)
        self._has_nan_1d_kernel = create_has_nan_1d_kernel(dtype)

        # Precompute A^T A as block-tridiagonal (A is fixed across iterations).
        if data.p > 0:
            self._AtA_diag = wp.zeros((B, N, d, d), dtype=self._dtype, device=self._device)
            self._AtA_offdiag = wp.zeros((B, N - 1, d, d), dtype=self._dtype, device=self._device)
            self._eval_AT_A_kernel = create_block_syrk_kernel(N, data.A_rows, d, dtype=dtype)
            wp.launch(
                kernel=self._eval_AT_A_kernel,
                dim=(B, N, d, d),
                inputs=[
                    self._dtype(1.0), data.A_diag, data.A_offdiag,
                    self._dtype(0.0), self._AtA_diag, self._AtA_offdiag,
                ]
            )
        else:
            self._AtA_diag = wp.zeros((B, 0, 0, 0), dtype=self._dtype, device=self._device)
            self._AtA_offdiag = wp.zeros((B, 0, 0, 0), dtype=self._dtype, device=self._device)

        # G placeholders for the fused kernel when m == 0; same elision logic.
        if data.m > 0:
            self._kkt_G_D = data.G_diag
            self._kkt_G_E = data.G_offdiag
            rows_of_G_for_kkt = data.G_rows
        else:
            self._kkt_G_D = wp.zeros((B, 0, 0, 0), dtype=self._dtype, device=self._device)
            self._kkt_G_E = wp.zeros((B, 0, 0, 0), dtype=self._dtype, device=self._device)
            rows_of_G_for_kkt = 1

        self._update_kkt_kernel = create_update_kkt_kernel(
            num_blocks=N, block_size=d,
            p=data.p, m=data.m, rows_of_G=rows_of_G_for_kkt,
            dtype=dtype)

        # ---- matvec kernels ----
        self._eval_P_x_kernel = create_block_tridiag_gemv_kernel(N, d, dtype=dtype)

        if data.p > 0:
            self._eval_A_xn_kernel = create_block_bidiag_gemv_n_kernel(N, data.A_rows, d, dtype=dtype)
            self._eval_AT_xt_kernel = create_block_bidiag_gemv_t_kernel(N, data.A_rows, d, dtype=dtype)

        if data.m > 0:
            self._eval_G_xn_kernel = create_block_bidiag_gemv_n_kernel(N, data.G_rows, d, dtype=dtype)
            self._eval_GT_xt_kernel = create_block_bidiag_gemv_t_kernel(N, data.G_rows, d, dtype=dtype)

    def update_data(self, data: MultistageData, update_P: bool, update_A: bool, update_G: bool):
        if update_A and data.p > 0:
            wp.launch(
                kernel=self._eval_AT_A_kernel,
                dim=(self._batch_size, self.num_stages, self._block_size, self._block_size),
                inputs=[
                    self._dtype(1.0), data.A_diag, data.A_offdiag,
                    self._dtype(0.0), self._AtA_diag, self._AtA_offdiag,
                ]
            )

    @nvtx.annotate("MultistageKKTSolver::update_kkt")
    def update_kkt(self, data: MultistageData, delta: wp.array, x_reg: wp.array, z_reg: wp.array, z_reg_inv: wp.array) -> None:
        """KKT[b] = P[b] + diag(x_reg[b]) + (1/delta[b])*A[b]^T A[b] + G[b]^T diag(z_reg_inv[b]) G[b].

        The condensed multistage backend uses z_reg_inv. The explicit diagonal z_reg is
        accepted for interface symmetry but not used here.
        """

        B = self._batch_size
        N = self.num_stages
        d = self._block_size

        self._kkt_offdiag_blocks.zero_()

        # Launch covers k in [0, N+1) so threads at k = N can write the
        # final block of z_reg_inv; KKT writes are guarded by k < N (and
        # k < N-1 for the off-diagonal).
        wp.launch(
            kernel=self._update_kkt_kernel,
            dim=(B, N + 1, d, d),
            inputs=[
                data.P_diag,
                data.P_offdiag,
                x_reg,
                self._AtA_diag, self._AtA_offdiag,
                delta,
                self._kkt_G_D, self._kkt_G_E,
                z_reg_inv,
                self._kkt_diag_blocks,
                self._kkt_offdiag_blocks,
                self._delta_inv,
                self._z_reg_inv,
            ]
        )

    @nvtx.annotate("MultistageKKTSolver::factor")
    def factor(self) -> bool:
        self._cholesky_factor_launch()
        # socu doesn't expose per-batch info; fall back to a NaN scan over
        # the whole 4-D buffer (one bool for the whole batch -- same convention
        # as the dense backend's batched factor).
        stream = wp.get_stream("cuda")
        self._nan_flag.zero_()
        for blocks in (self._kkt_diag_blocks, self._kkt_offdiag_blocks):
            flat = blocks.flatten()
            if flat.shape[0] > 0:
                wp.launch(self._has_nan_1d_kernel, dim=flat.shape[0], inputs=[flat, self._nan_flag],
                          device=self._device)
        wp.copy(self._nan_flag_host, self._nan_flag)
        wp.synchronize_stream(stream)
        return int(self._nan_flag_host.numpy()[0]) == 0

    @nvtx.annotate("MultistageKKTSolver::solve")
    def solve(self, data: MultistageData,
              rhs_x: wp.array, rhs_y: wp.array, rhs_z: wp.array,
              delta_x: wp.array, delta_y: wp.array, delta_z: wp.array):
        B = self._batch_size
        N = self.num_stages
        d = self._block_size

        # delta_x = rhs_x + (1/delta)*A^T*rhs_y + G^T * z_reg_inv * rhs_z
        wp.copy(delta_x, rhs_x)

        if data.p > 0:
            # _work_n = A^T rhs_y; delta_x += (1/delta) * _work_n
            wp.launch(
                kernel=self._eval_AT_xt_kernel,
                dim=(B, N, d),
                inputs=[
                    self._dtype(1.0),
                    data.A_diag, data.A_offdiag,
                    rhs_y,
                    self._dtype(0.0),
                    self._work_n,
                ]
            )
            wp.launch(self._add_scaled_rows_kernel, dim=(B, data.n), inputs=[delta_x, self._delta_inv, self._work_n],
                      device=self._device)

        if data.m > 0:
            # delta_x += G^T * (z_reg_inv * rhs_z); reuse delta_z as scratch.
            wp.map(wp.mul, self._z_reg_inv, rhs_z, out=delta_z)
            wp.launch(
                kernel=self._eval_GT_xt_kernel, dim=(B, N, d),
                inputs=[
                    self._dtype(1.0),
                    data.G_diag, data.G_offdiag,
                    delta_z,
                    self._dtype(1.0),
                    delta_x,
                ]
            )

        # Stage rhs into the (B, N, d, 1) socu buffer (zero-copy reshape view).
        wp.copy(self._kkt_rhs_flat, delta_x)
        self._cholesky_solve_launch()
        wp.copy(delta_x, self._kkt_rhs_flat)

        # delta_y = (A * delta_x - rhs_y) / delta
        if data.p > 0:
            wp.launch(
                kernel=self._eval_A_xn_kernel,
                dim=(B, N + 1, data.A_rows),
                inputs=[
                    self._dtype(1.0),
                    data.A_diag, data.A_offdiag,
                    delta_x,
                    self._dtype(0.0),
                    delta_y,
                ]
            )
            wp.launch(self._sub_scale_rows_kernel, dim=(B, data.p), inputs=[delta_y, rhs_y, self._delta_inv],
                      device=self._device)

        # delta_z = z_reg_inv * (G * delta_x - rhs_z)
        if data.m > 0:
            wp.launch(
                kernel=self._eval_G_xn_kernel,
                dim=(B, N + 1, data.G_rows),
                inputs=[
                    self._dtype(1.0),
                    data.G_diag, data.G_offdiag,
                    delta_x,
                    self._dtype(0.0),
                    delta_z,
                ]
            )
            wp.map(sub_mul, delta_z, rhs_z, self._z_reg_inv, out=delta_z)

    @nvtx.annotate("MultistageKKTSolver::eval_P_x")
    def eval_P_x(self, data: MultistageData, alpha: float, x: wp.array, z: wp.array):
        wp.launch(
            self._eval_P_x_kernel,
            dim=(self._batch_size, self.num_stages, self._block_size),
            inputs=[
                self._dtype(alpha),
                data.P_diag,
                data.P_offdiag,
                x,
                self._dtype(0.0),
                z,
            ]
        )

    @nvtx.annotate("MultistageKKTSolver::eval_A_xn")
    def eval_A_xn(self, data: MultistageData, alpha_n: float, xn: wp.array, zn: wp.array):
        wp.launch(
            self._eval_A_xn_kernel,
            dim=(self._batch_size, self.num_stages + 1, data.A_rows),
            inputs=[
                self._dtype(alpha_n),
                data.A_diag, data.A_offdiag,
                xn,
                self._dtype(0.0),
                zn,
            ]
        )

    @nvtx.annotate("MultistageKKTSolver::eval_AT_xt")
    def eval_AT_xt(self, data: MultistageData, alpha_t: float, xt: wp.array, zt: wp.array):
        wp.launch(
            self._eval_AT_xt_kernel,
            dim=(self._batch_size, self.num_stages, self._block_size),
            inputs=[
                self._dtype(alpha_t),
                data.A_diag, data.A_offdiag,
                xt,
                self._dtype(0.0),
                zt,
            ]
        )

    @nvtx.annotate("MultistageKKTSolver::eval_G_xn")
    def eval_G_xn(self, data: MultistageData, alpha_n: float, xn: wp.array, zn: wp.array):
        wp.launch(
            self._eval_G_xn_kernel,
            dim=(self._batch_size, self.num_stages + 1, data.G_rows),
            inputs=[
                self._dtype(alpha_n),
                data.G_diag, data.G_offdiag,
                xn,
                self._dtype(0.0),
                zn,
            ]
        )

    @nvtx.annotate("MultistageKKTSolver::eval_GT_xt")
    def eval_GT_xt(self, data: MultistageData, alpha_t: float, xt: wp.array, zt: wp.array):
        wp.launch(
            self._eval_GT_xt_kernel,
            dim=(self._batch_size, self.num_stages, self._block_size),
            inputs=[
                self._dtype(alpha_t),
                data.G_diag, data.G_offdiag,
                xt,
                self._dtype(0.0),
                zt,
            ]
        )
