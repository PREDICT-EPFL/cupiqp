import cupy as cp
import warp as wp
import nvtx

from socu.block_tridiag_solver import (
    create_cholesky_factor_launch,
    create_cholesky_solve_launch,
    calculate_off_diag_storage_len,
)

from ..kkt_solver import KKTSolverBase
from .multistage_data import MultistageData
from .multistage_utils import (
    create_block_tridiag_diaad_kernel,
    create_block_tridiag_gead_kernel,
    create_block_bidiag_gemv_n_kernel,
    create_block_bidiag_gemv_t_kernel,
    create_block_tridiag_gemv_kernel,
    create_block_syrk_kernel,
    create_weighted_block_syrk_kernel,
)


class MultistageKKTSolver(KKTSolverBase):
    """
    Multi-stage KKT solver with block-wise Cholesky factorization. All operations use native block structures.
    """
    def __init__(self, data: MultistageData, block_size: int):
        super().__init__()
        N = data.num_blocks
        d = block_size

        self._block_size = d
        self.num_stages = N

        self._delta_inv = cp.zeros(1, dtype=cp.float64)  # for kernels that need a pointer to delta_inv
        self._scalar_one = cp.ones(1, dtype=cp.float64)  # for kernels that need a pointer to a scalar 1
        self._x_reg = cp.zeros(data.n, dtype=cp.float64)
        self._z_reg_inv = cp.zeros(data.m, dtype=cp.float64)

        # ---- Cholesky solver storage ----
        self._kkt_diag_blocks = wp.zeros((N, d, d), dtype=wp.float64, device="cuda")
        self._kkt_offdiag_blocks = wp.zeros(
            (calculate_off_diag_storage_len(N), d, d), dtype=wp.float64, device="cuda"
        )
        self._kkt_rhs = wp.zeros((N, d, 1), dtype=wp.float64, device="cuda")

        self._cholesky_factor_launch = create_cholesky_factor_launch(
            self._kkt_diag_blocks, self._kkt_offdiag_blocks,
            device="cuda", dtype=wp.float64, use_cuda_graph=True,
        )
        self._cholesky_solve_launch = create_cholesky_solve_launch(
            self._kkt_diag_blocks, self._kkt_offdiag_blocks, self._kkt_rhs,
            device="cuda", dtype=wp.float64, use_cuda_graph=True,
        )

        self._btd_diaad_kernel = create_block_tridiag_diaad_kernel(d, wp.float64)  # used to add diag(x_reg) to KKT matrix
        self._btd_gead_kernel = create_block_tridiag_gead_kernel(N, d, wp.float64)  # used to add P and (1/delta)*AtA to KKT matrix

        # Precompute A^T A as block-tridiagonal (A is fixed)
        if data.p > 0:
            self._AtA_diag = wp.zeros((N, d, d), dtype=wp.float64, device="cuda")
            self._AtA_offdiag = wp.zeros((N - 1, d, d), dtype=wp.float64, device="cuda")
            self._eval_AT_A_kernel = create_block_syrk_kernel(N, data._A.rows_of_blocks, d)
            wp.launch(
                kernel=self._eval_AT_A_kernel,
                dim=(N, d, d),
                inputs=[
                    wp.float64(1.0), data._A.D, data._A.E,
                    wp.float64(0.0), self._AtA_diag, self._AtA_offdiag,
                ],
            )

        if data.m > 0:
            self._eval_GT_zreg_G_kernel = create_weighted_block_syrk_kernel(N, data._G.rows_of_blocks, d)

        # ---- Kernels for matvec operations ----
        self._eval_P_x_kernel = create_block_tridiag_gemv_kernel(N, d, wp.float64)

        if data.p > 0:
            self._eval_A_xn_kernel = create_block_bidiag_gemv_n_kernel(N, data._A.rows_of_blocks, d, wp.float64)
            self._eval_AT_xt_kernel = create_block_bidiag_gemv_t_kernel(N, data._A.rows_of_blocks, d, wp.float64)

        if data.m > 0:
            self._eval_G_xn_kernel = create_block_bidiag_gemv_n_kernel(N, data._G.rows_of_blocks, d, wp.float64)
            self._eval_GT_xt_kernel = create_block_bidiag_gemv_t_kernel(N, data._G.rows_of_blocks, d, wp.float64)

    @nvtx.annotate("MultistageKKTSolver::update_kkt")
    def update_kkt(self, data: MultistageData, delta: float, x_reg: cp.ndarray, z_reg: cp.ndarray) -> None:
        """
        KKT = P + diag(x_reg) + (1/delta)*A^T*A + G^T*diag(z_reg_inv)*G
        """
        cp.reciprocal(delta, out=self._delta_inv)
        self._x_reg[:] = x_reg
        cp.reciprocal(z_reg, out=self._z_reg_inv)

        N = self.num_stages
        d = self._block_size

        self._kkt_diag_blocks.zero_()
        self._kkt_offdiag_blocks.zero_()

        # kkt += P
        wp.launch(
            kernel=self._btd_gead_kernel,
            dim=(N, d, d),
            inputs=[
                self._scalar_one,
                data._P.diag_blocks.data,
                data._P.off_diag_blocks_lower.data,
                self._kkt_diag_blocks,
                self._kkt_offdiag_blocks,
            ],
        )

        # kkt += diag(x_reg)
        wp.launch(
            kernel=self._btd_diaad_kernel,
            dim=(data.n,),
            inputs=[self._x_reg, self._kkt_diag_blocks],
        )

        # kkt += (1/delta) * A^T A
        if data.p > 0:
            wp.launch(
                kernel=self._btd_gead_kernel,
                dim=(N, d, d),
                inputs=[
                    self._delta_inv,
                    self._AtA_diag,
                    self._AtA_offdiag,
                    self._kkt_diag_blocks,
                    self._kkt_offdiag_blocks,
                ],
            )

        # kkt += G^T diag(z_reg_inv) G  (weighted SYRK, accumulated directly)
        if data.m > 0:
            wp.launch(
                kernel=self._eval_GT_zreg_G_kernel,
                dim=(N, d, d),
                inputs=[
                    wp.float64(1.0),
                    data._G.D, data._G.E,
                    self._z_reg_inv,
                    wp.float64(1.0),
                    self._kkt_diag_blocks,
                    self._kkt_offdiag_blocks,
                ],
            )

    @nvtx.annotate("MultistageKKTSolver::factor")
    def factor(self) -> bool:
        self._cholesky_factor_launch()

        diag_has_nan = cp.isnan(cp.from_dlpack(self._kkt_diag_blocks)).any()
        offdiag_has_nan = cp.isnan(cp.from_dlpack(self._kkt_offdiag_blocks)).any()
        return not (diag_has_nan or offdiag_has_nan)

    @nvtx.annotate("MultistageKKTSolver::solve")
    def solve(self, data: MultistageData, rhs_x: cp.ndarray, rhs_y: cp.ndarray, rhs_z: cp.ndarray, delta_x: cp.ndarray, delta_y: cp.ndarray, delta_z: cp.ndarray):
        N = self.num_stages
        d = self._block_size

        # delta_x = rhs_x + (1/delta)*A^T*rhs_y + G^T*diag(z_reg_inv)*rhs_z
        delta_x[:] = rhs_x

        if data.p > 0:
            # delta_x += (1/delta) * A^T * rhs_y
            wp.launch(
                kernel=self._eval_AT_xt_kernel,
                dim=(N, d),
                inputs=[
                    self._delta_inv,
                    data._A.D, data._A.E,
                    rhs_y,
                    wp.float64(1.0),
                    delta_x,
                ],
            )

        if data.m > 0:
            # delta_x += G^T * z_reg_inv * rhs_z (use delta_z as temp buffer for z_reg_inv * rhs_z)
            cp.multiply(self._z_reg_inv, rhs_z, out=delta_z)
            wp.launch(
                kernel=self._eval_GT_xt_kernel, dim=(N, d),
                inputs=[
                    wp.float64(1.0),
                    data._G.D, data._G.E,
                    delta_z,
                    wp.float64(1.0),
                    delta_x,
                ],
            )

        # Copy to warp rhs, solve, copy back
        dst = cp.from_dlpack(wp.to_dlpack(self._kkt_rhs))
        src = cp.asarray(delta_x, dtype=dst.dtype).reshape(dst.shape)
        cp.copyto(dst, src)

        self._cholesky_solve_launch()

        delta_x[:] = cp.asarray(self._kkt_rhs).reshape((data.n,))

        # dy = (1/delta) * (A * dx - rhs_y)
        if data.p > 0:
            wp.launch(
                kernel=self._eval_A_xn_kernel,
                dim=(N + 1, data._A.rows_of_blocks),
                inputs=[
                    wp.float64(1.0),
                    data._A.D, data._A.E,
                    delta_x,
                    wp.float64(0.0),
                    delta_y,
                ],
            )
            delta_y -= rhs_y
            delta_y *= self._delta_inv

        # dz = z_reg_inv * (G * dx - rhs_z)
        if data.m > 0:
            wp.launch(
                kernel=self._eval_G_xn_kernel,
                dim=(N + 1, data._G.rows_of_blocks),
                inputs=[
                    wp.float64(1.0),
                    data._G.D, data._G.E,
                    delta_x,
                    wp.float64(0.0),
                    delta_z,
                ],
            )
            delta_z -= rhs_z
            delta_z *= self._z_reg_inv

    @nvtx.annotate("MultistageKKTSolver::eval_P_x")
    def eval_P_x(self, data: MultistageData, alpha: float, x: cp.ndarray, z: cp.ndarray):
        N = self.num_stages
        d = self._block_size
        wp.launch(
            self._eval_P_x_kernel, dim=(N, d),
            inputs=[
                wp.float64(alpha),
                data._P.diag_blocks.data,
                data._P.off_diag_blocks_lower.data,
                x,
                wp.float64(0.0),
                z,
            ],
        )

    @nvtx.annotate("MultistageKKTSolver::eval_A_xn_and_AT_xt")
    def eval_A_xn_and_AT_xt(self, data: MultistageData, alpha_n: float, xn: cp.ndarray, alpha_t: float, xt: cp.ndarray, zn: cp.ndarray, zt: cp.ndarray):
        N = self.num_stages
        d = self._block_size
        # zn = alpha_n * A * xn
        wp.launch(
            self._eval_A_xn_kernel, dim=(N + 1, data._A.rows_of_blocks),
            inputs=[
                wp.float64(alpha_n),
                data._A.D,
                data._A.E,
                xn,
                wp.float64(0.0),
                zn,
            ],
        )
        # zt = alpha_t * A^T * xt
        wp.launch(
            self._eval_AT_xt_kernel, dim=(N, d),
            inputs=[
                wp.float64(alpha_t),
                data._A.D,
                data._A.E,
                xt,
                wp.float64(0.0),
                zt,
            ],
        )

    @nvtx.annotate("MultistageKKTSolver::eval_G_xn_and_GT_xt")
    def eval_G_xn_and_GT_xt(self, data: MultistageData, alpha_n: float, xn: cp.ndarray, alpha_t: float, xt: cp.ndarray, zn: cp.ndarray, zt: cp.ndarray):
        N = self.num_stages
        d = self._block_size
        # zn = alpha_n * G * xn
        wp.launch(
            self._eval_G_xn_kernel,
            dim=(N + 1, data._G.rows_of_blocks),
            inputs=[
                wp.float64(alpha_n),
                data._G.D,
                data._G.E,
                xn,
                wp.float64(0.0),
                zn,
            ],
        )
        # zt = alpha_t * G^T * xt
        wp.launch(
            self._eval_GT_xt_kernel,
            dim=(N, d),
            inputs=[
                wp.float64(alpha_t),
                data._G.D,
                data._G.E,
                xt,
                wp.float64(0.0),
                zt,
            ],
        )

    @nvtx.annotate("MultistageKKTSolver::eval_G_xn")
    def eval_G_xn(self, data: MultistageData, alpha_n: float, xn: cp.ndarray, zn: cp.ndarray):
        N = self.num_stages
        r_G = data._G.rows_of_blocks
        # zn = alpha_n * G * xn
        wp.launch(
            self._eval_G_xn_kernel, dim=(N + 1, r_G),
            inputs=[
                wp.float64(alpha_n),
                data._G.D,
                data._G.E,
                xn,
                wp.float64(0.0),
                zn,
            ],
        )
