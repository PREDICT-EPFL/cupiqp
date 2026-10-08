import nvtx
import warp as wp

from ..kkt_solver import KKTSolverBase
from .dense_data import DenseData
from .dense_cholesky import CholeskyInplaceSolver, BatchedCholeskyInplaceSolver
from .dense_kkt_solver_kernels import (
    create_update_kkt_kernel,
    create_solve_pre_cholesky_kernel,
    create_solve_post_cholesky_kernel
)
from .dense_matmul import CublasHandle, DenseGemv, DenseSyrk


def use_cusolver_potrf(rows):
    return True if rows > 32 else False



class DenseKKTSolver(KKTSolverBase):
    """
    Dense KKT solver.

    Eliminates Delta_y and Delta_z to form:
    (P + diag(x_reg) + (1/delta)*A^T*A + G^T*diag(z_reg_inv)*G) Delta_x = rhs.
    """
    def __init__(self, data: DenseData):
        n, p, m = data.n, data.p, data.m
        B = data.batch_size
        super().__init__(B, data.dtype, data.device)
        self._batch_size = B
        self._dtype = data.dtype
        self._device = data.device

        # Pre-allocated workspace -- all (B, ...) shapes
        self._delta = wp.empty((B,), dtype=self._dtype, device=self._device)
        self._delta_inv = wp.empty((B,), dtype=self._dtype, device=self._device)
        self._z_reg_inv = wp.empty((B, m), dtype=self._dtype, device=self._device)
        self._z_reg_inv_sqrt = wp.empty((B, m), dtype=self._dtype, device=self._device)
        self._kkt_mat = wp.empty((B, n, n), dtype=self._dtype, device=self._device)
        self._AtA = wp.zeros((B, n, n), dtype=self._dtype, device=self._device) if p > 0 else wp.zeros((B, 0, 0), dtype=self._dtype, device=self._device)
        self._G_scaled = wp.empty((B, m, n), dtype=self._dtype, device=self._device) if m > 0 else wp.zeros((B, 0, 0), dtype=self._dtype, device=self._device)
        self._work_n_AT = wp.empty((B, n), dtype=self._dtype, device=self._device) if p > 0 else wp.empty((B, 0), dtype=self._dtype, device=self._device)
        self._work_n_GT = wp.empty((B, n), dtype=self._dtype, device=self._device) if m > 0 else wp.empty((B, 0), dtype=self._dtype, device=self._device)

        self._update_kkt_kernel, self._update_kkt_kernel_launch_dim = create_update_kkt_kernel(n, p, m, dtype=self._dtype)
        self._solve_pre_cholesky_kernel = create_solve_pre_cholesky_kernel(p, m, dtype=self._dtype)
        self._solve_post_cholesky_kernel = create_solve_post_cholesky_kernel(p, m, dtype=self._dtype)

        # Matrix products, each with its own Warp / cuBLAS choice for its shape
        # (see dense_matmul.py). They share one cuBLAS handle, created only if
        # one of them uses cuBLAS.
        cublas = CublasHandle()
        self._P_gemv = DenseGemv(B, n, n, self._dtype, cublas)
        self._A_gemv = DenseGemv(B, p, n, self._dtype, cublas)
        self._G_gemv = DenseGemv(B, m, n, self._dtype, cublas)
        self._AtA_syrk = DenseSyrk(B, p, n, self._dtype, cublas)
        self._GtG_syrk = DenseSyrk(B, m, n, self._dtype, cublas)

        if B > 1:
            self._cholesky_solver = BatchedCholeskyInplaceSolver(n, B, dtype=self._dtype)
        else:
            self._cholesky_solver = CholeskyInplaceSolver(n, dtype=self._dtype)

        # cuSOLVER writes one info entry per matrix ((1,) for the single
        # solver) straight into the status buffer.
        self._factor_status = self._cholesky_solver.factor_status

        if p > 0:
            self._compute_AtA(data)

    def _compute_AtA(self, data: DenseData):
        """Compute AtA = A^T * A."""
        self._AtA_syrk(data.A, self._AtA)

    def update_data(self, data: DenseData, update_P: bool, update_A: bool, update_G: bool):
        if update_A and data.p > 0:
            self._compute_AtA(data)

    @nvtx.annotate("DenseKKTSolver::update_kkt")
    def update_kkt(self, data: DenseData, delta: wp.array, x_reg: wp.array, z_reg: wp.array, z_reg_inv: wp.array) -> None:
        """Assemble the condensed KKT matrix (CUDA graph safe)."""

        # For inactive rows G[i], z_reg_inv[i] is already zero.
        # Therefore the contribution of G[i] to the condensed KKT is zero.
        wp.launch(
            kernel=self._update_kkt_kernel,
            dim=(self._batch_size, self._update_kkt_kernel_launch_dim),
            inputs=[
                data.P, self._AtA, data.G, delta, x_reg, z_reg_inv,
                self._delta_inv, self._z_reg_inv, self._z_reg_inv_sqrt,
                self._kkt_mat, self._G_scaled,
            ],
            device=self._device
        )
        if data.m > 0:
            self._GtG_syrk(self._G_scaled, self._kkt_mat, accumulate=True)

    @nvtx.annotate("DenseKKTSolver::factor")
    def factor(self) -> None:
        # B=1: CholeskyInplaceSolver expects (n, n); B>1: BatchedCholeskyInplaceSolver expects (B, n, n)
        factor_input = self._kkt_mat[0] if self._batch_size == 1 else self._kkt_mat
        self._cholesky_solver.factorize(factor_input)

    @nvtx.annotate("DenseKKTSolver::solve")
    def solve(self, data: DenseData, rhs_x, rhs_y, rhs_z, delta_x, delta_y, delta_z):
        """Solve the reduced KKT system and recover delta_y, delta_z."""
        n, p, m = data.n, data.p, data.m
        B = self._batch_size

        # work_n_AT = A^T @ rhs_y
        if p > 0:
            self.eval_AT_xt(data, 1.0, rhs_y, self._work_n_AT)
        # work_n_GT = G^T @ delta_z
        if m > 0:
            # delta_z = z_reg_inv * rhs_z
            wp.map(wp.mul, self._z_reg_inv, rhs_z, out=delta_z)
            self.eval_GT_xt(data, 1.0, delta_z, self._work_n_GT)

        wp.launch(
            kernel=self._solve_pre_cholesky_kernel,
            dim=(B, n),
            inputs=[rhs_x, self._delta_inv,
                    self._work_n_AT, self._work_n_GT, delta_x],
            device=self._device
        )

        # B=1: CholeskyInplaceSolver expects 1D/2D; B>1: BatchedCholeskyInplaceSolver expects (B, n).
        self._cholesky_solver.solve(delta_x[0] if B == 1 else delta_x)

        # Recover delta_y = (A @ delta_x - rhs_y) / delta
        if p > 0:
            self.eval_A_xn(data, 1.0, delta_x, delta_y)
        # Recover delta_z = (G @ delta_x - rhs_z) * z_reg_inv
        if m > 0:
            self.eval_G_xn(data, 1.0, delta_x, delta_z)

        if p > 0 or m > 0:
            wp.launch(
                kernel=self._solve_post_cholesky_kernel,
                dim=(B, p+m),
                inputs=[rhs_y, rhs_z, self._delta_inv, self._z_reg_inv,
                        delta_y, delta_z],
                device=self._device
            )
        self._write_solve_status(delta_x, delta_y, delta_z)

    @nvtx.annotate("DenseKKTSolver::eval_P_x")
    def eval_P_x(self, data: DenseData, alpha: float, x: wp.array, z: wp.array):
        self._P_gemv(data.P, x, z, alpha=alpha)

    @nvtx.annotate("DenseKKTSolver::eval_A_xn")
    def eval_A_xn(self, data: DenseData, alpha_n: float, xn: wp.array, zn: wp.array):
        self._A_gemv(data.A, xn, zn, alpha=alpha_n)

    @nvtx.annotate("DenseKKTSolver::eval_AT_xt")
    def eval_AT_xt(self, data: DenseData, alpha_t: float, xt: wp.array, zt: wp.array):
        self._A_gemv(data.A, xt, zt, transpose=True, alpha=alpha_t)

    @nvtx.annotate("DenseKKTSolver::eval_G_xn")
    def eval_G_xn(self, data: DenseData, alpha_n: float, xn: wp.array, zn: wp.array):
        self._G_gemv(data.G, xn, zn, alpha=alpha_n)

    @nvtx.annotate("DenseKKTSolver::eval_GT_xt")
    def eval_GT_xt(self, data: DenseData, alpha_t: float, xt: wp.array, zt: wp.array):
        self._G_gemv(data.G, xt, zt, transpose=True, alpha=alpha_t)
