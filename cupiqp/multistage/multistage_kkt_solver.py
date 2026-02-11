import cupy as cp
import cupyx.scipy.sparse as cpsp
import warp as wp
import nvtx

# import scipy.sparse as sp

from socu.block_tridiag_solver import (
    create_cholesky_factor_launch,
    create_cholesky_solve_launch,
    calculate_off_diag_storage_len,
)


from ..kkt_solver import KKTSolverBase
from ..sparse.sparse_data import SparseData
from .multistage_data import MultistageData
from .multistage_utils import (
    create_csr_add_btd_kernel,
    create_add_on_diag_kernel
)


class MultistageKKTSolver(KKTSolverBase):
    """
    Multi-stage KKT solver with block-wise Cholesky factorization.
    """
    def __init__(self, data: MultistageData, block_size: int):
        super().__init__()
        self._delta = cp.zeros(1, dtype=cp.float64)
        self._x_reg = cp.zeros(data.n, dtype=cp.float64)
        self._z_reg_inv = cp.zeros(data.m, dtype=cp.float64)

        self._kkt_sparsity_csr = data.P + data.A.T @ data.A + data.G.T @ data.G + cpsp.eye(data.n, format='csr', dtype=cp.float64)
        self._kkt_sparsity_csr = cpsp.csr_matrix(self._kkt_sparsity_csr)
        self._block_size = block_size
        if self._kkt_sparsity_csr.shape[0] % block_size != 0:
            raise ValueError("KKT matrix size must be divisible by block_size")
        self.num_stages = self._kkt_sparsity_csr.shape[0] // block_size

        self._AtA = data.A.T @ data.A  # cachable since A is fixed
        self._GtG_scaled = data.G.T @ data.G  # preallocate memory to store G^T * diag(z_reg_inv) * G

        self._kkt_diag_blocks = wp.zeros((self.num_stages, block_size, block_size), dtype=wp.float64, device="cuda")
        self._kkt_offdiag_blocks = wp.zeros((calculate_off_diag_storage_len(self.num_stages), block_size, block_size), dtype=wp.float64, device="cuda")
        self._kkt_rhs = wp.zeros((self.num_stages, block_size, 1), dtype=wp.float64, device="cuda")

        self._cholesky_factor_launch = create_cholesky_factor_launch(
            self._kkt_diag_blocks, self._kkt_offdiag_blocks, device="cuda", dtype=wp.float64, use_cuda_graph=True
        )
        self._cholesky_solve_launch = create_cholesky_solve_launch(
            self._kkt_diag_blocks, self._kkt_offdiag_blocks, self._kkt_rhs, device="cuda", dtype=wp.float64, use_cuda_graph=True
        )
        
        self._add_to_btd_diag_kernel = create_add_on_diag_kernel(block_size, wp.float64)
        self._csr_add_to_btd_kernel = create_csr_add_btd_kernel(self.num_stages, block_size, dtype=wp.float64)


    @nvtx.annotate("MultistageKKTSolver::_update_kkt")
    def _update_kkt(self, data: SparseData) -> None:
        """
        Compute the KKT matrix:
        KKT = P + diag(x_reg) + 1/delta*A^T*A + G^T*(z_reg_inv)*G
        """
        self._kkt_diag_blocks.zero_()
        self._kkt_offdiag_blocks.zero_()

        # kkt += P
        wp.launch(
            kernel=self._csr_add_to_btd_kernel,
            dim=(data.n,),
            inputs=[
                wp.float64(1.0),
                data.P.indptr,
                data.P.indices,
                data.P.data,
                self._kkt_diag_blocks,
                self._kkt_offdiag_blocks
            ],
        )

        # kkt += diag(x_reg)
        wp.launch(
            kernel=self._add_to_btd_diag_kernel,
            dim=(data.n,),
            inputs=[self._x_reg, self._kkt_diag_blocks],
        )
        
        if data.p > 0:
            wp.launch(
                kernel=self._csr_add_to_btd_kernel,
                dim=(data.n,),
                inputs=[
                    wp.float64(1.0 / self._delta[0]),
                    self._AtA.indptr,
                    self._AtA.indices,
                    self._AtA.data,
                    self._kkt_diag_blocks,
                    self._kkt_offdiag_blocks,
                ],
            )

        if data.m > 0:
            # ! This is inefficient!
            self._GtG_scaled = data.G.T @ cpsp.diags(self._z_reg_inv) @ data.G
            wp.launch(
                kernel=self._csr_add_to_btd_kernel,
                dim=(data.n,),
                inputs=[
                    wp.float64(1.0),
                    self._GtG_scaled.indptr,
                    self._GtG_scaled.indices,
                    self._GtG_scaled.data,
                    self._kkt_diag_blocks,
                    self._kkt_offdiag_blocks,
                ],
            )

    @nvtx.annotate("MultistageKKTSolver::update_scalings_and_factor")
    def update_scalings_and_factor(self, data: MultistageData, delta: cp.ndarray, x_reg: cp.ndarray, z_reg: cp.ndarray) -> bool:
        self._delta[:] = delta
        self._x_reg[:] = x_reg
        cp.reciprocal(z_reg, out=self._z_reg_inv)

        self._update_kkt(data)
        self._cholesky_factor_launch()

        diag_has_nan = cp.isnan(cp.from_dlpack(self._kkt_diag_blocks)).any()
        offdiag_has_nan = cp.isnan(cp.from_dlpack(self._kkt_offdiag_blocks)).any()
        return not (diag_has_nan or offdiag_has_nan)

    @nvtx.annotate("MultistageKKTSolver::solve")
    def solve(self, data: MultistageData, rhs_x: cp.ndarray, rhs_y: cp.ndarray, rhs_z: cp.ndarray, delta_x: cp.ndarray, delta_y: cp.ndarray, delta_z: cp.ndarray):
        # solve KKT * dx = rhs_x + 1/delta*A^T*rhs_y + G^T*diag(z_reg_inv)*rhs_z
        delta_x[:] = rhs_x
        if data.p > 0:
            delta_x += 1/self._delta[0] * data.A.T @ rhs_y
        if data.m > 0:
            delta_x += data.G.T @ (self._z_reg_inv * rhs_z)

        # rhs_x + 1/delta*A^T*rhs_y + G^T*diag(z_reg_inv)*rhs_z
        dst = cp.from_dlpack(wp.to_dlpack(self._kkt_rhs))  # zero-copy view of Warp memory
        src = cp.asarray(delta_x, dtype=dst.dtype).reshape(dst.shape)
        cp.copyto(dst, src)
        
        self._cholesky_solve_launch()

        delta_x[:] = cp.asarray(self._kkt_rhs).reshape((data.n,))
        
        # recover delta_y and delta_z
        # dy = 1/delta * (A * dx - r_y)
        delta_y[:] = data.A @ delta_x
        delta_y -= rhs_y
        delta_y /= self._delta[0]
        # dz = z_reg_inv * (G * dx - r_z)
        delta_z[:] = data.G @ delta_x
        delta_z -= rhs_z
        delta_z *= self._z_reg_inv
    

    @nvtx.annotate("MultistageKKTSolver::eval_P_x")
    def eval_P_x(self, data: MultistageData, alpha: float, x: cp.ndarray, z: cp.ndarray):
        # TODO: customize kernels for this
        z[:] = data.P @ x * alpha

    @nvtx.annotate("MultistageKKTSolver::eval_A_xn_and_AT_xt")
    def eval_A_xn_and_AT_xt(self, data: MultistageData, alpha_n: float, xn: cp.ndarray, alpha_t: float, xt: cp.ndarray, zn: cp.ndarray, zt: cp.ndarray):
        # TODO: customize kernels for this
        zn[:] = (data.A @ xn) * alpha_n
        zt[:] = (data.A.T @ xt) * alpha_t
    
    @nvtx.annotate("MultistageKKTSolver::eval_G_xn_and_GT_xt")
    def eval_G_xn_and_GT_xt(self, data: MultistageData, alpha_n: float, xn: cp.ndarray, alpha_t: float, xt: cp.ndarray, zn: cp.ndarray, zt: cp.ndarray):
        # TODO: customize kernels for this
        zn[:] = (data.G @ xn) * alpha_n
        zt[:] = (data.G.T @ xt) * alpha_t
    