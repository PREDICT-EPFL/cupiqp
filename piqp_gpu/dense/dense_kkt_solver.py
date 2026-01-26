import cupy as cp
from cupyx.scipy.linalg import solve_triangular
import cupyx
from cupy_backends.cuda.libs import cusolver
import nvtx

from ..kkt_solver import KKTSolverBase, KKTUpdateOptions
from .dense_data import DenseData

class DenseKKTSolver(KKTSolverBase):
    """
    Dense KKT solver.
    
    It eliminate Delta_y and Delta_z to form the following system:
    P + x_reg + 1/delta*A^T*A + G^T*(z_reg)^-1*G) Delta_x = rhs_x + 1/delta*A^T*rhs_y + G^T*diag((z_reg)^-1)*rhs_z

    x_reg and z_reg are both diagonal, so we only need to store their diagonals.

    Then we can solve for Delta_y and Delta_z accordingly.
    """
    def __init__(self, data: DenseData):
        super().__init__()

        self._delta = cp.nan
        self._x_reg = cp.nan * cp.ones(data.n)  # used to store the diag(P) + reg_x, i.e., [... P_ii + reg_x_i ...]
        self._z_reg_inv = cp.nan * cp.ones(data.m)  # used to store the inverse of diag(W+delta*I), i.e., [... (s_i/z_i + delta)^-1 ...]
        
        self._kkt_mat = cp.zeros((data.n, data.n))  # Placeholder for KKT matrix
        self._rhs = cp.zeros(data.n, dtype=cp.float64)

        self._AtA = data.A.T @ data.A if data.p > 0 else cp.zeros((0, 0))
        self._AtA_scaled = cp.zeros_like(self._AtA)   # placeholder for 1/delta* A^T*A
        self._Gt_scaled = cp.zeros((data.n, data.m), dtype=cp.float64)  # placeholder for G^T * diag(z_reg_inv)
        self._GtG_scaled = cp.zeros((data.n, data.n), dtype=cp.float64)  # placeholder for G^T * diag(z_reg_inv) * G


    def update_data(self, data: DenseData, update_options: KKTUpdateOptions):
        if update_options == KKTUpdateOptions.KKT_UPDATE_A and data.p > 0:
            self._AtA = data.A.T @ data.A

    @nvtx.annotate("DenseKKTSolver::_update_kkt")
    def _update_kkt(self, data: DenseData, x_reg: cp.ndarray) -> None:
        """
        Compute the KKT matrix:
        KKT = P + x_reg + 1/delta*A^T*A + G^T*(z_reg)^-1*G
        """
        # ! can do these only for lower diagonal part
        self._kkt_mat[:, :] = data.P
        self._kkt_mat.flat[::data.n + 1] += x_reg
        
        # condense A part: kkt += 1/delta * A^T * A
        if data.p > 0:
            self._AtA_scaled[:, :] = self._AtA
            cp.multiply(self._AtA_scaled, 1.0 / self._delta, out=self._AtA_scaled)
            self._kkt_mat += self._AtA_scaled

        # condense G part: kkt += G^T * diag(z_reg_inv) * G
        if data.m > 0:
            self._Gt_scaled[:, :] = data.G.T
            cp.multiply(self._Gt_scaled, self._z_reg_inv, out=self._Gt_scaled)
            cp.matmul(self._Gt_scaled, data.G, out=self._GtG_scaled)
            self._kkt_mat += self._GtG_scaled
    
    @nvtx.annotate("DenseKKTSolver::update_scalings_and_factor")
    def update_scalings_and_factor(self, data: DenseData, delta: float, x_reg: cp.ndarray, z_reg: cp.ndarray) -> bool:
        """
        z_reg = [delta+s_i/z_i for i in 1, ..., m]
        """
        self._delta = delta
        cp.reciprocal(z_reg, out=self._z_reg_inv)
        self._update_kkt(data, x_reg)
        try:
            with cupyx.errstate(linalg='raise'):  # raise exception on factorization failure
                # TODO: this is not efficient since it generates tmp rather than in-place factorization
                # Should investigate cupy raw bindings to call cusolver directly.
                # Refer to: https://github.com/cupy/cupy/blob/main/cupy/linalg/_decomposition.py
                self._kkt_mat[:, :] = cp.linalg.cholesky(self._kkt_mat)
                cp.cuda.get_current_stream().synchronize()
            return True
        except cp.linalg.LinAlgError:  # TODO: need to investigate specific exception type raise by cupy.linalg.cholesky
            return False
    
    @nvtx.annotate("DenseKKTSolver::solve")
    def solve(self, data: DenseData, rhs_x: cp.ndarray, rhs_y: cp.ndarray, rhs_z: cp.ndarray, delta_x: cp.ndarray, delta_y: cp.ndarray, delta_z: cp.ndarray):
        """
        Solve the KKT system using the factorized KKT matrix.
        """
        # solve KKT * dx = rhs_x + 1/delta*A^T*rhs_y + G^T*diag(z_reg_inv)*rhs_z
        self._rhs[:] = rhs_x
        if data.p > 0:
            self._rhs += 1/self._delta * data.A.T @ rhs_y
        if data.m > 0:
            self._rhs += self._Gt_scaled @ rhs_z
        
        # TODO: use cusolver directly to do inplace operation for better performance
        # forward substitution
        delta_x[:] = solve_triangular(self._kkt_mat, self._rhs, lower=True, overwrite_b=True)
        # backward substitution
        delta_x[:] = solve_triangular(self._kkt_mat, delta_x, lower=True, trans='T', overwrite_b=True)
        # recover delta_y and delta_z
        # dy = 1/delta * (A * dx - r_y)
        delta_y[:] = data.A @ delta_x
        delta_y -= rhs_y
        delta_y /= self._delta
        # dz = (W+delta*I)^-1 * (G * dx - r_z)
        delta_z[:] = data.G @ delta_x
        delta_z -= rhs_z
        delta_z *= self._z_reg_inv

    def eval_P_x(self, data: DenseData, alpha: float, x: cp.ndarray, z: cp.ndarray):
        z[:] = data.P @ x * alpha
    
    def eval_A_xn_and_AT_xt(self, data: DenseData, alpha_n: float, xn: cp.ndarray, alpha_t: float, xt: cp.ndarray, zn: cp.ndarray, zt: cp.ndarray):
        zn[:] = (data.A @ xn) * alpha_n
        zt[:] = (data.A.T @ xt) * alpha_t
    
    def eval_G_xn_and_GT_xt(self, data: DenseData, alpha_n: float, xn: cp.ndarray, alpha_t: float, xt: cp.ndarray, zn: cp.ndarray, zt: cp.ndarray):
        zn[:] = (data.G @ xn) * alpha_n
        zt[:] = (data.G.T @ xt) * alpha_t