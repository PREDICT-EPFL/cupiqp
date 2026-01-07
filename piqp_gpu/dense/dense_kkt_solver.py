import cupy as cp
from cupyx.scipy.linalg import solve_triangular
from cupy_backends.cuda.libs import cusolver

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
        super().__init__(data)

        self._delta = cp.nan
        self._x_reg = cp.nan * cp.ones(data.n)  # used to store the diag(P) + reg_x, i.e., [... P_ii + reg_x_i ...]
        self._z_reg_inv = cp.nan * cp.ones(data.m)  # used to store the inverse of diag(W+delta*I), i.e., [... (s_i/z_i + delta)^-1 ...]
        
        self._kkt_mat = cp.zeros((data.n, data.n))  # Placeholder for KKT matrix

        self._AtA = data.A.T @ data.A if data.p > 0 else cp.zeros((0, 0))


    def update_data(self, data: DenseData, update_options: KKTUpdateOptions):
        if update_options == KKTUpdateOptions.KKT_UPDATE_A and data.p > 0:
            self._AtA = data.A.T @ data.A

    def _update_kkt(self, data: DenseData, x_reg: cp.ndarray) -> None:
        """
        Compute the KKT matrix:
        KKT = P + x_reg + 1/delta*A^T*A + G^T*(z_reg)^-1*G
        """
        # ! can do these only for lower diagonal part
        self._kkt_mat = data.P + cp.diag(x_reg)
        
        if data.p > 0:
            self._kkt_mat += (1.0 / self._delta) * self._AtA

        if data.m > 0:
            self._kkt_mat += data.G.T @ cp.diag(self._z_reg_inv) @ data.G
    
    def update_scalings_and_factor(self, data: DenseData, delta: float, x_reg: cp.ndarray, z_reg: cp.ndarray) -> bool:
        """
        z_reg = [delta+s_i/z_i for i in 1, ..., m]
        """
        self._delta = delta
        self._z_reg_inv = 1.0 / z_reg
        self._update_kkt(data, x_reg)
        try:
            self._kkt_mat = cp.linalg.cholesky(self._kkt_mat)
            return True
        except cp.linalg.LinAlgError:  # TODO: need to investigate specific exception type raise by cupy.linalg.cholesky
            return False
        
    def solve(self, data: DenseData, rhs_x, rhs_y, rhs_z):
        """
        Solve the KKT system using the factorized KKT matrix.
        """
        # Solve KKT * dx = rhs_x + 1/delta*A^T*rhs_y + G^T*diag(z_reg_inv)*rhs_z
        lhs_x_tmp = rhs_x.copy()
        if data.p > 0:
            lhs_x_tmp += 1/self._delta * data.A.T @ rhs_y
        if data.m > 0:
            lhs_x_tmp += data.G.T @ (self._z_reg_inv * rhs_z)
        # Solve L * L^T * dx = effective_rhs
        
        y = solve_triangular(self._kkt_mat, lhs_x_tmp, lower=True, overwrite_b=False)  # TODO: inplace
        lhs_x = solve_triangular(self._kkt_mat, y, lower=True, trans='T', overwrite_b=False)  # TODO: inplace

        # dy = 1/delta * (A * dx - r_y)
        lhs_y = 1/self._delta * (data.A @ lhs_x - rhs_y) if data.p > 0 else cp.array([])

        # dz = (W+delta*I)^-1 * (G * dx - r_z)
        lhs_z = self._z_reg_inv * (data.G @ lhs_x - rhs_z) if data.m > 0 else cp.array([])

        return lhs_x, lhs_y, lhs_z

    def eval_P_x(self, data: DenseData, alpha: float, x: cp.ndarray, z: cp.ndarray):
        z[:] = data.P @ x * alpha
    
    def eval_A_xn_and_AT_xt(self, data: DenseData, alpha_n: float, xn: cp.ndarray, alpha_t: float, xt: cp.ndarray, zn: cp.ndarray, zt: cp.ndarray):
        zn[:] = (data.A @ xn) * alpha_n
        zt[:] = (data.A.T @ xt) * alpha_t
    
    def eval_G_xn_and_GT_xt(self, data: DenseData, alpha_n: float, xn: cp.ndarray, alpha_t: float, xt: cp.ndarray, zn: cp.ndarray, zt: cp.ndarray):
        zn[:] = (data.G @ xn) * alpha_n
        zt[:] = (data.G.T @ xt) * alpha_t