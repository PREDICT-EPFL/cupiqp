import cupy as cp
from cupyx.scipy.linalg import solve_triangular
from cupy_backends.cuda.libs import cusolver

from ..typedef import Vector
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
        
    def solve(self, data: DenseData, rhs_x: cp.ndarray, rhs_y: cp.ndarray, rhs_z: cp.ndarray, delta_x: cp.ndarray, delta_y: cp.ndarray, delta_z: cp.ndarray):
        """
        Solve the KKT system using the factorized KKT matrix.
        """
        # solve KKT * dx = rhs_x + 1/delta*A^T*rhs_y + G^T*diag(z_reg_inv)*rhs_z
        self._rhs[:] = rhs_x
        if data.p > 0:
            self._rhs += 1/self._delta * data.A.T @ rhs_y
        # ! need to be careful when union(idx_hl, idx_hu) != all indices of m
        if data.m > 0:
            self._rhs += data.G.T @ (self._z_reg_inv * rhs_z)
        
        # forward substitution
        delta_x[:] = solve_triangular(self._kkt_mat, self._rhs, lower=True, overwrite_b=True)
        # backward substitution
        delta_x[:] = solve_triangular(self._kkt_mat, delta_x, lower=True, trans='T', overwrite_b=True)
        # dy = 1/delta * (A * dx - r_y), dz = (W+delta*I)^-1 * (G * dx - r_z)
        delta_y[:] = 1/self._delta * (data.A @ delta_x - rhs_y) if data.p > 0 else cp.array([])
        delta_z[:] = self._z_reg_inv * (data.G @ delta_x - rhs_z) if data.m > 0 else cp.array([])
    
    def eval_P_x(self, data: DenseData, alpha: float, x: cp.ndarray, z: cp.ndarray):
        z[:] = data.P @ x * alpha
    
    def eval_A_xn_and_AT_xt(self, data: DenseData, alpha_n: float, xn: cp.ndarray, alpha_t: float, xt: cp.ndarray, zn: cp.ndarray, zt: cp.ndarray):
        zn[:] = (data.A @ xn) * alpha_n
        zt[:] = (data.A.T @ xt) * alpha_t
    
    def eval_G_xn_and_GT_xt(self, data: DenseData, alpha_n: float, xn: cp.ndarray, alpha_t: float, xt: cp.ndarray, zn: cp.ndarray, zt: cp.ndarray):
        zn[:] = (data.G @ xn) * alpha_n
        zt[:] = (data.G.T @ xt) * alpha_t