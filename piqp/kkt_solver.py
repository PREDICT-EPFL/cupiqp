from abc import ABC, abstractmethod
import numpy as np
from scipy.linalg import solve_triangular, cholesky, ldl
import scipy.sparse as sp

from piqp.typedef import Vector, Matrix
from piqp.data import Data
from piqp.kkt_fwd import KKTUpdateOptions


class KKTSolverBase(ABC):
    """Represent the system with the following form:
    [P+x_reg     A^T      G^T    ] [Delta_x] = [rhs_x]  
    [A         -delta*I     0    ] [Delta_y] = [rhs_y]
    [G           0     -(z_reg)  ] [Delta_z] = [rhs_z]

    where W = diag(s_i/z_i) for i=1,...,m
    Actually we should note the delta*I as regularization term since it's not necessarily delta*I if there are contributions from the box constriants, which are denoted as reg_x and reg_z.
    """
    def __init__(self, data: Data):
        self._x_reg = np.nan * np.ones(data.n)
        self._z_reg_inv = np.nan * np.ones(data.m)

    @abstractmethod
    def update_scalings_and_factor(self, data: Data, delta: float, x_reg: np.ndarray, z_reg: np.ndarray) -> bool:
        pass

    @abstractmethod
    def solve(self, data: Data, rhs_x, rhs_y, rhs_z):
        pass

    def eval_P_x(self, data: Data, alpha: float, x):
        """
        Evaluate alpha * P * x
        """
        # TODO: in PIQP cpp code the data stores upper triangular part of P only, so we need to adjust accordingly
        return alpha * (data.P @ x)

    def eval_A_xn_and_AT_xt(self, data: Data, alpha_n: float, alpha_t: float, xn, xt):
        """
        Evaluate Ax and A^T xt with scaling factors alpha_n and alpha_t:
        zn = alpha_n * A * xn, zt = alpha_t * A^T * xt
        """
        zn = alpha_n * (data.A @ xn)
        zt = alpha_t * (data.A.T @ xt)
        return zn, zt
    
    def eval_G_xn_and_GT_xt(self, data: Data, alpha_n: float, alpha_t: float, xn, xt):
        """
        Evaluate Gx and G^T xt with scaling factors alpha_n and alpha_t:
        zn = alpha_n * G * xn, zt = alpha_t * G^T * xt
        """
        zn = alpha_n * (data.G @ xn)
        zt = alpha_t * (data.G.T @ xt)
        return zn, zt


class DenseKKTSolver(KKTSolverBase):
    """
    Dense KKT solver.
    
    It eliminate Delta_y and Delta_z to form the following system:
    P + x_reg + 1/delta*A^T*A + G^T*(z_reg)^-1*G) Delta_x = rhs_x + 1/delta*A^T*rhs_y + G^T*diag((z_reg)^-1)*rhs_z

    x_reg and z_reg are both diagonal, so we only need to store their diagonals.

    Then we can solve for Delta_y and Delta_z accordingly.
    """
    def __init__(self, data: Data):
        super().__init__(data)

        self._delta = np.nan
        self._x_reg = np.nan * np.ones(data.n)  # used to store the diag(P) + reg_x, i.e., [... P_ii + reg_x_i ...]
        self._z_reg_inv = np.nan * np.ones(data.m)  # used to store the inverse of diag(W+delta*I), i.e., [... (s_i/z_i + delta)^-1 ...]
        
        self._kkt_mat = np.zeros((data.n, data.n))  # Placeholder for KKT matrix

        self._AtA = data.A.T @ data.A if data.p > 0 else np.zeros((0, 0))

    def update_data(self, data: Data, update_options: KKTUpdateOptions):
        if update_options == KKTUpdateOptions.KKT_UPDATE_A and data.p > 0:
            self._AtA = data.A.T @ data.A

    def _update_kkt(self, data: Data, x_reg: np.ndarray) -> None:
        """
        Compute the KKT matrix:
        KKT = P + x_reg + 1/delta*A^T*A + G^T*(z_reg)^-1*G
        """
        # ! can do these only for lower diagonal part
        self._kkt_mat = data.P + np.diag(x_reg)
        
        if data.p > 0:
            self._kkt_mat += (1.0 / self._delta) * self._AtA

        if data.m > 0:
            self._kkt_mat += data.G.T @ np.diag(self._z_reg_inv) @ data.G
    
    def update_scalings_and_factor(self, data: Data, delta: float, x_reg: np.ndarray, z_reg: np.ndarray) -> bool:
        """
        z_reg = [delta+s_i/z_i for i in 1, ..., m]
        """
        self._delta = delta
        self._z_reg_inv = 1.0 / z_reg
        self._update_kkt(data, x_reg)
        try:
            self._kkt_mat = cholesky(self._kkt_mat, lower=True, overwrite_a=True, check_finite=True)  # ? overwrite_a seems not making a difference?
            return True
        except np.linalg.LinAlgError:
            return False
        
    def solve(self, data: Data, rhs_x, rhs_y, rhs_z):
        """
        Solve the KKT system using the factorized KKT matrix.
        """
        # Solve KKT * dx = rhs_x + 1/delta*A^T*rhs_y + G^T*diag(z_reg_inv)*rhs_z
        lhs_x_tmp = rhs_x.copy()
        if data.p > 0:
            lhs_x_tmp += 1/self._delta * data.A.T @ rhs_y
        if data.m > 0:
            lhs_x_tmp += data.G.T @ np.diag(self._z_reg_inv) @ rhs_z
        # Solve L * L^T * dx = effective_rhs
        y = solve_triangular(self._kkt_mat, lhs_x_tmp, lower=True, overwrite_b=False)  # TODO: inplace
        lhs_x = solve_triangular(self._kkt_mat.T, y, lower=False, overwrite_b=False)  # TODO: inplace

        # dy = 1/delta * (A * dx - r_y)
        lhs_y = 1/self._delta * (data.A @ lhs_x - rhs_y) if data.p > 0 else np.array([])

        # dz = (W+delta*I)^-1 * (G * dx - r_z)
        lhs_z = np.diag(self._z_reg_inv) @ (data.G @ lhs_x - rhs_z) if data.m > 0 else np.array([])

        return lhs_x, lhs_y, lhs_z


class SparseKKTSolver(KKTSolverBase):
    """
    Sparse KKT solver.
    """
    def __init__(self, data: Data):
        super().__init__(data)
        print("Initializing SparseKKTSolver...")
        # ! at this moment it's not really sparse. We're justing not eliminating A and G to use ldlt factorization on the full KKT matrix.
        self._delta = np.nan
        self._x_reg = np.nan * np.ones(data.n)  # used to store the diag(P) + reg_x, i.e., [... P_ii + reg_x_i ...]
        self._z_reg_inv = np.nan * np.ones(data.m)  # used to store the inverse of diag(W+delta*I), i.e., [... (s_i/z_i + delta)^-1 ...]
        
        self._kkt_mat = np.zeros((data.n + data.p + data.m, data.n + data.p + data.m))  # Placeholder for KKT matrix

        self._AtA = data.A.T @ data.A if data.p > 0 else sp.csc_matrix((0, 0))

    def update_scalings_and_factor(self, data: Data, delta: float, x_reg: np.ndarray, z_reg: np.ndarray) -> bool:
        self._delta = delta
        self._x_reg = x_reg
        self._z_reg_inv = 1.0 / z_reg

        n, p, m = data.n, data.p, data.m
        self._kkt_mat[:n, :n] = data.P + np.diag(self._x_reg)
        self._kkt_mat[:n, n:n+p] = data.A.T
        self._kkt_mat[:n, n+p:] = data.G.T
        self._kkt_mat[n:n+p, :n] = data.A
        self._kkt_mat[n:n+p, n:n+p] = -self._delta * np.eye(p)
        self._kkt_mat[n+p:, :n] = data.G
        self._kkt_mat[n+p:, n+p:] = -np.diag(1.0 / self._z_reg_inv)

        # lu, d, perm = ldl(self._kkt_mat, lower=True, hermitian=True, check_finite=True)
        # raise NotImplementedError("SparseKKTSolver is not yet implemented.")
        return True

    def solve(self, data: Data, rhs_x, rhs_y, rhs_z):
        rhs = np.concatenate([rhs_x, rhs_y, rhs_z])
        lhs = np.linalg.solve(self._kkt_mat, rhs)
        delta_x = lhs[:len(rhs_x)]
        delta_y = lhs[len(rhs_x):len(rhs_x)+len(rhs_y)]
        delta_z = lhs[len(rhs_x)+len(rhs_y):]
        return delta_x, delta_y, delta_z
