import numpy as np

from .data import Data
from .settings import Settings
from .kkt_solver import KKTSolverBase, DenseKKTSolver, SparseKKTSolver
from .results import Variables

class KKTSystem:
    """
    The KKT system handles the full KKT condition.
    """
    def __init__(self, data: Data):
        self._data = data
        self._kkt_solver = DenseKKTSolver(self._data)
        self._x_reg = np.nan * np.ones(self._data.n)
        self._z_reg = np.nan * np.ones(self._data.m)

        self._delta = np.nan

        # store the value of slack and dual variables value at this iteration, will be used in recovering the slack step: S*delta_z + Z*delta_s = r_s
        self._m_s_u = np.zeros(self._data.m)
        self._m_s_l = np.zeros(self._data.m)
        self._m_z_u_inv = np.zeros(self._data.m)
        self._m_z_l_inv = np.zeros(self._data.m)
        self._m_s_bu = np.zeros(self._data.num_xu)
        self._m_s_bl = np.zeros(self._data.num_xl)
        self._m_z_bu_inv = np.zeros(self._data.num_xu)
        self._m_z_bl_inv = np.zeros(self._data.num_xl)



    def update_scalings_and_factor(self, data: Data, rho: float, delta: float, vars: Variables) -> bool:
        self._delta = delta

        # store the current slack and dual variable values at this iteration
        self._m_s_u = vars.s_u
        self._m_s_l = vars.s_l
        self._m_s_bu = vars.s_bu
        self._m_s_bl = vars.s_bl
        self._m_z_u_inv = 1. / vars.z_u
        self._m_z_l_inv = 1. / vars.z_l
        self._m_z_bu_inv = 1. / vars.z_bu
        self._m_z_bl_inv = 1. / vars.z_bl

        # TODO: vars will be used if we have box constraints
        w_bu_delta = vars.s_bu / vars.z_bu + self._delta
        w_bl_delta = vars.s_bl / vars.z_bl + self._delta
        self._x_reg = rho * np.ones(self._data.n)
        self._x_reg[data.idx_xu] += + 1./w_bu_delta
        self._x_reg[data.idx_xl] += + 1./w_bl_delta

        w_u_delta_inv = 1. / (vars.s_u / vars.z_u + self._delta)
        w_l_delta_inv = 1. / (vars.s_l / vars.z_l + self._delta)
        self._z_reg = 1. / (w_u_delta_inv + w_l_delta_inv)

        return self._kkt_solver.update_scalings_and_factor(data, delta, self._x_reg, self._z_reg)

    def solve(self, data: Data, settings: Settings, rhs: Variables, lhs: Variables) -> None:
        
        w_bu_delta_inv = 1. / (rhs.s_bu / rhs.z_bu + self._delta)
        w_bl_delta_inv = 1. / (rhs.s_bl / rhs.z_bl + self._delta)

        rhs_z_u = rhs.z_u - self._m_z_u_inv * rhs.s_u  # rhs_z_u - inv(Z_u) * r_s_u
        rhs_z_l = rhs.z_l - self._m_z_l_inv * rhs.s_l  # rhs_z_l - inv(Z_l) * r_s_l
        rhs_z_bu = rhs.z_bu - self._m_z_bu_inv * rhs.s_bu  # rhs_z_bu - inv(Z_bu) * r_s_bu
        rhs_z_bl = rhs.z_bl - self._m_z_bl_inv * rhs.s_bl  # rhs_z_bl - inv(Z_bl) * r_s_bl

        rhs_x_bar = rhs.x
        rhs_x_bar[data.idx_xu] += w_bu_delta_inv * rhs_z_bu
        rhs_x_bar[data.idx_xl] -= w_bl_delta_inv * rhs_z_bl

        rhs_y = rhs.y

        w_u_delta_inv = 1. / (rhs.s_u / rhs.z_u + self._delta)
        w_l_delta_inv = 1. / (rhs.s_l / rhs.z_l + self._delta)

        rhs_z_bar = 1./ (w_u_delta_inv + w_l_delta_inv) * (w_u_delta_inv * rhs_z_u - w_l_delta_inv * rhs_z_l)

        delta_x, delta_y, delta_z = self._kkt_solver.solve(data, rhs_x_bar, rhs_y, rhs_z_bar)

        # recover primal/dual step from kkt solution
        lhs.x = delta_x  # delta_x
        lhs.y = delta_y  # delta_y
        lhs.z_u = w_u_delta_inv * (data.G @ delta_x - rhs_z_u)   # delta_z_u
        lhs.z_l = w_l_delta_inv * (-data.G @ delta_x - rhs_z_l)  # delta_z_l
        lhs.z_bu = w_bu_delta_inv * (delta_x[data.idx_xu] - rhs.z_bu + self._m_z_bu_inv * rhs.s_bu)  # delta_z_bu
        lhs.z_bl = -w_bl_delta_inv * (delta_x[data.idx_xl] + rhs.z_bl - self._m_z_bl_inv * rhs.s_bl)  # delta_z_bl

        # recover slack variable steps
        lhs.s_u = self._m_z_u_inv * (rhs.s_u - self._m_s_u * lhs.z_u)  # delta_s_u = inv(Z_u) (r_s_u - S_u delta_z_u)
        lhs.s_l = self._m_z_l_inv * (rhs.s_l - self._m_s_l * lhs.z_l)  # delta_s_l = inv(Z_l) (r_s_l - S_l delta_z_l)
        lhs.s_bu = self._m_z_bu_inv * (rhs.s_bu - self._m_s_bu * lhs.z_bu)  # delta_s_bu = inv(Z_bu) (r_s_bu - S_bu delta_z_bu)
        lhs.s_bl = self._m_z_bl_inv * (rhs.s_bl - self._m_s_bl * lhs.z_bl)  # delta_s_bl = inv(Z_bl) (r_s_bl - S_bl delta_z_bl)
        
        # return lhs

    def eval_P_x(self, data: Data, alpha: float, x: np.ndarray) -> np.ndarray:
        """
        Evaluate alpha * P * x
        """
        return self._kkt_solver.eval_P_x(data, alpha, x)
    
    def eval_A_xn_and_AT_xt(self, data: Data, alpha_n: float, alpha_t: float, xn: np.ndarray, xt: np.ndarray):
        """
        Evaluate Ax and A^T xt with scaling factors alpha_n and alpha_t:
        zn = alpha_n * A * xn, 
        zt = alpha_t * A^T * xt
        """
        return self._kkt_solver.eval_A_xn_and_AT_xt(data, alpha_n, alpha_t, xn, xt)
    
    def eval_G_xn_and_GT_xt(self, data: Data, alpha_n: float, alpha_t: float, xn: np.ndarray, xt: np.ndarray):
        """
        Evaluate Gx and G^T xt with scaling factors alpha_n and alpha_t:
        zn = alpha_n * G * xn, 
        zt = alpha_t * G^T * xt
        """
        return self._kkt_solver.eval_G_xn_and_GT_xt(data, alpha_n, alpha_t, xn, xt)
    
    def kkt_matrix(self, rho: float, delta: float, vars: Variables) -> np.ndarray:
        """
        Only for testing purpose: return the full KKT matrix with given rho and delta
        """
        n, p, m = self._data.n, self._data.p, self._data.m
        data = self._data
        kkt_mat = np.zeros((self._data.n + self._data.p + 4*self._data.m + 2*(self._data.num_xl + self._data.num_xu),
                            self._data.n + self._data.p + 4*self._data.m + 2*(self._data.num_xl + self._data.num_xu)))
        
        # fill in P, A related parts
        kkt_mat[:n, :n] = self._data.P + rho * np.eye(n)
        kkt_mat[n:n+p, :n] = self._data.A
        kkt_mat[:n, n:n+p] = self._data.A.T
        kkt_mat[n:n+p, n:n+p] = -delta * np.eye(p)

        # fill in G related parts
        rows_start = n + p
        rows_end = rows_start + 2*m
        kkt_mat[rows_start:rows_end, :n] = np.vstack((data.G, -data.G))
        kkt_mat[:n, rows_start:rows_end] = np.hstack((data.G.T, -data.G.T))
        kkt_mat[rows_start:rows_start+m, rows_start:rows_start+m] = -delta * np.eye(data.m)
        kkt_mat[rows_start+m:rows_end, rows_start+m:rows_end] = -delta * np.eye(data.m)

        # fill in box constraint related parts
        rows_start += 2*m
        rows_end = rows_start + data.num_xu + data.num_xl
        kkt_mat[rows_start:rows_end, :n] = np.vstack((np.eye(n)[data.idx_xu, :], -np.eye(n)[data.idx_xl, :]))
        kkt_mat[:n, rows_start:rows_end] = np.hstack((np.eye(n)[:, data.idx_xu], -np.eye(n)[:, data.idx_xl]))
        kkt_mat[rows_start:rows_start+data.num_xu, rows_start:rows_start+data.num_xu] = -delta * np.eye(data.num_xu)
        kkt_mat[rows_start+data.num_xu:rows_end, rows_start+data.num_xu:rows_end] = -delta * np.eye(data.num_xl)

        # fill in slack rows for h_u
        rows_start += data.num_xu + data.num_xl
        rows_end = rows_start + data.m
        kkt_mat[rows_start:rows_end, n+p:n+p+m] = np.diag(vars.s_u)
        kkt_mat[rows_start:rows_end, rows_start:rows_end] = np.diag(vars.z_u)
        kkt_mat[n+p:n+p+m, rows_start:rows_end] = np.eye(data.m)

        rows_start += data.m
        rows_end = rows_start + data.m
        kkt_mat[rows_start:rows_end, n+p+m:n+p+2*m] = np.diag(vars.s_l)
        kkt_mat[rows_start:rows_end, rows_start:rows_end] = np.diag(vars.z_l)
        kkt_mat[n+p+m:n+p+2*m, rows_start:rows_end] = np.eye(data.m)

        rows_start += data.m
        rows_end = rows_start + data.num_xu
        kkt_mat[rows_start:rows_end, n+p+2*m:n+p+2*m+data.num_xu] = np.diag(vars.s_bu)
        kkt_mat[rows_start:rows_end, rows_start:rows_end] = np.diag(vars.z_bu)
        kkt_mat[n+p+2*m:n+p+2*m+data.num_xu, rows_start:rows_end] = np.eye(data.num_xu)

        rows_start += data.num_xu
        rows_end = rows_start + data.num_xl
        kkt_mat[rows_start:rows_end, n+p+2*m+data.num_xu:n+p+2*m+data.num_xu+data.num_xl] = np.diag(vars.s_bl)
        kkt_mat[rows_start:rows_end, rows_start:rows_end] = np.diag(vars.z_bl)
        kkt_mat[n+p+2*m+data.num_xu:n+p+2*m+data.num_xu+data.num_xl, rows_start:rows_end] = np.eye(data.num_xl)

        return kkt_mat
    
    def kkt_solution(self, rho: float, delta: float, vars: Variables, rhs: Variables) -> Variables:
        """
        Only for testing purpose: return the KKT solution for given rhs
        """
        n, p, m = self._data.n, self._data.p, self._data.m
        kkt_mat = self.kkt_matrix(rho, delta, vars)
        rhs_vector = np.hstack((rhs.x, rhs.y, rhs.z_u, rhs.z_l, rhs.z_bu, rhs.z_bl, rhs.s_u, rhs.s_l, rhs.s_bu, rhs.s_bl))
        sol = np.linalg.solve(kkt_mat, rhs_vector)
        assert np.abs(np.max(kkt_mat @ sol - rhs_vector)) < 1e-8, "KKT solution verification failed!"
        lhs = Variables(self._data)
        lhs.x = sol[:n]
        lhs.y = sol[n:n + p]
        lhs.z_u = sol[n + p:n + p + m]
        lhs.z_l = sol[n + p + m:n + p + 2*m]

        idx = n + p + 2*m
        lhs.z_bu = sol[idx : idx + self._data.num_xu]

        idx += self._data.num_xu
        lhs.z_bl = sol[idx : idx + self._data.num_xl]

        idx += self._data.num_xl
        lhs.s_u = sol[idx : idx + m]

        idx += m
        lhs.s_l = sol[idx : idx + m]

        idx += m
        lhs.s_bu = sol[idx : idx + self._data.num_xu]

        idx += self._data.num_xu
        lhs.s_bl = sol[idx : idx + self._data.num_xl]

        return lhs
    
    