import cupy as cp
import nvtx

from .data import Data
from .settings import Settings
from .sparse.sparse_kkt_solver import SparseKKTSolver
from .dense.dense_kkt_solver import DenseKKTSolver
from .results import Variables

class KKTSystem:
    """
    The KKT system handles the full KKT condition.
    """
    def __init__(self, data: Data, settings: Settings):
        self._data = data
        self._x_reg = cp.nan * cp.ones(self._data.n)
        self._z_reg = cp.nan * cp.ones(self._data.m)

        self._delta = cp.nan

        if settings.kkt_solver == "sparse_ldlt":
            self._kkt_solver = SparseKKTSolver(self._data)
        elif settings.kkt_solver == "dense_cholesky":
            self._kkt_solver = DenseKKTSolver(self._data)
        else:
            raise ValueError(f"Unsupported kkt_solver: {settings.kkt_solver}")

        # store the value of slack and dual variables value at this iteration, will be used in recovering the slack step: S*delta_z + Z*delta_s = r_s
        # allocate for max possible size, but we will only use part of them according to idx_hu and idx_hl. 
        self._m_s_u = cp.zeros(self._data.m)
        self._m_s_l = cp.zeros(self._data.m)
        self._m_z_u_inv = cp.zeros(self._data.m)
        self._m_z_l_inv = cp.zeros(self._data.m)
        # allocate for max possible size, but we will only use part of them according to idx_xu and idx_xl. 
        # TODO: can be optimized later to reduce memory usage
        self._m_s_bu = cp.zeros(self._data.n)
        self._m_s_bl = cp.zeros(self._data.n)
        self._m_z_bu_inv = cp.zeros(self._data.n)
        self._m_z_bl_inv = cp.zeros(self._data.n)

        # pre-allocate memory for some variables used in factor and solve
        self._w_u_delta_inv = cp.zeros(self._data.num_hu)   # store 1./(s_u / z_u + delta)
        self._w_l_delta_inv = cp.zeros(self._data.num_hl)   # store 1./(s_l / z_l + delta)
        self._w_bu_delta_inv = cp.zeros(self._data.num_xu)  # store 1./(s_bu / z_bu + delta)
        self._w_bl_delta_inv = cp.zeros(self._data.num_xl)  # store 1./(s_bl / z_bl + delta)


    @nvtx.annotate("KKTSystem::update_scalings_and_factor")
    def update_scalings_and_factor(self, data: Data, rho: float, delta: float, vars: Variables) -> bool:
        """
        Update the scaling factors and refactor the KKT matrix.

        The variable vars is the current primal/dual variable values at this iteration, i.e., values of x, y, z_u, z_l, s_u, s_l, z_bu, z_bl, s_bu, s_bl at the current iteration.
        """
        self._delta = delta

        # store the current slack and dual variable values at this iteration
        self._m_s_u[:] = vars.s_u
        self._m_s_l[:] = vars.s_l
        self._m_s_bu[:] = vars.s_bu
        self._m_s_bl[:] = vars.s_bl
        self._m_z_u_inv[data.idx_hu] = 1. / vars.z_u[data.idx_hu]
        self._m_z_l_inv[data.idx_hl] = 1. / vars.z_l[data.idx_hl]
        self._m_z_bu_inv[data.idx_xu] = 1. / vars.z_bu[data.idx_xu]
        self._m_z_bl_inv[data.idx_xl] = 1. / vars.z_bl[data.idx_xl]

        # eliminate the box constraints by adding their contribution to x_reg and z_reg
        self._w_bu_delta_inv[:] = 1. / (self._m_s_bu[data.idx_xu] * self._m_z_bu_inv[data.idx_xu] + self._delta)
        self._w_bl_delta_inv[:] = 1. / (self._m_s_bl[data.idx_xl] * self._m_z_bl_inv[data.idx_xl] + self._delta)
        self._x_reg[:] = rho * cp.ones(self._data.n)
        self._x_reg[data.idx_xu] += self._w_bu_delta_inv
        self._x_reg[data.idx_xl] += self._w_bl_delta_inv

        self._w_u_delta_inv[:] = 1. / (vars.s_u[data.idx_hu] / vars.z_u[data.idx_hu] + self._delta)
        self._w_l_delta_inv[:] = 1. / (vars.s_l[data.idx_hl] / vars.z_l[data.idx_hl] + self._delta)
        self._z_reg[:] = cp.zeros(self._data.m)
        self._z_reg[data.idx_hu] += self._w_u_delta_inv
        self._z_reg[data.idx_hl] += self._w_l_delta_inv
        self._z_reg[:] = 1. / self._z_reg
        return self._kkt_solver.update_scalings_and_factor(data, delta, self._x_reg, self._z_reg) # ! this is implicitly assuming idx_hu and idx_hl cover all indices of inequalities 0:m
    
    @nvtx.annotate("KKTSystem::solve")
    def solve(self, data: Data, settings: Settings, rhs: Variables, lhs: Variables) -> None:
        rhs_z_u = rhs.z_u[data.idx_hu] - self._m_z_u_inv[data.idx_hu] * rhs.s_u[data.idx_hu]  # rhs_z_u - inv(Z_u) * r_s_u
        rhs_z_l = rhs.z_l[data.idx_hl] - self._m_z_l_inv[data.idx_hl] * rhs.s_l[data.idx_hl]  # rhs_z_l - inv(Z_l) * r_s_l
        rhs_z_bu = rhs.z_bu[data.idx_xu] - self._m_z_bu_inv[data.idx_xu] * rhs.s_bu[data.idx_xu]  # rhs_z_bu - inv(Z_bu) * r_s_bu
        rhs_z_bl = rhs.z_bl[data.idx_xl] - self._m_z_bl_inv[data.idx_xl] * rhs.s_bl[data.idx_xl]  # rhs_z_bl - inv(Z_bl) * r_s_bl

        rhs_x_bar = rhs.x.copy()
        rhs_x_bar[data.idx_xu] += self._w_bu_delta_inv * rhs_z_bu
        rhs_x_bar[data.idx_xl] -= self._w_bl_delta_inv * rhs_z_bl

        rhs_y = rhs.y.copy()

        # rhs_z_bar = (1./ (w_u_delta_inv + w_l_delta_inv)) * (w_u_delta_inv * rhs_z_u - w_l_delta_inv * rhs_z_l)
        tmp = cp.zeros(data.m)
        tmp[data.idx_hu] += self._w_u_delta_inv * rhs_z_u
        tmp[data.idx_hl] -= self._w_l_delta_inv * rhs_z_l
        rhs_z_bar = self._z_reg * tmp

        delta_z = cp.ones(self._data.m)
        self._kkt_solver.solve(data, rhs_x_bar, rhs_y, rhs_z_bar, lhs.x, lhs.y, delta_z)
        delta_x = lhs.x  # reference
        lhs.z_u[data.idx_hu] = self._w_u_delta_inv * (data.G[data.idx_hu, :] @ delta_x - rhs_z_u)   # delta_z_u
        lhs.z_l[data.idx_hl] = self._w_l_delta_inv * (-data.G[data.idx_hl, :] @ delta_x - rhs_z_l)  # delta_z_l
        lhs.z_bu[data.idx_xu] = self._w_bu_delta_inv * (delta_x[data.idx_xu] - rhs.z_bu[data.idx_xu] + self._m_z_bu_inv[data.idx_xu] * rhs.s_bu[data.idx_xu])  # delta_z_bu
        lhs.z_bl[data.idx_xl] = -self._w_bl_delta_inv * (delta_x[data.idx_xl] + rhs.z_bl[data.idx_xl] - self._m_z_bl_inv[data.idx_xl] * rhs.s_bl[data.idx_xl])  # delta_z_bl

        # recover slack variable steps
        lhs.s_u[data.idx_hu] = self._m_z_u_inv[data.idx_hu] * (rhs.s_u[data.idx_hu] - self._m_s_u[data.idx_hu] * lhs.z_u[data.idx_hu])  # delta_s_u = inv(Z_u) (r_s_u - S_u delta_z_u)
        lhs.s_l[data.idx_hl] = self._m_z_l_inv[data.idx_hl] * (rhs.s_l[data.idx_hl] - self._m_s_l[data.idx_hl] * lhs.z_l[data.idx_hl])  # delta_s_l = inv(Z_l) (r_s_l - S_l delta_z_l)
        lhs.s_bu[data.idx_xu] = self._m_z_bu_inv[data.idx_xu] * (rhs.s_bu[data.idx_xu] - self._m_s_bu[data.idx_xu] * lhs.z_bu[data.idx_xu])  # delta_s_bu = inv(Z_bu) (r_s_bu - S_bu delta_z_bu)
        lhs.s_bl[data.idx_xl] = self._m_z_bl_inv[data.idx_xl] * (rhs.s_bl[data.idx_xl] - self._m_s_bl[data.idx_xl] * lhs.z_bl[data.idx_xl])  # delta_s_bl = inv(Z_bl) (r_s_bl - S_bl delta_z_bl)


    @nvtx.annotate("KKTSystem::eval_P_x")
    def eval_P_x(self, data: Data, alpha: float, x: cp.ndarray, z: cp.ndarray):
        """
        Evaluate alpha * P * x
        """
        self._kkt_solver.eval_P_x(data, alpha, x, z)
    
    @nvtx.annotate("KKTSystem::eval_A_xn_and_AT_xt")
    def eval_A_xn_and_AT_xt(self, data: Data, alpha_n: float, alpha_t: float, xn: cp.ndarray, xt: cp.ndarray, zn: cp.ndarray, zt: cp.ndarray):
        """
        Evaluate Ax and A^T xt with scaling factors alpha_n and alpha_t:
        zn = alpha_n * A * xn, 
        zt = alpha_t * A^T * xt
        """
        self._kkt_solver.eval_A_xn_and_AT_xt(data, alpha_n, xn, alpha_t, xt, zn, zt)
    
    @nvtx.annotate("KKTSystem::eval_G_xn_and_GT_xt")
    def eval_G_xn_and_GT_xt(self, data: Data, alpha_n: float, alpha_t: float, xn: cp.ndarray, xt: cp.ndarray, zn: cp.ndarray, zt: cp.ndarray):
        """
        Evaluate Gx and G^T xt with scaling factors alpha_n and alpha_t:
        zn = alpha_n * G * xn, 
        zt = alpha_t * G^T * xt
        """
        self._kkt_solver.eval_G_xn_and_GT_xt(data, alpha_n, xn, alpha_t, xt, zn, zt)
    
    def kkt_matrix(self, rho: float, delta: float, vars: Variables) -> cp.ndarray:
        """
        NOTICE: this method is only for testing purpose. It returns the full KKT matrix with given rho and delta
        """
        n, p, m = self._data.n, self._data.p, self._data.m
        data = self._data
        kkt_mat = cp.zeros((self._data.n + self._data.p + 4*self._data.m + 2*(self._data.num_xl + self._data.num_xu),
                            self._data.n + self._data.p + 4*self._data.m + 2*(self._data.num_xl + self._data.num_xu)))
        
        # fill in P, A related parts
        kkt_mat[:n, :n] = self._data.P + rho * cp.eye(n)
        kkt_mat[n:n+p, :n] = self._data.A
        kkt_mat[:n, n:n+p] = self._data.A.T
        kkt_mat[n:n+p, n:n+p] = -delta * cp.eye(p)

        # fill in G related parts
        rows_start = n + p
        rows_end = rows_start + 2*m
        kkt_mat[rows_start:rows_end, :n] = cp.vstack((data.G, -data.G))
        kkt_mat[:n, rows_start:rows_end] = cp.hstack((data.G.T, -data.G.T))
        kkt_mat[rows_start:rows_start+m, rows_start:rows_start+m] = -delta * cp.eye(data.m)
        kkt_mat[rows_start+m:rows_end, rows_start+m:rows_end] = -delta * cp.eye(data.m)

        # fill in box constraint related parts
        rows_start += 2*m
        rows_end = rows_start + data.num_xu + data.num_xl
        kkt_mat[rows_start:rows_end, :n] = cp.vstack((cp.eye(n)[data.idx_xu, :], -cp.eye(n)[data.idx_xl, :]))
        kkt_mat[:n, rows_start:rows_end] = cp.hstack((cp.eye(n)[:, data.idx_xu], -cp.eye(n)[:, data.idx_xl]))
        kkt_mat[rows_start:rows_start+data.num_xu, rows_start:rows_start+data.num_xu] = -delta * cp.eye(data.num_xu)
        kkt_mat[rows_start+data.num_xu:rows_end, rows_start+data.num_xu:rows_end] = -delta * cp.eye(data.num_xl)

        # fill in slack rows for h_u
        rows_start += data.num_xu + data.num_xl
        rows_end = rows_start + data.m
        kkt_mat[rows_start:rows_end, n+p:n+p+m] = cp.diag(vars.s_u)
        kkt_mat[rows_start:rows_end, rows_start:rows_end] = cp.diag(vars.z_u)
        kkt_mat[n+p:n+p+m, rows_start:rows_end] = cp.eye(data.m)

        rows_start += data.m
        rows_end = rows_start + data.m
        kkt_mat[rows_start:rows_end, n+p+m:n+p+2*m] = cp.diag(vars.s_l)
        kkt_mat[rows_start:rows_end, rows_start:rows_end] = cp.diag(vars.z_l)
        kkt_mat[n+p+m:n+p+2*m, rows_start:rows_end] = cp.eye(data.m)

        rows_start += data.m
        rows_end = rows_start + data.num_xu
        kkt_mat[rows_start:rows_end, n+p+2*m:n+p+2*m+data.num_xu] = cp.diag(vars.s_bu)
        kkt_mat[rows_start:rows_end, rows_start:rows_end] = cp.diag(vars.z_bu)
        kkt_mat[n+p+2*m:n+p+2*m+data.num_xu, rows_start:rows_end] = cp.eye(data.num_xu)

        rows_start += data.num_xu
        rows_end = rows_start + data.num_xl
        kkt_mat[rows_start:rows_end, n+p+2*m+data.num_xu:n+p+2*m+data.num_xu+data.num_xl] = cp.diag(vars.s_bl)
        kkt_mat[rows_start:rows_end, rows_start:rows_end] = cp.diag(vars.z_bl)
        kkt_mat[n+p+2*m+data.num_xu:n+p+2*m+data.num_xu+data.num_xl, rows_start:rows_end] = cp.eye(data.num_xl)

        return kkt_mat
    
    def kkt_solution(self, rho: float, delta: float, vars: Variables, rhs: Variables) -> Variables:
        """
        Only for testing purpose: return the KKT solution for given rhs
        """
        n, p, m = self._data.n, self._data.p, self._data.m
        kkt_mat = self.kkt_matrix(rho, delta, vars)
        rhs_vector = cp.hstack((rhs.x, rhs.y, rhs.z_u, rhs.z_l, rhs.z_bu, rhs.z_bl, rhs.s_u, rhs.s_l, rhs.s_bu, rhs.s_bl))
        sol =  cp.linalg.solve(kkt_mat, rhs_vector)
        assert cp.abs(cp.max(kkt_mat @ sol - rhs_vector)) < 1e-8, "KKT solution verification failed!"
        lhs = Variables(n, p, m, self._data.num_xu, self._data.num_xl)
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
    
    