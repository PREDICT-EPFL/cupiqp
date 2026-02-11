import cupy as cp
import warp as wp
import nvtx

from .data import Data
from .settings import Settings
from .sparse.sparse_kkt_solver import SparseKKTSolver
from .dense.dense_kkt_solver import DenseKKTSolver
from .multistage.multistage_kkt_solver import MultistageKKTSolver
from .results import Variables


wp.config.enable_backward = False  # disable backward mode, cut down kernel compile time


class KKTSystem:
    """
    The KKT system handles the full KKT condition.
    """
    def __init__(self, data: Data, settings: Settings):
        self._data = data
        self._x_reg = cp.nan * cp.ones(self._data.n)
        self._z_reg = cp.nan * cp.ones(self._data.m)

        self._work_x = cp.nan * cp.zeros(self._data.n)
        self._work_z = cp.nan * cp.zeros(self._data.m)

        self._delta = cp.nan

        if settings.kkt_solver == "sparse_ldlt":
            self._kkt_solver = SparseKKTSolver(self._data)
        elif settings.kkt_solver == "dense_cholesky":
            self._kkt_solver = DenseKKTSolver(self._data)
        elif settings.kkt_solver == "multistage_block_cholesky":
            self._kkt_solver = MultistageKKTSolver(self._data, settings.multistage_block_size)
        else:
            raise ValueError(f"Unsupported kkt_solver: {settings.kkt_solver}")

        # store the value of slack and dual variables value at this iteration, will be used in recovering the slack step: S*delta_z + Z*delta_s = r_s
        # allocate for max possible size, but we will only use part of them according to idx_hu and idx_hl. 
        self._m_s_u = cp.zeros(self._data.num_hu)
        self._m_s_l = cp.zeros(self._data.num_hl)
        self._m_z_u_inv = cp.zeros(self._data.num_hu)
        self._m_z_l_inv = cp.zeros(self._data.num_hl)
        # allocate for max possible size, but we will only use part of them according to idx_xu and idx_xl. 
        # TODO: can be optimized later to reduce memory usage
        self._m_s_bu = cp.zeros(self._data.num_xu)
        self._m_s_bl = cp.zeros(self._data.num_xl)
        self._m_z_bu_inv = cp.zeros(self._data.num_xu)
        self._m_z_bl_inv = cp.zeros(self._data.num_xl)

        # pre-allocate memory for some variables used in factor and solve
        self._w_u_delta_inv = cp.zeros(self._data.num_hu)   # store 1./(s_u / z_u + delta)
        self._w_l_delta_inv = cp.zeros(self._data.num_hl)   # store 1./(s_l / z_l + delta)
        self._w_bu_delta_inv = cp.zeros(self._data.num_xu)  # store 1./(s_bu / z_bu + delta)
        self._w_bl_delta_inv = cp.zeros(self._data.num_xl)  # store 1./(s_bl / z_bl + delta)

        # pre-allocate memory for some variables used to store updated rhs in solve
        self._updated_rhs_z_u = cp.zeros(self._data.num_hu)
        self._updated_rhs_z_l = cp.zeros(self._data.num_hl)
        self._updated_rhs_z_bu = cp.zeros(self._data.num_xu)
        self._updated_rhs_z_bl = cp.zeros(self._data.num_xl)

        # create kernels
        self._eliminate_slacks_kernel = create_eliminate_slacks_kernel(self._data.num_hu, self._data.num_hl, self._data.num_xu, self._data.num_xl)
        self._eliminate_duals_kernel = create_eliminate_duals_kernel(self._data.n, self._data.m, self._data.num_hu, self._data.num_hl, self._data.num_xu, self._data.num_xl)
        self._recover_duals_kernel = create_recover_duals_kernel(self._data.num_hl, self._data.num_hu, self._data.num_xu, self._data.num_xl)
        self._recover_slacks_kernel = create_recover_slacks_kernel(self._data.num_hu, self._data.num_hl, self._data.num_xu, self._data.num_xl)

    @nvtx.annotate("KKTSystem::update_scalings_and_factor")
    def update_scalings_and_factor(self, data: Data, rho: float, delta: float, vars: Variables) -> bool:
        """
        Update the scaling factors and refactor the KKT matrix.

        The variable vars is the current primal/dual variable values at this iteration, i.e., values of x, y, z_u, z_l, s_u, s_l, z_bu, z_bl, s_bu, s_bl at the current iteration.
        """
        with nvtx.annotate("KKTSystem::update_scalings_and_factor::update_regularizations"):
            self._delta = delta

            # store the current slack and dual variable values at this iteration
            self._m_s_u[:] = vars.s_u
            self._m_s_l[:] = vars.s_l
            self._m_s_bu[:] = vars.s_bu
            self._m_s_bl[:] = vars.s_bl
            cp.reciprocal(vars.z_u, out=self._m_z_u_inv)  # better than self._m_z_u_inv[:] = 1. / vars.z_u since it avoids temporary allocation
            cp.reciprocal(vars.z_l, out=self._m_z_l_inv)
            cp.reciprocal(vars.z_bu, out=self._m_z_bu_inv)
            cp.reciprocal(vars.z_bl, out=self._m_z_bl_inv)

            # eliminate the box constraints by adding their contribution to x_reg and z_reg
            # self._w_bu_delta_inv[:] = 1. / (self._m_s_bu * self._m_z_bu_inv + self._delta)
            cp.multiply(self._m_s_bu, self._m_z_bu_inv, out=self._w_bu_delta_inv)
            cp.add(self._w_bu_delta_inv, self._delta, out=self._w_bu_delta_inv)
            cp.reciprocal(self._w_bu_delta_inv, out=self._w_bu_delta_inv)
            # self._w_bl_delta_inv[:] = 1. / (self._m_s_bl * self._m_z_bl_inv + self._delta)
            cp.multiply(self._m_s_bl, self._m_z_bl_inv, out=self._w_bl_delta_inv)
            cp.add(self._w_bl_delta_inv, self._delta, out=self._w_bl_delta_inv)
            cp.reciprocal(self._w_bl_delta_inv, out=self._w_bl_delta_inv)

            self._x_reg[:] = rho
            self._x_reg[data.idx_xu] += self._w_bu_delta_inv
            self._x_reg[data.idx_xl] += self._w_bl_delta_inv

            # self._w_u_delta_inv[:] = 1. / (vars.s_u / vars.z_u + self._delta)
            cp.multiply(self._m_s_u, self._m_z_u_inv, out=self._w_u_delta_inv)
            cp.add(self._w_u_delta_inv, self._delta, out=self._w_u_delta_inv)
            cp.reciprocal(self._w_u_delta_inv, out=self._w_u_delta_inv)
            # self._w_l_delta_inv[:] = 1. / (vars.s_l / vars.z_l + self._delta)
            cp.multiply(self._m_s_l, self._m_z_l_inv, out=self._w_l_delta_inv)
            cp.add(self._w_l_delta_inv, self._delta, out=self._w_l_delta_inv)
            cp.reciprocal(self._w_l_delta_inv, out=self._w_l_delta_inv)

            self._z_reg.fill(0.)
            self._z_reg[data.idx_hu] += self._w_u_delta_inv
            self._z_reg[data.idx_hl] += self._w_l_delta_inv
            cp.reciprocal(self._z_reg, out=self._z_reg)  # self._z_reg[:] = 1. / self._z_reg

        factor_success = self._kkt_solver.update_scalings_and_factor(data, delta, self._x_reg, self._z_reg) # ! this is implicitly assuming idx_hu and idx_hl cover all indices of inequalities 0:m
        return factor_success
    
    @nvtx.annotate("KKTSystem::solve")
    def solve(self, data: Data, settings: Settings, rhs: Variables, lhs: Variables) -> None:
        with nvtx.annotate("KKTSystem::solve::prepare_rhs"):
            # ------ elliminate slack variables from rhs
            # # ! ALTERNATIVE IMPLEMENTATION (pure cupy operations)
            # # rhs_z_u - inv(Z_u) * r_s_u
            # cp.multiply(self._m_z_u_inv, rhs.s_u, out=self._updated_rhs_z_u)
            # cp.subtract(rhs.z_u, self._updated_rhs_z_u, out=self._updated_rhs_z_u)
            # # rhs_z_l - inv(Z_l) * r_s_l
            # cp.multiply(self._m_z_l_inv, rhs.s_l, out=self._updated_rhs_z_l)
            # cp.subtract(rhs.z_l, self._updated_rhs_z_l, out=self._updated_rhs_z_l)
            # # rhs_z_bu - inv(Z_bu) * r_s_bu
            # cp.multiply(self._m_z_bu_inv, rhs.s_bu, out=self._updated_rhs_z_bu)
            # cp.subtract(rhs.z_bu, self._updated_rhs_z_bu, out=self._updated_rhs_z_bu)
            # # rhs_z_bl - inv(Z_bl) * r_s_bl
            # cp.multiply(self._m_z_bl_inv, rhs.s_bl, out=self._updated_rhs_z_bl)
            # cp.subtract(rhs.z_bl, self._updated_rhs_z_bl, out=self._updated_rhs_z_bl)

            wp.launch(
                kernel=self._eliminate_slacks_kernel,
                dim=self._data.num_hu+self._data.num_hl+self._data.num_xu+self._data.num_xl,
                inputs=[rhs.z_u, rhs.s_u, self._m_z_u_inv, self._updated_rhs_z_u,
                        rhs.z_l, rhs.s_l, self._m_z_l_inv, self._updated_rhs_z_l,
                        rhs.z_bu, rhs.s_bu, self._m_z_bu_inv, self._updated_rhs_z_bu,
                        rhs.z_bl, rhs.s_bl, self._m_z_bl_inv, self._updated_rhs_z_bl],
                device="cuda",
            )

            # ------ elliminate dual variables from rhs to yield one single rhs_z passing to kkt solver
            
            # To avoid avoid extra allocation, we use:
            # self._work_x to hold modified rhs_x (to be passed to KKTSolver), self._work_z to hold modified rhs_z (to be passed to KKTSolver)
            # use lhs.z_* to hold temporary value self._w_u_delta_inv * self._updated_rhs_z_u, and so on

            # The below code is equivalent to:
            # self._work_x[:] = rhs.x
            # self._work_x[data.idx_xu] += self._w_bu_delta_inv * self._updated_rhs_z_bu
            # self._work_x[data.idx_xl] -= self._w_bl_delta_inv * self._updated_rhs_z_bl
            # self._work_z[:] = 0.
            # self._work_z[data.idx_hu] += self._w_u_delta_inv * self._updated_rhs_z_u
            # self._work_z[data.idx_hl] -= self._w_l_delta_inv * self._updated_rhs_z_l
            # self._work_z[:] *= self._z_reg

            # # ! ALTERNATIVE IMPLEMENTATION (pure cupy operations)
            # self._work_x[:] = rhs.x
            # cp.multiply(self._w_bu_delta_inv, self._updated_rhs_z_bu, out=lhs.z_bu)
            # cp.add.at(self._work_x, data.idx_xu, lhs.z_bu)
            # cp.multiply(self._w_bl_delta_inv, self._updated_rhs_z_bl, out=lhs.z_bl)
            # cp.negative(lhs.z_bl, out=lhs.z_bl)
            # cp.add.at(self._work_x, data.idx_xl, lhs.z_bl)

            # self._work_z.fill(0)  # faster than cp.zeros assignment
            # cp.multiply(self._w_u_delta_inv, self._updated_rhs_z_u, out=lhs.z_u) # use lhs.z_u as temporary storage
            # cp.add.at(self._work_z, data.idx_hu, lhs.z_u)
            # cp.multiply(self._w_l_delta_inv, self._updated_rhs_z_l, out=lhs.z_l)
            # cp.negative(lhs.z_l, out=lhs.z_l)
            # cp.add.at(self._work_z, data.idx_hl, lhs.z_l)
            # self._work_z[:] *= self._z_reg

            wp.launch(
                kernel=self._eliminate_duals_kernel,
                dim=self._data.n + self._data.m,
                inputs=[
                    # for updating rhs_x
                    self._data.idx_xu,
                    self._data.idx_xl,
                    rhs.x,
                    self._w_bu_delta_inv,
                    self._w_bl_delta_inv,
                    self._updated_rhs_z_bu,
                    self._updated_rhs_z_bl,
                    self._work_x,
                    # for updating rhs_z
                    self._data.idx_hu,
                    self._data.idx_hl,
                    self._w_u_delta_inv,
                    self._w_l_delta_inv,
                    self._updated_rhs_z_u,
                    self._updated_rhs_z_l,
                    self._z_reg,
                    self._work_z,  # placeholder for the updated rhs_z
                ],
                device="cuda",
            )

        self._kkt_solver.solve(data, self._work_x, rhs.y, self._work_z, lhs.x, lhs.y, self._work_z)  # ! the second _work_z is used to hold delta_z, but useless anyway. Can be further optimized.

        with nvtx.annotate("KKTSystem::solve::recover_lhs"):
            self._work_z[:] = data.G @ lhs.x  # G * delta_x, where delta_x is stored in lhs.x

            # ----- recover dual variables on lhs
            # The below code is equivalent to:
            # lhs.z_u[:] = self._w_u_delta_inv * (G_dx[data.idx_hu] - self._updated_rhs_z_u)   # delta_z_u
            # lhs.z_l[:] = self._w_l_delta_inv * (-G_dx[data.idx_hl] - self._updated_rhs_z_l)  # delta_z_l
            # lhs.z_bu[:] = self._w_bu_delta_inv * (lhs.x[data.idx_xu] - rhs.z_bu + self._m_z_bu_inv * rhs.s_bu)  # delta_z_bu
            # lhs.z_bl[:] = -self._w_bl_delta_inv * (lhs.x[data.idx_xl] + rhs.z_bl - self._m_z_bl_inv * rhs.s_bl)  # delta_z_bl

            # # ! ALTERNATIVE IMPLEMENTATION (pure cupy operations)
            # lhs.z_u[:] = self._work_z[data.idx_hu]
            # lhs.z_u -= self._updated_rhs_z_u
            # lhs.z_u *= self._w_u_delta_inv

            # lhs.z_l[:] = self._work_z[data.idx_hl]
            # lhs.z_l *= -1.
            # lhs.z_l -= self._updated_rhs_z_l
            # lhs.z_l *= self._w_l_delta_inv

            # cp.multiply(self._m_z_bu_inv, rhs.s_bu, out=lhs.z_bu)
            # lhs.z_bu += lhs.x[data.idx_xu]
            # lhs.z_bu -= rhs.z_bu
            # lhs.z_bu *= self._w_bu_delta_inv

            # cp.multiply(self._m_z_bl_inv, rhs.s_bl, out=lhs.z_bl)
            # lhs.z_bl -= lhs.x[data.idx_xl]
            # lhs.z_bl -= rhs.z_bl
            # lhs.z_bl *= self._w_bl_delta_inv

            wp.launch(
                kernel=self._recover_duals_kernel,
                dim=self._data.num_hu+self._data.num_hl+self._data.num_xu+self._data.num_xl,
                inputs=[
                    self._work_z, lhs.x,
                    self._data.idx_hu, self._w_u_delta_inv, self._updated_rhs_z_u, lhs.z_u,
                    self._data.idx_hl, self._w_l_delta_inv, self._updated_rhs_z_l, lhs.z_l,
                    self._data.idx_xu, self._w_bu_delta_inv, self._m_z_bu_inv, rhs.z_bu, rhs.s_bu, lhs.z_bu,
                    self._data.idx_xl, self._w_bl_delta_inv, self._m_z_bl_inv, rhs.z_bl, rhs.s_bl, lhs.z_bl],
                device="cuda",
            )

            # ----- recover slack variable on lhs
            # The below code is equivalent to:
            # lhs.s_u[:] = self._m_z_u_inv * (rhs.s_u - self._m_s_u * lhs.z_u)  # delta_s_u = inv(Z_u) (r_s_u - S_u delta_z_u)
            # lhs.s_l[:] = self._m_z_l_inv * (rhs.s_l - self._m_s_l * lhs.z_l)  # delta_s_l = inv(Z_l) (r_s_l - S_l delta_z_l)
            # lhs.s_bu[:] = self._m_z_bu_inv * (rhs.s_bu - self._m_s_bu * lhs.z_bu)  # delta_s_bu = inv(Z_bu) (r_s_bu - S_bu delta_z_bu)
            # lhs.s_bl[:] = self._m_z_bl_inv * (rhs.s_bl - self._m_s_bl * lhs.z_bl)  # delta_s_bl = inv(Z_bl) (r_s_bl - S_bl delta_z_bl)

            # # ! ALTERNATIVE IMPLEMENTATION (pure cupy operations)
            # cp.multiply(self._m_s_u, lhs.z_u, out=lhs.s_u)
            # cp.subtract(rhs.s_u, lhs.s_u, out=lhs.s_u)
            # cp.multiply(self._m_z_u_inv, lhs.s_u, out=lhs.s_u)

            # cp.multiply(self._m_s_l, lhs.z_l, out=lhs.s_l)
            # cp.subtract(rhs.s_l, lhs.s_l, out=lhs.s_l)
            # cp.multiply(self._m_z_l_inv, lhs.s_l, out=lhs.s_l)

            # cp.multiply(self._m_s_bu, lhs.z_bu, out=lhs.s_bu)
            # cp.subtract(rhs.s_bu, lhs.s_bu, out=lhs.s_bu)
            # cp.multiply(self._m_z_bu_inv, lhs.s_bu, out=lhs.s_bu)

            # cp.multiply(self._m_s_bl, lhs.z_bl, out=lhs.s_bl)
            # cp.subtract(rhs.s_bl, lhs.s_bl, out=lhs.s_bl)
            # cp.multiply(self._m_z_bl_inv, lhs.s_bl, out=lhs.s_bl)

            wp.launch(
                kernel=self._recover_slacks_kernel,
                dim=self._data.num_hu+self._data.num_hl+self._data.num_xu+self._data.num_xl,
                inputs=[rhs.s_u, lhs.z_u, self._m_s_u, self._m_z_u_inv, lhs.s_u,
                        rhs.s_l, lhs.z_l, self._m_s_l, self._m_z_l_inv, lhs.s_l,
                        rhs.s_bu, lhs.z_bu, self._m_s_bu, self._m_z_bu_inv, lhs.s_bu,
                        rhs.s_bl, lhs.z_bl, self._m_s_bl, self._m_z_bl_inv, lhs.s_bl],
                device="cuda",
            )

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
    

def create_eliminate_duals_kernel(nx: int, nz: int, num_hu: int, num_hl: int, num_xu: int, num_xl: int):
    """Create kernel specialized for eliminating duals. Performs the operation:

        rhs.x[data.idx_xu] += self._w_bu_delta_inv * self._updated_rhs_z_bu
        rhs.x[data.idx_xl] -= self._w_bl_delta_inv * self._updated_rhs_z_bl

        rhs.z = 0.
        rhs.z[data.idx_hu] += self._w_u_delta_inv * self._updated_rhs_z_u
        rhs.z[data.idx_hl] -= self._w_l_delta_inv * self._updated_rhs_z_l
        rhs.z[:] *= self._z_reg
    """
    @wp.kernel
    def eliminate_duals_kernel(
        # prepare new rhs_x
        idx_xu: wp.array(dtype=wp.int32),
        idx_xl: wp.array(dtype=wp.int32),
        rhs_x: wp.array(dtype=wp.float64),
        w_bu_delta_inv: wp.array(dtype=wp.float64),
        w_bl_delta_inv: wp.array(dtype=wp.float64),
        rhs_z_bu: wp.array(dtype=wp.float64),
        rhs_z_bl: wp.array(dtype=wp.float64),
        rhs_x_updated: wp.array(dtype=wp.float64),  # placeholder for the updated rhs_x
        # prepare new rhs_z
        idx_hu: wp.array(dtype=wp.int32),
        idx_hl: wp.array(dtype=wp.int32),
        w_u_delta_inv: wp.array(dtype=wp.float64),
        w_l_delta_inv: wp.array(dtype=wp.float64),
        rhs_z_u: wp.array(dtype=wp.float64),
        rhs_z_l: wp.array(dtype=wp.float64),
        z_reg: wp.array(dtype=wp.float64),
        rhs_z_updated: wp.array(dtype=wp.float64),  # placeholder for the updated rhs_z
    ):
        t = wp.tid()
        nx_static = wp.static(nx)
        nz_static = wp.static(nz)
        num_hu_static = wp.static(num_hu)
        num_hl_static = wp.static(num_hl)
        num_xu_static = wp.static(num_xu)
        num_xl_static = wp.static(num_xl)

        # rhs.x[data.idx_xu] += self._w_bu_delta_inv * self._updated_rhs_z_bu
        # rhs.x[data.idx_xl] -= self._w_bl_delta_inv * self._updated_rhs_z_bl
        if t < nx_static:
            rhs_x_updated[t] = rhs_x[t]
            if t < num_xu_static:
                rhs_x_updated[idx_xu[t]] += w_bu_delta_inv[t] * rhs_z_bu[t]
            if t < num_xl_static:
                rhs_x_updated[idx_xl[t]] -= w_bl_delta_inv[t] * rhs_z_bl[t]

        # rhs.z = 0.
        # rhs.z[data.idx_hu] += self._w_u_delta_inv * self._updated_rhs_z_u
        # rhs.z[data.idx_hl] -= self._w_l_delta_inv * self._updated_rhs_z_l
        # rhs.z[:] *= self._z_reg
        elif t < nx_static + nz_static:
            tz = t - nx_static
            rhs_z_updated[tz] = wp.float64(0.)
            if tz < num_hu_static:
                rhs_z_updated[idx_hu[tz]] += w_u_delta_inv[tz] * rhs_z_u[tz]
            if tz < num_hl_static:
                rhs_z_updated[idx_hl[tz]] -= w_l_delta_inv[tz] * rhs_z_l[tz]
            rhs_z_updated[tz] *= z_reg[tz]

        else:
            return

    return eliminate_duals_kernel


def create_eliminate_slacks_kernel(num_hu: int, num_hl: int, num_xu: int, num_xl: int):
    """Create kernel specialized for eliminating slacks. Performs the operation:
    
        updated_rhs_z_u = rhs_z_u - inv(Z_u) * r_s_u
        updated_rhs_z_l = rhs_z_l - inv(Z_l) * r_s_l
        updated_rhs_z_bu = rhs_z_bu - inv(Z_bu) * r_s_bu
        updated_rhs_z_bl = rhs_z_bl - inv(Z_bl) * r_s_bl
    """
    @wp.kernel
    def eliminate_slacks_kernel(
        # h_u
        rhs_z_u: wp.array(dtype=wp.float64),
        rhs_s_u: wp.array(dtype=wp.float64),
        result_z_u_inv: wp.array(dtype=wp.float64),
        updated_rhs_z_u: wp.array(dtype=wp.float64),
        # h_l
        rhs_z_l: wp.array(dtype=wp.float64),
        rhs_s_l: wp.array(dtype=wp.float64),
        result_z_l_inv: wp.array(dtype=wp.float64),
        updated_rhs_z_l: wp.array(dtype=wp.float64),
        # x_u
        rhs_z_bu: wp.array(dtype=wp.float64),
        rhs_s_bu: wp.array(dtype=wp.float64),
        result_z_bu_inv: wp.array(dtype=wp.float64),
        updated_rhs_z_bu: wp.array(dtype=wp.float64),
        # x_l
        rhs_z_bl: wp.array(dtype=wp.float64),
        rhs_s_bl: wp.array(dtype=wp.float64),
        result_z_bl_inv: wp.array(dtype=wp.float64),
        updated_rhs_z_bl: wp.array(dtype=wp.float64),
    ):
        t = wp.tid()
        num_hu_static = wp.static(num_hu)
        num_hl_static = wp.static(num_hl)
        num_xu_static = wp.static(num_xu)
        num_xl_static = wp.static(num_xl)

        if t < num_hu_static:
            updated_rhs_z_u[t] = -result_z_u_inv[t] * rhs_s_u[t] + rhs_z_u[t]
        elif t < num_hu_static + num_hl_static:
            offset = num_hu_static
            updated_rhs_z_l[t - offset] = -result_z_l_inv[t - offset] * rhs_s_l[t - offset] + rhs_z_l[t - offset]
        elif t < num_hu_static + num_hl_static + num_xu_static:
            offset = num_hu_static + num_hl_static
            updated_rhs_z_bu[t - offset] = -result_z_bu_inv[t - offset] * rhs_s_bu[t - offset] + rhs_z_bu[t - offset]
        elif t < num_hu_static + num_hl_static + num_xu_static + num_xl_static:
            offset = num_hu_static + num_hl_static + num_xu_static
            updated_rhs_z_bl[t - offset] = -result_z_bl_inv[t - offset] * rhs_s_bl[t - offset] + rhs_z_bl[t - offset]
        else:
            return

    return eliminate_slacks_kernel


def create_recover_duals_kernel(num_hu: int, num_hl: int, num_xu: int, num_xl: int):
    """Create kernel specialized for recovering duals. Performs the operation:

        lhs.z_u[:] = self._w_u_delta_inv * (G_dx[data.idx_hu] - self._updated_rhs_z_u)
        lhs.z_l[:] = self._w_l_delta_inv * (-G_dx[data.idx_hl] - self._updated_rhs_z_l)
        lhs.z_bu[:] = self._w_bu_delta_inv * (lhs.x[data.idx_xu] - rhs.z_bu + self._m_z_bu_inv * rhs.s_bu)
        lhs.z_bl[:] = -self._w_bl_delta_inv * (lhs.x[data.idx_xl] + rhs.z_bl - self._m_z_bl_inv * rhs.s_bl)
    """
    @wp.kernel
    def recover_duals_kernel(
        G_dx: wp.array(dtype=wp.float64),
        lhs_x: wp.array(dtype=wp.float64),
        # h_u
        idx_hu: wp.array(dtype=wp.int32),
        w_u_delta_inv: wp.array(dtype=wp.float64),
        rhs_z_u: wp.array(dtype=wp.float64),
        lhs_z_u: wp.array(dtype=wp.float64),
        # h_l
        idx_hl: wp.array(dtype=wp.int32),
        w_l_delta_inv: wp.array(dtype=wp.float64),
        rhs_z_l: wp.array(dtype=wp.float64),
        lhs_z_l: wp.array(dtype=wp.float64),
        # x_u
        idx_xu: wp.array(dtype=wp.int32),
        w_bu_delta_inv: wp.array(dtype=wp.float64),
        m_z_bu_inv: wp.array(dtype=wp.float64),
        rhs_z_bu: wp.array(dtype=wp.float64),
        rhs_s_bu: wp.array(dtype=wp.float64),
        lhs_z_bu: wp.array(dtype=wp.float64),
        # x_l
        idx_xl: wp.array(dtype=wp.int32),
        w_bl_delta_inv: wp.array(dtype=wp.float64),
        m_z_bl_inv: wp.array(dtype=wp.float64),
        rhs_z_bl: wp.array(dtype=wp.float64),
        rhs_s_bl: wp.array(dtype=wp.float64),
        lhs_z_bl: wp.array(dtype=wp.float64),
    ):
        t = wp.tid()
        num_hu_static = wp.static(num_hu)
        num_hl_static = wp.static(num_hl)
        num_xu_static = wp.static(num_xu)
        num_xl_static = wp.static(num_xl)

        # lhs.z_u[:] = self._w_u_delta_inv * (G_dx[data.idx_hu] - self._updated_rhs_z_u)
        if t < num_hu_static:
            lhs_z_u[t] = G_dx[idx_hu[t]] - rhs_z_u[t]
            lhs_z_u[t] *= w_u_delta_inv[t]
        # lhs.z_l[:] = self._w_l_delta_inv * (-G_dx[data.idx_hl] - self._updated_rhs_z_l)
        elif t < num_hu_static + num_hl_static:
            offset = num_hu_static
            lhs_z_l[t - offset] = -G_dx[idx_hl[t - offset]] - rhs_z_l[t - offset]
            lhs_z_l[t - offset] *= w_l_delta_inv[t - offset]
        # lhs.z_bu[:] = self._w_bu_delta_inv * (lhs.x[data.idx_xu] - rhs.z_bu + self._m_z_bu_inv * rhs.s_bu)
        elif t < num_hu_static + num_hl_static + num_xu_static:
            offset = num_hu_static + num_hl_static
            lhs_z_bu[t - offset] = lhs_x[idx_xu[t - offset]] - rhs_z_bu[t - offset] + m_z_bu_inv[t - offset] * rhs_s_bu[t - offset]
            lhs_z_bu[t - offset] *= w_bu_delta_inv[t - offset]
        # lhs.z_bl[:] = -self._w_bl_delta_inv * (lhs.x[data.idx_xl] + rhs.z_bl - self._m_z_bl_inv * rhs.s_bl)
        elif t < num_hu_static + num_hl_static + num_xu_static + num_xl_static:
            offset = num_hu_static + num_hl_static + num_xu_static
            lhs_z_bl[t - offset] = lhs_x[idx_xl[t - offset]] + rhs_z_bl[t - offset] - m_z_bl_inv[t - offset] * rhs_s_bl[t - offset]
            lhs_z_bl[t - offset] *= -w_bl_delta_inv[t - offset]
        else:
            return

    return recover_duals_kernel


def create_recover_slacks_kernel(num_hu: int, num_hl: int, num_xu: int, num_xl: int):
    """Create kernel specialized for eliminating slacks. Performs the operation:
    
        updated_lhs_z_u = inv(Z_u) (r_s_u - S_u lhs_z_u)
        updated_lhs_s_l = inv(Z_l) (r_s_l - S_l lhs_z_l)
        updated_lhs_s_bu = inv(Z_bu) (r_s_bu - S_bu lhs_z_bu)
        updated_lhs_s_bl = inv(Z_bl) (r_s_bl - S_bl lhs_z_bl)
    """
    @wp.kernel
    def recover_slacks_kernel(
        # h_u
        rhs_s_u: wp.array(dtype=wp.float64),
        lhs_z_u: wp.array(dtype=wp.float64),
        result_s_u: wp.array(dtype=wp.float64),
        result_z_u_inv: wp.array(dtype=wp.float64),
        updated_lhs_s_u: wp.array(dtype=wp.float64),
        # h_l
        rhs_s_l: wp.array(dtype=wp.float64),
        lhs_z_l: wp.array(dtype=wp.float64),
        result_s_l: wp.array(dtype=wp.float64),
        result_z_l_inv: wp.array(dtype=wp.float64),
        updated_lhs_s_l: wp.array(dtype=wp.float64),
        # x_u
        rhs_s_bu: wp.array(dtype=wp.float64),
        lhs_z_bu: wp.array(dtype=wp.float64),
        result_s_bu: wp.array(dtype=wp.float64),
        result_z_bu_inv: wp.array(dtype=wp.float64),
        updated_lhs_s_bu: wp.array(dtype=wp.float64),
        # x_l
        rhs_s_bl: wp.array(dtype=wp.float64),
        lhs_z_bl: wp.array(dtype=wp.float64),
        result_s_bl: wp.array(dtype=wp.float64),
        result_z_bl_inv: wp.array(dtype=wp.float64),
        updated_lhs_s_bl: wp.array(dtype=wp.float64),
    ):
        t = wp.tid()
        num_hu_static = wp.static(num_hu)
        num_hl_static = wp.static(num_hl)
        num_xu_static = wp.static(num_xu)
        num_xl_static = wp.static(num_xl)

        if t < num_hu_static:
            # explicitly first do -result_s_u[t] * lhs_z_u[t], then add rhs_s_u[t], to trigger FMA, which is faster and more accurate
            updated_lhs_s_u[t] = result_z_u_inv[t] * (-result_s_u[t] * lhs_z_u[t] + rhs_s_u[t])
        elif t < num_hu_static + num_hl_static:
            offset = num_hu_static
            updated_lhs_s_l[t - offset] = result_z_l_inv[t - offset] * (-result_s_l[t - offset] * lhs_z_l[t - offset] + rhs_s_l[t - offset])
        elif t < num_hu_static + num_hl_static + num_xu_static:
            offset = num_hu_static + num_hl_static
            updated_lhs_s_bu[t - offset] = result_z_bu_inv[t - offset] * (-result_s_bu[t - offset] * lhs_z_bu[t - offset] + rhs_s_bu[t - offset])
        elif t < num_hu_static + num_hl_static + num_xu_static + num_xl_static:
            offset = num_hu_static + num_hl_static + num_xu_static
            updated_lhs_s_bl[t - offset] = result_z_bl_inv[t - offset] * (-result_s_bl[t - offset] * lhs_z_bl[t - offset] + rhs_s_bl[t - offset])
        else:
            return

    return recover_slacks_kernel
