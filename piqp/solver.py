import numpy as np
from typing import Tuple

from .utils import print_matlab_format
from .settings import Settings
from .data import Data
from .results import Result, Status, Variables
from .kkt_systems import KKTSystem

class SolverBase:
    def __init__(self):

        self.settings = Settings()
        self._data: Data = None
        self._result = Result()  # store the values of primal and dual variables of current iteration
        self._step: Variables = Variables() # store the step direction of primal and dual variables

        self._kkt_system = None
    
    def setup(self, P, c, A, b, G, h_u, h_l, x_u, x_l):
        self._data = Data(P, c, A, b, G, h_u, h_l, x_u, x_l)

        # self._result.x = np.zeros(self._data.n)
        # self._result.y = np.zeros(self._data.p)
        # self._result.z_l = -self._data.h_l
        # self._result.z_u = self._data.h_u
        # self._result.z_bl = -self._data.x_l
        # self._result.z_bu = self._data.x_u
        # self._result.s_u = np.zeros(self._data.m)
        # self._result.s_l = np.zeros(self._data.m)
        # self._result.s_bl = np.zeros(self._data.n)  # Slack variables for bound constraints (lower)
        # self._result.s_bu = np.zeros(self._data.n)  # Slack variables for bound constraints (upper)

        self._result.rho = self.settings.rho_init
        self._result.delta = self.settings.delta_init
        self._step = Variables(self._data.n, self._data.p, self._data.m)

        self._kkt_system = KKTSystem(self._data)

        
        

    def solve(self):
        self._solve_impl()

    def _solve_impl(self):
        self._result.status = Status.PIQP_UNSOLVED        

        ## ----------- first iteration --------------
        # eq(12) in Roland Schwan 2023 paper

        self._result.s_l = np.ones(self._data.m)
        self._result.s_u = np.ones(self._data.m)
        self._result.s_bl = np.ones(self._data.n)
        self._result.s_bu = np.ones(self._data.n)
        self._result.z_l = np.ones(self._data.m)
        self._result.z_u = np.ones(self._data.m)
        self._result.z_bl = np.ones(self._data.n)
        self._result.z_bu = np.ones(self._data.n)

        self._kkt_system.update_scalings_and_factor(
            self._data,
            self._result.rho,
            self._result.delta,
            self._result
        )

        res = Result()
        res.x = -self._data.c
        res.y = self._data.b
        res.z_l = -self._data.h_l
        res.z_u = self._data.h_u
        res.z_bl = -self._data.x_l
        res.z_bu = self._data.x_u
        res.s_l = np.zeros(self._data.m)
        res.s_u = np.zeros(self._data.m)
        res.s_bl = np.zeros(self._data.n)
        res.s_bu = np.zeros(self._data.n)

        full_kkt = self._kkt_system.kkt_matrix(self._result.rho, self._result.delta, res)
        print("The full KKT matrix is: ")
        print_matlab_format(full_kkt, name="KKT_Matrix")
        print("res:", res)

        self._kkt_system.solve(self._data, self.settings, res, self._result)  # getting an initial point of _result

        prox_vars = Variables(self._data.n, self._data.p, self._data.m)
        prox_vars.x = self._result.x
        prox_vars.y = self._result.y
        prox_vars.z_l = self._result.z_l
        prox_vars.z_u = self._result.z_u

        print("self._result:", self._result)

        ## ---------- remaining iterations -------------
        for iter in range(1, self.settings.max_iter):
            self._result.info.iter = iter

            self._result.info.prev_primal_res = ...

            if self.settings.verbose:
                print(
                    f"{self._result.info.iter:3d}   "
                    f"{float(self._result.info.primal_obj): .5e}   "
                    f"{float(self._result.info.dual_obj): .5e}   "
                    f"{float(self._result.info.duality_gap): .5e}   "
                    f"{float(self._result.info.primal_res): .5e}   "
                    f"{float(self._result.info.dual_res): .5e}   "
                    f"{float(self._result.info.rho): .3e}   "
                    f"{float(self._result.info.delta): .3e}   "
                    f"{float(self._result.info.mu): .3e}   "
                    f"{float(self._result.info.primal_step): .4f}   "
                    f"{float(self._result.info.dual_step): .4f}",
                    flush=True
                )

            factor_success = self._kkt_system.update_scalings_and_factor(self._data, self._result.rho, self._result.delta, self._step)
            assert factor_success, "KKT matrix factorization failed."

            self._kkt_system.solve(self._data, self.settings, res, self._step)

            # step in the non-negative orthant
            alpha_s, alpha_z = self._calculate_step()

            # avoid getting too close to the boundary
            self._result.info.primal_step = alpha_s * self.settings.tau
            self._result.info.dual_step = alpha_z * self.settings.tau

            # ------------------ update variables ------------------
            self._result.x += self._result.info.primal_step * self._step.x
            self._result.y += self._result.info.dual_step * self._step.y
            self._result.z_l += self._result.info.dual_step * self._step.z_l
            self._result.z_u += self._result.info.dual_step * self._step.z_u
            self._result.z_bl += self._result.info.dual_step * self._step.z_bl
            self._result.z_bu += self._result.info.dual_step * self._step.z_bu
            self._result.s_l += self._result.info.primal_step * self._step.s_l
            self._result.s_u += self._result.info.primal_step * self._step.s_u
            self._result.s_bl += self._result.info.primal_step * self._step.s_bl
            self._result.s_bu += self._result.info.primal_step * self._step.s_bu

            mu_prev: float = self._result.info.mu
            self._result.info.mu = self._calculate_mu()
            mu_rate: float = max(0., (mu_prev - self._result.info.mu) / mu_prev)


            # ------------------ update regularization ------------------
            self._update_residuals_nr()



            # Update the hyper parameters rho and delta

            
    def _calculate_step(self) -> Tuple[float, float]:
        """
        Compute the step length of the slack variables and dual variables. Make sure they remain non-negative.
        """
        alpha_s = 1.0
        alpha_z = 1.0


        if self._data.m > 0:
            # s_i + alpha_s * delta_s_i >= 0  =>  if delta_s_i < 0, alpha_s <= -s_i / delta_s_i
            
            # Vectorized computation of step sizes
            # for s_l and s_u (inequality constraint slacks)
            mask_s_l = self._step.s_l < 0
            mask_s_u = self._step.s_u < 0
            
            if np.any(mask_s_l):
                alpha_s = min(alpha_s, np.min(-self._result.s_l[mask_s_l] / self._step.s_l[mask_s_l]))
            if np.any(mask_s_u):
                alpha_s = min(alpha_s, np.min(-self._result.s_u[mask_s_u] / self._step.s_u[mask_s_u]))
            
            # for z_l and z_u (dual variables)
            mask_z_l = self._step.z_l < 0
            mask_z_u = self._step.z_u < 0
            
            if np.any(mask_z_l):
                alpha_z = min(alpha_z, np.min(-self._result.z_l[mask_z_l] / self._step.z_l[mask_z_l]))
            if np.any(mask_z_u):
                alpha_z = min(alpha_z, np.min(-self._result.z_u[mask_z_u] / self._step.z_u[mask_z_u]))

        if self._data.n_xl > 0:
            # for s_bl and z_bl (bound constraint slacks and duals - lower)
            mask_s_bl = self._step.s_bl < 0
            mask_z_bl = self._step.z_bl < 0
            
            if np.any(mask_s_bl):
                alpha_s = min(alpha_s, np.min(-self._result.s_bl[mask_s_bl] / self._step.s_bl[mask_s_bl]))
            if np.any(mask_z_bl):
                alpha_z = min(alpha_z, np.min(-self._result.z_bl[mask_z_bl] / self._step.z_bl[mask_z_bl]))

        if self._data.n_xu > 0:
            # for s_bu and z_bu (bound constraint slacks and duals - upper)
            mask_s_bu = self._step.s_bu < 0
            mask_z_bu = self._step.z_bu < 0
            
            if np.any(mask_s_bu):
                alpha_s = min(alpha_s, np.min(-self._result.s_bu[mask_s_bu] / self._step.s_bu[mask_s_bu]))
            if np.any(mask_z_bu):
                alpha_z = min(alpha_z, np.min(-self._result.z_bu[mask_z_bu] / self._step.z_bu[mask_z_bu]))
        
        return alpha_s, alpha_z
    
    
    def _calculate_mu(self) -> float:
        mu = (self._result.s_l.dot(self._result.z_l)
                + self._result.s_u.dot(self._result.z_u) \
                + self._result.s_bl.dot(self._result.z_bl) \
                + self._result.s_bu.dot(self._result.z_bu)) \
                / (self._data.n_hl + self._data.n_hu + self._data.n_xl + self._data.n_xu)
        return mu


    def _update_residuals_nr(self):
        """
        Compute the non-regularized primal and dual residuals:
        primal_residual = ||[A*x-b; G*x-h+s]||_inf
        dual_residual = ||P*x + c + AT*y + GT*z||_inf
        """
        # we calculate these term here first to be able to reuse temporary vectors
        # res_nr.y = -A * x
        # work_x = A^T * y
        A_x, At_xt = self._kkt_system.eval_A_xn_and_AT_xt(self._data, -1., 1., self._result.x, self._result.y)
        # res_nr.z_u = -G * x
        # res_nr.z_l = G * x
        # work_x += G^T * (z_u - z_l)
        # work_z.noalias() = m_result.z_u - m_result.z_l

        self._data.P @ self._result.x + self._data.c + self._data.A.transpose() * self._result.y + self._data.G.transpose() * (self._result.z_u - self._result.z_l) + bound_terms

        

# class DenseSolver(SolverBase):
#     def __init__(self):
#         super().__init__()
#         self._kkt_system = De

#     def _solve_impl(self):
#         # Implement the dense solver logic here
#         self._result.status = Status.PIQP_SOLVED
#         # Fill in other result fields as necessary