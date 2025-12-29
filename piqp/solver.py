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
        self._step: Variables = None # store the step direction of primal and dual variables

        self._kkt_system = None
    
    def setup(self, P, c, A, b, G, h_u, h_l, x_u, x_l):
        self._data = Data(P, c, A, b, G, h_u, h_l, x_u, x_l)

        self._result.info.rho = self.settings.rho_init
        self._result.info.delta = self.settings.delta_init
        self._step = Variables(self._data.n, self._data.p, self._data.m, self._data.num_xu, self._data.num_xl)

        self._kkt_system = KKTSystem(self._data)

        self._res_nr = Variables(self._data.n, self._data.p, self._data.m, self._data.num_xu, self._data.num_xl)  # used to store the non-regularized residuals
        self._res_r = Variables(self._data.n, self._data.p, self._data.m, self._data.num_xu, self._data.num_xl)  # used to store the regularized residuals
        
        

    def solve(self):
        self._solve_impl()

    def _solve_impl(self):
        self._result.info.status = Status.PIQP_UNSOLVED 
        self._result.info.iter = 0
        self._result.info.reg_limit = self.settings.reg_lower_limit
        self._result.info.factor_retires = 0
        self._result.info.no_primal_update = 0
        self._result.info.no_dual_update = 0
        self._result.info.mu = 0.
        self._result.info.primal_step = 0.
        self._result.info.dual_step = 0.
        self._result.info.rho = self.settings.rho_init
        self._result.info.delta = self.settings.delta_init     

        if self.settings.verbose:
            print("iter   prim_obj        dual_obj        duality_gap    prim_res       dual_res       rho          delta        mu           p_step    d_step\n")  

        ## ----------- initial iteration --------------
        # eq(12) in Roland Schwan 2023 paper

        self._result.x = np.zeros(self._data.n)
        self._result.y = np.zeros(self._data.p)
        self._result.s_l = np.zeros(self._data.m)
        self._result.s_u = np.zeros(self._data.m)
        self._result.s_bl = np.zeros(self._data.n)
        self._result.s_bu = np.zeros(self._data.n)
        self._result.z_l = np.zeros(self._data.m)
        self._result.z_u = np.zeros(self._data.m)
        self._result.z_bl = np.zeros(self._data.n)
        self._result.z_bu = np.zeros(self._data.n)

        self._result.s_l[self._data.idx_hl] = 1.0
        self._result.z_l[self._data.idx_hl] = 1.0
        self._result.s_u[self._data.idx_hu] = 1.0
        self._result.z_u[self._data.idx_hu] = 1.0
        self._result.s_bl[self._data.idx_xl] = 1.0
        self._result.z_bl[self._data.idx_xl] = 1.0
        self._result.s_bu[self._data.idx_xu] = 1.0
        self._result.z_bu[self._data.idx_xu] = 1.0

        self._kkt_system.update_scalings_and_factor(
            self._data,
            self._result.info.rho,
            self._result.info.delta,
            self._result
        )

        self._res = Result()  # used to store the right hand side of KKT system
        self._res.x = -self._data.c.copy()
        self._res.y = self._data.b.copy()
        self._res.z_l = -self._data.h_l.copy()
        self._res.z_u = self._data.h_u.copy()
        self._res.z_bl = -self._data.x_l.copy()
        self._res.z_bu = self._data.x_u.copy()
        self._res.s_l = np.zeros(self._data.m)
        self._res.s_u = np.zeros(self._data.m)
        self._res.s_bl = np.zeros(self._data.n)
        self._res.s_bu = np.zeros(self._data.n)

        # full_kkt = self._kkt_system.kkt_matrix(self._result.rho, self._result.delta, res)
        # print("The full KKT matrix is: ")
        # print_matlab_format(full_kkt, name="KKT_Matrix")
        # print("res:", res)

        self._kkt_system.solve(self._data, self.settings, self._res, self._result)  # getting an initial point of _result

        if self.settings.debug:
            print("Initial point after solving KKT system:", self._result)

        ## ----------- keep z and s non-negative --------------
        # this is according to the IV.A part of Roland Schwan 2023 paper
        delta_s = 0.0
        if self._data.m > 0:
            delta_s = max(delta_s, -self._result.s_l.min())
            delta_s = max(delta_s, -self._result.s_u.min())

        if self._data.num_xl > 0:
            delta_s = max(delta_s, -self._result.s_bl.min())
        if self._data.num_xu > 0:
            delta_s = max(delta_s, -self._result.s_bu.min())

        delta_z = 0.0
        if self._data.m > 0:
            delta_z = max(delta_z, -self._result.z_l.min())
            delta_z = max(delta_z, -self._result.z_u.min())
        
        if self._data.num_xl > 0:
            delta_z = max(delta_z, -self._result.z_bl.min())
        if self._data.num_xu > 0:
            delta_z = max(delta_z, -self._result.z_bu.min())

        self._result.s_l[self._data.idx_hl] += delta_s
        self._result.z_l[self._data.idx_hl] += delta_z
        self._result.s_u[self._data.idx_hu] += delta_s
        self._result.z_u[self._data.idx_hu] += delta_z

        self._result.s_bl[self._data.idx_xl] += delta_s
        self._result.z_bl[self._data.idx_xl] += delta_z
        self._result.s_bu[self._data.idx_xu] += delta_s
        self._result.z_bu[self._data.idx_xu] += delta_z

        self._result.info.mu = max(self._calculate_mu(), 1e-10)
        if self.settings.debug:
            print("Initial mu:", self._result.info.mu)

        # put s and z on the central path
        for idx in self._data.idx_hu:
            c = self._result.z_u[idx] - delta_z
            self._result.z_u[idx] = (c + np.sqrt(c * c + 4 * self._result.info.mu)) / 2
            self._result.s_u[idx] = self._result.z_u[idx] - c

        for idx in self._data.idx_hl:
            c = self._result.z_l[idx] - delta_z
            self._result.z_l[idx] = (c + np.sqrt(c * c + 4 * self._result.info.mu)) / 2
            self._result.s_l[idx] = self._result.z_l[idx] - c

        for idx in self._data.idx_xu:
            c = self._result.z_bu[idx] - delta_z
            self._result.z_bu[idx] = (c + np.sqrt(c * c + 4 * self._result.info.mu)) / 2
            self._result.s_bu[idx] = self._result.z_bu[idx] - c

        for idx in self._data.idx_xl:
            c = self._result.z_bl[idx] - delta_z
            self._result.z_bl[idx] = (c + np.sqrt(c * c + 4 * self._result.info.mu)) / 2
            self._result.s_bl[idx] = self._result.z_bl[idx] - c

        if self.settings.debug:
            print("self._result:", self._result)

        self._result.info.mu = self._calculate_mu()
        # print("Initial mu:", self._result.info.mu)


        self._prox_vars = Variables(self._data.n, self._data.p, self._data.m, self._data.num_xu, self._data.num_xl)
        self._prox_vars.x = self._result.x
        self._prox_vars.y = self._result.y
        self._prox_vars.z_l = self._result.z_l
        self._prox_vars.z_u = self._result.z_u

        print("Initial point set. Starting iterations...\n", self._prox_vars)

        ## ---------------------------------------------
        ## ---------- remaining iterations -------------
        ## ---------------------------------------------
        for iter in range(self.settings.max_iter):
            self._result.info.iter = iter

            if iter == 0:
                self._update_residuals_nr()
                self._result.info.prev_primal_res = self._result.info.primal_res
                self._result.info.prev_dual_res = self._result.info.dual_res

            # ? why not update both here?
            self._update_residuals_r()


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

            factor_success = self._kkt_system.update_scalings_and_factor(self._data, self._result.info.rho, self._result.info.delta, self._result)
            assert factor_success, "KKT matrix factorization failed."

            
            # ------------------ predictor step ------------------

            if self.settings.debug:
                print("before predictor step, result is: ", self._result)
                print("before predictor step, res is: ", self._res)

            # Short derivation:
            # Complementarity (elementwise): s_i * z_i = mu (usually written s * z = mu e).
            # Predictor (affine) aims for the affine step that drives complementarity to zero, so require (s + Δs) ∘ (z + Δz) = 0.
            # Expand: s ∘ z + S Δz + Z Δs + Δs ∘ Δz = 0, where S = diag(s), Z = diag(z).
            # Drop the quadratic term Δs ∘ Δz (first‑order Newton linearization) to get the linear system S Δz + Z Δs = - s ∘ z.
            # Thus the predictor RHS for the slack/dual complementarity equations is - s ∘ z (elementwise product), which is exactly what the four lines set for the different constraint groups.
            # In words: those lines build the complementarity residual r_s = - s .* z so the KKT solve computes Δs, Δz satisfying S Δz + Z Δs = r_s (the linearized complementarity equation) for the predictor (affine) direction. The .array() calls implement the elementwise product s .* z.
            self._res.s_l = -self._result.s_l * self._result.z_l
            self._res.s_u = -self._result.s_u * self._result.z_u
            self._res.s_bl = -self._result.s_bl * self._result.z_bl
            self._res.s_bu = -self._result.s_bu * self._result.z_bu
            
            

            if self.settings.debug:
                print("predictor step rhs is: res= ", self._res)

            self._kkt_system.solve(self._data, self.settings, self._res, self._step)

            if self.settings.debug:
                print("predictor step is:", self._step)

            # step in the non-negative orthant
            alpha_s, alpha_z = self._calculate_step()

            # avoid getting to close to the boundary
            alpha_s *= self.settings.tau
            alpha_z *= self.settings.tau
            # print("alpha_s, alpha_z:", alpha_s, alpha_z)

            # ------------------ compute centering parameter sigma ------------------
            self._result.info.sigma = 0.
            self._result.info.sigma += np.dot(self._result.s_l + alpha_s * self._step.s_l, self._result.z_l + alpha_z * self._step.z_l)
            self._result.info.sigma += np.dot(self._result.s_u + alpha_s * self._step.s_u, self._result.z_u + alpha_z * self._step.z_u)
            self._result.info.sigma += np.dot(self._result.s_bl + alpha_s * self._step.s_bl, self._result.z_bl + alpha_z * self._step.z_bl)
            self._result.info.sigma += np.dot(self._result.s_bu + alpha_s * self._step.s_bu, self._result.z_bu + alpha_z * self._step.z_bu)

            self._result.info.sigma /= self._result.info.mu * (self._data.m + self._data.m + self._data.num_xl + self._data.num_xu)
            self._result.info.sigma = max(0., min(1., self._result.info.sigma))
            self._result.info.sigma = self._result.info.sigma ** 3


            # ------------------ corrector step ------------------
            self._res.s_l += -self._step.s_l * self._step.z_l + self._result.info.sigma * self._result.info.mu
            self._res.s_u += -self._step.s_u * self._step.z_u + self._result.info.sigma * self._result.info.mu
            self._res.s_bl += -self._step.s_bl * self._step.z_bl + self._result.info.sigma * self._result.info.mu
            self._res.s_bu += -self._step.s_bu * self._step.z_bu + self._result.info.sigma * self._result.info.mu

            if self.settings.debug:
                print("corrector step rhs is: res= ", self._res)
            self._kkt_system.solve(self._data, self.settings, self._res, self._step)

            if self.settings.debug:
                print("corrector step is:", self._step)

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
            mu_rate: float = max(0., (mu_prev - self._result.info.mu) / mu_prev)  # r in Algorithm 2 in Roland Schwan 2023 paper


            # ------------------ update regularization ------------------
            self._update_residuals_nr()

            # TODO: more conditions to add in if clause
            if self._result.info.dual_res < 0.95 * self._result.info.prev_dual_res:
                self._prox_vars.x = self._result.x.copy()
                self._result.info.rho = max(self._result.info.reg_limit, (1. - mu_rate) * self._result.info.rho)
            else:
                self._result.info.rho = max(self._result.info.reg_limit, (1. - 0.666 * mu_rate) * self._result.info.rho)

            if self._result.info.primal_res < 0.95 * self._result.info.prev_primal_res:
                self._prox_vars.y = self._result.y.copy()
                self._prox_vars.z_l = self._result.z_l.copy()
                self._prox_vars.z_u = self._result.z_u.copy()
                self._prox_vars.z_bu = self._result.z_bu.copy()
                self._prox_vars.z_bl = self._result.z_bl.copy()
                
                self._result.info.delta = max(self._result.info.reg_limit, (1. - mu_rate) * self._result.info.delta)
            else:
                self._result.info.delta = max(self._result.info.reg_limit, (1. - 0.666 * mu_rate) * self._result.info.delta)


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

        if self._data.num_xl > 0:
            # for s_bl and z_bl (bound constraint slacks and duals - lower)
            mask_s_bl = self._step.s_bl < 0
            mask_z_bl = self._step.z_bl < 0
            
            if np.any(mask_s_bl):
                alpha_s = min(alpha_s, np.min(-self._result.s_bl[mask_s_bl] / self._step.s_bl[mask_s_bl]))
            if np.any(mask_z_bl):
                alpha_z = min(alpha_z, np.min(-self._result.z_bl[mask_z_bl] / self._step.z_bl[mask_z_bl]))

        if self._data.num_xu > 0:
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
                / (self._data.m + self._data.m + self._data.num_xl + self._data.num_xu)
        return float(mu)


    def _update_residuals_nr(self):
        """
        Compute the non-regularized primal and dual residuals:
        primal_residual = ||[A*x-b; G*x-h+s]||_inf
        dual_residual = ||P*x + c + AT*y + GT*z||_inf

        res_nr.x = -(P*x + c + A^T*y + G^T*(z_u - z_l) + z_bu - z_bl)
        res_nr.y = -(A*x - b)
        res_nr.z_l = -(G*x + s_l - hl)
        res_nr.z_u = -(-G*x + s_u + hu)

        also updates the primal and dual objectives and duality gap in self._result.info
        """
        # we calculate these term here first to be able to reuse temporary vectors
        minus_P_x = self._kkt_system.eval_P_x(self._data, -1., self._result.x)
        # res_nr.y = -A * x
        # work_x = A^T * y
        minus_A_x, AT_y = self._kkt_system.eval_A_xn_and_AT_xt(self._data, -1., 1., self._result.x, self._result.y)
        # res_nr.z_u = -G * x
        # res_nr.z_l = G * x
        # work_x += G^T * (z_u - z_l)
        # work_z.noalias() = m_result.z_u - m_result.z_l

        G_x, GT_zu_minus_zl = self._kkt_system.eval_G_xn_and_GT_xt(self._data, 1., 1., self._result.x, self._result.z_u - self._result.z_l)

        # ------------ update primal / dual objectives and duality gap ------------
        self._result.info.primal_obj = -0.5 * np.dot(minus_P_x, self._result.x) + np.dot(self._data.c, self._result.x)
        # dual objective is: 0.5 x^T P x - b^T y - h_u^T z_u + h_l^T z_l - x_u^T z_bu + x_l^T z_bl
        self._result.info.dual_obj = 0.5 * np.dot(minus_P_x, self._result.x)
        self._result.info.dual_obj += - np.dot(self._data.b, self._result.y)
        self._result.info.dual_obj += np.dot(self._data.h_l, self._result.z_l)
        self._result.info.dual_obj += - np.dot(self._data.h_u, self._result.z_u)
        self._result.info.dual_obj += np.dot(self._data.x_l, self._result.z_bl)
        self._result.info.dual_obj += - np.dot(self._data.x_u, self._result.z_bu)
        

        self._result.info.duality_gap = abs(self._result.info.primal_obj - self._result.info.dual_obj)
        self._result.info.duality_gap_rel = ... # TODO

        # duality_gap_rel = duality_gap / max(1, duality_gap_rel_norm)
        # where duality_gap_rel_norm is a scale estimate computed from the unscaled absolute contributions to the cost (e.g. |x^T P x|, |c^T x|, |b^T y|, |h_l^T z_l|, |h_u^T z_u|, |x_l^T z_bl|, |x_u^T z_bu|), each passed through the preconditioner unscale_cost.

        # res_nr.x = -(P*x + c + A^T*y + G^T*(z_u - z_l) + z_bu - z_bl)
        self._res_nr.x = minus_P_x - self._data.c - AT_y - GT_zu_minus_zl - (self._result.z_bu - self._result.z_bl)
        # res_nr.y = -(A*x - b)
        self._res_nr.y = minus_A_x + self._data.b
        # TODO: need to consider which index contains the constraints
        self._res_nr.z_l = G_x - self._result.s_l - self._data.h_l
        self._res_nr.z_u = -G_x - self._result.s_u + self._data.h_u
        self._res_nr.z_bl = self._result.x - self._result.s_bl - self._data.x_l
        self._res_nr.z_bu = - (self._result.x + self._result.s_bu - self._data.x_u)


        # ------------ update primal and dual residuals ------------
        self._result.info.prev_primal_res = self._result.info.primal_res
        self._result.info.prev_dual_res = self._result.info.dual_res

        self._result.info.primal_res = self._primal_res_nr()


        primal_rel_norm = 0.
        primal_rel_norm = max(primal_rel_norm, np.linalg.norm(minus_A_x, ord=np.inf))
        primal_rel_norm = max(primal_rel_norm, np.linalg.norm(self._data.b, ord=np.inf))
        primal_rel_norm = max(primal_rel_norm, np.linalg.norm(G_x, ord=np.inf))
        primal_rel_norm = max(primal_rel_norm, np.linalg.norm(self._data.h_u, ord=np.inf))
        primal_rel_norm = max(primal_rel_norm, np.linalg.norm(self._data.h_l, ord=np.inf))
        primal_rel_norm = max(primal_rel_norm, np.linalg.norm(self._data.x_u, ord=np.inf))
        primal_rel_norm = max(primal_rel_norm, np.linalg.norm(self._data.x_l, ord=np.inf))
        primal_rel_norm = max(primal_rel_norm, np.linalg.norm(self._result.s_u, ord=np.inf))
        primal_rel_norm = max(primal_rel_norm, np.linalg.norm(self._result.s_l, ord=np.inf))
        primal_rel_norm = max(primal_rel_norm, np.linalg.norm(self._result.s_bu, ord=np.inf))
        primal_rel_norm = max(primal_rel_norm, np.linalg.norm(self._result.s_bl, ord=np.inf))
        self._result.info.primal_res_rel = self._result.info.primal_res / max(1., primal_rel_norm)

        self._result.info.dual_res = np.linalg.norm(self._res_nr.x, ord=np.inf)
        dual_res_norm = np.linalg.norm(minus_P_x, ord=np.inf)
        dual_res_norm = max(dual_res_norm, np.linalg.norm(self._data.c, ord=np.inf))
        dual_res_norm = max(dual_res_norm, np.linalg.norm(AT_y, ord=np.inf))
        dual_res_norm = max(dual_res_norm, np.linalg.norm(GT_zu_minus_zl, ord=np.inf))
        dual_res_norm = max(dual_res_norm, np.linalg.norm(self._result.z_bu - self._result.z_bl, ord=np.inf))
        self._result.info.dual_res_rel = self._result.info.dual_res / max(1., dual_res_norm)
        

    def _update_residuals_r(self):
        """
        Compute the regularized primal and dual residuals. The computation is based on the non-regularized residuals computed in _update_residuals_nr.
        It adds the regularization terms to the non-regularized residuals to obtain the regularized residuals.
        """
        # update the rhs of the KKT system
        # r_x = -(P*x + c + A^T*y + G^T*(z_u - z_l)) - rho * (x - x_prox)
        self._res.x = self._res_nr.x - self._result.info.rho * (self._result.x - self._prox_vars.x)
        # r_y = -(A*x - b - delta*(y - lamda))  # TODO: I understand the formula, but why is it computed this way? I copied the following line from the original C++ code.
        self._res.y = self._res_nr.y - self._result.info.delta * (self._prox_vars.y - self._result.y)
        self._res.z_l = self._res_nr.z_l - self._result.info.delta * (self._prox_vars.z_l - self._result.z_l)
        self._res.z_u = self._res_nr.z_u - self._result.info.delta * (self._prox_vars.z_u - self._result.z_u)
        self._res.z_bl[self._data.idx_xl] = self._res_nr.z_bl[self._data.idx_xl] - self._result.info.delta * (self._prox_vars.z_bl[self._data.idx_xl] - self._result.z_bl[self._data.idx_xl])
        self._res.z_bu[self._data.idx_xu] = self._res_nr.z_bu[self._data.idx_xu] - self._result.info.delta * (self._prox_vars.z_bu[self._data.idx_xu] - self._result.z_bu[self._data.idx_xu])
        
        primal_rel_scaling = self._result.info.primal_res / self._result.info.primal_res_rel if self._result.info.primal_res_rel > 0 else 1.
        dual_rel_scaling = self._result.info.dual_res / self._result.info.dual_res_rel if self._result.info.dual_res_rel > 0 else 1.

        self._result.info.primal_res_reg = self._primal_res_r()
        self._result.info.primal_res_reg_rel = self._result.info.primal_res_reg / primal_rel_scaling
        self._result.info.dual_res_reg = self._dual_res_r()
        self._result.info.dual_res_reg_rel = self._result.info.dual_res_reg / dual_rel_scaling

        self._result.info.primal_prox_inf = self._primal_prox_inf() * self._result.info.delta
        self._result.info.dual_prox_inf = self._dual_prox_inf() * self._result.info.rho

    def _primal_res_nr(self) -> float:
        inf = 0.
        inf = max(inf, np.max(np.abs(self._res_nr.y)))
        inf = max(inf, np.max(np.abs(self._res_nr.z_u)))
        inf = max(inf, np.max(np.abs(self._res_nr.z_l)))
        # inf = max(inf, np.max(np.abs(self._res_nr.z_bu)))
        # inf = max(inf, np.max(np.abs(self._res_nr.z_bl)))
        # ! I don't understand here. Why it is not taking the abs value of z_bl and z_bu?
        # ! This is just copied from the cpp implementation.
        inf = max(inf, np.max(self._res_nr.z_bu))
        inf = max(inf, np.max(self._res_nr.z_bl))
        return inf


    def _primal_res_r(self) -> float:
        # TODO: fix this according to the cpp implementation
        # ! primal_res_r and primal_res_nr are doing the same stuff, but using self._res and self._res_nr respectively
        # primal_res_r = np.max(np.abs(np.concatenate([
        #     self._res.y,
        #     self._res.z_u + self._res.s_u - self._res.z_l - self._res.s_l
        # ])))
        # return primal_res_r
        inf = 0.
        inf = max(inf, np.max(np.abs(self._res.y)))
        inf = max(inf, np.max(np.abs(self._res.z_u)))
        inf = max(inf, np.max(np.abs(self._res.z_l)))
        inf = max(inf, np.max(np.abs(self._res.z_bu)))
        inf = max(inf, np.max(np.abs(self._res.z_bl)))
        return inf
    
    def _dual_res_nr(self) -> float:
        return self._res_nr.x
    
    def _dual_res_r(self) -> float:
        return self._res.x
    
    def _primal_prox_inf(self) -> float:
        inf = 0.
        inf = max(inf, np.linalg.norm(self._result.y - self._prox_vars.y, ord=np.inf))
        inf = max(inf, np.linalg.norm(self._result.z_l - self._prox_vars.z_l, ord=np.inf))
        inf = max(inf, np.linalg.norm(self._result.z_u - self._prox_vars.z_u, ord=np.inf))
        inf = max(inf, np.linalg.norm(self._result.z_bl - self._prox_vars.z_bl, ord=np.inf))
        inf = max(inf, np.linalg.norm(self._result.z_bu - self._prox_vars.z_bu, ord=np.inf))
        return inf
    
    def _dual_prox_inf(self) -> float:
        return np.linalg.norm(self._result.x - self._prox_vars.x, ord=np.inf)
    

# class DenseSolver(SolverBase):
#     def __init__(self):
#         super().__init__()
#         self._kkt_system = De

#     def _solve_impl(self):
#         # Implement the dense solver logic here
#         self._result.status = Status.PIQP_SOLVED
#         # Fill in other result fields as necessary