import jax.numpy as jnp
from typing import Tuple

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
        self._step = Variables(self._data.n, self._data.p, self._data.m)

        self._kkt_system = KKTSystem(self._data)

        self._res_nr = Variables(self._data.n, self._data.p, self._data.m)  # used to store the non-regularized residuals
        self._res_r = Variables(self._data.n, self._data.p, self._data.m)  # used to store the regularized residuals
        
        

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

        self._result.x = jnp.zeros(self._data.n)
        self._result.y = jnp.zeros(self._data.p)
        # ! using jnp.nan because it raises error when the implementation is wrong, making debugging easier
        self._result.s_l = jnp.nan * jnp.ones(self._data.m)
        self._result.s_u = jnp.nan * jnp.ones(self._data.m)
        self._result.s_bl = jnp.nan * jnp.ones(self._data.n)
        self._result.s_bu = jnp.nan * jnp.ones(self._data.n)
        self._result.z_l = jnp.nan * jnp.ones(self._data.m)
        self._result.z_u = jnp.nan * jnp.ones(self._data.m)
        self._result.z_bl = jnp.nan * jnp.ones(self._data.n)
        self._result.z_bu = jnp.nan * jnp.ones(self._data.n)

        self._result.s_l = self._result.s_l.at[self._data.idx_hl].set(1.0)
        self._result.z_l = self._result.z_l.at[self._data.idx_hl].set(1.0)
        self._result.s_u = self._result.s_u.at[self._data.idx_hu].set(1.0)
        self._result.z_u = self._result.z_u.at[self._data.idx_hu].set(1.0)
        self._result.s_bl = self._result.s_bl.at[self._data.idx_xl].set(1.0)
        self._result.z_bl = self._result.z_bl.at[self._data.idx_xl].set(1.0)
        self._result.s_bu = self._result.s_bu.at[self._data.idx_xu].set(1.0)
        self._result.z_bu = self._result.z_bu.at[self._data.idx_xu].set(1.0)

        self._kkt_system.update_scalings_and_factor(
            self._data,
            self._result.info.rho,
            self._result.info.delta,
            self._result
        )

        self._res = Result()  # used to store the right hand side of KKT system
        self._res.x = -self._data.c.copy()
        self._res.y = self._data.b.copy()
        self._res.z_l = jnp.nan * jnp.zeros(self._data.m)
        self._res.z_l = self._res.z_l.at[self._data.idx_hl].set(-self._data.h_l[self._data.idx_hl])
        self._res.z_u = jnp.nan * jnp.zeros(self._data.m)
        self._res.z_u = self._res.z_u.at[self._data.idx_hu].set(self._data.h_u[self._data.idx_hu])
        self._res.z_bl = jnp.nan * jnp.zeros(self._data.n)
        self._res.z_bl = self._res.z_bl.at[self._data.idx_xl].set(-self._data.x_l[self._data.idx_xl])
        self._res.z_bu = jnp.nan * jnp.zeros(self._data.n)
        self._res.z_bu = self._res.z_bu.at[self._data.idx_xu].set(self._data.x_u[self._data.idx_xu])
        
        self._res.s_l = jnp.nan * jnp.zeros(self._data.m)
        self._res.s_u = jnp.nan * jnp.zeros(self._data.m)
        self._res.s_bl = jnp.nan * jnp.zeros(self._data.n)
        self._res.s_bu = jnp.nan * jnp.zeros(self._data.n)
        self._res.s_l = self._res.s_l.at[self._data.idx_hl].set(0.)
        self._res.s_u = self._res.s_u.at[self._data.idx_hu].set(0.)
        self._res.s_bl = self._res.s_bl.at[self._data.idx_xl].set(0.)
        self._res.s_bu = self._res.s_bu.at[self._data.idx_xu].set(0.)

        self._kkt_system.solve(self._data, self.settings, self._res, self._result)  # getting an initial point of _result

        if self.settings.debug:
            print("Initial point after solving KKT system:", self._result)

        ## ----------- keep z and s non-negative --------------
        # this is according to the IV.A part of Roland Schwan 2023 paper
        delta_s = 0.0
        if self._data.num_hl > 0:
            delta_s = jnp.maximum(delta_s, -self._result.s_l[self._data.idx_hl].min())
        if self._data.num_hu > 0:
            delta_s = jnp.maximum(delta_s, -self._result.s_u[self._data.idx_hu].min())

        if self._data.num_xl > 0:
            delta_s = jnp.maximum(delta_s, -self._result.s_bl[self._data.idx_xl].min())
        if self._data.num_xu > 0:
            delta_s = jnp.maximum(delta_s, -self._result.s_bu[self._data.idx_xu].min())

        delta_z = 0.0
        if self._data.num_hl > 0:
            delta_z = jnp.maximum(delta_z, -self._result.z_l[self._data.idx_hl].min())
        if self._data.num_hu > 0:
            delta_z = jnp.maximum(delta_z, -self._result.z_u[self._data.idx_hu].min())
        
        if self._data.num_xl > 0:
            delta_z = jnp.maximum(delta_z, -self._result.z_bl[self._data.idx_xl].min())
        if self._data.num_xu > 0:
            delta_z = jnp.maximum(delta_z, -self._result.z_bu[self._data.idx_xu].min())

        self._result.s_l = self._result.s_l.at[self._data.idx_hl].add(delta_s)
        self._result.z_l = self._result.z_l.at[self._data.idx_hl].add(delta_z)
        self._result.s_u = self._result.s_u.at[self._data.idx_hu].add(delta_s)
        self._result.z_u = self._result.z_u.at[self._data.idx_hu].add(delta_z)

        self._result.s_bl = self._result.s_bl.at[self._data.idx_xl].add(delta_s)
        self._result.z_bl = self._result.z_bl.at[self._data.idx_xl].add(delta_z)
        self._result.s_bu = self._result.s_bu.at[self._data.idx_xu].add(delta_s)
        self._result.z_bu = self._result.z_bu.at[self._data.idx_xu].add(delta_z)

        self._result.info.mu = jnp.maximum(self._calculate_mu(), 1e-10)
        if self.settings.debug:
            print("Initial mu:", self._result.info.mu)

        # put s and z on the central path
        for idx in self._data.idx_hu:
            c = self._result.z_u[idx] - delta_z
            self._result.z_u = self._result.z_u.at[idx].set((c + jnp.sqrt(c * c + 4 * self._result.info.mu)) / 2)
            self._result.s_u = self._result.s_u.at[idx].set(self._result.z_u[idx] - c)

        for idx in self._data.idx_hl:
            c = self._result.z_l[idx] - delta_z
            self._result.z_l = self._result.z_l.at[idx].set((c + jnp.sqrt(c * c + 4 * self._result.info.mu)) / 2)
            self._result.s_l = self._result.s_l.at[idx].set(self._result.z_l[idx] - c)

        for idx in self._data.idx_xu:
            c = self._result.z_bu[idx] - delta_z
            self._result.z_bu = self._result.z_bu.at[idx].set((c + jnp.sqrt(c * c + 4 * self._result.info.mu)) / 2)
            self._result.s_bu = self._result.s_bu.at[idx].set(self._result.z_bu[idx] - c)

        for idx in self._data.idx_xl:
            c = self._result.z_bl[idx] - delta_z
            self._result.z_bl = self._result.z_bl.at[idx].set((c + jnp.sqrt(c * c + 4 * self._result.info.mu)) / 2)
            self._result.s_bl = self._result.s_bl.at[idx].set(self._result.z_bl[idx] - c)

        if self.settings.debug:
            print("self._result:", self._result)

        self._result.info.mu = self._calculate_mu()

        self._prox_vars = Variables(self._data.n, self._data.p, self._data.m)
        self._prox_vars.x = self._result.x
        self._prox_vars.y = self._result.y
        self._prox_vars.z_l = self._result.z_l
        self._prox_vars.z_u = self._result.z_u

        if self.settings.debug:
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

            # ------------------ compute centering parameter sigma ------------------
            self._result.info.sigma = 0.
            self._result.info.sigma += jnp.dot(self._result.s_l + alpha_s * self._step.s_l, self._result.z_l + alpha_z * self._step.z_l)
            self._result.info.sigma += jnp.dot(self._result.s_u + alpha_s * self._step.s_u, self._result.z_u + alpha_z * self._step.z_u)
            self._result.info.sigma += jnp.dot(self._result.s_bl[self._data.idx_xl] + alpha_s * self._step.s_bl[self._data.idx_xl], self._result.z_bl[self._data.idx_xl] + alpha_z * self._step.z_bl[self._data.idx_xl])
            self._result.info.sigma += jnp.dot(self._result.s_bu[self._data.idx_xu] + alpha_s * self._step.s_bu[self._data.idx_xu], self._result.z_bu[self._data.idx_xu] + alpha_z * self._step.z_bu[self._data.idx_xu])

            self._result.info.sigma /= self._result.info.mu * (self._data.m + self._data.m + self._data.num_xl + self._data.num_xu)
            self._result.info.sigma = jnp.clip(self._result.info.sigma, 0., 1.)
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
            mu_rate: float = jnp.maximum(0., (mu_prev - self._result.info.mu) / mu_prev)  # r in Algorithm 2 in Roland Schwan 2023 paper


            # ------------------ update regularization ------------------
            self._update_residuals_nr()

            # TODO: more conditions to add in if clause
            if self._result.info.dual_res < 0.95 * self._result.info.prev_dual_res:
                self._prox_vars.x = self._result.x.copy()
                self._result.info.rho = jnp.maximum(self._result.info.reg_limit, (1. - mu_rate) * self._result.info.rho)
            else:
                self._result.info.rho = jnp.maximum(self._result.info.reg_limit, (1. - 0.666 * mu_rate) * self._result.info.rho)

            if self._result.info.primal_res < 0.95 * self._result.info.prev_primal_res:
                self._prox_vars.y = self._result.y.copy()
                self._prox_vars.z_l = self._result.z_l.copy()
                self._prox_vars.z_u = self._result.z_u.copy()
                self._prox_vars.z_bu = self._result.z_bu.copy()
                self._prox_vars.z_bl = self._result.z_bl.copy()
                
                self._result.info.delta = jnp.maximum(self._result.info.reg_limit, (1. - mu_rate) * self._result.info.delta)
            else:
                self._result.info.delta = jnp.maximum(self._result.info.reg_limit, (1. - 0.666 * mu_rate) * self._result.info.delta)


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
            
            if jnp.any(mask_s_l):
                alpha_s = jnp.minimum(alpha_s, jnp.min(-self._result.s_l[mask_s_l] / self._step.s_l[mask_s_l]))
            if jnp.any(mask_s_u):
                alpha_s = jnp.minimum(alpha_s, jnp.min(-self._result.s_u[mask_s_u] / self._step.s_u[mask_s_u]))
            
            # for z_l and z_u (dual variables)
            mask_z_l = self._step.z_l < 0
            mask_z_u = self._step.z_u < 0
            
            if jnp.any(mask_z_l):
                alpha_z = jnp.minimum(alpha_z, jnp.min(-self._result.z_l[mask_z_l] / self._step.z_l[mask_z_l]))
            if jnp.any(mask_z_u):
                alpha_z = jnp.minimum(alpha_z, jnp.min(-self._result.z_u[mask_z_u] / self._step.z_u[mask_z_u]))

        if self._data.num_xl > 0:
            # for s_bl and z_bl (bound constraint slacks and duals - lower)
            mask_s_bl = self._step.s_bl < 0
            mask_z_bl = self._step.z_bl < 0
            
            if jnp.any(mask_s_bl):
                alpha_s = jnp.minimum(alpha_s, jnp.min(-self._result.s_bl[mask_s_bl] / self._step.s_bl[mask_s_bl]))
            if jnp.any(mask_z_bl):
                alpha_z = jnp.minimum(alpha_z, jnp.min(-self._result.z_bl[mask_z_bl] / self._step.z_bl[mask_z_bl]))

        if self._data.num_xu > 0:
            # for s_bu and z_bu (bound constraint slacks and duals - upper)
            mask_s_bu = self._step.s_bu < 0
            mask_z_bu = self._step.z_bu < 0
            
            if jnp.any(mask_s_bu):
                alpha_s = jnp.minimum(alpha_s, jnp.min(-self._result.s_bu[mask_s_bu] / self._step.s_bu[mask_s_bu]))
            if jnp.any(mask_z_bu):
                alpha_z = jnp.minimum(alpha_z, jnp.min(-self._result.z_bu[mask_z_bu] / self._step.z_bu[mask_z_bu]))
        
        return alpha_s, alpha_z
    

    def _calculate_mu(self) -> float:
        mu = (self._result.s_l[self._data.idx_hl].dot(self._result.z_l[self._data.idx_hl])
                + self._result.s_u[self._data.idx_hu].dot(self._result.z_u[self._data.idx_hu]) \
                + self._result.s_bl[self._data.idx_xl].dot(self._result.z_bl[self._data.idx_xl]) \
                + self._result.s_bu[self._data.idx_xu].dot(self._result.z_bu[self._data.idx_xu])) \
                / (self._data.num_hl + self._data.num_hu + self._data.num_xl + self._data.num_xu)
        return mu


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

        # TODO: Need to reconsider this if idx_hu and idx_hl are not full
        tmp = jnp.zeros(self._data.m)
        tmp = tmp[self._data.idx_hu] + self._result.z_u[self._data.idx_hu]
        tmp = tmp[self._data.idx_hl] - self._result.z_l[self._data.idx_hl]
        G_x, GT_zu_minus_zl = self._kkt_system.eval_G_xn_and_GT_xt(self._data, 1., 1., self._result.x, tmp)

        # ------------ update primal / dual objectives and duality gap ------------
        self._result.info.primal_obj = -0.5 * jnp.dot(minus_P_x, self._result.x) + jnp.dot(self._data.c, self._result.x)
        # dual objective is: 0.5 x^T P x - b^T y - h_u^T z_u + h_l^T z_l - x_u^T z_bu + x_l^T z_bl
        self._result.info.dual_obj = 0.5 * jnp.dot(minus_P_x, self._result.x)
        self._result.info.dual_obj += - jnp.dot(self._data.b, self._result.y)
        self._result.info.dual_obj += jnp.dot(self._data.h_l[self._data.idx_hl], self._result.z_l[self._data.idx_hl])
        self._result.info.dual_obj += - jnp.dot(self._data.h_u[self._data.idx_hu], self._result.z_u[self._data.idx_hu])
        self._result.info.dual_obj += jnp.dot(self._data.x_l[self._data.idx_xl], self._result.z_bl[self._data.idx_xl])
        self._result.info.dual_obj += - jnp.dot(self._data.x_u[self._data.idx_xu], self._result.z_bu[self._data.idx_xu])
        

        self._result.info.duality_gap = jnp.abs(self._result.info.primal_obj - self._result.info.dual_obj)
        self._result.info.duality_gap_rel = ... # TODO

        # duality_gap_rel = duality_gap / max(1, duality_gap_rel_norm)
        # where duality_gap_rel_norm is a scale estimate computed from the unscaled absolute contributions to the cost (e.g. |x^T P x|, |c^T x|, |b^T y|, |h_l^T z_l|, |h_u^T z_u|, |x_l^T z_bl|, |x_u^T z_bu|), each passed through the preconditioner unscale_cost.

        # res_nr.x = -(P*x + c + A^T*y + G^T*(z_u - z_l) + z_bu - z_bl)
        # TODO: Need to reconsider this if idx_hu and idx_hl are not full
        self._res_nr.x = minus_P_x - self._data.c - AT_y - GT_zu_minus_zl
        self._res_nr.x = self._res_nr.x.at[self._data.idx_xl].add(self._result.z_bl[self._data.idx_xl])
        self._res_nr.x = self._res_nr.x.at[self._data.idx_xu].add(-self._result.z_bu[self._data.idx_xu])
        # res_nr.y = -(A*x - b)
        self._res_nr.y = minus_A_x + self._data.b
        # TODO: need to consider which index contains the constraints
        self._res_nr.z_l = self._res_nr.z_l.at[self._data.idx_hl].set((G_x - self._result.s_l - self._data.h_l)[self._data.idx_hl])
        self._res_nr.z_u = self._res_nr.z_u.at[self._data.idx_hu].set((-G_x - self._result.s_u + self._data.h_u)[self._data.idx_hu])
        self._res_nr.z_bl = self._res_nr.z_bl.at[self._data.idx_xl].set((self._result.x - self._result.s_bl - self._data.x_l)[self._data.idx_xl])
        self._res_nr.z_bu = self._res_nr.z_bu.at[self._data.idx_xu].set(- (self._result.x + self._result.s_bu - self._data.x_u)[self._data.idx_xu])

        # ------------ update primal and dual residuals ------------
        self._result.info.prev_primal_res = self._result.info.primal_res
        self._result.info.prev_dual_res = self._result.info.dual_res

        self._result.info.primal_res = self._primal_res_nr()


        primal_rel_norm = jnp.linalg.norm(minus_A_x, ord=jnp.inf)
        primal_rel_norm = jnp.maximum(primal_rel_norm, jnp.linalg.norm(self._data.b, ord=jnp.inf))
        primal_rel_norm = jnp.maximum(primal_rel_norm, jnp.linalg.norm(G_x, ord=jnp.inf))
        primal_rel_norm = jnp.maximum(primal_rel_norm, jnp.linalg.norm(self._data.h_u, ord=jnp.inf))
        primal_rel_norm = jnp.maximum(primal_rel_norm, jnp.linalg.norm(self._data.h_l, ord=jnp.inf))
        primal_rel_norm = jnp.maximum(primal_rel_norm, jnp.linalg.norm(self._data.x_u, ord=jnp.inf))
        primal_rel_norm = jnp.maximum(primal_rel_norm, jnp.linalg.norm(self._data.x_l, ord=jnp.inf))
        primal_rel_norm = jnp.maximum(primal_rel_norm, jnp.linalg.norm(self._result.s_u, ord=jnp.inf))
        primal_rel_norm = jnp.maximum(primal_rel_norm, jnp.linalg.norm(self._result.s_l, ord=jnp.inf))
        primal_rel_norm = jnp.maximum(primal_rel_norm, jnp.linalg.norm(self._result.s_bu, ord=jnp.inf))
        primal_rel_norm = jnp.maximum(primal_rel_norm, jnp.linalg.norm(self._result.s_bl, ord=jnp.inf))
        self._result.info.primal_res_rel = self._result.info.primal_res / jnp.maximum(1., primal_rel_norm)

        self._result.info.dual_res = jnp.linalg.norm(self._res_nr.x, ord=jnp.inf)
        dual_res_norm = jnp.linalg.norm(minus_P_x, ord=jnp.inf)
        dual_res_norm = jnp.maximum(dual_res_norm, jnp.linalg.norm(self._data.c, ord=jnp.inf))
        dual_res_norm = jnp.maximum(dual_res_norm, jnp.linalg.norm(AT_y, ord=jnp.inf))
        dual_res_norm = jnp.maximum(dual_res_norm, jnp.linalg.norm(GT_zu_minus_zl, ord=jnp.inf))
        # TODO: Need to reconsider this if idx_hu and idx_hl are not full
        dual_res_norm = jnp.maximum(dual_res_norm, jnp.linalg.norm(self._result.z_bu - self._result.z_bl, ord=jnp.inf))
        self._result.info.dual_res_rel = self._result.info.dual_res / jnp.maximum               (1., dual_res_norm)
        

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
        self._res.z_bl = self._res.z_bl.at[self._data.idx_xl].set(self._res_nr.z_bl[self._data.idx_xl] - self._result.info.delta * (self._prox_vars.z_bl[self._data.idx_xl] - self._result.z_bl[self._data.idx_xl]))
        self._res.z_bu = self._res.z_bu.at[self._data.idx_xu].set(self._res_nr.z_bu[self._data.idx_xu] - self._result.info.delta * (self._prox_vars.z_bu[self._data.idx_xu] - self._result.z_bu[self._data.idx_xu]))
        
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
        inf = jnp.maximum(inf, jnp.linalg.norm(self._res_nr.y, ord=jnp.inf))
        inf = jnp.maximum(inf, jnp.linalg.norm(self._res_nr.z_u[self._data.idx_hu], ord=jnp.inf))
        inf = jnp.maximum(inf, jnp.linalg.norm(self._res_nr.z_l[self._data.idx_hl], ord=jnp.inf))
        # ! I don't understand here. Why it is not taking the abs value of z_bl and z_bu?
        # ! This is just copied from the cpp implementation.
        inf = jnp.maximum(inf, jnp.linalg.norm(self._res_nr.z_bu[self._data.idx_xu], ord=jnp.inf))
        inf = jnp.maximum(inf, jnp.linalg.norm(self._res_nr.z_bl[self._data.idx_xl], ord=jnp.inf))
        return inf


    def _primal_res_r(self) -> float:
        inf = 0.
        inf = jnp.maximum(inf, jnp.linalg.norm(self._res.y, ord=jnp.inf))
        inf = jnp.maximum(inf, jnp.linalg.norm(self._res.z_u[self._data.idx_hu], ord=jnp.inf))
        inf = jnp.maximum(inf, jnp.linalg.norm(self._res.z_l[self._data.idx_hl], ord=jnp.inf))
        inf = jnp.maximum(inf, jnp.linalg.norm(self._res.z_bu[self._data.idx_xu], ord=jnp.inf))
        inf = jnp.maximum(inf, jnp.linalg.norm(self._res.z_bl[self._data.idx_xl], ord=jnp.inf))
        return inf
    
    def _dual_res_nr(self) -> float:
        return jnp.linalg.norm(self._res_nr.x, ord=jnp.inf)
    
    def _dual_res_r(self) -> float:
        return jnp.linalg.norm(self._res.x, ord=jnp.inf)
    
    def _primal_prox_inf(self) -> float:
        inf = 0.
        inf = jnp.maximum(inf, jnp.linalg.norm(self._result.y - self._prox_vars.y, ord=jnp.inf))
        inf = jnp.maximum(inf, jnp.linalg.norm(self._result.z_l[self._data.idx_hl] - self._prox_vars.z_l[self._data.idx_hl], ord=jnp.inf))
        inf = jnp.maximum(inf, jnp.linalg.norm(self._result.z_u[self._data.idx_hu] - self._prox_vars.z_u[self._data.idx_hu], ord=jnp.inf))
        inf = jnp.maximum(inf, jnp.linalg.norm(self._result.z_bl[self._data.idx_xl] - self._prox_vars.z_bl[self._data.idx_xl], ord=jnp.inf))
        inf = jnp.maximum(inf, jnp.linalg.norm(self._result.z_bu[self._data.idx_xu] - self._prox_vars.z_bu[self._data.idx_xu], ord=jnp.inf))
        return inf
    
    def _dual_prox_inf(self) -> float:
        return jnp.linalg.norm(self._result.x - self._prox_vars.x, ord=jnp.inf)
    