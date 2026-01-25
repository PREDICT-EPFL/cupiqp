import cupy as cp
from typing import Tuple
import nvtx

from .settings import Settings
from .data import Data
from .results import Result, Status, Variables
from .kkt_systems import KKTSystem

class SolverBase:
    def __init__(self):

        self.settings = Settings()
        self._data: Data = None
        self._result = Result()    # store the values of primal, dual and slack variables of current iteration, and other information
        self._step = Variables()   # used to store the step direction of primal and dual variables
        self._res_nr = Variables()  # used to store the non-regularized residuals
        self._res = Variables()  # used to store the regularized residuals
        self._prox_vars = Variables()  # used to store the proximal variables

        self._kkt_system = None
    
    @nvtx.annotate("Solver::setup")
    def setup(self, P, c, A, b, G, h_u, h_l, x_u, x_l):
        self._data = Data(P, c, A, b, G, h_u, h_l, x_u, x_l)
        self._result.init(self._data)
        self._result.info.rho = self.settings.rho_init
        self._result.info.delta = self.settings.delta_init
        self._result.init(self._data)
        
        self._step.init(self._data)
        self._res_nr.init(self._data)
        self._res.init(self._data)
        self._prox_vars.init(self._data)

        self._kkt_system = KKTSystem(self._data, self.settings)

        self._work_z_1 = cp.empty(self._data.m)  # used to store intermediate results in _update_residuals_nr
        self._work_z_2 = cp.empty(self._data.m)  # used to store intermediate results in _update_residuals_nr

        self._work_z = cp.empty(self._data.num_hl + self._data.num_hu + self._data.num_xl + self._data.num_xu)  # used in _calculate_step to hold all concatenated slack or dual steps / results
        self._work_s = cp.empty(self._data.num_hl + self._data.num_hu + self._data.num_xl + self._data.num_xu)  # used in _calculate_step to hold all concatenated slack or dual steps / results
        self._alpha_sz = cp.empty(2) # step lengths of slack and dual variables [alpha_s, alpha_z]
        
        

    def solve(self) -> Status:
        if self.settings.verbose:
            if self.settings.kkt_solver == "dense_cholesky":
                print("dense backend:")
                print(f"variables n = {self._data.n}")
                print(f"equality constraints p = {self._data.p}")
                print(f"inequality constraints m = {self._data.m}")
            else:
                print("sparse backend:")
                print(f"variables n = {self._data.n}, nnz(P) = {self._data.P.nnz}")
                print(f"equality constraints p = {self._data.p}, nnz(A) = {self._data.A.nnz}")
                print(f"inequality constraints m = {self._data.m}, nnz(G) = {self._data.G.nnz}")
            print(f"inequality lower bounds n_h_l = {self._data.num_hl}")
            print(f"inequality upper bounds n_h_u = {self._data.num_hu}")
            print(f"variable lower bounds n_x_l = {self._data.num_xl}")
            print(f"variable upper bounds n_x_u = {self._data.num_xu}")
            print("")
        return self._solve_impl()

    def _solve_impl(self) -> Status:
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
            print("iter  prim_obj       dual_obj       duality_gap   prim_res      dual_res      rho         delta       mu          p_step   d_step")  

        ## ----------- initial iteration --------------
        # eq(12) in Roland Schwan 2023 paper
        with nvtx.annotate("Solver::initialization"):
            self._result.x.fill(0.0)
            self._result.y.fill(0.0)

            self._result.s_l.fill(1.0)
            self._result.z_l.fill(1.0)
            self._result.s_u.fill(1.0)
            self._result.z_u.fill(1.0)
            self._result.s_bl.fill(1.0)
            self._result.z_bl.fill(1.0)
            self._result.s_bu.fill(1.0)
            self._result.z_bu.fill(1.0)

            self._kkt_system.update_scalings_and_factor(
                self._data,
                self._result.info.rho,
                self._result.info.delta,
                self._result
            )

            self._res.x[:] = self._data.c
            self._res.x *= -1.0
            self._res.y[:] = self._data.b
            cp.take(self._data.h_l, self._data.idx_hl, out=self._res.z_l)
            self._res.z_l *= -1.0
            cp.take(self._data.h_u, self._data.idx_hu, out=self._res.z_u)
            cp.take(self._data.x_l, self._data.idx_xl, out=self._res.z_bl)
            self._res.z_bl *= -1.0
            cp.take(self._data.x_u, self._data.idx_xu, out=self._res.z_bu)
            self._res.s_l[:] = 0.
            self._res.s_u[:] = 0.
            self._res.s_bl[:] = 0.
            self._res.s_bu[:] = 0.

            self._kkt_system.solve(self._data, self.settings, self._res, self._result)  # getting an initial point of _result

            if self.settings.debug:
                print("Initial point after solving KKT system:", self._result)

            ## ----------- keep z and s non-negative --------------
            # this is according to the IV.A part of Roland Schwan 2023 paper
            offset = 0
            self._work_s[offset:offset+self._data.num_hl] = self._result.s_l
            self._work_z[offset:offset+self._data.num_hl] = self._result.z_l
            offset += self._data.num_hl
            self._work_s[offset:offset+self._data.num_hu] = self._result.s_u
            self._work_z[offset:offset+self._data.num_hu] = self._result.z_u
            offset += self._data.num_hu
            self._work_s[offset:offset+self._data.num_xl] = self._result.s_bl
            self._work_z[offset:offset+self._data.num_xl] = self._result.z_bl
            offset += self._data.num_xl
            self._work_s[offset:offset+self._data.num_xu] = self._result.s_bu
            self._work_z[offset:offset+self._data.num_xu] = self._result.z_bu
            offset += self._data.num_xu
            delta_s = -cp.min(self._work_s[:offset])  # single D2H transfer
            delta_z = -cp.min(self._work_z[:offset])  # single D2H transfer

            self._result.s_l += delta_s
            self._result.z_l += delta_z
            self._result.s_u += delta_s
            self._result.z_u += delta_z

            self._result.s_bl += delta_s
            self._result.z_bl += delta_z
            self._result.s_bu += delta_s
            self._result.z_bu += delta_z

            # need to make sure mu is positive here, otherwise in the next step (put s and z on central path) sqrt(mu) the computed z_* will be zeros
            self._result.info.mu = cp.maximum(self._calculate_mu(), 1e-10)
            if self.settings.debug:
                print("Initial mu:", self._result.info.mu)

            # put s and z on the central path
            # Do the following: c = z* - delta-z; z = (c + sqrt(c^2 + 4*mu)) / 2; s = z - c
            for s, z in zip(
                [self._result.s_l, self._result.s_u, self._result.s_bl, self._result.s_bu], 
                [self._result.z_l, self._result.z_u, self._result.z_bl, self._result.z_bu]
                ):
                cp.subtract(z, delta_z, out=s)
                cp.power(s, 2, out=z)
                z += 4. * self._result.info.mu
                cp.sqrt(z, out=z)
                z += s
                z /= 2.
                cp.subtract(z, s, out=s)

            if self.settings.debug:
                print("self._result:", self._result)

            self._result.info.mu = self._calculate_mu()

            self._prox_vars.x[:] = self._result.x
            self._prox_vars.y[:] = self._result.y
            self._prox_vars.z_l[:] = self._result.z_l
            self._prox_vars.z_u[:] = self._result.z_u
            self._prox_vars.z_bl[:] = self._result.z_bl
            self._prox_vars.z_bu[:] = self._result.z_bu

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

            # ? The convergence criteria seems different from the one in the paper
            if ((self._result.info.primal_res < self.settings.eps_abs or self._result.info.primal_res_rel < self.settings.eps_rel) and
                (self._result.info.dual_res < self.settings.eps_abs or self._result.info.dual_res_rel < self.settings.eps_rel) and
                (not self.settings.check_duality_gap or self._result.info.duality_gap < self.settings.eps_duality_gap_abs or self._result.info.duality_gap_rel < self.settings.eps_duality_gap_rel)):
                self._result.info.status = Status.PIQP_SOLVED
                return self._result.info.status
            
            # ? why not update both here?
            self._update_residuals_r()

            if (self._result.info.no_dual_update > cp.minimum(5., self.settings.reg_finetune_dual_update_threshold) and
                self._result.info.primal_prox_inf > self.settings.infeasibility_threshold and
                (self._result.info.primal_res_reg < self.settings.eps_abs or self._result.info.primal_res_reg_rel < self.settings.eps_rel)):
                self._result.info.status = Status.PIQP_PRIMAL_INFEASIBLE
                return self._result.info.status
            
            if (self._result.info.no_primal_update > cp.minimum(5., self.settings.reg_finetune_primal_update_threshold) and
                self._result.info.dual_prox_inf > self.settings.infeasibility_threshold and
                (self._result.info.dual_res_reg < self.settings.eps_abs or self._result.info.dual_res_reg_rel < self.settings.eps_rel)):
                self._result.info.status = Status.PIQP_DUAL_INFEASIBLE
                return self._result.info.status
            

            if self.settings.verbose:
                print(
                    f"{self._result.info.iter:3d}   "
                    f"{float(self._result.info.primal_obj): .5e}   "
                    f"{float(self._result.info.dual_obj): .5e}  "
                    f"{float(self._result.info.duality_gap): .5e}  "
                    f"{float(self._result.info.primal_res): .5e}  "
                    f"{float(self._result.info.dual_res): .5e}  "
                    f"{float(self._result.info.rho): .3e}  "
                    f"{float(self._result.info.delta): .3e}  "
                    f"{float(self._result.info.mu): .3e}  "
                    f"{float(self._result.info.primal_step): .4f}  "
                    f"{float(self._result.info.dual_step): .4f}",
                    flush=True
                )

            while self._result.info.factor_retires < self.settings.max_factor_retires:
                factor_succeeded = self._kkt_system.update_scalings_and_factor(self._data, self._result.info.rho, self._result.info.delta, self._result)
                if factor_succeeded:
                    break
                else:
                    self._result.info.factor_retires += 1
                    self._result.info.rho *= 100.
                    self._result.info.delta *= 100.
                    self._result.info.reg_limit = cp.minimum(10 * self._result.info.reg_limit, self.settings.eps_abs)
            
            if self._result.info.factor_retires >= self.settings.max_factor_retires:
                self._result.info.status = Status.PIQP_NUMERICAL_ISSUES
                return self._result.info.status
            
            # reset factor retires for next iteration
            self._result.info.factor_retires = 0

            
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
            self._res.s_l[:] = -self._result.s_l * self._result.z_l
            self._res.s_u[:] = -self._result.s_u * self._result.z_u
            self._res.s_bl[:] = -self._result.s_bl * self._result.z_bl
            self._res.s_bu[:] = -self._result.s_bu * self._result.z_bu
            
            

            if self.settings.debug:
                print("predictor step rhs is: res= ", self._res)

            self._kkt_system.solve(self._data, self.settings, self._res, self._step)

            if self.settings.debug:
                print("predictor step is:", self._step)

            # step in the non-negative orthant
            self._calculate_step()

            # avoid getting to close to the boundary
            self._alpha_sz *= self.settings.tau

            # ------------------ compute centering parameter sigma ------------------
            self._result.info.sigma = 0.
            self._result.info.sigma += cp.dot(self._result.s_l + self._alpha_sz[0] * self._step.s_l, self._result.z_l + self._alpha_sz[1] * self._step.z_l)
            self._result.info.sigma += cp.dot(self._result.s_u + self._alpha_sz[0] * self._step.s_u, self._result.z_u + self._alpha_sz[1] * self._step.z_u)
            self._result.info.sigma += cp.dot(self._result.s_bl + self._alpha_sz[0] * self._step.s_bl, self._result.z_bl + self._alpha_sz[1] * self._step.z_bl)
            self._result.info.sigma += cp.dot(self._result.s_bu + self._alpha_sz[0] * self._step.s_bu, self._result.z_bu + self._alpha_sz[1] * self._step.z_bu)
            self._result.info.sigma /= self._result.info.mu * cp.float64(self._data.num_hl + self._data.num_hu + self._data.num_xl + self._data.num_xu)
            self._result.info.sigma = cp.maximum(0., cp.minimum(1., self._result.info.sigma))
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
            self._calculate_step()
            # avoid getting too close to the boundary
            self._alpha_sz *= self.settings.tau
            self._result.info.primal_step = self._alpha_sz[0]
            self._result.info.dual_step = self._alpha_sz[1]

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

            mu_prev = self._result.info.mu
            self._result.info.mu = self._calculate_mu()
            mu_rate = cp.maximum(0., (mu_prev - self._result.info.mu) / mu_prev)  # r in Algorithm 2 in Roland Schwan 2023 paper


            # ------------------ update regularization ------------------
            self._update_residuals_nr()

            # TODO: more conditions to add in if clause
            if self._result.info.dual_res < 0.95 * self._result.info.prev_dual_res:
                self._prox_vars.x[:] = self._result.x
                self._result.info.rho = cp.maximum(self._result.info.reg_limit, (1. - mu_rate) * self._result.info.rho)
            else:
                self._result.info.rho = cp.maximum(self._result.info.reg_limit, (1. - 0.666 * mu_rate) * self._result.info.rho)

            if self._result.info.primal_res < 0.95 * self._result.info.prev_primal_res:
                self._prox_vars.y[:] = self._result.y
                self._prox_vars.z_l[:] = self._result.z_l
                self._prox_vars.z_u[:] = self._result.z_u
                self._prox_vars.z_bu[:] = self._result.z_bu
                self._prox_vars.z_bl[:] = self._result.z_bl
                
                self._result.info.delta = cp.maximum(self._result.info.reg_limit, (1. - mu_rate) * self._result.info.delta)
            else:
                self._result.info.delta = cp.maximum(self._result.info.reg_limit, (1. - 0.666 * mu_rate) * self._result.info.delta)


        self._result.info.status = Status.PIQP_MAX_ITER_REACHED
        return self._result.info.status

    @nvtx.annotate("Solver::_calculate_step")
    def _calculate_step(self) -> None:
        """
        Compute the step length of the slack variables and dual variables. Make sure they remain non-negative.
        Vectorized implementation to minimize GPU kernel launches and synchronization.
        """
        num_hl, num_hu, num_xl, num_xu = self._data.num_hl, self._data.num_hu, self._data.num_xl, self._data.num_xu

        if not hasattr(self, '_calculate_step_cuda_graphs'):
            self._calculate_step_cuda_graphs = {}
            self._calculate_step_cuda_graphs_capture_count = 0

        key = (self._step.buffer_ptr, self._result.buffer_ptr)
        
        if key not in self._calculate_step_cuda_graphs:
            self._calculate_step_cuda_graphs_capture_count += 1
            print(f"Solver::_calculate_step capturing CUDA graph (occurrence {self._calculate_step_cuda_graphs_capture_count})...")
            stream = cp.cuda.Stream(non_blocking=True)
            stream.begin_capture()
            with stream:
                # first compute alpha_s, use self._work_s to concatenate step, use self._work_z to concatenate result
                offset = 0
                self._work_s[offset : offset+num_hl] = self._step.s_l
                self._work_z[offset : offset+num_hl] = self._result.s_l
                offset += num_hl
                self._work_s[offset:offset + num_hu] = self._step.s_u
                self._work_z[offset:offset + num_hu] = self._result.s_u
                offset += num_hu
                self._work_s[offset:offset + num_xl] = self._step.s_bl
                self._work_z[offset:offset + num_xl] = self._result.s_bl
                offset += num_xl
                self._work_s[offset:offset + num_xu] = self._step.s_bu
                self._work_z[offset:offset + num_xu] = self._result.s_bu
                offset += num_xu

                # if step < 0, must limit alpha <= -s / step, otherwise take full step 1.0
                self._work_s[:offset] = cp.where(self._work_s[:offset] < 0, -self._work_z[:offset] / self._work_s[:offset], 1.)
                self._alpha_sz[0] = cp.min(self._work_s[:offset]) # alpha_s

                # then compute alpha_z, use self._work_s to concatenate step, use self._work_z to concatenate result
                offset = 0
                self._work_z[offset : offset+num_hl] = self._step.z_l
                self._work_s[offset : offset+num_hl] = self._result.z_l
                offset += num_hl
                self._work_z[offset : offset+num_hu] = self._step.z_u
                self._work_s[offset : offset+num_hu] = self._result.z_u
                offset += num_hu
                self._work_z[offset : offset+num_xl] = self._step.z_bl
                self._work_s[offset : offset+num_xl] = self._result.z_bl
                offset += num_xl
                self._work_z[offset : offset+num_xu] = self._step.z_bu
                self._work_s[offset : offset+num_xu] = self._result.z_bu
                offset += num_xu

                self._work_z[:offset] = cp.where(self._work_z[:offset] < 0, -self._work_s[:offset] / self._work_z[:offset], 1.)
                self._alpha_sz[1] = cp.min(self._work_z[:offset]) # alpha_z

            self._calculate_step_cuda_graphs[key] = stream.end_capture()

        self._calculate_step_cuda_graphs[key].launch()

    @nvtx.annotate("Solver::_calculate_mu")
    def _calculate_mu(self) -> float:
        mu = (cp.dot(self._result.s_l, self._result.z_l)
                + cp.dot(self._result.s_u, self._result.z_u) \
                + cp.dot(self._result.s_bl, self._result.z_bl) \
                + cp.dot(self._result.s_bu, self._result.z_bu)) \
                / (self._data.num_hl + self._data.num_hu + self._data.num_xl + self._data.num_xu)
        return mu


    @nvtx.annotate("Solver::_update_residuals_nr")
    def _update_residuals_nr(self):
        """
        Compute the non-regularized residuals, which reflects the residuals of the KKT conditions excluding the regularization terms in Schwan 2023 paper eq(6)

            res_nr.x = -(P*x + c + A^T*y + G^T*(z_u - z_l) + z_bu - z_bl)  # residual of KKT stationarity
            res_nr.y = -(A*x - b)                                          # residual of KKT primal feasibility: A*x = b
            res_nr.z_l = G*x - s_l - hl                                    # residual of KKT primal feasibility: hl <= G*x
            res_nr.z_u = -G*x - s_u + hu                                   # residual of KKT primal feasibility: G*x <= hu
            res_nr.z_bu = -(x - s_bu - xu)                                 # residual of KKT primal feasibility: xl <= x
            res_nr.z_bl = x - s_bl - xl                                    # residual of KKT primal feasibility: x <= xu

        primal_residual = ||[A*x - b; G*x - s_l - hl; -G*x - s_u + hu; -x + s_bu + xu; x - s_bl - xl]||_inf
        dual_residual = ||P*x + c + A^T*y + G^T*(z_u - z_l) + z_bu - z_bl||_inf

        primal_residual_norm = ||A*x; b; G*x; h_u; s_u; h_l; s_l; x_l; s_bu; x_u; s_bl||_inf
        dual_residual_norm = ||P*x; c; A^T*y; G^T*(z_u - z_l) + z_bu - z_bl||_inf
        

        Also updates the primal and dual objectives and duality gap in self._result.info:
            - primal_obj = 0.5 x^T P x + c^T x
            - dual_obj = -0.5 x^T P x - b^T y - h_u^T z_u + h_l^T z_l - x_u^T z_bu + x_l^T z_bl
        """
        # we calculate these term here first to be able to reuse temporary vectors
        self._kkt_system.eval_P_x(self._data, -1., self._result.x, self._res_nr.x)
        dual_res_norm = cp.linalg.norm(self._res_nr.x, ord=cp.inf) # dual_res_norm = max(||P*x||_inf, ||c||_inf, ||A^T*y + G^T*(z_u - z_l) + z_bu - z_bl||_inf), will be updated below
        
        # AT_y = self._res.x  # use self._step.x as temporary storage
        self._kkt_system.eval_A_xn_and_AT_xt(self._data, -1., 1., self._result.x, self._result.y, self._res_nr.y, self._res.x)  # store -A*x in res_nr.y
        primal_rel_norm = cp.linalg.norm(self._res_nr.y, ord=cp.inf) if self._data.p > 0 else 0.  # primal_rel_norm will be updated below

        self._work_z_1.fill(0.)
        self._work_z_1[self._data.idx_hu] += self._result.z_u
        self._work_z_1[self._data.idx_hl] -= self._result.z_l
        
        G_x = self._work_z_2 # reuse self._work_z_2 to store G*x
        GT_zu_minus_zl = self._step.x  # reuse self._step.x as temporary storage
        self._kkt_system.eval_G_xn_and_GT_xt(self._data, 1., 1., self._result.x, self._work_z_1, G_x, GT_zu_minus_zl)

        # ------------ update primal / dual objectives and duality gap ------------
        # primal objective: 0.5 x^T P x + c^T x
        # dual objective is: -0.5 x^T P x - b^T y - h_u^T z_u + h_l^T z_l - x_u^T z_bu + x_l^T z_bl

        # -x^T P x
        tmp = cp.dot(self._res_nr.x, self._result.x)  # self._res_nr.x currently holds -P*x
        self._result.info.primal_obj = -0.5 * tmp
        self._result.info.dual_obj = 0.5 * tmp
        duality_gap_rel_norm = cp.abs(tmp)
        # c^T x
        tmp = cp.dot(self._data.c, self._result.x)
        self._result.info.primal_obj += tmp
        duality_gap_rel_norm = cp.maximum(cp.abs(tmp), duality_gap_rel_norm)
        # -b^T y
        tmp = -cp.dot(self._data.b, self._result.y)
        self._result.info.dual_obj += tmp
        duality_gap_rel_norm = cp.maximum(cp.abs(tmp), duality_gap_rel_norm)
        # h_l^T z_l
        tmp = cp.dot(self._data.h_l[self._data.idx_hl], self._result.z_l)
        self._result.info.dual_obj += tmp
        duality_gap_rel_norm = cp.maximum(cp.abs(tmp), duality_gap_rel_norm)
        # -h_u^T z_u
        tmp = -cp.dot(self._data.h_u[self._data.idx_hu], self._result.z_u)
        self._result.info.dual_obj += tmp
        duality_gap_rel_norm = cp.maximum(cp.abs(tmp), duality_gap_rel_norm)
        # x_l^T z_bl
        tmp = cp.dot(self._data.x_l[self._data.idx_xl], self._result.z_bl)
        self._result.info.dual_obj += tmp
        duality_gap_rel_norm = cp.maximum(cp.abs(tmp), duality_gap_rel_norm)
        # -x_u^T z_bu
        tmp = -cp.dot(self._data.x_u[self._data.idx_xu], self._result.z_bu)
        self._result.info.dual_obj += tmp
        duality_gap_rel_norm = cp.maximum(cp.abs(tmp), duality_gap_rel_norm)
        
        self._result.info.duality_gap = cp.abs(self._result.info.primal_obj - self._result.info.dual_obj)
        self._result.info.duality_gap_rel = self._result.info.duality_gap / cp.maximum(1., duality_gap_rel_norm)

        # res_nr.x = -(P*x + c + A^T*y + G^T*(z_u - z_l) + z_bu - z_bl)
        # self._res_nr.x is computed as -P*x above
        self._res_nr.x -= self._data.c
        self._res_nr.x -= self._res.x  # self._res.x holds A^T*y
        self._res_nr.x -= GT_zu_minus_zl
        self._res_nr.x[self._data.idx_xl] += self._result.z_bl
        self._res_nr.x[self._data.idx_xu] -= self._result.z_bu
        
        # res_nr.y = -(A*x - b)
        self._res_nr.y += self._data.b

        # res_nr.z_l = G*x - s_l - hl  =>  self._res_nr.z_l[:] = (G_x[self._data.idx_hl] - self._result.s_l - self._data.h_l[self._data.idx_hl])
        cp.take(G_x, self._data.idx_hl, out=self._res_nr.z_l)
        cp.subtract(self._res_nr.z_l, self._result.s_l, out=self._res_nr.z_l)
        cp.subtract(self._res_nr.z_l, self._data.h_l[self._data.idx_hl], out=self._res_nr.z_l)  # TODO: this creates a tmp array, optimize?
        
        # res_nr.z_u = -G*x - s_u + hu  =>  self._res_nr.z_u[:] = (-G_x[self._data.idx_hu] - self._result.s_u + self._data.h_u[self._data.idx_hu])
        cp.take(G_x, self._data.idx_hu, out=self._res_nr.z_u)
        cp.negative(self._res_nr.z_u, out=self._res_nr.z_u)
        cp.subtract(self._res_nr.z_u, self._result.s_u, out=self._res_nr.z_u)
        cp.add(self._res_nr.z_u, self._data.h_u[self._data.idx_hu], out=self._res_nr.z_u)  # TODO: this creates a tmp array, optimize?

        # res_nr.z_bl = x - s_bl - xl  =>  self._res_nr.z_bl[:] = (self._result.x[self._data.idx_xl] - self._result.s_bl - self._data.x_l[self._data.idx_xl])
        cp.take(self._result.x, self._data.idx_xl, out=self._res_nr.z_bl)
        cp.subtract(self._res_nr.z_bl, self._result.s_bl, out=self._res_nr.z_bl)
        cp.subtract(self._res_nr.z_bl, self._data.x_l[self._data.idx_xl], out=self._res_nr.z_bl)  # TODO: this creates a tmp array, optimize?

        # res_nr.z_bu = -(x + s_bu - xu)  =>  self._res_nr.z_bu[:] = - (self._result.x[self._data.idx_xu] + self._result.s_bu - self._data.x_u[self._data.idx_xu])
        cp.take(self._result.x, self._data.idx_xu, out=self._res_nr.z_bu)
        cp.add(self._res_nr.z_bu, self._result.s_bu, out=self._res_nr.z_bu)
        cp.subtract(self._res_nr.z_bu, self._data.x_u[self._data.idx_xu], out=self._res_nr.z_bu)  # TODO: this creates a tmp array, optimize?
        cp.negative(self._res_nr.z_bu, out=self._res_nr.z_bu)


        # ------------ update primal and dual residuals ------------
        self._result.info.prev_primal_res = self._result.info.primal_res
        self._result.info.prev_dual_res = self._result.info.dual_res

        self._result.info.primal_res = self._primal_res_nr()

        # primal_rel_norm is computed as ||-A*x||_inf above, now update it with other terms
        primal_rel_norm = cp.maximum(primal_rel_norm, cp.linalg.norm(self._data.b, ord=cp.inf)) if self._data.p > 0 else primal_rel_norm
        primal_rel_norm = cp.maximum(primal_rel_norm, cp.linalg.norm(G_x[self._data.idx_hu], ord=cp.inf)) if self._data.num_hu > 0 else primal_rel_norm
        primal_rel_norm = cp.maximum(primal_rel_norm, cp.linalg.norm(G_x[self._data.idx_hl], ord=cp.inf)) if self._data.num_hl > 0 else primal_rel_norm
        primal_rel_norm = cp.maximum(primal_rel_norm, cp.linalg.norm(self._data.h_u[self._data.idx_hu], ord=cp.inf)) if self._data.num_hu > 0 else primal_rel_norm
        primal_rel_norm = cp.maximum(primal_rel_norm, cp.linalg.norm(self._data.h_l[self._data.idx_hl], ord=cp.inf)) if self._data.num_hl > 0 else primal_rel_norm
        primal_rel_norm = cp.maximum(primal_rel_norm, cp.linalg.norm(self._data.x_u[self._data.idx_xu], ord=cp.inf)) if self._data.num_xu > 0 else primal_rel_norm
        primal_rel_norm = cp.maximum(primal_rel_norm, cp.linalg.norm(self._data.x_l[self._data.idx_xl], ord=cp.inf)) if self._data.num_xl > 0 else primal_rel_norm
        primal_rel_norm = cp.maximum(primal_rel_norm, cp.linalg.norm(self._result.s_u, ord=cp.inf)) if self._data.num_hu > 0 else primal_rel_norm
        primal_rel_norm = cp.maximum(primal_rel_norm, cp.linalg.norm(self._result.s_l, ord=cp.inf)) if self._data.num_hl > 0 else primal_rel_norm
        primal_rel_norm = cp.maximum(primal_rel_norm, cp.linalg.norm(self._result.s_bu, ord=cp.inf)) if self._data.num_xu > 0 else primal_rel_norm
        primal_rel_norm = cp.maximum(primal_rel_norm, cp.linalg.norm(self._result.s_bl, ord=cp.inf)) if self._data.num_xl > 0 else primal_rel_norm
        self._result.info.primal_res_rel = self._result.info.primal_res / cp.maximum(1., primal_rel_norm)

        # dual_res_norm = max(||P*x||_inf, ||c||_inf, ||A^T*y + G^T*(z_u - z_l) + z_bu - z_bl||_inf)
        self._result.info.dual_res = self._dual_res_nr()
        dual_res_norm = cp.maximum(dual_res_norm, cp.linalg.norm(self._data.c, ord=cp.inf))  # ||P*x||_inf was calculated before
        # self._res.x currently holds A^T*y
        self._res.x += GT_zu_minus_zl
        self._res.x[self._data.idx_xl] -= self._result.z_bl
        self._res.x[self._data.idx_xu] += self._result.z_bu
        dual_res_norm = cp.maximum(dual_res_norm, cp.linalg.norm(self._res.x, ord=cp.inf))
        self._result.info.dual_res_rel = self._result.info.dual_res / cp.maximum(1., dual_res_norm)
        

    @nvtx.annotate("Solver::_update_residuals_r")
    def _update_residuals_r(self):
        """
        Compute the regularized primal and dual residuals. The computation is based on the non-regularized residuals computed in _update_residuals_nr.
        It adds the regularization terms to the non-regularized residuals to obtain the regularized residuals.
        """
        # update the rhs of the KKT system
        # r_x = -(P*x + c + A^T*y + G^T*(z_u - z_l)) - rho * (x - x_prox)
        self._res.x[:] = self._res_nr.x - self._result.info.rho * (self._result.x - self._prox_vars.x)
        # r_y = -(A*x - b - delta*(y - lamda))  # TODO: I understand the formula, but why is it computed this way? I copied the following line from the original C++ code.
        self._res.y[:] = self._res_nr.y - self._result.info.delta * (self._prox_vars.y - self._result.y)
        self._res.z_l[:] = self._res_nr.z_l - self._result.info.delta * (self._prox_vars.z_l - self._result.z_l)
        self._res.z_u[:] = self._res_nr.z_u - self._result.info.delta * (self._prox_vars.z_u - self._result.z_u)
        self._res.z_bl[:] = self._res_nr.z_bl - self._result.info.delta * (self._prox_vars.z_bl - self._result.z_bl)
        self._res.z_bu[:] = self._res_nr.z_bu - self._result.info.delta * (self._prox_vars.z_bu - self._result.z_bu)
        
        primal_rel_scaling = self._result.info.primal_res / self._result.info.primal_res_rel if self._result.info.primal_res_rel > 0 else 1.
        dual_rel_scaling = self._result.info.dual_res / self._result.info.dual_res_rel if self._result.info.dual_res_rel > 0 else 1.

        self._result.info.primal_res_reg = self._primal_res_r()
        self._result.info.primal_res_reg_rel = self._result.info.primal_res_reg / primal_rel_scaling
        self._result.info.dual_res_reg = self._dual_res_r()
        self._result.info.dual_res_reg_rel = self._result.info.dual_res_reg / dual_rel_scaling

        self._result.info.primal_prox_inf = self._primal_prox_inf() * self._result.info.delta
        self._result.info.dual_prox_inf = self._dual_prox_inf() * self._result.info.rho

    @nvtx.annotate("Solver::_primal_res_nr")
    def _primal_res_nr(self) -> float:
        inf = cp.float64(0.)
        inf = cp.maximum(inf, cp.linalg.norm(self._res_nr.y, ord=cp.inf)) if self._data.p > 0 else inf
        inf = cp.maximum(inf, cp.linalg.norm(self._res_nr.z_u, ord=cp.inf)) if self._data.num_hu > 0 else inf
        inf = cp.maximum(inf, cp.linalg.norm(self._res_nr.z_l, ord=cp.inf)) if self._data.num_hl > 0 else inf
        #! The values of z_u, z_l, z_bu, z_bl with active indices should all be positive.
        #! can optionally use cp.maximum, if more efficient
        inf = cp.maximum(inf, cp.linalg.norm(self._res_nr.z_bu, ord=cp.inf)) if self._data.num_xu > 0 else inf
        inf = cp.maximum(inf, cp.linalg.norm(self._res_nr.z_bl, ord=cp.inf)) if self._data.num_xl > 0 else inf
        return inf


    @nvtx.annotate("Solver::_primal_res_r")
    def _primal_res_r(self) -> float:
        inf = cp.float64(0.)
        inf = cp.maximum(inf, cp.linalg.norm(self._res.y, ord=cp.inf)) if self._data.p > 0 else inf
        inf = cp.maximum(inf, cp.linalg.norm(self._res.z_u, ord=cp.inf)) if self._data.num_hu > 0 else inf
        inf = cp.maximum(inf, cp.linalg.norm(self._res.z_l, ord=cp.inf)) if self._data.num_hl > 0 else inf
        inf = cp.maximum(inf, cp.linalg.norm(self._res.z_bu, ord=cp.inf)) if self._data.num_xu > 0 else inf
        inf = cp.maximum(inf, cp.linalg.norm(self._res.z_bl, ord=cp.inf)) if self._data.num_xl > 0 else inf
        return inf
    
    @nvtx.annotate("Solver::_dual_res_nr")
    def _dual_res_nr(self) -> float:
        return cp.linalg.norm(self._res_nr.x, ord=cp.inf)
    
    @nvtx.annotate("Solver::_dual_res_r")
    def _dual_res_r(self) -> float:
        return cp.linalg.norm(self._res.x, ord=cp.inf)
    
    @nvtx.annotate("Solver::_primal_prox_inf")
    def _primal_prox_inf(self) -> float:
        inf = cp.float64(0.)
        inf = cp.maximum(inf, cp.linalg.norm(self._result.y - self._prox_vars.y, ord=cp.inf)) if self._data.p > 0 else inf
        inf = cp.maximum(inf, cp.linalg.norm(self._result.z_l - self._prox_vars.z_l, ord=cp.inf)) if self._data.num_hl > 0 else inf
        inf = cp.maximum(inf, cp.linalg.norm(self._result.z_u - self._prox_vars.z_u, ord=cp.inf)) if self._data.num_hu > 0 else inf
        inf = cp.maximum(inf, cp.linalg.norm(self._result.z_bl - self._prox_vars.z_bl, ord=cp.inf)) if self._data.num_xl > 0 else inf
        inf = cp.maximum(inf, cp.linalg.norm(self._result.z_bu - self._prox_vars.z_bu, ord=cp.inf)) if self._data.num_xu > 0 else inf
        return inf
    
    @nvtx.annotate("Solver::_dual_prox_inf")
    def _dual_prox_inf(self) -> float:
        return cp.linalg.norm(self._result.x - self._prox_vars.x, ord=cp.inf)
    