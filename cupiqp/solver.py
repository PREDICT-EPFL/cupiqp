import cupy as cp
import warp as wp
from typing import Optional, Any
import nvtx

from .settings import Settings
from .data import Data
from .results import Result, Status, Variables
from .kkt_systems import KKTSystem


wp.config.quiet = True  # disable warp module initialization messages.
wp.config.enable_backward = False  # disable backward mode, cut down kernel compile time
wp.init()


class SolverBase:
    def __init__(self):

        self.settings = Settings()
        self._data: Data = None
        self._result = Result()    # store the values of primal, dual and slack variables of current iteration, and other information
        self._step = Variables()   # used to store the step direction of primal and dual variables
        self._res_nr = Variables()  # used to store the non-regularized residuals
        self._res = Variables()  # used to store the regularized residuals
        self._prox_vars = Variables()  # used to store the proximal variables
        self._kkt_system = KKTSystem()

    @property
    def result(self):
        return self._result
    
    @nvtx.annotate("Solver::setup")
    def setup(self, P, c, A, b, G, h_u, h_l, x_u, x_l):
        if self.settings.kkt_solver == "dense_cholesky":
            from .dense.dense_data import DenseData
            self._data = DenseData(P, c, A, b, G, h_u, h_l, x_u, x_l)
        elif self.settings.kkt_solver == "sparse_ldlt":
            from .sparse.sparse_data import SparseData
            self._data = SparseData(P, c, A, b, G, h_u, h_l, x_u, x_l)
        elif self.settings.kkt_solver == "multistage_block_cholesky":
            from .multistage.multistage_data import MultistageData
            self._data = MultistageData(P, c, A, b, G, h_u, h_l, x_u, x_l)
        else:
            raise ValueError(f"Unknown kkt_solver type: {self.settings.kkt_solver}")
        self._result.init(self._data)
        self._result.info.rho[0] = self.settings.rho_init
        self._result.info.delta[0] = self.settings.delta_init
        self._result.init(self._data)
        
        self._step.init(self._data)
        self._res_nr.init(self._data)
        self._res.init(self._data)
        self._prox_vars.init(self._data)

        self._kkt_system.init(self._data, self.settings)

        self._work_z_1 = cp.empty(self._data.m)  # used to store intermediate results in _update_residuals_nr
        self._work_z_2 = cp.empty(self._data.m)  # used to store intermediate results in _update_residuals_nr

        self._work_z = cp.empty(self._data.num_hl + self._data.num_hu + self._data.num_xl + self._data.num_xu)  # used in _calculate_step to hold all concatenated slack or dual steps / results
        self._work_s = cp.empty(self._data.num_hl + self._data.num_hu + self._data.num_xl + self._data.num_xu)  # used in _calculate_step to hold all concatenated slack or dual steps / results
        self._alpha_sz = cp.empty(2) # step lengths of slack and dual variables [alpha_s, alpha_z]
        self._work_primals = cp.empty(self._data.n)
        self._work_duals = cp.empty(self._data.p + self._data.num_hl + self._data.num_hu + self._data.num_xl + self._data.num_xu)  # used to hold the concatenated dual variables for computing the residuals in _update_residuals_nr
        self._work_residual = cp.empty(())
        self._work_reduce = cp.empty(8)  # used to hold the intermediate results of the reductions related to s_l, s_u, s_bl, s_bu and z_l, z_u, z_bl, z_bu
        
        self._update_residuals_r_kernel = create_update_residual_r_kernel(
            self._data.n, self._data.p, self._data.num_hl, self._data.num_hu, self._data.num_xl, self._data.num_xu
        )
        # Pre-allocated scalar buffers for CUDA-graph-safe norm computations in _update_residuals_nr / _update_residuals_r
        self._work_primal_rel_norm = cp.empty(1, dtype=cp.float64)  # running max of primal relative norm terms
        self._work_dual_res_norm = cp.empty(1, dtype=cp.float64)    # running max of dual residual norm terms
        self._work_norm_temp = cp.empty(1, dtype=cp.float64)        # temp scalar for individual norm results

        self._mu_prev = cp.empty(1)
        self._mu_rate = cp.empty(1)

        self._setup_done = True

    def update(self,
               P: Optional[Any] = None,
               c: Optional[Any] = None,
               A: Optional[Any] = None,
               b: Optional[Any] = None,
               G: Optional[Any] = None,
               h_u: Optional[Any] = None,
               h_l: Optional[Any] = None,
               x_u: Optional[Any] = None,
               x_l: Optional[Any] = None,
               check_validity: bool = False,
               ):
        """Update problem data between solves without a full setup().

        Only numerical values are updated; dimensions and sparsity patterns
        must remain unchanged.  Call ``solve()`` again after ``update()``.

        Args:
            check_validity: If True, validate dimensions/sparsity of the new data.
                   Defaults to False for maximum performance (skips D2H syncs
                   from sparsity pattern validation in the sparse backend).
                   The caller must ensure that the finite/infinite bound
                   structure is unchanged (same indices have finite bounds
                   as in ``setup()``).
        """
        if not self._setup_done:
            raise RuntimeError("Solver not setup yet. Call setup() first.")

        if P is not None:
            self._data.set_P(P, check=check_validity)
        if c is not None:
            self._data.set_c(c, check=check_validity)
        if A is not None:
            self._data.set_A(A, check=check_validity)
        if b is not None:
            self._data.set_b(b, check=check_validity)
        if G is not None:
            self._data.set_G(G, check=check_validity)
        if h_u is not None:
            self._data.set_h_u(h_u, check=check_validity)
        if h_l is not None:
            self._data.set_h_l(h_l, check=check_validity)
        if x_u is not None:
            self._data.set_x_u(x_u, check=check_validity)
        if x_l is not None:
            self._data.set_x_l(x_l, check=check_validity)

        self._data._compute_constraints_rhs_inf_norm()
        self._kkt_system.update_data(self._data, P is not None, A is not None, G is not None)

    def solve(self) -> Status:
        if self.settings.verbose:
            if self.settings.kkt_solver == "dense_cholesky":
                print("dense backend:")
                print(f"variables n = {self._data.n}")
                print(f"equality constraints p = {self._data.p}")
                print(f"inequality constraints m = {self._data.m}")
            elif self.settings.kkt_solver == "sparse_ldlt":
                print("sparse backend:")
                print(f"variables n = {self._data.n}, nnz(P) = {self._data.P.nnz}")
                print(f"equality constraints p = {self._data.p}, nnz(A) = {self._data.A.nnz}")
                print(f"inequality constraints m = {self._data.m}, nnz(G) = {self._data.G.nnz}")
            elif self.settings.kkt_solver == "multistage_block_cholesky":
                print("multistage backend:")
                print(f"variables n = {self._data.n}, num_diag_blocks(P) = {self._data.P.num_diag_blocks}, block_size(P) = ({self._data.P.block_size}, {self._data.P.block_size})")
                print(f"equality constraints p = {self._data.p}, num_diag_blocks(A) = {self._data.A.N}, block_size(A) = ({self._data.A.rows_of_blocks}, {self._data.A.cols_of_blocks})")
                print(f"inequality constraints m = {self._data.m}, num_diag_blocks(G) = {self._data.G.N}, block_size(G) = ({self._data.G.rows_of_blocks}, {self._data.G.cols_of_blocks})")
            else:
                raise ValueError(f"Unsupported kkt_solver type: {self.settings.kkt_solver}")
            
            print(f"inequality lower bounds n_h_l = {self._data.num_hl}")
            print(f"inequality upper bounds n_h_u = {self._data.num_hu}")
            print(f"variable lower bounds n_x_l = {self._data.num_xl}")
            print(f"variable upper bounds n_x_u = {self._data.num_xu}")
            print("")
        return self._solve_impl()

    def _solve_impl(self) -> Status:
        self._result.info.status = Status.PIQP_UNSOLVED 
        self._result.info.iter = 0
        self._result.info.reg_limit[0] = self.settings.reg_lower_limit
        self._result.info.factor_retires = 0
        self._result.info.no_primal_update = 0
        self._result.info.no_dual_update = 0
        self._result.info.mu[0] = 0.
        self._result.info.primal_step[0] = 0.
        self._result.info.dual_step[0] = 0.
        self._result.info.rho[0] = self.settings.rho_init
        self._result.info.delta[0] = self.settings.delta_init

        if self.settings.verbose:
            print("iter  prim_obj       dual_obj       duality_gap   prim_res      dual_res      rho         delta       mu          p_step   d_step")  

        ## ----------- initial iteration --------------
        self._initial_guess()

        ## ---------------------------------------------
        ## ---------- remaining iterations -------------
        ## ---------------------------------------------
        for iter in range(self.settings.max_iter):
            with nvtx.annotate(f"Solver::ipm_iteration"):
                self._result.info.iter = iter
                if iter == 0:
                    self._update_residuals_nr()
                    self._result.info.prev_primal_res[:] = self._result.info.primal_res
                    self._result.info.prev_dual_res[:] = self._result.info.dual_res

                # ? The convergence criteria seems different from the one in the paper
                if ((self._result.info.primal_res < self.settings.eps_abs or self._result.info.primal_res_rel < self.settings.eps_rel) and
                    (self._result.info.dual_res < self.settings.eps_abs or self._result.info.dual_res_rel < self.settings.eps_rel) and
                    (not self.settings.check_duality_gap or self._result.info.duality_gap < self.settings.eps_duality_gap_abs or self._result.info.duality_gap_rel < self.settings.eps_duality_gap_rel)):
                    self._result.info.status = Status.PIQP_SOLVED
                    return self._result.info.status
                
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
                
                # avoid possibility of converging to a local minimum -> decrease the minimum regularization value
                if ((self._result.info.no_primal_update > self.settings.reg_finetune_primal_update_threshold and
                    self._result.info.rho[0] == self._result.info.reg_limit[0] and
                    self._result.info.reg_limit[0] != self.settings.reg_finetune_lower_limit) or
                    (self._result.info.no_dual_update > self.settings.reg_finetune_dual_update_threshold and
                    self._result.info.delta[0] == self._result.info.reg_limit[0] and
                    self._result.info.reg_limit[0] != self.settings.reg_finetune_lower_limit)):
                    if (self._result.info.dual_prox_inf < self.settings.infeasibility_threshold and self._result.info.primal_prox_inf < self.settings.infeasibility_threshold):
                        self._result.info.reg_limit[0] = self.settings.reg_finetune_lower_limit
                        self._result.info.no_primal_update = 0
                        self._result.info.no_dual_update = 0
                
                if self.settings.verbose:
                    self._print_iteration_info()

                self._update_and_factorize_kkt()
                if self._result.info.status == Status.PIQP_NUMERICAL_ISSUES:
                    return self._result.info.status

                if self._data.num_hl + self._data.num_hu + self._data.num_xl + self._data.num_xu == 0:
                    # since there are no inequalities we can take full Newton steps
                    self._run_full_newton_step()
                    self._update_residuals_nr()
                    self._update_rho_delta_without_ineq()
                else:
                    self._run_predictor_corrector()
                    self._update_residuals_nr()
                    self._update_rho_delta_with_ineq()
                
        self._result.info.status = Status.PIQP_MAX_ITER_REACHED
        return self._result.info.status        
        

    @nvtx.annotate("Solver::_initial_guess")
    def _initial_guess(self):
        # eq(12) in Roland Schwan 2023 paper
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

        if self._data.num_hl + self._data.num_hu + self._data.num_xl + self._data.num_xu > 0:
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
            delta_s = -cp.min(self._work_s[:offset]) # single D2H transfer
            delta_z = -cp.min(self._work_z[:offset]) # single D2H transfer

            self._result.s_l += delta_s
            self._result.z_l += delta_z
            self._result.s_u += delta_s
            self._result.z_u += delta_z

            self._result.s_bl += delta_s
            self._result.z_bl += delta_z
            self._result.s_bu += delta_s
            self._result.z_bu += delta_z

            # need to make sure mu is positive here, otherwise in the next step (put s and z on central path) sqrt(mu) the computed z_* will be zeros
            self._calculate_mu()
            cp.clip(self._result.info.mu, 1e-10, None, out=self._result.info.mu)

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
                z += 4. * self._result.info.mu[0]
                cp.sqrt(z, out=z)
                z += s
                z /= 2.
                cp.subtract(z, s, out=s)

            if self.settings.debug:
                print("self._result:", self._result)

            self._calculate_mu()

        self._prox_vars.x[:] = self._result.x
        self._prox_vars.y[:] = self._result.y
        self._prox_vars.z_l[:] = self._result.z_l
        self._prox_vars.z_u[:] = self._result.z_u
        self._prox_vars.z_bl[:] = self._result.z_bl
        self._prox_vars.z_bu[:] = self._result.z_bu

    @nvtx.annotate("Solver::_print_iteration_info")
    def _print_iteration_info(self):
        """Print iteration verbose info."""
        print(
            f"{self._result.info.iter:3d}   "
            f"{float(self._result.info.primal_obj[0]): .5e}   "
            f"{float(self._result.info.dual_obj[0]): .5e}  "
            f"{float(self._result.info.duality_gap[0]): .5e}  "
            f"{float(self._result.info.primal_res[0]): .5e}  "
            f"{float(self._result.info.dual_res[0]): .5e}  "
            f"{float(self._result.info.rho[0]): .3e}  "
            f"{float(self._result.info.delta[0]): .3e}  "
            f"{float(self._result.info.mu[0]): .3e}  "
            f"{float(self._result.info.primal_step[0]): .4f}  "
            f"{float(self._result.info.dual_step[0]): .4f}",
            flush=True
        )

    @nvtx.annotate("Solver::_update_and_factorize_kkt")
    def _update_and_factorize_kkt(self) -> None:
        """Update the KKT matrix and refactorize."""
        while self._result.info.factor_retires < self.settings.max_factor_retires:
            factor_succeeded = self._kkt_system.update_scalings_and_factor(self._data, self._result.info.rho, self._result.info.delta, self._result)
            if factor_succeeded:
                break
            else:
                self._result.info.factor_retires += 1
                self._result.info.rho *= 100.
                self._result.info.delta *= 100.
                self._result.info.reg_limit[:] = cp.minimum(10 * self._result.info.reg_limit, self.settings.eps_abs)
        
        if self._result.info.factor_retires >= self.settings.max_factor_retires:
            self._result.info.status = Status.PIQP_NUMERICAL_ISSUES
        
        # reset factor retires for next iteration
        self._result.info.factor_retires = 0

    @nvtx.annotate("Solver::_run_full_newton_step")
    def _run_full_newton_step(self):
        self._kkt_system.solve(self._data, self.settings, self._res, self._step)
        self._result.info.primal_step[:] = 1.0
        self._result.info.dual_step[:] = 1.0
        self._result.x += self._result.info.primal_step * self._step.x
        self._result.y += self._result.info.dual_step * self._step.y

    @nvtx.annotate("Solver::_run_predictor_corrector")
    def _run_predictor_corrector(self):
        """Predictor-corrector steps + variable update + mu calculation."""
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
        with nvtx.annotate("Solver::prepare_predictor_step"):
            cp.multiply(self._result.s_l, self._result.z_l, out=self._res.s_l)
            self._res.s_l *= -1.
            cp.multiply(self._result.s_u, self._result.z_u, out=self._res.s_u)
            self._res.s_u *= -1.
            cp.multiply(self._result.s_bl, self._result.z_bl, out=self._res.s_bl)
            self._res.s_bl *= -1.
            cp.multiply(self._result.s_bu, self._result.z_bu, out=self._res.s_bu)
            self._res.s_bu *= -1.
            

        if self.settings.debug:
            print("predictor step rhs is: res= ", self._res)

        self._kkt_system.solve(self._data, self.settings, self._res, self._step)

        if self.settings.debug:
            print("predictor step is:", self._step)

        # step in the non-negative orthant
        self._calculate_step()

        # avoid getting too close to the boundary
        self._alpha_sz *= self.settings.tau

        # ------------------ compute centering parameter sigma ------------------
        self._calculate_sigma()

        # ------------------ corrector step ------------------
        with nvtx.annotate("Solver::prepare_corrector_step"):
            # self._res.s_l += -self._step.s_l * self._step.z_l + self._result.info.sigma * self._result.info.mu
            # self._res.s_u += -self._step.s_u * self._step.z_u + self._result.info.sigma * self._result.info.mu
            # self._res.s_bl += -self._step.s_bl * self._step.z_bl + self._result.info.sigma * self._result.info.mu
            # self._res.s_bu += -self._step.s_bu * self._step.z_bu + self._result.info.sigma * self._result.info.mu
            tmp_sigma_mu = self._result.info.sigma * self._result.info.mu
            cp.multiply(self._step.s_l, self._step.z_l, out=self._work_duals[:self._data.num_hl])
            self._res.s_l -= self._work_duals[:self._data.num_hl]
            self._res.s_l += tmp_sigma_mu
            cp.multiply(self._step.s_u, self._step.z_u, out=self._work_duals[:self._data.num_hu])
            self._res.s_u -= self._work_duals[:self._data.num_hu]
            self._res.s_u += tmp_sigma_mu
            cp.multiply(self._step.s_bl, self._step.z_bl, out=self._work_duals[:self._data.num_xl])
            self._res.s_bl -= self._work_duals[:self._data.num_xl]
            self._res.s_bl += tmp_sigma_mu
            cp.multiply(self._step.s_bu, self._step.z_bu, out=self._work_duals[:self._data.num_xu])
            self._res.s_bu -= self._work_duals[:self._data.num_xu]
            self._res.s_bu += tmp_sigma_mu

        if self.settings.debug:
            print("corrector step rhs is: res= ", self._res)
        self._kkt_system.solve(self._data, self.settings, self._res, self._step)

        if self.settings.debug:
            print("corrector step is:", self._step)

        # step in the non-negative orthant
        self._calculate_step()
        # avoid getting too close to the boundary
        self._alpha_sz *= self.settings.tau
        self._result.info.primal_step[:] = self._alpha_sz[0]
        self._result.info.dual_step[:] = self._alpha_sz[1]

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

        # ------------------ update mu and mu_rate for adaptive regularization ------------------
        self._mu_prev[:] = self._result.info.mu
        self._calculate_mu()
        cp.subtract(self._mu_prev, self._result.info.mu, out=self._mu_rate)
        cp.divide(self._mu_rate, self._mu_prev, out=self._mu_rate)
        cp.maximum(self._mu_rate, 0., out=self._mu_rate)

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
            # print(f"Solver::_calculate_step capturing CUDA graph (occurrence {self._calculate_step_cuda_graphs_capture_count})...")
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
    def _calculate_mu(self) -> None:
        if not hasattr(self, '_calculate_mu_cuda_graphs'):
            self._calculate_mu_cuda_graphs = {}
            self._calculate_mu_cuda_graphs_capture_count = 0
        key = (self._result.buffer_ptr, self._result.info.mu.data.ptr)
        if key not in self._calculate_mu_cuda_graphs:
            self._calculate_mu_cuda_graphs_capture_count += 1
            # print(f"Solver::_calculate_mu capturing CUDA graph (occurrence {self._calculate_mu_cuda_graphs_capture_count})...")
            stream = cp.cuda.Stream(non_blocking=True)
            stream.begin_capture()
            with stream:
                cp.dot(self._result.s_l, self._result.z_l, out=self._work_reduce[0:1])
                cp.dot(self._result.s_u, self._result.z_u, out=self._work_reduce[1:2])
                cp.dot(self._result.s_bl, self._result.z_bl, out=self._work_reduce[2:3])
                cp.dot(self._result.s_bu, self._result.z_bu, out=self._work_reduce[3:4])
                cp.sum(self._work_reduce[0:4], out=self._result.info.mu[:], keepdims=True)
                self._result.info.mu /= (self._data.num_hl + self._data.num_hu + self._data.num_xl + self._data.num_xu)
            self._calculate_mu_cuda_graphs[key] = stream.end_capture()

        self._calculate_mu_cuda_graphs[key].launch()

    @nvtx.annotate("Solver::_calculate_sigma")
    def _calculate_sigma(self) -> None:
        if not hasattr(self, '_calculate_sigma_cuda_graphs'):
            self._calculate_sigma_cuda_graphs = {}
            self._calculate_sigma_cuda_graphs_capture_count = 0

        key = (self._alpha_sz.data.ptr,
                self._result.buffer_ptr,
                self._step.buffer_ptr,
                self._result.info.sigma.data.ptr,
                self._result.info.mu.data.ptr
            )
        
        if key not in self._calculate_sigma_cuda_graphs:
            self._calculate_sigma_cuda_graphs_capture_count += 1
            # print(f"Solver::_calculate_sigma capturing CUDA graph (occurrence {self._calculate_sigma_cuda_graphs_capture_count})...")
            stream = cp.cuda.Stream(non_blocking=True)
            stream.begin_capture()
            with stream:
                self._result.info.sigma[:] = 0.
                self._result.info.sigma += cp.dot(self._result.s_l + self._alpha_sz[0] * self._step.s_l, self._result.z_l + self._alpha_sz[1] * self._step.z_l)
                self._result.info.sigma += cp.dot(self._result.s_u + self._alpha_sz[0] * self._step.s_u, self._result.z_u + self._alpha_sz[1] * self._step.z_u)
                self._result.info.sigma += cp.dot(self._result.s_bl + self._alpha_sz[0] * self._step.s_bl, self._result.z_bl + self._alpha_sz[1] * self._step.z_bl)
                self._result.info.sigma += cp.dot(self._result.s_bu + self._alpha_sz[0] * self._step.s_bu, self._result.z_bu + self._alpha_sz[1] * self._step.z_bu)
                
                cp.divide(self._result.info.sigma, self._result.info.mu, out=self._result.info.sigma)
                self._result.info.sigma /= self._data.num_hl + self._data.num_hu + self._data.num_xl + self._data.num_xu
                cp.clip(self._result.info.sigma, 0., 1., out=self._result.info.sigma)
                cp.power(self._result.info.sigma, 3., out=self._result.info.sigma)

            self._calculate_sigma_cuda_graphs[key] = stream.end_capture()

        self._calculate_sigma_cuda_graphs[key].launch()


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
        if not hasattr(self, '_update_residuals_nr_cuda_graphs'):
            self._update_residuals_nr_cuda_graphs = {}
            self._update_residuals_nr_cuda_graphs_capture_count = 0

        key = (self._res.buffer_ptr, 
               self._result.buffer_ptr,
               self._prox_vars.buffer_ptr,
               self._res_nr.buffer_ptr)
        # cuSPARSE/cuBLAS operations (outside graph capture for now)
        self._kkt_system.eval_P_x(self._data, -1., self._result.x, self._res_nr.x)
        # ||P*x||_inf -> _work_dual_res_norm (will be updated further inside graph)
        cp.absolute(self._res_nr.x, out=self._work_primals)
        cp.max(self._work_primals, out=self._work_dual_res_norm, keepdims=True)

        self._kkt_system.eval_A_xn(self._data, -1., self._result.x, self._res_nr.y)  # store -A*x in res_nr.y
        self._kkt_system.eval_AT_xt(self._data, 1., self._result.y, self._res.x)  # add -A^T*y to res_nr.y
        # ||A*x||_inf -> _work_primal_rel_norm (will be updated further inside graph)
        if self._data.p > 0:
            cp.absolute(self._res_nr.y, out=self._work_duals[:self._data.p])
            cp.max(self._work_duals[:self._data.p], out=self._work_primal_rel_norm, keepdims=True)
        else:
            self._work_primal_rel_norm.fill(0.)

        self._work_z_1.fill(0.)
        self._work_z_1[self._data.idx_hu] += self._result.z_u
        self._work_z_1[self._data.idx_hl] -= self._result.z_l

        G_x = self._work_z_2 # reuse self._work_z_2 to store G*x
        GT_zu_minus_zl = self._step.x  # reuse self._step.x as temporary storage
        self._kkt_system.eval_G_xn(self._data, 1., self._result.x, G_x)
        self._kkt_system.eval_GT_xt(self._data, 1., self._work_z_1, GT_zu_minus_zl)

        if key not in self._update_residuals_nr_cuda_graphs:
            self._update_residuals_nr_cuda_graphs_capture_count += 1
            # print(f"Solver::_update_residuals_nr capturing CUDA graph (occurrence {self._update_residuals_nr_cuda_graphs_capture_count})...")
            stream = cp.cuda.Stream(non_blocking=True)
            stream.begin_capture()
            with stream:
                # ------------ update primal / dual objectives and duality gap ------------
                # primal objective: 0.5 x^T P x + c^T x
                # dual objective is: -0.5 x^T P x - b^T y - h_u^T z_u + h_l^T z_l - x_u^T z_bu + x_l^T z_bl

                # use self._work_reduce to hold intermediate terms for primal and dual objectives:
                # index |  content
                #  0    |   0.5 * x^T P x
                #  1    |   c^T x
                #  2    |   0.5 * x^T P x (copy of idx 0 for easier reduction)
                #  3    |   b^T y
                #  4    |   -h_l^T z_l
                #  5    |   h_u^T z_u
                #  6    |   -x_l^T z_bl
                #  7    |   x_u^T z_bu

                cp.dot(self._res_nr.x, self._result.x, out=self._work_reduce[0:1])
                self._work_reduce[0:1] *= -0.5  # hold 0.5 * x^T P x
                cp.dot(self._data.c, self._result.x, out=self._work_reduce[1:2]) # hold c^T x

                cp.multiply(self._work_reduce[0:1], 1., out=self._work_reduce[2:3])  # hold 0.5 * x^T P x
                cp.dot(self._data.b, self._result.y, out=self._work_reduce[3:4]) # hold b^T y
                cp.dot(self._data.h_l[self._data.idx_hl], self._result.z_l, out=self._work_reduce[4:5])  # hold h_l^T z_l
                self._work_reduce[4:5] *= -1.  # hold - h_l^T z_l
                cp.dot(self._data.h_u[self._data.idx_hu], self._result.z_u, out=self._work_reduce[5:6])  # hold h_u^T z_u
                cp.dot(self._data.x_l[self._data.idx_xl], self._result.z_bl, out=self._work_reduce[6:7])
                self._work_reduce[6:7] *= -1.  # hold - x_l^T z_bl
                cp.dot(self._data.x_u[self._data.idx_xu], self._result.z_bu, out=self._work_reduce[7:8])  # hold x_u^T z_bu

                cp.sum(self._work_reduce[0:2], out=self._result.info.primal_obj, keepdims=True)  # primal_obj = 0.5 x^T P x + c^T x
                cp.sum(self._work_reduce[2:8], out=self._result.info.dual_obj, keepdims=True)
                self._result.info.dual_obj *= -1.  # dual_obj = -b^T y - h_u^T z_u + h_l^T z_l - x_u^T z_bu + x_l^T z_bl

                # duality_gap = abs(primal_obj - dual_obj)
                cp.subtract(self._result.info.primal_obj, self._result.info.dual_obj, out=self._result.info.duality_gap)
                
                # duality_gap_rel_norm = max(abs(0.5 x'Px, b'y, h_u'z_u, h_l'z_l, x_u'z_bu, x_l'z_bl))
                # duality_gap_relative = duality_gap / maximum(1., duality_gap_rel_norm)
                cp.abs(self._work_reduce, out=self._work_reduce)
                cp.max(self._work_reduce[0:8], out=self._result.info.duality_gap_rel, keepdims=True)
                cp.abs(self._result.info.duality_gap, out=self._result.info.duality_gap)
                cp.maximum(self._result.info.duality_gap_rel, 1., out=self._result.info.duality_gap_rel)
                cp.divide(self._result.info.duality_gap, self._result.info.duality_gap_rel, out=self._result.info.duality_gap_rel)

                # ------------ update non-regulerized residuals ------------
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
                self._result.info.prev_primal_res[:] = self._result.info.primal_res
                self._result.info.prev_dual_res[:] = self._result.info.dual_res

                self._result.info.primal_res[:] = self._primal_res_nr()

                # primal_rel_norm: update running max (initialized outside graph with ||A*x||_inf)
                # _work_z_1 is free at this point (only used before graph for cuSPARSE input)
                if self._data.num_hu > 0:
                    cp.take(G_x, self._data.idx_hu, out=self._work_z_1[:self._data.num_hu])
                    cp.absolute(self._work_z_1[:self._data.num_hu], out=self._work_z_1[:self._data.num_hu])
                    cp.max(self._work_z_1[:self._data.num_hu], out=self._work_norm_temp, keepdims=True)
                    cp.maximum(self._work_primal_rel_norm, self._work_norm_temp, out=self._work_primal_rel_norm)

                if self._data.num_hl > 0:
                    cp.take(G_x, self._data.idx_hl, out=self._work_z_1[:self._data.num_hl])
                    cp.absolute(self._work_z_1[:self._data.num_hl], out=self._work_z_1[:self._data.num_hl])
                    cp.max(self._work_z_1[:self._data.num_hl], out=self._work_norm_temp, keepdims=True)
                    cp.maximum(self._work_primal_rel_norm, self._work_norm_temp, out=self._work_primal_rel_norm)

                if self._data.num_hu > 0:
                    cp.absolute(self._result.s_u, out=self._work_z_1[:self._data.num_hu])
                    cp.max(self._work_z_1[:self._data.num_hu], out=self._work_norm_temp, keepdims=True)
                    cp.maximum(self._work_primal_rel_norm, self._work_norm_temp, out=self._work_primal_rel_norm)

                if self._data.num_hl > 0:
                    cp.absolute(self._result.s_l, out=self._work_z_1[:self._data.num_hl])
                    cp.max(self._work_z_1[:self._data.num_hl], out=self._work_norm_temp, keepdims=True)
                    cp.maximum(self._work_primal_rel_norm, self._work_norm_temp, out=self._work_primal_rel_norm)

                if self._data.num_xu > 0:
                    cp.absolute(self._result.s_bu, out=self._work_z[:self._data.num_xu])
                    cp.max(self._work_z[:self._data.num_xu], out=self._work_norm_temp, keepdims=True)
                    cp.maximum(self._work_primal_rel_norm, self._work_norm_temp, out=self._work_primal_rel_norm)

                if self._data.num_xl > 0:
                    cp.absolute(self._result.s_bl, out=self._work_z[:self._data.num_xl])
                    cp.max(self._work_z[:self._data.num_xl], out=self._work_norm_temp, keepdims=True)
                    cp.maximum(self._work_primal_rel_norm, self._work_norm_temp, out=self._work_primal_rel_norm)

                cp.maximum(self._work_primal_rel_norm, self._data._constraints_rhs_inf_norm, out=self._work_primal_rel_norm)
                # Store max(1, primal_rel_norm) for use by _update_residuals_r
                cp.maximum(self._work_primal_rel_norm, 1., out=self._work_primal_rel_norm)
                cp.divide(self._result.info.primal_res, self._work_primal_rel_norm, out=self._result.info.primal_res_rel)

                # dual_res_norm: update running max (initialized outside graph with ||P*x||_inf)
                self._result.info.dual_res[:] = self._dual_res_nr()

                # ||c||_inf
                cp.absolute(self._data.c, out=self._work_primals)
                cp.max(self._work_primals, out=self._work_norm_temp, keepdims=True)
                cp.maximum(self._work_dual_res_norm, self._work_norm_temp, out=self._work_dual_res_norm)

                # ||A^T*y + G^T*(z_u - z_l) + z_bu - z_bl||_inf
                self._res.x += GT_zu_minus_zl
                self._res.x[self._data.idx_xl] -= self._result.z_bl
                self._res.x[self._data.idx_xu] += self._result.z_bu
                cp.absolute(self._res.x, out=self._work_primals)
                cp.max(self._work_primals, out=self._work_norm_temp, keepdims=True)
                cp.maximum(self._work_dual_res_norm, self._work_norm_temp, out=self._work_dual_res_norm)

                # store max(1, dual_res_norm) for use by _update_residuals_r
                cp.maximum(self._work_dual_res_norm, 1., out=self._work_dual_res_norm)
                cp.divide(self._result.info.dual_res, self._work_dual_res_norm, out=self._result.info.dual_res_rel)

            self._update_residuals_nr_cuda_graphs[key] = stream.end_capture()

        self._update_residuals_nr_cuda_graphs[key].launch()


    @nvtx.annotate("Solver::_update_residuals_r")
    def _update_residuals_r(self):
        """
        Compute the regularized primal and dual residuals. The computation is based on the non-regularized residuals computed in _update_residuals_nr.
        It adds the regularization terms to the non-regularized residuals to obtain the regularized residuals.
        """

        if not hasattr(self, '_update_residuals_r_cuda_graphs'):
            self._update_residuals_r_cuda_graphs = {}
            self._update_residuals_r_cuda_graphs_capture_count = 0

        key = (self._res.buffer_ptr, 
               self._result.buffer_ptr,
               self._prox_vars.buffer_ptr,
               self._res_nr.buffer_ptr)
        
        if key not in self._update_residuals_r_cuda_graphs:
            self._update_residuals_r_cuda_graphs_capture_count += 1
            # print(f"Solver::_update_residuals_r capturing CUDA graph (occurrence {self._update_residuals_r_cuda_graphs_capture_count})...")
            # !IMPORTANT! Use the same stream for warp and cupy to make sure the captured graph contains both the warp kernel and the cupy operations, and they can be executed together without unnecessary stream synchronization
            stream = cp.cuda.ExternalStream(wp.get_stream().cuda_stream)
            stream.begin_capture()
            with stream:

                # update the rhs of the KKT system
                # self._res.x[:] = self._res_nr.x - self._result.info.rho * (self._result.x - self._prox_vars.x)
                # self._res.y[:] = self._res_nr.y - self._result.info.delta * (self._prox_vars.y - self._result.y)
                # self._res.z_l[:] = self._res_nr.z_l - self._result.info.delta * (self._prox_vars.z_l - self._result.z_l)
                # self._res.z_u[:] = self._res_nr.z_u - self._result.info.delta * (self._prox_vars.z_u - self._result.z_u)
                # self._res.z_bl[:] = self._res_nr.z_bl - self._result.info.delta * (self._prox_vars.z_bl - self._result.z_bl)
                # self._res.z_bu[:] = self._res_nr.z_bu - self._result.info.delta * (self._prox_vars.z_bu - self._result.z_bu)
                USE_WARP_IMPLEMENTATION = True
                if USE_WARP_IMPLEMENTATION:
                    wp.launch(
                        kernel=self._update_residuals_r_kernel,
                        dim=self._data.n + self._data.p + self._data.num_hl + self._data.num_hu + self._data.num_xl + self._data.num_xu,
                        inputs=[
                            self._result.info.rho, self._result.info.delta,
                            self._res_nr.x, self._res_nr.y, self._res_nr.z_l, self._res_nr.z_u, self._res_nr.z_bl, self._res_nr.z_bu,
                            self._result.x, self._result.y, self._result.z_l, self._result.z_u, self._result.z_bl, self._result.z_bu,
                            self._prox_vars.x, self._prox_vars.y, self._prox_vars.z_l, self._prox_vars.z_u, self._prox_vars.z_bl, self._prox_vars.z_bu,
                            self._res.x, self._res.y, self._res.z_l, self._res.z_u, self._res.z_bl, self._res.z_bu
                        ],
                        device="cuda",
                    )
                else:
                    cp.subtract(self._result.x, self._prox_vars.x, out=self._res.x)
                    self._res.x *= self._result.info.rho
                    cp.subtract(self._res_nr.x, self._res.x, out=self._res.x)
                    cp.subtract(self._prox_vars.y, self._result.y, out=self._res.y)
                    self._res.y *= self._result.info.delta
                    cp.subtract(self._res_nr.y, self._res.y, out=self._res.y)
                    cp.subtract(self._prox_vars.z_l, self._result.z_l, out=self._res.z_l)
                    self._res.z_l *= self._result.info.delta
                    cp.subtract(self._res_nr.z_l, self._res.z_l, out=self._res.z_l)
                    cp.subtract(self._prox_vars.z_u, self._result.z_u, out=self._res.z_u)
                    self._res.z_u *= self._result.info.delta
                    cp.subtract(self._res_nr.z_u, self._res.z_u, out=self._res.z_u)
                    cp.subtract(self._prox_vars.z_bl, self._result.z_bl, out=self._res.z_bl)
                    self._res.z_bl *= self._result.info.delta
                    cp.subtract(self._res_nr.z_bl, self._res.z_bl, out=self._res.z_bl)
                    cp.subtract(self._prox_vars.z_bu, self._result.z_bu, out=self._res.z_bu)
                    self._res.z_bu *= self._result.info.delta
                    cp.subtract(self._res_nr.z_bu, self._res.z_bu, out=self._res.z_bu)

                self._result.info.primal_res_reg[:] = self._primal_res_r()
                # primal_rel_scaling = self._result.info.primal_res / self._result.info.primal_res_rel if self._result.info.primal_res_rel > 0 else 1.
                # self._result.info.primal_res_reg_rel[:] = self._result.info.primal_res_reg / primal_rel_scaling
                cp.divide(self._result.info.primal_res, self._result.info.primal_res_rel, out=self._result.info.primal_res_reg_rel)
                self._result.info.primal_res_reg_rel[:] = cp.where(self._result.info.primal_res_rel > 0,
                        self._result.info.primal_res_reg_rel, cp.asarray(1.0, dtype=self._result.info.primal_res_reg_rel.dtype))
                cp.divide(self._result.info.primal_res_reg, self._result.info.primal_res_reg_rel, out=self._result.info.primal_res_reg_rel)

                self._result.info.dual_res_reg[:] = self._dual_res_r()
                # dual_rel_scaling = self._result.info.dual_res / self._result.info.dual_res_rel if self._result.info.dual_res_rel > 0 else 1.
                # self._result.info.dual_res_reg_rel[:] = self._result.info.dual_res_reg / dual_rel_scaling
                cp.divide(self._result.info.dual_res, self._result.info.dual_res_rel, out=self._result.info.dual_res_reg_rel)
                self._result.info.dual_res_reg_rel[:] = cp.where(self._result.info.dual_res_rel > 0,
                        self._result.info.dual_res_reg_rel, cp.asarray(1.0, dtype=self._result.info.dual_res_reg_rel.dtype))
                cp.divide(self._result.info.dual_res_reg, self._result.info.dual_res_reg_rel, out=self._result.info.dual_res_reg_rel)

                self._result.info.primal_prox_inf[:] = self._primal_prox_inf()
                self._result.info.primal_prox_inf *= self._result.info.delta
                self._result.info.dual_prox_inf[:] = self._dual_prox_inf()
                self._result.info.dual_prox_inf *= self._result.info.rho
        
            self._update_residuals_r_cuda_graphs[key] = stream.end_capture()

        self._update_residuals_r_cuda_graphs[key].launch()
        

    @nvtx.annotate("Solver::_primal_res_nr")
    def _primal_res_nr(self) -> float:
        offset = 0
        self._work_duals[:self._data.p] = self._res_nr.y
        offset += self._data.p
        self._work_duals[offset : offset+self._data.num_hu] = self._res_nr.z_u
        offset += self._data.num_hu
        self._work_duals[offset : offset+self._data.num_hl] = self._res_nr.z_l
        offset += self._data.num_hl
        self._work_duals[offset : offset+self._data.num_xu] = self._res_nr.z_bu
        offset += self._data.num_xu
        self._work_duals[offset : offset+self._data.num_xl] = self._res_nr.z_bl
        offset += self._data.num_xl
        # inf-norm without allocating a scalar output
        cp.absolute(self._work_duals[:offset], out=self._work_duals[:offset])
        cp.max(self._work_duals[:offset], out=self._work_residual)
        return self._work_residual

    @nvtx.annotate("Solver::_primal_res_r")
    def _primal_res_r(self) -> float:
        offset = 0
        self._work_duals[:self._data.p] = self._res.y
        offset = self._data.p
        self._work_duals[offset : offset+self._data.num_hu] = self._res.z_u
        offset += self._data.num_hu
        self._work_duals[offset : offset+self._data.num_hl] = self._res.z_l
        offset += self._data.num_hl
        self._work_duals[offset : offset+self._data.num_xu] = self._res.z_bu
        offset += self._data.num_xu
        self._work_duals[offset : offset+self._data.num_xl] = self._res.z_bl
        offset += self._data.num_xl
        cp.absolute(self._work_duals[:offset], out=self._work_duals[:offset])
        cp.max(self._work_duals[:offset], out=self._work_residual)
        return self._work_residual
    
    @nvtx.annotate("Solver::_dual_res_nr")
    def _dual_res_nr(self) -> float:
        cp.absolute(self._res_nr.x, out=self._work_primals)
        cp.max(self._work_primals, out=self._work_residual)
        return self._work_residual
    
    @nvtx.annotate("Solver::_dual_res_r")
    def _dual_res_r(self) -> float:
        cp.absolute(self._res.x, out=self._work_primals)
        cp.max(self._work_primals, out=self._work_residual)
        return self._work_residual
    
    @nvtx.annotate("Solver::_primal_prox_inf")
    def _primal_prox_inf(self) -> float:
        cp.subtract(self._result.y, self._prox_vars.y, out=self._work_duals[:self._data.p])
        offset = self._data.p
        cp.subtract(self._result.z_l, self._prox_vars.z_l, out=self._work_duals[offset : offset+self._data.num_hl])
        offset += self._data.num_hl
        cp.subtract(self._result.z_u, self._prox_vars.z_u, out=self._work_duals[offset : offset+self._data.num_hu])
        offset += self._data.num_hu
        cp.subtract(self._result.z_bl, self._prox_vars.z_bl, out=self._work_duals[offset : offset+self._data.num_xl])
        offset += self._data.num_xl
        cp.subtract(self._result.z_bu, self._prox_vars.z_bu, out=self._work_duals[offset : offset+self._data.num_xu])
        offset += self._data.num_xu
        cp.absolute(self._work_duals[:offset], out=self._work_duals[:offset])
        cp.max(self._work_duals[:offset], out=self._work_residual)
        return self._work_residual

    @nvtx.annotate("Solver::_dual_prox_inf")
    def _dual_prox_inf(self) -> float:
        cp.subtract(self._result.x, self._prox_vars.x, out=self._work_primals)
        cp.absolute(self._work_primals, out=self._work_primals)
        cp.max(self._work_primals, out=self._work_residual)
        return self._work_residual
    
    @nvtx.annotate("Solver::_update_rho_delta_with_ineq")
    def _update_rho_delta_with_ineq(self) -> None:
        """Update rho/delta based on residual progress (host-side branching)."""
        if self._result.info.dual_res < 0.95 * self._result.info.prev_dual_res or \
            (self._result.info.dual_res < self.settings.eps_abs or self._result.info.dual_res_rel < self.settings.eps_rel) or \
            (self._result.info.rho == self.settings.reg_finetune_lower_limit and self._result.info.dual_prox_inf < self.settings.infeasibility_threshold):
            self._prox_vars.x[:] = self._result.x
            self._result.info.rho[:] = cp.maximum(self._result.info.reg_limit, (1. - self._mu_rate) * self._result.info.rho)
        else:
            self._result.info.no_primal_update += 1
            if self._result.info.iter < 5 or self._result.info.dual_prox_inf < self.settings.infeasibility_threshold:
                self._result.info.rho[:] = cp.maximum(self._result.info.reg_limit, (1. - 0.666 * self._mu_rate) * self._result.info.rho)

        if self._result.info.primal_res < 0.95 * self._result.info.prev_primal_res or \
            (self._result.info.primal_res < self.settings.eps_abs or self._result.info.primal_res_rel < self.settings.eps_rel) or \
            (self._result.info.delta == self.settings.reg_finetune_lower_limit and self._result.info.primal_prox_inf < self.settings.infeasibility_threshold):
            self._prox_vars.y[:] = self._result.y
            self._prox_vars.z_l[:] = self._result.z_l
            self._prox_vars.z_u[:] = self._result.z_u
            self._prox_vars.z_bu[:] = self._result.z_bu
            self._prox_vars.z_bl[:] = self._result.z_bl
            self._result.info.delta[:] = cp.maximum(self._result.info.reg_limit, (1. - self._mu_rate) * self._result.info.delta)
        else:
            self._result.info.no_dual_update += 1
            if self._result.info.iter < 5 or self._result.info.primal_prox_inf < self.settings.infeasibility_threshold:
                self._result.info.delta[:] = cp.maximum(self._result.info.reg_limit, (1. - 0.666 * self._mu_rate) * self._result.info.delta)

    @nvtx.annotate("Solver::_update_rho_delta_without_ineq")
    def _update_rho_delta_without_ineq(self) -> None:
        """Update rho/delta based on residual progress (host-side branching)."""
        if self._result.info.dual_res < 0.95 * self._result.info.prev_dual_res or \
            self._result.info.dual_res < self.settings.eps_abs or \
                self._result.info.dual_res_rel < self.settings.eps_rel:                
            self._prox_vars.x[:] = self._result.x
            self._result.info.rho[:] = cp.maximum(self._result.info.reg_limit, 0.1 * self._result.info.rho)                
        else:                
            self._result.info.no_primal_update += 1
            if self._result.info.iter < 5 or self._result.info.dual_prox_inf < self.settings.infeasibility_threshold:
                self._result.info.rho[:] = cp.maximum(self._result.info.reg_limit, 0.5 * self._result.info.rho)
                    
        if self._result.info.primal_res < 0.95 * self._result.info.prev_primal_res or \
            self._result.info.primal_res < self.settings.eps_abs or \
                self._result.info.primal_res_rel < self.settings.eps_rel:
            self._prox_vars.y[:] = self._result.y
            self._result.info.delta[:] = cp.maximum(self._result.info.reg_limit, 0.1 * self._result.info.delta)
        else:
            self._result.info.no_dual_update += 1
            if self._result.info.iter < 5 or self._result.info.primal_prox_inf < self.settings.infeasibility_threshold:
                self._result.info.delta[:] = cp.maximum(self._result.info.reg_limit, 0.5 * self._result.info.delta)



def create_update_residual_r_kernel(n: int, p: int, num_hl: int, num_hu: int, num_xl: int, num_xu: int):
    """
    Perform the following operations:
        self._res.x[:] = self._res_nr.x - self._result.info.rho * (self._result.x - self._prox_vars.x)
        self._res.y[:] = self._res_nr.y - self._result.info.delta * (self._prox_vars.y - self._result.y)
        self._res.z_l[:] = self._res_nr.z_l - self._result.info.delta * (self._prox_vars.z_l - self._result.z_l)
        self._res.z_u[:] = self._res_nr.z_u - self._result.info.delta * (self._prox_vars.z_u - self._result.z_u)
        self._res.z_bl[:] = self._res_nr.z_bl - self._result.info.delta * (self._prox_vars.z_bl - self._result.z_bl)
        self._res.z_bu[:] = self._res_nr.z_bu - self._result.info.delta * (self._prox_vars.z_bu - self._result.z_bu)
    """
    @wp.kernel
    def update_residual_r_kernel(
        rho: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        delta: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        res_nr_x: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        res_nr_y: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        res_nr_z_l: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        res_nr_z_u: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        res_nr_z_bl: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        res_nr_z_bu: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        result_x: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        result_y: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        result_z_l: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        result_z_u: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        result_z_bl: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        result_z_bu: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        prox_var_x: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        prox_var_y: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        prox_var_z_l: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        prox_var_z_u: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        prox_var_z_bl: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        prox_var_z_bu: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        res_r_x: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        res_r_y: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        res_r_z_l: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        res_r_z_u: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        res_r_z_bl: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        res_r_z_bu: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
    ):
        t = wp.tid()
        n_static = wp.static(n)
        p_static = wp.static(p)
        num_hl_static = wp.static(num_hl)
        num_hu_static = wp.static(num_hu)
        num_xl_static = wp.static(num_xl)
        num_xu_static = wp.static(num_xu)

        if t < n_static:
            res_r_x[t] = -rho[0] * (result_x[t] - prox_var_x[t]) + res_nr_x[t]
        elif t < n_static + p_static:
            idx = t - n_static
            res_r_y[idx] = delta[0] * (result_y[idx] - prox_var_y[idx]) + res_nr_y[idx]
        elif t < n_static + p_static + num_hl_static:
            idx = t - n_static - p_static
            res_r_z_l[idx] = delta[0] * (result_z_l[idx] - prox_var_z_l[idx]) + res_nr_z_l[idx]
        elif t < n_static + p_static + num_hl_static + num_hu_static:
            idx = t - n_static - p_static - num_hl_static
            res_r_z_u[idx] = delta[0] * (result_z_u[idx] - prox_var_z_u[idx]) + res_nr_z_u[idx]
        elif t < n_static + p_static + num_hl_static + num_hu_static + num_xl_static:
            idx = t - n_static - p_static - num_hl_static - num_hu_static
            res_r_z_bl[idx] = delta[0] * (result_z_bl[idx] - prox_var_z_bl[idx]) + res_nr_z_bl[idx]
        elif t < n_static + p_static + num_hl_static + num_hu_static + num_xl_static + num_xu_static:
            idx = t - n_static - p_static - num_hl_static - num_hu_static - num_xl_static
            res_r_z_bu[idx] = delta[0] * (result_z_bu[idx] - prox_var_z_bu[idx]) + res_nr_z_bu[idx]
        else:
            return

    return update_residual_r_kernel
