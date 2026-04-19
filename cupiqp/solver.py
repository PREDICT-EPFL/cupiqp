import numpy as np
import cupy as cp
import warp as wp
from typing import Optional, Any
import nvtx

from .settings import Settings
from .data import Data
from .results import Result, Status, Variables, InfoHost
from .kkt_systems import KKTSystem
from .preconditioner import RuizEquilibration
from .utils import cuda_graph_capture


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
        self._preconditioner = None

    @property
    def result(self):
        return self._result
    
    @nvtx.annotate("Solver::setup")
    def setup(self, P, c, A=None, b=None, G=None, h_u=None, h_l=None, x_u=None, x_l=None):
        # Detect if user provided batched (3D P) or non-batched (2D P) data.
        # DenseData auto-unsqueezes non-batched to (1, ...) internally,
        # but we track this so solve() returns the right type.
        self._user_batched = (hasattr(P, 'ndim') and P.ndim == 3) or (isinstance(P, (list, tuple)) and len(P) > 1)

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

        # Apply Ruiz equilibration preconditioner
        if self.settings.preconditioner_iter > 0:
            if self.settings.kkt_solver == "dense_cholesky":
                from .dense.dense_preconditioner import DenseRuizEquilibration
                PreconditionerClass = DenseRuizEquilibration
            elif self.settings.kkt_solver == "sparse_ldlt":
                from .sparse.sparse_preconditioner import SparseRuizEquilibration
                PreconditionerClass = SparseRuizEquilibration
            elif self.settings.kkt_solver == "multistage_block_cholesky":
                from .multistage.multistage_preconditioner import MultistageRuizEquilibration
                PreconditionerClass = MultistageRuizEquilibration
            else:
                raise ValueError(f"No preconditioner for kkt_solver type: {self.settings.kkt_solver}")
            self._preconditioner = PreconditionerClass(
                self._data.batch_size, self._data.n, self._data.p, self._data.m,
                self._data.idx_xl, self._data.idx_xu,
            )
            self._preconditioner.scale_data(
                self._data,
                self.settings.preconditioner_scale_cost,
                self.settings.preconditioner_iter,
            )
            self._data._compute_constraints_rhs_inf_norm()
        else:
            self._preconditioner = None

        data = self._data
        B = data.batch_size

        self._result = Result(B)
        self._result.init(self._data)
        self._result.info.rho[:] = self.settings.rho_init
        self._result.info.delta[:] = self.settings.delta_init

        self._step.init(self._data)
        self._res_nr.init(self._data)
        self._res.init(self._data)
        self._prox_vars.init(self._data)

        self._kkt_system.init(self._data, self.settings)
        self._info_host = InfoHost(B)

        self._work_z_1 = cp.empty((data.batch_size, data.m))  # used to store intermediate results in _update_residuals_nr
        self._work_z_2 = cp.empty((data.batch_size, data.m))  # used to store intermediate results in _update_residuals_nr

        self._work_z = cp.empty((data.batch_size, data.num_ineq))  # used in _calculate_step to hold all concatenated slack or dual steps / results
        self._work_s = cp.empty((data.batch_size, data.num_ineq))  # used in _calculate_step to hold all concatenated slack or dual steps / results
        self._work_primals = cp.empty((data.batch_size, data.n))
        self._work_duals = cp.empty((data.batch_size, data.p + data.num_ineq))  # used to hold the concatenated dual variables for computing the residuals in _update_residuals_nr
        self._work_residual = cp.empty((data.batch_size, ))
        self._work_reduce = cp.empty((data.batch_size, 8))  # used to hold the intermediate results of the reductions related to s_l, s_u, s_bl, s_bu and z_l, z_u, z_bl, z_bu
        
        self._update_residuals_r_kernel = create_update_residual_r_kernel(
            self._data.n, self._data.p, self._data.num_hl+self._data.num_hu+self._data.num_xl+self._data.num_xu  # only n is used by the kernel; others kept for API compatibility
        )
        # Pre-allocated (B,) buffers for CUDA-graph-safe norm computations in _update_residuals_nr / _update_residuals_r
        self._work_primal_rel_norm = cp.empty(B, dtype=cp.float64)  # running max of primal relative norm terms
        self._work_dual_res_norm = cp.empty(B, dtype=cp.float64)    # running max of dual residual norm terms
        self._work_norm_temp = cp.empty(B, dtype=cp.float64)        # temp (B,) for individual norm results

        self._mu_prev = cp.empty((data.batch_size, 1))
        self._mu_rate = cp.empty((data.batch_size, 1))

        self._enable_iterative_refinement = self.settings.iterative_refinement_always_enabled

        # Pre-compute residual unscaling factors for preconditioner.
        # These are cached arrays used every iteration in _update_residuals_nr
        # to compute norms in the original (unscaled) coordinate system.
        # When no preconditioner is active, all factors are 1.0 (identity).
        n, p, m = self._data.n, self._data.p, self._data.m
        pc = self._preconditioner
        if pc is not None:
            self._c_scaling_inv = pc.c_scaling_inv                          # (B,)
            self._unscale_dual_res_factor = pc.c_scaling_inv[:, None] * pc.delta_inv[:, :n]  # (B, n)
            self._unscale_primal_res_eq_factor = pc.delta_inv[:, n:n + p] if p > 0 else cp.empty((B, 0))  # (B, p)
            self._unscale_primal_res_ineq_hu = pc.delta_inv[:, n + p + self._data.idx_hu] if self._data.num_hu > 0 else cp.empty((B, 0))  # (B, num_hu)
            self._unscale_primal_res_ineq_hl = pc.delta_inv[:, n + p + self._data.idx_hl] if self._data.num_hl > 0 else cp.empty((B, 0))  # (B, num_hl)
            self._unscale_primal_res_b_xu = pc.delta_b_inv[:, self._data.idx_xu] if self._data.num_xu > 0 else cp.empty((B, 0))  # (B, num_xu)
            self._unscale_primal_res_b_xl = pc.delta_b_inv[:, self._data.idx_xl] if self._data.num_xl > 0 else cp.empty((B, 0))  # (B, num_xl)
            # Compute unscaled constraints RHS norm from original (pre-scaling) bounds — (B,)
            self._constraints_rhs_inf_norm_unscaled = cp.zeros(B, dtype=cp.float64)
            if p > 0:
                cp.maximum(self._constraints_rhs_inf_norm_unscaled,
                           cp.max(cp.abs(pc.unscale_primal_res_eq(self._data.b)), axis=1),
                           out=self._constraints_rhs_inf_norm_unscaled)
            if self._data.num_hu > 0:
                cp.maximum(self._constraints_rhs_inf_norm_unscaled,
                           cp.max(cp.abs(pc.unscale_primal_res_ineq(self._data.h_u[:, self._data.idx_hu], self._data.idx_hu)), axis=1),
                           out=self._constraints_rhs_inf_norm_unscaled)
            if self._data.num_hl > 0:
                cp.maximum(self._constraints_rhs_inf_norm_unscaled,
                           cp.max(cp.abs(pc.unscale_primal_res_ineq(self._data.h_l[:, self._data.idx_hl], self._data.idx_hl)), axis=1),
                           out=self._constraints_rhs_inf_norm_unscaled)
            if self._data.num_xu > 0:
                cp.maximum(self._constraints_rhs_inf_norm_unscaled,
                           cp.max(cp.abs(pc.unscale_primal_res_b(self._data.x_u[:, self._data.idx_xu], self._data.idx_xu)), axis=1),
                           out=self._constraints_rhs_inf_norm_unscaled)
            if self._data.num_xl > 0:
                cp.maximum(self._constraints_rhs_inf_norm_unscaled,
                           cp.max(cp.abs(pc.unscale_primal_res_b(self._data.x_l[:, self._data.idx_xl], self._data.idx_xl)), axis=1),
                           out=self._constraints_rhs_inf_norm_unscaled)
        else:
            self._c_scaling_inv = cp.ones(B, dtype=cp.float64)                                                                               # (B,)
            self._unscale_dual_res_factor = cp.ones((B, n), dtype=cp.float64)                                                                # (B, n)
            self._unscale_primal_res_eq_factor = cp.ones((B, p), dtype=cp.float64) if p > 0 else cp.empty((B, 0))                            # (B, p)
            self._unscale_primal_res_ineq_hu = cp.ones((B, self._data.num_hu), dtype=cp.float64) if self._data.num_hu > 0 else cp.empty((B, 0))  # (B, num_hu)
            self._unscale_primal_res_ineq_hl = cp.ones((B, self._data.num_hl), dtype=cp.float64) if self._data.num_hl > 0 else cp.empty((B, 0))  # (B, num_hl)
            self._unscale_primal_res_b_xu = cp.ones((B, self._data.num_xu), dtype=cp.float64) if self._data.num_xu > 0 else cp.empty((B, 0))    # (B, num_xu)
            self._unscale_primal_res_b_xl = cp.ones((B, self._data.num_xl), dtype=cp.float64) if self._data.num_xl > 0 else cp.empty((B, 0))    # (B, num_xl)
            self._constraints_rhs_inf_norm_unscaled = self._data._constraints_rhs_inf_norm                                                   # (B,)

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

        # Apply preconditioner scaling to updated data
        if self._preconditioner is not None:
            if self.settings.preconditioner_reuse_on_update:
                self._preconditioner.apply_scaling(self._data)
            else:
                self._preconditioner.reset()
                self._preconditioner.scale_data(
                    self._data,
                    self.settings.preconditioner_scale_cost,
                    self.settings.preconditioner_iter,
                )

        self._data._compute_constraints_rhs_inf_norm()
        self._kkt_system.update_data(self._data, P is not None, A is not None, G is not None)

    def solve(self) -> list:
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
        self._result.info._status_value[:] = Status.PIQP_UNSOLVED.value
        self._result.info.iter[:] = 0
        self._result.info.reg_limit[:] = self.settings.reg_lower_limit
        self._result.info.factor_retires[:] = 0
        self._result.info.no_primal_update[:] = 0
        self._result.info.no_dual_update[:] = 0
        self._result.info.mu[:] = 0.
        self._result.info.primal_step[:] = 0.
        self._result.info.dual_step[:] = 0.
        self._result.info.rho[:] = self.settings.rho_init
        self._result.info.delta[:] = self.settings.delta_init

        if self.settings.verbose:
            print("iter  prim_obj       dual_obj       duality_gap   prim_res      dual_res      rho         delta       mu          p_step   d_step")

        ## ----------- initial iteration --------------
        self._initial_guess()

        ## ---------------------------------------------
        ## ---------- remaining iterations -------------
        ## ---------------------------------------------
        for iter in range(self.settings.max_iter):
            with nvtx.annotate(f"Solver::ipm_iteration"):
                self._result.info.iter[:] = iter
                if iter == 0:
                    self._update_residuals_nr()
                    self._result.info.prev_primal_res[:] = self._result.info.primal_res
                    self._result.info.prev_dual_res[:] = self._result.info.dual_res

                self._update_residuals_r()

                # fetch all info to host all at once, at the cost of one D2H memcpy
                self._result.info.to_host(self._info_host)  # CPU: numpy (B, num_fields) buffer
                info_host = self._info_host 

                # ============================================================
                # Per-problem termination check — ALL ON CPU (host-side numpy)
                # h = info_host (numpy mirror), status/no_*_update are numpy arrays.
                # Vectorized over batch: no Python loops, just numpy boolean ops.
                # All problems keep running until every one has terminated.
                # ============================================================
                settings = self.settings
                still_unsolved = (self._result.info._status_value == Status.PIQP_UNSOLVED.value)  # CPU: numpy bool (B,)

                # convergence check
                primal_ok = (info_host.primal_res < settings.eps_abs) | (info_host.primal_res_rel < settings.eps_rel)
                dual_ok = (info_host.dual_res < settings.eps_abs) | (info_host.dual_res_rel < settings.eps_rel)
                converged = primal_ok & dual_ok
                if settings.check_duality_gap:
                    gap_ok = (info_host.duality_gap < settings.eps_duality_gap_abs) | (info_host.duality_gap_rel < settings.eps_duality_gap_rel)
                    converged &= gap_ok
                self._result.info._status_value[still_unsolved & converged] = Status.PIQP_SOLVED.value  # CPU write

                # primal infeasibility check
                primal_infeasible = (
                    (self._result.info.no_dual_update > min(5, settings.reg_finetune_dual_update_threshold)) &
                    (info_host.primal_prox_inf > settings.infeasibility_threshold) &
                    ((info_host.primal_res_reg < settings.eps_abs) | (info_host.primal_res_reg_rel < settings.eps_rel))
                )
                self._result.info._status_value[still_unsolved & ~converged & primal_infeasible] = Status.PIQP_PRIMAL_INFEASIBLE.value  # CPU write

                # dual infeasibility check
                dual_infeasible = (
                    (self._result.info.no_primal_update > min(5, settings.reg_finetune_primal_update_threshold)) &
                    (info_host.dual_prox_inf > settings.infeasibility_threshold) &
                    ((info_host.dual_res_reg < settings.eps_abs) | (info_host.dual_res_reg_rel < settings.eps_rel))
                )
                self._result.info._status_value[still_unsolved & ~converged & ~primal_infeasible & dual_infeasible] = Status.PIQP_DUAL_INFEASIBLE.value  # CPU write

                # exit if all problems have terminated
                if np.all(self._result.info._status_value != Status.PIQP_UNSOLVED.value):
                    break
                
                # avoid getting too close to boundary which can result in a division by zero
                epsilon = float(cp.finfo(cp.float64).eps)
                boundary_shifted = False
                if self._data.num_hl > 0 and bool(cp.any(self._result.z_l < epsilon)):
                    self._result.z_l += (self._result.z_l < epsilon) * epsilon
                    boundary_shifted = True
                if self._data.num_hu > 0 and bool(cp.any(self._result.z_u < epsilon)):
                    self._result.z_u += (self._result.z_u < epsilon) * epsilon
                    boundary_shifted = True
                if self._data.num_xl > 0 and bool(cp.min(self._result.z_bl) < epsilon):
                    self._result.z_bl += epsilon
                    boundary_shifted = True
                if self._data.num_xu > 0 and bool(cp.min(self._result.z_bu) < epsilon):
                    self._result.z_bu += epsilon
                    boundary_shifted = True
                if boundary_shifted:
                    print("Boundary shifted to avoid division by zero")
                    self._calculate_mu()
                
                # avoid possibility of converging to a local minimum -> decrease the minimum regularization value (vectorized)
                finetune_mask = (
                    ((self._result.info.no_primal_update > self.settings.reg_finetune_primal_update_threshold) &
                     (info_host.rho == info_host.reg_limit) &
                     (info_host.reg_limit != self.settings.reg_finetune_lower_limit)) |
                    ((self._result.info.no_dual_update > self.settings.reg_finetune_dual_update_threshold) &
                     (info_host.delta == info_host.reg_limit) &
                     (info_host.reg_limit != self.settings.reg_finetune_lower_limit))
                )
                finetune_mask &= (info_host.dual_prox_inf < self.settings.infeasibility_threshold) & (info_host.primal_prox_inf < self.settings.infeasibility_threshold)
                if np.any(finetune_mask):
                    self._result.info.reg_limit[finetune_mask] = self.settings.reg_finetune_lower_limit
                    self._result.info.no_primal_update[finetune_mask] = 0
                    self._result.info.no_dual_update[finetune_mask] = 0

                if self.settings.verbose:
                    self._print_iteration_info()

                self._update_and_factorize_kkt()
                if np.any(self._result.info._status_value == Status.PIQP_NUMERICAL_ISSUES.value):
                    break

                if self._data.num_hl + self._data.num_hu + self._data.num_xl + self._data.num_xu == 0:
                    # since there are no inequalities we can take full Newton steps
                    self._run_full_newton_step()
                    self._update_residuals_nr()
                    self._update_rho_delta_without_ineq()
                else:
                    self._run_predictor_corrector()
                    self._update_residuals_nr()
                    self._update_rho_delta_with_ineq()

        # Mark remaining unsolved as max iter reached
        self._result.info._status_value[self._result.info._status_value == Status.PIQP_UNSOLVED.value] = Status.PIQP_MAX_ITER_REACHED.value
        if self._preconditioner is not None:
            self._preconditioner.unscale_solution(self._result, self._data)
        statuses = self._result.info.status
        if self._user_batched:
            return statuses
        else:
            return statuses[0]

    @nvtx.annotate("Solver::_initial_guess")
    def _initial_guess(self):
        # eq(12) in Roland Schwan 2023 paper
        self._result.x.fill(0.0)
        self._result.y.fill(0.0)
        self._result.s_all.fill(1.0)
        self._result.z_all.fill(1.0)

        self._kkt_system.update_scalings_and_factor(
            self._data,
            self.settings,
            self._enable_iterative_refinement,
            self._result.info.rho,
            self._result.info.delta,
            self._result
        )

        self._res.x[:] = -self._data.c
        self._res.y[:] = self._data.b
        if self._data.num_hl > 0:
            self._res.z_l[:] = -self._data.h_l[:, self._data.idx_hl]
        if self._data.num_hu > 0:
            self._res.z_u[:] = self._data.h_u[:, self._data.idx_hu]
        if self._data.num_xl > 0:
            self._res.z_bl[:] = -self._data.x_l[:, self._data.idx_xl]
        if self._data.num_xu > 0:
            self._res.z_bu[:] = self._data.x_u[:, self._data.idx_xu]
        self._res.s_all[:] = 0.

        self._kkt_system.solve(self._data, self.settings, self._res, self._result)  # getting an initial point of _result

        if self.settings.debug:
            print("Initial point after solving KKT system:", self._result)

        if self._data.num_hl + self._data.num_hu + self._data.num_xl + self._data.num_xu > 0:
            ## ----------- keep z and s non-negative --------------
            # this is according to the IV.A part of Roland Schwan 2023 paper
            delta_s = -cp.min(self._result.s_all, axis=1, keepdims=True)  # (B, 1)
            delta_z = -cp.min(self._result.z_all, axis=1, keepdims=True)  # (B, 1)
            self._result.s_all += delta_s
            self._result.z_all += delta_z

            # need to make sure mu is positive here, otherwise in the next step (put s and z on central path) sqrt(mu) the computed z_* will be zeros
            self._calculate_mu()
            cp.clip(self._result.info.mu, 1e-10, None, out=self._result.info.mu)

            if self.settings.debug:
                print("Initial mu:", self._result.info.mu)

            # put s and z on the central path
            # c = z - delta_z; z = (c + sqrt(c^2 + 4*mu)) / 2; s = z - c
            cp.subtract(self._result.z_all, delta_z, out=self._result.s_all)
            cp.power(self._result.s_all, 2, out=self._result.z_all)
            self._result.z_all += 4. * self._result.info.mu[:, None]
            cp.sqrt(self._result.z_all, out=self._result.z_all)
            self._result.z_all += self._result.s_all
            self._result.z_all /= 2.
            cp.subtract(self._result.z_all, self._result.s_all, out=self._result.s_all)

            if self.settings.debug:
                print("self._result:", self._result)

            self._calculate_mu()

        self._prox_vars.primals_all[:] = self._result.primals_all
        self._prox_vars.duals_all[:] = self._result.duals_all

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
        retries = 0
        while retries < self.settings.max_factor_retires:
            factor_succeeded = self._kkt_system.update_scalings_and_factor(
                self._data, self.settings, self._enable_iterative_refinement,
                self._result.info.rho, self._result.info.delta, self._result)
            if factor_succeeded:
                break
            else:
                if not self._enable_iterative_refinement:
                    self._enable_iterative_refinement = True
                retries += 1
                self._result.info.rho *= 100.
                self._result.info.delta *= 100.
                self._result.info.reg_limit[:] = cp.minimum(10 * self._result.info.reg_limit, self.settings.eps_abs)

        if retries >= self.settings.max_factor_retires:
            # Mark all still-unsolved problems as numerical issues
            still_unsolved = (self._result.info._status_value == Status.PIQP_UNSOLVED.value)
            self._result.info._status_value[still_unsolved] = Status.PIQP_NUMERICAL_ISSUES.value

    @nvtx.annotate("Solver::_run_full_newton_step")
    def _run_full_newton_step(self):
        self._kkt_system.solve(self._data, self.settings, self._res, self._step)
        self._result.info.primal_step[:] = 1.0
        self._result.info.dual_step[:] = 1.0
        self._result.x += self._result.info.primal_step[:, None] * self._step.x
        self._result.y += self._result.info.dual_step[:, None] * self._step.y

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
            cp.multiply(self._result.s_all, self._result.z_all, out=self._res.s_all)
            self._res.s_all *= -1.
            

        if self.settings.debug:
            print("predictor step rhs is: res= ", self._res)

        self._kkt_system.solve(self._data, self.settings, self._res, self._step)

        if self.settings.debug:
            print("predictor step is:", self._step)

        # step in the non-negative orthant
        self._calculate_step()

        # ------------------ compute centering parameter sigma ------------------
        self._calculate_sigma()

        # ------------------ corrector step ------------------
        with nvtx.annotate("Solver::prepare_corrector_step"):
            # self._res.s_l += -self._step.s_l * self._step.z_l + self._result.info.sigma * self._result.info.mu
            # self._res.s_u += -self._step.s_u * self._step.z_u + self._result.info.sigma * self._result.info.mu
            # self._res.s_bl += -self._step.s_bl * self._step.z_bl + self._result.info.sigma * self._result.info.mu
            # self._res.s_bu += -self._step.s_bu * self._step.z_bu + self._result.info.sigma * self._result.info.mu
            tmp_sigma_mu = self._result.info.sigma * self._result.info.mu
            cp.multiply(self._step.s_all, self._step.z_all, out=self._work_s)
            self._res.s_all -= self._work_s
            self._res.s_all += tmp_sigma_mu[:, None]

        if self.settings.debug:
            print("corrector step rhs is: res= ", self._res)
        self._kkt_system.solve(self._data, self.settings, self._res, self._step)

        if self.settings.debug:
            print("corrector step is:", self._step)

        # step in the non-negative orthant
        self._calculate_step()
        self._update_vars_after_corrector_step()
        self._calculate_mu()
        cp.subtract(self._mu_prev, self._result.info.mu[:, None], out=self._mu_rate)
        cp.divide(self._mu_rate, self._mu_prev, out=self._mu_rate)
        cp.maximum(self._mu_rate, 0., out=self._mu_rate)

    @cuda_graph_capture(enable=lambda self: self.settings.enable_cuda_graph)
    def _update_vars_after_corrector_step(self):
        # ------------------ update variables ------------------
        self._result.primals_all += self._result.info.primal_step[:, None] * self._step.primals_all
        self._result.duals_all += self._result.info.dual_step[:, None] * self._step.duals_all
        # ------------------ update mu and mu_rate for adaptive regularization ------------------
        self._mu_prev[:] = self._result.info.mu[:, None]

    @nvtx.annotate("Solver::_calculate_step")
    @cuda_graph_capture(enable=lambda self: self.settings.enable_cuda_graph)
    def _calculate_step(self) -> None:
        """
        Compute the step length of the slack variables and dual variables. Make sure they remain non-negative.
        Vectorized implementation to minimize GPU kernel launches and synchronization.
        """
        # alpha_s: step length for slacks
        self._work_s[:] = cp.where(self._step.s_all < 0, -self._result.s_all / self._step.s_all, 1.)
        self._result.info.primal_step[:] = cp.min(self._work_s, axis=1)  # alpha_s
        self._result.info.primal_step *= self.settings.tau  # avoid getting too close to the boundary

        # alpha_z: step length for duals
        self._work_z[:] = cp.where(self._step.z_all < 0, -self._result.z_all / self._step.z_all, 1.)
        self._result.info.dual_step[:] = cp.min(self._work_z, axis=1)  # alpha_z
        self._result.info.dual_step *= self.settings.tau  # avoid getting too close to the boundary

    @nvtx.annotate("Solver::_calculate_mu")
    @cuda_graph_capture(enable=lambda self: self.settings.enable_cuda_graph)
    def _calculate_mu(self) -> None:
        cp.multiply(self._result.s_all, self._result.z_all, out=self._work_s)
        cp.sum(self._work_s, axis=1, out=self._result.info.mu)
        self._result.info.mu /= self._data.num_ineq

    @nvtx.annotate("Solver::_calculate_sigma")
    @cuda_graph_capture(enable=lambda self: self.settings.enable_cuda_graph)
    def _calculate_sigma(self) -> None:
        # s_trial = s + alpha_s * ds,  z_trial = z + alpha_z * dz
        cp.multiply(self._result.info.primal_step[:, None], self._step.s_all, out=self._work_s) # s_trial in _work_s
        self._work_s += self._result.s_all
        cp.multiply(self._result.info.dual_step[:, None], self._step.z_all, out=self._work_z)  # z_trial in _work_z
        self._work_z += self._result.z_all
        cp.multiply(self._work_s, self._work_z, out=self._work_s)  # reuse _work_s to hold s_trial * z_trial
        cp.sum(self._work_s, axis=1, out=self._result.info.sigma)

        cp.divide(self._result.info.sigma, self._result.info.mu, out=self._result.info.sigma)
        self._result.info.sigma /= self._data.num_ineq
        cp.clip(self._result.info.sigma, 0., 1., out=self._result.info.sigma)
        cp.power(self._result.info.sigma, 3., out=self._result.info.sigma)

    @nvtx.annotate("Solver::_update_residuals_nr")
    @cuda_graph_capture(enable=lambda self: self.settings.enable_cuda_graph)
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
        # cuSPARSE/cuBLAS operations
        self._kkt_system.eval_P_x(self._data, -1., self._result.x, self._res_nr.x)
        # ||unscale_dual_res(P*x)||_inf -> _work_dual_res_norm (will be updated further inside graph)
        cp.absolute(self._res_nr.x, out=self._work_primals)
        self._work_primals *= self._unscale_dual_res_factor
        cp.max(self._work_primals, axis=1, out=self._work_dual_res_norm)

        if self._data.p > 0:
            self._kkt_system.eval_A_xn(self._data, -1., self._result.x, self._res_nr.y)  # store -A*x in res_nr.y
            self._kkt_system.eval_AT_xt(self._data, 1., self._result.y, self._res.x)  # store A^T*y in res.x
        else:
            self._res.x.fill(0.)  # no equality constraints, A^T*y = 0
        # ||unscale_primal_res_eq(A*x)||_inf -> _work_primal_rel_norm (will be updated further inside graph)
        if self._data.p > 0:
            cp.absolute(self._res_nr.y, out=self._work_duals[:, :self._data.p])
            self._work_duals[:, :self._data.p] *= self._unscale_primal_res_eq_factor
            cp.max(self._work_duals[:, :self._data.p], axis=1, out=self._work_primal_rel_norm)
        else:
            self._work_primal_rel_norm.fill(0.)

        self._work_z_1.fill(0.)
        self._work_z_1[:, self._data.idx_hu] += self._result.z_u
        self._work_z_1[:, self._data.idx_hl] -= self._result.z_l

        G_x = self._work_z_2  # reuse self._work_z_2 to store G*x
        GT_zu_minus_zl = self._step.x  # reuse self._step.x as temporary storage
        # NOTE: cublasDgemmStridedBatched performs many memset if m=0, so better not call it when unnecessary
        if self._data.m > 0:
            self._kkt_system.eval_G_xn(self._data, 1., self._result.x, G_x)
            self._kkt_system.eval_GT_xt(self._data, 1., self._work_z_1, GT_zu_minus_zl)
        else:
            G_x.fill(0.)
            GT_zu_minus_zl.fill(0)

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

        # Per-problem dot products via sum of elementwise product along axis=1
        # _work_reduce is (B, 8)
        cp.sum(self._res_nr.x * self._result.x, axis=1, out=self._work_reduce[:, 0])
        self._work_reduce[:, 0] *= -0.5  # hold 0.5 * x^T P x
        cp.sum(self._data.c * self._result.x, axis=1, out=self._work_reduce[:, 1])  # hold c^T x

        self._work_reduce[:, 2] = self._work_reduce[:, 0]  # hold 0.5 * x^T P x
        cp.sum(self._data.b * self._result.y, axis=1, out=self._work_reduce[:, 3])  # hold b^T y
        cp.sum(self._data.h_l[:, self._data.idx_hl] * self._result.z_l, axis=1, out=self._work_reduce[:, 4])
        self._work_reduce[:, 4] *= -1.  # hold - h_l^T z_l
        cp.sum(self._data.h_u[:, self._data.idx_hu] * self._result.z_u, axis=1, out=self._work_reduce[:, 5])  # hold h_u^T z_u
        cp.sum(self._data.x_l[:, self._data.idx_xl] * self._result.z_bl, axis=1, out=self._work_reduce[:, 6])
        self._work_reduce[:, 6] *= -1.  # hold - x_l^T z_bl
        cp.sum(self._data.x_u[:, self._data.idx_xu] * self._result.z_bu, axis=1, out=self._work_reduce[:, 7])  # hold x_u^T z_bu

        cp.sum(self._work_reduce[:, 0:2], axis=1, out=self._result.info.primal_obj)  # primal_obj = 0.5 * x^T P x + c^T x
        cp.sum(self._work_reduce[:, 2:8], axis=1, out=self._result.info.dual_obj)
        self._result.info.dual_obj *= -1.  # dual_obj = -b^T y - h_u^T z_u + h_l^T z_l - x_u^T z_bu + x_l^T z_bl

        # duality_gap = abs(primal_obj - dual_obj) (in scaled space, then unscale all three)
        cp.subtract(self._result.info.primal_obj, self._result.info.dual_obj, out=self._result.info.duality_gap)

        # Unscale objectives and duality gap from scaled to original space
        self._result.info.primal_obj *= self._c_scaling_inv
        self._result.info.dual_obj *= self._c_scaling_inv
        self._result.info.duality_gap *= self._c_scaling_inv

        # duality_gap_rel_norm = max(abs(unscaled terms))
        # Unscale the work_reduce terms before computing duality_gap_rel
        self._work_reduce *= self._c_scaling_inv[:, None]
        cp.abs(self._work_reduce, out=self._work_reduce)
        cp.max(self._work_reduce[:, 0:8], axis=1, out=self._result.info.duality_gap_rel)
        cp.abs(self._result.info.duality_gap, out=self._result.info.duality_gap)
        cp.maximum(self._result.info.duality_gap_rel, 1., out=self._result.info.duality_gap_rel)
        cp.divide(self._result.info.duality_gap, self._result.info.duality_gap_rel, out=self._result.info.duality_gap_rel)

        # ------------ update non-regulerized residuals ------------
        # res_nr.x = -(P*x + c + A^T*y + G^T*(z_u - z_l) + z_bu - z_bl)
        # self._res_nr.x is computed as -P*x above
        self._res_nr.x -= self._data.c
        self._res_nr.x -= self._res.x  # self._res.x holds A^T*y
        self._res_nr.x -= GT_zu_minus_zl
        self._res_nr.x[:, self._data.idx_xl] += self._data.x_b_scaling[:, self._data.idx_xl] * self._result.z_bl
        self._res_nr.x[:, self._data.idx_xu] -= self._data.x_b_scaling[:, self._data.idx_xu] * self._result.z_bu

        # res_nr.y = -(A*x - b)
        self._res_nr.y += self._data.b

        # res_nr.z_l = G*x - s_l - hl
        self._res_nr.z_l[:] = G_x[:, self._data.idx_hl]
        cp.subtract(self._res_nr.z_l, self._result.s_l, out=self._res_nr.z_l)
        cp.subtract(self._res_nr.z_l, self._data.h_l[:, self._data.idx_hl], out=self._res_nr.z_l)

        # res_nr.z_u = -G*x - s_u + hu
        self._res_nr.z_u[:] = -G_x[:, self._data.idx_hu]
        cp.subtract(self._res_nr.z_u, self._result.s_u, out=self._res_nr.z_u)
        cp.add(self._res_nr.z_u, self._data.h_u[:, self._data.idx_hu], out=self._res_nr.z_u)

        # res_nr.z_bl = x_b_scaling*x - s_bl - xl
        self._res_nr.z_bl[:] = self._result.x[:, self._data.idx_xl]
        self._res_nr.z_bl *= self._data.x_b_scaling[:, self._data.idx_xl]
        cp.subtract(self._res_nr.z_bl, self._result.s_bl, out=self._res_nr.z_bl)
        cp.subtract(self._res_nr.z_bl, self._data.x_l[:, self._data.idx_xl], out=self._res_nr.z_bl)

        # res_nr.z_bu = -(x_b_scaling*x + s_bu - xu)
        self._res_nr.z_bu[:] = self._result.x[:, self._data.idx_xu]
        self._res_nr.z_bu *= self._data.x_b_scaling[:, self._data.idx_xu]
        cp.add(self._res_nr.z_bu, self._result.s_bu, out=self._res_nr.z_bu)
        cp.subtract(self._res_nr.z_bu, self._data.x_u[:, self._data.idx_xu], out=self._res_nr.z_bu)
        cp.negative(self._res_nr.z_bu, out=self._res_nr.z_bu)


        # ------------ update primal and dual residuals ------------
        self._result.info.prev_primal_res[:] = self._result.info.primal_res
        self._result.info.prev_dual_res[:] = self._result.info.dual_res

        self._result.info.primal_res[:] = self._primal_res_nr()

        # primal_rel_norm: update running max (initialized outside graph with ||unscale(A*x)||_inf)
        # All terms are unscaled before taking norms to match PIQP C++ convergence check.
        # _work_z_1 is free at this point (only used before graph for cuSPARSE input)
        if self._data.num_hu > 0:
            self._work_z_1[:, :self._data.num_hu] = cp.abs(G_x[:, self._data.idx_hu])
            self._work_z_1[:, :self._data.num_hu] *= self._unscale_primal_res_ineq_hu
            cp.max(self._work_z_1[:, :self._data.num_hu], axis=1, out=self._work_norm_temp)
            cp.maximum(self._work_primal_rel_norm, self._work_norm_temp, out=self._work_primal_rel_norm)

        if self._data.num_hl > 0:
            self._work_z_1[:, :self._data.num_hl] = cp.abs(G_x[:, self._data.idx_hl])
            self._work_z_1[:, :self._data.num_hl] *= self._unscale_primal_res_ineq_hl
            cp.max(self._work_z_1[:, :self._data.num_hl], axis=1, out=self._work_norm_temp)
            cp.maximum(self._work_primal_rel_norm, self._work_norm_temp, out=self._work_primal_rel_norm)

        if self._data.num_hu > 0:
            cp.absolute(self._result.s_u, out=self._work_z_1[:, :self._data.num_hu])
            self._work_z_1[:, :self._data.num_hu] *= self._unscale_primal_res_ineq_hu
            cp.max(self._work_z_1[:, :self._data.num_hu], axis=1, out=self._work_norm_temp)
            cp.maximum(self._work_primal_rel_norm, self._work_norm_temp, out=self._work_primal_rel_norm)

        if self._data.num_hl > 0:
            cp.absolute(self._result.s_l, out=self._work_z_1[:, :self._data.num_hl])
            self._work_z_1[:, :self._data.num_hl] *= self._unscale_primal_res_ineq_hl
            cp.max(self._work_z_1[:, :self._data.num_hl], axis=1, out=self._work_norm_temp)
            cp.maximum(self._work_primal_rel_norm, self._work_norm_temp, out=self._work_primal_rel_norm)

        if self._data.num_xu > 0:
            cp.absolute(self._result.s_bu, out=self._work_z[:, :self._data.num_xu])
            self._work_z[:, :self._data.num_xu] *= self._unscale_primal_res_b_xu
            cp.max(self._work_z[:, :self._data.num_xu], axis=1, out=self._work_norm_temp)
            cp.maximum(self._work_primal_rel_norm, self._work_norm_temp, out=self._work_primal_rel_norm)

        if self._data.num_xl > 0:
            cp.absolute(self._result.s_bl, out=self._work_z[:, :self._data.num_xl])
            self._work_z[:, :self._data.num_xl] *= self._unscale_primal_res_b_xl
            cp.max(self._work_z[:, :self._data.num_xl], axis=1, out=self._work_norm_temp)
            cp.maximum(self._work_primal_rel_norm, self._work_norm_temp, out=self._work_primal_rel_norm)

        cp.maximum(self._work_primal_rel_norm, self._constraints_rhs_inf_norm_unscaled, out=self._work_primal_rel_norm)
        # Store max(1, primal_rel_norm) for use by _update_residuals_r
        cp.maximum(self._work_primal_rel_norm, 1., out=self._work_primal_rel_norm)
        cp.divide(self._result.info.primal_res, self._work_primal_rel_norm, out=self._result.info.primal_res_rel)

        # dual_res_norm: update running max (initialized outside graph with ||unscale(P*x)||_inf)
        self._result.info.dual_res[:] = self._dual_res_nr()

        # ||unscale_dual_res(c)||_inf
        cp.absolute(self._data.c, out=self._work_primals)
        self._work_primals *= self._unscale_dual_res_factor
        cp.max(self._work_primals, axis=1, out=self._work_norm_temp)
        cp.maximum(self._work_dual_res_norm, self._work_norm_temp, out=self._work_dual_res_norm)

        # ||unscale_dual_res(A^T*y + G^T*(z_u - z_l) + x_b_scaling*(z_bu - z_bl))||_inf
        self._res.x += GT_zu_minus_zl
        self._res.x[:, self._data.idx_xl] -= self._data.x_b_scaling[:, self._data.idx_xl] * self._result.z_bl
        self._res.x[:, self._data.idx_xu] += self._data.x_b_scaling[:, self._data.idx_xu] * self._result.z_bu
        cp.absolute(self._res.x, out=self._work_primals)
        self._work_primals *= self._unscale_dual_res_factor
        cp.max(self._work_primals, axis=1, out=self._work_norm_temp)
        cp.maximum(self._work_dual_res_norm, self._work_norm_temp, out=self._work_dual_res_norm)

        # store max(1, dual_res_norm) for use by _update_residuals_r
        cp.maximum(self._work_dual_res_norm, 1., out=self._work_dual_res_norm)
        cp.divide(self._result.info.dual_res, self._work_dual_res_norm, out=self._result.info.dual_res_rel)

    @nvtx.annotate("Solver::_update_residuals_r")
    @cuda_graph_capture(enable=lambda self: self.settings.enable_cuda_graph)
    def _update_residuals_r(self):
        """
        Compute the regularized primal and dual residuals. The computation is based on the non-regularized residuals computed in _update_residuals_nr.
        It adds the regularization terms to the non-regularized residuals to obtain the regularized residuals.
        """
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
                dim=(self._data.batch_size, self._data.n + self._data.p + self._data.num_ineq),
                inputs=[
                    self._result.info.rho, self._result.info.delta,
                    self._res_nr.x, self._res_nr.duals_all,
                    self._result.x, self._result.duals_all,
                    self._prox_vars.x, self._prox_vars.duals_all,
                    self._res.x, self._res.duals_all
                ],
                device="cuda",
                stream=wp.Stream(cuda_stream=cp.cuda.get_current_stream().ptr)
            )
        else:
            cp.subtract(self._result.x, self._prox_vars.x, out=self._res.x)
            self._res.x *= self._result.info.rho[:, None]
            cp.subtract(self._res_nr.x, self._res.x, out=self._res.x)
            cp.subtract(self._prox_vars.duals_all, self._result.duals_all, out=self._res.duals_all)
            self._res.duals_all *= self._result.info.delta[:, None]
            cp.subtract(self._res_nr.duals_all, self._res.duals_all, out=self._res.duals_all)

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
        
        

    @nvtx.annotate("Solver::_primal_res_nr")
    def _primal_res_nr(self):
        offset = 0
        self._work_duals[:, :self._data.p] = self._res_nr.y
        self._work_duals[:, :self._data.p] *= self._unscale_primal_res_eq_factor
        offset += self._data.p
        self._work_duals[:, offset:offset+self._data.num_hu] = self._res_nr.z_u
        self._work_duals[:, offset:offset+self._data.num_hu] *= self._unscale_primal_res_ineq_hu
        offset += self._data.num_hu
        self._work_duals[:, offset:offset+self._data.num_hl] = self._res_nr.z_l
        self._work_duals[:, offset:offset+self._data.num_hl] *= self._unscale_primal_res_ineq_hl
        offset += self._data.num_hl
        self._work_duals[:, offset:offset+self._data.num_xu] = self._res_nr.z_bu
        self._work_duals[:, offset:offset+self._data.num_xu] *= self._unscale_primal_res_b_xu
        offset += self._data.num_xu
        self._work_duals[:, offset:offset+self._data.num_xl] = self._res_nr.z_bl
        self._work_duals[:, offset:offset+self._data.num_xl] *= self._unscale_primal_res_b_xl
        offset += self._data.num_xl
        if offset > 0:
            cp.absolute(self._work_duals[:, :offset], out=self._work_duals[:, :offset])
            cp.max(self._work_duals[:, :offset], axis=1, out=self._work_residual)
        else:
            self._work_residual.fill(0.)
        return self._work_residual

    @nvtx.annotate("Solver::_primal_res_r")
    def _primal_res_r(self):
        offset = 0
        self._work_duals[:, :self._data.p] = self._res.y
        self._work_duals[:, :self._data.p] *= self._unscale_primal_res_eq_factor
        offset = self._data.p
        self._work_duals[:, offset:offset+self._data.num_hu] = self._res.z_u
        self._work_duals[:, offset:offset+self._data.num_hu] *= self._unscale_primal_res_ineq_hu
        offset += self._data.num_hu
        self._work_duals[:, offset:offset+self._data.num_hl] = self._res.z_l
        self._work_duals[:, offset:offset+self._data.num_hl] *= self._unscale_primal_res_ineq_hl
        offset += self._data.num_hl
        self._work_duals[:, offset:offset+self._data.num_xu] = self._res.z_bu
        self._work_duals[:, offset:offset+self._data.num_xu] *= self._unscale_primal_res_b_xu
        offset += self._data.num_xu
        self._work_duals[:, offset:offset+self._data.num_xl] = self._res.z_bl
        self._work_duals[:, offset:offset+self._data.num_xl] *= self._unscale_primal_res_b_xl
        offset += self._data.num_xl
        if offset > 0:
            cp.absolute(self._work_duals[:, :offset], out=self._work_duals[:, :offset])
            cp.max(self._work_duals[:, :offset], axis=1, out=self._work_residual)
        else:
            self._work_residual.fill(0.)
        return self._work_residual

    @nvtx.annotate("Solver::_dual_res_nr")
    def _dual_res_nr(self):
        # Unscale dual residual before computing inf-norm
        cp.absolute(self._res_nr.x, out=self._work_primals)
        self._work_primals *= self._unscale_dual_res_factor
        cp.max(self._work_primals, axis=1, out=self._work_residual)
        return self._work_residual
    
    @nvtx.annotate("Solver::_dual_res_r")
    def _dual_res_r(self):
        cp.absolute(self._res.x, out=self._work_primals)
        self._work_primals *= self._unscale_dual_res_factor
        cp.max(self._work_primals, axis=1, out=self._work_residual)
        return self._work_residual
    
    @nvtx.annotate("Solver::_primal_prox_inf")
    def _primal_prox_inf(self):
        if self._work_duals.shape[1] > 0:
            cp.subtract(self._result.duals_all, self._prox_vars.duals_all, out=self._work_duals)
            cp.absolute(self._work_duals, out=self._work_duals)
            cp.max(self._work_duals, axis=1, out=self._work_residual)
        else:
            self._work_residual.fill(0.)
        return self._work_residual

    @nvtx.annotate("Solver::_dual_prox_inf")
    def _dual_prox_inf(self):
        cp.subtract(self._result.x, self._prox_vars.x, out=self._work_primals)
        cp.absolute(self._work_primals, out=self._work_primals)
        cp.max(self._work_primals, axis=1, out=self._work_residual)
        return self._work_residual
    
    @nvtx.annotate("Solver::_update_rho_delta_with_ineq")
    def _update_rho_delta_with_ineq(self) -> None:
        """Update rho/delta based on residual progress — branchless via cp.where."""
        info = self._result.info
        settings = self.settings
        mu_rate = self._mu_rate.ravel()  # (B,)

        # --- Rho update ---
        dual_improved = (
            (info.dual_res < 0.95 * info.prev_dual_res) |
            (info.dual_res < settings.eps_abs) | (info.dual_res_rel < settings.eps_rel) |
            ((info.rho == settings.reg_finetune_lower_limit) & (info.dual_prox_inf < settings.infeasibility_threshold))
        )
        rho_fast = cp.maximum(info.reg_limit, (1. - mu_rate) * info.rho)
        rho_slow = cp.maximum(info.reg_limit, (1. - 0.666 * mu_rate) * info.rho)
        rho_slow_decay_ok = (~dual_improved) & ((info.iter[0] < 5) | (info.dual_prox_inf < settings.infeasibility_threshold))
        info.rho[:] = cp.where(dual_improved, rho_fast, cp.where(rho_slow_decay_ok, rho_slow, info.rho))
        self._prox_vars.x[:] = cp.where(dual_improved[:, None], self._result.x, self._prox_vars.x)
        self._result.info.no_primal_update += cp.asnumpy((~dual_improved).astype(cp.int32))

        # --- Delta update ---
        primal_improved = (
            (info.primal_res < 0.95 * info.prev_primal_res) |
            (info.primal_res < settings.eps_abs) | (info.primal_res_rel < settings.eps_rel) |
            ((info.delta == settings.reg_finetune_lower_limit) & (info.primal_prox_inf < settings.infeasibility_threshold))
        )
        delta_fast = cp.maximum(info.reg_limit, (1. - mu_rate) * info.delta)
        delta_slow = cp.maximum(info.reg_limit, (1. - 0.666 * mu_rate) * info.delta)
        delta_slow_decay_ok = (~primal_improved) & ((info.iter[0] < 5) | (info.primal_prox_inf < settings.infeasibility_threshold))
        info.delta[:] = cp.where(primal_improved, delta_fast, cp.where(delta_slow_decay_ok, delta_slow, info.delta))
        self._prox_vars.duals_all[:] = cp.where(primal_improved[:, None], self._result.duals_all, self._prox_vars.duals_all)
        self._result.info.no_dual_update += cp.asnumpy((~primal_improved).astype(cp.int32))

    @nvtx.annotate("Solver::_update_rho_delta_without_ineq")
    def _update_rho_delta_without_ineq(self) -> None:
        """Update rho/delta based on residual progress — branchless via cp.where."""
        info = self._result.info
        settings = self.settings

        # --- Rho update ---
        dual_improved = (
            (info.dual_res < 0.95 * info.prev_dual_res) |
            (info.dual_res < settings.eps_abs) |
            (info.dual_res_rel < settings.eps_rel)
        )
        rho_fast = cp.maximum(info.reg_limit, 0.1 * info.rho)
        rho_slow = cp.maximum(info.reg_limit, 0.5 * info.rho)
        rho_slow_decay_ok = (~dual_improved) & ((info.iter[0] < 5) | (info.dual_prox_inf < settings.infeasibility_threshold))
        info.rho[:] = cp.where(dual_improved, rho_fast, cp.where(rho_slow_decay_ok, rho_slow, info.rho))
        self._prox_vars.x[:] = cp.where(dual_improved[:, None], self._result.x, self._prox_vars.x)
        self._result.info.no_primal_update += cp.asnumpy((~dual_improved).astype(cp.int32))

        # --- Delta update ---
        primal_improved = (
            (info.primal_res < 0.95 * info.prev_primal_res) |
            (info.primal_res < settings.eps_abs) |
            (info.primal_res_rel < settings.eps_rel)
        )
        delta_fast = cp.maximum(info.reg_limit, 0.1 * info.delta)
        delta_slow = cp.maximum(info.reg_limit, 0.5 * info.delta)
        delta_slow_decay_ok = (~primal_improved) & ((info.iter[0] < 5) | (info.primal_prox_inf < settings.infeasibility_threshold))
        info.delta[:] = cp.where(primal_improved, delta_fast, cp.where(delta_slow_decay_ok, delta_slow, info.delta))
        self._prox_vars.y[:] = cp.where(primal_improved[:, None], self._result.y, self._prox_vars.y)
        self._result.info.no_dual_update += cp.asnumpy((~primal_improved).astype(cp.int32))



def create_update_residual_r_kernel(n: int, p: int, num_ineq: int):
    """Performs the following operations using contiguous duals_all
        res.x[b]         = res_nr.x[b]         - rho[b]   * (result.x[b]         - prox.x[b])
        res.duals_all[b] = res_nr.duals_all[b] + delta[b] * (result.duals_all[b] - prox.duals_all[b])
    """
    @wp.kernel
    def update_residual_r_kernel(
        rho: wp.array(dtype=wp.float64),             # (B,)         # type: ignore
        delta: wp.array(dtype=wp.float64),           # (B,)         # type: ignore
        res_nr_x: wp.array2d(dtype=wp.float64),      # (B, n)       # type: ignore
        res_nr_dual: wp.array2d(dtype=wp.float64),   # (B, p+nineq) # type: ignore
        result_x: wp.array2d(dtype=wp.float64),      # (B, n)       # type: ignore
        result_dual: wp.array2d(dtype=wp.float64),   # (B, p+nineq) # type: ignore
        prox_x: wp.array2d(dtype=wp.float64),        # (B, n)       # type: ignore
        prox_dual: wp.array2d(dtype=wp.float64),     # (B, p+nineq) # type: ignore
        res_r_x: wp.array2d(dtype=wp.float64),       # (B, n)       # type: ignore
        res_r_dual: wp.array2d(dtype=wp.float64),    # (B, p+nineq) # type: ignore
    ):
        b, t = wp.tid()
        n_static = wp.static(n)
        num_duals_static = wp.static(p + num_ineq)

        if t < n_static:
            res_r_x[b, t] = -rho[b] * (result_x[b, t] - prox_x[b, t]) + res_nr_x[b, t]
        elif t < n_static + num_duals_static:
            idx = t - n_static
            res_r_dual[b, idx] = delta[b] * (result_dual[b, idx] - prox_dual[b, idx]) + res_nr_dual[b, idx]

    return update_residual_r_kernel
