from abc import ABC, abstractmethod
from typing import Optional, Any

import numpy as np
import cupy as cp
import warp as wp
import nvtx

from .settings import Settings
from .data import Data
from .results import Result, Status, Variables, InfoHost
from .kkt_systems import KKTSystem
from .utils import cuda_graph_capture
from .solver_kernels import (
    create_prepare_predictor_step_kernel,
    create_prepare_corrector_step_kernel,
    create_update_vars_after_corrector_step_kernel,
    create_boundary_shift_kernel,
    create_calculate_step_kernel,
    create_calculate_sigma_kernel,
    create_calculate_mu_kernel,
    create_update_residuals_r_kernel,
    create_prepare_zu_minus_zl_and_zbu_minus_zbl_kernel,
    create_update_residual_nr_kernel,
    create_update_rho_delta_with_ineq_kernel,
    create_update_rho_delta_without_ineq_kernel,
    create_run_full_newton_step_kernel,
)


wp.config.quiet = True  # disable warp module initialization messages.
wp.config.enable_backward = False  # disable backward mode, cut down kernel compile time
wp.init()


class SolverBase(ABC):
    """Abstract base for the cuPIQP solver."""

    def __init__(self):
        self._settings = Settings()
        self._data: Data = None
        self._result = Result()    # store the values of primal, dual and slack variables of current iteration, and other information
        self._step = Variables()   # used to store the step direction of primal and dual variables
        self._res_nr = Variables()  # used to store the non-regularized residuals
        self._res = Variables()  # used to store the regularized residuals
        self._prox_vars = Variables()  # used to store the proximal variables
        self._kkt_system = KKTSystem()
        self._preconditioner = None

    @property
    def settings(self) -> Settings:
        return self._settings

    @settings.setter
    def settings(self, value: Settings) -> None:
        self._settings = value

    @property
    def result(self):
        return self._result
    
    @nvtx.annotate("Solver::setup")
    def setup(self, P, c, A=None, b=None, G=None, h_u=None, h_l=None, x_u=None, x_l=None):
        # Detect if user provided batched (3D P) or non-batched (2D P) data.
        # DenseData auto-unsqueezes non-batched to (1, ...) internally,
        # but we track this so solve() returns the right type.
        self._user_batched = (hasattr(P, 'ndim') and P.ndim == 3) or (isinstance(P, (list, tuple)) and len(P) > 1)

        self._data = self._init_data(P, c, A, b, G, h_u, h_l, x_u, x_l)
        self._preconditioner = self._init_preconditioner()
        if self.settings.preconditioner_iter > 0:
            self._preconditioner.scale_data(
                self._data,
                self.settings.preconditioner_scale_cost,
                self.settings.preconditioner_iter,
            )

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
        
        self._init_warp_kernels()

        self._work_x = cp.empty((B, self._data.n), dtype=cp.float64)

        self._tau_device = cp.empty(1, dtype=cp.float64)
        self._tau_device[0] = self.settings.tau  # device copy used by warp kernels
        self._tau_host = float(self.settings.tau)  # host cache -- only H2D when tau actually changes

        # Pre-allocated (B,) buffers for CUDA-graph-safe norm computations in _update_residuals_nr / _update_residuals_r
        self._work_primal_rel_norm = cp.empty(B, dtype=cp.float64)  # running max of primal relative norm terms
        self._work_dual_res_norm = cp.empty(B, dtype=cp.float64)    # running max of dual residual norm terms
        self._work_norm_temp = cp.empty(B, dtype=cp.float64)        # temp (B,) for individual norm results

        self._enable_iterative_refinement = self.settings.iterative_refinement_always_enabled

        # Unscaled-RHS inf-norm. When preconditioner_iter == 0 the stored
        # factors are identity, so this reduces to the inf-norm of the user-
        # space b / h_l/u / x_l/u — same answer, single code path.
        self._constraints_rhs_inf_norm_unscaled = cp.zeros(B, dtype=cp.float64)
        self._preconditioner.compute_constraints_rhs_inf_norm_unscaled(
            self._data, self._constraints_rhs_inf_norm_unscaled,
        )

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

        if self.settings.preconditioner_iter > 0:
            self._preconditioner.unscale_data(self._data)

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

        matrix_changed = P is not None or A is not None or G is not None

        # Apply preconditioner scaling to updated data.
        preconditioner_did_fresh_ruiz = False
        if self.settings.preconditioner_iter > 0:
            reuse = self.settings.preconditioner_reuse_on_update or not matrix_changed
            if reuse:
                self._preconditioner.reuse_scaling(self._data)
            else:
                self._preconditioner.reset()
                self._preconditioner.scale_data(
                    self._data,
                    self.settings.preconditioner_scale_cost,
                    self.settings.preconditioner_iter,
                )
                preconditioner_did_fresh_ruiz = True

        self._preconditioner.compute_constraints_rhs_inf_norm_unscaled(
            self._data, self._constraints_rhs_inf_norm_unscaled,
        )
        # Fresh Ruiz produces new factors that re-scale ALL of P/A/G in place,
        # even matrices the user didn't pass. The KKT solver caches things
        # like A^T A keyed off those scaled values, so flag everything as
        # changed in that case.
        self._kkt_system.update_data(
            self._data,
            (P is not None) or preconditioner_did_fresh_ruiz,
            (A is not None) or preconditioner_did_fresh_ruiz,
            (G is not None) or preconditioner_did_fresh_ruiz,
        )

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
        # Refresh tau only if the user changed settings.tau between solves because it requires H2D memcpy
        if self._tau_host != self.settings.tau:
            self._tau_device[0] = self.settings.tau
            self._tau_host = float(self.settings.tau)
        self._result.info.factor_retires[:] = 0
        self._result.info.no_primal_update[:] = 0
        self._result.info.no_dual_update[:] = 0
        self._result.info.mu[:] = 0.
        self._result.info.primal_step[:] = 0.
        self._result.info.dual_step[:] = 0.
        self._result.info.rho[:] = self.settings.rho_init
        self._result.info.delta[:] = self.settings.delta_init

        if self.settings.verbose:
            if self._data.batch_size == 1:
                print("iter  prim_obj       dual_obj       duality_gap   prim_res      dual_res      rho         delta       mu          p_step   d_step")
            else:
                # Match the column widths used in ``_print_iteration_info``
                # so header + data right-align to the same edge.
                B = self._data.batch_size
                counter_w = max(2 * len(str(B)) + 1, len("solved"))
                print(
                    f"{'iter':>4}  "
                    f"{'solved':>{counter_w}}  "
                    f"{'gap_max':>12}  "
                    f"{'p_res_max':>12}  "
                    f"{'d_res_max':>12}  "
                    f"{'rho_max':>10}  "
                    f"{'delta_max':>10}  "
                    f"{'mu_max':>10}  "
                    f"{'p_step':>6}  "
                    f"{'d_step':>6}"
                )

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
                    (info_host.no_dual_update > min(5, settings.reg_finetune_dual_update_threshold)) &
                    (info_host.primal_prox_inf > settings.infeasibility_threshold) &
                    ((info_host.primal_res_reg < settings.eps_abs) | (info_host.primal_res_reg_rel < settings.eps_rel))
                )
                self._result.info._status_value[still_unsolved & ~converged & primal_infeasible] = Status.PIQP_PRIMAL_INFEASIBLE.value  # CPU write

                # dual infeasibility check
                dual_infeasible = (
                    (info_host.no_primal_update > min(5, settings.reg_finetune_primal_update_threshold)) &
                    (info_host.dual_prox_inf > settings.infeasibility_threshold) &
                    ((info_host.dual_res_reg < settings.eps_abs) | (info_host.dual_res_reg_rel < settings.eps_rel))
                )
                self._result.info._status_value[still_unsolved & ~converged & ~primal_infeasible & dual_infeasible] = Status.PIQP_DUAL_INFEASIBLE.value  # CPU write

                if self.settings.verbose:
                    self._print_iteration_info()

                # exit if all problems have terminated
                if np.all(self._result.info._status_value != Status.PIQP_UNSOLVED.value):
                    break

                # avoid getting too close to boundary which can result in a division by zero
                if self._data.num_ineq > 0:
                    wp.launch(
                        kernel=self._boundary_shift_kernel,
                        dim=(self._data.batch_size,
                             self._data.num_hl + self._data.num_hu
                             + self._data.num_xl + self._data.num_xu),
                        inputs=[
                            self._result.z_l, self._result.z_u,
                            self._result.z_bl, self._result.z_bu,
                        ],
                        device="cuda",
                        stream=wp.Stream(cuda_stream=cp.cuda.get_current_stream().ptr),
                    )
                    self._calculate_mu()
                
                # avoid possibility of converging to a local minimum -> decrease the minimum regularization value (vectorized)
                finetune_mask = (
                    ((info_host.no_primal_update > self.settings.reg_finetune_primal_update_threshold) &
                     (info_host.rho == info_host.reg_limit) &
                     (info_host.reg_limit != self.settings.reg_finetune_lower_limit)) |
                    ((info_host.no_dual_update > self.settings.reg_finetune_dual_update_threshold) &
                     (info_host.delta == info_host.reg_limit) &
                     (info_host.reg_limit != self.settings.reg_finetune_lower_limit))
                )
                finetune_mask &= (info_host.dual_prox_inf < self.settings.infeasibility_threshold) & (info_host.primal_prox_inf < self.settings.infeasibility_threshold)
                if np.any(finetune_mask):
                    self._result.info.reg_limit[finetune_mask] = self.settings.reg_finetune_lower_limit
                    finetune_mask_dev = cp.asarray(finetune_mask)
                    self._result.info.no_primal_update[finetune_mask_dev] = 0
                    self._result.info.no_dual_update[finetune_mask_dev] = 0

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
        if self.settings.preconditioner_iter > 0:
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
            self._preconditioner,
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

        self._kkt_system.solve(self._data, self._preconditioner, self.settings, self._res, self._result)  # getting an initial point of _result

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
        info_host = self._info_host
        B = self._data.batch_size

        if B == 1:
            print(
                f"{self._result.info.iter[0]:3d}   "
                f"{info_host.primal_obj[0]: .5e}   "
                f"{info_host.dual_obj[0]: .5e}  "
                f"{info_host.duality_gap[0]: .5e}  "
                f"{info_host.primal_res[0]: .5e}  "
                f"{info_host.dual_res[0]: .5e}  "
                f"{info_host.rho[0]: .3e}  "
                f"{info_host.delta[0]: .3e}  "
                f"{info_host.mu[0]: .3e}  "
                f"{info_host.primal_step[0]: .4f}  "
                f"{info_host.dual_step[0]: .4f}",
                flush=True,
            )
        
        else:
            solved  = B - int((self._result.info._status_value == Status.PIQP_UNSOLVED.value).sum())
            counter = f"{solved}/{B}"
            counter_w = max(2 * len(str(B)) + 1, len("solved"))
            print(
                f"{self._result.info.iter[0]:>4d}  "
                f"{counter:>{counter_w}}  "
                f"{info_host.duality_gap.max():>12.5e}  "
                f"{info_host.primal_res.max():>12.5e}  "
                f"{info_host.dual_res.max():>12.5e}  "
                f"{info_host.rho.max():>10.3e}  "
                f"{info_host.delta.max():>10.3e}  "
                f"{info_host.mu.max():>10.3e}  "
                f"{info_host.primal_step.min():>6.4f}  "
                f"{info_host.dual_step.min():>6.4f}",
                flush=True,
            )

    @nvtx.annotate("Solver::_update_and_factorize_kkt")
    def _update_and_factorize_kkt(self) -> None:
        """Update the KKT matrix and refactorize."""
        retries = 0
        while retries < self.settings.max_factor_retires:
            factor_succeeded = self._kkt_system.update_scalings_and_factor(
                self._data, self._preconditioner, self.settings, self._enable_iterative_refinement,
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

    @abstractmethod
    def _init_data(self, P, c, A, b, G, h_u, h_l, x_u, x_l):
        """Backend-specific data construction hook."""

    @abstractmethod
    def _init_preconditioner(self):
        """Backend-specific Ruiz preconditioner construction hook."""

    def _init_warp_kernels(self) -> None:
        if self._data.num_ineq > 0:
            self._boundary_shift_kernel = create_boundary_shift_kernel(
                self._data.num_hl, self._data.num_hu,
                self._data.num_xl, self._data.num_xu,
            )
            self._prepare_predictor_step_kernel = create_prepare_predictor_step_kernel()
            self._prepare_corrector_step_kernel = create_prepare_corrector_step_kernel()
            self._update_vars_after_corrector_step_kernel = create_update_vars_after_corrector_step_kernel(
                n_primal=self._data.n + self._data.num_ineq, n_dual=self._data.p + self._data.num_ineq,
            )

        # Tile-based kernels
        self._update_residuals_r_kernel = create_update_residuals_r_kernel(
            self._data.n, self._data.p,
            int(self._data.num_hu), int(self._data.num_hl),
            int(self._data.num_xu), int(self._data.num_xl),
        )
        self._prepare_zu_minus_zl_and_zbu_minus_zbl_kernel = create_prepare_zu_minus_zl_and_zbu_minus_zbl_kernel(
            self._data.m, self._data.n,
        )
        self._update_residual_nr_kernel = create_update_residual_nr_kernel(
            self._data.n, self._data.p, self._data.m,
            self._data.num_hl, self._data.num_hu, self._data.num_xl, self._data.num_xu,
        )
        if self._data.num_ineq > 0:
            self._calculate_sigma_kernel = create_calculate_sigma_kernel(self._data.num_ineq)
            self._calculate_step_kernel = create_calculate_step_kernel(self._data.num_ineq)
            self._calculate_mu_kernel = create_calculate_mu_kernel(self._data.num_ineq)
            self._update_rho_delta_with_ineq_kernel = create_update_rho_delta_with_ineq_kernel(
                self._data.n, self._data.p + self._data.num_ineq,
            )
        else:
            self._run_full_newton_step_kernel = create_run_full_newton_step_kernel(self._data.n, self._data.p)
            self._update_rho_delta_without_ineq_kernel = create_update_rho_delta_without_ineq_kernel(
                self._data.n, self._data.p,
            )

    @nvtx.annotate("Solver::_run_full_newton_step")
    def _run_full_newton_step(self):
        self._kkt_system.solve(self._data, self._preconditioner, self.settings, self._res, self._step)
        wp.launch(
            kernel=self._run_full_newton_step_kernel,
            dim=(self._data.batch_size, self._data.n + self._data.p),
            inputs=[
                self._step.x, self._step.y,
                self._result.x, self._result.y,
                self._result.info.primal_step, self._result.info.dual_step,
            ],
            device="cuda",
            stream=wp.Stream(cuda_stream=cp.cuda.get_current_stream().ptr),
        )

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
        
        # one fused kernel: res.s_all[b, i] = -s_all[b, i] * z_all[b, i].
        wp.launch(
            kernel=self._prepare_predictor_step_kernel,
            dim=(self._data.batch_size, self._data.num_ineq),
            inputs=[self._result.s_all, self._result.z_all, self._res.s_all],
            device="cuda",
            stream=wp.Stream(cuda_stream=cp.cuda.get_current_stream().ptr),
        )

        if self.settings.debug:
            print("predictor step rhs is: res= ", self._res)

        self._kkt_system.solve(self._data, self._preconditioner, self.settings, self._res, self._step)

        if self.settings.debug:
            print("predictor step is:", self._step)

        # step in the non-negative orthant
        self._calculate_step()

        # ------------------ compute centering parameter sigma ------------------
        self._calculate_sigma()

        # ------------------ corrector step ------------------
        # self._res.s_l += -self._step.s_l * self._step.z_l + self._result.info.sigma * self._result.info.mu
        # self._res.s_u += -self._step.s_u * self._step.z_u + self._result.info.sigma * self._result.info.mu
        # self._res.s_bl += -self._step.s_bl * self._step.z_bl + self._result.info.sigma * self._result.info.mu
        # self._res.s_bu += -self._step.s_bu * self._step.z_bu + self._result.info.sigma * self._result.info.mu
        wp.launch(
            kernel=self._prepare_corrector_step_kernel,
            dim=(self._data.batch_size, self._data.num_ineq),
            inputs=[
                self._step.s_all, self._step.z_all,
                self._result.info.sigma, self._result.info.mu,
                self._res.s_all,
            ],
            device="cuda",
            stream=wp.Stream(cuda_stream=cp.cuda.get_current_stream().ptr),
        )

        if self.settings.debug:
            print("corrector step rhs is: res= ", self._res)
        self._kkt_system.solve(self._data, self._preconditioner, self.settings, self._res, self._step)

        if self.settings.debug:
            print("corrector step is:", self._step)

        # step in the non-negative orthant
        self._calculate_step()
        self._update_vars_after_corrector_step()
        self._calculate_mu()

    @cuda_graph_capture(enable=lambda self: self.settings.enable_cuda_graph)
    def _update_vars_after_corrector_step(self):
        # self._result.primals_all += self._result.info.primal_step[:, None] * self._step.primals_all
        # self._result.duals_all += self._result.info.dual_step[:, None] * self._step.duals_all
        n_primal = self._data.n + self._data.num_ineq
        n_dual   = self._data.p + self._data.num_ineq
        wp.launch(
            kernel=self._update_vars_after_corrector_step_kernel,
            dim=(self._data.batch_size, n_primal + n_dual),
            inputs=[
                self._result.info.primal_step,
                self._result.info.dual_step,
                self._step.primals_all,
                self._step.duals_all,
                self._result.primals_all,
                self._result.duals_all,
            ],
            device="cuda",
            stream=wp.Stream(cuda_stream=cp.cuda.get_current_stream().ptr),
        )

    @nvtx.annotate("Solver::_calculate_step")
    @cuda_graph_capture(enable=lambda self: self.settings.enable_cuda_graph)
    def _calculate_step(self) -> None:
        STEP_BLOCK_DIM = 256
        wp.launch_tiled(
            kernel=self._calculate_step_kernel,
            dim=[self._data.batch_size],
            inputs=[
                self._result.s_all, self._result.z_all,
                self._step.s_all, self._step.z_all,
                self._tau_device,
                self._result.info.primal_step, self._result.info.dual_step,
            ],
            block_dim=STEP_BLOCK_DIM,
            device="cuda",
            stream=wp.Stream(cuda_stream=cp.cuda.get_current_stream().ptr),
        )

    @nvtx.annotate("Solver::_calculate_mu")
    @cuda_graph_capture(enable=lambda self: self.settings.enable_cuda_graph)
    def _calculate_mu(self) -> None:
        MU_BLOCK_DIM = 256
        wp.launch_tiled(
            kernel=self._calculate_mu_kernel,
            dim=[self._data.batch_size],
            inputs=[
                self._result.s_all, self._result.z_all,
                self._result.info.mu,
            ],
            block_dim=MU_BLOCK_DIM,
            device="cuda",
            stream=wp.Stream(cuda_stream=cp.cuda.get_current_stream().ptr),
        )

    @nvtx.annotate("Solver::_calculate_sigma")
    @cuda_graph_capture(enable=lambda self: self.settings.enable_cuda_graph)
    def _calculate_sigma(self) -> None:
        SIGMA_BLOCK_DIM = 256
        wp.launch_tiled(
            kernel=self._calculate_sigma_kernel,
            dim=[self._data.batch_size],
            inputs=[
                self._result.s_all, self._result.z_all,
                self._step.s_all, self._step.z_all,
                self._result.info.primal_step, self._result.info.dual_step,
                self._result.info.mu,
                self._result.info.sigma,
            ],
            block_dim=SIGMA_BLOCK_DIM,
            device="cuda",
            stream=wp.Stream(cuda_stream=cp.cuda.get_current_stream().ptr),
        )

    @nvtx.annotate("Solver::_update_residuals_nr")
    @cuda_graph_capture(enable=lambda self: self.settings.enable_cuda_graph)
    def _update_residuals_nr(self):
        r"""Compute non-regularized KKT residuals + objective values +
        relative norms (used for convergence checks).

        All variables (``x``, ``y``, ``z_l``, ``z_u``, ``z_bl``, ``z_bu``,
        ``s_*``) and data (``P``, ``c``, ``A``, ``b``, ``G``, ``h_*``,
        ``x_*``) are stored in the **scaled** problem space (Ruiz
        preconditioner). The bound rows pick up an extra ``x_b_scaling``
        factor because ``x_l <= x <= x_u`` becomes
        ``x_l_scaled <= x_b_scaling * x_scaled <= x_u_scaled`` after
        scaling. Convergence norms are reported in the **unscaled** problem
        space — magnitudes are restored via ``delta_inv``, ``delta_b_inv``,
        ``cost_scaling_inv`` from the preconditioner.

        Residual formulas (scaled space):

            res_nr.x    = -(P*x + c + A^T*y + G^T*(z_u - z_l)
                            + x_b_scaling*(z_bu - z_bl))
            res_nr.y    = -(A*x - b)
            res_nr.z_l  =   G*x[idx_hl] - s_l - h_l[idx_hl]
            res_nr.z_u  = -G*x[idx_hu] - s_u + h_u[idx_hu]
            res_nr.z_bl =   x_b_scaling[idx_xl]*x[idx_xl] - s_bl - x_l[idx_xl]
            res_nr.z_bu = -(x_b_scaling[idx_xu]*x[idx_xu] + s_bu - x_u[idx_xu])

        Convergence norms (unscaled, infinity norm per batch):

            primal_res     = max over the 5 dual segments of
                                 ||u_p_seg .* res_nr_seg||_inf
                             where u_p_seg is the per-segment primal
                             unscale factor:
                                 [y]:    delta_inv[:, n : n+p]
                                 [z_l]:  delta_inv[:, n+p+idx_hl]
                                 [z_u]:  delta_inv[:, n+p+idx_hu]
                                 [z_bl]: delta_b_inv[:, idx_xl]
                                 [z_bu]: delta_b_inv[:, idx_xu]

            dual_res       = cost_scaling_inv * ||delta_inv[:, :n] .* res_nr.x||_inf

        Relative-norm denominators (also unscaled, max over magnitudes
        that go into the corresponding residual):

            primal_rel     = max( ||u_p_y .* A*x||,
                                  ||u_p_zl .* G*x[idx_hl]||,
                                  ||u_p_zu .* G*x[idx_hu]||,
                                  ||u_p_zl .* s_l||,  ||u_p_zu .* s_u||,
                                  ||u_p_zbl .* s_bl||, ||u_p_zbu .* s_bu||,
                                  constraints_rhs_inf_norm_unscaled )

            dual_rel       = cost_scaling_inv * max(
                                  ||delta_inv[:, :n] .* P*x||,
                                  ||delta_inv[:, :n] .* c||,
                                  ||delta_inv[:, :n] .* (A^T*y + G^T*(z_u-z_l)
                                       + x_b_scaling*(z_bu-z_bl))|| )

            primal_res_rel = primal_res / max(1, primal_rel)
            dual_res_rel   = dual_res   / max(1, dual_rel)

        Objectives and duality gap (unscaled to original problem space via
        ``cost_scaling_inv``):

            primal_obj   = ( 0.5 x^T P x + c^T x ) * cost_scaling_inv
            dual_obj     = -( 0.5 x^T P x + b^T y + h_u^T z_u - h_l^T z_l
                              + x_u^T z_bu - x_l^T z_bl ) * cost_scaling_inv
            duality_gap  = |primal_obj - dual_obj|
            duality_gap_rel
                         = duality_gap / max(1, cost_scaling_inv *
                              max_k |w_k|)
                           where {w_k} is the set of seven obj sub-terms
                           used above (0.5 x^T P x, c^T x, b^T y, h_u^T z_u,
                           h_l^T z_l, x_u^T z_bu, x_l^T z_bl).
        """
        pc      = self._preconditioner
        data    = self._data
        result  = self._result
        res_nr  = self._res_nr
        info    = result.info
        wp_stream = wp.Stream(cuda_stream=cp.cuda.get_current_stream().ptr)

        self._kkt_system.eval_P_x(data, -1., result.x, res_nr.x)

        if data.p > 0:
            self._kkt_system.eval_A_xn(data, 1., result.x, self._res.y)
            self._kkt_system.eval_AT_xt(data, 1., result.y, self._res.x)
        else:
            self._res.y.fill(0.)
            self._res.x.fill(0.)

        # build work_z_1 (G^T * (z_u_scatter - z_l_scatter))
        # and self._work_x (x_b_scaling*(z_bu_scatter - z_bl_scattered))
        wp.launch(
            kernel=self._prepare_zu_minus_zl_and_zbu_minus_zbl_kernel,
            dim=(data.batch_size, data.m + data.n),
            inputs=[
                result.z_u, result.z_l,
                result.z_bl, result.z_bu,
                pc.x_b_scaling,
                self._kkt_system._inv_idx_hu, self._kkt_system._inv_idx_hl,
                self._kkt_system._inv_idx_xu, self._kkt_system._inv_idx_xl,
                self._work_z_1, self._work_x,
            ],
            stream=wp_stream,
        )

        G_x = self._work_z_2
        GT_zu_minus_zl = self._step.x
        if data.m > 0:
            self._kkt_system.eval_G_xn(data, 1., result.x, G_x)
            self._kkt_system.eval_GT_xt(data, 1., self._work_z_1, GT_zu_minus_zl)
        else:
            G_x.fill(0.)
            GT_zu_minus_zl.fill(0.)

        wp.launch_tiled(
            kernel=self._update_residual_nr_kernel,
            dim=[data.batch_size],
            inputs=[
                res_nr.x,            # minus_Px
                self._res.y,         # A_x = A*x
                self._res.x,         # AT_y
                G_x,
                GT_zu_minus_zl,      # GT_zh_assembled
                self._work_x,        # zb_assembled = x_b_scaling*(z_bu - z_bl)
                # Data
                data.c, data.b, data.h_l, data.h_u, data.x_l, data.x_u,
                # Result variables
                result.x, result.y,
                result.z_l, result.z_u, result.z_bl, result.z_bu,
                result.s_l, result.s_u, result.s_bl, result.s_bu,
                # Preconditioner
                pc.x_b_scaling, pc.cost_scaling_inv,
                pc.delta_inv, pc.delta_b_inv,
                self._constraints_rhs_inf_norm_unscaled,
                # Index maps
                data.idx_hl, data.idx_hu, data.idx_xl, data.idx_xu,
                # Residual outputs
                res_nr.x, res_nr.y,
                res_nr.z_l, res_nr.z_u, res_nr.z_bl, res_nr.z_bu,
                # Info outputs
                info.primal_obj,
                info.dual_obj,
                info.duality_gap,
                info.duality_gap_rel,
                info.primal_res,
                info.primal_res_rel,
                info.dual_res,
                info.dual_res_rel,
                info.prev_primal_res,
                info.prev_dual_res,
            ],
            block_dim=256,
            stream=wp_stream,
        )

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
        pc = self._preconditioner
        wp.launch_tiled(
            kernel=self._update_residuals_r_kernel,
            dim=[self._data.batch_size],
            inputs=[
                self._result.info.rho, self._result.info.delta,
                self._res_nr.x, self._res_nr.duals_all,
                self._result.x, self._result.duals_all,
                self._prox_vars.x, self._prox_vars.duals_all,
                self._res.x, self._res.duals_all,
                pc.dual_res_unscale_factor, pc.primal_res_unscale_factor,
                self._result.info.primal_res, self._result.info.primal_res_rel,
                self._result.info.dual_res, self._result.info.dual_res_rel,
                self._result.info.primal_res_reg, self._result.info.primal_res_reg_rel,
                self._result.info.dual_res_reg, self._result.info.dual_res_reg_rel,
                self._result.info.primal_prox_inf, self._result.info.dual_prox_inf,
            ],
            block_dim=256,
            device="cuda",
            stream=wp.Stream(cuda_stream=cp.cuda.get_current_stream().ptr),
        )

    @nvtx.annotate("Solver::_primal_res_nr")
    def _primal_res_nr(self):
        pc = self._preconditioner
        n, p = self._data.n, self._data.p
        offset = 0
        self._work_duals[:, :p] = self._res_nr.y
        self._work_duals[:, :p] *= pc.delta_inv[:, n:n + p]
        offset += p
        self._work_duals[:, offset:offset+self._data.num_hu] = self._res_nr.z_u
        self._work_duals[:, offset:offset+self._data.num_hu] *= pc.delta_inv[:, n + p + self._data.idx_hu]
        offset += self._data.num_hu
        self._work_duals[:, offset:offset+self._data.num_hl] = self._res_nr.z_l
        self._work_duals[:, offset:offset+self._data.num_hl] *= pc.delta_inv[:, n + p + self._data.idx_hl]
        offset += self._data.num_hl
        self._work_duals[:, offset:offset+self._data.num_xu] = self._res_nr.z_bu
        self._work_duals[:, offset:offset+self._data.num_xu] *= pc.delta_b_inv[:, self._data.idx_xu]
        offset += self._data.num_xu
        self._work_duals[:, offset:offset+self._data.num_xl] = self._res_nr.z_bl
        self._work_duals[:, offset:offset+self._data.num_xl] *= pc.delta_b_inv[:, self._data.idx_xl]
        offset += self._data.num_xl
        if offset > 0:
            cp.absolute(self._work_duals[:, :offset], out=self._work_duals[:, :offset])
            cp.max(self._work_duals[:, :offset], axis=1, out=self._work_residual)
        else:
            self._work_residual.fill(0.)
        return self._work_residual

    @nvtx.annotate("Solver::_primal_res_r")
    def _primal_res_r(self):
        pc = self._preconditioner
        n, p = self._data.n, self._data.p
        offset = 0
        self._work_duals[:, :p] = self._res.y
        self._work_duals[:, :p] *= pc.delta_inv[:, n:n + p]
        offset = p
        self._work_duals[:, offset:offset+self._data.num_hu] = self._res.z_u
        self._work_duals[:, offset:offset+self._data.num_hu] *= pc.delta_inv[:, n + p + self._data.idx_hu]
        offset += self._data.num_hu
        self._work_duals[:, offset:offset+self._data.num_hl] = self._res.z_l
        self._work_duals[:, offset:offset+self._data.num_hl] *= pc.delta_inv[:, n + p + self._data.idx_hl]
        offset += self._data.num_hl
        self._work_duals[:, offset:offset+self._data.num_xu] = self._res.z_bu
        self._work_duals[:, offset:offset+self._data.num_xu] *= pc.delta_b_inv[:, self._data.idx_xu]
        offset += self._data.num_xu
        self._work_duals[:, offset:offset+self._data.num_xl] = self._res.z_bl
        self._work_duals[:, offset:offset+self._data.num_xl] *= pc.delta_b_inv[:, self._data.idx_xl]
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
        pc = self._preconditioner
        n = self._data.n
        cp.absolute(self._res_nr.x, out=self._work_primals)
        self._work_primals *= pc.delta_inv[:, :n]
        self._work_primals *= pc.cost_scaling_inv[:, None]
        cp.max(self._work_primals, axis=1, out=self._work_residual)
        return self._work_residual

    @nvtx.annotate("Solver::_dual_res_r")
    def _dual_res_r(self):
        pc = self._preconditioner
        n = self._data.n
        cp.absolute(self._res.x, out=self._work_primals)
        self._work_primals *= pc.delta_inv[:, :n]
        self._work_primals *= pc.cost_scaling_inv[:, None]
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
        info = self._result.info
        settings = self.settings
        n = self._data.n
        num_duals = self._data.p + self._data.num_ineq
        wp.launch(
            kernel=self._update_rho_delta_with_ineq_kernel,
            dim=(self._data.batch_size, n + num_duals),
            inputs=[
                info.dual_res, info.prev_dual_res, info.dual_res_rel, info.dual_prox_inf,
                info.primal_res, info.prev_primal_res, info.primal_res_rel, info.primal_prox_inf,
                info.reg_limit,
                info.rho, info.delta,
                info.no_primal_update, info.no_dual_update,
                self._result.x, self._prox_vars.x,
                self._result.duals_all, self._prox_vars.duals_all,
                settings.eps_abs,
                settings.eps_rel,
                settings.reg_finetune_lower_limit,
                settings.infeasibility_threshold,
                info.iter[0],
            ],
            device="cuda",
            stream=wp.Stream(cuda_stream=cp.cuda.get_current_stream().ptr),
        )

    @nvtx.annotate("Solver::_update_rho_delta_without_ineq")
    def _update_rho_delta_without_ineq(self) -> None:
        info = self._result.info
        settings = self.settings
        n = self._data.n
        p = self._data.p
        wp.launch(
            kernel=self._update_rho_delta_without_ineq_kernel,
            dim=(self._data.batch_size, n + p),
            inputs=[
                info.dual_res, info.prev_dual_res, info.dual_res_rel, info.dual_prox_inf,
                info.primal_res, info.prev_primal_res, info.primal_res_rel, info.primal_prox_inf,
                info.reg_limit,
                info.rho, info.delta,
                info.no_primal_update, info.no_dual_update,
                self._result.x, self._prox_vars.x,
                self._result.y, self._prox_vars.y,
                settings.eps_abs,
                settings.eps_rel,
                settings.infeasibility_threshold,
                info.iter[0],
            ],
            device="cuda",
            stream=wp.Stream(cuda_stream=cp.cuda.get_current_stream().ptr),
        )



SolverBase = Solver
