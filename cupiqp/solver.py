from abc import ABC, abstractmethod
from typing import Optional, Any, Literal, List

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
    create_init_guess_rhs_kernel,
    create_init_guess_project_to_central_path_kernel,
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
    create_backward_assemble_rhs_kernel,
    create_backward_unscale_lhs_kernel,
    create_backward_compute_vector_grad_kernel,
    create_backward_copy_kernel,
    create_backward_pack_full_layout_kernel,
)


wp.config.quiet = True  # disable warp module initialization messages.
wp.config.enable_backward = False  # disable backward mode, cut down kernel compile time
wp.init()


class SolverBase(ABC):
    """Abstract base for the cuPIQP solver."""

    def __init__(self, dtype: Literal["float32", "float64"] = "float64"):
        if dtype not in ("float32", "float64"):
            raise ValueError(
                f"Solver dtype must be 'float32' or 'float64'; got {dtype!r}."
            )
        self._settings = Settings.for_dtype(dtype)
        self._data: Data = None
        self._result = Result()    # store the values of primal, dual and slack variables of current iteration, and other information
        self._step = Variables()   # used to store the step direction of primal and dual variables
        self._res_nr = Variables()  # used to store the non-regularized residuals
        self._res = Variables()  # used to store the regularized residuals
        self._prox_vars = Variables()  # used to store the proximal variables
        self._kkt_system = KKTSystem()
        self._preconditioner = None
        self._setup_done = False

    @property
    def settings(self) -> Settings:
        """Solver configuration (a ``Settings`` dataclass).

        Mutate its fields before ``setup()`` or between solves, e.g.
        ``solver.settings.verbose = True``.
        """
        return self._settings

    @settings.setter
    def settings(self, value: Settings) -> None:
        self._settings = value

    @property
    def data(self) -> Data:
        """The problem data built by ``setup()`` (a ``Data`` subclass), or
        ``None`` before ``setup()`` has been called."""
        return self._data

    @property
    def result(self) -> Result:
        """The latest solution and per-problem info (a ``Result``), populated
        by ``solve()``."""
        return self._result
    
    @nvtx.annotate("Solver::setup")
    def setup(self, P, c, A=None, b=None, G=None, h_u=None, h_l=None, x_u=None, x_l=None):
        """Bind the problem data and prepare the solver for ``solve()``.

        Fixes the problem *structure* - array shapes, which constraint
        blocks are present, the sparsity pattern (sparse backend), and the
        finite/infinite pattern of the bounds - and allocates all GPU
        buffers, the KKT system, and the preconditioner. Call this **once**
        per solver instance, then call ``solve()``.

        Pass a single problem (2D ``P``) or a batch (3D ``P`` with a leading
        batch axis, or a list of matrices); the batch size is inferred here
        and sets the shape of the result.

        Parameters
        ----------
        P : GPU array
            Quadratic cost, shape ``(n, n)`` or batched ``(B, n, n)``. Must
            be symmetric positive semidefinite. Required.
        c : GPU array
            Linear cost, shape ``(n,)`` or ``(B, n)``. Required.
        A, b : GPU array, optional
            Equality constraints ``A x = b``; shapes ``(p, n)`` and ``(p,)``
            (or batched). Omit for no equality constraints.
        G, h_l, h_u : GPU array, optional
            Two-sided inequalities ``h_l <= G x <= h_u``; ``G`` is
            ``(m, n)`` (or batched) and the bounds are ``(m,)``. Use ``-inf``
            / ``+inf`` entries for one-sided rows.
        x_l, x_u : GPU array, optional
            Element-wise box bounds ``x_l <= x <= x_u``, shape ``(n,)`` (or
            batched). Use ``+/-inf`` for unbounded entries.

        Raises
        ------
        RuntimeError
            If ``setup()`` has already been called on this instance. The
            structure is fixed after setup - create a new solver for a
            different structure, or use ``update()`` to change only the
            numerical values.
        TypeError
            If an input is not a GPU array of the kind this backend expects
            (e.g. a CPU ``numpy`` array, or a dense matrix passed to the
            sparse backend). See the backend's class docstring for the exact
            accepted types.

        See Also
        --------
        solve : run the solver after setup.
        update : change numerical data without a full re-setup.
        """
        if self._setup_done:
            raise RuntimeError(
                "setup() may only be called once per solver instance; "
                "create a new solver instance to set up a different problem."
            )

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
        self._info_host = InfoHost(B, dtype=self._data.dtype)
        # Problems in the batch that have terminated must not evolve while other problems continue iterating.
        self._unsolved_mask = cp.ones(B, dtype=cp.bool_)

        self._dtype = self._data.dtype
        self._work_z_1 = cp.empty((data.batch_size, data.m), dtype=self._dtype)  # used to store intermediate results in _update_residuals_nr
        self._work_z_2 = cp.empty((data.batch_size, data.m), dtype=self._dtype)  # used to store intermediate results in _update_residuals_nr

        self._work_z = cp.empty((data.batch_size, data.num_ineq), dtype=self._dtype)  # used in _calculate_step to hold all concatenated slack or dual steps / results
        self._work_s = cp.empty((data.batch_size, data.num_ineq), dtype=self._dtype)  # used in _calculate_step to hold all concatenated slack or dual steps / results
        self._work_primals = cp.empty((data.batch_size, data.n), dtype=self._dtype)
        self._work_duals = cp.empty((data.batch_size, data.p + data.num_ineq), dtype=self._dtype)  # used to hold the concatenated dual variables for computing the residuals in _update_residuals_nr
        self._work_residual = cp.empty((data.batch_size, ), dtype=self._dtype)
        self._work_reduce = cp.empty((data.batch_size, 8), dtype=self._dtype)  # used to hold the intermediate results of the reductions related to s_l, s_u, s_bl, s_bu and z_l, z_u, z_bl, z_bu

        self._init_warp_kernels()

        self._work_x = cp.empty((B, self._data.n), dtype=self._dtype)

        self._tau_device = cp.empty(1, dtype=self._dtype)
        self._tau_device[0] = self.settings.tau  # device copy used by warp kernels
        self._tau_host = float(self.settings.tau)  # host cache -- only H2D when tau actually changes

        # Pre-allocated (B,) buffers for CUDA-graph-safe norm computations in _update_residuals_nr / _update_residuals_r
        self._work_primal_rel_norm = cp.empty(B, dtype=self._dtype)  # running max of primal relative norm terms
        self._work_dual_res_norm = cp.empty(B, dtype=self._dtype)    # running max of dual residual norm terms
        self._work_norm_temp = cp.empty(B, dtype=self._dtype)        # temp (B,) for individual norm results

        if self.settings.enable_grad:
            # Working variables for implicit differentiation
            self._work_grad_rhs = Variables()
            self._work_grad_rhs.init(self._data)
            # User cotangent input (caller packs kwargs into this) and the user-space adjoint solution buffer.
            self._grad_in = Variables()
            self._grad_in.init(self._data)
            self._backward_adjoint_vector = Variables()
            self._backward_adjoint_vector.init(self._data)
            # Pre-zeroed Variables used as a placeholder for None cotangents
            # in the fused pack kernel — the kernel can't read None, so
            # absent kwargs get substituted with the corresponding field of
            # this zero buffer.
            self._zero_grad_in = Variables()
            self._zero_grad_in.init(self._data)
            self._zero_grad_in._primal_buffer.fill(0.0)
            self._zero_grad_in._dual_buffer.fill(0.0)
            # Full-layout scatter buffers feeding the matrix and vector
            # gradient assemblies. ineq groups live in length-m; bound
            # groups live in length-n.
            self._lam_zu_full  = cp.empty((B, data.m), dtype=self._dtype)
            self._lam_zl_full  = cp.empty((B, data.m), dtype=self._dtype)
            self._lam_zbu_full = cp.empty((B, data.n), dtype=self._dtype)
            self._lam_zbl_full = cp.empty((B, data.n), dtype=self._dtype)
            self._zu_full      = cp.empty((B, data.m), dtype=self._dtype)
            self._zl_full      = cp.empty((B, data.m), dtype=self._dtype)

        self._enable_iterative_refinement = self.settings.iterative_refinement_always_enabled

        # Unscaled-RHS inf-norm. When preconditioner_iter == 0 the stored
        # factors are identity, so this reduces to the inf-norm of the user-
        # space b / h_l/u / x_l/u — same answer, single code path.
        self._constraints_rhs_inf_norm_unscaled = cp.zeros(B, dtype=self._dtype)
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
        """Change the numerical problem data, then ``solve()`` again.

        The fast path for re-solving a problem of the **same structure** -
        for example a moving target ``b`` or a re-linearized ``P`` in
        receding-horizon control. It reuses every GPU allocation from
        ``setup()``, so only the values change; shapes, sparsity patterns,
        and which blocks are present must stay the same (create a new solver
        for a structural change). Bound *values* may change freely, including
        which entries are ``+/-inf`` - a bound can flip between finite and
        infinite without re-``setup()``.

        Any argument left as ``None`` keeps its current value. After
        ``update()``, call ``solve()`` to get the new solution.

        Parameters
        ----------
        P, c, A, b, G, h_u, h_l, x_u, x_l : GPU array, optional
            New values for the corresponding problem block. ``None`` (the
            default) leaves that block unchanged. Must match the original
            shapes / sparsity pattern set at ``setup()``.
        check_validity : bool, default: False
            If ``True``, validate the dimensions and sparsity of the new
            data. Defaults to ``False`` for speed (validation forces
            device-to-host syncs in the sparse backend). When ``False``, you
            must still keep the shapes and sparsity patterns of ``P``/``A``/
            ``G`` unchanged; bound values (including which entries are
            ``+/-inf``) may change.
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

        # NOTE: Since we allow changing h_l/h_u containing arbitrary +inf/-inf, 
        # an inequality row G[i] can switch between active (a finite
        # bound) and inactive (both bounds infinite) between updates.
        # If either of h_l or h_u are updated, we need to update G 
        # because for sparse kkt solver we need to set the inactive rows to 0
        ineq_bound_pattern_may_change = h_l is not None or h_u is not None

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
            (G is not None) or preconditioner_did_fresh_ruiz or ineq_bound_pattern_may_change,
        )

    def solve(self) -> List[Status]:
        """Solve the QP set up by ``setup()`` and return the solve status.

        Runs the proximal interior-point iterations on the GPU. The full
        solution (primal ``x``, dual, and slack variables) and per-problem
        diagnostics are written to ``solver.result``; this method returns the
        status for convenience.

        Returns
        -------
        Status or list of Status
            For a single problem, the ``Status``. For a batched ``setup()``,
            a list of ``B`` of them (one per problem). ``CUPIQP_SOLVED``
            means the problem converged to tolerance. The list form is always
            available as ``solver.result.info.status``.

        Notes
        -----
        Read the solution from ``solver.result`` after solving - e.g.
        ``solver.result.x`` (shape ``(B, n)``) and
        ``solver.result.info.status``. Set ``solver.settings.verbose = True``
        to print a per-iteration log. After ``setup()`` you may ``solve()``
        repeatedly, optionally calling ``update()`` in between to change the
        numerical data.
        """
        if self.settings.verbose:
            try:
                from importlib.metadata import version
                _ver = version("cupiqp")
            except Exception:
                _ver = ""
            _w = 58
            print("-" * _w)
            print(f"cuPIQP v{_ver} - GPU-accelerated PIQP solver".strip().center(_w))
            print("(c) Fenglong Song".center(_w))
            print("Ecole Polytechnique Federale de Lausanne (EPFL) 2026".center(_w))
            print("-" * _w)
            if self.settings.kkt_solver == "dense_cholesky":
                print("dense backend:")
                print(f"batch size B = {self._data.batch_size}")
                print(f"variables n = {self._data.n}")
                print(f"equality constraints p = {self._data.p}")
                print(f"inequality constraints m = {self._data.m}")
            elif self.settings.kkt_solver == "sparse_ldlt":
                print("sparse backend:")
                print(f"batch size B = {self._data.batch_size}")
                print(f"variables n = {self._data.n}, nnz(P) = {self._data.P.nnz}")
                print(f"equality constraints p = {self._data.p}, nnz(A) = {self._data.A.nnz}")
                print(f"inequality constraints m = {self._data.m}, nnz(G) = {self._data.G.nnz}")
            elif self.settings.kkt_solver == "multistage_block_cholesky":
                print("multistage backend:")
                print(f"batch size B = {self._data.batch_size}")
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

    def _solve_impl(self) -> List[Status]:
        self._result.info.status_value[:] = Status.CUPIQP_UNSOLVED.value
        self._unsolved_mask.fill(True)
        self._result.info.iter[:] = 0
        self._result.info.iter_total = 0
        self._iter = 0  # global IPM iteration counter (host scalar)
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
        still_unsolved = np.ones(self._data.batch_size, dtype=np.bool_)

        ## ---------------------------------------------
        ## ---------- remaining iterations -------------
        ## ---------------------------------------------
        for iter in range(self.settings.max_iter):
            with nvtx.annotate(f"Solver::ipm_iteration"):
                self._iter = iter
                self._result.info.iter[still_unsolved] = iter
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

                # convergence check
                primal_ok = (info_host.primal_res < settings.eps_abs) | (info_host.primal_res_rel < settings.eps_rel)
                dual_ok = (info_host.dual_res < settings.eps_abs) | (info_host.dual_res_rel < settings.eps_rel)
                converged = primal_ok & dual_ok
                if settings.check_duality_gap:
                    gap_ok = (info_host.duality_gap < settings.eps_duality_gap_abs) | (info_host.duality_gap_rel < settings.eps_duality_gap_rel)
                    converged &= gap_ok
                solved = still_unsolved & converged
                self._result.info.status_value[solved] = Status.CUPIQP_SOLVED.value  # CPU write

                # primal infeasibility check
                primal_infeasible = still_unsolved & ~converged & (
                    (info_host.no_dual_update > min(5, settings.reg_finetune_dual_update_threshold)) &
                    (info_host.primal_prox_inf > settings.infeasibility_threshold) &
                    ((info_host.primal_res_reg < settings.eps_abs) | (info_host.primal_res_reg_rel < settings.eps_rel))
                )
                self._result.info.status_value[primal_infeasible] = Status.CUPIQP_PRIMAL_INFEASIBLE.value  # CPU write

                # dual infeasibility check
                dual_infeasible = still_unsolved & ~converged & ~primal_infeasible & (
                    (info_host.no_primal_update > min(5, settings.reg_finetune_primal_update_threshold)) &
                    (info_host.dual_prox_inf > settings.infeasibility_threshold) &
                    ((info_host.dual_res_reg < settings.eps_abs) | (info_host.dual_res_reg_rel < settings.eps_rel))
                )
                self._result.info.status_value[dual_infeasible] = Status.CUPIQP_DUAL_INFEASIBLE.value  # CPU write

                newly_terminated = solved | primal_infeasible | dual_infeasible
                mask_changed = np.any(newly_terminated)
                if mask_changed:
                    still_unsolved[newly_terminated] = False

                if self.settings.verbose:
                    self._print_iteration_info()

                if mask_changed:
                    # No subsequent GPU work is launched when the entire batch is done.
                    if not np.any(still_unsolved):
                        break
                    self._unsolved_mask.set(still_unsolved)

                # avoid getting too close to boundary which can result in a division by zero
                if self._data.num_ineq > 0:
                    wp.launch(
                        kernel=self._boundary_shift_kernel,
                        dim=(self._data.batch_size,
                             self._data.num_hl + self._data.num_hu
                             + self._data.num_xl + self._data.num_xu),
                        inputs=[
                            self._unsolved_mask,
                            self._data.finite_mask_hl, self._data.finite_mask_hu,
                            self._data.finite_mask_xl, self._data.finite_mask_xu,
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
                finetune_mask &= still_unsolved
                if np.any(finetune_mask):
                    self._result.info.reg_limit[finetune_mask] = self.settings.reg_finetune_lower_limit
                    finetune_mask_dev = cp.asarray(finetune_mask)
                    self._result.info.no_primal_update[finetune_mask_dev] = 0
                    self._result.info.no_dual_update[finetune_mask_dev] = 0

                self._update_and_factorize_kkt()
                if np.any(self._result.info.status_value == Status.CUPIQP_NUMERICAL_ISSUES.value):
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

        self._result.info.iter_total = int(self._iter)
        # Mark remaining unsolved as max iter reached
        self._result.info.status_value[self._result.info.status_value == Status.CUPIQP_UNSOLVED.value] = Status.CUPIQP_MAX_ITER_REACHED.value
        if self.settings.verbose:
            self._print_summary()
        if self.settings.preconditioner_iter > 0:
            self._preconditioner.unscale_solution(self._result, self._data)
        statuses = self._result.info.status
        
        return statuses

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

        total_t = (self._data.n + self._data.p
                   + self._data.num_hl + self._data.num_hu
                   + self._data.num_xl + self._data.num_xu)
        wp.launch(
            kernel=self._initial_guess_rhs_kernel,
            dim=(self._data.batch_size, total_t),
            inputs=[
                self._data.c, self._data.b,
                self._data.h_l, self._data.h_u,
                self._data.x_l, self._data.x_u,
                self._data.finite_mask_hl, self._data.finite_mask_hu,
                self._data.finite_mask_xl, self._data.finite_mask_xu,
                self._res.x, self._res.y,
                self._res.z_l, self._res.z_u,
                self._res.z_bl, self._res.z_bu,
            ],
            device="cuda",
            stream=wp.Stream(cuda_stream=cp.cuda.get_current_stream().ptr),
        )
        self._res.s_all[:] = 0.

        self._kkt_system.solve(self._data, self._preconditioner, self.settings, self._res, self._result)  # getting an initial point of _result

        if self._data.num_hl + self._data.num_hu + self._data.num_xl + self._data.num_xu > 0:
            ## ----------- keep z and s non-negative --------------
            # this is according to the IV.A part of Roland Schwan 2023 paper.
            # Uses pre-allocated (B, 1) scratch buffers — see Solver.setup
            # for the rationale (no transient cupy allocs in the solve path).
            delta_s = -cp.min(self._result.s_all, axis=1, keepdims=True)  # (B, 1)
            delta_z = -cp.min(self._result.z_all, axis=1, keepdims=True)  # (B, 1)
            self._result.s_all += delta_s
            self._result.z_all += delta_z

            # need to make sure mu is positive here, otherwise in the next step (put s and z on central path) sqrt(mu) the computed z_* will be zeros
            self._calculate_mu()
            cp.clip(self._result.info.mu, 1e-10, None, out=self._result.info.mu)

            # put s and z on the central path
            # c = z - delta_z; z = (c + sqrt(c^2 + 4*mu)) / 2; s = z - c
            wp.launch(
                kernel=self._init_guess_project_to_central_path_kernel,
                dim=(self._data.batch_size, self._data.num_ineq),
                inputs=[delta_z, self._result.info.mu, self._data.finite_mask_all,
                        self._result.z_all, self._result.s_all],
                device="cuda",
                stream=wp.Stream(cuda_stream=cp.cuda.get_current_stream().ptr),
            )

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
                f"{self._iter:3d}   "
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
            solved  = B - int((self._result.info.status_value == Status.CUPIQP_UNSOLVED.value).sum())
            counter = f"{solved}/{B}"
            counter_w = max(2 * len(str(B)) + 1, len("solved"))
            print(
                f"{self._iter:>4d}  "
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

    @nvtx.annotate("Solver::_print_summary")
    def _print_summary(self):
        statuses = self._result.info.status
        labels = {
            "Solved":             Status.CUPIQP_SOLVED,
            "Max iter reached":   Status.CUPIQP_MAX_ITER_REACHED,
            "Primal infeasible":  Status.CUPIQP_PRIMAL_INFEASIBLE,
            "Dual infeasible":    Status.CUPIQP_DUAL_INFEASIBLE,
            "Numerical issues":   Status.CUPIQP_NUMERICAL_ISSUES,
        }
        print(f"\nFinished in {self._result.info.iter_total} iterations", flush=True)
        for name, status in labels.items():
            count = statuses.count(status)
            if count > 0:
                print(f"  {name + ':':<20} {count}/{len(statuses)}", flush=True)

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
            still_unsolved = (self._result.info.status_value == Status.CUPIQP_UNSOLVED.value)
            self._result.info.status_value[still_unsolved] = Status.CUPIQP_NUMERICAL_ISSUES.value

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
                dtype=self._data.dtype
                )
            self._prepare_predictor_step_kernel = create_prepare_predictor_step_kernel(dtype=self._data.dtype)
            self._prepare_corrector_step_kernel = create_prepare_corrector_step_kernel(dtype=self._data.dtype)
            self._update_vars_after_corrector_step_kernel = create_update_vars_after_corrector_step_kernel(
                n=self._data.n, p=self._data.p, num_ineq=self._data.num_ineq,
                dtype=self._data.dtype
                )

        # Tile-based kernels
        self._update_residuals_r_kernel = create_update_residuals_r_kernel(
            self._data.n, self._data.p,
            int(self._data.num_hu), int(self._data.num_hl),
            int(self._data.num_xu), int(self._data.num_xl),
            dtype=self._data.dtype
            )
        self._prepare_zu_minus_zl_and_zbu_minus_zbl_kernel = create_prepare_zu_minus_zl_and_zbu_minus_zbl_kernel(
            self._data.m, self._data.n,
            has_x_l=self._data.has_x_l, has_x_u=self._data.has_x_u,
            dtype=self._data.dtype
            )
        self._update_residual_nr_kernel = create_update_residual_nr_kernel(
            self._data.n, self._data.p, self._data.m,
            self._data.num_hl, self._data.num_hu, self._data.num_xl, self._data.num_xu,
            dtype=self._data.dtype
            )
        self._initial_guess_rhs_kernel = create_init_guess_rhs_kernel(
            self._data.n, self._data.p,
            self._data.num_hl, self._data.num_hu,
            self._data.num_xl, self._data.num_xu,
            dtype=self._data.dtype,
        )
        if self._data.num_ineq > 0:
            self._calculate_sigma_kernel = create_calculate_sigma_kernel(self._data.num_ineq, dtype=self._data.dtype)
            self._calculate_step_kernel = create_calculate_step_kernel(self._data.num_ineq, dtype=self._data.dtype)
            self._calculate_mu_kernel = create_calculate_mu_kernel(self._data.num_ineq, dtype=self._data.dtype)
            self._init_guess_project_to_central_path_kernel = create_init_guess_project_to_central_path_kernel(dtype=self._data.dtype)
            self._update_rho_delta_with_ineq_kernel = create_update_rho_delta_with_ineq_kernel(
                self._data.n, self._data.p + self._data.num_ineq,
                dtype=self._data.dtype
                )
        else:
            self._run_full_newton_step_kernel = create_run_full_newton_step_kernel(self._data.n, self._data.p, dtype=self._data.dtype)
            self._update_rho_delta_without_ineq_kernel = create_update_rho_delta_without_ineq_kernel(
                self._data.n, self._data.p,
                dtype=self._data.dtype
                )

        # Adjoint/backward-pass kernels
        if self.settings.enable_grad:
            n, p = self._data.n, self._data.p
            nhu, nhl = self._data.num_hu, self._data.num_hl
            nxu, nxl = self._data.num_xu, self._data.num_xl
            precond_on = self.settings.preconditioner_iter > 0
            self._backward_assemble_rhs_kernel = create_backward_assemble_rhs_kernel(
                n, p, nhu, nhl, nxu, nxl, precond_on, dtype=self._data.dtype)
            self._backward_unscale_lhs_kernel = create_backward_unscale_lhs_kernel(
                n, p, nhu, nhl, nxu, nxl, precond_on, dtype=self._data.dtype)
            self._backward_compute_vector_grad_kernel = create_backward_compute_vector_grad_kernel(
                n, p, nhu, nhl, nxu, nxl, dtype=self._data.dtype)
            self._backward_pack_full_layout_kernel = create_backward_pack_full_layout_kernel(
                self._data.m, n, nxl, nxu, dtype=self._data.dtype)
            self._backward_copy_kernel = create_backward_copy_kernel(
                n, p, nhu, nhl, nxu, nxl, dtype=self._data.dtype)

    @nvtx.annotate("Solver::_run_full_newton_step")
    def _run_full_newton_step(self):
        self._kkt_system.solve(self._data, self._preconditioner, self.settings, self._res, self._step)
        wp.launch(
            kernel=self._run_full_newton_step_kernel,
            dim=(self._data.batch_size, self._data.n + self._data.p),
            inputs=[
                self._unsolved_mask,
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
            inputs=[self._result.s_all, self._result.z_all, self._data.finite_mask_all, self._res.s_all],
            device="cuda",
            stream=wp.Stream(cuda_stream=cp.cuda.get_current_stream().ptr),
        )

        self._kkt_system.solve(self._data, self._preconditioner, self.settings, self._res, self._step)

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
                self._data.finite_mask_all,
                self._res.s_all,
            ],
            device="cuda",
            stream=wp.Stream(cuda_stream=cp.cuda.get_current_stream().ptr),
        )

        self._kkt_system.solve(self._data, self._preconditioner, self.settings, self._res, self._step)

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
                self._unsolved_mask,
                self._data.finite_mask_all,
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
                self._data.finite_mask_all,
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
                self._data.finite_mask_all, self._data.num_finite_bounds,
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
                self._data.finite_mask_all, self._data.num_finite_bounds,
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
                data.finite_mask_hl, data.finite_mask_hu, data.finite_mask_xl, data.finite_mask_xu,
                # Result variables
                result.x, result.y,
                result.z_l, result.z_u, result.z_bl, result.z_bu,
                result.s_l, result.s_u, result.s_bl, result.s_bu,
                # Preconditioner
                pc.x_b_scaling, pc.cost_scaling_inv,
                pc.delta_inv, pc.delta_b_inv,
                self._constraints_rhs_inf_norm_unscaled,
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
        self._work_duals[:, offset:offset+self._data.num_hu] *= pc.delta_inv[:, n + p:n + p + self._data.num_hu]
        offset += self._data.num_hu
        self._work_duals[:, offset:offset+self._data.num_hl] = self._res_nr.z_l
        self._work_duals[:, offset:offset+self._data.num_hl] *= pc.delta_inv[:, n + p:n + p + self._data.num_hl]
        offset += self._data.num_hl
        self._work_duals[:, offset:offset+self._data.num_xu] = self._res_nr.z_bu
        self._work_duals[:, offset:offset+self._data.num_xu] *= pc.delta_b_inv[:, :self._data.num_xu]
        offset += self._data.num_xu
        self._work_duals[:, offset:offset+self._data.num_xl] = self._res_nr.z_bl
        self._work_duals[:, offset:offset+self._data.num_xl] *= pc.delta_b_inv[:, :self._data.num_xl]
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
        self._work_duals[:, offset:offset+self._data.num_hu] *= pc.delta_inv[:, n + p:n + p + self._data.num_hu]
        offset += self._data.num_hu
        self._work_duals[:, offset:offset+self._data.num_hl] = self._res.z_l
        self._work_duals[:, offset:offset+self._data.num_hl] *= pc.delta_inv[:, n + p:n + p + self._data.num_hl]
        offset += self._data.num_hl
        self._work_duals[:, offset:offset+self._data.num_xu] = self._res.z_bu
        self._work_duals[:, offset:offset+self._data.num_xu] *= pc.delta_b_inv[:, :self._data.num_xu]
        offset += self._data.num_xu
        self._work_duals[:, offset:offset+self._data.num_xl] = self._res.z_bl
        self._work_duals[:, offset:offset+self._data.num_xl] *= pc.delta_b_inv[:, :self._data.num_xl]
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
                self._unsolved_mask,
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
                wp.int32(self._iter),
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
                self._unsolved_mask,
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
                wp.int32(self._iter),  # self._iter is int64, need to convert to int32
            ],
            device="cuda",
            stream=wp.Stream(cuda_stream=cp.cuda.get_current_stream().ptr),
        )

    @nvtx.annotate("Solver::_compute_adjoint")
    def _compute_adjoint(self, grad: Variables, sol: Variables) -> None:
        r"""Solve the adjoint KKT system :math:`K^\top \lambda = -\partial L / \partial v`.

        Backend-agnostic: the math operates only on cupy arrays and the
        cached KKT factor (which every backend's ``_kkt_system`` exposes
        with a uniform ``solve(..., transpose=True)`` API). Used by every
        subclass's ``grad()`` to obtain the lambdas; per-backend matrix
        and vector gradient assembly happens in the caller.

        Parameters
        ----------
        grad : Variables
            User cotangents :math:`\partial L / \partial v` for every
            variable group (``x, y, z_u, z_l, z_{bu}, z_{bl}, s_u, s_l,
            s_{bu}, s_{bl}``). Absent constraint groups have zero-sized
            fields and the kernel's t-range naturally skips them.
        sol : Variables
            **Output.** The adjoint solution :math:`\lambda` is written
            in-place into ``sol`` with the same field layout as ``grad``.
            Each ``sol.<field>`` aliases the corresponding lambda.

        Notes
        -----
        Cotangents in ``grad`` are interpreted in **user (un-scaled)**
        space. The adjoint KKT system is solved in scaled space when
        ``preconditioner_iter > 0``, but the scaled-to-user push-back is
        applied here so that ``sol`` is written in **user space**. The
        per-backend ``grad()`` only contributes the matrix-gradient
        push-back on top of that.

        Raises
        ------
        RuntimeError
            If :meth:`solve` has not been called yet (no cached KKT factor).
        """
        if not getattr(self, "_setup_done", False) or self._result is None:
            raise RuntimeError(
                f"{type(self).__name__}.grad() requires a prior solve(); "
                f"call setup() and solve() before grad()."
            )

        data = self._data
        settings = self.settings
        kkt_system = self._kkt_system
        precond = self._preconditioner
        B = data.batch_size
        n, p = data.n, data.p
        wp_stream = wp.Stream(cuda_stream=cp.cuda.get_current_stream().ptr)

        rhs = self._work_grad_rhs              # Variables pre-allocated in setup (scaled-space RHS)

        # ---- Step 1 (fused): rhs = -grad_v L, scaled to scaled space
        rhs_total = n + p + 2 * grad.num_ineq
        wp.launch(
            kernel=self._backward_assemble_rhs_kernel,
            dim=(B, rhs_total),
            inputs=[
                grad.x, grad.y,
                grad.z_u, grad.z_l, grad.z_bu, grad.z_bl,
                grad.s_u, grad.s_l, grad.s_bu, grad.s_bl,
                precond.delta, precond.delta_b,
                precond.delta_inv, precond.delta_b_inv,
                precond.cost_scaling_inv,
                rhs.x, rhs.y,
                rhs.z_u, rhs.z_l, rhs.z_bu, rhs.z_bl,
                rhs.s_u, rhs.s_l, rhs.s_bu, rhs.s_bl,
            ],
            device="cuda",
            stream=wp_stream,
        )

        # ---- Step 2: K^T sol = rhs (reuses cached forward factor).
        kkt_system.solve(data, precond, settings, rhs, sol, transpose=True)

        # ---- Step 3 (fused): un-scale sol from scaled space to user
        # space in place.
        wp.launch(
            kernel=self._backward_unscale_lhs_kernel,
            dim=(B, n + p + grad.num_ineq),
            inputs=[
                sol.x, sol.y,
                sol.z_u, sol.z_l, sol.z_bu, sol.z_bl,
                precond.delta, precond.delta_b, precond.cost_scaling,
            ],
            device="cuda",
            stream=wp_stream,
        )

    def _compute_vector_gradients(self, grad: Variables, sol: Variables) -> None:
        data = self._data
        B = data.batch_size
        wp.launch(
            kernel=self._backward_compute_vector_grad_kernel,
            dim=(B, data.n + data.p + data.num_ineq),
            inputs=[
                grad.x, grad.y,
                grad.z_u, grad.z_l, grad.z_bu, grad.z_bl,
                sol.x, sol.y,
                sol.z_u, sol.z_l, sol.z_bu, sol.z_bl,
            ],
            device="cuda",
            stream=wp.Stream(cuda_stream=cp.cuda.get_current_stream().ptr),
        )

    @nvtx.annotate("Solver::grad")
    def backward(self,
             grad_x=None, grad_y=None,
             grad_z_u=None, grad_z_l=None, grad_z_bu=None, grad_z_bl=None,
             grad_s_u=None, grad_s_l=None, grad_s_bu=None, grad_s_bl=None):
        r"""Compute gradients of an outer scalar :math:`L` w.r.t. problem
        data, given upstream cotangents on the solution variables.

        Orchestration (backend-agnostic):

        1. Pack the per-field cotangent kwargs into ``self._grad_in``
           (a pre-allocated ``Variables``); missing kwargs are treated
           as zeros.
        2. Solve the adjoint KKT system via :meth:`_compute_adjoint`,
           producing user-space adjoint vectors.
        3. Scatter the four active-size lambda groups (``z_u, z_l,
           z_bu, z_bl``) and the two active-size ineq result groups
           into full-``m`` / full-``n`` buffers
           (``self._lam_z*_full``, ``self._z*_full``). Both the dG
           outer product and the ``dh_*`` / ``dx_*`` vector gradients
           consume these full-layout buffers.
        4. Delegate to :meth:`_compute_data_gradients` for backend-
           specific matrix-gradient assembly + ``Data`` subclass
           construction.

        Returns the backend's ``Data`` subclass populated with the
        gradients in user space. Cotangents and returned gradients are
        interpreted in user (un-scaled) space throughout — the
        adjoint solve and scatter chain handles all preconditioner
        bookkeeping internally.

        Raises
        ------
        RuntimeError
            If :meth:`solve` has not been called yet (no cached KKT factor).
        """
        if not self.settings.enable_grad:
            raise RuntimeError("Set enable_grad to True to enable gradient computation.")

        if not getattr(self, "_setup_done", False) or self._result is None:
            raise RuntimeError(
                f"{type(self).__name__}.grad() requires a prior solve(); "
                f"call setup() and solve() before grad()."
            )

        data = self._data
        B = data.batch_size
        wp_stream = wp.Stream(cuda_stream=cp.cuda.get_current_stream().ptr)

        # ---- Step 1
        zeros = self._zero_grad_in
        pack_total = data.n + data.p + 2 * zeros.num_ineq
        if pack_total > 0:
            wp.launch(
                kernel=self._backward_copy_kernel,
                dim=(B, pack_total),
                inputs=[
                    grad_x    if grad_x    is not None else zeros.x,
                    grad_y    if grad_y    is not None else zeros.y,
                    grad_z_u  if grad_z_u  is not None else zeros.z_u,
                    grad_z_l  if grad_z_l  is not None else zeros.z_l,
                    grad_z_bu if grad_z_bu is not None else zeros.z_bu,
                    grad_z_bl if grad_z_bl is not None else zeros.z_bl,
                    grad_s_u  if grad_s_u  is not None else zeros.s_u,
                    grad_s_l  if grad_s_l  is not None else zeros.s_l,
                    grad_s_bu if grad_s_bu is not None else zeros.s_bu,
                    grad_s_bl if grad_s_bl is not None else zeros.s_bl,
                    self._grad_in.x,    self._grad_in.y,
                    self._grad_in.z_u,  self._grad_in.z_l,
                    self._grad_in.z_bu, self._grad_in.z_bl,
                    self._grad_in.s_u,  self._grad_in.s_l,
                    self._grad_in.s_bu, self._grad_in.s_bl,
                ],
                device="cuda",
                stream=wp_stream,
            )

        # ---- Step 2: adjoint KKT solve
        self._compute_adjoint(self._grad_in, self._backward_adjoint_vector)

        # ---- Step 3: write into full-layout buffers
        wp.launch(
            kernel=self._backward_pack_full_layout_kernel,
            dim=(B, 4 * data.m + data.num_xu + data.num_xl),
            inputs=[
                self._backward_adjoint_vector.z_u, self._backward_adjoint_vector.z_l,
                self._backward_adjoint_vector.z_bu, self._backward_adjoint_vector.z_bl,
                self._result.z_u, self._result.z_l,
                self._lam_zu_full, self._lam_zl_full,
                self._lam_zbu_full, self._lam_zbl_full,
                self._zu_full, self._zl_full,
            ],
            device="cuda",
            stream=wp_stream,
        )

        # ---- Step 4: backend-specific matrix and vector gradient
        # assembly + Data subclass construction.
        return self._compute_data_gradients(self._backward_adjoint_vector)

    @abstractmethod
    def _compute_data_gradients(self, sol_adj: Variables):
        """Build and return the backend's ``Data`` subclass populated
        with user-space gradients ``(P, c, A, b, G, h_u, h_l, x_u,
        x_l)``.

        Implementations read user-space adjoint lambdas from
        ``sol_adj`` (active-only sizes), the user-space primal/dual
        solution from ``self._result``, and the pre-scattered full-
        layout buffers ``self._lam_z*_full`` / ``self._z*_full`` set
        up by :meth:`grad`'s scatter step.
        """
