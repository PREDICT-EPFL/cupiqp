from abc import ABC, abstractmethod
from typing import Optional, Any, List, Union

import numpy as np
import warp as wp
import nvtx

from .settings import Settings
from .data import Data
from .results import Result, Status, Variables, InfoHost
from .kkt_systems import KKTSystem
from .utils import cuda_graph_capture
from .solver_kernels import (
    REDUCTION_BLOCK_DIM,
    create_init_guess_rhs_kernel,
    create_init_guess_center_kernel,
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
    create_update_smoothing_residual_nr_kernel,
    create_smoothing_prepare_kernel,
    create_smoothing_apply_step_kernel,
    create_update_rho_delta_with_ineq_kernel,
    create_update_prox_vars_kernel,
    create_update_rho_delta_without_ineq_kernel,
    create_run_full_newton_step_kernel,
    create_factor_retry_kernel,
    create_apply_finetune_kernel,
    create_backward_assemble_rhs_kernel,
    create_backward_unscale_lhs_kernel,
    create_backward_compute_vector_grad_kernel,
    create_backward_copy_kernel,
    create_backward_pack_full_layout_kernel
)


wp.config.quiet = True  # disable warp module initialization messages.
wp.config.enable_backward = False  # disable backward mode, cut down kernel compile time
wp.init()


class SolverBase(ABC):
    """Abstract base for the cuPIQP solver."""

    def __init__(self, dtype: Union[type[wp.float32], type[wp.float64]] = wp.float64, stream=None):
        if dtype is not wp.float32 and dtype is not wp.float64:
            raise TypeError(
                f"Solver dtype must be wp.float32 or wp.float64; got {dtype!r}."
            )
        # One CUDA stream per solver instance: every public entry point makes
        # it Warp's current stream, so all kernels, copies, library calls and
        # CUDA graphs of this solver are issued and captured on it. A stream
        # given to the constructor is borrowed (never destroyed here).
        if stream is not None and not isinstance(stream, wp.Stream):
            raise TypeError(
                "stream must be a warp.Stream or None; got "
                f"{type(stream).__name__}. Order other frameworks' streams "
                "with the solver through cupiqp.torch / cupiqp.jax, or with "
                "warp events on solver.stream."
            )
        self._stream = stream if stream is not None else wp.Stream("cuda")
        self._owns_stream = stream is None
        # All solver data lives on the device of the solver stream.
        self._device = self._stream.device
        self._dtype = dtype
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

        # Gradient smoothing (qpax-style relaxed-KKT differentiation) state.
        # The mode is frozen at setup(); the relaxation buffers (see setup) are
        # only allocated when it is on. The backward outer products read the
        # linearization point passed in by backward(): ``self._result`` (the true
        # optimum) in standard mode, or ``self._result_smoothed`` (the user-space
        # relaxed iterate) in smoothed mode.
        self._grad_smoothing = None
        self._result_scaled = Variables()
        self._result_smoothed = Variables()

    def __del__(self):
        """Release stream-bound components before the solver stream.

        Library handles (cuDSS, cuSPARSE, cuBLAS, cuSOLVER) and captured CUDA
        graphs are bound to ``self._stream``; their teardown may touch that
        stream, so it must still exist. Attribute release order is otherwise
        arbitrary, hence the explicit order here.
        """
        d = getattr(self, "__dict__", None)
        if not d:
            return
        for name in [n for n in d if n.startswith("_cuda_graphs_")]:
            d.pop(name, None)
        for name in ("_kkt_system", "_preconditioner", "_grad_data", "_data"):
            d.pop(name, None)
        # The stream goes last; a borrowed stream is only dereferenced.

    @property
    def stream(self) -> wp.Stream:
        """The CUDA stream this solver runs on, as a ``warp.Stream``.

        Every ``setup`` / ``update`` / ``solve`` / ``backward`` call is issued on
        it. It is the ``warp.Stream`` given to the constructor, or one the
        solver created. The stream is created blocking with respect to the
        legacy default stream, so work issued on the default stream (plain
        cupy / torch / numba code) is ordered with the solver automatically;
        producers or consumers on other streams must wait on this stream (or
        on an event recorded on it) themselves.
        """
        return self._stream

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
    def _setup_impl(self, P, c, A=None, b=None, G=None, h_u=None, h_l=None, x_u=None, x_l=None):
        """Bind the problem data and prepare the solver for ``solve()``.

        Backends call this from their public ``setup()`` inside the solver's
        stream scope, after validating their own argument conventions.

        Fixes the problem *structure* - array shapes, which constraint
        blocks are present, the sparsity pattern (sparse backend), and the
        finite/infinite pattern of the bounds - and allocates all GPU
        buffers, the KKT system, and the preconditioner. Call this **once**
        per solver instance, then call ``solve()``.

        Parameters
        ----------
        P : GPU array
            Quadratic cost, shape ``(n, n)``. Must be symmetric positive
            semidefinite. Required.
        c : GPU array
            Linear cost, shape ``(n,)``. Required.
        A, b : GPU array, optional
            Equality constraints ``A x = b``; shapes ``(p, n)`` and ``(p,)``.
            Omit for no equality constraints.
        G, h_l, h_u : GPU array, optional
            Two-sided inequalities ``h_l <= G x <= h_u``; ``G`` is
            ``(m, n)`` and the bounds are ``(m,)``. Use ``-inf``
            / ``+inf`` entries for one-sided rows.
        x_l, x_u : GPU array, optional
            Element-wise box bounds ``x_l <= x <= x_u``, shape ``(n,)``.
            Use ``+/-inf`` for unbounded entries.

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

        if not self.settings.verify_settings():
            raise ValueError(
                "Invalid solver settings; check the Settings field values."
            )

        self._data = self._init_data(P, c, A, b, G, h_u, h_l, x_u, x_l)

        self._preconditioner = self._init_preconditioner()
        if self.settings.preconditioner_iter > 0:
            self._preconditioner.scale_data(
                self._data,
                self.settings.preconditioner_scale_cost,
                self.settings.preconditioner_iter
            )

        data = self._data
        B = data.batch_size

        self._result = Result(B)
        self._result.init(self._data)
        self._result.info.rho = self.settings.rho_init
        self._result.info.delta = self.settings.delta_init

        self._step.init(self._data)
        self._res_nr.init(self._data)
        self._res.init(self._data)
        self._prox_vars.init(self._data)

        self._kkt_system.init(self._data, self.settings)
        self._info_host = InfoHost(B, dtype=self._data.dtype)
        # Problems in the batch that have terminated must not evolve while other problems continue iterating.
        self._unsolved_mask = wp.full(B, True, dtype=wp.bool, device=self._device)
        self._finetune_mask = wp.zeros(B, dtype=wp.bool, device=self._device)
        # Per-iteration improvement flags, passed from the rho/delta update to the prox update.
        self._dual_improved = wp.zeros(B, dtype=wp.bool, device=self._device)
        self._primal_improved = wp.zeros(B, dtype=wp.bool, device=self._device)

        self._work_z_1 = wp.empty((B, data.m), dtype=self._dtype, device=self._device)  # used to store intermediate results in _update_residuals_nr
        self._work_z_2 = wp.empty((B, data.m), dtype=self._dtype, device=self._device)  # used to store intermediate results in _update_residuals_nr
        self._work_z = wp.empty((B, data.num_ineq), dtype=self._dtype, device=self._device)  # scratch of one dual/slack row width
        self._work_primals = wp.empty((B, data.n), dtype=self._dtype, device=self._device)

        self._init_warp_kernels()

        self._work_x = wp.empty((B, data.n), dtype=self._dtype, device=self._device)

        self._tau_device = wp.full(1, self.settings.tau, dtype=self._dtype, device=self._device)  # device copy used by warp kernels
        self._tau_host = float(self.settings.tau)  # host cache -- only H2D when tau actually changes

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
            # in the fused pack kernel: the kernel can't read None, so
            # absent kwargs get substituted with the corresponding field of
            # this zero buffer.
            self._zero_grad_in = Variables()
            self._zero_grad_in.init(self._data)
            self._zero_grad_in.primals_all.zero_()
            self._zero_grad_in.duals_all.zero_()
            # Full-layout scatter buffers feeding the matrix and vector
            # gradient assemblies. ineq groups live in length-m; bound
            # groups live in length-n.
            self._lam_zu_full  = wp.empty((B, data.m), dtype=self._dtype, device=self._device)
            self._lam_zl_full  = wp.empty((B, data.m), dtype=self._dtype, device=self._device)
            self._lam_zbu_full = wp.empty((B, data.n), dtype=self._dtype, device=self._device)
            self._lam_zbl_full = wp.empty((B, data.n), dtype=self._dtype, device=self._device)
            self._zu_full      = wp.empty((B, data.m), dtype=self._dtype, device=self._device)
            self._zl_full      = wp.empty((B, data.m), dtype=self._dtype, device=self._device)


        self._grad_smoothing = bool(self.settings.enable_grad and self.settings.gradient_smoothing)
        if self._grad_smoothing:
            self._result_scaled.init(self._data)
            self._result_smoothed.init(self._data)
            self._smoothing_mu_scaled = wp.empty((B,), dtype=self._dtype, device=self._device)            # cost_scaling * mu
            self._smoothing_alpha_s = wp.empty((B,), dtype=self._dtype, device=self._device)              # primal step length
            self._smoothing_alpha_z = wp.empty((B,), dtype=self._dtype, device=self._device)              # dual step length
            self._smoothing_residual_norm = wp.zeros((1,), dtype=self._dtype, device=self._device)
            self._smoothing_residual_norm_host = wp.zeros(1, dtype=self._dtype, device="cpu", pinned=True)

        self._enable_iterative_refinement = self.settings.iterative_refinement_always_enabled

        # Unscaled-RHS inf-norm. When preconditioner_iter == 0 the stored
        # factors are identity, so this reduces to the inf-norm of the user-
        # space b / h_l/u / x_l/u -- same answer, single code path.
        self._constraints_rhs_inf_norm_unscaled = wp.zeros((B,), dtype=self._dtype, device=self._device)
        self._preconditioner.compute_constraints_rhs_inf_norm_unscaled(
            self._data, self._constraints_rhs_inf_norm_unscaled
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
               check_validity: bool = False
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
        with wp.ScopedStream(self._stream):
            self._update_impl(P, c, A, b, G, h_u, h_l, x_u, x_l, check_validity)

    def _update_impl(self, P, c, A, b, G, h_u, h_l, x_u, x_l, check_validity):
        """Body of :meth:`update`; the caller holds the solver's stream scope."""
        if not self._setup_done:
            raise RuntimeError("Solver not setup yet. Call setup() first.")

        # TODO: in the future should allow only update some problems of the whole batch, and only marks the updated ones as unsolved
        self._result.info.status_value[:] = Status.CUPIQP_UNSOLVED.value

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
                    self.settings.preconditioner_iter
                )
                preconditioner_did_fresh_ruiz = True

        self._preconditioner.compute_constraints_rhs_inf_norm_unscaled(
            self._data, self._constraints_rhs_inf_norm_unscaled
        )
        # Fresh Ruiz produces new factors that re-scale ALL of P/A/G in place,
        # even matrices the user didn't pass. The KKT solver caches things
        # like A^T A keyed off those scaled values, so flag everything as
        # changed in that case.
        self._kkt_system.update_data(
            self._data,
            (P is not None) or preconditioner_did_fresh_ruiz,
            (A is not None) or preconditioner_did_fresh_ruiz,
            (G is not None) or preconditioner_did_fresh_ruiz or ineq_bound_pattern_may_change
        )

    def solve(self) -> List[Status]:
        """Solve the QP set up by ``setup()`` and return the solve status.

        Runs the proximal interior-point iterations on the GPU. The full
        solution (primal ``x``, dual, and slack variables) and per-problem
        diagnostics are written to ``solver.result``; this method returns the
        status for convenience.

        Returns
        -------
        list of Status
            One ``Status`` per problem in the batch (a list of length 1 for a
            single problem). ``CUPIQP_SOLVED`` means the problem converged to
            tolerance. The same list is available as
            ``solver.result.info.status``.

        Notes
        -----
        Read the solution from ``solver.result`` after solving - e.g.
        ``solver.result.x`` (a ``(B, n)`` Warp array) and
        ``solver.result.info.status``. Set ``solver.settings.verbose = True``
        to print a per-iteration log. After ``setup()`` you may ``solve()``
        repeatedly, optionally calling ``update()`` in between to change the
        numerical data.
        """
        with wp.ScopedStream(self._stream):
            return self._solve_impl()

    def _solve_impl(self) -> List[Status]:
        """Body of :meth:`solve`; the caller holds the solver's stream scope."""
        if not self._setup_done:
            raise RuntimeError("Solver not setup yet. Call setup() first.")

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
            self._print_problem_size()

            print(f"inequality lower bounds n_h_l = {self._data.num_hl}")
            print(f"inequality upper bounds n_h_u = {self._data.num_hu}")
            print(f"variable lower bounds n_x_l = {self._data.num_xl}")
            print(f"variable upper bounds n_x_u = {self._data.num_xu}")
            print("")

        info = self._result.info
        info.status_value[:] = Status.CUPIQP_UNSOLVED.value
        self._unsolved_mask.fill_(True)
        info.iter[:] = 0
        info.iter_total = 0
        self._iter = 0  # global IPM iteration counter (host scalar)
        info.reg_limit = self.settings.reg_lower_limit
        # Refresh tau only if the user changed settings.tau between solves because it requires H2D memcpy
        if self._tau_host != self.settings.tau:
            self._tau_device.fill_(self.settings.tau)
            self._tau_host = float(self.settings.tau)
        info.factor_retires[:] = 0
        info.no_primal_update.zero_()
        info.no_dual_update.zero_()
        info.mu = 0.
        info.primal_step = 0.
        info.dual_step = 0.
        info.rho = self.settings.rho_init
        info.delta = self.settings.delta_init

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
                info.iter[still_unsolved] = iter
                if iter == 0:
                    self._update_residuals_nr()
                    wp.copy(info.prev_primal_res, info.primal_res)
                    wp.copy(info.prev_dual_res, info.dual_res)

                self._update_residuals_r()

                # fetch all info to host all at once, at the cost of one D2H memcpy
                info.to_host(self._info_host)  # CPU: numpy (num_fields, B) buffer
                info_host = self._info_host

                # ============================================================
                # Per-problem termination check -- ALL ON CPU (host-side numpy)
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
                info.status_value[solved] = Status.CUPIQP_SOLVED.value  # CPU write

                # primal infeasibility check
                primal_infeasible = still_unsolved & ~converged & (
                    (info_host.no_dual_update > min(5, settings.reg_finetune_dual_update_threshold)) &
                    (info_host.primal_prox_inf > settings.infeasibility_threshold) &
                    ((info_host.primal_res_reg < settings.eps_abs) | (info_host.primal_res_reg_rel < settings.eps_rel))
                )
                info.status_value[primal_infeasible] = Status.CUPIQP_PRIMAL_INFEASIBLE.value  # CPU write

                # dual infeasibility check
                dual_infeasible = still_unsolved & ~converged & ~primal_infeasible & (
                    (info_host.no_primal_update > min(5, settings.reg_finetune_primal_update_threshold)) &
                    (info_host.dual_prox_inf > settings.infeasibility_threshold) &
                    ((info_host.dual_res_reg < settings.eps_abs) | (info_host.dual_res_reg_rel < settings.eps_rel))
                )
                info.status_value[dual_infeasible] = Status.CUPIQP_DUAL_INFEASIBLE.value  # CPU write

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
                    self._upload_mask(self._unsolved_mask, still_unsolved)

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
                        device=self._device
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
                    self._upload_mask(self._finetune_mask, finetune_mask)
                    wp.launch(
                        kernel=self._apply_finetune_kernel,
                        dim=(self._data.batch_size,),
                        inputs=[self._finetune_mask, info.reg_limit,
                                info.no_primal_update, info.no_dual_update,
                                self._dtype(self.settings.reg_finetune_lower_limit)],
                        device=self._device
                    )

                self._update_and_factorize_kkt()
                if np.any(info.status_value == Status.CUPIQP_NUMERICAL_ISSUES.value):
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

        info.iter_total = int(self._iter)
        # Mark remaining unsolved as max iter reached
        info.status_value[info.status_value == Status.CUPIQP_UNSOLVED.value] = Status.CUPIQP_MAX_ITER_REACHED.value
        if self.settings.verbose:
            self._print_summary()
        # Capture the converged iterate in SCALED coordinates before it is
        # unscaled below. Gradient smoothing warm-starts its relaxed Newton loop
        # from this converged point, which must live in the same scaled frame as
        # the data / KKT factor. (The frozen rho/delta are read straight from
        # self._result.info at backward time, so they are not copied here.)
        if self._grad_smoothing:
            self._result_scaled.copy_from(self._result)
        if self.settings.preconditioner_iter > 0:
            self._preconditioner.unscale_solution(self._result, self._data)
        statuses = info.status

        return statuses

    @staticmethod
    def _upload_mask(device_mask: wp.array, host_mask: np.ndarray) -> None:
        """Copy a host boolean batch mask into its device counterpart."""
        wp.copy(device_mask, wp.array(host_mask, dtype=wp.bool, device="cpu", copy=False))

    @nvtx.annotate("Solver::_initial_guess")
    def _initial_guess(self):
        # eq(12) in Roland Schwan 2023 paper
        self._result.x.zero_()
        self._result.y.zero_()
        self._result.s_all.fill_(1.0)
        self._result.z_all.fill_(1.0)

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
            device=self._device
        )
        self._res.s_all.zero_()

        self._kkt_system.solve(self._data, self._preconditioner, self.settings, self._res, self._result)  # getting an initial point of _result

        if self._data.num_ineq > 0:
            ## ----------- keep z and s non-negative, then put them on the central path --------------
            # this is according to the IV.A part of Roland Schwan 2023 paper.
            wp.launch_tiled(
                kernel=self._init_guess_center_kernel,
                dim=[self._data.batch_size],
                inputs=[self._data.finite_mask_all, self._data.num_finite_bounds,
                        self._result.s_all, self._result.z_all, self._result.info.mu],
                block_dim=REDUCTION_BLOCK_DIM,
                device=self._device
            )
            self._calculate_mu()

        self._prox_vars.copy_from(self._result)

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
                flush=True
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
                flush=True
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
        info = self._result.info
        retries = 0
        while retries < self.settings.max_factor_retires:
            factor_succeeded = self._kkt_system.update_scalings_and_factor(
                self._data, self._preconditioner, self.settings, self._enable_iterative_refinement,
                info.rho, info.delta, self._result)
            if factor_succeeded:
                break
            else:
                if not self._enable_iterative_refinement:
                    self._enable_iterative_refinement = True
                retries += 1
                wp.launch(
                    kernel=self._factor_retry_kernel,
                    dim=(self._data.batch_size,),
                    inputs=[info.rho, info.delta, info.reg_limit, self._dtype(self.settings.eps_abs)],
                    device=self._device
                )

        if retries >= self.settings.max_factor_retires:
            # Mark all still-unsolved problems as numerical issues
            still_unsolved = (info.status_value == Status.CUPIQP_UNSOLVED.value)
            info.status_value[still_unsolved] = Status.CUPIQP_NUMERICAL_ISSUES.value

    @abstractmethod
    def _init_data(
        self,
        P: Any,
        c: wp.array,
        A: Optional[Any],
        b: Optional[wp.array],
        G: Optional[Any],
        h_u: Optional[wp.array],
        h_l: Optional[wp.array],
        x_u: Optional[wp.array],
        x_l: Optional[wp.array]
    ) -> Data:
        """Backend-specific data construction hook.

        Receives the inputs of the public ``setup()`` after its boundary
        conversion: vectors are Warp arrays of the solver dtype, and ``P``,
        ``A``, ``G`` are in the backend's matrix form (a Warp array for dense,
        a CSR triple for sparse, a ``(diag, offdiag)`` pair for multistage).
        """

    @abstractmethod
    def _init_preconditioner(self):
        """Backend-specific Ruiz preconditioner construction hook."""

    @abstractmethod
    def _print_problem_size(self):
        """Backend-specific verbose banner: backend name and problem sizes."""

    def _init_warp_kernels(self) -> None:
        """Create (and thus compile) every kernel of the IPM loop.

        The block-reduction kernels depend only on the dtype; the element-wise
        kernels are specialized to the block widths fixed at setup.
        """
        d = self._data
        dtype = d.dtype
        if d.num_ineq > 0:
            self._boundary_shift_kernel = create_boundary_shift_kernel(
                d.num_hl, d.num_hu, d.num_xl, d.num_xu, dtype=dtype)
            self._prepare_predictor_step_kernel = create_prepare_predictor_step_kernel(dtype=dtype)
            self._prepare_corrector_step_kernel = create_prepare_corrector_step_kernel(dtype=dtype)
            self._update_vars_after_corrector_step_kernel = create_update_vars_after_corrector_step_kernel(
                n=d.n, p=d.p, num_ineq=d.num_ineq, dtype=dtype)
            self._init_guess_center_kernel = create_init_guess_center_kernel(dtype)
            self._calculate_sigma_kernel = create_calculate_sigma_kernel(dtype)
            self._calculate_step_kernel = create_calculate_step_kernel(dtype)
            self._calculate_mu_kernel = create_calculate_mu_kernel(dtype)
            self._update_rho_delta_with_ineq_kernel = create_update_rho_delta_with_ineq_kernel(dtype)
            self._update_prox_vars_kernel = create_update_prox_vars_kernel(
                d.n, d.p + d.num_ineq, dtype=dtype)
        else:
            self._run_full_newton_step_kernel = create_run_full_newton_step_kernel(d.n, d.p, dtype=dtype)
            self._update_rho_delta_without_ineq_kernel = create_update_rho_delta_without_ineq_kernel(
                d.n, d.p, dtype=dtype)

        self._initial_guess_rhs_kernel = create_init_guess_rhs_kernel(
            d.n, d.p, int(d.num_hl), int(d.num_hu), int(d.num_xl), int(d.num_xu), dtype=dtype)
        self._update_residuals_r_kernel = create_update_residuals_r_kernel(dtype)
        self._prepare_zu_minus_zl_and_zbu_minus_zbl_kernel = create_prepare_zu_minus_zl_and_zbu_minus_zbl_kernel(
            d.m, d.n, has_h_l=d.has_h_l, has_h_u=d.has_h_u, has_x_l=d.has_x_l, has_x_u=d.has_x_u, dtype=dtype)
        self._update_residual_nr_kernel = create_update_residual_nr_kernel(dtype)
        self._factor_retry_kernel = create_factor_retry_kernel(dtype)
        self._apply_finetune_kernel = create_apply_finetune_kernel(dtype)

        if self.settings.enable_grad and self.settings.gradient_smoothing:
            self._update_smoothing_residual_nr_kernel = create_update_smoothing_residual_nr_kernel(dtype)
            self._smoothing_prepare_kernel = create_smoothing_prepare_kernel(dtype)
            self._smoothing_apply_step_kernel = create_smoothing_apply_step_kernel(dtype)
            if d.num_ineq == 0:
                self._calculate_step_kernel = create_calculate_step_kernel(dtype)

        # Adjoint/backward-pass kernels.
        if self.settings.enable_grad:
            n, p = d.n, d.p
            nhu, nhl = d.num_hu, d.num_hl
            nxu, nxl = d.num_xu, d.num_xl
            precond_on = self.settings.preconditioner_iter > 0
            self._backward_assemble_rhs_kernel = create_backward_assemble_rhs_kernel(
                n, p, nhu, nhl, nxu, nxl, precond_on, dtype=dtype)
            self._backward_unscale_lhs_kernel = create_backward_unscale_lhs_kernel(
                n, p, nhu, nhl, nxu, nxl, precond_on, dtype=dtype)
            self._backward_compute_vector_grad_kernel = create_backward_compute_vector_grad_kernel(
                n, p, nhu, nhl, nxu, nxl, dtype=dtype)
            self._backward_pack_full_layout_kernel = create_backward_pack_full_layout_kernel(
                d.m, n, nhl, nhu, nxl, nxu, dtype=dtype)
            self._backward_copy_kernel = create_backward_copy_kernel(
                n, p, nhu, nhl, nxu, nxl, dtype=dtype)

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
            device=self._device
        )

    @nvtx.annotate("Solver::_run_predictor_corrector")
    def _run_predictor_corrector(self):
        """Predictor-corrector steps + variable update + mu calculation."""
        # ------------------ predictor step ------------------
        # Short derivation:
        # Complementarity (elementwise): s_i * z_i = mu (usually written S * z = mu e).
        # Predictor (affine) aims for the affine step that drives complementarity to zero, so require (s + ds) o (z + dz) = 0.
        # Expand: s o z + S dz + Z ds + ds o dz = 0, where S = diag(s), Z = diag(z).
        # Drop the quadratic term ds o dz (first-order Newton linearization) to get the linear system S dz + Z ds = - s o z.
        # Thus the predictor RHS for the slack/dual complementarity equations is - s o z (elementwise product).

        # one fused kernel: res.s_all[b, i] = -s_all[b, i] * z_all[b, i].
        wp.launch(
            kernel=self._prepare_predictor_step_kernel,
            dim=(self._data.batch_size, self._data.num_ineq),
            inputs=[self._result.s_all, self._result.z_all, self._data.finite_mask_all, self._res.s_all],
            device=self._device
        )

        self._kkt_system.solve(self._data, self._preconditioner, self.settings, self._res, self._step)

        # step in the non-negative orthant
        self._calculate_step()

        # ------------------ compute centering parameter sigma ------------------
        self._calculate_sigma()

        # ------------------ corrector step ------------------
        # res.s += -step.s * step.z + sigma * mu   (on finite-bound entries)
        wp.launch(
            kernel=self._prepare_corrector_step_kernel,
            dim=(self._data.batch_size, self._data.num_ineq),
            inputs=[
                self._step.s_all, self._step.z_all,
                self._result.info.sigma, self._result.info.mu,
                self._data.finite_mask_all,
                self._res.s_all,
            ],
            device=self._device
        )

        self._kkt_system.solve(self._data, self._preconditioner, self.settings, self._res, self._step)

        # step in the non-negative orthant
        self._calculate_step()
        self._update_vars_after_corrector_step()
        self._calculate_mu()

    @cuda_graph_capture(enable=lambda self: self.settings.enable_cuda_graph)
    def _update_vars_after_corrector_step(self):
        # result.primals_all += primal_step * step.primals_all
        # result.duals_all += dual_step * step.duals_all
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
            device=self._device
        )

    @nvtx.annotate("Solver::_calculate_step")
    @cuda_graph_capture(enable=lambda self: self.settings.enable_cuda_graph)
    def _calculate_step(self) -> None:
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
            block_dim=REDUCTION_BLOCK_DIM,
            device=self._device
        )

    @nvtx.annotate("Solver::_calculate_mu")
    @cuda_graph_capture(enable=lambda self: self.settings.enable_cuda_graph)
    def _calculate_mu(self) -> None:
        """Calculate mu (the duality measure)."""
        wp.launch_tiled(
            kernel=self._calculate_mu_kernel,
            dim=[self._data.batch_size],
            inputs=[
                self._result.s_all, self._result.z_all,
                self._data.finite_mask_all, self._data.num_finite_bounds,
                self._result.info.mu,
            ],
            block_dim=REDUCTION_BLOCK_DIM,
            device=self._device
        )

    @nvtx.annotate("Solver::_calculate_sigma")
    @cuda_graph_capture(enable=lambda self: self.settings.enable_cuda_graph)
    def _calculate_sigma(self) -> None:
        """Calculate sigma (the centering parameter)."""
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
            block_dim=REDUCTION_BLOCK_DIM,
            device=self._device
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
        space - magnitudes are restored via ``delta_inv``, ``delta_b_inv``,
        ``cost_scaling_inv`` from the preconditioner.

        Residual formulas (scaled space):

            res_nr.x    = -(P*x + c + A^T*y + G^T*(z_u - z_l)
                            + x_b_scaling*(z_bu - z_bl))
            res_nr.y    = -(A*x - b)
            res_nr.z_l  =   G*x - s_l - h_l          (finite lower rows)
            res_nr.z_u  = -G*x - s_u + h_u           (finite upper rows)
            res_nr.z_bl =   x_b_scaling*x - s_bl - x_l
            res_nr.z_bu = -(x_b_scaling*x + s_bu - x_u)

        Convergence norms (unscaled, infinity norm per batch):

            primal_res     = max over the 5 dual segments of
                                 ||u_p_seg .* res_nr_seg||_inf
                             where u_p_seg is the per-segment primal
                             unscale factor (delta_inv / delta_b_inv slices).

            dual_res       = cost_scaling_inv * ||delta_inv[:, :n] .* res_nr.x||_inf

        Relative-norm denominators (also unscaled, max over magnitudes
        that go into the corresponding residual):

            primal_rel     = max( ||u_p_y .* A*x||,
                                  ||u_p_zl .* G*x||, ||u_p_zu .* G*x||,
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

        self._kkt_system.eval_P_x(data, -1., result.x, res_nr.x)

        if data.p > 0:
            self._kkt_system.eval_A_xn(data, 1., result.x, self._res.y)
            self._kkt_system.eval_AT_xt(data, 1., result.y, self._res.x)
        else:
            self._res.y.zero_()
            self._res.x.zero_()

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
            device=self._device
        )

        G_x = self._work_z_2
        GT_zu_minus_zl = self._step.x
        if data.m > 0:
            self._kkt_system.eval_G_xn(data, 1., result.x, G_x)
            self._kkt_system.eval_GT_xt(data, 1., self._work_z_1, GT_zu_minus_zl)
        else:
            G_x.zero_()
            GT_zu_minus_zl.zero_()

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
            block_dim=REDUCTION_BLOCK_DIM,
            device=self._device
        )

    @nvtx.annotate("Solver::_update_residuals_r")
    @cuda_graph_capture(enable=lambda self: self.settings.enable_cuda_graph)
    def _update_residuals_r(self):
        """
        Compute the regularized primal and dual residuals. The computation is based on the non-regularized residuals computed in _update_residuals_nr.
        It adds the regularization terms to the non-regularized residuals to obtain the regularized residuals.
        """
        # update the rhs of the KKT system
        # res.x[:] = res_nr.x - rho * (result.x - prox_vars.x)
        # res.duals_all[:] = res_nr.duals_all + delta * (result.duals_all - prox_vars.duals_all)
        pc = self._preconditioner
        info = self._result.info
        wp.launch_tiled(
            kernel=self._update_residuals_r_kernel,
            dim=[self._data.batch_size],
            inputs=[
                info.rho, info.delta,
                self._res_nr.x, self._res_nr.duals_all,
                self._result.x, self._result.duals_all,
                self._prox_vars.x, self._prox_vars.duals_all,
                self._res.x, self._res.duals_all,
                pc.dual_res_unscale_factor, pc.primal_res_unscale_factor,
                info.primal_res, info.primal_res_rel,
                info.dual_res, info.dual_res_rel,
                info.primal_res_reg, info.primal_res_reg_rel,
                info.dual_res_reg, info.dual_res_reg_rel,
                info.primal_prox_inf, info.dual_prox_inf,
            ],
            block_dim=REDUCTION_BLOCK_DIM,
            device=self._device
        )

    @nvtx.annotate("Solver::_update_rho_delta_with_ineq")
    def _update_rho_delta_with_ineq(self) -> None:
        info = self._result.info
        settings = self.settings
        n = self._data.n
        num_duals = self._data.p + self._data.num_ineq
        # Two launches: the improvement flags read rho/delta, so they are decided
        # once per problem, before rho/delta are overwritten, and then handed to
        # the element-wise prox update.
        wp.launch(
            kernel=self._update_rho_delta_with_ineq_kernel,
            dim=(self._data.batch_size,),
            inputs=[
                self._unsolved_mask,
                info.dual_res, info.prev_dual_res, info.dual_res_rel, info.dual_prox_inf,
                info.primal_res, info.prev_primal_res, info.primal_res_rel, info.primal_prox_inf,
                info.reg_limit,
                info.rho, info.delta,
                info.no_primal_update, info.no_dual_update,
                self._dual_improved, self._primal_improved,
                self._dtype(settings.eps_abs),
                self._dtype(settings.eps_rel),
                self._dtype(settings.reg_finetune_lower_limit),
                self._dtype(settings.infeasibility_threshold),
                wp.int32(self._iter),
            ],
            device=self._device
        )
        wp.launch(
            kernel=self._update_prox_vars_kernel,
            dim=(self._data.batch_size, n + num_duals),
            inputs=[
                self._unsolved_mask,
                self._dual_improved, self._primal_improved,
                self._result.x, self._prox_vars.x,
                self._result.duals_all, self._prox_vars.duals_all,
            ],
            device=self._device
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
                self._dtype(settings.eps_abs),
                self._dtype(settings.eps_rel),
                self._dtype(settings.infeasibility_threshold),
                wp.int32(self._iter),  # self._iter is int64, need to convert to int32
            ],
            device=self._device
        )

    # ------------------------------------------------------------------
    # Gradient smoothing
    # ------------------------------------------------------------------
    def _run_smoothing_steps(self) -> None:
        """Run the relaxed Newton loop for gradient smoothing.

        From the scaled converged solution (``self._result_scaled``), drive
        ``s * z -> mu`` and leave (a) the cached KKT factor at the relaxed point
        and (b) the user-space relaxed iterate in ``self._result_smoothed`` --
        which ``backward()`` then passes as the linearization point to the
        adjoint solve and the matrix/vector gradient assembly.
        """
        data = self._data
        pc = self._preconditioner
        settings = self.settings
        rv = self._result_smoothed

        # Restart from the immutable scaled converged solution.
        rv.copy_from(self._result_scaled)

        # mu in scaled space: s_scaled * z_scaled = cost_scaling * mu_user
        # (the Ruiz constraint scaling cancels in the s*z product; cost scaling
        # does not). cost_scaling is 1 unless preconditioner_scale_cost is set.
        # Floor s, z at sqrt(mu_scaled) on finite-bound entries so the first
        # factorization of H = Q + G^T diag(z/s) G stays well conditioned.
        wp.launch(
            kernel=self._smoothing_prepare_kernel,
            dim=(data.batch_size, max(data.num_ineq, 1)),
            inputs=[pc.cost_scaling, self._dtype(settings.gradient_smoothing_mu),
                    data.finite_mask_all, rv.s_all, rv.z_all, self._smoothing_mu_scaled],
            device=self._device
        )

        # Frozen regularization = the converged solve's final rho/delta
        rho, delta = self._result.info.rho, self._result.info.delta

        converged = False
        for _ in range(settings.gradient_smoothing_max_iter):
            self._kkt_system.update_scalings_and_factor(
                data, pc, settings, False, rho, delta, rv)
            norm = self._calculate_smoothing_step_rhs()
            if norm < settings.gradient_smoothing_tol:
                converged = True
                break
            self._kkt_system.solve(data, pc, settings, self._res, self._step)
            self._apply_smoothing_step()
        if not converged:
            # Check the iterate left by the last step before giving up.
            self._kkt_system.update_scalings_and_factor(
                data, pc, settings, False, rho, delta, rv)
            norm = self._calculate_smoothing_step_rhs()
            if not norm < settings.gradient_smoothing_tol:
                raise RuntimeError(
                    "Gradient smoothing did not converge: the relaxed KKT residual "
                    f"is {norm:.3e} after {settings.gradient_smoothing_max_iter} "
                    f"iterations (gradient_smoothing_tol = "
                    f"{settings.gradient_smoothing_tol:.3e}). Increase "
                    "gradient_smoothing_max_iter or relax gradient_smoothing_tol."
                )

        if settings.preconditioner_iter > 0:
            pc.unscale_solution(rv, data)

    def _calculate_smoothing_step_rhs(self) -> float:
        """Assemble the relaxed Newton RHS (negative non-regularized KKT residual
        with complementarity target mu) into ``self._res`` and return its
        scaled-space inf-norm. Sign convention matches the forward KKT solve
        RHS: x/y/z rows are the negative stationarity/feasibility residuals and
        the s-row is ``mu_scaled - s*z``.
        """
        data = self._data
        pc = self._preconditioner
        result_smooth = self._result_smoothed
        res = self._res
        wp_stream = wp.get_stream("cuda")

        self._kkt_system.eval_P_x(data, -1.0, result_smooth.x, res.x)        # minus_Px
        if data.p > 0:
            self._kkt_system.eval_A_xn(data, 1.0, result_smooth.x, res.y)    # A_x
            self._kkt_system.eval_AT_xt(data, 1.0, result_smooth.y, self._work_x)  # AT_y
        else:
            self._work_x.zero_()

        # z_u - z_l -> work_z_1 ; x_b_scaling*(z_bu - z_bl) -> work_primals
        if data.num_ineq > 0:
            wp.launch(
                kernel=self._prepare_zu_minus_zl_and_zbu_minus_zbl_kernel,
                dim=(data.batch_size, data.m + data.n),
                inputs=[result_smooth.z_u, result_smooth.z_l, result_smooth.z_bl, result_smooth.z_bu, pc.x_b_scaling,
                        self._work_z_1, self._work_primals],
                device=self._device
            )
        else:
            self._work_primals.zero_()
        # G_x -> work_z_2 ; G^T(z_u - z_l) -> step.x
        if data.m > 0:
            self._kkt_system.eval_G_xn(data, 1.0, result_smooth.x, self._work_z_2)
            self._kkt_system.eval_GT_xt(data, 1.0, self._work_z_1, self._step.x)
        else:
            self._step.x.zero_()

        # --- fused elementwise assembly + relaxed s-row + inf-norm ---
        # norm_out is an atomic-max accumulator; zero it before the launch.
        self._smoothing_residual_norm.zero_()
        wp.launch_tiled(
            kernel=self._update_smoothing_residual_nr_kernel,
            dim=[data.batch_size],
            inputs=[
                res.x, res.y, self._work_x, self._work_z_2, self._step.x, self._work_primals,
                data.c, data.b, data.h_l, data.h_u, data.x_l, data.x_u,
                data.finite_mask_hl, data.finite_mask_hu, data.finite_mask_xl, data.finite_mask_xu,
                result_smooth.x, result_smooth.z_l, result_smooth.z_u, result_smooth.z_bl, result_smooth.z_bu, result_smooth.s_l, result_smooth.s_u, result_smooth.s_bl, result_smooth.s_bu,
                pc.x_b_scaling, self._smoothing_mu_scaled,
                res.x, res.y, res.z_l, res.z_u, res.z_bl, res.z_bu,
                res.s_l, res.s_u, res.s_bl, res.s_bu,
                self._smoothing_residual_norm,
            ],
            block_dim=REDUCTION_BLOCK_DIM,
            device=self._device
        )
        wp.copy(self._smoothing_residual_norm_host, self._smoothing_residual_norm)
        wp.synchronize_stream(wp_stream)
        return float(self._smoothing_residual_norm_host.numpy()[0])

    def _apply_smoothing_step(self) -> None:
        """Take a damped Newton step on ``self._result_smoothed``. The per-side
        fraction-to-boundary step lengths ``tau * min(1, min_i -v_i / dv_i)`` are
        computed with the forward solve's step-length kernel, then a single
        *common* step ``alpha = min(alpha_s, alpha_z)`` is applied to all of
        (x, y, s, z). Stepping s and z together keeps them positive and drives
        ``s * z`` to ``mu`` in a few iterations -- separate primal/dual steps let
        the product overshoot near the boundary. The step rule only affects the
        path; the relaxed fixed point ``s * z = mu`` (hence the gradient) is the
        same.
        """
        data = self._data
        rv = self._result_smoothed
        step = self._step

        if data.num_ineq == 0:
            self._smoothing_alpha_s.fill_(1.0)
            self._smoothing_alpha_z.fill_(1.0)
        else:
            wp.launch_tiled(
                kernel=self._calculate_step_kernel,
                dim=[data.batch_size],
                inputs=[
                    rv.s_all, rv.z_all, data.finite_mask_all,
                    step.s_all, step.z_all, self._tau_device,
                    self._smoothing_alpha_s, self._smoothing_alpha_z,
                ],
                block_dim=REDUCTION_BLOCK_DIM,
                device=self._device
            )

        wp.launch(
            kernel=self._smoothing_apply_step_kernel,
            dim=(data.batch_size, data.n + data.p + 2 * data.num_ineq),
            inputs=[
                self._smoothing_alpha_s, self._smoothing_alpha_z, self._dtype(self.settings.tau),
                data.finite_mask_all,
                step.x, step.y, step.s_all, step.z_all,
                rv.x, rv.y, rv.s_all, rv.z_all,
            ],
            device=self._device
        )

    @nvtx.annotate("Solver::_compute_adjoint")
    def _compute_adjoint(self, grad: Variables, sol: Variables) -> None:
        r"""Solve the adjoint KKT system :math:`K^\top \lambda = -\partial L / \partial v`.

        Backend-agnostic: the math operates only on the variable buffers and
        the cached KKT factor (which every backend's ``_kkt_system`` exposes
        with a uniform ``solve(..., transpose=True)`` API). Used by every
        subclass's ``backward()`` to obtain the lambdas; per-backend matrix
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
        per-backend ``_compute_data_gradients`` only contributes the
        matrix-gradient push-back on top of that.

        Raises
        ------
        RuntimeError
            If :meth:`solve` has not been called yet (no cached KKT factor).
        """
        if not getattr(self, "_setup_done", False) or self._result is None:
            raise RuntimeError(
                f"{type(self).__name__}.backward() requires a prior solve(); "
                f"call setup() and solve() before backward()."
            )

        data = self._data
        settings = self.settings
        kkt_system = self._kkt_system
        precond = self._preconditioner
        B = data.batch_size
        n, p = data.n, data.p

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
            device=self._device
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
            device=self._device
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
            device=self._device
        )

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
        gradients in user space. Cotangents are GPU arrays of shape
        ``(B, k)`` in the solver dtype; the returned gradients are the
        solver's Warp arrays and are overwritten by the next ``backward()``.
        Cotangents and returned gradients are interpreted in user (un-scaled)
        space throughout - the adjoint solve and scatter chain handles all
        preconditioner bookkeeping internally.

        Raises
        ------
        RuntimeError
            If :meth:`solve` has not been called yet (no cached KKT factor).
        """
        with wp.ScopedStream(self._stream):
            return self._backward_impl(grad_x, grad_y, grad_z_u, grad_z_l, grad_z_bu, grad_z_bl,
                                       grad_s_u, grad_s_l, grad_s_bu, grad_s_bl)

    @nvtx.annotate("Solver::backward")
    def _backward_impl(self, grad_x, grad_y, grad_z_u, grad_z_l, grad_z_bu, grad_z_bl,
                       grad_s_u, grad_s_l, grad_s_bu, grad_s_bl):
        """Body of :meth:`backward`; the caller holds the solver's stream scope."""
        if not self.settings.enable_grad:
            raise RuntimeError("Set enable_grad to True to enable gradient computation.")

        if not getattr(self, "_setup_done", False) or self._result is None:
            raise RuntimeError(
                f"{type(self).__name__}.backward() requires a prior solve(); "
                f"call setup() and solve() before backward()."
            )

        if not (self._result.info.status_value == Status.CUPIQP_SOLVED.value).all():
            raise RuntimeError(
                "Gradient computation requires every problem in the batch to be "
                "solved (status CUPIQP_SOLVED) since the last setup()/update(); "
                "call solve() and check the status before backward()."
            )

        # The smoothing mode is fixed at setup() (it gates buffer allocation).
        if self.settings.gradient_smoothing != self._grad_smoothing:
            raise RuntimeError(
                "settings.gradient_smoothing was changed after setup(); the "
                "smoothing mode is fixed at setup. Create a new solver to change it."
            )

        if self._grad_smoothing:
            # Settings are mutable between backward() calls; an invalid
            # smoothing value must raise here, not produce a NaN gradient.
            if not self.settings.verify_settings():
                raise ValueError(
                    "Invalid solver settings; check the Settings field values."
                )
            self._run_smoothing_steps()
            result_for_calc_grad = self._result_smoothed
        else:
            result_for_calc_grad = self._result

        data = self._data
        B = data.batch_size

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
                device=self._device
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
                result_for_calc_grad.z_u, result_for_calc_grad.z_l,
                self._lam_zu_full, self._lam_zl_full,
                self._lam_zbu_full, self._lam_zbl_full,
                self._zu_full, self._zl_full,
            ],
            device=self._device
        )

        # ---- Step 4: backend-specific matrix and vector gradient
        # assembly + Data subclass construction.
        return self._compute_data_gradients(self._backward_adjoint_vector, result_for_calc_grad)

    @abstractmethod
    def _compute_data_gradients(self, sol_adj: Variables, linearization_point: Variables):
        """Build and return the backend's ``Data`` subclass populated
        with user-space gradients ``(P, c, A, b, G, h_u, h_l, x_u,
        x_l)``.

        Implementations read user-space adjoint lambdas from
        ``sol_adj`` (active-only sizes), the user-space primal/dual
        ``linearization_point`` (``self._result`` normally, or the
        relaxed iterate ``self._result_smoothed`` under gradient
        smoothing), and the pre-scattered full-layout buffers
        ``self._lam_z*_full`` / ``self._z*_full`` set up by
        :meth:`backward`'s scatter step.
        """
