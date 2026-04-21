import math

import cupy as cp
import warp as wp
import nvtx

from .results import Variables
from .data import Data
from .settings import Settings
from .utils import cuda_graph_capture

class KKTSystem:
    """
    The KKT system handles the full KKT condition with non-symmetric matrix.

    Solves the block linear system:

        [ P+rho*I   A^T      G^T      -G^T     I_n      -I_n                                         ] [ dx    ]   [ r_x    ]
        [ A        -d*I_p                                                                            ] [ dy    ]   [ r_y    ]
        [ G                 -d*I_m                                I_m                                ] [ dz_hu ]   [ r_z_hu ]
        [ -G                         -d*I_m                               I_m                        ] [ dz_hl ]   [ r_z_hl ]
        [ I_n                                  -d*I_n                             I_n                ] [ dz_xu ]   [ r_z_xu ]
        [ -I_n                                          -d*I_n                            I_n        ] [ dz_xl ]   [ r_z_xl ]
        [                   S_hu                                  Z_hu                               ] [ ds_hu ] = [ r_s_hu ]
        [                            S_hl                                  Z_hl                      ] [ ds_hl ]   [ r_s_hl ]
        [                                      S_xu                                  Z_xu            ] [ ds_xu ]   [ r_s_xu ]
        [                                               S_xl                                  Z_xl   ] [ ds_xl ]   [ r_s_xl ]

    where:
        - P, A, G        : problem data (cost, equality, inequality matrices)
        - rho            : proximal regularization parameter
        - d (delta)      : regularization on dual variables
        - S_hu, S_hl     : diagonal slack matrices for inequality upper/lower bounds
        - S_xu, S_xl     : diagonal slack matrices for variable upper/lower bounds
        - Z_hu, Z_hl     : diagonal dual variable matrices for inequality upper/lower bounds
        - Z_xu, Z_xl     : diagonal dual variable matrices for variable upper/lower bounds
        - n, p, m        : number of primal variables, equality constraints, inequality constraints
    """
    def __init__(self):
        return

    def init(self, data: Data, settings: Settings):
        self._settings = settings
        self._backend = settings.kkt_solver
        self._use_iterative_refinement = False
        self._batch_size = data.batch_size
        B = self._batch_size
        n, p, m = data.n, data.p, data.m

        # Per-problem regularization
        self._rho = cp.empty(B, dtype=cp.float64)
        self._delta = cp.empty(B, dtype=cp.float64)
        self._x_reg = cp.empty((B, n), dtype=cp.float64)
        self._z_reg = cp.empty((B, m), dtype=cp.float64)

        # used to store the rhs of the condensed KKT system (per problem)
        # K_condensed * [dx; dy; dz] = [rhs_x_bar; rhs_y_bar; rhs_z_bar], 
        # where K_condensed is the condensed KKT matrix after eliminating duals of inequalities and box constraints and all slacks.
        # Since eliminating slacks and duals does not change rhs_y, we only need to store rhs_x_bar and rhs_z_bar.

        self._rhs_x_bar = cp.empty((B, n), dtype=cp.float64)
        self._rhs_z_bar = cp.empty((B, m), dtype=cp.float64) if m > 0 else cp.empty((B, 0), dtype=cp.float64)

        # Work buffers
        self._work_x = cp.empty((B, n), dtype=cp.float64)
        self._work_z = cp.empty((B, m), dtype=cp.float64) if m > 0 else cp.empty((B, 0), dtype=cp.float64)

        # KKT solver backend
        if settings.kkt_solver == "dense_cholesky":
            from .dense.dense_kkt_solver import DenseKKTSolver
            self._kkt_solver = DenseKKTSolver(data)
        elif settings.kkt_solver == "sparse_ldlt":
            from .sparse.sparse_kkt_solver import SparseKKTSolver
            self._kkt_solver = SparseKKTSolver(data, use_deterministic_mode=settings.use_deterministic_mode_for_cudss)
        elif settings.kkt_solver == "multistage_block_cholesky":
            from .multistage.multistage_kkt_solver import MultistageKKTSolver
            self._kkt_solver = MultistageKKTSolver(data)
        else:
            raise ValueError(f"Unsupported kkt_solver: {settings.kkt_solver}")

        # Contiguous internal buffers with layout [hl | hu | xl | xu] matching s_all/z_all.
        num_ineq = data.num_hl + data.num_hu + data.num_xl + data.num_xu
        hl, hu, xl, xu = data.num_hl, data.num_hu, data.num_xl, data.num_xu
        # Store slack values at current iteration
        self._m_s_all = cp.zeros((B, num_ineq), dtype=cp.float64)
        self._m_s_l = self._m_s_all[:, :hl]
        self._m_s_u = self._m_s_all[:, hl:hl+hu]
        self._m_s_bl = self._m_s_all[:, hl+hu:hl+hu+xl]
        self._m_s_bu = self._m_s_all[:, hl+hu+xl:hl+hu+xl+xu]

        # Store 1/z values at current iteration
        self._m_z_inv_all = cp.zeros((B, num_ineq), dtype=cp.float64)
        self._m_z_l_inv = self._m_z_inv_all[:, :hl]
        self._m_z_u_inv = self._m_z_inv_all[:, hl:hl+hu]
        self._m_z_bl_inv = self._m_z_inv_all[:, hl+hu:hl+hu+xl]
        self._m_z_bu_inv = self._m_z_inv_all[:, hl+hu+xl:hl+hu+xl+xu]

        # Store 1/(s/z + delta) used in factor and solve
        self._w_delta_inv_all = cp.zeros((B, num_ineq), dtype=cp.float64)
        self._w_l_delta_inv = self._w_delta_inv_all[:, :hl]
        self._w_u_delta_inv = self._w_delta_inv_all[:, hl:hl+hu]
        self._w_bl_delta_inv = self._w_delta_inv_all[:, hl+hu:hl+hu+xl]
        self._w_bu_delta_inv = self._w_delta_inv_all[:, hl+hu+xl:hl+hu+xl+xu]

        # Updated rhs after eliminating slacks
        self._updated_rhs_z_all = cp.zeros((B, num_ineq), dtype=cp.float64)
        self._updated_rhs_z_l = self._updated_rhs_z_all[:, :hl]
        self._updated_rhs_z_u = self._updated_rhs_z_all[:, hl:hl+hu]
        self._updated_rhs_z_bl = self._updated_rhs_z_all[:, hl+hu:hl+hu+xl]
        self._updated_rhs_z_bu = self._updated_rhs_z_all[:, hl+hu+xl:hl+hu+xl+xu]

        # Pre-allocated buffers for condensed KKT iterative refinement
        self._iter_refine_error_xyz = cp.zeros((B, n + p + m), dtype=cp.float64)
        self._iter_refine_delta_xyz = cp.zeros((B, n + p + m), dtype=cp.float64)

        # Create Warp kernels
        self._update_regulerization_step_1_kernel = create_update_regularizations_step_1_kernel()
        self._update_regulerization_step_2_kernel = create_update_regularizations_step_2_kernel(n, m)
        self._eliminate_slacks_kernel = create_eliminate_slacks_kernel()
        self._eliminate_slacks_transposed_kernel = create_eliminate_slacks_transposed_kernel()
        self._eliminate_duals_kernel = create_eliminate_duals_kernel(n, m)
        self._recover_duals_kernel = create_recover_duals_kernel(data.num_hu, data.num_hl, data.num_xu, data.num_xl)
        self._recover_slacks_kernel = create_recover_slacks_kernel()
        self._recover_slacks_transposed_kernel = create_recover_slacks_transposed_kernel()

        # Precompute inverse index maps for gather-pattern kernels.
        # inv_idx_xu[j] = i such that idx_xu[i] == j, or -1 if variable j has no upper bound.
        _build_inv_idx_kernel = create_build_inverse_index_kernel()
        self._inv_idx_xu = wp.full(n, value=-1, dtype=wp.int32, device="cuda")
        self._inv_idx_xl = wp.full(n, value=-1, dtype=wp.int32, device="cuda")
        self._inv_idx_hu = wp.full(m, value=-1, dtype=wp.int32, device="cuda") if m > 0 else wp.zeros(0, dtype=wp.int32, device="cuda")
        self._inv_idx_hl = wp.full(m, value=-1, dtype=wp.int32, device="cuda") if m > 0 else wp.zeros(0, dtype=wp.int32, device="cuda")
        if data.num_xu > 0:
            wp.launch(_build_inv_idx_kernel, dim=data.num_xu,
                      inputs=[data.idx_xu, self._inv_idx_xu], device="cuda")
        if data.num_xl > 0:
            wp.launch(_build_inv_idx_kernel, dim=data.num_xl,
                      inputs=[data.idx_xl, self._inv_idx_xl], device="cuda")
        if data.num_hu > 0:
            wp.launch(_build_inv_idx_kernel, dim=data.num_hu,
                      inputs=[data.idx_hu, self._inv_idx_hu], device="cuda")
        if data.num_hl > 0:
            wp.launch(_build_inv_idx_kernel, dim=data.num_hl,
                      inputs=[data.idx_hl, self._inv_idx_hl], device="cuda")

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @nvtx.annotate("KKTSystem::update_data")
    def update_data(self, data: Data, update_P: bool = False, update_A: bool = False, update_G: bool = False):
        self._kkt_solver.update_data(data, update_P, update_A, update_G)

    @nvtx.annotate("KKTSystem::update_scalings_and_factor")
    def update_scalings_and_factor(self, data: Data, settings: Settings, iterative_refinement: bool, rho: cp.ndarray, delta: cp.ndarray, vars: Variables) -> bool:
        """Update regularization terms and factor the KKT matrix.

        TODO: When iterative_refinement (IR) is True, adds static regularization to improve factorization stability. The solve() method will then run IR.

        The variable vars is the current primal/dual variable values at this iteration, i.e., values of x, y, z_u, z_l, s_u, s_l, z_bu, z_bl, s_bu, s_bl at the current iteration.
        """
        self._update_reg_and_kkt(data, delta, rho, vars)
        self._use_iterative_refinement = iterative_refinement
        factor_success = self._kkt_solver.factor() # ! this is implicitly assuming idx_hu and idx_hl cover all indices of inequalities 0:m
        return factor_success

    @nvtx.annotate("KKTSystem::_update_reg_and_kkt")
    @cuda_graph_capture(key=lambda self, data, delta, rho, vars: (vars.buffer_ptr, delta.data.ptr, rho.data.ptr), enable=lambda self: self._settings.enable_cuda_graph)
    def _update_reg_and_kkt(self, data: Data, delta: cp.ndarray, rho: cp.ndarray, vars: Variables):
        """Update the regularization terms x_reg and z_reg for the condensed KKT system after eliminating slacks and duals of inequalities and box constraints. 
        Also update the condensed KKT matrix with the new regularization terms."""
        self._rho[:] = rho
        self._delta[:] = delta

        USE_WARP_IMPLEMENTATION = True
        B = self._batch_size
        if USE_WARP_IMPLEMENTATION:
            wp_stream = wp.Stream(cuda_stream=cp.cuda.get_current_stream().ptr)
            wp.launch(
                kernel=self._update_regulerization_step_1_kernel,
                dim=(B, data.num_ineq),
                inputs=[vars.s_all, vars.z_all,
                        self._m_s_all, self._m_z_inv_all, self._w_delta_inv_all,
                        delta],
                device="cuda",
                stream=wp_stream,
            )
            wp.launch(
                kernel=self._update_regulerization_step_2_kernel,
                dim=(B, data.n + data.m),
                inputs=[
                    self._inv_idx_xu,
                    self._inv_idx_xl,
                    self._inv_idx_hu,
                    self._inv_idx_hl,
                    self._w_bu_delta_inv,
                    self._w_bl_delta_inv,
                    data.x_b_scaling,
                    rho,
                    self._x_reg,
                    self._w_u_delta_inv,
                    self._w_l_delta_inv,
                    self._z_reg,
                ],
                device="cuda",
                stream=wp_stream,
            )
        else:
            # cache s, 1/z and w = 1/(s/z + delta)
            self._m_s_all[:] = vars.s_all
            cp.reciprocal(vars.z_all, out=self._m_z_inv_all)
            cp.multiply(self._m_s_all, self._m_z_inv_all, out=self._w_delta_inv_all)
            cp.add(self._w_delta_inv_all, delta[:, None], out=self._w_delta_inv_all)
            cp.reciprocal(self._w_delta_inv_all, out=self._w_delta_inv_all)

            # compute x_reg (B, n) and z_reg (B, m)
            self._x_reg[:] = rho[:, None]
            xbs = data.x_b_scaling  # (B, n)
            if data.num_xu > 0:
                xbs_xu = xbs[:, data.idx_xu]  # (B, num_xu)
                self._x_reg[:, data.idx_xu] += xbs_xu * xbs_xu * self._w_bu_delta_inv
            if data.num_xl > 0:
                xbs_xl = xbs[:, data.idx_xl]  # (B, num_xl)
                self._x_reg[:, data.idx_xl] += xbs_xl * xbs_xl * self._w_bl_delta_inv

            if data.m > 0:
                self._z_reg[:] = 0.0
                if data.num_hu > 0:
                    self._z_reg[:, data.idx_hu] += self._w_u_delta_inv
                if data.num_hl > 0:
                    self._z_reg[:, data.idx_hl] += self._w_l_delta_inv
                cp.reciprocal(self._z_reg, out=self._z_reg)

        # Update KKT matrix
        self._kkt_solver.update_kkt(data, delta, self._x_reg, self._z_reg)
    
    @nvtx.annotate("KKTSystem::solve")
    def solve(self, data: Data, settings: Settings, rhs: Variables, lhs: Variables,
              transpose: bool = False) -> None:
        """Solve either ``K v = rhs`` (default) or ``K^T λ = rhs`` (``transpose=True``).

        The condensed K_c factor built during the last ``update_scalings_and_factor``
        is reused in both directions: K_c is symmetric, so its cuDSS factor inverts
        K_c and K_c^T identically. Only the slack-side Schur elimination and back-
        substitution differ between forward and transposed — handled by swapping
        ``_eliminate_slacks_kernel`` / ``_recover_slacks_kernel`` for their
        ``_transposed`` counterparts.

        When ``transpose=True`` the system being solved (for implicit-diff / VJP) is::

            [ P+rho*I   A^T   G^T  -G^T   I_n  -I_n                              ]
            [ A        -d*I                                                      ]
            [ G                -d*I              S_hu                            ]
            [ -G                    -d*I              S_hl                       ]
            [ I_n                        -d*I               S_xu                 ]
            [ -I_n                             -d*I              S_xl            ]
            [                    I_m                        Z_hu                 ]
            [                          I_m                        Z_hl           ]
            [                               I_n                         Z_xu     ]
            [                                    I_n                        Z_xl ]

        which has ``updated_rhs_z = rhs.z - (S/Z) * rhs.s`` (vs forward's
        ``rhs.z - (1/Z) * rhs.s``) and slack recovery ``lhs.s = (rhs.s - lhs.z) / Z``
        (vs forward's ``(rhs.s - S * lhs.z) / Z``).
        """
        self._prepare_rhs(data, rhs, transpose=transpose)
        self._kkt_solver.solve(data, self._rhs_x_bar, rhs.y, self._rhs_z_bar, lhs.x, lhs.y, self._work_z)  # ! the second _work_z is used to hold delta_z, but useless anyway. Can be further optimized.
        # Iterative refinement applies to the forward condensed solve only (K_c
        # is symmetric so it *could* run for transpose too, but that path is for
        # implicit-differentiation gradients where the minor regularization-floor
        # error is accepted).
        # TODO: currently only do IR in the QP solve procedure, no IR in the implicit differentiation.
        if not transpose and self._use_iterative_refinement and settings.iterative_refinement_max_iter > 0:
            self.iterative_refinement(
                data, settings,
                self._rhs_x_bar, rhs.y, self._rhs_z_bar,
                lhs.x, lhs.y, self._work_z)
        self._recover_lhs(data, rhs, lhs, transpose=transpose)

    @nvtx.annotate("KKTSystem::_prepare_rhs")
    @cuda_graph_capture(key=lambda self, data, rhs, transpose=False: (rhs.buffer_ptr, bool(transpose)), enable=lambda self: self._settings.enable_cuda_graph)
    def _prepare_rhs(self, data: Data, rhs: Variables, transpose: bool = False):
        """Build the condensed KKT rhs by eliminating slacks and duals.

        The dual elimination (bound → x, inequality → z) is direction-agnostic —
        the reduced system for (x, y, z_cond) has the same LHS in both K and K^T.
        Only the preceding slack elimination differs: ``_eliminate_slacks_kernel``
        for forward, ``_eliminate_slacks_transposed_kernel`` for transposed.
        """
        B = self._batch_size
        wp_stream = wp.Stream(cuda_stream=cp.cuda.get_current_stream().ptr)

        if transpose:
            # K^T slack row:  I lhs_z + Z lhs_s = rhs.s → S/Z scaling on rhs_s.
            wp.launch(
                kernel=self._eliminate_slacks_transposed_kernel,
                dim=(B, data.num_ineq),
                inputs=[rhs.z_all, rhs.s_all, self._m_s_all, self._m_z_inv_all,
                        self._updated_rhs_z_all],
                device="cuda",
                stream=wp_stream,
            )
        else:
            # K slack row:  S lhs_z + Z lhs_s = rhs.s → 1/Z scaling on rhs_s.
            wp.launch(
                kernel=self._eliminate_slacks_kernel,
                dim=(B, data.num_ineq),
                inputs=[rhs.z_all, rhs.s_all, self._m_z_inv_all, self._updated_rhs_z_all],
                device="cuda",
                stream=wp_stream,
            )

        wp.launch(
            kernel=self._eliminate_duals_kernel,
            dim=(B, data.n + data.m),
            inputs=[
                self._inv_idx_xu, self._inv_idx_xl,
                self._inv_idx_hu, self._inv_idx_hl,
                rhs.x,
                self._w_bu_delta_inv, self._w_bl_delta_inv,
                data._x_b_scaling,
                self._updated_rhs_z_bu, self._updated_rhs_z_bl,
                self._rhs_x_bar,
                self._w_u_delta_inv, self._w_l_delta_inv,
                self._updated_rhs_z_u, self._updated_rhs_z_l,
                self._z_reg,
                self._rhs_z_bar,
            ],
            device="cuda",
            stream=wp_stream,
        )

        # # ---- ALTERNATIVE: pure CuPy implementation ----
        # # Eliminate slacks: forward has updated_rhs_z = rhs_z - inv(Z) * rhs_s,
        # #                   transpose has updated_rhs_z = rhs_z - (S/Z) * rhs_s.
        # if transpose:
        #     cp.multiply(self._m_s_all, self._m_z_inv_all, out=self._updated_rhs_z_all)
        #     cp.multiply(self._updated_rhs_z_all, rhs.s_all, out=self._updated_rhs_z_all)
        # else:
        #     cp.multiply(self._m_z_inv_all, rhs.s_all, out=self._updated_rhs_z_all)
        # cp.subtract(rhs.z_all, self._updated_rhs_z_all, out=self._updated_rhs_z_all)
        #
        # # Eliminate duals → rhs_x_bar, rhs_z_bar  (direction-agnostic)
        # self._rhs_x_bar[:] = rhs.x
        # if data.num_xu > 0:
        #     xbs_xu = data.x_b_scaling[:, data.idx_xu]
        #     self._rhs_x_bar[:, data.idx_xu] += xbs_xu * self._w_bu_delta_inv * self._updated_rhs_z_bu
        # if data.num_xl > 0:
        #     xbs_xl = data.x_b_scaling[:, data.idx_xl]
        #     self._rhs_x_bar[:, data.idx_xl] -= xbs_xl * self._w_bl_delta_inv * self._updated_rhs_z_bl
        #
        # if data.m > 0:
        #     self._rhs_z_bar[:] = 0.0
        #     if data.num_hu > 0:
        #         self._rhs_z_bar[:, data.idx_hu] += self._w_u_delta_inv * self._updated_rhs_z_u
        #     if data.num_hl > 0:
        #         self._rhs_z_bar[:, data.idx_hl] -= self._w_l_delta_inv * self._updated_rhs_z_l
        #     self._rhs_z_bar *= self._z_reg

    @nvtx.annotate("KKTSystem::_recover_lhs")
    @cuda_graph_capture(key=lambda self, data, rhs, lhs, transpose=False: (rhs.buffer_ptr, lhs.buffer_ptr, bool(transpose)), enable=lambda self: self._settings.enable_cuda_graph)
    def _recover_lhs(self, data: Data, rhs: Variables, lhs: Variables, transpose: bool = False):
        """Back-substitute lhs.z_* and lhs.s_* from the condensed (lhs.x, lhs.y).

        The dual back-sub (z_u, z_l, z_bu, z_bl) is direction-agnostic — the row
        equations for those blocks have the same structure in K and K^T once the
        slack has been eliminated. Only the final slack recovery differs:
        forward uses ``lhs.s = (rhs.s - S lhs.z) / Z``, transpose drops the S
        factor.
        """
        B = self._batch_size
        wp_stream = wp.Stream(cuda_stream=cp.cuda.get_current_stream().ptr)

        if data.m > 0:
            self.eval_G_xn(data, 1., lhs.x, self._work_z)

        if data.num_ineq > 0:
            wp.launch(
                kernel=self._recover_duals_kernel,
                dim=(B, data.num_ineq),
                inputs=[
                    self._work_z, lhs.x,
                    data.idx_hu, self._w_u_delta_inv, self._updated_rhs_z_u, lhs.z_u,
                    data.idx_hl, self._w_l_delta_inv, self._updated_rhs_z_l, lhs.z_l,
                    data.idx_xu, self._w_bu_delta_inv, self._m_z_bu_inv, rhs.z_bu, rhs.s_bu, lhs.z_bu,
                    data.idx_xl, self._w_bl_delta_inv, self._m_z_bl_inv, rhs.z_bl, rhs.s_bl, lhs.z_bl,
                    data.x_b_scaling],
                device="cuda",
                stream=wp_stream,
            )
            if transpose:
                # lhs.s = (rhs.s - lhs.z) / Z   (no factor of S)
                wp.launch(
                    kernel=self._recover_slacks_transposed_kernel,
                    dim=(B, data.num_ineq),
                    inputs=[rhs.s_all, lhs.z_all, self._m_z_inv_all, lhs.s_all],
                    device="cuda",
                    stream=wp_stream,
                )
            else:
                # lhs.s = (rhs.s - S lhs.z) / Z
                wp.launch(
                    kernel=self._recover_slacks_kernel,
                    dim=(B, data.num_ineq),
                    inputs=[rhs.s_all, lhs.z_all, self._m_s_all, self._m_z_inv_all, lhs.s_all],
                    device="cuda",
                    stream=wp_stream,
                )

        # # ---- ALTERNATIVE: pure CuPy implementation ----
        # # Recover duals (direction-agnostic)
        # if data.num_hu > 0:
        #     lhs.z_u[:] = self._work_z[:, data.idx_hu]
        #     lhs.z_u -= self._updated_rhs_z_u
        #     lhs.z_u *= self._w_u_delta_inv
        #
        # if data.num_hl > 0:
        #     lhs.z_l[:] = self._work_z[:, data.idx_hl]
        #     lhs.z_l *= -1.0
        #     lhs.z_l -= self._updated_rhs_z_l
        #     lhs.z_l *= self._w_l_delta_inv
        #
        # if data.num_xu > 0:
        #     xbs_xu = data.x_b_scaling[:, data.idx_xu]
        #     cp.multiply(self._m_z_bu_inv, rhs.s_bu, out=lhs.z_bu)
        #     lhs.z_bu += xbs_xu * lhs.x[:, data.idx_xu]
        #     lhs.z_bu -= rhs.z_bu
        #     lhs.z_bu *= self._w_bu_delta_inv
        #
        # if data.num_xl > 0:
        #     xbs_xl = data.x_b_scaling[:, data.idx_xl]
        #     cp.multiply(self._m_z_bl_inv, rhs.s_bl, out=lhs.z_bl)
        #     lhs.z_bl -= xbs_xl * lhs.x[:, data.idx_xl]
        #     lhs.z_bl -= rhs.z_bl
        #     lhs.z_bl *= self._w_bl_delta_inv
        #
        # # Recover slacks
        # if data.num_ineq > 0:
        #     if transpose:
        #         cp.subtract(rhs.s_all, lhs.z_all, out=lhs.s_all)
        #         cp.multiply(lhs.s_all, self._m_z_inv_all, out=lhs.s_all)
        #     else:
        #         cp.multiply(self._m_s_all, lhs.z_all, out=lhs.s_all)
        #         cp.subtract(rhs.s_all, lhs.s_all, out=lhs.s_all)
        #         cp.multiply(self._m_z_inv_all, lhs.s_all, out=lhs.s_all)

    @nvtx.annotate("KKTSystem::iterative_refinement")
    def iterative_refinement(self, data: Data, settings: Settings,
                             rhs_x: cp.ndarray, rhs_y: cp.ndarray, rhs_z: cp.ndarray,
                             lhs_x: cp.ndarray, lhs_y: cp.ndarray, lhs_z: cp.ndarray) -> bool:
        """Iterative refinement on the condensed 3-block KKT system.

        Refines (lhs_x, lhs_y, lhs_z) in-place so that
            K_condensed * [lhs_x; lhs_y; lhs_z] ≈ [rhs_x; rhs_y; rhs_z].

        All arrays are (B, k) shaped.
        Matches PIQP's solve() IR loop (kkt_system.tpp lines 294-339).
        Returns False if a non-finite residual is encountered.
        """
        n, p, m = data.n, data.p, data.m
        ref_err_x = self._iter_refine_error_xyz[:, :n]
        ref_err_y = self._iter_refine_error_xyz[:, n:n+p]
        ref_err_z = self._iter_refine_error_xyz[:, n+p:]
        ref_lhs_x = self._iter_refine_delta_xyz[:, :n]
        ref_lhs_y = self._iter_refine_delta_xyz[:, n:n+p]
        ref_lhs_z = self._iter_refine_delta_xyz[:, n+p:]

        rhs_norm = float(cp.max(cp.abs(rhs_x)))
        if data.p > 0:
            rhs_norm = max(rhs_norm, float(cp.max(cp.abs(rhs_y))))
        if data.m > 0:
            rhs_norm = max(rhs_norm, float(cp.max(cp.abs(rhs_z))))

        # Initial error computed on first iteration; subsequent iterations
        # reuse ref_err from candidate evaluation at end of previous iteration.
        refine_error = math.inf
        VERBOSE_IR = False
        tol = settings.iterative_refinement_eps_abs + settings.iterative_refinement_eps_rel * rhs_norm

        for i in range(settings.iterative_refinement_max_iter):
            # Compute error: initial solve (i==0) or candidate (i>0)
            if i == 0:
                self.get_refinement_error(
                    data, lhs_x, lhs_y, lhs_z, rhs_x, rhs_y, rhs_z,
                    ref_err_x, ref_err_y, ref_err_z)
            else:
                self.get_refinement_error(
                    data, ref_lhs_x, ref_lhs_y, ref_lhs_z, rhs_x, rhs_y, rhs_z,
                    ref_err_x, ref_err_y, ref_err_z)

            prev_refine_error = refine_error
            refine_error = float(cp.linalg.norm(self._iter_refine_error_xyz[:, :n+p+m].reshape(-1), ord=cp.inf))

            if VERBOSE_IR:
                if i == 0:
                    print(f"  IR iter {i}: error={refine_error:.2e}, tol={tol:.2e}")
                else:
                    print(f"  IR iter {i}: error={refine_error:.2e}, improvement={prev_refine_error/refine_error:.2f}x")

            if not math.isfinite(refine_error):
                if VERBOSE_IR:
                    print(f"  IR: non-finite error, aborting")
                return False

            if refine_error <= tol:
                if VERBOSE_IR:
                    print(f"  IR: converged at iter {i}")
                if i > 0:
                    cp.copyto(lhs_x, ref_lhs_x)
                    cp.copyto(lhs_y, ref_lhs_y)
                    cp.copyto(lhs_z, ref_lhs_z)
                break

            if i > 0:
                improvement_rate = prev_refine_error / refine_error
                if improvement_rate < settings.iterative_refinement_min_improvement_rate:
                    if improvement_rate > 1.0:
                        if VERBOSE_IR:
                            print(f"  IR: slow improvement ({improvement_rate:.2f}x < {settings.iterative_refinement_min_improvement_rate:.1f}x), accepting candidate")
                        cp.copyto(lhs_x, ref_lhs_x)
                        cp.copyto(lhs_y, ref_lhs_y)
                        cp.copyto(lhs_z, ref_lhs_z)
                    else:
                        if VERBOSE_IR:
                            print(f"  IR: no improvement ({improvement_rate:.2f}x), rejecting candidate")
                    break
                # Accept candidate
                cp.copyto(lhs_x, ref_lhs_x)
                cp.copyto(lhs_y, ref_lhs_y)
                cp.copyto(lhs_z, ref_lhs_z)

            # Solve for correction: K * ref_lhs = ref_err
            self._kkt_solver.solve(data,
                                   ref_err_x, ref_err_y, ref_err_z,
                                   ref_lhs_x, ref_lhs_y, ref_lhs_z)

            # Build candidate in ref_lhs: ref_lhs = lhs + correction
            ref_lhs_x += lhs_x
            ref_lhs_y += lhs_y
            ref_lhs_z += lhs_z

        return True

    @nvtx.annotate("KKTSystem::mul_condensed_kkt")
    def mul_condensed_kkt(self, data: Data,
                          lhs_x: cp.ndarray, lhs_y: cp.ndarray, lhs_z: cp.ndarray,
                          rhs_x: cp.ndarray, rhs_y: cp.ndarray, rhs_z: cp.ndarray) -> None:
        """
        Compute the matrix-vector product with the condensed (reduced) KKT matrix:

            [ P + x_reg*I   A^T       G^T    ] [ lhs_x ]   [ rhs_x ]
            [ A            -delta*I    0     ] [ lhs_y ] = [ rhs_y ]
            [ G              0       -z_reg  ] [ lhs_z ]   [ rhs_z ]

        where x_reg and z_reg incorporate the eliminated slack/bound contributions.
        Requires update_scalings_and_factor() to have been called first (sets _x_reg, _z_reg, _delta).

        All output arrays (rhs_x, rhs_y, rhs_z) are overwritten.
        """
        # rhs_x = P*lhs_x
        self.eval_P_x(data, 1., lhs_x, rhs_x)
        rhs_x += self._x_reg * lhs_x

        # A block: rhs_y = A*lhs_x, rhs_x += A^T*lhs_y
        if data.p > 0:
            self.eval_A_xn(data, 1., lhs_x, rhs_y)
            self.eval_AT_xt(data, 1., lhs_y, self._work_x)
            rhs_x += self._work_x
            rhs_y -= self._delta[:, None] * lhs_y

        # G block: rhs_z = G*lhs_x, rhs_x += G^T*lhs_z
        if data.m > 0:
            self.eval_G_xn(data, 1., lhs_x, rhs_z)
            self.eval_GT_xt(data, 1., lhs_z, self._work_x)
            rhs_x += self._work_x
            rhs_z -= self._z_reg * lhs_z

    @nvtx.annotate("KKTSystem::get_refinement_error")
    def get_refinement_error(self, data: Data,
                         lhs_x: cp.ndarray, lhs_y: cp.ndarray, lhs_z: cp.ndarray,
                         rhs_x: cp.ndarray, rhs_y: cp.ndarray, rhs_z: cp.ndarray,
                         err_x: cp.ndarray, err_y: cp.ndarray, err_z: cp.ndarray) -> float:
        """
        Compute the residual error of the condensed KKT solve:
            err = rhs - KKT * lhs
        and return ||err||_inf = max(||err_x||_inf, ||err_y||_inf, ||err_z||_inf).

        Args:
            lhs_x, lhs_y, lhs_z: current solution (primal, eq dual, ineq dual)
            rhs_x, rhs_y, rhs_z: right-hand side of condensed KKT system
            err_x, err_y, err_z: output arrays overwritten with residual
        Returns:
            Infinity norm of the residual.
        """
        # err = rhs - KKT * lhs
        self.mul_condensed_kkt(data, lhs_x, lhs_y, lhs_z, err_x, err_y, err_z)
        cp.subtract(rhs_x, err_x, out=err_x)
        cp.subtract(rhs_y, err_y, out=err_y)
        cp.subtract(rhs_z, err_z, out=err_z)

        # return max(||err_x||_inf, ||err_y||_inf, ||err_z||_inf)
        norm = float(cp.max(cp.abs(err_x)))
        if err_y.size > 0:
            norm = max(norm, float(cp.max(cp.abs(err_y))))
        if err_z.size > 0:
            norm = max(norm, float(cp.max(cp.abs(err_z))))
        return norm

    @nvtx.annotate("KKTSystem::eval_P_x")
    def eval_P_x(self, data: Data, alpha: float, x: cp.ndarray, z: cp.ndarray):
        """
        Evaluate alpha * P * x
        """
        self._kkt_solver.eval_P_x(data, alpha, x, z)
    
    @nvtx.annotate("KKTSystem::eval_A_xn")
    def eval_A_xn(self, data: Data, alpha_n: float, xn: cp.ndarray, zn: cp.ndarray):
        """
        Evaluate Ax with scaling factor alpha_n:
        zn = alpha_n * A * xn
        """
        self._kkt_solver.eval_A_xn(data, alpha_n, xn, zn)

    @nvtx.annotate("KKTSystem::eval_AT_xt")
    def eval_AT_xt(self, data: Data, alpha_t: float, xt: cp.ndarray, zt: cp.ndarray):
        """
        Evaluate A^T xt with scaling factor alpha_t:
        zt = alpha_t * A^T * xt
        """
        self._kkt_solver.eval_AT_xt(data, alpha_t, xt, zt)

    @nvtx.annotate("KKTSystem::eval_G_xn")
    def eval_G_xn(self, data: Data, alpha_n: float, xn: cp.ndarray, zn: cp.ndarray):
        """
        Evaluate Gx with scaling factor alpha_n:
        zn = alpha_n * G * xn
        """
        self._kkt_solver.eval_G_xn(data, alpha_n, xn, zn)

    @nvtx.annotate("KKTSystem::eval_GT_xt")
    def eval_GT_xt(self, data: Data, alpha_t: float, xt: cp.ndarray, zt: cp.ndarray):
        """
        Evaluate G^T xt with scaling factor alpha_t:
        zt = alpha_t * G^T * xt
        """
        self._kkt_solver.eval_GT_xt(data, alpha_t, xt, zt)


def create_build_inverse_index_kernel():
    """Create a kernel that builds an inverse index map: inv_idx[idx[t]] = t."""
    @wp.kernel
    def build_inverse_index_kernel(
        idx: wp.array(dtype=wp.int32),      # type: ignore
        inv_idx: wp.array(dtype=wp.int32),   # type: ignore
    ):
        t = wp.tid()
        inv_idx[idx[t]] = t
    return build_inverse_index_kernel


def create_update_regularizations_step_1_kernel():
    """Create kernel operating on contiguous s_all/z_all buffers. Performs:

        self._m_s_u = vars.s_u
        self._m_s_l = vars.s_l
        self._m_s_bu = vars.s_bu
        self._m_s_bl = vars.s_bl
        self._m_z_u_inv = 1. / vars.z_u
        self._m_z_l_inv = 1. / vars.z_l
        self._m_z_bu_inv = 1. / vars.z_bu
        self._m_z_bl_inv = 1. / vars.z_bl

        self._w_bu_delta_inv = 1. / (self._m_s_bu * self._m_z_bu_inv + delta)
        self._w_bl_delta_inv = 1. / (self._m_s_bl * self._m_z_bl_inv + delta)
        self._w_u_delta_inv = 1. / (self._m_s_u * self._m_z_u_inv + delta)
        self._w_l_delta_inv = 1. / (self._m_s_l * self._m_z_l_inv + delta)

        Since s and z are stored contiguously, it becomes:

        m_s_all[b, i]         = vars_s_all[b, i]
        m_z_inv_all[b, i]     = 1.0 / vars_z_all[b, i]
        w_delta_inv_all[b, i] = 1.0 / (s * z_inv + delta[b])
    """
    @wp.kernel
    def update_regularizations_step_1_kernel(
        vars_s_all: wp.array2d(dtype=wp.float64),       # type: ignore
        vars_z_all: wp.array2d(dtype=wp.float64),       # type: ignore
        m_s_all: wp.array2d(dtype=wp.float64),           # type: ignore
        m_z_inv_all: wp.array2d(dtype=wp.float64),       # type: ignore
        w_delta_inv_all: wp.array2d(dtype=wp.float64),   # type: ignore
        delta: wp.array(dtype=wp.float64),               # type: ignore
    ):
        b, i = wp.tid()
        s = vars_s_all[b, i]
        z_inv = wp.float64(1.0) / vars_z_all[b, i]
        m_s_all[b, i] = s
        m_z_inv_all[b, i] = z_inv
        w_delta_inv_all[b, i] = wp.float64(1.0) / (s * z_inv + delta[b])
    return update_regularizations_step_1_kernel


def create_update_regularizations_step_2_kernel(nx: int, nz: int):
    """Create kernel specialized for computing the regularization terms for x and z
    using a gather pattern.

    Equivalent to:
        x_reg[:] = rho
        x_reg[idx_xu] += w_bu_delta_inv
        x_reg[idx_xl] += w_bl_delta_inv

        z_reg[:] = 0.
        z_reg[idx_hu] += w_u_delta_inv
        z_reg[idx_hl] += w_l_delta_inv
        z_reg[:] = 1. / z_reg

    Each thread writes only to its own unique slot (x_reg[b, t] or z_reg[b, tz]),
    using inverse index maps to gather contributions.
    """
    @wp.kernel
    def update_regularizations_step_2_kernel(
        inv_idx_xu: wp.array(dtype=wp.int32),  # type: ignore
        inv_idx_xl: wp.array(dtype=wp.int32),  # type: ignore
        inv_idx_hu: wp.array(dtype=wp.int32),  # type: ignore
        inv_idx_hl: wp.array(dtype=wp.int32),  # type: ignore
        w_bu_delta_inv: wp.array2d(dtype=wp.float64),  # type: ignore
        w_bl_delta_inv: wp.array2d(dtype=wp.float64),  # type: ignore
        x_b_scaling: wp.array2d(dtype=wp.float64),  # type: ignore
        rho: wp.array(dtype=wp.float64),  # type: ignore
        x_reg: wp.array2d(dtype=wp.float64),  # type: ignore
        w_u_delta_inv: wp.array2d(dtype=wp.float64),  # type: ignore
        w_l_delta_inv: wp.array2d(dtype=wp.float64),  # type: ignore
        z_reg: wp.array2d(dtype=wp.float64),  # type: ignore
    ):
        b, t = wp.tid()
        nx_static = wp.static(nx)
        nz_static = wp.static(nz)

        if t < nx_static:
            val = rho[b]
            xb_scaling = x_b_scaling[b, t]
            xb_scaling_squared = xb_scaling * xb_scaling
            ixu = inv_idx_xu[t]
            ixl = inv_idx_xl[t]
            if ixu >= 0:
                val = val + xb_scaling_squared * w_bu_delta_inv[b, ixu]
            if ixl >= 0:
                val = val + xb_scaling_squared * w_bl_delta_inv[b, ixl]
            x_reg[b, t] = val
        elif t < nx_static + nz_static:
            tz = t - nx_static
            val = wp.float64(0.)
            ihu = inv_idx_hu[tz]
            ihl = inv_idx_hl[tz]
            if ihu >= 0:
                val = val + w_u_delta_inv[b, ihu]
            if ihl >= 0:
                val = val + w_l_delta_inv[b, ihl]
            z_reg[b, tz] = wp.float64(1.0) / val

    return update_regularizations_step_2_kernel


def create_eliminate_duals_kernel(nx: int, nz: int):
    """Create kernel specialized for eliminating duals using a gather pattern.

    Equivalent to:
        rhs_x_updated[:] = rhs_x
        rhs_x_updated[idx_xu] += w_bu_delta_inv * rhs_z_bu
        rhs_x_updated[idx_xl] -= w_bl_delta_inv * rhs_z_bl

        rhs_z_updated[:] = 0.
        rhs_z_updated[idx_hu] += w_u_delta_inv * rhs_z_u
        rhs_z_updated[idx_hl] -= w_l_delta_inv * rhs_z_l
        rhs_z_updated[:] *= z_reg

    Each thread writes only to its own unique slot (rhs_x_updated[t] or
    rhs_z_updated[tz]), using inverse index maps to gather contributions.
    """
    @wp.kernel
    def eliminate_duals_kernel(
        # inverse index maps (gather lookups)
        inv_idx_xu: wp.array(dtype=wp.int32),  # type: ignore
        inv_idx_xl: wp.array(dtype=wp.int32),  # type: ignore
        inv_idx_hu: wp.array(dtype=wp.int32),  # type: ignore
        inv_idx_hl: wp.array(dtype=wp.int32),  # type: ignore
        # prepare new rhs_x
        rhs_x: wp.array2d(dtype=wp.float64),  # type: ignore
        w_bu_delta_inv: wp.array2d(dtype=wp.float64),  # type: ignore
        w_bl_delta_inv: wp.array2d(dtype=wp.float64),  # type: ignore
        x_b_scaling: wp.array2d(dtype=wp.float64),  # type: ignore
        rhs_z_bu: wp.array2d(dtype=wp.float64),  # type: ignore
        rhs_z_bl: wp.array2d(dtype=wp.float64),  # type: ignore
        rhs_x_updated: wp.array2d(dtype=wp.float64),  # type: ignore
        # prepare new rhs_z
        w_u_delta_inv: wp.array2d(dtype=wp.float64),  # type: ignore
        w_l_delta_inv: wp.array2d(dtype=wp.float64),  # type: ignore
        rhs_z_u: wp.array2d(dtype=wp.float64),  # type: ignore
        rhs_z_l: wp.array2d(dtype=wp.float64),  # type: ignore
        z_reg: wp.array2d(dtype=wp.float64),  # type: ignore
        rhs_z_updated: wp.array2d(dtype=wp.float64),  # type: ignore
    ):
        b, t = wp.tid()
        nx_static = wp.static(nx)
        nz_static = wp.static(nz)

        if t < nx_static:
            val = rhs_x[b, t]
            xb_scaling = x_b_scaling[b, t]
            ixu = inv_idx_xu[t]
            ixl = inv_idx_xl[t]
            if ixu >= 0:
                val = val + xb_scaling * w_bu_delta_inv[b, ixu] * rhs_z_bu[b, ixu]
            if ixl >= 0:
                val = val - xb_scaling * w_bl_delta_inv[b, ixl] * rhs_z_bl[b, ixl]
            rhs_x_updated[b, t] = val

        elif t < nx_static + nz_static:
            tz = t - nx_static
            val = wp.float64(0.)
            ihu = inv_idx_hu[tz]
            ihl = inv_idx_hl[tz]
            if ihu >= 0:
                val = val + w_u_delta_inv[b, ihu] * rhs_z_u[b, ihu]
            if ihl >= 0:
                val = val - w_l_delta_inv[b, ihl] * rhs_z_l[b, ihl]
            rhs_z_updated[b, tz] = val * z_reg[b, tz]

    return eliminate_duals_kernel


def create_eliminate_slacks_kernel():
    """Batched element-wise kernel for eliminating slacks for inequalities.

        updated_rhs_z_all[b, i] = rhs_z_all[b, i] - m_z_inv_all[b, i] * rhs_s_all[b, i]
    """
    @wp.kernel
    def eliminate_slacks_kernel(
        rhs_z_all: wp.array2d(dtype=wp.float64),          # type: ignore
        rhs_s_all: wp.array2d(dtype=wp.float64),          # type: ignore
        m_z_inv_all: wp.array2d(dtype=wp.float64),        # type: ignore
        updated_rhs_z_all: wp.array2d(dtype=wp.float64),  # type: ignore
    ):
        b, i = wp.tid()
        updated_rhs_z_all[b, i] = -m_z_inv_all[b, i] * rhs_s_all[b, i] + rhs_z_all[b, i]

    return eliminate_slacks_kernel


def create_eliminate_slacks_transposed_kernel():
    """Transposed (K^T) variant of eliminate_slacks. Scales rhs_s by W = S/Z instead
    of 1/Z, because row 7..10 of K^T have S in the off-diagonal (vs. I in K).

        updated_rhs_z_all[b, i] = rhs_z_all[b, i] - m_s_all[b, i] * m_z_inv_all[b, i] * rhs_s_all[b, i]
    """
    @wp.kernel
    def eliminate_slacks_transposed_kernel(
        rhs_z_all: wp.array2d(dtype=wp.float64),          # type: ignore
        rhs_s_all: wp.array2d(dtype=wp.float64),          # type: ignore
        m_s_all: wp.array2d(dtype=wp.float64),            # type: ignore
        m_z_inv_all: wp.array2d(dtype=wp.float64),        # type: ignore
        updated_rhs_z_all: wp.array2d(dtype=wp.float64),  # type: ignore
    ):
        b, i = wp.tid()
        w = m_s_all[b, i] * m_z_inv_all[b, i]  # W = S / Z
        updated_rhs_z_all[b, i] = -w * rhs_s_all[b, i] + rhs_z_all[b, i]

    return eliminate_slacks_transposed_kernel


def create_recover_duals_kernel(num_hu: int, num_hl: int, num_xu: int, num_xl: int):
    """Create kernel specialized for recovering duals. Performs the operation:

    Performs the operation:
        lhs.z_u = self._w_u_delta_inv * (G_dx[:, data.idx_hu] - self._updated_rhs_z_u)
        lhs.z_l = self._w_l_delta_inv * (-G_dx[:, data.idx_hl] - self._updated_rhs_z_l)
        lhs.z_bu = self._w_bu_delta_inv * (xbs * lhs.x[:, data.idx_xu] - rhs.z_bu + self._m_z_bu_inv * rhs.s_bu)
        lhs.z_bl = -self._w_bl_delta_inv * (xbs * lhs.x[:, data.idx_xl] + rhs.z_bl - self._m_z_bl_inv * rhs.s_bl)
    """
    @wp.kernel
    def recover_duals_kernel(
        G_dx: wp.array2d(dtype=wp.float64),  # type: ignore
        lhs_x: wp.array2d(dtype=wp.float64),  # type: ignore
        # h_u
        idx_hu: wp.array(dtype=wp.int32),  # type: ignore
        w_u_delta_inv: wp.array2d(dtype=wp.float64),  # type: ignore
        rhs_z_u: wp.array2d(dtype=wp.float64),  # type: ignore
        lhs_z_u: wp.array2d(dtype=wp.float64),  # type: ignore
        # h_l
        idx_hl: wp.array(dtype=wp.int32),  # type: ignore
        w_l_delta_inv: wp.array2d(dtype=wp.float64),  # type: ignore
        rhs_z_l: wp.array2d(dtype=wp.float64),  # type: ignore
        lhs_z_l: wp.array2d(dtype=wp.float64),  # type: ignore
        # x_u
        idx_xu: wp.array(dtype=wp.int32),  # type: ignore
        w_bu_delta_inv: wp.array2d(dtype=wp.float64),  # type: ignore
        m_z_bu_inv: wp.array2d(dtype=wp.float64),  # type: ignore
        rhs_z_bu: wp.array2d(dtype=wp.float64),  # type: ignore
        rhs_s_bu: wp.array2d(dtype=wp.float64),  # type: ignore
        lhs_z_bu: wp.array2d(dtype=wp.float64),  # type: ignore
        # x_l
        idx_xl: wp.array(dtype=wp.int32),  # type: ignore
        w_bl_delta_inv: wp.array2d(dtype=wp.float64),  # type: ignore
        m_z_bl_inv: wp.array2d(dtype=wp.float64),  # type: ignore
        rhs_z_bl: wp.array2d(dtype=wp.float64),  # type: ignore
        rhs_s_bl: wp.array2d(dtype=wp.float64),  # type: ignore
        lhs_z_bl: wp.array2d(dtype=wp.float64),  # type: ignore
        # x_b_scaling
        x_b_scaling: wp.array2d(dtype=wp.float64),  # type: ignore
    ):
        b, t = wp.tid()
        num_hu_static = wp.static(num_hu)
        num_hl_static = wp.static(num_hl)
        num_xu_static = wp.static(num_xu)
        num_xl_static = wp.static(num_xl)

        if t < num_hu_static:
            lhs_z_u[b, t] = (G_dx[b, idx_hu[t]] - rhs_z_u[b, t]) * w_u_delta_inv[b, t]
        elif t < num_hu_static + num_hl_static:
            j = t - num_hu_static
            lhs_z_l[b, j] = (-G_dx[b, idx_hl[j]] - rhs_z_l[b, j]) * w_l_delta_inv[b, j]
        elif t < num_hu_static + num_hl_static + num_xu_static:
            j = t - num_hu_static - num_hl_static
            idx = idx_xu[j]
            lhs_z_bu[b, j] = (x_b_scaling[b, idx] * lhs_x[b, idx] - rhs_z_bu[b, j] + m_z_bu_inv[b, j] * rhs_s_bu[b, j]) * w_bu_delta_inv[b, j]
        elif t < num_hu_static + num_hl_static + num_xu_static + num_xl_static:
            j = t - num_hu_static - num_hl_static - num_xu_static
            idx = idx_xl[j]
            lhs_z_bl[b, j] = -(x_b_scaling[b, idx] * lhs_x[b, idx] + rhs_z_bl[b, j] - m_z_bl_inv[b, j] * rhs_s_bl[b, j]) * w_bl_delta_inv[b, j]
        else:
            return

    return recover_duals_kernel


def create_recover_slacks_kernel():
    """Create kernel specialized for eliminating slacks. Performs the operation:

        updated_lhs_z_u = inv(Z_u) (r_s_u - S_u lhs_z_u)
        updated_lhs_s_l = inv(Z_l) (r_s_l - S_l lhs_z_l)
        updated_lhs_s_bu = inv(Z_bu) (r_s_bu - S_bu lhs_z_bu)
        updated_lhs_s_bl = inv(Z_bl) (r_s_bl - S_bl lhs_z_bl)

        Since s and z are stored contiguously, it becomes:
        lhs_s_all[t] = m_z_inv_all[t] * (-m_s_all[t] * lhs_z_all[t] + rhs_s_all[t])

        The expression is written as (-m_s) * lhs_z + rhs_s to trigger FMA on GPU.
    """
    @wp.kernel
    def recover_slacks_kernel(
        rhs_s_all: wp.array2d(dtype=wp.float64),    # type: ignore
        lhs_z_all: wp.array2d(dtype=wp.float64),    # type: ignore
        m_s_all: wp.array2d(dtype=wp.float64),      # type: ignore
        m_z_inv_all: wp.array2d(dtype=wp.float64),  # type: ignore
        lhs_s_all: wp.array2d(dtype=wp.float64),    # type: ignore
    ):
        b, i = wp.tid()
        lhs_s_all[b, i] = m_z_inv_all[b, i] * ((-m_s_all[b, i]) * lhs_z_all[b, i] + rhs_s_all[b, i])

    return recover_slacks_kernel


def create_recover_slacks_transposed_kernel():
    """Transposed (K^T) variant of recover_slacks. The slack rows in K^T read
    ``I lhs_z + Z lhs_s = rhs_s``, so ``lhs_s = inv(Z) (rhs_s - lhs_z)``.

        lhs_s_all[b, i] = m_z_inv_all[b, i] * (-lhs_z_all[b, i] + rhs_s_all[b, i])
    """
    @wp.kernel
    def recover_slacks_transposed_kernel(
        rhs_s_all: wp.array2d(dtype=wp.float64),    # type: ignore
        lhs_z_all: wp.array2d(dtype=wp.float64),    # type: ignore
        m_z_inv_all: wp.array2d(dtype=wp.float64),  # type: ignore
        lhs_s_all: wp.array2d(dtype=wp.float64),    # type: ignore
    ):
        b, i = wp.tid()
        lhs_s_all[b, i] = m_z_inv_all[b, i] * (-lhs_z_all[b, i] + rhs_s_all[b, i])

    return recover_slacks_transposed_kernel