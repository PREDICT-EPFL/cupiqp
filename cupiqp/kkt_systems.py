import math

import cupy as cp
import warp as wp
import nvtx

from .results import Variables
from .data import Data
from .settings import Settings
from .sparse.sparse_kkt_solver import SparseKKTSolver
from .dense.dense_kkt_solver import DenseKKTSolver
from .multistage.multistage_kkt_solver import MultistageKKTSolver


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
        self._use_iterative_refinement = False

        self._x_reg = cp.nan * cp.ones(data.n)
        self._z_reg = cp.nan * cp.ones(data.m)
        self._P_diag = cp.empty(data.n)

        # used to store the rhs of the condensed KKT system
        # K_condensed * [dx; dy; dz] = [rhs_x_bar; rhs_y_bar; rhs_z_bar], 
        # where K_condensed is the condensed KKT matrix after eliminating duals of inequalities and box constraints and all slacks.
        # Since eliminating slacks and duals does not change rhs_y, we only need to store rhs_x_bar and rhs_z_bar.
        self._rhs_x_bar = cp.empty(data.n)
        self._rhs_z_bar = cp.empty(data.m)

        self._work_x = cp.nan * cp.zeros(data.n)
        self._work_z = cp.nan * cp.zeros(data.m)

        if settings.kkt_solver == "sparse_ldlt":
            self._kkt_solver = SparseKKTSolver(data, use_deterministic_mode=settings.use_deterministic_mode_for_cudss)
        elif settings.kkt_solver == "dense_cholesky":
            self._kkt_solver = DenseKKTSolver(data)
        elif settings.kkt_solver == "multistage_block_cholesky":
            self._kkt_solver = MultistageKKTSolver(data)
        else:
            raise ValueError(f"Unsupported kkt_solver: {settings.kkt_solver}")

        # store the value of slack and dual variables value at this iteration, will be used in recovering the slack step: S*delta_z + Z*delta_s = r_s
        # allocate for max possible size, but we will only use part of them according to idx_hu and idx_hl. 
        self._m_s_u = cp.zeros(data.num_hu)
        self._m_s_l = cp.zeros(data.num_hl)
        self._m_z_u_inv = cp.zeros(data.num_hu)
        self._m_z_l_inv = cp.zeros(data.num_hl)
        # allocate for max possible size, but we will only use part of them according to idx_xu and idx_xl. 
        # TODO: can be optimized later to reduce memory usage
        self._m_s_bu = cp.zeros(data.num_xu)
        self._m_s_bl = cp.zeros(data.num_xl)
        self._m_z_bu_inv = cp.zeros(data.num_xu)
        self._m_z_bl_inv = cp.zeros(data.num_xl)

        # pre-allocate memory for some variables used in factor and solve
        self._w_u_delta_inv = cp.zeros(data.num_hu)   # store 1./(s_u / z_u + delta)
        self._w_l_delta_inv = cp.zeros(data.num_hl)   # store 1./(s_l / z_l + delta)
        self._w_bu_delta_inv = cp.zeros(data.num_xu)  # store 1./(s_bu / z_bu + delta)
        self._w_bl_delta_inv = cp.zeros(data.num_xl)  # store 1./(s_bl / z_bl + delta)

        # pre-allocate memory for some variables used to store updated rhs in solve
        self._updated_rhs_z_u = cp.zeros(data.num_hu)
        self._updated_rhs_z_l = cp.zeros(data.num_hl)
        self._updated_rhs_z_bu = cp.zeros(data.num_xu)
        self._updated_rhs_z_bl = cp.zeros(data.num_xl)

        # pre-allocate memory for condensed KKT iterative refinement (operates on x, y, z blocks only)
        self._iter_refine_error_xyz = cp.zeros(data.n + data.p + data.m)
        self._iter_refine_delta_xyz = cp.zeros(data.n+data.p+data.m)

        # create kernels
        self._update_regulerization_step_1_kernel = create_update_regularizations_step_1_kernel(data.num_hu, data.num_hl, data.num_xu, data.num_xl)
        self._update_regulerization_step_2_kernel = create_update_regularizations_step_2_kernel(data.n, data.m)
        self._eliminate_slacks_kernel = create_eliminate_slacks_kernel(data.num_hu, data.num_hl, data.num_xu, data.num_xl)
        self._eliminate_duals_kernel = create_eliminate_duals_kernel(data.n, data.m)
        self._recover_duals_kernel = create_recover_duals_kernel(data.num_hu, data.num_hl, data.num_xu, data.num_xl)
        self._recover_slacks_kernel = create_recover_slacks_kernel(data.num_hu, data.num_hl, data.num_xu, data.num_xl)

        # Precompute inverse index maps for gather-pattern kernels.
        # inv_idx_xu[j] = i such that idx_xu[i] == j, or -1 if variable j has no upper bound.
        _build_inv_idx_kernel = create_build_inverse_index_kernel()
        self._inv_idx_xu = wp.full(data.n, value=-1, dtype=wp.int32, device="cuda")
        self._inv_idx_xl = wp.full(data.n, value=-1, dtype=wp.int32, device="cuda")
        self._inv_idx_hu = wp.full(data.m, value=-1, dtype=wp.int32, device="cuda") if data.m > 0 else wp.zeros(0, dtype=wp.int32, device="cuda")
        self._inv_idx_hl = wp.full(data.m, value=-1, dtype=wp.int32, device="cuda") if data.m > 0 else wp.zeros(0, dtype=wp.int32, device="cuda")
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
            
    @nvtx.annotate("KKTSystem::update_data")
    def update_data(self, data: Data, update_P: bool = False, update_A: bool = False, update_G: bool = False):
        self._kkt_solver.update_data(data, update_P, update_A, update_G)

    @nvtx.annotate("KKTSystem::update_scalings_and_factor")
    def update_scalings_and_factor(self, data: Data, settings: Settings, iterative_refinement: bool, rho: cp.ndarray, delta: cp.ndarray, vars: Variables) -> bool:
        """
        Update the scaling factors and refactor the KKT matrix.

        TODO: When iterative_refinement (IR) is True, adds static regularization to improve factorization stability. The solve() method will then run IR.

        The variable vars is the current primal/dual variable values at this iteration, i.e., values of x, y, z_u, z_l, s_u, s_l, z_bu, z_bl, s_bu, s_bl at the current iteration.
        """
        with nvtx.annotate("KKTSystem::update_scalings_and_factor::update_regularizations"):
            if not hasattr(self, '_update_scaling_and_factor_cuda_graphs'):
                self._update_scaling_and_factor_cuda_graphs = {}
                self._update_scaling_and_factor_cuda_graphs_capture_count = 0

            key = (vars.buffer_ptr, 
                self._m_s_u.data.ptr, self._m_s_l.data.ptr, self._m_s_bu.data.ptr, self._m_s_bl.data.ptr,
                self._m_z_u_inv.data.ptr, self._m_z_l_inv.data.ptr, self._m_z_bu_inv.data.ptr, self._m_z_bl_inv.data.ptr,
                self._w_u_delta_inv.data.ptr, self._w_l_delta_inv.data.ptr, self._w_bu_delta_inv.data.ptr, self._w_bl_delta_inv.data.ptr,
                self._x_reg.data.ptr, self._z_reg.data.ptr,
                iterative_refinement,
                )

            if key not in self._update_scaling_and_factor_cuda_graphs:
                self._update_scaling_and_factor_cuda_graphs_capture_count += 1
                # print(f"KKTSystems::_update_scaling_and_factor capturing CUDA graph (occurrence {self._update_scaling_and_factor_cuda_graphs_capture_count})...")
                stream = cp.cuda.Stream(non_blocking=True)
                wp_stream = wp.Stream(cuda_stream=stream.ptr)

                stream.begin_capture()
                with stream:

                    USE_WARP_IMPLEMENTATION = True  # set to False to use pure cupy implementation for updating regularization, which is easier to debug but slower.
                    if USE_WARP_IMPLEMENTATION:
                        wp.launch(
                            kernel=self._update_regulerization_step_1_kernel,
                            dim=data.num_hu + data.num_hl + data.num_xu + data.num_xl,
                            inputs=[vars.s_u, vars.s_l, vars.s_bu, vars.s_bl,
                                    vars.z_u, vars.z_l, vars.z_bu, vars.z_bl,
                                    self._m_s_u, self._m_s_l, self._m_s_bu, self._m_s_bl,
                                    self._m_z_u_inv, self._m_z_l_inv, self._m_z_bu_inv, self._m_z_bl_inv,
                                    self._w_u_delta_inv, self._w_l_delta_inv, self._w_bu_delta_inv, self._w_bl_delta_inv,
                                    delta],
                            device="cuda",
                            stream=wp_stream,
                        )

                        wp.launch(
                            kernel=self._update_regulerization_step_2_kernel,
                            dim=data.n + data.m,
                            inputs=[
                                self._inv_idx_xu,
                                self._inv_idx_xl,
                                self._inv_idx_hu,
                                self._inv_idx_hl,
                                self._w_bu_delta_inv,
                                self._w_bl_delta_inv,
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
                        # store the current slack and dual variable values at this iteration
                        self._m_s_u[:] = vars.s_u
                        self._m_s_l[:] = vars.s_l
                        self._m_s_bu[:] = vars.s_bu
                        self._m_s_bl[:] = vars.s_bl
                        cp.reciprocal(vars.z_u, out=self._m_z_u_inv)  # better than self._m_z_u_inv[:] = 1. / vars.z_u since it avoids temporary allocation
                        cp.reciprocal(vars.z_l, out=self._m_z_l_inv)
                        cp.reciprocal(vars.z_bu, out=self._m_z_bu_inv)
                        cp.reciprocal(vars.z_bl, out=self._m_z_bl_inv)

                        # eliminate the box constraints by adding their contribution to x_reg and z_reg
                        # self._w_bu_delta_inv[:] = 1. / (self._m_s_bu * self._m_z_bu_inv + delta)
                        cp.multiply(self._m_s_bu, self._m_z_bu_inv, out=self._w_bu_delta_inv)
                        cp.add(self._w_bu_delta_inv, delta, out=self._w_bu_delta_inv)
                        cp.reciprocal(self._w_bu_delta_inv, out=self._w_bu_delta_inv)
                        # self._w_bl_delta_inv[:] = 1. / (self._m_s_bl * self._m_z_bl_inv + delta)
                        cp.multiply(self._m_s_bl, self._m_z_bl_inv, out=self._w_bl_delta_inv)
                        cp.add(self._w_bl_delta_inv, delta, out=self._w_bl_delta_inv)
                        cp.reciprocal(self._w_bl_delta_inv, out=self._w_bl_delta_inv)
                        # self._w_u_delta_inv[:] = 1. / (vars.s_u / vars.z_u + delta)
                        cp.multiply(self._m_s_u, self._m_z_u_inv, out=self._w_u_delta_inv)
                        cp.add(self._w_u_delta_inv, delta, out=self._w_u_delta_inv)
                        cp.reciprocal(self._w_u_delta_inv, out=self._w_u_delta_inv)
                        # self._w_l_delta_inv[:] = 1. / (vars.s_l / vars.z_l + delta)
                        cp.multiply(self._m_s_l, self._m_z_l_inv, out=self._w_l_delta_inv)
                        cp.add(self._w_l_delta_inv, delta, out=self._w_l_delta_inv)
                        cp.reciprocal(self._w_l_delta_inv, out=self._w_l_delta_inv)

                        self._x_reg[:] = rho[0]
                        self._x_reg[data.idx_xu] += self._w_bu_delta_inv
                        self._x_reg[data.idx_xl] += self._w_bl_delta_inv
                        self._z_reg.fill(0.)
                        self._z_reg[data.idx_hu] += self._w_u_delta_inv
                        self._z_reg[data.idx_hl] += self._w_l_delta_inv
                        cp.reciprocal(self._z_reg, out=self._z_reg)

                    # TODO: in PIQP, if iterative_refinement is enabled, they add a small perturbation to the diagonal for improved stability. We can consider adding this as well.
                    self._kkt_solver.update_kkt(data, delta, self._x_reg, self._z_reg)

                self._update_scaling_and_factor_cuda_graphs[key] = stream.end_capture()

            self._update_scaling_and_factor_cuda_graphs[key].launch()

        self._rho = rho
        self._delta = delta
        self._use_iterative_refinement = iterative_refinement
        factor_success = self._kkt_solver.factor() # ! this is implicitly assuming idx_hu and idx_hl cover all indices of inequalities 0:m
        return factor_success
    
    @nvtx.annotate("KKTSystem::solve")
    def solve(self, data: Data, settings: Settings, rhs: Variables, lhs: Variables) -> None:
        stream_cp = cp.cuda.get_current_stream()
        stream_wp = wp.Stream(cuda_stream=stream_cp.ptr)
        
        with nvtx.annotate("KKTSystem::solve::prepare_rhs"):
            if not hasattr(self, '_prepare_rhs_cuda_graphs'):
                self._prepare_rhs_cuda_graphs = {}

            key = (rhs.buffer_ptr,)

            if key not in self._prepare_rhs_cuda_graphs:
                # stream_cp_capture and stream_wp_capture are used to launch kernels to capture cuda graph, but the actual computation is captured in stream_cp and stream_wp
                stream_cp_capture = cp.cuda.Stream(non_blocking=True)
                stream_wp_capture = wp.Stream(cuda_stream=stream_cp_capture.ptr)

                stream_cp_capture.begin_capture()
                with stream_cp_capture:
                    wp.launch(
                        kernel=self._eliminate_slacks_kernel,
                        dim=data.num_hu+data.num_hl+data.num_xu+data.num_xl,
                        inputs=[rhs.z_u, rhs.s_u, self._m_z_u_inv, self._updated_rhs_z_u,
                                rhs.z_l, rhs.s_l, self._m_z_l_inv, self._updated_rhs_z_l,
                                rhs.z_bu, rhs.s_bu, self._m_z_bu_inv, self._updated_rhs_z_bu,
                                rhs.z_bl, rhs.s_bl, self._m_z_bl_inv, self._updated_rhs_z_bl],
                        device="cuda",
                        stream=stream_wp_capture,
                    )
                    wp.launch(
                        kernel=self._eliminate_duals_kernel,
                        dim=data.n + data.m,
                        inputs=[
                            self._inv_idx_xu, self._inv_idx_xl,
                            self._inv_idx_hu, self._inv_idx_hl,
                            rhs.x,
                            self._w_bu_delta_inv, self._w_bl_delta_inv,
                            self._updated_rhs_z_bu, self._updated_rhs_z_bl,
                            self._rhs_x_bar,
                            self._w_u_delta_inv, self._w_l_delta_inv,
                            self._updated_rhs_z_u, self._updated_rhs_z_l,
                            self._z_reg,
                            self._rhs_z_bar,
                        ],
                        device="cuda",
                        stream=stream_wp_capture,
                    )

                    # # ! ALTERNATIVE IMPLEMENTATION (pure cupy operations)
                    # ------ elliminate slack variables from rhs
                    # # rhs_z_u - inv(Z_u) * r_s_u
                    # cp.multiply(self._m_z_u_inv, rhs.s_u, out=self._updated_rhs_z_u)
                    # cp.subtract(rhs.z_u, self._updated_rhs_z_u, out=self._updated_rhs_z_u)
                    # # rhs_z_l - inv(Z_l) * r_s_l
                    # cp.multiply(self._m_z_l_inv, rhs.s_l, out=self._updated_rhs_z_l)
                    # cp.subtract(rhs.z_l, self._updated_rhs_z_l, out=self._updated_rhs_z_l)
                    # # rhs_z_bu - inv(Z_bu) * r_s_bu
                    # cp.multiply(self._m_z_bu_inv, rhs.s_bu, out=self._updated_rhs_z_bu)
                    # cp.subtract(rhs.z_bu, self._updated_rhs_z_bu, out=self._updated_rhs_z_bu)
                    # # rhs_z_bl - inv(Z_bl) * r_s_bl
                    # cp.multiply(self._m_z_bl_inv, rhs.s_bl, out=self._updated_rhs_z_bl)
                    # cp.subtract(rhs.z_bl, self._updated_rhs_z_bl, out=self._updated_rhs_z_bl)

                    # ------ elliminate dual variables from rhs to yield one single rhs_z passing to kkt solver
                    # To avoid avoid extra allocation, we use:
                    # self._work_x to hold modified rhs_x (to be passed to KKTSolver), self._work_z to hold modified rhs_z (to be passed to KKTSolver)
                    # use lhs.z_* to hold temporary value self._w_u_delta_inv * self._updated_rhs_z_u, and so on

                    # The below code is equivalent to:
                    # self._work_x[:] = rhs.x
                    # self._work_x[data.idx_xu] += self._w_bu_delta_inv * self._updated_rhs_z_bu
                    # self._work_x[data.idx_xl] -= self._w_bl_delta_inv * self._updated_rhs_z_bl
                    # self._work_z[:] = 0.
                    # self._work_z[data.idx_hu] += self._w_u_delta_inv * self._updated_rhs_z_u
                    # self._work_z[data.idx_hl] -= self._w_l_delta_inv * self._updated_rhs_z_l
                    # self._work_z[:] *= self._z_reg
                    
                    # self._work_x[:] = rhs.x
                    # cp.multiply(self._w_bu_delta_inv, self._updated_rhs_z_bu, out=lhs.z_bu)
                    # cp.add.at(self._work_x, data.idx_xu, lhs.z_bu)
                    # cp.multiply(self._w_bl_delta_inv, self._updated_rhs_z_bl, out=lhs.z_bl)
                    # cp.negative(lhs.z_bl, out=lhs.z_bl)
                    # cp.add.at(self._work_x, data.idx_xl, lhs.z_bl)
                    # self._work_z.fill(0)  # faster than cp.zeros assignment
                    # cp.multiply(self._w_u_delta_inv, self._updated_rhs_z_u, out=lhs.z_u) # use lhs.z_u as temporary storage
                    # cp.add.at(self._work_z, data.idx_hu, lhs.z_u)
                    # cp.multiply(self._w_l_delta_inv, self._updated_rhs_z_l, out=lhs.z_l)
                    # cp.negative(lhs.z_l, out=lhs.z_l)
                    # cp.add.at(self._work_z, data.idx_hl, lhs.z_l)
                    # self._work_z[:] *= self._z_reg

                self._prepare_rhs_cuda_graphs[key] = stream_cp_capture.end_capture()

            self._prepare_rhs_cuda_graphs[key].launch()

        self._kkt_solver.solve(data, self._rhs_x_bar, rhs.y, self._rhs_z_bar, lhs.x, lhs.y, self._work_z)  # ! the second _work_z is used to hold delta_z, but useless anyway. Can be further optimized.

        if self._use_iterative_refinement and settings.iterative_refinement_max_iter > 0:
            self.iterative_refinement(
                data, settings,
                self._rhs_x_bar, rhs.y, self._rhs_z_bar,
                lhs.x, lhs.y, self._work_z)

        with nvtx.annotate("KKTSystem::solve::recover_lhs"):
            if not hasattr(self, '_recover_lhs_cuda_graphs'):
                self._recover_lhs_cuda_graphs = {}

            key = (
                rhs.buffer_ptr, lhs.buffer_ptr, self._work_z.data.ptr, 
                self._w_u_delta_inv.data.ptr, self._w_l_delta_inv.data.ptr, 
                self._w_bu_delta_inv.data.ptr, self._w_bl_delta_inv.data.ptr,
                )

            if key not in self._recover_lhs_cuda_graphs:
                stream_cp_capture = cp.cuda.Stream(non_blocking=True)
                stream_wp_capture = wp.Stream(cuda_stream=stream_cp_capture.ptr)

                stream_cp_capture.begin_capture()
                with stream_cp_capture:
                    self.eval_G_xn(data, 1., lhs.x, self._work_z)  # G * delta_x, where delta_x is stored in lhs.x
                    wp.launch(
                        kernel=self._recover_duals_kernel,
                        dim=data.num_hu+data.num_hl+data.num_xu+data.num_xl,
                        inputs=[
                            self._work_z, lhs.x,
                            data.idx_hu, self._w_u_delta_inv, self._updated_rhs_z_u, lhs.z_u,
                            data.idx_hl, self._w_l_delta_inv, self._updated_rhs_z_l, lhs.z_l,
                            data.idx_xu, self._w_bu_delta_inv, self._m_z_bu_inv, rhs.z_bu, rhs.s_bu, lhs.z_bu,
                            data.idx_xl, self._w_bl_delta_inv, self._m_z_bl_inv, rhs.z_bl, rhs.s_bl, lhs.z_bl],
                        device="cuda",
                        stream=stream_wp_capture,
                    )
                    wp.launch(
                        kernel=self._recover_slacks_kernel,
                        dim=data.num_hu+data.num_hl+data.num_xu+data.num_xl,
                        inputs=[rhs.s_u, lhs.z_u, self._m_s_u, self._m_z_u_inv, lhs.s_u,
                                rhs.s_l, lhs.z_l, self._m_s_l, self._m_z_l_inv, lhs.s_l,
                                rhs.s_bu, lhs.z_bu, self._m_s_bu, self._m_z_bu_inv, lhs.s_bu,
                                rhs.s_bl, lhs.z_bl, self._m_s_bl, self._m_z_bl_inv, lhs.s_bl],
                        device="cuda",
                        stream=stream_wp_capture,
                    )

                    # # ! ALTERNATIVE IMPLEMENTATION (pure cupy operations)

                    # ----- recover dual variables on lhs
                    # The below code is equivalent to:
                    # lhs.z_u[:] = self._w_u_delta_inv * (G_dx[data.idx_hu] - self._updated_rhs_z_u)   # delta_z_u
                    # lhs.z_l[:] = self._w_l_delta_inv * (-G_dx[data.idx_hl] - self._updated_rhs_z_l)  # delta_z_l
                    # lhs.z_bu[:] = self._w_bu_delta_inv * (lhs.x[data.idx_xu] - rhs.z_bu + self._m_z_bu_inv * rhs.s_bu)  # delta_z_bu
                    # lhs.z_bl[:] = -self._w_bl_delta_inv * (lhs.x[data.idx_xl] + rhs.z_bl - self._m_z_bl_inv * rhs.s_bl)  # delta_z_bl
                    
                    # lhs.z_u[:] = self._work_z[data.idx_hu]
                    # lhs.z_u -= self._updated_rhs_z_u
                    # lhs.z_u *= self._w_u_delta_inv

                    # lhs.z_l[:] = self._work_z[data.idx_hl]
                    # lhs.z_l *= -1.
                    # lhs.z_l -= self._updated_rhs_z_l
                    # lhs.z_l *= self._w_l_delta_inv

                    # cp.multiply(self._m_z_bu_inv, rhs.s_bu, out=lhs.z_bu)
                    # lhs.z_bu += lhs.x[data.idx_xu]
                    # lhs.z_bu -= rhs.z_bu
                    # lhs.z_bu *= self._w_bu_delta_inv

                    # cp.multiply(self._m_z_bl_inv, rhs.s_bl, out=lhs.z_bl)
                    # lhs.z_bl -= lhs.x[data.idx_xl]
                    # lhs.z_bl -= rhs.z_bl
                    # lhs.z_bl *= self._w_bl_delta_inv
                    
                    # ----- recover slack variable on lhs
                    # The below code is equivalent to:
                    # lhs.s_u[:] = self._m_z_u_inv * (rhs.s_u - self._m_s_u * lhs.z_u)  # delta_s_u = inv(Z_u) (r_s_u - S_u delta_z_u)
                    # lhs.s_l[:] = self._m_z_l_inv * (rhs.s_l - self._m_s_l * lhs.z_l)  # delta_s_l = inv(Z_l) (r_s_l - S_l delta_z_l)
                    # lhs.s_bu[:] = self._m_z_bu_inv * (rhs.s_bu - self._m_s_bu * lhs.z_bu)  # delta_s_bu = inv(Z_bu) (r_s_bu - S_bu delta_z_bu)
                    # lhs.s_bl[:] = self._m_z_bl_inv * (rhs.s_bl - self._m_s_bl * lhs.z_bl)  # delta_s_bl = inv(Z_bl) (r_s_bl - S_bl delta_z_bl)

                    # cp.multiply(self._m_s_u, lhs.z_u, out=lhs.s_u)
                    # cp.subtract(rhs.s_u, lhs.s_u, out=lhs.s_u)
                    # cp.multiply(self._m_z_u_inv, lhs.s_u, out=lhs.s_u)

                    # cp.multiply(self._m_s_l, lhs.z_l, out=lhs.s_l)
                    # cp.subtract(rhs.s_l, lhs.s_l, out=lhs.s_l)
                    # cp.multiply(self._m_z_l_inv, lhs.s_l, out=lhs.s_l)

                    # cp.multiply(self._m_s_bu, lhs.z_bu, out=lhs.s_bu)
                    # cp.subtract(rhs.s_bu, lhs.s_bu, out=lhs.s_bu)
                    # cp.multiply(self._m_z_bu_inv, lhs.s_bu, out=lhs.s_bu)

                    # cp.multiply(self._m_s_bl, lhs.z_bl, out=lhs.s_bl)
                    # cp.subtract(rhs.s_bl, lhs.s_bl, out=lhs.s_bl)
                    # cp.multiply(self._m_z_bl_inv, lhs.s_bl, out=lhs.s_bl)

                self._recover_lhs_cuda_graphs[key] = stream_cp_capture.end_capture()

            self._recover_lhs_cuda_graphs[key].launch()

    @nvtx.annotate("KKTSystem::iterative_refinement")
    def iterative_refinement(self, data: Data, settings: Settings,
                             rhs_x: cp.ndarray, rhs_y: cp.ndarray, rhs_z: cp.ndarray,
                             lhs_x: cp.ndarray, lhs_y: cp.ndarray, lhs_z: cp.ndarray) -> bool:
        """Iterative refinement on the condensed 3-block KKT system.

        Refines (lhs_x, lhs_y, lhs_z) in-place so that
            K_condensed * [lhs_x; lhs_y; lhs_z] ≈ [rhs_x; rhs_y; rhs_z].

        Matches PIQP's solve() IR loop (kkt_system.tpp lines 294-339).
        Returns False if a non-finite residual is encountered.
        """
        ref_err_x = self._iter_refine_error_xyz[:data.n]
        ref_err_y = self._iter_refine_error_xyz[data.n:data.n+data.p]
        ref_err_z = self._iter_refine_error_xyz[data.n+data.p:]
        ref_lhs_x = self._iter_refine_delta_xyz[:data.n]
        ref_lhs_y = self._iter_refine_delta_xyz[data.n:data.n+data.p]
        ref_lhs_z = self._iter_refine_delta_xyz[data.n+data.p:]

        rhs_norm = float(cp.max(cp.abs(rhs_x)))
        if data.p > 0:
            rhs_norm = max(rhs_norm, float(cp.max(cp.abs(rhs_y))))
        if data.m > 0:
            rhs_norm = max(rhs_norm, float(cp.max(cp.abs(rhs_z))))

        # Initial error computed on first iteration; subsequent iterations
        # reuse ref_err from candidate evaluation at end of previous iteration.
        refine_error = math.inf
        VERBOSE_IR = True
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
            refine_error = float(cp.linalg.norm(self._iter_refine_error_xyz[:data.n + data.p + data.m], ord=cp.inf))

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
            rhs_y -= self._delta[0] * lhs_y

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
        idx: wp.array(dtype=wp.int32),      # pyright: ignore[reportInvalidTypeForm]
        inv_idx: wp.array(dtype=wp.int32),   # pyright: ignore[reportInvalidTypeForm]
    ):
        t = wp.tid()
        inv_idx[idx[t]] = t
    return build_inverse_index_kernel


def create_update_regularizations_step_1_kernel(num_hu: int, num_hl: int, num_xu: int, num_xl: int):
    """Create kernel specialized for ..., which will be used in factor and solve. Performs the operation:

        self._m_s_u[:] = vars.s_u
        self._m_s_l[:] = vars.s_l
        self._m_s_bu[:] = vars.s_bu
        self._m_s_bl[:] = vars.s_bl

        self._m_z_u_inv[:] = 1. / vars.z_u
        self._m_z_l_inv[:] = 1. / vars.z_l
        self._m_z_bu_inv[:] = 1. / vars.z_bu
        self._m_z_bl_inv[:] = 1. / vars.z_bl

        self._w_bu_delta_inv[:] = 1. / (self._m_s_bu * self._m_z_bu_inv + delta)
        self._w_bl_delta_inv[:] = 1. / (self._m_s_bl * self._m_z_bl_inv + delta)
        self._w_u_delta_inv[:] = 1. / (self._m_s_u * self._m_z_u_inv + delta)
        self._w_l_delta_inv[:] = 1. / (self._m_s_l * self._m_z_l_inv + delta)
    """
    @wp.kernel
    def update_regularizations_step_1_kernel(
        vars_s_u: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        vars_s_l: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        vars_s_bu: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        vars_s_bl: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        vars_z_u: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        vars_z_l: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        vars_z_bu: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        vars_z_bl: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        m_s_u: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        m_s_l: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        m_s_bu: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        m_s_bl: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        m_z_u_inv: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        m_z_l_inv: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        m_z_bu_inv: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        m_z_bl_inv: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        w_u_delta_inv: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        w_l_delta_inv: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        w_bu_delta_inv: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        w_bl_delta_inv: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        delta: wp.array(dtype=wp.float64)  # pyright: ignore[reportInvalidTypeForm]
    ):
        t = wp.tid()
        num_hu_static = wp.static(num_hu)
        num_hl_static = wp.static(num_hl)
        num_xu_static = wp.static(num_xu)
        num_xl_static = wp.static(num_xl)

        if t < num_hu_static:
            m_s_u[t] = vars_s_u[t]
            m_z_u_inv[t] = wp.float64(1.0) / vars_z_u[t]
            w_u_delta_inv[t] = wp.float64(1.0) / (m_s_u[t] * m_z_u_inv[t] + delta[0])
        elif t < num_hu_static + num_hl_static:
            t_hl = t - num_hu_static
            m_s_l[t_hl] = vars_s_l[t_hl]
            m_z_l_inv[t_hl] = wp.float64(1.0) / vars_z_l[t_hl]
            w_l_delta_inv[t_hl] = wp.float64(1.0) / (m_s_l[t_hl] * m_z_l_inv[t_hl] + delta[0])
        elif t < num_hu_static + num_hl_static + num_xu_static:
            t_xu = t - num_hu_static - num_hl_static
            m_s_bu[t_xu] = vars_s_bu[t_xu]
            m_z_bu_inv[t_xu] = wp.float64(1.0) / vars_z_bu[t_xu]
            w_bu_delta_inv[t_xu] = wp.float64(1.0) / (m_s_bu[t_xu] * m_z_bu_inv[t_xu] + delta[0])
        elif t < num_hu_static + num_hl_static + num_xu_static + num_xl_static:
            t_xl = t - num_hu_static - num_hl_static - num_xu_static
            m_s_bl[t_xl] = vars_s_bl[t_xl]
            m_z_bl_inv[t_xl] = wp.float64(1.0) / vars_z_bl[t_xl]
            w_bl_delta_inv[t_xl] = wp.float64(1.0) / (m_s_bl[t_xl] * m_z_bl_inv[t_xl] + delta[0])
        else:
            return
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

    Each thread writes only to its own unique slot (x_reg[t] or z_reg[tz]),
    using inverse index maps to gather contributions.
    """
    @wp.kernel
    def update_regularizations_step_2_kernel(
        inv_idx_xu: wp.array(dtype=wp.int32),  # pyright: ignore[reportInvalidTypeForm]
        inv_idx_xl: wp.array(dtype=wp.int32),  # pyright: ignore[reportInvalidTypeForm]
        inv_idx_hu: wp.array(dtype=wp.int32),  # pyright: ignore[reportInvalidTypeForm]
        inv_idx_hl: wp.array(dtype=wp.int32),  # pyright: ignore[reportInvalidTypeForm]
        w_bu_delta_inv: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        w_bl_delta_inv: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        rho: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        x_reg: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        w_u_delta_inv: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        w_l_delta_inv: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        z_reg: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
    ):
        t = wp.tid()
        nx_static = wp.static(nx)
        nz_static = wp.static(nz)

        if t < nx_static:
            val = rho[0]
            ixu = inv_idx_xu[t]
            ixl = inv_idx_xl[t]
            if ixu >= 0:
                val = val + w_bu_delta_inv[ixu]
            if ixl >= 0:
                val = val + w_bl_delta_inv[ixl]
            x_reg[t] = val
        elif t < nx_static + nz_static:
            tz = t - nx_static
            val = wp.float64(0.)
            ihu = inv_idx_hu[tz]
            ihl = inv_idx_hl[tz]
            if ihu >= 0:
                val = val + w_u_delta_inv[ihu]
            if ihl >= 0:
                val = val + w_l_delta_inv[ihl]
            z_reg[tz] = wp.float64(1.0) / val

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
        inv_idx_xu: wp.array(dtype=wp.int32),  # pyright: ignore[reportInvalidTypeForm]
        inv_idx_xl: wp.array(dtype=wp.int32),  # pyright: ignore[reportInvalidTypeForm]
        inv_idx_hu: wp.array(dtype=wp.int32),  # pyright: ignore[reportInvalidTypeForm]
        inv_idx_hl: wp.array(dtype=wp.int32),  # pyright: ignore[reportInvalidTypeForm]
        # prepare new rhs_x
        rhs_x: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        w_bu_delta_inv: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        w_bl_delta_inv: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        rhs_z_bu: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        rhs_z_bl: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        rhs_x_updated: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        # prepare new rhs_z
        w_u_delta_inv: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        w_l_delta_inv: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        rhs_z_u: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        rhs_z_l: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        z_reg: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        rhs_z_updated: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
    ):
        t = wp.tid()
        nx_static = wp.static(nx)
        nz_static = wp.static(nz)

        if t < nx_static:
            val = rhs_x[t]
            ixu = inv_idx_xu[t]
            ixl = inv_idx_xl[t]
            if ixu >= 0:
                val = val + w_bu_delta_inv[ixu] * rhs_z_bu[ixu]
            if ixl >= 0:
                val = val - w_bl_delta_inv[ixl] * rhs_z_bl[ixl]
            rhs_x_updated[t] = val

        elif t < nx_static + nz_static:
            tz = t - nx_static
            val = wp.float64(0.)
            ihu = inv_idx_hu[tz]
            ihl = inv_idx_hl[tz]
            if ihu >= 0:
                val = val + w_u_delta_inv[ihu] * rhs_z_u[ihu]
            if ihl >= 0:
                val = val - w_l_delta_inv[ihl] * rhs_z_l[ihl]
            rhs_z_updated[tz] = val * z_reg[tz]

    return eliminate_duals_kernel


def create_eliminate_slacks_kernel(num_hu: int, num_hl: int, num_xu: int, num_xl: int):
    """Create kernel specialized for eliminating slacks. Performs the operation:
    
        updated_rhs_z_u = rhs_z_u - inv(Z_u) * r_s_u
        updated_rhs_z_l = rhs_z_l - inv(Z_l) * r_s_l
        updated_rhs_z_bu = rhs_z_bu - inv(Z_bu) * r_s_bu
        updated_rhs_z_bl = rhs_z_bl - inv(Z_bl) * r_s_bl
    """
    @wp.kernel
    def eliminate_slacks_kernel(
        # h_u
        rhs_z_u: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        rhs_s_u: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        result_z_u_inv: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        updated_rhs_z_u: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        # h_l
        rhs_z_l: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        rhs_s_l: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        result_z_l_inv: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        updated_rhs_z_l: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        # x_u
        rhs_z_bu: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        rhs_s_bu: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        result_z_bu_inv: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        updated_rhs_z_bu: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        # x_l
        rhs_z_bl: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        rhs_s_bl: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        result_z_bl_inv: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        updated_rhs_z_bl: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
    ):
        t = wp.tid()
        num_hu_static = wp.static(num_hu)
        num_hl_static = wp.static(num_hl)
        num_xu_static = wp.static(num_xu)
        num_xl_static = wp.static(num_xl)

        if t < num_hu_static:
            updated_rhs_z_u[t] = -result_z_u_inv[t] * rhs_s_u[t] + rhs_z_u[t]
        elif t < num_hu_static + num_hl_static:
            offset = num_hu_static
            updated_rhs_z_l[t - offset] = -result_z_l_inv[t - offset] * rhs_s_l[t - offset] + rhs_z_l[t - offset]
        elif t < num_hu_static + num_hl_static + num_xu_static:
            offset = num_hu_static + num_hl_static
            updated_rhs_z_bu[t - offset] = -result_z_bu_inv[t - offset] * rhs_s_bu[t - offset] + rhs_z_bu[t - offset]
        elif t < num_hu_static + num_hl_static + num_xu_static + num_xl_static:
            offset = num_hu_static + num_hl_static + num_xu_static
            updated_rhs_z_bl[t - offset] = -result_z_bl_inv[t - offset] * rhs_s_bl[t - offset] + rhs_z_bl[t - offset]
        else:
            return

    return eliminate_slacks_kernel


def create_recover_duals_kernel(num_hu: int, num_hl: int, num_xu: int, num_xl: int):
    """Create kernel specialized for recovering duals. Performs the operation:

        lhs.z_u[:] = self._w_u_delta_inv * (G_dx[data.idx_hu] - self._updated_rhs_z_u)
        lhs.z_l[:] = self._w_l_delta_inv * (-G_dx[data.idx_hl] - self._updated_rhs_z_l)
        lhs.z_bu[:] = self._w_bu_delta_inv * (lhs.x[data.idx_xu] - rhs.z_bu + self._m_z_bu_inv * rhs.s_bu)
        lhs.z_bl[:] = -self._w_bl_delta_inv * (lhs.x[data.idx_xl] + rhs.z_bl - self._m_z_bl_inv * rhs.s_bl)
    """
    @wp.kernel
    def recover_duals_kernel(
        G_dx: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        lhs_x: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        # h_u
        idx_hu: wp.array(dtype=wp.int32),  # pyright: ignore[reportInvalidTypeForm]
        w_u_delta_inv: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        rhs_z_u: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        lhs_z_u: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        # h_l
        idx_hl: wp.array(dtype=wp.int32),  # pyright: ignore[reportInvalidTypeForm]
        w_l_delta_inv: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        rhs_z_l: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        lhs_z_l: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        # x_u
        idx_xu: wp.array(dtype=wp.int32),  # pyright: ignore[reportInvalidTypeForm]
        w_bu_delta_inv: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        m_z_bu_inv: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        rhs_z_bu: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        rhs_s_bu: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        lhs_z_bu: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        # x_l
        idx_xl: wp.array(dtype=wp.int32),  # pyright: ignore[reportInvalidTypeForm]
        w_bl_delta_inv: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        m_z_bl_inv: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        rhs_z_bl: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        rhs_s_bl: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        lhs_z_bl: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
    ):
        t = wp.tid()
        num_hu_static = wp.static(num_hu)
        num_hl_static = wp.static(num_hl)
        num_xu_static = wp.static(num_xu)
        num_xl_static = wp.static(num_xl)

        # lhs.z_u[:] = self._w_u_delta_inv * (G_dx[data.idx_hu] - self._updated_rhs_z_u)
        if t < num_hu_static:
            lhs_z_u[t] = G_dx[idx_hu[t]] - rhs_z_u[t]
            lhs_z_u[t] *= w_u_delta_inv[t]
        # lhs.z_l[:] = self._w_l_delta_inv * (-G_dx[data.idx_hl] - self._updated_rhs_z_l)
        elif t < num_hu_static + num_hl_static:
            offset = num_hu_static
            lhs_z_l[t - offset] = -G_dx[idx_hl[t - offset]] - rhs_z_l[t - offset]
            lhs_z_l[t - offset] *= w_l_delta_inv[t - offset]
        # lhs.z_bu[:] = self._w_bu_delta_inv * (lhs.x[data.idx_xu] - rhs.z_bu + self._m_z_bu_inv * rhs.s_bu)
        elif t < num_hu_static + num_hl_static + num_xu_static:
            offset = num_hu_static + num_hl_static
            lhs_z_bu[t - offset] = lhs_x[idx_xu[t - offset]] - rhs_z_bu[t - offset] + m_z_bu_inv[t - offset] * rhs_s_bu[t - offset]
            lhs_z_bu[t - offset] *= w_bu_delta_inv[t - offset]
        # lhs.z_bl[:] = -self._w_bl_delta_inv * (lhs.x[data.idx_xl] + rhs.z_bl - self._m_z_bl_inv * rhs.s_bl)
        elif t < num_hu_static + num_hl_static + num_xu_static + num_xl_static:
            offset = num_hu_static + num_hl_static + num_xu_static
            lhs_z_bl[t - offset] = lhs_x[idx_xl[t - offset]] + rhs_z_bl[t - offset] - m_z_bl_inv[t - offset] * rhs_s_bl[t - offset]
            lhs_z_bl[t - offset] *= -w_bl_delta_inv[t - offset]
        else:
            return

    return recover_duals_kernel


def create_recover_slacks_kernel(num_hu: int, num_hl: int, num_xu: int, num_xl: int):
    """Create kernel specialized for eliminating slacks. Performs the operation:
    
        updated_lhs_z_u = inv(Z_u) (r_s_u - S_u lhs_z_u)
        updated_lhs_s_l = inv(Z_l) (r_s_l - S_l lhs_z_l)
        updated_lhs_s_bu = inv(Z_bu) (r_s_bu - S_bu lhs_z_bu)
        updated_lhs_s_bl = inv(Z_bl) (r_s_bl - S_bl lhs_z_bl)
    """
    @wp.kernel
    def recover_slacks_kernel(
        # h_u
        rhs_s_u: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        lhs_z_u: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        result_s_u: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        result_z_u_inv: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        updated_lhs_s_u: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        # h_l
        rhs_s_l: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        lhs_z_l: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        result_s_l: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        result_z_l_inv: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        updated_lhs_s_l: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        # x_u
        rhs_s_bu: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        lhs_z_bu: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        result_s_bu: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        result_z_bu_inv: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        updated_lhs_s_bu: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        # x_l
        rhs_s_bl: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        lhs_z_bl: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        result_s_bl: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        result_z_bl_inv: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
        updated_lhs_s_bl: wp.array(dtype=wp.float64),  # pyright: ignore[reportInvalidTypeForm]
    ):
        t = wp.tid()
        num_hu_static = wp.static(num_hu)
        num_hl_static = wp.static(num_hl)
        num_xu_static = wp.static(num_xu)
        num_xl_static = wp.static(num_xl)

        if t < num_hu_static:
            # explicitly first do -result_s_u[t] * lhs_z_u[t], then add rhs_s_u[t], to trigger FMA, which is faster and more accurate
            updated_lhs_s_u[t] = result_z_u_inv[t] * (-result_s_u[t] * lhs_z_u[t] + rhs_s_u[t])
        elif t < num_hu_static + num_hl_static:
            offset = num_hu_static
            updated_lhs_s_l[t - offset] = result_z_l_inv[t - offset] * (-result_s_l[t - offset] * lhs_z_l[t - offset] + rhs_s_l[t - offset])
        elif t < num_hu_static + num_hl_static + num_xu_static:
            offset = num_hu_static + num_hl_static
            updated_lhs_s_bu[t - offset] = result_z_bu_inv[t - offset] * (-result_s_bu[t - offset] * lhs_z_bu[t - offset] + rhs_s_bu[t - offset])
        elif t < num_hu_static + num_hl_static + num_xu_static + num_xl_static:
            offset = num_hu_static + num_hl_static + num_xu_static
            updated_lhs_s_bl[t - offset] = result_z_bl_inv[t - offset] * (-result_s_bl[t - offset] * lhs_z_bl[t - offset] + rhs_s_bl[t - offset])
        else:
            return

    return recover_slacks_kernel
