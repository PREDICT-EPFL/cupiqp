import cupy as cp
import nvtx

from .solver import SolverBase
from .utils import cuda_graph_capture
from .solver_kernels import (
    create_prepare_predictor_step_kernel,
    create_prepare_corrector_step_kernel,
    create_update_vars_after_corrector_step_kernel,
    create_boundary_shift_kernel,
    create_init_guess_rhs_kernel,
    create_init_guess_project_to_central_path_kernel,
)
from .dense.dense_solver import DenseSolver
from .sparse.sparse_solver import SparseSolver
from .multistage.multistage_solver import MultistageSolver

from .dense.dense_preconditioner import DenseRuizEquilibration
from .sparse.sparse_preconditioner import SparseRuizEquilibration
from .multistage.multistage_preconditioner import MultistageRuizEquilibration


class LargeProblemSolverBase(SolverBase):
    """Cupy re-implementation of some methods of SolverBase.

    Replacing the warp-tile inner-loop kernels with cupy kernels.
    Use this class when ``max(n, p, m)`` is large enough that warp
    tile-kernel compile time dominates first-solve latency and cupy
    axis-1 reductions amortize their per-launch overhead. Numerically
    agrees with the regular backend solver to solver tolerance.
    """

    def _init_warp_kernels(self) -> None:
        # Subset: only the easy-warp kernels (no shape-specialized tile
        # kernels). The base class's _init_warp_kernels also builds the
        # tile-kernel families that takes long time to compile when the
        # problem size is large
        if self._data.num_ineq > 0:
            self._boundary_shift_kernel = create_boundary_shift_kernel(
                self._data.num_hl, self._data.num_hu,
                self._data.num_xl, self._data.num_xu,
                dtype=self._data.dtype)
            self._prepare_predictor_step_kernel = create_prepare_predictor_step_kernel(dtype=self._data.dtype)
            self._prepare_corrector_step_kernel = create_prepare_corrector_step_kernel(dtype=self._data.dtype)
            self._update_vars_after_corrector_step_kernel = create_update_vars_after_corrector_step_kernel(
                n=self._data.n, p=self._data.p, num_ineq=self._data.num_ineq,
                dtype=self._data.dtype)
            self._init_guess_project_to_central_path_kernel = create_init_guess_project_to_central_path_kernel(dtype=self._data.dtype)

        self._initial_guess_rhs_kernel = create_init_guess_rhs_kernel(
            self._data.n, self._data.p,
            int(self._data.num_hl), int(self._data.num_hu),
            int(self._data.num_xl), int(self._data.num_xu),
            dtype=self._data.dtype,
        )

    @nvtx.annotate("LargeProblemSolverBase::_run_full_newton_step")
    def _run_full_newton_step(self):
        self._kkt_system.solve(self._data, self._preconditioner, self.settings, self._res, self._step)
        active = self._unsolved_mask
        self._result.info.primal_step[:] = cp.where(active, 1.0, self._result.info.primal_step)
        self._result.info.dual_step[:] = cp.where(active, 1.0, self._result.info.dual_step)
        self._result.x += active[:, None] * self._step.x
        self._result.y += active[:, None] * self._step.y

    @nvtx.annotate("LargeProblemSolverBase::_calculate_step")
    @cuda_graph_capture(enable=lambda self: self.settings.enable_cuda_graph)
    def _calculate_step(self) -> None:
        # alpha_s: step length for slacks
        self._work_s[:] = cp.where(self._step.s_all < 0, -self._result.s_all / self._step.s_all, 1.)
        self._result.info.primal_step[:] = cp.min(self._work_s, axis=1)  # alpha_s
        self._result.info.primal_step *= self.settings.tau

        # alpha_z: step length for duals
        self._work_z[:] = cp.where(self._step.z_all < 0, -self._result.z_all / self._step.z_all, 1.)
        self._result.info.dual_step[:] = cp.min(self._work_z, axis=1)  # alpha_z
        self._result.info.dual_step *= self.settings.tau

    @nvtx.annotate("LargeProblemSolverBase::_calculate_mu")
    @cuda_graph_capture(enable=lambda self: self.settings.enable_cuda_graph)
    def _calculate_mu(self) -> None:
        # mu = sum_i(finite_mask * s * z) / num_finite_bounds, matching
        # SolverBase's warp kernel: infinite-bound slots are masked out and the
        # divisor is the per-problem finite-bound count (not num_ineq). When a
        # problem has no finite bounds the masked sum is 0, so 0 / max(0, 1) = 0
        # reproduces the kernel's denom>0 guard.
        cp.multiply(self._result.s_all, self._result.z_all, out=self._work_s)
        self._work_s *= self._data.finite_mask_all
        cp.sum(self._work_s, axis=1, out=self._result.info.mu)
        cp.divide(self._result.info.mu,
                  cp.maximum(self._data.num_finite_bounds, 1.),
                  out=self._result.info.mu)

    @nvtx.annotate("LargeProblemSolverBase::_calculate_sigma")
    @cuda_graph_capture(enable=lambda self: self.settings.enable_cuda_graph)
    def _calculate_sigma(self) -> None:
        # s_trial = s + alpha_s * ds,  z_trial = z + alpha_z * dz
        cp.multiply(self._result.info.primal_step[:, None], self._step.s_all, out=self._work_s)
        self._work_s += self._result.s_all
        cp.multiply(self._result.info.dual_step[:, None], self._step.z_all, out=self._work_z)
        self._work_z += self._result.z_all
        cp.multiply(self._work_s, self._work_z, out=self._work_s)  # s_trial * z_trial
        self._work_s *= self._data.finite_mask_all                 # mask infinite-bound slots
        cp.sum(self._work_s, axis=1, out=self._result.info.sigma)  # acc

        # sigma = clip(acc / (mu * num_finite_bounds), 0, 1)**3, or 0 when the
        # denominator is non-positive -- matching SolverBase's warp kernel.
        denom = self._result.info.mu * self._data.num_finite_bounds
        positive = denom > 0.
        cp.divide(self._result.info.sigma, cp.where(positive, denom, 1.),
                  out=self._result.info.sigma)
        cp.clip(self._result.info.sigma, 0., 1., out=self._result.info.sigma)
        cp.power(self._result.info.sigma, 3., out=self._result.info.sigma)
        self._result.info.sigma[:] = cp.where(positive, self._result.info.sigma, 0.)

    @nvtx.annotate("LargeProblemSolverBase::_update_residuals_nr")
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
            res_nr.z_l  =   G*x - s_l - h_l
            res_nr.z_u  = -G*x - s_u + h_u
            res_nr.z_bl =   x_b_scaling*x - s_bl - x_l
            res_nr.z_bu = -(x_b_scaling*x + s_bu - x_u)

        Convergence norms (unscaled, infinity norm per batch):

            primal_res     = max over the 5 dual segments of
                                 ||u_p_seg .* res_nr_seg||_inf
                             where u_p_seg is the per-segment primal
                             unscale factor:
                                 [y]:    delta_inv[:, n : n+p]
                                 [z_l]:  delta_inv[:, n+p : n+p+m]
                                 [z_u]:  delta_inv[:, n+p : n+p+m]
                                 [z_bl]: delta_b_inv[:, :n]
                                 [z_bu]: delta_b_inv[:, :n]

            dual_res       = cost_scaling_inv * ||delta_inv[:, :n] .* res_nr.x||_inf

        Relative-norm denominators (also unscaled, max over magnitudes
        that go into the corresponding residual):

            primal_rel     = max( ||u_p_y .* A*x||,
                                  ||u_p_zl .* G*x||,
                                  ||u_p_zu .* G*x||,
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

        This is the cupy-axis-reduction implementation; ``SolverBase``'s
        version in cupiqp/solver.py uses fused warp tile kernels and is
        numerically equivalent.
        """
        pc = self._preconditioner
        n, p = self._data.n, self._data.p
        # cuSPARSE/cuBLAS operations
        self._kkt_system.eval_P_x(self._data, -1., self._result.x, self._res_nr.x)
        # ||unscale_dual_res(P*x)||_inf -> _work_dual_res_norm
        cp.absolute(self._res_nr.x, out=self._work_primals)
        self._work_primals *= pc.delta_inv[:, :n]
        self._work_primals *= pc.cost_scaling_inv[:, None]
        cp.max(self._work_primals, axis=1, out=self._work_dual_res_norm)

        if self._data.p > 0:
            self._kkt_system.eval_A_xn(self._data, -1., self._result.x, self._res_nr.y)
            self._kkt_system.eval_AT_xt(self._data, 1., self._result.y, self._res.x)
        else:
            self._res.x.fill(0.)
        if self._data.p > 0:
            cp.absolute(self._res_nr.y, out=self._work_duals[:, :self._data.p])
            self._work_duals[:, :self._data.p] *= pc.delta_inv[:, n:n + p]
            cp.max(self._work_duals[:, :self._data.p], axis=1, out=self._work_primal_rel_norm)
        else:
            self._work_primal_rel_norm.fill(0.)

        # zu_minus_zl is always full (B, m); an omitted inequality side has
        # (B, 0) duals and contributes nothing, so guard each on its presence.
        self._work_z_1.fill(0.)
        if self._data.num_hu > 0:
            self._work_z_1[:, :self._data.m] += self._result.z_u
        if self._data.num_hl > 0:
            self._work_z_1[:, :self._data.m] -= self._result.z_l

        G_x = self._work_z_2
        GT_zu_minus_zl = self._step.x
        if self._data.m > 0:
            self._kkt_system.eval_G_xn(self._data, 1., self._result.x, G_x)
            self._kkt_system.eval_GT_xt(self._data, 1., self._work_z_1, GT_zu_minus_zl)
        else:
            G_x.fill(0.)
            GT_zu_minus_zl.fill(0)

        # Mask infinite bounds to 0 before they enter any cupy arithmetic.
        # data.h_l/h_u/x_l/x_u carry +/-inf (or a >PIQP_INF sentinel) on
        # inactive rows, while their dual/slack stays at the central-path
        # value, so an unmasked product would be inf*0 = NaN (and the
        # residual would subtract an infinity). finite_mask_* is 1.0 on
        # finite rows and 0.0 on infinite ones. The bounds are loop-invariant,
        # but kept here to mirror SolverBase's per-iteration warp kernel.
        h_l_masked = cp.where(self._data.finite_mask_hl > 0.5, self._data.h_l, 0.0)
        h_u_masked = cp.where(self._data.finite_mask_hu > 0.5, self._data.h_u, 0.0)
        x_l_masked = cp.where(self._data.finite_mask_xl > 0.5, self._data.x_l, 0.0)
        x_u_masked = cp.where(self._data.finite_mask_xu > 0.5, self._data.x_u, 0.0)

        # ------------ update primal / dual objectives and duality gap ------------
        cp.sum(self._res_nr.x * self._result.x, axis=1, out=self._work_reduce[:, 0])
        self._work_reduce[:, 0] *= -0.5  # 0.5 * x^T P x
        cp.sum(self._data.c * self._result.x, axis=1, out=self._work_reduce[:, 1])

        self._work_reduce[:, 2] = self._work_reduce[:, 0]
        cp.sum(self._data.b * self._result.y, axis=1, out=self._work_reduce[:, 3])
        cp.sum(h_l_masked * self._result.z_l, axis=1, out=self._work_reduce[:, 4])
        self._work_reduce[:, 4] *= -1.
        cp.sum(h_u_masked * self._result.z_u, axis=1, out=self._work_reduce[:, 5])
        cp.sum(x_l_masked * self._result.z_bl, axis=1, out=self._work_reduce[:, 6])
        self._work_reduce[:, 6] *= -1.
        cp.sum(x_u_masked * self._result.z_bu, axis=1, out=self._work_reduce[:, 7])

        cp.sum(self._work_reduce[:, 0:2], axis=1, out=self._result.info.primal_obj)
        cp.sum(self._work_reduce[:, 2:8], axis=1, out=self._result.info.dual_obj)
        self._result.info.dual_obj *= -1.

        cp.subtract(self._result.info.primal_obj, self._result.info.dual_obj, out=self._result.info.duality_gap)

        # Unscale objectives and duality gap from scaled to original space
        self._result.info.primal_obj *= pc.cost_scaling_inv
        self._result.info.dual_obj *= pc.cost_scaling_inv
        self._result.info.duality_gap *= pc.cost_scaling_inv

        self._work_reduce *= pc.cost_scaling_inv[:, None]
        cp.abs(self._work_reduce, out=self._work_reduce)
        cp.max(self._work_reduce[:, 0:8], axis=1, out=self._result.info.duality_gap_rel)
        cp.abs(self._result.info.duality_gap, out=self._result.info.duality_gap)
        cp.maximum(self._result.info.duality_gap_rel, 1., out=self._result.info.duality_gap_rel)
        cp.divide(self._result.info.duality_gap, self._result.info.duality_gap_rel, out=self._result.info.duality_gap_rel)

        # ------------ update non-regulerized residuals ------------
        self._res_nr.x -= self._data.c
        self._res_nr.x -= self._res.x  # self._res.x holds A^T*y
        self._res_nr.x -= GT_zu_minus_zl
        # Box blocks are optional: z_bl/z_bu are (B, 0) when that side was
        # omitted at setup(), so guard each contribution on its presence.
        if self._data.num_xl > 0:
            self._res_nr.x[:, :self._data.n] += self._preconditioner.x_b_scaling[:, :self._data.n] * self._result.z_bl
        if self._data.num_xu > 0:
            self._res_nr.x[:, :self._data.n] -= self._preconditioner.x_b_scaling[:, :self._data.n] * self._result.z_bu

        # res_nr.y = -(A*x - b)
        self._res_nr.y += self._data.b

        # res_nr.z_l = (G*x - s_l - h_l) on finite rows; 0 on inactive rows.
        # The trailing mask multiply zeros the inactive rows so the shared
        # _primal_res_nr() inf-norm (which does not re-mask) is correct.
        if self._data.num_hl > 0:
            self._res_nr.z_l[:] = G_x[:, :self._data.m]
            cp.subtract(self._res_nr.z_l, self._result.s_l, out=self._res_nr.z_l)
            cp.subtract(self._res_nr.z_l, h_l_masked, out=self._res_nr.z_l)
            self._res_nr.z_l *= self._data.finite_mask_hl

        # res_nr.z_u = (-G*x - s_u + h_u) on finite rows; 0 on inactive rows.
        if self._data.num_hu > 0:
            self._res_nr.z_u[:] = -G_x[:, :self._data.m]
            cp.subtract(self._res_nr.z_u, self._result.s_u, out=self._res_nr.z_u)
            cp.add(self._res_nr.z_u, h_u_masked, out=self._res_nr.z_u)
            self._res_nr.z_u *= self._data.finite_mask_hu

        # res_nr.z_bl = (x_b_scaling*x - s_bl - x_l) on finite rows; 0 otherwise.
        # Skipped entirely when the lower box block is absent (z_bl is (B, 0)).
        if self._data.num_xl > 0:
            self._res_nr.z_bl[:] = self._result.x[:, :self._data.n]
            self._res_nr.z_bl *= self._preconditioner.x_b_scaling[:, :self._data.n]
            cp.subtract(self._res_nr.z_bl, self._result.s_bl, out=self._res_nr.z_bl)
            cp.subtract(self._res_nr.z_bl, x_l_masked, out=self._res_nr.z_bl)
            self._res_nr.z_bl *= self._data.finite_mask_xl

        # res_nr.z_bu = -(x_b_scaling*x + s_bu - x_u) on finite rows; 0 otherwise.
        # Skipped entirely when the upper box block is absent (z_bu is (B, 0)).
        if self._data.num_xu > 0:
            self._res_nr.z_bu[:] = self._result.x[:, :self._data.n]
            self._res_nr.z_bu *= self._preconditioner.x_b_scaling[:, :self._data.n]
            cp.add(self._res_nr.z_bu, self._result.s_bu, out=self._res_nr.z_bu)
            cp.subtract(self._res_nr.z_bu, x_u_masked, out=self._res_nr.z_bu)
            cp.negative(self._res_nr.z_bu, out=self._res_nr.z_bu)
            self._res_nr.z_bu *= self._data.finite_mask_xu

        # ------------ update primal and dual residuals ------------
        self._result.info.prev_primal_res[:] = self._result.info.primal_res
        self._result.info.prev_dual_res[:] = self._result.info.dual_res

        self._result.info.primal_res[:] = self._primal_res_nr()

        # primal_rel_norm: update running max
        # G*x, s_*: zero the inactive rows (finite_mask_*) so they do not
        # inflate the relative-norm denominator -- inactive slacks sit at the
        # central-path value, not 0, so this masking is required to match
        # SolverBase's warp-kernel relative norm.
        if self._data.num_hu > 0:
            self._work_z_1[:, :self._data.num_hu] = cp.abs(G_x[:, :self._data.m])
            self._work_z_1[:, :self._data.num_hu] *= pc.delta_inv[:, n + p:n + p + self._data.m]
            self._work_z_1[:, :self._data.num_hu] *= self._data.finite_mask_hu
            cp.max(self._work_z_1[:, :self._data.num_hu], axis=1, out=self._work_norm_temp)
            cp.maximum(self._work_primal_rel_norm, self._work_norm_temp, out=self._work_primal_rel_norm)

        if self._data.num_hl > 0:
            self._work_z_1[:, :self._data.num_hl] = cp.abs(G_x[:, :self._data.m])
            self._work_z_1[:, :self._data.num_hl] *= pc.delta_inv[:, n + p:n + p + self._data.m]
            self._work_z_1[:, :self._data.num_hl] *= self._data.finite_mask_hl
            cp.max(self._work_z_1[:, :self._data.num_hl], axis=1, out=self._work_norm_temp)
            cp.maximum(self._work_primal_rel_norm, self._work_norm_temp, out=self._work_primal_rel_norm)

        if self._data.num_hu > 0:
            cp.absolute(self._result.s_u, out=self._work_z_1[:, :self._data.num_hu])
            self._work_z_1[:, :self._data.num_hu] *= pc.delta_inv[:, n + p:n + p + self._data.m]
            self._work_z_1[:, :self._data.num_hu] *= self._data.finite_mask_hu
            cp.max(self._work_z_1[:, :self._data.num_hu], axis=1, out=self._work_norm_temp)
            cp.maximum(self._work_primal_rel_norm, self._work_norm_temp, out=self._work_primal_rel_norm)

        if self._data.num_hl > 0:
            cp.absolute(self._result.s_l, out=self._work_z_1[:, :self._data.num_hl])
            self._work_z_1[:, :self._data.num_hl] *= pc.delta_inv[:, n + p:n + p + self._data.m]
            self._work_z_1[:, :self._data.num_hl] *= self._data.finite_mask_hl
            cp.max(self._work_z_1[:, :self._data.num_hl], axis=1, out=self._work_norm_temp)
            cp.maximum(self._work_primal_rel_norm, self._work_norm_temp, out=self._work_primal_rel_norm)

        if self._data.num_xu > 0:
            cp.absolute(self._result.s_bu, out=self._work_z[:, :self._data.num_xu])
            self._work_z[:, :self._data.num_xu] *= pc.delta_b_inv[:, :self._data.n]
            self._work_z[:, :self._data.num_xu] *= self._data.finite_mask_xu
            cp.max(self._work_z[:, :self._data.num_xu], axis=1, out=self._work_norm_temp)
            cp.maximum(self._work_primal_rel_norm, self._work_norm_temp, out=self._work_primal_rel_norm)

        if self._data.num_xl > 0:
            cp.absolute(self._result.s_bl, out=self._work_z[:, :self._data.num_xl])
            self._work_z[:, :self._data.num_xl] *= pc.delta_b_inv[:, :self._data.n]
            self._work_z[:, :self._data.num_xl] *= self._data.finite_mask_xl
            cp.max(self._work_z[:, :self._data.num_xl], axis=1, out=self._work_norm_temp)
            cp.maximum(self._work_primal_rel_norm, self._work_norm_temp, out=self._work_primal_rel_norm)

        cp.maximum(self._work_primal_rel_norm, self._constraints_rhs_inf_norm_unscaled, out=self._work_primal_rel_norm)
        cp.maximum(self._work_primal_rel_norm, 1., out=self._work_primal_rel_norm)
        cp.divide(self._result.info.primal_res, self._work_primal_rel_norm, out=self._result.info.primal_res_rel)

        # dual_res_norm: update running max
        self._result.info.dual_res[:] = self._dual_res_nr()

        # ||unscale_dual_res(c)||_inf
        cp.absolute(self._data.c, out=self._work_primals)
        self._work_primals *= pc.delta_inv[:, :n]
        self._work_primals *= pc.cost_scaling_inv[:, None]
        cp.max(self._work_primals, axis=1, out=self._work_norm_temp)
        cp.maximum(self._work_dual_res_norm, self._work_norm_temp, out=self._work_dual_res_norm)

        # ||unscale_dual_res(A^T*y + G^T*(z_u - z_l) + x_b_scaling*(z_bu - z_bl))||_inf
        self._res.x += GT_zu_minus_zl
        # Box blocks are optional (z_bl/z_bu are (B, 0) when omitted); guard each.
        if self._data.num_xl > 0:
            self._res.x[:, :self._data.n] -= self._preconditioner.x_b_scaling[:, :self._data.n] * self._result.z_bl
        if self._data.num_xu > 0:
            self._res.x[:, :self._data.n] += self._preconditioner.x_b_scaling[:, :self._data.n] * self._result.z_bu
        cp.absolute(self._res.x, out=self._work_primals)
        self._work_primals *= pc.delta_inv[:, :n]
        self._work_primals *= pc.cost_scaling_inv[:, None]
        cp.max(self._work_primals, axis=1, out=self._work_norm_temp)
        cp.maximum(self._work_dual_res_norm, self._work_norm_temp, out=self._work_dual_res_norm)

        cp.maximum(self._work_dual_res_norm, 1., out=self._work_dual_res_norm)
        cp.divide(self._result.info.dual_res, self._work_dual_res_norm, out=self._result.info.dual_res_rel)

    @nvtx.annotate("LargeProblemSolverBase::_update_residuals_r")
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
        cp.subtract(self._result.x, self._prox_vars.x, out=self._res.x)
        self._res.x *= self._result.info.rho[:, None]
        cp.subtract(self._res_nr.x, self._res.x, out=self._res.x)
        cp.subtract(self._prox_vars.duals_all, self._result.duals_all, out=self._res.duals_all)
        self._res.duals_all *= self._result.info.delta[:, None]
        cp.subtract(self._res_nr.duals_all, self._res.duals_all, out=self._res.duals_all)

        self._result.info.primal_res_reg[:] = self._primal_res_r()
        cp.divide(self._result.info.primal_res, self._result.info.primal_res_rel, out=self._result.info.primal_res_reg_rel)
        self._result.info.primal_res_reg_rel[:] = cp.where(
            self._result.info.primal_res_rel > 0,
            self._result.info.primal_res_reg_rel,
            cp.asarray(1.0, dtype=self._result.info.primal_res_reg_rel.dtype),
        )
        cp.divide(self._result.info.primal_res_reg, self._result.info.primal_res_reg_rel, out=self._result.info.primal_res_reg_rel)

        self._result.info.dual_res_reg[:] = self._dual_res_r()
        cp.divide(self._result.info.dual_res, self._result.info.dual_res_rel, out=self._result.info.dual_res_reg_rel)
        self._result.info.dual_res_reg_rel[:] = cp.where(
            self._result.info.dual_res_rel > 0,
            self._result.info.dual_res_reg_rel,
            cp.asarray(1.0, dtype=self._result.info.dual_res_reg_rel.dtype),
        )
        cp.divide(self._result.info.dual_res_reg, self._result.info.dual_res_reg_rel, out=self._result.info.dual_res_reg_rel)

        self._result.info.primal_prox_inf[:] = self._primal_prox_inf()
        self._result.info.primal_prox_inf *= self._result.info.delta
        self._result.info.dual_prox_inf[:] = self._dual_prox_inf()
        self._result.info.dual_prox_inf *= self._result.info.rho

    @nvtx.annotate("LargeProblemSolverBase::_update_rho_delta_with_ineq")
    def _update_rho_delta_with_ineq(self) -> None:
        info = self._result.info
        settings = self.settings
        active = self._unsolved_mask

        # --- Rho update ---
        dual_improved = active & (
            (info.dual_res < 0.95 * info.prev_dual_res) |
            (info.dual_res < settings.eps_abs) | (info.dual_res_rel < settings.eps_rel) |
            ((info.rho == settings.reg_finetune_lower_limit) & (info.dual_prox_inf < settings.infeasibility_threshold))
        )
        rho_fast = cp.maximum(info.reg_limit, 0.1 * info.rho)
        rho_slow = cp.maximum(info.reg_limit, 0.5 * info.rho)
        rho_slow_decay_ok = active & (~dual_improved) & ((self._iter < 5) | (info.dual_prox_inf < settings.infeasibility_threshold))
        info.rho[:] = cp.where(dual_improved, rho_fast, cp.where(rho_slow_decay_ok, rho_slow, info.rho))
        self._prox_vars.x[:] = cp.where(dual_improved[:, None], self._result.x, self._prox_vars.x)
        info.no_primal_update[:] = cp.where(
            active, cp.where(dual_improved, 0, info.no_primal_update + 1),
            info.no_primal_update,
        )

        # --- Delta update ---
        primal_improved = active & (
            (info.primal_res < 0.95 * info.prev_primal_res) |
            (info.primal_res < settings.eps_abs) | (info.primal_res_rel < settings.eps_rel) |
            ((info.delta == settings.reg_finetune_lower_limit) & (info.primal_prox_inf < settings.infeasibility_threshold))
        )
        delta_fast = cp.maximum(info.reg_limit, 0.1 * info.delta)
        delta_slow = cp.maximum(info.reg_limit, 0.5 * info.delta)
        delta_slow_decay_ok = active & (~primal_improved) & ((self._iter < 5) | (info.primal_prox_inf < settings.infeasibility_threshold))
        info.delta[:] = cp.where(primal_improved, delta_fast, cp.where(delta_slow_decay_ok, delta_slow, info.delta))
        self._prox_vars.duals_all[:] = cp.where(primal_improved[:, None], self._result.duals_all, self._prox_vars.duals_all)
        info.no_dual_update[:] = cp.where(
            active, cp.where(primal_improved, 0, info.no_dual_update + 1),
            info.no_dual_update,
        )

    @nvtx.annotate("LargeProblemSolverBase::_update_rho_delta_without_ineq")
    def _update_rho_delta_without_ineq(self) -> None:
        info = self._result.info
        settings = self.settings
        active = self._unsolved_mask

        # --- Rho update ---
        dual_improved = active & (
            (info.dual_res < 0.95 * info.prev_dual_res) |
            (info.dual_res < settings.eps_abs) |
            (info.dual_res_rel < settings.eps_rel)
        )
        rho_fast = cp.maximum(info.reg_limit, 0.1 * info.rho)
        rho_slow = cp.maximum(info.reg_limit, 0.5 * info.rho)
        rho_slow_decay_ok = active & (~dual_improved) & ((self._iter < 5) | (info.dual_prox_inf < settings.infeasibility_threshold))
        info.rho[:] = cp.where(dual_improved, rho_fast, cp.where(rho_slow_decay_ok, rho_slow, info.rho))
        self._prox_vars.x[:] = cp.where(dual_improved[:, None], self._result.x, self._prox_vars.x)
        info.no_primal_update[:] = cp.where(
            active, cp.where(dual_improved, 0, info.no_primal_update + 1),
            info.no_primal_update,
        )

        # --- Delta update ---
        primal_improved = active & (
            (info.primal_res < 0.95 * info.prev_primal_res) |
            (info.primal_res < settings.eps_abs) |
            (info.primal_res_rel < settings.eps_rel)
        )
        delta_fast = cp.maximum(info.reg_limit, 0.1 * info.delta)
        delta_slow = cp.maximum(info.reg_limit, 0.5 * info.delta)
        delta_slow_decay_ok = active & (~primal_improved) & ((self._iter < 5) | (info.primal_prox_inf < settings.infeasibility_threshold))
        info.delta[:] = cp.where(primal_improved, delta_fast, cp.where(delta_slow_decay_ok, delta_slow, info.delta))
        self._prox_vars.y[:] = cp.where(primal_improved[:, None], self._result.y, self._prox_vars.y)
        info.no_dual_update[:] = cp.where(
            active, cp.where(primal_improved, 0, info.no_dual_update + 1),
            info.no_dual_update,
        )


class DenseLargeProblemSolver(LargeProblemSolverBase, DenseSolver):
    """Dense backend with cupy-axis-reduction inner-loop kernels."""

    def _init_preconditioner(self):
        # Override DenseSolver: skip the warp tile-kernel compile cliff.
        return DenseRuizEquilibration(
            self._data.batch_size, self._data.n, self._data.p, self._data.m,
            has_h_l=self._data.has_h_l, has_h_u=self._data.has_h_u,
            has_x_l=self._data.has_x_l, has_x_u=self._data.has_x_u,
            active_x_bound=self._data.active_x_bound,
            use_warp_tile_kernels=False,
        )


class SparseLargeProblemSolver(LargeProblemSolverBase, SparseSolver):
    """Sparse backend with cupy-axis-reduction inner-loop kernels."""

    def _init_preconditioner(self):
        # Override SparseSolver: skip the warp tile-kernel compile cliff.
        return SparseRuizEquilibration(
            self._data.batch_size, self._data.n, self._data.p, self._data.m,
            has_h_l=self._data.has_h_l, has_h_u=self._data.has_h_u,
            has_x_l=self._data.has_x_l, has_x_u=self._data.has_x_u,
            active_x_bound=self._data.active_x_bound,
            use_warp_tile_kernels=False,
        )


class MultistageLargeProblemSolver(LargeProblemSolverBase, MultistageSolver):
    """Multistage backend with cupy-axis-reduction inner-loop kernels."""

    def _init_preconditioner(self):
        # Override MultistageSolver: skip the warp tile-kernel compile
        # cliff. Multistage needs the block layout via ``data=``.
        return MultistageRuizEquilibration(
            self._data.batch_size, self._data.n, self._data.p, self._data.m,
            has_h_l=self._data.has_h_l, has_h_u=self._data.has_h_u,
            has_x_l=self._data.has_x_l, has_x_u=self._data.has_x_u,
            active_x_bound=self._data.active_x_bound,
            data=self._data,
            use_warp_tile_kernels=False,
        )

