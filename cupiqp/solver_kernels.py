"""Warp kernels of the interior-point iteration shared by every backend.

The block-reduction kernels (step lengths, mu, sigma, residual norms) read
their widths from the array shapes: one CUDA block per batch entry, each
thread striding over the row, so they are compiled once per dtype for any
problem size. The element-wise kernels keep compile-time block widths.
"""
import functools

import warp as wp
from .utils import to_warp_dtype
from .typedef import (
    STATUS_SOLVED, STATUS_PRIMAL_INFEASIBLE, STATUS_DUAL_INFEASIBLE, STATUS_UNSOLVED,
    STATUS_NUMERICAL_ISSUES,
)
from .settings import SettingsFloatIdx, SettingsIntIdx


# Threads per batch entry in the block-reduction kernels below. Each thread
# strides over the row it reduces, so the kernels work for any problem width
# and are compiled once per dtype rather than once per shape.
REDUCTION_BLOCK_DIM = 256


@functools.lru_cache(maxsize=None)
def create_calculate_step_kernel(dtype=wp.float64):
    """Fused block-reduction kernel for step lengths (primal and dual).

    For each batch ``b``, computes over finite-bound entries only::

        alpha_s[b] = tau * min_i( ds[b,i] < 0 ? -s[b,i]/ds[b,i] : 1.0 )
        alpha_z[b] = tau * min_i( dz[b,i] < 0 ? -z[b,i]/dz[b,i] : 1.0 )

    Inactive entries have mask value 0.0 and contribute candidate step 1.0,
    so they never restrict the line search.

    ``tau`` is read from the device configuration. Dispatch with ``wp.launch_tiled(..., dim=[B], block_dim=REDUCTION_BLOCK_DIM)``:
    one CUDA block per batch entry; every thread reduces a strided subset of
    the row, then the block combines the partial minima.
    """
    dtype = to_warp_dtype(dtype)

    @wp.func
    def step_candidate(a: dtype, b: dtype) -> dtype:    # type: ignore
        return wp.where(a < dtype(0.0), -b / a, dtype(1.0))

    @wp.func
    def step_candidate_masked(a: dtype, b: dtype, mask: dtype) -> dtype:    # type: ignore
        return wp.where(mask > dtype(0.5), step_candidate(a, b), dtype(1.0))

    @wp.kernel
    def calculate_step_kernel(
        s_all: wp.array2d(dtype=dtype),        # (B, num_ineq)  # type: ignore
        z_all: wp.array2d(dtype=dtype),        # (B, num_ineq)  # type: ignore
        finite_mask_all: wp.array2d(dtype=dtype),   # (B, num_ineq)  # type: ignore
        step_s_all: wp.array2d(dtype=dtype),   # (B, num_ineq)  # type: ignore
        step_z_all: wp.array2d(dtype=dtype),   # (B, num_ineq)  # type: ignore
        settings: wp.array(dtype=dtype),            # DeviceSettings.floats  # type: ignore
        alpha_s: wp.array(dtype=dtype),        # (B,) output    # type: ignore
        alpha_z: wp.array(dtype=dtype),        # (B,) output    # type: ignore
    ):
        b, i = wp.tid()
        num_ineq = s_all.shape[1]
        min_s = dtype(wp.inf)
        min_z = dtype(wp.inf)
        for k in range(i, num_ineq, wp.block_dim()):
            mask = finite_mask_all[b, k]
            min_s = wp.min(min_s, step_candidate_masked(step_s_all[b, k], s_all[b, k], mask))
            min_z = wp.min(min_z, step_candidate_masked(step_z_all[b, k], z_all[b, k], mask))
        red_s = wp.tile_min(wp.tile(min_s))
        red_z = wp.tile_min(wp.tile(min_z))
        if i == 0:
            tau = settings[SettingsFloatIdx.tau]
            alpha_s[b] = red_s[0] * tau
            alpha_z[b] = red_z[0] * tau

    return calculate_step_kernel


@functools.lru_cache(maxsize=None)
def create_calculate_mu_kernel(dtype=wp.float64):
    """Fused block-reduction kernel for the duality measure mu.

    For each batch ``b``, computes:

        mu[b] = sum_i( mask[b, i] * s[b, i] * z[b, i] ) / num_finite_bounds[b]

    (``0`` when the problem has no finite bound). Dispatch with
    ``wp.launch_tiled(..., dim=[B], block_dim=REDUCTION_BLOCK_DIM)``.
    """
    dtype = to_warp_dtype(dtype)

    @wp.kernel
    def calculate_mu_kernel(
        s_all: wp.array2d(dtype=dtype),  # (B, num_ineq)  # type: ignore
        z_all: wp.array2d(dtype=dtype),  # (B, num_ineq)  # type: ignore
        finite_mask_all: wp.array2d(dtype=dtype),  # (B, num_ineq)  # type: ignore
        num_finite_bounds: wp.array(dtype=dtype),  # (B,)  # type: ignore
        mu:    wp.array(dtype=dtype),    # (B,) output    # type: ignore
    ):
        b, i = wp.tid()
        num_ineq = s_all.shape[1]
        acc = dtype(0.0)
        for k in range(i, num_ineq, wp.block_dim()):
            acc += s_all[b, k] * z_all[b, k] * finite_mask_all[b, k]
        total = wp.tile_sum(wp.tile(acc))
        if i == 0:
            denom = num_finite_bounds[b]
            if denom > dtype(0.0):
                mu[b] = total[0] / denom
            else:
                mu[b] = dtype(0.0)

    return calculate_mu_kernel


@functools.lru_cache(maxsize=None)
def create_calculate_sigma_kernel(dtype=wp.float64):
    """Fused block-reduction kernel for the centering parameter sigma.

    For each batch ``b``, computes:

        s_trial[i] = s[b, i] + alpha_s[b] * ds[b, i]
        z_trial[i] = z[b, i] + alpha_z[b] * dz[b, i]
        acc        = sum_i( mask[i] * s_trial[i] * z_trial[i] )
        sigma[b]   = clip( acc / (mu[b] * num_finite_bounds[b]), 0, 1 ) ** 3

    or ``0`` when the denominator is not positive. Dispatch with
    ``wp.launch_tiled(..., dim=[B], block_dim=REDUCTION_BLOCK_DIM)``.
    """
    dtype = to_warp_dtype(dtype)

    @wp.kernel
    def calculate_sigma_kernel(
        s_all: wp.array2d(dtype=dtype),        # (B, num_ineq)  # type: ignore
        z_all: wp.array2d(dtype=dtype),        # (B, num_ineq)  # type: ignore
        step_s_all: wp.array2d(dtype=dtype),   # (B, num_ineq)  # type: ignore
        step_z_all: wp.array2d(dtype=dtype),   # (B, num_ineq)  # type: ignore
        finite_mask_all: wp.array2d(dtype=dtype),   # (B, num_ineq)  # type: ignore
        num_finite_bounds: wp.array(dtype=dtype),  # (B,)       # type: ignore
        primal_step: wp.array(dtype=dtype),    # (B,) alpha_s   # type: ignore
        dual_step: wp.array(dtype=dtype),      # (B,) alpha_z   # type: ignore
        mu: wp.array(dtype=dtype),             # (B,)           # type: ignore
        sigma: wp.array(dtype=dtype),          # (B,) output    # type: ignore
    ):
        b, i = wp.tid()
        num_ineq = s_all.shape[1]
        alpha_s = primal_step[b]
        alpha_z = dual_step[b]
        acc = dtype(0.0)
        for k in range(i, num_ineq, wp.block_dim()):
            acc += (s_all[b, k] + alpha_s * step_s_all[b, k]) * (z_all[b, k] + alpha_z * step_z_all[b, k]) * finite_mask_all[b, k]
        total = wp.tile_sum(wp.tile(acc))
        if i == 0:
            denominator = mu[b] * num_finite_bounds[b]
            if denominator > dtype(0.0):
                val = total[0] / denominator
                val = wp.clamp(val, dtype(0.0), dtype(1.0))
                sigma[b] = val * val * val
            else:
                sigma[b] = dtype(0.0)

    return calculate_sigma_kernel


@functools.lru_cache(maxsize=None)
def create_init_guess_center_kernel(dtype=wp.float64):
    """Shift the initial slacks and duals into the positive orthant and put
    them on the central path (section IV.A of Schwan et al. 2023).

    For each batch ``b``::

        delta_s = -min_i s[b, i];   delta_z = -min_i z[b, i]     (all entries)
        s += delta_s;               z += delta_z
        mu = max( sum_i(mask * s * z) / num_finite_bounds, 1e-10 )
        on finite-bound entries:   c = z - delta_z
                                   z = (c + sqrt(c^2 + 4 mu)) / 2;  s = z - c
        on infinite-bound entries: z = 0;  s = 0

    ``mu`` receives the clipped duality measure used for the projection;
    the caller recomputes mu afterwards. Dispatch with
    ``wp.launch_tiled(..., dim=[B], block_dim=REDUCTION_BLOCK_DIM)``.
    """
    dtype = to_warp_dtype(dtype)

    @wp.kernel
    def init_guess_center_kernel(
        finite_mask_all: wp.array2d(dtype=dtype),  # type: ignore  (B, num_ineq)
        num_finite_bounds: wp.array(dtype=dtype),  # type: ignore  (B,)
        s_all:   wp.array2d(dtype=dtype),   # type: ignore  (B, num_ineq) in-out
        z_all:   wp.array2d(dtype=dtype),   # type: ignore  (B, num_ineq) in-out
        mu:      wp.array(dtype=dtype),     # type: ignore  (B,) out
    ):
        b, i = wp.tid()
        num_ineq = s_all.shape[1]
        bd = wp.block_dim()

        min_s = dtype(wp.inf)
        min_z = dtype(wp.inf)
        for k in range(i, num_ineq, bd):
            min_s = wp.min(min_s, s_all[b, k])
            min_z = wp.min(min_z, z_all[b, k])
        red_s = wp.tile_min(wp.tile(min_s))
        red_z = wp.tile_min(wp.tile(min_z))
        delta_s = -red_s[0]
        delta_z = -red_z[0]

        acc = dtype(0.0)
        for k in range(i, num_ineq, bd):
            s = s_all[b, k] + delta_s
            z = z_all[b, k] + delta_z
            s_all[b, k] = s
            z_all[b, k] = z
            acc += s * z * finite_mask_all[b, k]
        total = wp.tile_sum(wp.tile(acc))

        denom = num_finite_bounds[b]
        mu_b = dtype(0.0)
        if denom > dtype(0.0):
            mu_b = total[0] / denom
        # mu must be positive here, otherwise sqrt(mu) in the projection gives z = 0.
        mu_b = wp.max(mu_b, dtype(1e-10))

        for k in range(i, num_ineq, bd):
            if finite_mask_all[b, k] > dtype(0.5):
                c_bk = z_all[b, k] - delta_z
                z_new = (c_bk + wp.sqrt(c_bk * c_bk + dtype(4.0) * mu_b)) * dtype(0.5)
                z_all[b, k] = z_new
                s_all[b, k] = z_new - c_bk
            else:
                z_all[b, k] = dtype(0.0)
                s_all[b, k] = dtype(0.0)
        if i == 0:
            mu[b] = mu_b

    return init_guess_center_kernel


@functools.lru_cache(maxsize=None)
def create_update_residuals_r_kernel(dtype=wp.float64):
    """Single fused kernel for the whole ``_update_residuals_r`` body.

    Residual-unscaling factors are passed pre-combined as
    ``dual_res_unscale_factor`` (B, n) and ``primal_res_unscale_factor``
    (B, num_duals), materialized once per Ruiz update by the preconditioner.

    Stage 1 -- build regularized residuals from non-regularized ones. The
    ``delta`` / ``rho`` kernel args are the IPM proximal-step scalars; they
    are *not* the preconditioner's ``delta`` / ``delta_inv``:

        res.x[b]         = res_nr.x[b]         - rho[b]   * (result.x[b]         - prox.x[b])
        res.duals_all[b] = res_nr.duals_all[b] + delta[b] * (result.duals_all[b] - prox.duals_all[b])

    Stage 2 -- four reductions over the freshly computed residuals + the
    primal/dual prox-infeasibility measures:

        dual_res_reg[b]    = max_i ( |res.x[b, i]|                         * dual_res_unscale_factor[b, i] )
        dual_prox_inf[b]   = rho[b]   * max_i ( |result.x[b, i]     - prox.x[b, i]| )
        primal_prox_inf[b] = delta[b] * max_i ( |result.duals_all[b, i] - prox.duals_all[b, i]| )
        primal_res_reg[b]  = max_j ( |res.duals_all[b, j]|                 * primal_res_unscale_factor[b, j] )

    Stage 3 -- scalar finalize (thread 0 only):

        dual_res_reg_rel[b]   = dual_res_reg[b]   * dual_res_rel[b]   / dual_res[b]   if dual_res_rel[b]   > 0 else dual_res_reg[b]
        primal_res_reg_rel[b] = primal_res_reg[b] * primal_res_rel[b] / primal_res[b] if primal_res_rel[b] > 0 else primal_res_reg[b]

    ``num_duals == 0`` (no eq, no ineq, no box) gives zero prox-infeasibility
    and residual. Dispatch: ``wp.launch_tiled(..., dim=[B], block_dim=REDUCTION_BLOCK_DIM)``.
    """
    dtype = to_warp_dtype(dtype)

    @wp.kernel
    def update_residuals_r_kernel(
        # Stage 1 inputs
        rho:           wp.array(dtype=dtype),    # type: ignore  (B,)
        delta:         wp.array(dtype=dtype),    # type: ignore  (B,)
        res_nr_x:      wp.array2d(dtype=dtype),  # type: ignore  (B, n)
        res_nr_duals:   wp.array2d(dtype=dtype), # type: ignore  (B, num_duals)
        result_x:      wp.array2d(dtype=dtype),  # type: ignore  (B, n)
        result_duals:  wp.array2d(dtype=dtype),  # type: ignore  (B, num_duals)
        prox_x:        wp.array2d(dtype=dtype),  # type: ignore  (B, n)
        prox_duals:    wp.array2d(dtype=dtype),  # type: ignore  (B, num_duals)
        # Stage 1 outputs
        res_x:         wp.array2d(dtype=dtype),  # type: ignore  (B, n)
        res_duals:     wp.array2d(dtype=dtype),  # type: ignore  (B, num_duals)
        # Pre-combined residual unscaling factors (from preconditioner)
        dual_res_unscale_factor:   wp.array2d(dtype=dtype),  # type: ignore  (B, n)
        primal_res_unscale_factor: wp.array2d(dtype=dtype),  # type: ignore  (B, num_duals)
        # Stage 3 scalar inputs
        primal_res:     wp.array(dtype=dtype),  # type: ignore  (B,)
        primal_res_rel: wp.array(dtype=dtype),  # type: ignore  (B,)
        dual_res:       wp.array(dtype=dtype),  # type: ignore  (B,)
        dual_res_rel:   wp.array(dtype=dtype),  # type: ignore  (B,)
        # Outputs
        primal_res_reg:     wp.array(dtype=dtype),  # type: ignore
        primal_res_reg_rel: wp.array(dtype=dtype),  # type: ignore
        dual_res_reg:       wp.array(dtype=dtype),  # type: ignore
        dual_res_reg_rel:   wp.array(dtype=dtype),  # type: ignore
        primal_prox_inf:    wp.array(dtype=dtype),  # type: ignore
        dual_prox_inf:      wp.array(dtype=dtype),  # type: ignore
    ):
        b, i = wp.tid()
        n = res_nr_x.shape[1]
        num_duals = res_nr_duals.shape[1]
        bd = wp.block_dim()
        rho_b = rho[b]
        delta_b = delta[b]

        # --- x-sized pipeline: stage 1 (build res.x) + stage 2 (dual_prox_inf, dual_res_reg) ---
        max_diff_x = dtype(0.0)
        max_res_x = dtype(0.0)
        for k in range(i, n, bd):
            diff_x = result_x[b, k] - prox_x[b, k]
            new_res_x = res_nr_x[b, k] - rho_b * diff_x
            res_x[b, k] = new_res_x
            max_diff_x = wp.max(max_diff_x, wp.abs(diff_x))
            max_res_x = wp.max(max_res_x, wp.abs(new_res_x) * dual_res_unscale_factor[b, k])
        red_diff_x = wp.tile_max(wp.tile(max_diff_x))
        red_res_x = wp.tile_max(wp.tile(max_res_x))
        if i == 0:
            dual_prox_inf[b] = red_diff_x[0] * rho_b
            dual_res_reg[b] = red_res_x[0]

        # --- duals-sized pipeline: stage 1 (build res.duals) + stage 2 (primal_prox_inf, primal_res_reg) ---
        max_diff_d = dtype(0.0)
        max_res_d = dtype(0.0)
        for k in range(i, num_duals, bd):
            diff_d = result_duals[b, k] - prox_duals[b, k]
            new_res_d = res_nr_duals[b, k] + delta_b * diff_d
            res_duals[b, k] = new_res_d
            max_diff_d = wp.max(max_diff_d, wp.abs(diff_d))
            max_res_d = wp.max(max_res_d, wp.abs(new_res_d) * primal_res_unscale_factor[b, k])
        red_diff_d = wp.tile_max(wp.tile(max_diff_d))
        red_res_d = wp.tile_max(wp.tile(max_res_d))

        # --- Stage 3: scalar finalize --------------------------------------
        if i == 0:
            primal_prox_inf[b] = red_diff_d[0] * delta_b
            primal_res_reg[b] = red_res_d[0]

            if primal_res_rel[b] > dtype(0.0):
                primal_res_reg_rel[b] = primal_res_reg[b] * primal_res_rel[b] / primal_res[b]
            else:
                primal_res_reg_rel[b] = primal_res_reg[b]

            if dual_res_rel[b] > dtype(0.0):
                dual_res_reg_rel[b] = dual_res_reg[b] * dual_res_rel[b] / dual_res[b]
            else:
                dual_res_reg_rel[b] = dual_res_reg[b]

    return update_residuals_r_kernel


@functools.lru_cache(maxsize=None)
def create_update_residual_nr_kernel(dtype=wp.float64):
    r"""Fused kernel to update non-regularized residuals

    - ``minus_Px``      = ``-P*x``                             from ``eval_P_x(alpha=-1)``  (also used as the in-place ``res_nr.x`` slot)
    - ``A_x``           = ``+A*x``                             from ``eval_A_xn(alpha=+1)``
    - ``AT_y``          = ``A^T * y``                          from ``eval_AT_xt``
    - ``G_x``           = ``G*x``                              from ``eval_G_xn``
    - ``GT_zh_assembled`` = ``G^T * (z_u - z_l)``              from ``eval_GT_xt``
    - ``zb_assembled``  = ``x_b_scaling * (z_bu - z_bl)``      from ``prepare_zu_minus_zl_and_zbu_minus_zbl_kernel``

    Compute:

        res_nr.x      = -(P*x + c + A^T*y + G^T*(z_u - z_l) + x_b_scaling*(z_bu - z_bl))
        res_nr.y      = -A*x + b
        res_nr.z_hl   =  G*x - s_l - h_l            on finite lower rows, else 0
        res_nr.z_hu   = -G*x - s_u + h_u            on finite upper rows, else 0
        res_nr.z_xl   =  x_b_scaling*x - s_bl - x_l on finite lower box bounds, else 0
        res_nr.z_xu   = -(x_b_scaling*x + s_bu - x_u) on finite upper box bounds, else 0

        primal_obj    = (0.5 x^T P x + c^T x) * cost_scaling_inv
        dual_obj      = -(0.5 x^T P x + b^T y + h_u^T z_u - h_l^T z_l + x_u^T z_bu - x_l^T z_bl) * cost_scaling_inv
        duality_gap   = |primal_obj - dual_obj|
        duality_gap_rel = duality_gap / max(1, max_k(cost_scaling_inv * |w_k|))   for w_k in the 7 obj-sum terms

        primal_res    = max over the 5 segments of  ||delta_inv_seg * res_nr_seg||_inf  (unscaled)
        primal_norm    = max(||A*x||, ||G*x[hu]||, ||G*x[hl]||,
                            ||s_u||,  ||s_l||,  ||s_bu||, ||s_bl||,
                            constraints_rhs_inf_norm[b])    -- all unscaled
        primal_res_rel = primal_res / max(1, primal_norm)

        dual_res      = ||delta_x_inv * res_nr.x||_inf * cost_scaling_inv
        dual_norm     = max(||P*x||, ||c||, ||A^T*y + G^T*(z_u-z_l) + x_b_scaling*(z_bu-z_bl)||) * delta_x_inv * cost_scaling_inv
        dual_res_rel  = dual_res / max(1, dual_norm)

    All widths come from the array shapes; an absent block is ``(B, 0)``.
    Dispatch: ``wp.launch_tiled(..., dim=[B], block_dim=REDUCTION_BLOCK_DIM)``.
    """
    dtype = to_warp_dtype(dtype)

    @wp.func
    def finite_value(v: dtype, mask: dtype) -> dtype:    # type: ignore
        return wp.where(mask > dtype(0.5), v, dtype(0.0))

    @wp.kernel
    def update_residual_nr_kernel(
        minus_Px:                   wp.array2d(dtype=dtype),  # type: ignore  (B, n)
        A_x:                        wp.array2d(dtype=dtype),  # type: ignore  (B, p)
        AT_y:                       wp.array2d(dtype=dtype),  # type: ignore  (B, n)
        G_x:                        wp.array2d(dtype=dtype),  # type: ignore  (B, m)
        GT_zh_assembled:            wp.array2d(dtype=dtype),  # type: ignore  (B, n)
        zb_assembled:               wp.array2d(dtype=dtype),  # type: ignore  (B, n)
        # Data
        data_c:                     wp.array2d(dtype=dtype),  # type: ignore  (B, n)
        data_b:                     wp.array2d(dtype=dtype),  # type: ignore  (B, p)
        data_h_l:                   wp.array2d(dtype=dtype),  # type: ignore  (B, m)
        data_h_u:                   wp.array2d(dtype=dtype),  # type: ignore  (B, m)
        data_x_l:                   wp.array2d(dtype=dtype),  # type: ignore  (B, n)
        data_x_u:                   wp.array2d(dtype=dtype),  # type: ignore  (B, n)
        finite_mask_hl:                  wp.array2d(dtype=dtype),  # type: ignore  (B, m)
        finite_mask_hu:                  wp.array2d(dtype=dtype),  # type: ignore  (B, m)
        finite_mask_xl:                  wp.array2d(dtype=dtype),  # type: ignore  (B, n)
        finite_mask_xu:                  wp.array2d(dtype=dtype),  # type: ignore  (B, n)
        # Variables at current iteration
        result_x:                   wp.array2d(dtype=dtype),  # type: ignore  (B, n)
        result_y:                   wp.array2d(dtype=dtype),  # type: ignore  (B, p)
        result_z_hl:                wp.array2d(dtype=dtype),  # type: ignore  (B, num_hl)
        result_z_hu:                wp.array2d(dtype=dtype),  # type: ignore  (B, num_hu)
        result_z_xl:                wp.array2d(dtype=dtype),  # type: ignore  (B, num_xl)
        result_z_xu:                wp.array2d(dtype=dtype),  # type: ignore  (B, num_xu)
        result_s_hl:                wp.array2d(dtype=dtype),  # type: ignore  (B, num_hl)
        result_s_hu:                wp.array2d(dtype=dtype),  # type: ignore  (B, num_hu)
        result_s_xl:                wp.array2d(dtype=dtype),  # type: ignore  (B, num_xl)
        result_s_xu:                wp.array2d(dtype=dtype),  # type: ignore  (B, num_xu)
        # Preconditioner
        x_b_scaling:                wp.array2d(dtype=dtype),  # type: ignore  (B, n)
        cost_scaling_inv:           wp.array(dtype=dtype),    # type: ignore  (B,)
        delta_inv:                  wp.array2d(dtype=dtype),  # type: ignore  (B, n+p+m)
        delta_b_inv:                wp.array2d(dtype=dtype),  # type: ignore  (B, n)
        constraints_rhs_inf_norm:   wp.array(dtype=dtype),    # type: ignore  (B,)
        # Residuals
        res_nr_x:                   wp.array2d(dtype=dtype),  # type: ignore  (B, n)
        res_nr_y:                   wp.array2d(dtype=dtype),  # type: ignore  (B, p)
        res_nr_z_hl:                wp.array2d(dtype=dtype),  # type: ignore  (B, num_hl)
        res_nr_z_hu:                wp.array2d(dtype=dtype),  # type: ignore  (B, num_hu)
        res_nr_z_xl:                wp.array2d(dtype=dtype),  # type: ignore  (B, num_xl)
        res_nr_z_xu:                wp.array2d(dtype=dtype),  # type: ignore  (B, num_xu)
        # Objectices and residuals
        info_primal_obj:            wp.array(dtype=dtype),  # type: ignore  shape (B,)
        info_dual_obj:              wp.array(dtype=dtype),  # type: ignore  shape (B,)
        info_duality_gap:           wp.array(dtype=dtype),  # type: ignore  shape (B,)
        info_duality_gap_rel:       wp.array(dtype=dtype),  # type: ignore  shape (B,)
        info_primal_res:            wp.array(dtype=dtype),  # type: ignore  shape (B,)
        info_primal_res_rel:        wp.array(dtype=dtype),  # type: ignore  shape (B,)
        info_dual_res:              wp.array(dtype=dtype),  # type: ignore  shape (B,)
        info_dual_res_rel:          wp.array(dtype=dtype),  # type: ignore  shape (B,)
        info_prev_primal_res:       wp.array(dtype=dtype),  # type: ignore  shape (B,)
        info_prev_dual_res:         wp.array(dtype=dtype),  # type: ignore  shape (B,)
    ):
        b, i = wp.tid()
        bd = wp.block_dim()
        n = minus_Px.shape[1]
        p = A_x.shape[1]
        num_hl = result_z_hl.shape[1]
        num_hu = result_z_hu.shape[1]
        num_xl = result_z_xl.shape[1]
        num_xu = result_z_xu.shape[1]
        csi = cost_scaling_inv[b]

        if i == 0:
            info_prev_primal_res[b] = info_primal_res[b]
            info_prev_dual_res[b]   = info_dual_res[b]

        # ===================== x-segment =====================
        # res_nr.x = -P*x - c - AT*y - GT*(z_u-z_l) - x_b_scaling*(z_bu-z_bl)
        max_dual_res = dtype(0.0)
        sum_xPx = dtype(0.0)
        sum_cx = dtype(0.0)
        max_Px = dtype(0.0)
        max_c = dtype(0.0)
        max_accum = dtype(0.0)
        for k in range(i, n, bd):
            mpx = minus_Px[b, k]
            xk = result_x[b, k]
            ck = data_c[b, k]
            aty = AT_y[b, k]
            gtz = GT_zh_assembled[b, k]
            zbk = zb_assembled[b, k]
            dxi = delta_inv[b, k]
            r = mpx - ck - aty - gtz - zbk
            res_nr_x[b, k] = r
            max_dual_res = wp.max(max_dual_res, wp.abs(dxi * r))
            sum_xPx += mpx * xk
            sum_cx += ck * xk
            max_Px = wp.max(max_Px, wp.abs(mpx * dxi))
            max_c = wp.max(max_c, wp.abs(ck * dxi))
            max_accum = wp.max(max_accum, wp.abs((aty + gtz + zbk) * dxi))
        red_dual_res = wp.tile_max(wp.tile(max_dual_res))
        red_xPx = wp.tile_sum(wp.tile(sum_xPx))
        red_cx = wp.tile_sum(wp.tile(sum_cx))
        red_Px = wp.tile_max(wp.tile(max_Px))
        red_c = wp.tile_max(wp.tile(max_c))
        red_accum = wp.tile_max(wp.tile(max_accum))

        dual_res = red_dual_res[0] * csi
        half_xT_Px = dtype(-0.5) * red_xPx[0]
        cT_x = red_cx[0]
        drn_max = wp.max(red_Px[0], wp.max(red_c[0], red_accum[0]))

        # ---- segment obj-sum scalars (default 0; overridden inside guards) ----
        bT_y    = dtype(0.0)
        hl_zhl  = dtype(0.0)
        hu_zhu  = dtype(0.0)
        xl_zxl  = dtype(0.0)
        xu_zxu  = dtype(0.0)

        # ---- primal_res / primal_rel_norm running maxes (scalar) ----
        primal_res  = dtype(0.0)
        primal_res_rel_norm = dtype(0.0)

        # ===================== y-segment =====================
        if p > 0:
            sum_by = dtype(0.0)
            max_res_y = dtype(0.0)
            max_Ax = dtype(0.0)
            for k in range(i, p, bd):
                bk = data_b[b, k]
                ax = A_x[b, k]
                dyi = delta_inv[b, n + k]
                sum_by += bk * result_y[b, k]
                r = -ax + bk
                res_nr_y[b, k] = r
                max_res_y = wp.max(max_res_y, wp.abs(r * dyi))
                max_Ax = wp.max(max_Ax, wp.abs(ax * dyi))
            red_by = wp.tile_sum(wp.tile(sum_by))
            red_res_y = wp.tile_max(wp.tile(max_res_y))
            red_Ax = wp.tile_max(wp.tile(max_Ax))
            bT_y = red_by[0]
            primal_res = wp.max(primal_res, red_res_y[0])
            primal_res_rel_norm = wp.max(primal_res_rel_norm, red_Ax[0])

        # ===================== z_l (hl) segment =====================
        if num_hl > 0:
            sum_hz = dtype(0.0)
            max_res = dtype(0.0)
            max_gx = dtype(0.0)
            max_s = dtype(0.0)
            for k in range(i, num_hl, bd):
                mask = finite_mask_hl[b, k]
                hl = finite_value(data_h_l[b, k], mask)
                zhl = result_z_hl[b, k] * mask
                sum_hz += hl * zhl
                gx = G_x[b, k]
                s = result_s_hl[b, k] * mask
                r = (gx - s - hl) * mask
                res_nr_z_hl[b, k] = r
                dzi = delta_inv[b, n + p + k]
                max_res = wp.max(max_res, wp.abs(r * dzi))
                max_gx = wp.max(max_gx, wp.abs(gx * mask * dzi))
                max_s = wp.max(max_s, wp.abs(s * dzi))
            red_hz = wp.tile_sum(wp.tile(sum_hz))
            red_res = wp.tile_max(wp.tile(max_res))
            red_gx = wp.tile_max(wp.tile(max_gx))
            red_s = wp.tile_max(wp.tile(max_s))
            hl_zhl = red_hz[0]
            primal_res = wp.max(primal_res, red_res[0])
            primal_res_rel_norm = wp.max(primal_res_rel_norm, wp.max(red_gx[0], red_s[0]))

        # ===================== z_u (hu) segment =====================
        if num_hu > 0:
            sum_hz = dtype(0.0)
            max_res = dtype(0.0)
            max_gx = dtype(0.0)
            max_s = dtype(0.0)
            for k in range(i, num_hu, bd):
                mask = finite_mask_hu[b, k]
                hu = finite_value(data_h_u[b, k], mask)
                zhu = result_z_hu[b, k] * mask
                sum_hz += hu * zhu
                gx = G_x[b, k]
                s = result_s_hu[b, k] * mask
                r = (-gx - s + hu) * mask
                res_nr_z_hu[b, k] = r
                dzi = delta_inv[b, n + p + k]
                max_res = wp.max(max_res, wp.abs(r * dzi))
                max_gx = wp.max(max_gx, wp.abs(gx * mask * dzi))
                max_s = wp.max(max_s, wp.abs(s * dzi))
            red_hz = wp.tile_sum(wp.tile(sum_hz))
            red_res = wp.tile_max(wp.tile(max_res))
            red_gx = wp.tile_max(wp.tile(max_gx))
            red_s = wp.tile_max(wp.tile(max_s))
            hu_zhu = red_hz[0]
            primal_res = wp.max(primal_res, red_res[0])
            primal_res_rel_norm = wp.max(primal_res_rel_norm, wp.max(red_gx[0], red_s[0]))

        # ===================== z_bl (xl) segment =====================
        if num_xl > 0:
            sum_xz = dtype(0.0)
            max_res = dtype(0.0)
            max_s = dtype(0.0)
            for k in range(i, num_xl, bd):
                mask = finite_mask_xl[b, k]
                xl = finite_value(data_x_l[b, k], mask)
                zxl = result_z_xl[b, k] * mask
                sum_xz += xl * zxl
                s = result_s_xl[b, k] * mask
                r = (x_b_scaling[b, k] * result_x[b, k] - s - xl) * mask
                res_nr_z_xl[b, k] = r
                dbi = delta_b_inv[b, k]
                max_res = wp.max(max_res, wp.abs(r * dbi))
                max_s = wp.max(max_s, wp.abs(s * dbi))
            red_xz = wp.tile_sum(wp.tile(sum_xz))
            red_res = wp.tile_max(wp.tile(max_res))
            red_s = wp.tile_max(wp.tile(max_s))
            xl_zxl = red_xz[0]
            primal_res = wp.max(primal_res, red_res[0])
            primal_res_rel_norm = wp.max(primal_res_rel_norm, red_s[0])

        # ===================== z_bu (xu) segment =====================
        if num_xu > 0:
            sum_xz = dtype(0.0)
            max_res = dtype(0.0)
            max_s = dtype(0.0)
            for k in range(i, num_xu, bd):
                mask = finite_mask_xu[b, k]
                xu = finite_value(data_x_u[b, k], mask)
                zxu = result_z_xu[b, k] * mask
                sum_xz += xu * zxu
                s = result_s_xu[b, k] * mask
                r = (-x_b_scaling[b, k] * result_x[b, k] - s + xu) * mask
                res_nr_z_xu[b, k] = r
                dbi = delta_b_inv[b, k]
                max_res = wp.max(max_res, wp.abs(r * dbi))
                max_s = wp.max(max_s, wp.abs(s * dbi))
            red_xz = wp.tile_sum(wp.tile(sum_xz))
            red_res = wp.tile_max(wp.tile(max_res))
            red_s = wp.tile_max(wp.tile(max_s))
            xu_zxu = red_xz[0]
            primal_res = wp.max(primal_res, red_res[0])
            primal_res_rel_norm = wp.max(primal_res_rel_norm, red_s[0])

        # ===================== finalize info =====================
        if i == 0:
            # primal/dual obj + duality_gap
            info_primal_obj[b] = csi * (half_xT_Px + cT_x)
            info_dual_obj[b] = csi * (-half_xT_Px - bT_y - hu_zhu + hl_zhl - xu_zxu + xl_zxl)
            info_duality_gap[b] = wp.abs(info_primal_obj[b] - info_dual_obj[b])

            # duality_gap_rel: cost_scaling_inv * max(|0.5xPx|, |cTx|, |bTy|, |hl_zhl|, |hu_zhu|, |xl_zxl|, |xu_zxu|)
            duality_gap_rel_norm = wp.abs(half_xT_Px)
            duality_gap_rel_norm = wp.max(duality_gap_rel_norm, wp.abs(cT_x))
            duality_gap_rel_norm = wp.max(duality_gap_rel_norm, wp.abs(bT_y))
            duality_gap_rel_norm = wp.max(duality_gap_rel_norm, wp.abs(hl_zhl))
            duality_gap_rel_norm = wp.max(duality_gap_rel_norm, wp.abs(hu_zhu))
            duality_gap_rel_norm = wp.max(duality_gap_rel_norm, wp.abs(xl_zxl))
            duality_gap_rel_norm = wp.max(duality_gap_rel_norm, wp.abs(xu_zxu))
            duality_gap_rel_norm = wp.max(csi * duality_gap_rel_norm, dtype(1.0))
            info_duality_gap_rel[b] = info_duality_gap[b] / duality_gap_rel_norm

            # primal_res / primal_res_rel
            info_primal_res[b] = primal_res
            prn = wp.max(primal_res_rel_norm, constraints_rhs_inf_norm[b])
            prn = wp.max(prn, dtype(1.0))
            info_primal_res_rel[b] = primal_res / prn

            # dual_res / dual_res_rel
            info_dual_res[b] = dual_res
            dual_res_rel_norm = wp.max(drn_max * csi, dtype(1.0))
            info_dual_res_rel[b] = dual_res / dual_res_rel_norm

    return update_residual_nr_kernel


@functools.lru_cache(maxsize=None)
def create_update_smoothing_residual_nr_kernel(dtype=wp.float64):
    r"""Fused kernel for the gradient-smoothing Newton steps RHS.

    The negative non-regularized KKT residual (sign convention of the forward
    KKT solve RHS) with the complementarity row driven to a target ``mu`` instead
    of 0. The matrix-vector products are precomputed (same inputs as
    :func:`create_update_residual_nr_kernel`); this kernel does the elementwise
    assembly of every residual block plus the scaled-space inf-norm used for the
    relaxation convergence test, in one launch.

        res.x      = -(P*x + c + A^T*y + G^T*(z_u - z_l) + x_b_scaling*(z_bu - z_bl))
        res.y      = -A*x + b
        res.z_hl   =  (G*x - s_l - h_l) * mask_hl
        res.z_hu   =  (-G*x - s_u + h_u) * mask_hu
        res.z_xl   =  (x_b_scaling*x - s_bl - x_l) * mask_xl
        res.z_xu   =  (-x_b_scaling*x - s_bu + x_u) * mask_xu
        res.s_*    =  (mu - s_* * z_*) * mask_*          (relaxed complementarity)
        norm_out   =  max over all residual blocks of |.|   (scaled space)

    ``mu`` is the per-batch scaled target ``cost_scaling * gradient_smoothing_mu``.
    ``norm_out`` is a one-element atomic-max accumulator over the batch; zero it
    before the launch. Dispatch: ``wp.launch_tiled(..., dim=[B], block_dim=REDUCTION_BLOCK_DIM)``.
    """
    dtype = to_warp_dtype(dtype)

    @wp.func
    def finite_value(v: dtype, mask: dtype) -> dtype:    # type: ignore
        return wp.where(mask > dtype(0.5), v, dtype(0.0))

    @wp.kernel
    def update_smoothing_residual_nr_kernel(
        # Precomputed matrix-vector products
        minus_Px:        wp.array2d(dtype=dtype),  # type: ignore  (B, n)
        A_x:             wp.array2d(dtype=dtype),  # type: ignore  (B, p)
        AT_y:            wp.array2d(dtype=dtype),  # type: ignore  (B, n)
        G_x:             wp.array2d(dtype=dtype),  # type: ignore  (B, m)
        GT_zh_assembled: wp.array2d(dtype=dtype),  # type: ignore  (B, n)
        zb_assembled:    wp.array2d(dtype=dtype),  # type: ignore  (B, n)
        # Data
        data_c:          wp.array2d(dtype=dtype),  # type: ignore  (B, n)
        data_b:          wp.array2d(dtype=dtype),  # type: ignore  (B, p)
        data_h_l:        wp.array2d(dtype=dtype),  # type: ignore  (B, m)
        data_h_u:        wp.array2d(dtype=dtype),  # type: ignore  (B, m)
        data_x_l:        wp.array2d(dtype=dtype),  # type: ignore  (B, n)
        data_x_u:        wp.array2d(dtype=dtype),  # type: ignore  (B, n)
        finite_mask_hl:  wp.array2d(dtype=dtype),  # type: ignore  (B, m)
        finite_mask_hu:  wp.array2d(dtype=dtype),  # type: ignore  (B, m)
        finite_mask_xl:  wp.array2d(dtype=dtype),  # type: ignore  (B, n)
        finite_mask_xu:  wp.array2d(dtype=dtype),  # type: ignore  (B, n)
        # Relaxed iterate
        result_x:        wp.array2d(dtype=dtype),  # type: ignore  (B, n)
        result_z_hl:     wp.array2d(dtype=dtype),  # type: ignore  (B, num_hl)
        result_z_hu:     wp.array2d(dtype=dtype),  # type: ignore  (B, num_hu)
        result_z_xl:     wp.array2d(dtype=dtype),  # type: ignore  (B, num_xl)
        result_z_xu:     wp.array2d(dtype=dtype),  # type: ignore  (B, num_xu)
        result_s_hl:     wp.array2d(dtype=dtype),  # type: ignore  (B, num_hl)
        result_s_hu:     wp.array2d(dtype=dtype),  # type: ignore  (B, num_hu)
        result_s_xl:     wp.array2d(dtype=dtype),  # type: ignore  (B, num_xl)
        result_s_xu:     wp.array2d(dtype=dtype),  # type: ignore  (B, num_xu)
        # Preconditioner + target
        x_b_scaling:     wp.array2d(dtype=dtype),  # type: ignore  (B, n)
        mu_scaled:       wp.array(dtype=dtype),    # type: ignore  (B,)
        # Outputs (RHS blocks)
        res_x:           wp.array2d(dtype=dtype),  # type: ignore  (B, n)
        res_y:           wp.array2d(dtype=dtype),  # type: ignore  (B, p)
        res_z_hl:        wp.array2d(dtype=dtype),  # type: ignore  (B, num_hl)
        res_z_hu:        wp.array2d(dtype=dtype),  # type: ignore  (B, num_hu)
        res_z_xl:        wp.array2d(dtype=dtype),  # type: ignore  (B, num_xl)
        res_z_xu:        wp.array2d(dtype=dtype),  # type: ignore  (B, num_xu)
        res_s_hl:        wp.array2d(dtype=dtype),  # type: ignore  (B, num_hl)
        res_s_hu:        wp.array2d(dtype=dtype),  # type: ignore  (B, num_hu)
        res_s_xl:        wp.array2d(dtype=dtype),  # type: ignore  (B, num_xl)
        res_s_xu:        wp.array2d(dtype=dtype),  # type: ignore  (B, num_xu)
        norm_out:        wp.array(dtype=dtype),    # type: ignore  (1,) global inf-norm (atomic max over batch); must be pre-zeroed
    ):
        b, i = wp.tid()
        bd = wp.block_dim()
        n = minus_Px.shape[1]
        p = A_x.shape[1]
        num_hl = result_z_hl.shape[1]
        num_hu = result_z_hu.shape[1]
        num_xl = result_z_xl.shape[1]
        num_xu = result_z_xu.shape[1]
        mu_b = mu_scaled[b]

        # ---- res.x = -(P*x + c + A^T*y + G^T*(z_u-z_l) + x_b_scaling*(z_bu-z_bl)) ----
        nrm = dtype(0.0)
        for k in range(i, n, bd):
            r = minus_Px[b, k] - data_c[b, k] - AT_y[b, k] - GT_zh_assembled[b, k] - zb_assembled[b, k]
            res_x[b, k] = r
            nrm = wp.max(nrm, wp.abs(r))

        # ---- res.y = -A*x + b ----
        for k in range(i, p, bd):
            r = -A_x[b, k] + data_b[b, k]
            res_y[b, k] = r
            nrm = wp.max(nrm, wp.abs(r))

        # ---- lower inequality (hl): z-row and relaxed s-row ----
        for k in range(i, num_hl, bd):
            mask = finite_mask_hl[b, k]
            hl = finite_value(data_h_l[b, k], mask)
            s_raw = result_s_hl[b, k]
            z_raw = result_z_hl[b, k]
            r_z = (G_x[b, k] - s_raw * mask - hl) * mask
            res_z_hl[b, k] = r_z
            r_s = mu_b * mask - (s_raw * z_raw) * mask
            res_s_hl[b, k] = r_s
            nrm = wp.max(nrm, wp.max(wp.abs(r_z), wp.abs(r_s)))

        # ---- upper inequality (hu) ----
        for k in range(i, num_hu, bd):
            mask = finite_mask_hu[b, k]
            hu = finite_value(data_h_u[b, k], mask)
            s_raw = result_s_hu[b, k]
            z_raw = result_z_hu[b, k]
            r_z = (-G_x[b, k] - s_raw * mask + hu) * mask
            res_z_hu[b, k] = r_z
            r_s = mu_b * mask - (s_raw * z_raw) * mask
            res_s_hu[b, k] = r_s
            nrm = wp.max(nrm, wp.max(wp.abs(r_z), wp.abs(r_s)))

        # ---- lower box (xl) ----
        for k in range(i, num_xl, bd):
            mask = finite_mask_xl[b, k]
            xl = finite_value(data_x_l[b, k], mask)
            s_raw = result_s_xl[b, k]
            z_raw = result_z_xl[b, k]
            r_z = (x_b_scaling[b, k] * result_x[b, k] - s_raw * mask - xl) * mask
            res_z_xl[b, k] = r_z
            r_s = mu_b * mask - (s_raw * z_raw) * mask
            res_s_xl[b, k] = r_s
            nrm = wp.max(nrm, wp.max(wp.abs(r_z), wp.abs(r_s)))

        # ---- upper box (xu) ----
        for k in range(i, num_xu, bd):
            mask = finite_mask_xu[b, k]
            xu = finite_value(data_x_u[b, k], mask)
            s_raw = result_s_xu[b, k]
            z_raw = result_z_xu[b, k]
            r_z = (-x_b_scaling[b, k] * result_x[b, k] - s_raw * mask + xu) * mask
            res_z_xu[b, k] = r_z
            r_s = mu_b * mask - (s_raw * z_raw) * mask
            res_s_xu[b, k] = r_s
            nrm = wp.max(nrm, wp.max(wp.abs(r_z), wp.abs(r_s)))

        red = wp.tile_max(wp.tile(nrm))
        # Reduce the per-batch inf-norms to a single global scalar across blocks.
        if i == 0:
            wp.atomic_max(norm_out, 0, red[0])

    return update_smoothing_residual_nr_kernel


@functools.lru_cache(maxsize=None)
def create_smoothing_prepare_kernel(dtype=wp.float64):
    """Prepare the relaxed iterate for the gradient-smoothing Newton loop.

    Per batch ``b``, with ``mu_scaled = cost_scaling[b] * mu_user``::

        mu_scaled_out[b] = mu_scaled
        on finite-bound entries: s = max(s, sqrt(mu_scaled)); z = max(z, sqrt(mu_scaled))

    The floor keeps the first factorization of ``H = Q + G^T diag(z/s) G``
    well conditioned. Launch with ``dim=(B, max(num_ineq, 1))``.
    """
    dtype = to_warp_dtype(dtype)

    @wp.kernel
    def smoothing_prepare_kernel(
        cost_scaling:  wp.array(dtype=dtype),     # type: ignore  (B,)
        mu_user:       dtype,                     # type: ignore
        finite_mask_all: wp.array2d(dtype=dtype), # type: ignore  (B, num_ineq)
        s_all:         wp.array2d(dtype=dtype),   # type: ignore  (B, num_ineq) in-out
        z_all:         wp.array2d(dtype=dtype),   # type: ignore  (B, num_ineq) in-out
        mu_scaled_out: wp.array(dtype=dtype),     # type: ignore  (B,) out
    ):
        b, i = wp.tid()
        mu_scaled = cost_scaling[b] * mu_user
        if i == 0:
            mu_scaled_out[b] = mu_scaled
        if i < s_all.shape[1]:
            if finite_mask_all[b, i] > dtype(0.5):
                floor = wp.sqrt(mu_scaled)
                s_all[b, i] = wp.max(s_all[b, i], floor)
                z_all[b, i] = wp.max(z_all[b, i], floor)

    return smoothing_prepare_kernel


@functools.lru_cache(maxsize=None)
def create_smoothing_apply_step_kernel(dtype=wp.float64):
    """Damped common Newton step of the gradient-smoothing loop.

    ``alpha_s`` / ``alpha_z`` hold the fraction-to-boundary step lengths
    ``tau * min_i(-v_i / dv_i)`` from the step-length kernel. Per batch::

        a = tau * min( min(alpha_s, tau), min(alpha_z, tau) )
        x += a * dx;  y += a * dy
        s += a * (ds * mask);  z += a * (dz * mask)

    Each per-side step is clamped at ``tau`` (a single active row would
    otherwise exceed the full Newton step), and the common step is damped once
    more by ``tau`` to keep the iterates centered. Launch with
    ``dim=(B, n + p + 2 * num_ineq)``.
    """
    dtype = to_warp_dtype(dtype)

    @wp.kernel
    def smoothing_apply_step_kernel(
        alpha_s:  wp.array(dtype=dtype),     # type: ignore  (B,)
        alpha_z:  wp.array(dtype=dtype),     # type: ignore  (B,)
        settings:      wp.array(dtype=dtype),     # type: ignore  DeviceSettings.floats
        finite_mask_all: wp.array2d(dtype=dtype),  # type: ignore  (B, num_ineq)
        step_x:   wp.array2d(dtype=dtype),   # type: ignore  (B, n)
        step_y:   wp.array2d(dtype=dtype),   # type: ignore  (B, p)
        step_s:   wp.array2d(dtype=dtype),   # type: ignore  (B, num_ineq)
        step_z:   wp.array2d(dtype=dtype),   # type: ignore  (B, num_ineq)
        x:        wp.array2d(dtype=dtype),   # type: ignore  (B, n) in-out
        y:        wp.array2d(dtype=dtype),   # type: ignore  (B, p) in-out
        s_all:    wp.array2d(dtype=dtype),   # type: ignore  (B, num_ineq) in-out
        z_all:    wp.array2d(dtype=dtype),   # type: ignore  (B, num_ineq) in-out
    ):
        b, t = wp.tid()
        n = x.shape[1]
        p = y.shape[1]
        num_ineq = s_all.shape[1]
        tau = settings[SettingsFloatIdx.tau]
        a = wp.min(wp.min(alpha_s[b], tau), wp.min(alpha_z[b], tau)) * tau
        if t < n:
            x[b, t] = x[b, t] + a * step_x[b, t]
        elif t < n + p:
            k = t - n
            y[b, k] = y[b, k] + a * step_y[b, k]
        elif t < n + p + num_ineq:
            k = t - n - p
            s_all[b, k] = s_all[b, k] + a * (step_s[b, k] * finite_mask_all[b, k])
        elif t < n + p + 2 * num_ineq:
            k = t - n - p - num_ineq
            z_all[b, k] = z_all[b, k] + a * (step_z[b, k] * finite_mask_all[b, k])

    return smoothing_apply_step_kernel


@functools.lru_cache(maxsize=None)
def create_apply_finetune_kernel(dtype=wp.float64):
    """Regularization fine-tuning on the problems selected by ``mask``::

        reg_limit = reg_finetune_lower_limit;  no_primal_update = 0;  no_dual_update = 0

    Launch with ``dim=(B,)``.
    """
    dtype = to_warp_dtype(dtype)

    @wp.kernel
    def apply_finetune_kernel(
        mask:             wp.array(dtype=wp.bool),   # type: ignore  (B,)
        reg_limit:        wp.array(dtype=dtype),     # type: ignore  (B,) in-out
        no_primal_update: wp.array(dtype=wp.int32),  # type: ignore  (B,) in-out
        no_dual_update:   wp.array(dtype=wp.int32),  # type: ignore  (B,) in-out
        settings:              wp.array(dtype=dtype),     # type: ignore  DeviceSettings.floats
    ):
        b = wp.tid()
        if mask[b]:
            reg_limit[b] = settings[SettingsFloatIdx.reg_finetune_lower_limit]
            no_primal_update[b] = wp.int32(0)
            no_dual_update[b] = wp.int32(0)

    return apply_finetune_kernel


@functools.lru_cache(maxsize=None)
def create_reset_solve_state_kernel(dtype=wp.float64):
    """Per-problem state reset at the start of a solve, from the device
    configuration::

        rho = rho_init;  delta = delta_init;  reg_limit = reg_lower_limit
        mu = primal_step = dual_step = 0
        no_primal_update = no_dual_update = 0
        status = UNSOLVED;  iteration = 0;  retries = 0;  active = True

    Reading the initial values from the configuration array keeps a captured
    prelude valid when the user changes ``Settings`` between replays. Launch
    with ``dim=(B,)``.
    """
    dtype = to_warp_dtype(dtype)

    @wp.kernel
    def reset_solve_state_kernel(
        settings:              wp.array(dtype=dtype),     # type: ignore  DeviceSettings.floats
        rho:              wp.array(dtype=dtype),     # type: ignore  (B,)
        delta:            wp.array(dtype=dtype),     # type: ignore  (B,)
        reg_limit:        wp.array(dtype=dtype),     # type: ignore  (B,)
        mu:               wp.array(dtype=dtype),     # type: ignore  (B,)
        primal_step:      wp.array(dtype=dtype),     # type: ignore  (B,)
        dual_step:        wp.array(dtype=dtype),     # type: ignore  (B,)
        no_primal_update: wp.array(dtype=wp.int32),  # type: ignore  (B,)
        no_dual_update:   wp.array(dtype=wp.int32),  # type: ignore  (B,)
        status:           wp.array(dtype=wp.int32),  # type: ignore  (B,)
        iteration:        wp.array(dtype=wp.int32),  # type: ignore  (B,)
        retries:          wp.array(dtype=wp.int32),  # type: ignore  (B,)
        active:           wp.array(dtype=wp.bool),   # type: ignore  (B,)
    ):
        b = wp.tid()
        rho[b] = settings[SettingsFloatIdx.rho_init]
        delta[b] = settings[SettingsFloatIdx.delta_init]
        reg_limit[b] = settings[SettingsFloatIdx.reg_lower_limit]
        mu[b] = dtype(0.0)
        primal_step[b] = dtype(0.0)
        dual_step[b] = dtype(0.0)
        no_primal_update[b] = wp.int32(0)
        no_dual_update[b] = wp.int32(0)
        status[b] = wp.static(STATUS_UNSOLVED)
        iteration[b] = wp.int32(0)
        retries[b] = wp.int32(0)
        active[b] = True

    return reset_solve_state_kernel


@functools.lru_cache(maxsize=None)
def create_update_termination_kernel(dtype=wp.float64):
    """Per-problem termination and finetune decisions of one IPM iteration.

    For every batch entry ``b`` that is still active::

        iteration[b] = current_iter[0]
        converged = primal_ok and dual_ok [and gap_ok if check_duality_gap]
        if converged:                  status = SOLVED
        elif primal infeasible:        status = PRIMAL_INFEASIBLE
        elif dual infeasible:          status = DUAL_INFEASIBLE
        active[b] = (status == UNSOLVED)

    where, with ``thr_p = min(5, primal_update_threshold)`` and
    ``thr_d = min(5, dual_update_threshold)``::

        primal_ok         = primal_res < eps_abs  or primal_res_rel < eps_rel
        dual_ok           = dual_res   < eps_abs  or dual_res_rel   < eps_rel
        gap_ok            = gap < eps_gap_abs     or gap_rel < eps_gap_rel
        primal infeasible = no_dual_update > thr_d and primal_prox_inf > infeas
                            and (primal_res_reg < eps_abs or primal_res_reg_rel < eps_rel)
        dual infeasible   = no_primal_update > thr_p and dual_prox_inf > infeas
                            and (dual_res_reg < eps_abs or dual_res_reg_rel < eps_rel)

    Then, for the entries that are still active after this update::

        finetune[b] = ((no_primal_update > primal_update_threshold and rho == reg_limit)
                       or (no_dual_update > dual_update_threshold and delta == reg_limit))
                      and reg_limit != reg_finetune_lower
                      and dual_prox_inf < infeas and primal_prox_inf < infeas

    and ``finetune[b] = False`` otherwise. These are the decisions the host
    loop of ``SolverBase.solve`` makes; the tolerances come from the device
    configuration and are compared in the solver dtype, as NumPy compares the
    host mirror. Launch with ``dim=(B,)``.
    """
    dtype = to_warp_dtype(dtype)

    @wp.kernel
    def update_termination_kernel(
        primal_res:         wp.array(dtype=dtype),     # type: ignore  (B,)
        primal_res_rel:     wp.array(dtype=dtype),     # type: ignore  (B,)
        dual_res:           wp.array(dtype=dtype),     # type: ignore  (B,)
        dual_res_rel:       wp.array(dtype=dtype),     # type: ignore  (B,)
        duality_gap:        wp.array(dtype=dtype),     # type: ignore  (B,)
        duality_gap_rel:    wp.array(dtype=dtype),     # type: ignore  (B,)
        primal_res_reg:     wp.array(dtype=dtype),     # type: ignore  (B,)
        primal_res_reg_rel: wp.array(dtype=dtype),     # type: ignore  (B,)
        dual_res_reg:       wp.array(dtype=dtype),     # type: ignore  (B,)
        dual_res_reg_rel:   wp.array(dtype=dtype),     # type: ignore  (B,)
        primal_prox_inf:    wp.array(dtype=dtype),     # type: ignore  (B,)
        dual_prox_inf:      wp.array(dtype=dtype),     # type: ignore  (B,)
        rho:                wp.array(dtype=dtype),     # type: ignore  (B,)
        delta:              wp.array(dtype=dtype),     # type: ignore  (B,)
        reg_limit:          wp.array(dtype=dtype),     # type: ignore  (B,)
        no_primal_update:   wp.array(dtype=wp.int32),  # type: ignore  (B,)
        no_dual_update:     wp.array(dtype=wp.int32),  # type: ignore  (B,)
        current_iter:       wp.array(dtype=wp.int32),  # type: ignore  (1,)
        settings_f:              wp.array(dtype=dtype),     # type: ignore  DeviceSettings.floats
        settings_i:              wp.array(dtype=wp.int32),  # type: ignore  DeviceSettings.ints
        active:             wp.array(dtype=wp.bool),   # type: ignore  (B,) in-out
        status:             wp.array(dtype=wp.int32),  # type: ignore  (B,) in-out
        iteration:          wp.array(dtype=wp.int32),  # type: ignore  (B,) in-out
        finetune:           wp.array(dtype=wp.bool),   # type: ignore  (B,) output
    ):
        b = wp.tid()
        if not active[b]:
            finetune[b] = False
            return
        iteration[b] = current_iter[0]
        eps_abs = settings_f[SettingsFloatIdx.eps_abs]
        eps_rel = settings_f[SettingsFloatIdx.eps_rel]
        check_duality_gap = settings_i[SettingsIntIdx.check_duality_gap]
        eps_gap_abs = settings_f[SettingsFloatIdx.eps_duality_gap_abs]
        eps_gap_rel = settings_f[SettingsFloatIdx.eps_duality_gap_rel]
        infeas_thresh = settings_f[SettingsFloatIdx.infeasibility_threshold]
        primal_update_threshold = settings_i[SettingsIntIdx.reg_finetune_primal_update_threshold]
        dual_update_threshold = settings_i[SettingsIntIdx.reg_finetune_dual_update_threshold]
        reg_finetune_lower = settings_f[SettingsFloatIdx.reg_finetune_lower_limit]

        primal_ok = (primal_res[b] < eps_abs) or (primal_res_rel[b] < eps_rel)
        dual_ok = (dual_res[b] < eps_abs) or (dual_res_rel[b] < eps_rel)
        converged = primal_ok and dual_ok
        if check_duality_gap != 0:
            gap_ok = (duality_gap[b] < eps_gap_abs) or (duality_gap_rel[b] < eps_gap_rel)
            converged = converged and gap_ok

        primal_infeasible = (
            (no_dual_update[b] > wp.min(wp.int32(5), dual_update_threshold))
            and (primal_prox_inf[b] > infeas_thresh)
            and ((primal_res_reg[b] < eps_abs) or (primal_res_reg_rel[b] < eps_rel))
        )
        dual_infeasible = (
            (no_primal_update[b] > wp.min(wp.int32(5), primal_update_threshold))
            and (dual_prox_inf[b] > infeas_thresh)
            and ((dual_res_reg[b] < eps_abs) or (dual_res_reg_rel[b] < eps_rel))
        )

        if converged:
            status[b] = wp.static(STATUS_SOLVED)
            active[b] = False
        elif primal_infeasible:
            status[b] = wp.static(STATUS_PRIMAL_INFEASIBLE)
            active[b] = False
        elif dual_infeasible:
            status[b] = wp.static(STATUS_DUAL_INFEASIBLE)
            active[b] = False

        if not active[b]:
            finetune[b] = False
            return
        regs_tunable = (
            (reg_limit[b] != reg_finetune_lower)
            and (dual_prox_inf[b] < infeas_thresh)
            and (primal_prox_inf[b] < infeas_thresh)
        )
        finetune[b] = regs_tunable and (
            ((no_primal_update[b] > primal_update_threshold) and (rho[b] == reg_limit[b]))
            or ((no_dual_update[b] > dual_update_threshold) and (delta[b] == reg_limit[b]))
        )

    return update_termination_kernel


@wp.kernel
def update_continue_flag_kernel(
    active:        wp.array(dtype=wp.bool),   # type: ignore  (B,)
    current_iter:  wp.array(dtype=wp.int32),  # type: ignore  (1,)
    settings_i:         wp.array(dtype=wp.int32),  # type: ignore  DeviceSettings.ints
    continue_flag: wp.array(dtype=wp.int32),  # type: ignore  (1,) output
):
    """``continue_flag[0] = 1`` if any problem is active and
    ``current_iter[0] < max_iter`` (read from the device configuration), else ``0``.

    One block reduces the whole batch, so no cross-block reset/atomic pattern
    is needed. Dispatch with
    ``wp.launch_tiled(..., dim=[1], block_dim=REDUCTION_BLOCK_DIM)``.
    """
    _, i = wp.tid()
    any_active = wp.int32(0)
    for k in range(i, active.shape[0], wp.block_dim()):
        if active[k]:
            any_active = wp.int32(1)
    total = wp.tile_max(wp.tile(any_active))
    if i == 0:
        if total[0] > wp.int32(0) and current_iter[0] < settings_i[SettingsIntIdx.max_iter]:
            continue_flag[0] = wp.int32(1)
        else:
            continue_flag[0] = wp.int32(0)


@functools.lru_cache(maxsize=None)
def create_factor_retry_kernel(dtype=wp.float64):
    """Per-problem reaction to the backend's factorization status.

    For an active problem ``b`` whose ``factor_status[b]`` is nonzero::

        retries[b] += 1
        if retries[b] >= max_factor_retires:
            status = NUMERICAL_ISSUES;  iteration = current_iter;  active = False
        else:
            rho *= 100;  delta *= 100;  reg_limit = min(10 * reg_limit, eps_abs)
            needs_retry[b] = 1

    ``needs_retry[b]`` is 0 for every other problem, so a reduction over it
    tells whether the batch must be refactored. Inactive problems are ignored
    whatever their factor status: they still pass through the batched
    factorization after terminating. Launch with ``dim=(B,)``.
    """
    dtype = to_warp_dtype(dtype)

    @wp.kernel
    def factor_retry_kernel(
        factor_status: wp.array(dtype=wp.int32),  # type: ignore  (B,) backend factor status
        current_iter:  wp.array(dtype=wp.int32),  # type: ignore  (1,)
        settings_f:         wp.array(dtype=dtype),     # type: ignore  DeviceSettings.floats
        settings_i:         wp.array(dtype=wp.int32),  # type: ignore  DeviceSettings.ints
        active:        wp.array(dtype=wp.bool),   # type: ignore  (B,) in-out
        status:        wp.array(dtype=wp.int32),  # type: ignore  (B,) in-out
        iteration:     wp.array(dtype=wp.int32),  # type: ignore  (B,) in-out
        retries:       wp.array(dtype=wp.int32),  # type: ignore  (B,) in-out
        rho:           wp.array(dtype=dtype),     # type: ignore  (B,) in-out
        delta:         wp.array(dtype=dtype),     # type: ignore  (B,) in-out
        reg_limit:     wp.array(dtype=dtype),     # type: ignore  (B,) in-out
        needs_retry:   wp.array(dtype=wp.int32),  # type: ignore  (B,) output
    ):
        b = wp.tid()
        needs_retry[b] = wp.int32(0)
        if active[b] and factor_status[b] != wp.int32(0):
            retries[b] = retries[b] + wp.int32(1)
            if retries[b] >= settings_i[SettingsIntIdx.max_factor_retires]:
                status[b] = wp.static(STATUS_NUMERICAL_ISSUES)
                iteration[b] = current_iter[0]
                active[b] = False
            else:
                rho[b] = rho[b] * dtype(100.0)
                delta[b] = delta[b] * dtype(100.0)
                reg_limit[b] = wp.min(dtype(10.0) * reg_limit[b], settings_f[SettingsFloatIdx.eps_abs])
                needs_retry[b] = wp.int32(1)

    return factor_retry_kernel


@wp.kernel
def mark_failed_kernel(
    failed:       wp.array(dtype=wp.int32),  # type: ignore  (B,) nonzero = failed
    current_iter: wp.array(dtype=wp.int32),  # type: ignore  (1,)
    active:       wp.array(dtype=wp.bool),   # type: ignore  (B,) in-out
    status:       wp.array(dtype=wp.int32),  # type: ignore  (B,) in-out
    iteration:    wp.array(dtype=wp.int32),  # type: ignore  (B,) in-out
):
    """Active problems with a nonzero ``failed`` flag become ``NUMERICAL_ISSUES``,
    record ``current_iter`` and leave the active set. Used after a KKT solve
    whose solution is not finite. Launch with ``dim=(B,)``.
    """
    b = wp.tid()
    if active[b] and failed[b] != wp.int32(0):
        status[b] = wp.static(STATUS_NUMERICAL_ISSUES)
        iteration[b] = current_iter[0]
        active[b] = False


@functools.lru_cache(maxsize=None)
def create_solution_finite_kernel(dtype=wp.float64):
    """``status[b] = 1`` if any entry of row ``b`` of ``x``, ``y`` or ``z`` is
    not finite, ``0`` otherwise. ``wp.isfinite`` is false for NaN and for
    +inf / -inf alike, so one test covers both failure modes.

    Block reduction, one CUDA block per batch entry: every thread strides over
    the concatenated row and the block combines the flags, so the status is
    written exactly once and needs no clearing beforehand. Dispatch with
    ``wp.launch_tiled(..., dim=[B], block_dim=REDUCTION_BLOCK_DIM)``.
    """
    dtype = to_warp_dtype(dtype)

    @wp.kernel
    def solution_finite_kernel(
        x:      wp.array2d(dtype=dtype),     # type: ignore  (B, n)
        y:      wp.array2d(dtype=dtype),     # type: ignore  (B, p)
        z:      wp.array2d(dtype=dtype),     # type: ignore  (B, m)
        status: wp.array(dtype=wp.int32),    # type: ignore  (B,) output
    ):
        b, i = wp.tid()
        n = x.shape[1]
        p = y.shape[1]
        total = n + p + z.shape[1]
        bad = wp.int32(0)
        for t in range(i, total, wp.block_dim()):
            if t < n:
                v = x[b, t]
            elif t < n + p:
                v = y[b, t - n]
            else:
                v = z[b, t - n - p]
            if not wp.isfinite(v):  # false for NaN, +inf and -inf
                bad = wp.int32(1)
        any_bad = wp.tile_max(wp.tile(bad))
        if i == 0:
            status[b] = any_bad[0]

    return solution_finite_kernel


@wp.kernel
def any_flag_kernel(
    flags: wp.array(dtype=wp.int32),  # type: ignore  (B,)
    out:   wp.array(dtype=wp.int32),  # type: ignore  (1,) output
):
    """``out[0] = 1`` if any ``flags[b]`` is nonzero, else ``0``. One block
    reduces the whole batch; dispatch with
    ``wp.launch_tiled(..., dim=[1], block_dim=REDUCTION_BLOCK_DIM)``.
    """
    _, i = wp.tid()
    acc = wp.int32(0)
    for k in range(i, flags.shape[0], wp.block_dim()):
        if flags[k] != wp.int32(0):
            acc = wp.int32(1)
    total = wp.tile_max(wp.tile(acc))
    if i == 0:
        out[0] = total[0]


@wp.kernel
def terminate_active_kernel(
    active:       wp.array(dtype=wp.bool),   # type: ignore  (B,) in-out
    current_iter: wp.array(dtype=wp.int32),  # type: ignore  (1,)
    new_status:   wp.int32,
    status:       wp.array(dtype=wp.int32),  # type: ignore  (B,) in-out
    iteration:    wp.array(dtype=wp.int32),  # type: ignore  (B,) in-out
):
    """Give every still-active problem ``new_status``, record ``current_iter``
    as its termination iteration and deactivate it. Launch with ``dim=(B,)``.
    """
    b = wp.tid()
    if active[b]:
        status[b] = new_status
        iteration[b] = current_iter[0]
        active[b] = False


@wp.kernel
def advance_iteration_kernel(current_iter: wp.array(dtype=wp.int32)):  # type: ignore
    """``current_iter[0] += 1``. Launch with ``dim=1``."""
    current_iter[0] = current_iter[0] + wp.int32(1)


def create_init_guess_rhs_kernel(n: int, p: int,
                                 num_hl: int, num_hu: int,
                                 num_xl: int, num_xu: int,
                                 dtype=wp.float64):
    """Initial-RHS assembly for :meth:`Solver._initial_guess`.

    A single launch writes all six res blocks::

        res.x [b, k] = -c   [b, k]                   for k in [0, n)
        res.y [b, k] =  b   [b, k]                   for k in [0, p)
        res.z_l [b, k] = -h_l[b, k] if finite_mask_hl[b, k] else 0   for k in [0, m)
        res.z_u [b, k] =  h_u[b, k] if finite_mask_hu[b, k] else 0   for k in [0, m)
        res.z_bl[b, k] = -x_l[b, k] if finite_mask_xl[b, k] else 0   for k in [0, n)
        res.z_bu[b, k] =  x_u[b, k] if finite_mask_xu[b, k] else 0   for k in [0, n)
    """
    dtype = to_warp_dtype(dtype)
    n_s   = wp.static(n)
    p_s   = wp.static(p)
    nhl_s = wp.static(num_hl)
    nhu_s = wp.static(num_hu)
    nxl_s = wp.static(num_xl)
    nxu_s = wp.static(num_xu)

    off_y   = wp.static(n)                                # start of y block
    off_zl  = wp.static(n + p)                            # start of z_l
    off_zu  = wp.static(n + p + num_hl)                   # start of z_u
    off_zbl = wp.static(n + p + num_hl + num_hu)          # start of z_bl
    off_zbu = wp.static(n + p + num_hl + num_hu + num_xl) # start of z_bu

    @wp.kernel
    def init_guess_rhs_kernel(
        c:      wp.array2d(dtype=dtype),        # type: ignore  (B, n)
        b_eq:   wp.array2d(dtype=dtype),        # type: ignore  (B, p)
        h_l:    wp.array2d(dtype=dtype),        # type: ignore  (B, m)
        h_u:    wp.array2d(dtype=dtype),        # type: ignore  (B, m)
        x_l:    wp.array2d(dtype=dtype),        # type: ignore  (B, n)
        x_u:    wp.array2d(dtype=dtype),        # type: ignore  (B, n)
        finite_mask_hl: wp.array2d(dtype=dtype),     # type: ignore  (B, m)
        finite_mask_hu: wp.array2d(dtype=dtype),     # type: ignore  (B, m)
        finite_mask_xl: wp.array2d(dtype=dtype),     # type: ignore  (B, n)
        finite_mask_xu: wp.array2d(dtype=dtype),     # type: ignore  (B, n)
        res_x:   wp.array2d(dtype=dtype),       # type: ignore  (B, n)   out
        res_y:   wp.array2d(dtype=dtype),       # type: ignore  (B, p)   out
        res_zl:  wp.array2d(dtype=dtype),       # type: ignore  (B, num_hl) out
        res_zu:  wp.array2d(dtype=dtype),       # type: ignore  (B, num_hu) out
        res_zbl: wp.array2d(dtype=dtype),       # type: ignore  (B, num_xl) out
        res_zbu: wp.array2d(dtype=dtype),       # type: ignore  (B, num_xu) out
    ):
        b, t = wp.tid()

        if t < n_s:
            res_x[b, t] = -c[b, t]
        elif t < n_s + p_s:
            k = t - off_y
            res_y[b, k] = b_eq[b, k]
        elif t < n_s + p_s + nhl_s:
            k = t - off_zl
            res_zl[b, k] = wp.where(finite_mask_hl[b, k] > dtype(0.5), -h_l[b, k], dtype(0.0))
        elif t < n_s + p_s + nhl_s + nhu_s:
            k = t - off_zu
            res_zu[b, k] = wp.where(finite_mask_hu[b, k] > dtype(0.5), h_u[b, k], dtype(0.0))
        elif t < n_s + p_s + nhl_s + nhu_s + nxl_s:
            k = t - off_zbl
            res_zbl[b, k] = wp.where(finite_mask_xl[b, k] > dtype(0.5), -x_l[b, k], dtype(0.0))
        elif t < n_s + p_s + nhl_s + nhu_s + nxl_s + nxu_s:
            k = t - off_zbu
            res_zbu[b, k] = wp.where(finite_mask_xu[b, k] > dtype(0.5), x_u[b, k], dtype(0.0))

    return init_guess_rhs_kernel


def create_prepare_predictor_step_kernel(dtype=wp.float64):
    dtype = to_warp_dtype(dtype)
    """Fused kernel for the predictor-step RHS assembly:

        res.s_all[b, i] = -s_all[b, i] * z_all[b, i]
    """
    @wp.kernel
    def prepare_predictor_step_kernel(
        s_all:     wp.array2d(dtype=dtype),  # type: ignore  (B, num_ineq)
        z_all:     wp.array2d(dtype=dtype),  # type: ignore  (B, num_ineq)
        finite_mask_all: wp.array2d(dtype=dtype),  # type: ignore  (B, num_ineq)
        res_s_all: wp.array2d(dtype=dtype),  # type: ignore  (B, num_ineq) output
    ):
        b, i = wp.tid()
        res_s_all[b, i] = wp.where(
            finite_mask_all[b, i] > dtype(0.5),
            -s_all[b, i] * z_all[b, i],
            dtype(0.0)
        )

    return prepare_predictor_step_kernel


def create_prepare_corrector_step_kernel(dtype=wp.float64):
    dtype = to_warp_dtype(dtype)
    """Fused kernel for the corrector-step RHS update:

        res.s_all[b, i] = res.s_all[b, i] - step.s_all[b, i] * step.z_all[b, i] + sigma[b] * mu[b]

        ``res.s_all`` already holds `-s*z` from the predictor step on entry;
        the corrector adds the second-order correction `-ds*dz` plus the
        centering term `sigma*mu` (broadcast scalar per batch).
    """
    @wp.kernel
    def prepare_corrector_step_kernel(
        step_s_all: wp.array2d(dtype=dtype),  # type: ignore  (B, num_ineq)
        step_z_all: wp.array2d(dtype=dtype),  # type: ignore  (B, num_ineq)
        sigma:      wp.array(dtype=dtype),    # type: ignore  (B,)
        mu:         wp.array(dtype=dtype),    # type: ignore  (B,)
        finite_mask_all: wp.array2d(dtype=dtype),  # type: ignore  (B, num_ineq)
        res_s_all:  wp.array2d(dtype=dtype),  # type: ignore  (B, num_ineq) in-out
    ):
        b, i = wp.tid()
        res_s_all[b, i] = res_s_all[b, i] - step_s_all[b, i] * step_z_all[b, i] + sigma[b] * mu[b]
        res_s_all[b, i] = res_s_all[b, i] * finite_mask_all[b, i]

    return prepare_corrector_step_kernel


def create_update_vars_after_corrector_step_kernel(n: int, p: int, num_ineq: int, dtype=wp.float64):
    dtype = to_warp_dtype(dtype)
    """Fused scaled-add for ``_update_vars_after_corrector_step``.

    The contiguous buffers are laid out as::

        primals_all = [x | s_l | s_u | s_bl | s_bu]
        duals_all   = [y | z_l | z_u | z_bl | z_bu]

    and this kernel applies the line search steps in place::

        result.x += primal_step * step.x
        result.s += primal_step * step.s
        result.y += dual_step   * step.y
        result.z += dual_step   * step.z

    Full-length cone entries whose finite-bound mask is zero are not part of the
    barrier problem. Those slack/dual entries are forced back to exactly zero
    during the update so public results remain PIQP-style: length ``m`` or ``n``
    with zero duals/slacks at infinite bounds.

    Dispatch: ``wp.launch(kernel, dim=(B, n + num_ineq + p + num_ineq))``.
    ``n``, ``p``, and ``num_ineq`` are compile-time constants so the per-thread
    branches specialize cleanly.
    """
    n_primal = n + num_ineq
    n_dual = p + num_ineq

    @wp.kernel
    def update_vars_after_corrector_step_kernel(
        active_mask:      wp.array(dtype=wp.bool), # type: ignore  (B,)
        finite_mask_all:       wp.array2d(dtype=dtype),  # type: ignore  (B, num_ineq)
        primal_step:      wp.array(dtype=dtype),    # type: ignore  (B,)
        dual_step:        wp.array(dtype=dtype),    # type: ignore  (B,)
        step_primals_all: wp.array2d(dtype=dtype),  # type: ignore  (B, n_primal)
        step_duals_all:   wp.array2d(dtype=dtype),  # type: ignore  (B, n_dual)
        primals_all:      wp.array2d(dtype=dtype),  # type: ignore  (B, n_primal) in-out
        duals_all:        wp.array2d(dtype=dtype),  # type: ignore  (B, n_dual) in-out
    ):
        b, t = wp.tid()
        if not active_mask[b]:
            return

        n_static = wp.static(n)
        p_static = wp.static(p)
        n_primal_static = wp.static(n_primal)
        n_dual_static = wp.static(n_dual)

        if t < n_static:
            primals_all[b, t] = primals_all[b, t] + primal_step[b] * step_primals_all[b, t]
        elif t < n_primal_static:
            k = t - n_static
            if finite_mask_all[b, k] > dtype(0.5):
                primals_all[b, t] = primals_all[b, t] + primal_step[b] * step_primals_all[b, t]
            else:
                primals_all[b, t] = dtype(0.0)
        else:
            j = t - n_primal_static
            if j < p_static:
                duals_all[b, j] = duals_all[b, j] + dual_step[b] * step_duals_all[b, j]
            elif j < n_dual_static:
                k = j - p_static
                if finite_mask_all[b, k] > dtype(0.5):
                    duals_all[b, j] = duals_all[b, j] + dual_step[b] * step_duals_all[b, j]
                else:
                    duals_all[b, j] = dtype(0.0)

    return update_vars_after_corrector_step_kernel


def create_run_full_newton_step_kernel(n: int, p: int, dtype=wp.float64):
    dtype = to_warp_dtype(dtype)
    """Fused post-solve variable update for the equality-only (no-inequality)
    path ``_run_full_newton_step``.

    After the KKT Newton solve, computes:

        result.x[b, :] += step.x[b, :]                  (alpha_primal = 1)
        result.y[b, :] += step.y[b, :]                  (alpha_dual   = 1)
        primal_step[b]  = 1.0
        dual_step[b]    = 1.0

    (A full Newton step since there are no inequality constraints to limit
    the step length.) Thread 0 of each batch writes the scalar step-length
    fields; the other threads apply the elementwise add on ``x`` or ``y``.

    Dispatch: ``wp.launch(kernel, dim=(B, n+p))``. ``n`` and ``p`` are
    compile-time constants so the per-thread branch fully specializes.
    """
    @wp.kernel
    def run_full_newton_step_kernel(
        active_mask: wp.array(dtype=wp.bool), # (B,)     # type: ignore
        step_x:      wp.array2d(dtype=dtype),  # (B, n)   # type: ignore
        step_y:      wp.array2d(dtype=dtype),  # (B, p)   # type: ignore
        result_x:    wp.array2d(dtype=dtype),  # (B, n)   # type: ignore
        result_y:    wp.array2d(dtype=dtype),  # (B, p)   # type: ignore
        primal_step: wp.array(dtype=dtype),    # (B,)     # type: ignore
        dual_step:   wp.array(dtype=dtype),    # (B,)     # type: ignore
    ):
        b, t = wp.tid()
        if not active_mask[b]:
            return
        n_static = wp.static(n)
        p_static = wp.static(p)

        if t < n_static:
            result_x[b, t] = result_x[b, t] + step_x[b, t]
        elif t < n_static + p_static:
            idx = t - n_static
            result_y[b, idx] = result_y[b, idx] + step_y[b, idx]

        # Thread 0 of each batch sets the scalar step-length outputs.
        if t == 0:
            primal_step[b] = dtype(1.0)
            dual_step[b] = dtype(1.0)

    return run_full_newton_step_kernel


def create_prepare_zu_minus_zl_and_zbu_minus_zbl_kernel(m: int, n: int,
                                                        has_h_l: bool, has_h_u: bool,
                                                        has_x_l: bool, has_x_u: bool,
                                                        dtype=wp.float64):
    dtype = to_warp_dtype(dtype)
    """Pre-matvec combination for _update_residuals_nr.

    Full-length layout: a present dual block is stored at full size with exactly
    0 on infinite-bound rows, so the combination is a plain elementwise
    difference (no compressed gather/scatter needed)::

          zu_minus_zl[:, i]  = z_u[:, i] - z_l[:, i]                  for i in [0, m)
          zbu_minus_zbl[:, j] = x_b_scaling[:, j] * (z_bu[:, j] - z_bl[:, j])  for j in [0, n)

    The output ``zu_minus_zl`` is always full ``(B, m)``. An omitted inequality
    side (``has_h_l`` / ``has_h_u`` False) has empty ``(B, 0)`` duals and
    contributes 0; likewise an omitted box block (``has_x_l`` / ``has_x_u``).
    All optional reads are guarded by the static presence flags.
    """

    @wp.kernel
    def prepare_zu_minus_zl_and_zbu_minus_zbl_kernel(
        z_u:              wp.array2d(dtype=dtype),  # type: ignore  (B, num_hu)
        z_l:              wp.array2d(dtype=dtype),  # type: ignore  (B, num_hl)
        z_bl:             wp.array2d(dtype=dtype),  # type: ignore  (B, num_xl)
        z_bu:             wp.array2d(dtype=dtype),  # type: ignore  (B, num_xu)
        x_b_scaling:      wp.array2d(dtype=dtype),  # type: ignore  (B, n)
        zu_minus_zl:      wp.array2d(dtype=dtype),  # type: ignore  (B, m) output
        zbu_minus_zbl:    wp.array2d(dtype=dtype),  # type: ignore  (B, n) output
    ):
        m_static = wp.static(m)
        b, i = wp.tid()

        if i < m_static:
            zu = dtype(0.0)
            zl = dtype(0.0)
            if wp.static(has_h_u):
                zu = z_u[b, i]
            if wp.static(has_h_l):
                zl = z_l[b, i]
            zu_minus_zl[b, i] = zu - zl
        else:
            j = i - m_static
            zbu = dtype(0.0)
            zbl = dtype(0.0)
            if wp.static(has_x_u):
                zbu = z_bu[b, j]
            if wp.static(has_x_l):
                zbl = z_bl[b, j]
            zbu_minus_zbl[b, j] = x_b_scaling[b, j] * (zbu - zbl)

    return prepare_zu_minus_zl_and_zbu_minus_zbl_kernel


@functools.lru_cache(maxsize=None)
def create_update_rho_delta_with_ineq_kernel(dtype=wp.float64):
    """Adaptive-regularization update for the inequality-constrained path.

    Per batch entry, the improvement flags are decided from the current
    (pre-update) ``rho`` / ``delta``::

        dual_improved   = dual_res < 0.95 * prev_dual_res  or  dual_res < eps_abs
                          or dual_res_rel < eps_rel
                          or (rho == reg_finetune_lower and dual_prox_inf < infeas_thresh)
        primal_improved = the same with the primal quantities and delta

    then ``rho`` / ``delta`` and the stagnation counters are updated, and the
    two flags are stored for ``update_prox_vars_kernel``, which must be launched
    right after this one.

    Launch with ``dim=(B,)``.
    """
    dtype = to_warp_dtype(dtype)

    @wp.kernel
    def update_rho_delta_with_ineq_kernel(
        active_mask:           wp.array(dtype=wp.bool), # type: ignore
        info_dual_res:         wp.array(dtype=dtype),   # type: ignore
        info_prev_dual_res:    wp.array(dtype=dtype),   # type: ignore
        info_dual_res_rel:     wp.array(dtype=dtype),   # type: ignore
        info_dual_prox_inf:    wp.array(dtype=dtype),   # type: ignore
        info_primal_res:       wp.array(dtype=dtype),   # type: ignore
        info_prev_primal_res:  wp.array(dtype=dtype),   # type: ignore
        info_primal_res_rel:   wp.array(dtype=dtype),   # type: ignore
        info_primal_prox_inf:  wp.array(dtype=dtype),   # type: ignore
        info_reg_limit:        wp.array(dtype=dtype),   # type: ignore
        info_rho:              wp.array(dtype=dtype),   # type: ignore  in-out (B,)
        info_delta:            wp.array(dtype=dtype),   # type: ignore  in-out (B,)
        info_no_primal_update: wp.array(dtype=wp.int32),     # type: ignore  in-out (B,)
        info_no_dual_update:   wp.array(dtype=wp.int32),     # type: ignore  in-out (B,)
        dual_improved_out:     wp.array(dtype=wp.bool), # type: ignore  output (B,)
        primal_improved_out:   wp.array(dtype=wp.bool), # type: ignore  output (B,)
        settings:                           wp.array(dtype=dtype),  # type: ignore  DeviceSettings.floats
        current_iter:          wp.array(dtype=wp.int32),     # type: ignore  (1,)
    ):
        b = wp.tid()
        settings_eps_abs = settings[SettingsFloatIdx.eps_abs]
        settings_eps_rel = settings[SettingsFloatIdx.eps_rel]
        settings_reg_finetune_lower = settings[SettingsFloatIdx.reg_finetune_lower_limit]
        settings_infeas_thresh = settings[SettingsFloatIdx.infeasibility_threshold]
        if not active_mask[b]:
            return
        iter_under_5 = (current_iter[0] < wp.int32(5))
        old_rho = info_rho[b]
        old_delta = info_delta[b]
        dual_improved = (
            (info_dual_res[b] < dtype(0.95) * info_prev_dual_res[b])
            or (info_dual_res[b] < settings_eps_abs)
            or (info_dual_res_rel[b] < settings_eps_rel)
            or ((old_rho == settings_reg_finetune_lower) and (info_dual_prox_inf[b] < settings_infeas_thresh))
        )
        primal_improved = (
            (info_primal_res[b] < dtype(0.95) * info_prev_primal_res[b])
            or (info_primal_res[b] < settings_eps_abs)
            or (info_primal_res_rel[b] < settings_eps_rel)
            or ((old_delta == settings_reg_finetune_lower) and (info_primal_prox_inf[b] < settings_infeas_thresh))
        )
        dual_improved_out[b] = dual_improved
        primal_improved_out[b] = primal_improved

        rho_fast = wp.max(info_reg_limit[b], dtype(0.1) * old_rho)
        rho_slow = wp.max(info_reg_limit[b], dtype(0.5) * old_rho)
        rho_slow_ok = (not dual_improved) and (
            iter_under_5 or (info_dual_prox_inf[b] < settings_infeas_thresh)
        )
        if dual_improved:
            info_rho[b] = rho_fast
        elif rho_slow_ok:
            info_rho[b] = rho_slow
        else:
            pass

        delta_fast = wp.max(info_reg_limit[b], dtype(0.1) * old_delta)
        delta_slow = wp.max(info_reg_limit[b], dtype(0.5) * old_delta)
        delta_slow_ok = (not primal_improved) and (
            iter_under_5 or (info_primal_prox_inf[b] < settings_infeas_thresh)
        )
        if primal_improved:
            info_delta[b] = delta_fast
        elif delta_slow_ok:
            info_delta[b] = delta_slow
        else:
            pass

        if dual_improved:
            info_no_primal_update[b] = wp.int32(0)
        else:
            info_no_primal_update[b] = info_no_primal_update[b] + wp.int32(1)
        if primal_improved:
            info_no_dual_update[b] = wp.int32(0)
        else:
            info_no_dual_update[b] = info_no_dual_update[b] + wp.int32(1)

    return update_rho_delta_with_ineq_kernel


def create_update_prox_vars_kernel(n: int, num_duals: int, dtype=wp.float64):
    """Proximal-center update from the flags of ``update_rho_delta_with_ineq_kernel``::

        prox_x[b, :]     = result_x[b, :]      if dual_improved[b]
        prox_duals[b, :] = result_duals[b, :]  if primal_improved[b]

    Launch with ``dim=(B, n + num_duals)`` after ``update_rho_delta_with_ineq_kernel``.
    """
    dtype = to_warp_dtype(dtype)

    @wp.kernel
    def update_prox_vars_kernel(
        active_mask:           wp.array(dtype=wp.bool), # type: ignore
        dual_improved:         wp.array(dtype=wp.bool), # type: ignore  (B,)
        primal_improved:       wp.array(dtype=wp.bool), # type: ignore  (B,)
        result_x:              wp.array2d(dtype=dtype), # type: ignore
        prox_x:                wp.array2d(dtype=dtype), # type: ignore
        result_duals:          wp.array2d(dtype=dtype), # type: ignore
        prox_duals:            wp.array2d(dtype=dtype), # type: ignore
    ):
        b, i = wp.tid()
        if not active_mask[b]:
            return
        n_static = wp.static(n)
        num_duals_static = wp.static(num_duals)
        if i < n_static:
            prox_x[b, i] = wp.where(dual_improved[b], result_x[b, i], prox_x[b, i])
        elif i < n_static + num_duals_static:
            t = i - n_static
            prox_duals[b, t] = wp.where(primal_improved[b], result_duals[b, t], prox_duals[b, t])

    return update_prox_vars_kernel


def create_update_rho_delta_without_ineq_kernel(n: int, p: int, dtype=wp.float64):
    dtype = to_warp_dtype(dtype)
    """Fused adaptive-regularization update for the equality-only path.
    """
    @wp.kernel
    def update_rho_delta_without_ineq_kernel(
        active_mask:           wp.array(dtype=wp.bool), # type: ignore
        info_dual_res:         wp.array(dtype=dtype),   # type: ignore
        info_prev_dual_res:    wp.array(dtype=dtype),   # type: ignore
        info_dual_res_rel:     wp.array(dtype=dtype),   # type: ignore
        info_dual_prox_inf:    wp.array(dtype=dtype),   # type: ignore
        info_primal_res:       wp.array(dtype=dtype),   # type: ignore
        info_prev_primal_res:  wp.array(dtype=dtype),   # type: ignore
        info_primal_res_rel:   wp.array(dtype=dtype),   # type: ignore
        info_primal_prox_inf:  wp.array(dtype=dtype),   # type: ignore
        info_reg_limit:        wp.array(dtype=dtype),   # type: ignore
        info_rho:              wp.array(dtype=dtype),   # type: ignore
        info_delta:            wp.array(dtype=dtype),   # type: ignore
        info_no_primal_update: wp.array(dtype=wp.int32),     # type: ignore
        info_no_dual_update:   wp.array(dtype=wp.int32),     # type: ignore
        result_x:              wp.array2d(dtype=dtype), # type: ignore
        prox_x:                wp.array2d(dtype=dtype), # type: ignore
        result_y:              wp.array2d(dtype=dtype), # type: ignore
        prox_y:                wp.array2d(dtype=dtype), # type: ignore
        settings:                           wp.array(dtype=dtype),  # type: ignore  DeviceSettings.floats
        current_iter:          wp.array(dtype=wp.int32),     # type: ignore  (1,)
    ):
        b, i = wp.tid()
        settings_eps_abs = settings[SettingsFloatIdx.eps_abs]
        settings_eps_rel = settings[SettingsFloatIdx.eps_rel]
        settings_infeas_thresh = settings[SettingsFloatIdx.infeasibility_threshold]
        if not active_mask[b]:
            return
        n_static = wp.static(n)
        p_static = wp.static(p)
        iter_under_5 = (current_iter[0] < wp.int32(5))
        # Unlike the inequality path, these flags never read rho / delta, which
        # thread (b, 0) overwrites below, so every thread of row b sees the same
        # flags and the update can stay fused in one launch.
        dual_improved = (
            (info_dual_res[b] < dtype(0.95) * info_prev_dual_res[b])
            or (info_dual_res[b] < settings_eps_abs)
            or (info_dual_res_rel[b] < settings_eps_rel)
        )
        primal_improved = (
            (info_primal_res[b] < dtype(0.95) * info_prev_primal_res[b])
            or (info_primal_res[b] < settings_eps_abs)
            or (info_primal_res_rel[b] < settings_eps_rel)
        )

        if i == 0:
            old_rho = info_rho[b]
            rho_fast = wp.max(info_reg_limit[b], dtype(0.1) * old_rho)
            rho_slow = wp.max(info_reg_limit[b], dtype(0.5) * old_rho)
            rho_slow_ok = (not dual_improved) and (
                iter_under_5 or (info_dual_prox_inf[b] < settings_infeas_thresh)
            )
            if dual_improved:
                info_rho[b] = rho_fast
            elif rho_slow_ok:
                info_rho[b] = rho_slow
            else:
                pass

            old_delta = info_delta[b]
            delta_fast = wp.max(info_reg_limit[b], dtype(0.1) * old_delta)
            delta_slow = wp.max(info_reg_limit[b], dtype(0.5) * old_delta)
            delta_slow_ok = (not primal_improved) and (
                iter_under_5 or (info_primal_prox_inf[b] < settings_infeas_thresh)
            )
            if primal_improved:
                info_delta[b] = delta_fast
            elif delta_slow_ok:
                info_delta[b] = delta_slow
            else:
                pass

            # Reset on improved, increment on stagnated.
            if dual_improved:
                info_no_primal_update[b] = wp.int32(0)
            else:
                info_no_primal_update[b] = info_no_primal_update[b] + wp.int32(1)
            if primal_improved:
                info_no_dual_update[b] = wp.int32(0)
            else:
                info_no_dual_update[b] = info_no_dual_update[b] + wp.int32(1)

        if i < n_static:
            prox_x[b, i] = wp.where(dual_improved, result_x[b, i], prox_x[b, i])
        elif i < n_static + p_static:
            t = i - n_static
            prox_y[b, t] = wp.where(primal_improved, result_y[b, t], prox_y[b, t])

    return update_rho_delta_without_ineq_kernel


def create_boundary_shift_kernel(num_hl: int, num_hu: int, num_xl: int, num_xu: int, dtype=wp.float64):
    dtype = to_warp_dtype(dtype)
    """Per-element ``z`` boundary shift to avoid division-by-zero in the IPM.

    Only finite-bound entries are shifted::

        if finite_mask_hl[b, k] and z_hl[b, k] < eps: z_hl[b, k] += eps
        if finite_mask_hu[b, k] and z_hu[b, k] < eps: z_hu[b, k] += eps
        if finite_mask_xl[b, k] and z_bl[b, k] < eps: z_bl[b, k] += eps
        if finite_mask_xu[b, k] and z_bu[b, k] < eps: z_bu[b, k] += eps

    Inactive entries correspond to infinite bounds in the full-length layout.
    They are set to exactly zero here instead of being shifted positive, because
    they are not barrier variables and must not affect step lengths, mu/sigma,
    residual norms, or public dual/slack outputs.
    """
    # IEEE 754 float64 machine epsilon (== np.finfo(np.float64).eps).
    EPS_F64 = wp.constant(dtype(2.220446049250313e-16))

    @wp.kernel
    def boundary_shift_kernel(
        active_mask: wp.array(dtype=wp.bool), # type: ignore  (B,)
        finite_mask_hl: wp.array2d(dtype=dtype),  # type: ignore  (B, num_hl)
        finite_mask_hu: wp.array2d(dtype=dtype),  # type: ignore  (B, num_hu)
        finite_mask_xl: wp.array2d(dtype=dtype),  # type: ignore  (B, num_xl)
        finite_mask_xu: wp.array2d(dtype=dtype),  # type: ignore  (B, num_xu)
        z_hl: wp.array2d(dtype=dtype),  # type: ignore  (B, num_hl)
        z_hu: wp.array2d(dtype=dtype),  # type: ignore  (B, num_hu)
        z_bl: wp.array2d(dtype=dtype),  # type: ignore  (B, num_xl)
        z_bu: wp.array2d(dtype=dtype),  # type: ignore  (B, num_xu)
    ):
        b, t = wp.tid()
        if not active_mask[b]:
            return

        n_hl = wp.static(num_hl)
        n_hu = wp.static(num_hu)
        n_xl = wp.static(num_xl)
        n_xu = wp.static(num_xu)

        if t < n_hl:
            if finite_mask_hl[b, t] > dtype(0.5):
                if z_hl[b, t] < EPS_F64:
                    z_hl[b, t] = z_hl[b, t] + EPS_F64
            else:
                z_hl[b, t] = dtype(0.0)
        elif t < n_hl + n_hu:
            i = t - n_hl
            if finite_mask_hu[b, i] > dtype(0.5):
                if z_hu[b, i] < EPS_F64:
                    z_hu[b, i] = z_hu[b, i] + EPS_F64
            else:
                z_hu[b, i] = dtype(0.0)
        elif t < n_hl + n_hu + n_xl:
            i = t - n_hl - n_hu
            if finite_mask_xl[b, i] > dtype(0.5):
                if z_bl[b, i] < EPS_F64:
                    z_bl[b, i] = z_bl[b, i] + EPS_F64
            else:
                z_bl[b, i] = dtype(0.0)
        elif t < n_hl + n_hu + n_xl + n_xu:
            i = t - n_hl - n_hu - n_xl
            if finite_mask_xu[b, i] > dtype(0.5):
                if z_bu[b, i] < EPS_F64:
                    z_bu[b, i] = z_bu[b, i] + EPS_F64
            else:
                z_bu[b, i] = dtype(0.0)

    return boundary_shift_kernel


def create_backward_assemble_rhs_kernel(
    n: int, p: int,
    num_hu: int, num_hl: int, num_xu: int, num_xl: int,
    precond_on: bool,
dtype=wp.float64):
    dtype = to_warp_dtype(dtype)
    r"""Assemble adjoint RHS."""
    @wp.kernel
    def backward_assemble_rhs_kernel(
        grad_x:    wp.array2d(dtype=dtype),  # type: ignore (B, n)
        grad_y:    wp.array2d(dtype=dtype),  # type: ignore (B, p)
        grad_z_u:  wp.array2d(dtype=dtype),  # type: ignore (B, num_hu)
        grad_z_l:  wp.array2d(dtype=dtype),  # type: ignore (B, num_hl)
        grad_z_bu: wp.array2d(dtype=dtype),  # type: ignore (B, num_xu)
        grad_z_bl: wp.array2d(dtype=dtype),  # type: ignore (B, num_xl)
        grad_s_u:  wp.array2d(dtype=dtype),  # type: ignore (B, num_hu)
        grad_s_l:  wp.array2d(dtype=dtype),  # type: ignore (B, num_hl)
        grad_s_bu: wp.array2d(dtype=dtype),  # type: ignore (B, num_xu)
        grad_s_bl: wp.array2d(dtype=dtype),  # type: ignore (B, num_xl)
        # Preconditioner factors
        delta:             wp.array2d(dtype=dtype),  # type: ignore (B, n+p+m)
        delta_b:           wp.array2d(dtype=dtype),  # type: ignore (B, n)
        delta_inv:         wp.array2d(dtype=dtype),  # type: ignore (B, n+p+m)
        delta_b_inv:       wp.array2d(dtype=dtype),  # type: ignore (B, n)
        cost_scaling_inv:  wp.array(dtype=dtype),    # type: ignore (B,)
        # Outputs (rhs_adj fields)
        rhs_x:    wp.array2d(dtype=dtype),  # type: ignore
        rhs_y:    wp.array2d(dtype=dtype),  # type: ignore
        rhs_z_u:  wp.array2d(dtype=dtype),  # type: ignore
        rhs_z_l:  wp.array2d(dtype=dtype),  # type: ignore
        rhs_z_bu: wp.array2d(dtype=dtype),  # type: ignore
        rhs_z_bl: wp.array2d(dtype=dtype),  # type: ignore
        rhs_s_u:  wp.array2d(dtype=dtype),  # type: ignore
        rhs_s_l:  wp.array2d(dtype=dtype),  # type: ignore
        rhs_s_bu: wp.array2d(dtype=dtype),  # type: ignore
        rhs_s_bl: wp.array2d(dtype=dtype),  # type: ignore
    ):
        b, t = wp.tid()
        n_static     = wp.static(n)
        p_static     = wp.static(p)
        nhu_static   = wp.static(num_hu)
        nhl_static   = wp.static(num_hl)
        nxu_static   = wp.static(num_xu)
        nxl_static   = wp.static(num_xl)
        precond      = wp.static(precond_on)

        # Cumulative offsets for the t-range dispatch.
        end_x   = n_static
        end_y   = end_x   + p_static
        end_zu  = end_y   + nhu_static
        end_zl  = end_zu  + nhl_static
        end_zbu = end_zl  + nxu_static
        end_zbl = end_zbu + nxl_static
        end_su  = end_zbl + nhu_static
        end_sl  = end_su  + nhl_static
        end_sbu = end_sl  + nxu_static
        end_sbl = end_sbu + nxl_static

        if t < end_x:
            i = t
            v = -grad_x[b, i]
            if precond:
                v = v * delta[b, i]
            rhs_x[b, i] = v

        elif t < end_y:
            i = t - end_x
            v = -grad_y[b, i]
            if precond:
                v = v * delta[b, n_static + i] * cost_scaling_inv[b]
            rhs_y[b, i] = v

        elif t < end_zu:
            i = t - end_y
            v = -grad_z_u[b, i]
            if precond:
                v = v * delta[b, n_static + p_static + i] * cost_scaling_inv[b]
            rhs_z_u[b, i] = v

        elif t < end_zl:
            i = t - end_zu
            v = -grad_z_l[b, i]
            if precond:
                v = v * delta[b, n_static + p_static + i] * cost_scaling_inv[b]
            rhs_z_l[b, i] = v

        elif t < end_zbu:
            i = t - end_zl
            v = -grad_z_bu[b, i]
            if precond:
                v = v * delta_b[b, i] * cost_scaling_inv[b]
            rhs_z_bu[b, i] = v

        elif t < end_zbl:
            i = t - end_zbu
            v = -grad_z_bl[b, i]
            if precond:
                v = v * delta_b[b, i] * cost_scaling_inv[b]
            rhs_z_bl[b, i] = v

        elif t < end_su:
            i = t - end_zbl
            v = -grad_s_u[b, i]
            if precond:
                v = v * delta_inv[b, n_static + p_static + i]
            rhs_s_u[b, i] = v

        elif t < end_sl:
            i = t - end_su
            v = -grad_s_l[b, i]
            if precond:
                v = v * delta_inv[b, n_static + p_static + i]
            rhs_s_l[b, i] = v

        elif t < end_sbu:
            i = t - end_sl
            v = -grad_s_bu[b, i]
            if precond:
                v = v * delta_b_inv[b, i]
            rhs_s_bu[b, i] = v

        elif t < end_sbl:
            i = t - end_sbu
            v = -grad_s_bl[b, i]
            if precond:
                v = v * delta_b_inv[b, i]
            rhs_s_bl[b, i] = v

        else:
            return

    return backward_assemble_rhs_kernel


def create_backward_unscale_lhs_kernel(
    n: int, p: int,
    num_hu: int, num_hl: int, num_xu: int, num_xl: int,
    precond_on: bool,
    dtype=wp.float64):
    dtype = to_warp_dtype(dtype)
    r"""Step 3 of the backward pass — un-scale ``sol`` (scaled-space
    backsolve output) into user-space lambdas **in place**, in
    Variables active-size layout.

    Inverse of :func:`create_assemble_grad_rhs_kernel`, restricted to
    the groups that actually need scaling. The 6 groups are dispatched
    by t-range over ``(B, n + p + num_hu + num_hl + num_xu + num_xl)``:

    * ``x``     :  ``sol *= cost_scaling * delta[:n]``
    * ``y``     :  ``sol *= delta[n:n+p]``
    * ``z_u``   :  ``sol[i] *= delta[n+p+i]``     (full-length)
    * ``z_l``   :  ``sol[i] *= delta[n+p+i]``     (full-length)
    * ``z_bu``  :  ``sol[i] *= delta_b[i]``        (full-length)
    * ``z_bl``  :  ``sol[i] *= delta_b[i]``        (full-length)
    """
    @wp.kernel
    def backward_unscale_lhs_kernel(
        # In-place sol (active sizes), Variables fields of the passed-in `sol`
        sol_x:    wp.array2d(dtype=dtype),  # type: ignore (B, n)
        sol_y:    wp.array2d(dtype=dtype),  # type: ignore (B, p)
        sol_z_u:  wp.array2d(dtype=dtype),  # type: ignore (B, num_hu)
        sol_z_l:  wp.array2d(dtype=dtype),  # type: ignore (B, num_hl)
        sol_z_bu: wp.array2d(dtype=dtype),  # type: ignore (B, num_xu)
        sol_z_bl: wp.array2d(dtype=dtype),  # type: ignore (B, num_xl)
        # Preconditioner factors
        delta:        wp.array2d(dtype=dtype),  # type: ignore (B, n+p+m)
        delta_b:      wp.array2d(dtype=dtype),  # type: ignore (B, n)
        cost_scaling: wp.array(dtype=dtype),    # type: ignore (B,)
    ):
        b, t = wp.tid()
        n_static     = wp.static(n)
        p_static     = wp.static(p)
        nhu_static   = wp.static(num_hu)
        nhl_static   = wp.static(num_hl)
        nxu_static   = wp.static(num_xu)
        nxl_static   = wp.static(num_xl)
        precond      = wp.static(precond_on)

        # Cumulative offsets for the t-range dispatch (s_* groups
        # excluded — they don't need un-scaling).
        end_x   = n_static
        end_y   = end_x   + p_static
        end_zu  = end_y   + nhu_static
        end_zl  = end_zu  + nhl_static
        end_zbu = end_zl  + nxu_static
        end_zbl = end_zbu + nxl_static

        if not precond:
            return

        if t < end_x:
            i = t
            sol_x[b, i] = sol_x[b, i] * cost_scaling[b] * delta[b, i]

        elif t < end_y:
            i = t - end_x
            sol_y[b, i] = sol_y[b, i] * delta[b, n_static + i]

        elif t < end_zu:
            i = t - end_y
            sol_z_u[b, i] = sol_z_u[b, i] * delta[b, n_static + p_static + i]

        elif t < end_zl:
            i = t - end_zu
            sol_z_l[b, i] = sol_z_l[b, i] * delta[b, n_static + p_static + i]

        elif t < end_zbu:
            i = t - end_zl
            sol_z_bu[b, i] = sol_z_bu[b, i] * delta_b[b, i]

        elif t < end_zbl:
            i = t - end_zbu
            sol_z_bl[b, i] = sol_z_bl[b, i] * delta_b[b, i]

        else:
            return

    return backward_unscale_lhs_kernel


def create_backward_compute_vector_grad_kernel(
    n: int, p: int,
    num_hu: int, num_hl: int, num_xu: int, num_xl: int,
dtype=wp.float64):
    dtype = to_warp_dtype(dtype)
    r"""Fused per-element sign-flip / copy that turns the user-space
    adjoint lambdas into the six vector gradients ``(dc, db, dh_u,
    dh_l, dx_u, dx_l)`` packed into ``sol``'s Variables layout.

    Field mapping (dispatched by t-range over
    ``(B, n + p + num_hu + num_hl + num_xu + num_xl)``):

    * ``sol.x``    = ``grad.x``     (copy)
    * ``sol.y``    = ``-grad.y``    (negate)
    * ``sol.z_u``  = ``-grad.z_u``  (negate)
    * ``sol.z_l``  =  ``grad.z_l``  (copy)
    * ``sol.z_bu`` = ``-grad.z_bu`` (negate)
    * ``sol.z_bl`` =  ``grad.z_bl`` (copy)
    """
    @wp.kernel
    def backward_compute_vector_grad_kernel(
        grad_x:    wp.array2d(dtype=dtype),  # type: ignore (B, n)
        grad_y:    wp.array2d(dtype=dtype),  # type: ignore (B, p)
        grad_z_u:  wp.array2d(dtype=dtype),  # type: ignore (B, num_hu)
        grad_z_l:  wp.array2d(dtype=dtype),  # type: ignore (B, num_hl)
        grad_z_bu: wp.array2d(dtype=dtype),  # type: ignore (B, num_xu)
        grad_z_bl: wp.array2d(dtype=dtype),  # type: ignore (B, num_xl)
        sol_x:    wp.array2d(dtype=dtype),   # type: ignore (B, n)
        sol_y:    wp.array2d(dtype=dtype),   # type: ignore (B, p)
        sol_z_u:  wp.array2d(dtype=dtype),   # type: ignore (B, num_hu)
        sol_z_l:  wp.array2d(dtype=dtype),   # type: ignore (B, num_hl)
        sol_z_bu: wp.array2d(dtype=dtype),   # type: ignore (B, num_xu)
        sol_z_bl: wp.array2d(dtype=dtype),   # type: ignore (B, num_xl)
    ):
        b, t = wp.tid()
        n_static   = wp.static(n)
        p_static   = wp.static(p)
        nhu_static = wp.static(num_hu)
        nhl_static = wp.static(num_hl)
        nxu_static = wp.static(num_xu)
        nxl_static = wp.static(num_xl)

        end_x   = n_static
        end_y   = end_x   + p_static
        end_zu  = end_y   + nhu_static
        end_zl  = end_zu  + nhl_static
        end_zbu = end_zl  + nxu_static
        end_zbl = end_zbu + nxl_static

        if t < end_x:
            i = t
            sol_x[b, i] = grad_x[b, i]

        elif t < end_y:
            i = t - end_x
            sol_y[b, i] = -grad_y[b, i]

        elif t < end_zu:
            i = t - end_y
            sol_z_u[b, i] = -grad_z_u[b, i]

        elif t < end_zl:
            i = t - end_zu
            sol_z_l[b, i] = grad_z_l[b, i]

        elif t < end_zbu:
            i = t - end_zl
            sol_z_bu[b, i] = -grad_z_bu[b, i]

        elif t < end_zbl:
            i = t - end_zbu
            sol_z_bl[b, i] = grad_z_bl[b, i]

        else:
            return

    return backward_compute_vector_grad_kernel


def create_backward_copy_kernel(
    n: int, p: int,
    num_hu: int, num_hl: int, num_xu: int, num_xl: int,
dtype=wp.float64):
    dtype = to_warp_dtype(dtype)
    @wp.kernel
    def backward_copy_kernel(
        in_x:    wp.array2d(dtype=dtype),  # type: ignore (B, n)
        in_y:    wp.array2d(dtype=dtype),  # type: ignore (B, p)
        in_z_u:  wp.array2d(dtype=dtype),  # type: ignore (B, num_hu)
        in_z_l:  wp.array2d(dtype=dtype),  # type: ignore (B, num_hl)
        in_z_bu: wp.array2d(dtype=dtype),  # type: ignore (B, num_xu)
        in_z_bl: wp.array2d(dtype=dtype),  # type: ignore (B, num_xl)
        in_s_u:  wp.array2d(dtype=dtype),  # type: ignore (B, num_hu)
        in_s_l:  wp.array2d(dtype=dtype),  # type: ignore (B, num_hl)
        in_s_bu: wp.array2d(dtype=dtype),  # type: ignore (B, num_xu)
        in_s_bl: wp.array2d(dtype=dtype),  # type: ignore (B, num_xl)
        out_x:    wp.array2d(dtype=dtype),  # type: ignore
        out_y:    wp.array2d(dtype=dtype),  # type: ignore
        out_z_u:  wp.array2d(dtype=dtype),  # type: ignore
        out_z_l:  wp.array2d(dtype=dtype),  # type: ignore
        out_z_bu: wp.array2d(dtype=dtype),  # type: ignore
        out_z_bl: wp.array2d(dtype=dtype),  # type: ignore
        out_s_u:  wp.array2d(dtype=dtype),  # type: ignore
        out_s_l:  wp.array2d(dtype=dtype),  # type: ignore
        out_s_bu: wp.array2d(dtype=dtype),  # type: ignore
        out_s_bl: wp.array2d(dtype=dtype),  # type: ignore
    ):
        b, t = wp.tid()
        n_s   = wp.static(n)
        p_s   = wp.static(p)
        nhu_s = wp.static(num_hu)
        nhl_s = wp.static(num_hl)
        nxu_s = wp.static(num_xu)
        nxl_s = wp.static(num_xl)

        end_x    = n_s
        end_y    = end_x    + p_s
        end_zu   = end_y    + nhu_s
        end_zl   = end_zu   + nhl_s
        end_zbu  = end_zl   + nxu_s
        end_zbl  = end_zbu  + nxl_s
        end_su   = end_zbl  + nhu_s
        end_sl   = end_su   + nhl_s
        end_sbu  = end_sl   + nxu_s
        end_sbl  = end_sbu  + nxl_s

        if t < end_x:
            i = t
            out_x[b, i] = in_x[b, i]
        elif t < end_y:
            i = t - end_x
            out_y[b, i] = in_y[b, i]
        elif t < end_zu:
            i = t - end_y
            out_z_u[b, i] = in_z_u[b, i]
        elif t < end_zl:
            i = t - end_zu
            out_z_l[b, i] = in_z_l[b, i]
        elif t < end_zbu:
            i = t - end_zl
            out_z_bu[b, i] = in_z_bu[b, i]
        elif t < end_zbl:
            i = t - end_zbu
            out_z_bl[b, i] = in_z_bl[b, i]
        elif t < end_su:
            i = t - end_zbl
            out_s_u[b, i] = in_s_u[b, i]
        elif t < end_sl:
            i = t - end_su
            out_s_l[b, i] = in_s_l[b, i]
        elif t < end_sbu:
            i = t - end_sl
            out_s_bu[b, i] = in_s_bu[b, i]
        elif t < end_sbl:
            i = t - end_sbu
            out_s_bl[b, i] = in_s_bl[b, i]
        else:
            return

    return backward_copy_kernel


def create_backward_pack_full_layout_kernel(m: int, n: int,
                                            num_hl: int, num_hu: int,
                                            num_xl: int, num_xu: int,
                                            dtype=wp.float64):
    dtype = to_warp_dtype(dtype)
    r"""Fused full-layout pack for the backward pass.

    The inequality-row gradient scratch (``lam_z*_full`` / ``z*_full``) is always
    full ``(B, m)`` because matrix gradients accumulate per row of ``G``. When an
    inequality side is present its active dual block has width ``m`` and is copied
    in directly; when it was omitted at setup() its ``sol_z*`` / ``result_z*``
    block is empty ``(B, 0)`` and the corresponding full buffer is filled with 0
    (the absent side contributes no dual). Box blocks are likewise optional: an
    omitted box side has zero-width ``num_xl`` / ``num_xu`` regions.

    Outputs:

    * ``lam_zu_full, lam_zl_full ∈ (B, m)`` — ``sol.z_u/z_l`` (0 if side absent).
    * ``lam_zbu_full ∈ (B, num_xu)``, ``lam_zbl_full ∈ (B, num_xl)`` — copy of
      ``sol.z_bu/z_bl`` (empty when that box side is absent).
    * ``zu_full, zl_full ∈ (B, m)`` — ``result.z_u/z_l`` (0 if side absent).
    """
    @wp.kernel
    def backward_pack_full_layout_kernel(
        # Active-only inputs (Variables fields)
        sol_z_u:    wp.array2d(dtype=dtype),  # type: ignore (B, num_hu)
        sol_z_l:    wp.array2d(dtype=dtype),  # type: ignore (B, num_hl)
        sol_z_bu:   wp.array2d(dtype=dtype),  # type: ignore (B, num_xu)
        sol_z_bl:   wp.array2d(dtype=dtype),  # type: ignore (B, num_xl)
        result_z_u: wp.array2d(dtype=dtype),  # type: ignore (B, num_hu)
        result_z_l: wp.array2d(dtype=dtype),  # type: ignore (B, num_hl)
        # Full-layout outputs (no pre-zero required).
        lam_zu_full:  wp.array2d(dtype=dtype),  # type: ignore (B, m)
        lam_zl_full:  wp.array2d(dtype=dtype),  # type: ignore (B, m)
        lam_zbu_full: wp.array2d(dtype=dtype),  # type: ignore (B, num_xu)
        lam_zbl_full: wp.array2d(dtype=dtype),  # type: ignore (B, num_xl)
        zu_full:      wp.array2d(dtype=dtype),  # type: ignore (B, m)
        zl_full:      wp.array2d(dtype=dtype),  # type: ignore (B, m)
    ):
        b, t = wp.tid()
        m_static = wp.static(m)
        num_xu_static = wp.static(num_xu)
        num_xl_static = wp.static(num_xl)
        has_hu = wp.static(num_hu > 0)
        has_hl = wp.static(num_hl > 0)

        # Six regions, one full-buffer position per thread. The four inequality-
        # row regions are full (B, m); absent box sides contribute zero-width
        # regions.
        end_lam_zu  = m_static
        end_lam_zl  = end_lam_zu  + m_static
        end_lam_zbu = end_lam_zl  + num_xu_static
        end_lam_zbl = end_lam_zbu + num_xl_static
        end_res_zu  = end_lam_zbl + m_static
        end_res_zl  = end_res_zu  + m_static

        if t < end_lam_zu:
            j = t
            if has_hu:
                lam_zu_full[b, j] = sol_z_u[b, j]
            else:
                lam_zu_full[b, j] = dtype(0.0)

        elif t < end_lam_zl:
            j = t - end_lam_zu
            if has_hl:
                lam_zl_full[b, j] = sol_z_l[b, j]
            else:
                lam_zl_full[b, j] = dtype(0.0)

        elif t < end_lam_zbu:
            j = t - end_lam_zl
            lam_zbu_full[b, j] = sol_z_bu[b, j]

        elif t < end_lam_zbl:
            j = t - end_lam_zbu
            lam_zbl_full[b, j] = sol_z_bl[b, j]

        elif t < end_res_zu:
            j = t - end_lam_zbl
            if has_hu:
                zu_full[b, j] = result_z_u[b, j]
            else:
                zu_full[b, j] = dtype(0.0)

        elif t < end_res_zl:
            j = t - end_res_zu
            if has_hl:
                zl_full[b, j] = result_z_l[b, j]
            else:
                zl_full[b, j] = dtype(0.0)

        else:
            return

    return backward_pack_full_layout_kernel
