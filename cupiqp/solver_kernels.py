import warp as wp


def create_prepare_predictor_step_kernel():
    """Fused kernel for the predictor-step RHS assembly:

        res.s_all[b, i] = -s_all[b, i] * z_all[b, i]
    """
    @wp.kernel
    def prepare_predictor_step_kernel(
        s_all:     wp.array2d(dtype=wp.float64),  # type: ignore  (B, num_ineq)
        z_all:     wp.array2d(dtype=wp.float64),  # type: ignore  (B, num_ineq)
        res_s_all: wp.array2d(dtype=wp.float64),  # type: ignore  (B, num_ineq) output
    ):
        b, i = wp.tid()
        res_s_all[b, i] = -s_all[b, i] * z_all[b, i]

    return prepare_predictor_step_kernel


def create_prepare_corrector_step_kernel():
    """Fused kernel for the corrector-step RHS update:

        res.s_all[b, i] = res.s_all[b, i] - step.s_all[b, i] * step.z_all[b, i] + sigma[b] * mu[b]

        ``res.s_all`` already holds `-s*z` from the predictor step on entry;
        the corrector adds the second-order correction `-ds*dz` plus the
        centering term `sigma*mu` (broadcast scalar per batch).
    """
    @wp.kernel
    def prepare_corrector_step_kernel(
        step_s_all: wp.array2d(dtype=wp.float64),  # type: ignore  (B, num_ineq)
        step_z_all: wp.array2d(dtype=wp.float64),  # type: ignore  (B, num_ineq)
        sigma:      wp.array(dtype=wp.float64),    # type: ignore  (B,)
        mu:         wp.array(dtype=wp.float64),    # type: ignore  (B,)
        res_s_all:  wp.array2d(dtype=wp.float64),  # type: ignore  (B, num_ineq) in-out
    ):
        b, i = wp.tid()
        sigma_mu = sigma[b] * mu[b]  # cheap redundant per-thread compute; no sync
        res_s_all[b, i] = res_s_all[b, i] - step_s_all[b, i] * step_z_all[b, i] + sigma_mu

    return prepare_corrector_step_kernel


def create_run_full_newton_step_kernel(n: int, p: int):
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
        step_x:      wp.array2d(dtype=wp.float64),  # (B, n)   # type: ignore
        step_y:      wp.array2d(dtype=wp.float64),  # (B, p)   # type: ignore
        result_x:    wp.array2d(dtype=wp.float64),  # (B, n)   # type: ignore
        result_y:    wp.array2d(dtype=wp.float64),  # (B, p)   # type: ignore
        primal_step: wp.array(dtype=wp.float64),    # (B,)     # type: ignore
        dual_step:   wp.array(dtype=wp.float64),    # (B,)     # type: ignore
    ):
        b, t = wp.tid()
        n_static = wp.static(n)
        p_static = wp.static(p)

        if t < n_static:
            result_x[b, t] = result_x[b, t] + step_x[b, t]
        elif t < n_static + p_static:
            idx = t - n_static
            result_y[b, idx] = result_y[b, idx] + step_y[b, idx]

        # Thread 0 of each batch sets the scalar step-length outputs.
        if t == 0:
            primal_step[b] = wp.float64(1.0)
            dual_step[b] = wp.float64(1.0)

    return run_full_newton_step_kernel


def create_calculate_step_kernel(num_ineq: int):
    """Fused block-reduction kernel for step lengths (primal and dual).

    For each batch ``b``, computes:

        alpha_s[b] = tau * min_i( ds[b,i] < 0 ? -s[b,i]/ds[b,i] : 1.0 )
        alpha_z[b] = tau * min_i( dz[b,i] < 0 ? -z[b,i]/dz[b,i] : 1.0 )

    Dispatch with ``wp.launch_tiled(..., dim=[B], block_dim=block_dim)``:
    one CUDA block per batch, threads cooperating to reduce ``num_ineq``-long
    rows via ``wp.tile_min``. The primal (s) and dual (z) paths share one
    block -- tiles from the s-pipeline go out of scope before the z-pipeline's
    loads, so shared memory is recycled between the two reductions. Thread 0
    finalizes both outputs by multiplying by ``tau``.
    """
    @wp.func
    def step_candidate(a: wp.float64, b: wp.float64) -> wp.float64:
        return wp.where(a < wp.float64(0.0), -b / a, wp.float64(1.0))

    @wp.kernel
    def calculate_step_kernel(
        s_all: wp.array2d(dtype=wp.float64),        # (B, num_ineq)  # type: ignore
        z_all: wp.array2d(dtype=wp.float64),        # (B, num_ineq)  # type: ignore
        step_s_all: wp.array2d(dtype=wp.float64),   # (B, num_ineq)  # type: ignore
        step_z_all: wp.array2d(dtype=wp.float64),   # (B, num_ineq)  # type: ignore
        tau: wp.array(dtype=wp.float64),            # (1,)           # type: ignore
        alpha_s: wp.array(dtype=wp.float64),        # (B,) output    # type: ignore
        alpha_z: wp.array(dtype=wp.float64),        # (B,) output    # type: ignore
    ):
        b, i = wp.tid()

        # Primal-step pipeline: alpha_s[b] = tau * min_i(cand_s[i])
        s_tile = wp.tile_load(s_all[b], shape=num_ineq)
        ds_tile = wp.tile_load(step_s_all[b], shape=num_ineq)
        cand_s = wp.tile_map(step_candidate, ds_tile, s_tile)
        min_s = wp.tile_min(cand_s)
        wp.tile_store(alpha_s, min_s, offset=b)  # store min_s into the corresponding batch of alpha_s

        # Dual-step pipeline: alpha_z[b] = tau * min_i(cand_z[i])
        z_tile = wp.tile_load(z_all[b], shape=num_ineq)
        dz_tile = wp.tile_load(step_z_all[b], shape=num_ineq)
        cand_z = wp.tile_map(step_candidate, dz_tile, z_tile)
        min_z = wp.tile_min(cand_z)
        wp.tile_store(alpha_z, min_z, offset=b)  # store min_z into the corresponding batch of alpha_z

        # Scalar tau multiply via thread 0 (tile_store is block-collective, so
        # the read-back is safe).
        if i == 0:
            alpha_s[b] = alpha_s[b] * tau[0]
            alpha_z[b] = alpha_z[b] * tau[0]

    return calculate_step_kernel


def create_calculate_mu_kernel(num_ineq: int):
    """Fused block-reduction kernel for the duality measure mu.

    For each batch ``b``, computes:

        mu[b] = sum_i( s[b, i] * z[b, i] ) / num_ineq

    Dispatch with ``wp.launch_tiled(..., dim=[B], block_dim=block_dim)``:
    one CUDA block per batch, threads cooperating on the ``num_ineq``-long
    reduction via ``wp.tile_sum``. Thread 0 performs the scalar
    ``/ num_ineq`` finalization after the block-collective ``tile_store``
    (which acts as a block-wide write fence).
    """
    @wp.kernel
    def calculate_mu_kernel(
        s_all: wp.array2d(dtype=wp.float64),  # (B, num_ineq)  # type: ignore
        z_all: wp.array2d(dtype=wp.float64),  # (B, num_ineq)  # type: ignore
        mu:    wp.array(dtype=wp.float64),    # (B,) output    # type: ignore
    ):
        b, tid = wp.tid()

        # Per-batch row loads, cooperatively filled by block threads.
        s_tile = wp.tile_load(s_all[b], shape=num_ineq)
        z_tile = wp.tile_load(z_all[b], shape=num_ineq)

        # Block-wide reduction: sum_i(s*z). Result is a (1,)-tile in shared mem.
        sum_tile = wp.tile_sum(s_tile * z_tile)
        wp.tile_store(mu, sum_tile, offset=b)

        # Scalar finalize by thread 0; safe because tile_store is block-collective.
        if tid == 0:
            mu[b] = mu[b] / wp.float64(num_ineq)

    return calculate_mu_kernel


def create_calculate_sigma_kernel(num_ineq: int):
    """Fused block-reduction kernel for the centering parameter sigma.

    For each batch ``b``, computes:

        s_trial[i] = s[b, i] + alpha_s[b] * ds[b, i]
        z_trial[i] = z[b, i] + alpha_z[b] * dz[b, i]
        acc        = sum_i( s_trial[i] * z_trial[i] )
        sigma[b]   = clip( acc / (mu[b] * num_ineq), 0, 1 ) ** 3

    Dispatch with ``wp.launch_tiled(..., dim=[B], block_dim=block_dim)``:
    one CUDA block per batch, ``block_dim`` threads cooperating to reduce the
    ``num_ineq``-long row via Warp's tile API (shared-memory block reduction).
    Thread 0 performs the scalar ``/mu``, ``clamp``, and ``^3`` finalization.
    """
    @wp.kernel
    def calculate_sigma_kernel(
        s_all: wp.array2d(dtype=wp.float64),        # (B, num_ineq)  # type: ignore
        z_all: wp.array2d(dtype=wp.float64),        # (B, num_ineq)  # type: ignore
        step_s_all: wp.array2d(dtype=wp.float64),   # (B, num_ineq)  # type: ignore
        step_z_all: wp.array2d(dtype=wp.float64),   # (B, num_ineq)  # type: ignore
        primal_step: wp.array(dtype=wp.float64),    # (B,) alpha_s   # type: ignore
        dual_step: wp.array(dtype=wp.float64),      # (B,) alpha_z   # type: ignore
        mu: wp.array(dtype=wp.float64),             # (B,)           # type: ignore
        sigma: wp.array(dtype=wp.float64),          # (B,) output    # type: ignore
    ):
        # launch_tiled gives us (batch_idx, thread_in_block).
        b, i = wp.tid()
        alpha_s = primal_step[b]
        alpha_z = dual_step[b]
        denominator = mu[b] * wp.float64(num_ineq)

        # Block-wide tile loads + elementwise fuse + reduction. All intermediates
        # stay in registers / shared memory (no DRAM round-trip).
        s_tile = wp.tile_load(s_all[b], shape=num_ineq)
        z_tile = wp.tile_load(z_all[b], shape=num_ineq)
        ds_tile = wp.tile_load(step_s_all[b], shape=num_ineq)
        dz_tile = wp.tile_load(step_z_all[b], shape=num_ineq)
        prod = (s_tile + alpha_s * ds_tile) * (z_tile + alpha_z * dz_tile)
        sum_tile = wp.tile_sum(prod)  # (1,)-tile, in shared memory

        # Write raw sum; tile_store is a block-synchronous collective op, so the
        # subsequent thread-0 read sees the final value. Thread 0 then overwrites
        # in place with the finalized sigma.
        wp.tile_store(sigma, sum_tile, offset=b)
        if i == 0:
            val = sigma[b] / denominator
            val = wp.clamp(val, wp.float64(0.0), wp.float64(1.0))
            sigma[b] = val * val * val

    return calculate_sigma_kernel


def create_update_residuals_r_kernel(
    n: int, p: int, num_hu: int, num_hl: int, num_xu: int, num_xl: int,
):
    """Single fused kernel for the whole ``_update_residuals_r`` body.

    Residual-unscaling factors are passed pre-combined as
    ``dual_res_unscale_factor`` (B, n) and ``primal_res_unscale_factor``
    (B, num_duals), materialized once per Ruiz update by the preconditioner
    (see ``RuizEquilibration._refresh_unscale_factors``). The kernel needs
    no preconditioner state or gather indices — every stage-2 reduction is
    a contiguous tile_max.

    Stage 1 -- build regularized residuals from non-regularized ones. The
    ``delta`` / ``rho`` kernel args are the IPM proximal-step scalars; they
    are *not* the preconditioner's ``delta`` / ``delta_inv``:

        res.x[b]         = res_nr.x[b]         - rho[b]   * (result.x[b]         - prox.x[b])
        res.duals_all[b] = res_nr.duals_all[b] + delta[b] * (result.duals_all[b] - prox.duals_all[b])

    Stage 2 -- four reductions over the freshly computed residuals + the
    primal/dual prox-infeasibility measures. The two ``*_unscale_factor``
    inputs hold:

        dual_res_unscale_factor[b, i]    = cost_scaling_inv[b] * delta_inv[b, i]
                                           for i ∈ [0, n)

        primal_res_unscale_factor[b, j]  = packed in Variables._dual_buffer
                                           order [y | z_l | z_u | z_bl | z_bu]:
                                             [y]:    delta_inv[b, n : n+p]
                                             [z_l]:  delta_inv[b, n + p + idx_hl]
                                             [z_u]:  delta_inv[b, n + p + idx_hu]
                                             [z_bl]: delta_b_inv[b, idx_xl]
                                             [z_bu]: delta_b_inv[b, idx_xu]

        dual_res_reg[b]    = max_i ( |res.x[b, i]|                         * dual_res_unscale_factor[b, i] )
        dual_prox_inf[b]   = rho[b]   * max_i ( |result.x[b, i]     - prox.x[b, i]| )
        primal_prox_inf[b] = delta[b] * max_i ( |result.duals_all[b, i] - prox.duals_all[b, i]| )
        primal_res_reg[b]  = max_j ( |res.duals_all[b, j]|                 * primal_res_unscale_factor[b, j] )

    Stage 3 -- scalar finalize (thread 0 only). ``delta`` / ``rho`` here
    are again the IPM proximal scalars, not preconditioner state:

        dual_res_reg_rel[b]   = dual_res_reg[b]   * dual_res_rel[b]   / dual_res[b]   if dual_res_rel[b]   > 0 else dual_res_reg[b]
        primal_res_reg_rel[b] = primal_res_reg[b] * primal_res_rel[b] / primal_res[b] if primal_res_rel[b] > 0 else primal_res_reg[b]

    ``p`` / ``num_hu`` / ``num_hl`` / ``num_xu`` / ``num_xl`` are compile-time
    constants. ``n`` is always > 0; ``num_duals == 0`` (no eq, no ineq, no
    box) is handled by thread 0 writing zeros.

    Tile reuse: stage 1's ``diff_x``, ``new_res_x``, ``diff_d``, ``new_res_d``
    tiles are kept in registers and fed straight into stage 2's reductions —
    no extra global loads.

    Dispatch: ``wp.launch_tiled(..., dim=[B], block_dim=256)`` -- one block
    per batch.
    """
    num_duals = p + num_hu + num_hl + num_xu + num_xl

    @wp.func
    def _abs_mul(a: wp.float64, b: wp.float64) -> wp.float64:
        return wp.abs(a) * b

    @wp.kernel
    def update_residuals_r_kernel(
        # Stage 1 inputs
        rho:           wp.array(dtype=wp.float64),    # type: ignore  (B,)
        delta:         wp.array(dtype=wp.float64),    # type: ignore  (B,)
        res_nr_x:      wp.array2d(dtype=wp.float64),  # type: ignore  (B, n)
        res_nr_duals:   wp.array2d(dtype=wp.float64), # type: ignore  (B, num_duals)
        result_x:      wp.array2d(dtype=wp.float64),  # type: ignore  (B, n)
        result_duals:  wp.array2d(dtype=wp.float64),  # type: ignore  (B, num_duals)
        prox_x:        wp.array2d(dtype=wp.float64),  # type: ignore  (B, n)
        prox_duals:    wp.array2d(dtype=wp.float64),  # type: ignore  (B, num_duals)
        # Stage 1 outputs
        res_x:         wp.array2d(dtype=wp.float64),  # type: ignore  (B, n)
        res_duals:     wp.array2d(dtype=wp.float64),  # type: ignore  (B, num_duals)
        # Pre-combined residual unscaling factors (from preconditioner)
        dual_res_unscale_factor:   wp.array2d(dtype=wp.float64),  # type: ignore  (B, n)
        primal_res_unscale_factor: wp.array2d(dtype=wp.float64),  # type: ignore  (B, num_duals)
        # Stage 3 scalar inputs
        primal_res:     wp.array(dtype=wp.float64),  # type: ignore  (B,)
        primal_res_rel: wp.array(dtype=wp.float64),  # type: ignore  (B,)
        dual_res:       wp.array(dtype=wp.float64),  # type: ignore  (B,)
        dual_res_rel:   wp.array(dtype=wp.float64),  # type: ignore  (B,)
        # Outputs
        primal_res_reg:     wp.array(dtype=wp.float64),  # type: ignore
        primal_res_reg_rel: wp.array(dtype=wp.float64),  # type: ignore
        dual_res_reg:       wp.array(dtype=wp.float64),  # type: ignore
        dual_res_reg_rel:   wp.array(dtype=wp.float64),  # type: ignore
        primal_prox_inf:    wp.array(dtype=wp.float64),  # type: ignore
        dual_prox_inf:      wp.array(dtype=wp.float64),  # type: ignore
    ):
        b, i = wp.tid()

        # --- x-sized pipeline: stage 1 (build res.x) + stage 2 (dual_prox_inf, dual_res_reg) ---
        res_nr_x_tile = wp.tile_load(res_nr_x[b],  shape=n)
        result_x_tile = wp.tile_load(result_x[b],  shape=n)
        prox_x_tile   = wp.tile_load(prox_x[b],    shape=n)
        diff_x_tile   = result_x_tile - prox_x_tile

        # Stage 1: res.x = res_nr.x - rho * (result.x - prox.x)
        new_res_x_tile = res_nr_x_tile - rho[b] * diff_x_tile
        wp.tile_store(res_x[b], new_res_x_tile)

        # Stage 2: dual_prox_inf = rho[b] * max |diff_x|  (reusing the tile)
        mt = wp.tile_max(wp.tile_map(wp.abs, diff_x_tile))
        wp.tile_store(dual_prox_inf, mt * rho[b], offset=b)

        # Stage 2: dual_res_reg = max |new_res_x * dual_res_unscale_factor|
        scale_x_tile = wp.tile_load(dual_res_unscale_factor[b], shape=n)
        mt = wp.tile_max(wp.tile_map(_abs_mul, new_res_x_tile, scale_x_tile))
        wp.tile_store(dual_res_reg, mt, offset=b)

        # --- duals-sized pipeline: stage 1 (build res.duals) + stage 2 (primal_prox_inf, primal_res_reg) ---
        if wp.static(num_duals > 0):
            nr_d_tile  = wp.tile_load(res_nr_duals[b],   shape=num_duals)
            r_d_tile   = wp.tile_load(result_duals[b],  shape=num_duals)
            p_d_tile   = wp.tile_load(prox_duals[b],    shape=num_duals)
            diff_d_tile = r_d_tile - p_d_tile

            # Stage 1: res.duals = res_nr.duals + delta * (result.duals - prox.duals)
            new_res_d_tile = nr_d_tile + delta[b] * diff_d_tile
            wp.tile_store(res_duals[b], new_res_d_tile)

            # Stage 2: primal_prox_inf = delta[b] * max |diff_d|
            mt = wp.tile_max(wp.tile_map(wp.abs, diff_d_tile))
            wp.tile_store(primal_prox_inf, mt * delta[b], offset=b)

            # Stage 2: primal_res_reg = max |new_res_d * primal_res_unscale_factor|
            scale_d_tile = wp.tile_load(primal_res_unscale_factor[b], shape=num_duals)
            mt = wp.tile_max(wp.tile_map(_abs_mul, new_res_d_tile, scale_d_tile))
            wp.tile_store(primal_res_reg, mt, offset=b)
        else:
            if i == 0:
                primal_prox_inf[b] = wp.float64(0.0)
                primal_res_reg[b]  = wp.float64(0.0)

        # --- Stage 3: scalar finalize --------------------------------------
        if i == 0:
            if primal_res_rel[b] > wp.float64(0.0):
                primal_res_reg_rel[b] = primal_res_reg[b] * primal_res_rel[b] / primal_res[b]
            else:
                primal_res_reg_rel[b] = primal_res_reg[b]

            if dual_res_rel[b] > wp.float64(0.0):
                dual_res_reg_rel[b] = dual_res_reg[b] * dual_res_rel[b] / dual_res[b]
            else:
                dual_res_reg_rel[b] = dual_res_reg[b]

    return update_residuals_r_kernel


def create_update_residual_nr_kernel(n: int, p: int, m: int, num_hl: int, num_hu: int, num_xl: int, num_xu: int):
    r"""Fused kernel to update non-regularized residuals

    - ``minus_Px``      = ``-P*x``                             from ``eval_P_x(alpha=-1)``  (also used as the in-place ``res_nr.x`` slot)
    - ``A_x``           = ``+A*x``                             from ``eval_A_xn(alpha=+1)`` (note: NOT aliased with ``res_nr.y``; see solver wiring)
    - ``AT_y``          = ``A^T * y``                          from ``eval_AT_xt``
    - ``G_x``           = ``G*x``                              from ``eval_G_xn``
    - ``GT_zh_assembled`` = ``G^T * (z_u - z_l)``              from ``eval_GT_xt`` (input to that matvec is built by the pre-matvec scatter into ``zu_minus_zl``)
    - ``zb_assembled``  = ``x_b_scaling * (z_bu - z_bl)``      from the pre-matvec scatter (``prepare_zu_minus_zl_and_zbu_minus_zbl_kernel``)

    Compute:

        res_nr.x      = -(P*x + c + A^T*y + G^T*(z_u - z_l) + x_b_scaling*(z_bu - z_bl))
        res_nr.y      = -A*x + b
        res_nr.z_hl   =  G*x[idx_hl] - s_l - h_l[idx_hl]
        res_nr.z_hu   = -G*x[idx_hu] - s_u + h_u[idx_hu]
        res_nr.z_xl   =  x_b_scaling[idx_xl]*x[idx_xl] - s_bl - x_l[idx_xl]
        res_nr.z_xu   = -(x_b_scaling[idx_xu]*x[idx_xu] + s_bu - x_u[idx_xu])

        primal_obj    = (0.5 x^T P x + c^T x) * cost_scaling_inv
        dual_obj      = -(0.5 x^T P x + b^T y + h_u^T z_u - h_l^T z_l + x_u^T z_bu - x_l^T z_bl) * cost_scaling_inv
        duality_gap   = |primal_obj - dual_obj|
        duality_gap_rel = duality_gap / max(1, max_k(cost_scaling_inv * |w_k|))   for w_k in the 7 unique obj-sum terms

        primal_res    = max over the 5 segments of  ||delta_inv_seg * res_nr_seg||_inf  (unscaled)
        primal_norm    = max(||A*x||, ||G*x[hu]||, ||G*x[hl]||,
                            ||s_u||,  ||s_l||,  ||s_bu||, ||s_bl||,
                            constraints_rhs_inf_norm[b])    -- all unscaled
        primal_res_rel = primal_res / max(1, primal_norm)

        dual_res      = ||delta_x_inv * res_nr.x||_inf * cost_scaling_inv
        dual_norm     = max(||P*x||, ||c||, ||A^T*y + G^T*(z_u-z_l) + x_b_scaling*(z_bu-z_bl)||) * delta_x_inv * cost_scaling_inv
        dual_res_rel  = dual_res / max(1, dual_norm)
    """
    @wp.kernel
    def update_residual_nr_kernel(
        minus_Px:                   wp.array2d(dtype=wp.float64),  # type: ignore  (B, n)
        A_x:                        wp.array2d(dtype=wp.float64),  # type: ignore  (B, p)
        AT_y:                       wp.array2d(dtype=wp.float64),  # type: ignore  (B, n)
        G_x:                        wp.array2d(dtype=wp.float64),  # type: ignore  (B, m)
        GT_zh_assembled:            wp.array2d(dtype=wp.float64),  # type: ignore  (B, n)
        zb_assembled:               wp.array2d(dtype=wp.float64),  # type: ignore  (B, n)
        # Data
        data_c:                     wp.array2d(dtype=wp.float64),  # type: ignore  (B, n)
        data_b:                     wp.array2d(dtype=wp.float64),  # type: ignore  (B, p)
        data_h_l:                   wp.array2d(dtype=wp.float64),  # type: ignore  (B, m)
        data_h_u:                   wp.array2d(dtype=wp.float64),  # type: ignore  (B, m)
        data_x_l:                   wp.array2d(dtype=wp.float64),  # type: ignore  (B, n)
        data_x_u:                   wp.array2d(dtype=wp.float64),  # type: ignore  (B, n)
        # Variables at current iteration
        result_x:                   wp.array2d(dtype=wp.float64),  # type: ignore  (B, n)
        result_y:                   wp.array2d(dtype=wp.float64),  # type: ignore  (B, p)
        result_z_hl:                wp.array2d(dtype=wp.float64),  # type: ignore  (B, num_hl)
        result_z_hu:                wp.array2d(dtype=wp.float64),  # type: ignore  (B, num_hu)
        result_z_xl:                wp.array2d(dtype=wp.float64),  # type: ignore  (B, num_xl)
        result_z_xu:                wp.array2d(dtype=wp.float64),  # type: ignore  (B, num_xu)
        result_s_hl:                wp.array2d(dtype=wp.float64),  # type: ignore  (B, num_hl)
        result_s_hu:                wp.array2d(dtype=wp.float64),  # type: ignore  (B, num_hu)
        result_s_xl:                wp.array2d(dtype=wp.float64),  # type: ignore  (B, num_xl)
        result_s_xu:                wp.array2d(dtype=wp.float64),  # type: ignore  (B, num_xu)
        # Preconditioner
        x_b_scaling:                wp.array2d(dtype=wp.float64),  # type: ignore  (B, n)
        cost_scaling_inv:           wp.array(dtype=wp.float64),    # type: ignore  (B,)
        delta_inv:                  wp.array2d(dtype=wp.float64),  # type: ignore  (B, n+p+m)
        delta_b_inv:                wp.array2d(dtype=wp.float64),  # type: ignore  (B, n)
        constraints_rhs_inf_norm:   wp.array(dtype=wp.float64),    # type: ignore  (B,)
        # Segment index maps
        idx_hl:                     wp.array(dtype=wp.int32),      # type: ignore
        idx_hu:                     wp.array(dtype=wp.int32),      # type: ignore
        idx_xl:                     wp.array(dtype=wp.int32),      # type: ignore
        idx_xu:                     wp.array(dtype=wp.int32),      # type: ignore
        # Residuals
        res_nr_x:                   wp.array2d(dtype=wp.float64),  # type: ignore  (B, n)
        res_nr_y:                   wp.array2d(dtype=wp.float64),  # type: ignore  (B, p)
        res_nr_z_hl:                wp.array2d(dtype=wp.float64),  # type: ignore  (B, num_hl)
        res_nr_z_hu:                wp.array2d(dtype=wp.float64),  # type: ignore  (B, num_hu)
        res_nr_z_xl:                wp.array2d(dtype=wp.float64),  # type: ignore  (B, num_xl)
        res_nr_z_xu:                wp.array2d(dtype=wp.float64),  # type: ignore  (B, num_xu)
        # Objectices and residuals
        info_primal_obj:            wp.array(dtype=wp.float64),  # type: ignore  shape (B,)
        info_dual_obj:              wp.array(dtype=wp.float64),  # type: ignore  shape (B,)
        info_duality_gap:           wp.array(dtype=wp.float64),  # type: ignore  shape (B,)
        info_duality_gap_rel:       wp.array(dtype=wp.float64),  # type: ignore  shape (B,)
        info_primal_res:            wp.array(dtype=wp.float64),  # type: ignore  shape (B,)
        info_primal_res_rel:        wp.array(dtype=wp.float64),  # type: ignore  shape (B,)
        info_dual_res:              wp.array(dtype=wp.float64),  # type: ignore  shape (B,)
        info_dual_res_rel:          wp.array(dtype=wp.float64),  # type: ignore  shape (B,)
        info_prev_primal_res:       wp.array(dtype=wp.float64),  # type: ignore  shape (B,)
        info_prev_dual_res:         wp.array(dtype=wp.float64),  # type: ignore  shape (B,)
    ):
        b, i = wp.tid()

        if i == 0:
            info_prev_primal_res[b] = info_primal_res[b]
            info_prev_dual_res[b]   = info_dual_res[b]

        minus_Px_tile        = wp.tile_load(minus_Px[b],        shape=n)
        x_tile               = wp.tile_load(result_x[b],        shape=n)
        c_tile               = wp.tile_load(data_c[b],          shape=n)
        AT_y_tile            = wp.tile_load(AT_y[b],            shape=n)
        GT_zh_assembled_tile = wp.tile_load(GT_zh_assembled[b], shape=n)
        zb_assembled_tile    = wp.tile_load(zb_assembled[b],    shape=n)
        delta_x_inv_tile     = wp.tile_load(delta_inv[b],       shape=n, offset=0)

        # ---- res_nr.x = -P*x - c - AT*y - GT*(z_u-z_l) - x_b_scaling*(z_bu-z_bl) ----
        res_nr_x_tile = minus_Px_tile - c_tile - AT_y_tile - GT_zh_assembled_tile - zb_assembled_tile
        wp.tile_store(res_nr_x[b], res_nr_x_tile)

        # ---- info.dual_res = cost_scaling_inv * ||delta_x_inv * res_nr.x||_inf ----
        dual_res = wp.tile_extract(
            wp.tile_max(wp.tile_map(wp.abs, delta_x_inv_tile * res_nr_x_tile)), 0,
        ) * cost_scaling_inv[b]
        if i == 0:
            info_dual_res[b] = dual_res

        half_xT_Px = wp.tile_extract(
            wp.float64(-0.5) * wp.tile_sum(minus_Px_tile * x_tile), 0,
        )
        cT_x = wp.tile_extract(wp.tile_sum(c_tile * x_tile), 0)

        # ---- segment obj-sum scalars (default 0; overridden inside guards) ----
        bT_y    = wp.float64(0.0)
        hl_zhl  = wp.float64(0.0)
        hu_zhu  = wp.float64(0.0)
        xl_zxl  = wp.float64(0.0)
        xu_zxu  = wp.float64(0.0)

        # ---- primal_res / primal_rel_norm running maxes (scalar) ----
        primal_res  = wp.float64(0.0)
        primal_res_rel_norm = wp.float64(0.0)

        # ---- dual_res_rel: 3-term running max ----
        Px_norm = wp.tile_extract(wp.tile_max(wp.tile_map(wp.abs, minus_Px_tile * delta_x_inv_tile)), 0)
        c_norm = wp.tile_extract(wp.tile_max(wp.tile_map(wp.abs, c_tile * delta_x_inv_tile)), 0)
        accum_x_tile = AT_y_tile + GT_zh_assembled_tile + zb_assembled_tile
        accum_norm = wp.tile_extract(wp.tile_max(wp.tile_map(wp.abs, accum_x_tile * delta_x_inv_tile)), 0)
        drn_max = wp.max(Px_norm, wp.max(c_norm, accum_norm))

        # ===================== y-segment =====================
        if wp.static(p > 0):
            delta_y_inv_tile = wp.tile_load(delta_inv[b], shape=p, offset=n)

            b_tile   = wp.tile_load(data_b[b], shape=p)
            y_tile   = wp.tile_load(result_y[b], shape=p)
            A_x_tile = wp.tile_load(A_x[b], shape=p)

            # b^T y obj-sum
            bT_y = wp.tile_extract(wp.tile_sum(b_tile * y_tile), 0)

            # res_nr.y = -A*x + b
            res_nr_y_tile = -A_x_tile + b_tile
            wp.tile_store(res_nr_y[b], res_nr_y_tile)

            # ||res_nr.y||_inf -> primal_res
            res_nr_y_norm = wp.tile_extract(wp.tile_max(wp.tile_map(wp.abs, res_nr_y_tile * delta_y_inv_tile)), 0)
            primal_res = wp.max(primal_res, res_nr_y_norm)

            # ||A*x||_inf -> primal_rel_norm
            A_x_norm = wp.tile_extract(wp.tile_max(wp.tile_map(wp.abs, A_x_tile * delta_y_inv_tile)), 0)
            primal_res_rel_norm = wp.max(primal_res_rel_norm, A_x_norm)

        # ===================== z_l (hl) segment =====================
        if wp.static(num_hl > 0):
            idx_hl_tile = wp.tile_load(idx_hl, shape=num_hl)
            delta_z_hl_inv_tile = wp.tile_load_indexed(
                delta_inv[b], indices=idx_hl_tile, shape=(num_hl,), offset=(n + p,), axis=0,
            )

            # h_l^T z_l obj-sum
            hl_tile  = wp.tile_load_indexed(data_h_l[b], indices=idx_hl_tile, shape=(num_hl,), axis=0)
            zhl_tile = wp.tile_load(result_z_hl[b], shape=num_hl)
            hl_zhl = wp.tile_extract(wp.tile_sum(hl_tile * zhl_tile), 0)

            # res_nr.z_l = G*x[idx_hl] - s_l - h_l[idx_hl]
            Gx_idx_hl_tile = wp.tile_load_indexed(G_x[b], indices=idx_hl_tile, shape=(num_hl,), axis=0)
            s_hl_tile = wp.tile_load(result_s_hl[b], shape=num_hl)
            res_nr_z_hl_tile = Gx_idx_hl_tile - s_hl_tile - hl_tile
            wp.tile_store(res_nr_z_hl[b], res_nr_z_hl_tile)

            # primal_res
            hl_res_norm = wp.tile_extract(
                wp.tile_max(wp.tile_map(wp.abs, res_nr_z_hl_tile * delta_z_hl_inv_tile)), 0,
            )
            primal_res = wp.max(primal_res, hl_res_norm)

            # primal_rel_norm: ||G*x[hl]||, ||s_l||
            gx_hl_norm = wp.tile_extract(
                wp.tile_max(wp.tile_map(wp.abs, Gx_idx_hl_tile * delta_z_hl_inv_tile)), 0,
            )
            s_hl_norm = wp.tile_extract(
                wp.tile_max(wp.tile_map(wp.abs, s_hl_tile * delta_z_hl_inv_tile)), 0,
            )
            primal_res_rel_norm = wp.max(primal_res_rel_norm, wp.max(gx_hl_norm, s_hl_norm))

        # ===================== z_u (hu) segment =====================
        if wp.static(num_hu > 0):
            idx_hu_tile = wp.tile_load(idx_hu, shape=num_hu)
            delta_z_hu_inv_tile = wp.tile_load_indexed(
                delta_inv[b], indices=idx_hu_tile, shape=(num_hu,), offset=(n + p,), axis=0,
            )

            hu_tile  = wp.tile_load_indexed(data_h_u[b], indices=idx_hu_tile, shape=(num_hu,), axis=0)
            zhu_tile = wp.tile_load(result_z_hu[b], shape=num_hu)
            hu_zhu = wp.tile_extract(wp.tile_sum(hu_tile * zhu_tile), 0)

            # res_nr.z_u = -G*x[idx_hu] - s_u + h_u[idx_hu]
            Gx_idx_hu_tile = wp.tile_load_indexed(G_x[b], indices=idx_hu_tile, shape=(num_hu,), axis=0)
            s_hu_tile = wp.tile_load(result_s_hu[b], shape=num_hu)
            res_nr_z_hu_tile = -Gx_idx_hu_tile - s_hu_tile + hu_tile
            wp.tile_store(res_nr_z_hu[b], res_nr_z_hu_tile)

            hu_res_norm = wp.tile_extract(
                wp.tile_max(wp.tile_map(wp.abs, res_nr_z_hu_tile * delta_z_hu_inv_tile)), 0,
            )
            primal_res = wp.max(primal_res, hu_res_norm)

            gx_hu_norm = wp.tile_extract(
                wp.tile_max(wp.tile_map(wp.abs, Gx_idx_hu_tile * delta_z_hu_inv_tile)), 0,
            )
            s_hu_norm = wp.tile_extract(
                wp.tile_max(wp.tile_map(wp.abs, s_hu_tile * delta_z_hu_inv_tile)), 0,
            )
            primal_res_rel_norm = wp.max(primal_res_rel_norm, wp.max(gx_hu_norm, s_hu_norm))

        # ===================== z_bl (xl) segment =====================
        if wp.static(num_xl > 0):
            idx_xl_tile = wp.tile_load(idx_xl, shape=num_xl)
            delta_b_xl_inv_tile = wp.tile_load_indexed(
                delta_b_inv[b], indices=idx_xl_tile, shape=(num_xl,), axis=0,
            )

            xl_tile  = wp.tile_load_indexed(data_x_l[b], indices=idx_xl_tile, shape=(num_xl,), axis=0)
            zxl_tile = wp.tile_load(result_z_xl[b], shape=num_xl)
            xl_zxl = wp.tile_extract(wp.tile_sum(xl_tile * zxl_tile), 0)

            # res_nr.z_bl = x_b_scaling[idx_xl]*x[idx_xl] - s_bl - x_l[idx_xl]
            x_idx_xl_tile = wp.tile_load_indexed(result_x[b], indices=idx_xl_tile, shape=(num_xl,), axis=0)
            x_b_scaling_idx_xl_tile = wp.tile_load_indexed(
                x_b_scaling[b], indices=idx_xl_tile, shape=(num_xl,), axis=0,
            )
            s_xl_tile = wp.tile_load(result_s_xl[b], shape=num_xl)
            res_nr_z_xl_tile = x_b_scaling_idx_xl_tile * x_idx_xl_tile - s_xl_tile - xl_tile
            wp.tile_store(res_nr_z_xl[b], res_nr_z_xl_tile)

            xl_res_norm = wp.tile_extract(
                wp.tile_max(wp.tile_map(wp.abs, res_nr_z_xl_tile * delta_b_xl_inv_tile)), 0,
            )
            primal_res = wp.max(primal_res, xl_res_norm)

            # primal_rel_norm: ||s_bl||
            s_xl_norm = wp.tile_extract(
                wp.tile_max(wp.tile_map(wp.abs, s_xl_tile * delta_b_xl_inv_tile)), 0,
            )
            primal_res_rel_norm = wp.max(primal_res_rel_norm, s_xl_norm)

        # ===================== z_bu (xu) segment =====================
        if wp.static(num_xu > 0):
            idx_xu_tile = wp.tile_load(idx_xu, shape=num_xu)
            delta_b_xu_inv_tile = wp.tile_load_indexed(
                delta_b_inv[b], indices=idx_xu_tile, shape=(num_xu,), axis=0,
            )

            xu_tile  = wp.tile_load_indexed(data_x_u[b], indices=idx_xu_tile, shape=(num_xu,), axis=0)
            zxu_tile = wp.tile_load(result_z_xu[b], shape=num_xu)
            xu_zxu = wp.tile_extract(wp.tile_sum(xu_tile * zxu_tile), 0)

            # res_nr.z_bu = -(x_b_scaling[idx_xu]*x[idx_xu] + s_bu - x_u[idx_xu])
            x_idx_xu_tile = wp.tile_load_indexed(result_x[b], indices=idx_xu_tile, shape=(num_xu,), axis=0)
            x_b_scaling_idx_xu_tile = wp.tile_load_indexed(
                x_b_scaling[b], indices=idx_xu_tile, shape=(num_xu,), axis=0,
            )
            s_xu_tile = wp.tile_load(result_s_xu[b], shape=num_xu)
            res_nr_z_xu_tile = -x_b_scaling_idx_xu_tile * x_idx_xu_tile - s_xu_tile + xu_tile
            wp.tile_store(res_nr_z_xu[b], res_nr_z_xu_tile)

            xu_res_norm = wp.tile_extract(
                wp.tile_max(wp.tile_map(wp.abs, res_nr_z_xu_tile * delta_b_xu_inv_tile)), 0,
            )
            primal_res = wp.max(primal_res, xu_res_norm)

            s_xu_norm = wp.tile_extract(
                wp.tile_max(wp.tile_map(wp.abs, s_xu_tile * delta_b_xu_inv_tile)), 0,
            )
            primal_res_rel_norm = wp.max(primal_res_rel_norm, s_xu_norm)

        # ===================== finalize info =====================
        if i == 0:
            # primal/dual obj + duality_gap
            info_primal_obj[b] = cost_scaling_inv[b] * (half_xT_Px + cT_x)
            info_dual_obj[b] = cost_scaling_inv[b] * (-half_xT_Px - bT_y - hu_zhu + hl_zhl - xu_zxu + xl_zxl)
            info_duality_gap[b] = wp.abs(info_primal_obj[b] - info_dual_obj[b])

            # duality_gap_rel: cost_scaling_inv * max(|0.5xPx|, |cTx|, |bTy|, |hl_zhl|, |hu_zhu|, |xl_zxl|, |xu_zxu|)
            duality_gap_rel_norm = wp.abs(half_xT_Px)
            duality_gap_rel_norm = wp.max(duality_gap_rel_norm, wp.abs(cT_x))
            duality_gap_rel_norm = wp.max(duality_gap_rel_norm, wp.abs(bT_y))
            duality_gap_rel_norm = wp.max(duality_gap_rel_norm, wp.abs(hl_zhl))
            duality_gap_rel_norm = wp.max(duality_gap_rel_norm, wp.abs(hu_zhu))
            duality_gap_rel_norm = wp.max(duality_gap_rel_norm, wp.abs(xl_zxl))
            duality_gap_rel_norm = wp.max(duality_gap_rel_norm, wp.abs(xu_zxu))
            duality_gap_rel_norm = wp.max(cost_scaling_inv[b] * duality_gap_rel_norm, wp.float64(1.0))
            info_duality_gap_rel[b] = info_duality_gap[b] / duality_gap_rel_norm

            # primal_res / primal_res_rel
            info_primal_res[b] = primal_res
            prn = wp.max(primal_res_rel_norm, constraints_rhs_inf_norm[b])
            prn = wp.max(prn, wp.float64(1.0))
            info_primal_res_rel[b] = primal_res / prn

            # dual_res_rel  (info.dual_res already = dr_x)
            dual_res_rel_norm = wp.max(drn_max * cost_scaling_inv[b], wp.float64(1.0))
            info_dual_res_rel[b] = dual_res / dual_res_rel_norm

    return update_residual_nr_kernel


def create_prepare_zu_minus_zl_and_zbu_minus_zbl_kernel(m: int, n: int):
    """Pre-matvec scatter-gather for _update_residuals_nr.

          zhu_minus_zhl = 0
          zhu_minus_zhl[:, idx_hu] += z_hu
          zhu_minus_zhl[:, idx_hl] -= z_hl

          zbu_minus_zbl = 0
          zbu_minus_zbl[:, idx_xu] += z_xu
          zbu_minus_zbl[:, idx_xl] -= z_xl
          zbu_minus_zbl *=  x_b_scaling[b, i]
    """

    @wp.kernel
    def prepare_zu_minus_zl_and_zbu_minus_zbl_kernel(
        z_u:              wp.array2d(dtype=wp.float64),  # type: ignore  (B, num_hu)
        z_l:              wp.array2d(dtype=wp.float64),  # type: ignore  (B, num_hl)
        z_bl:             wp.array2d(dtype=wp.float64),  # type: ignore  (B, num_xl)
        z_bu:             wp.array2d(dtype=wp.float64),  # type: ignore  (B, num_xu)
        x_b_scaling:      wp.array2d(dtype=wp.float64),  # type: ignore  (B, n)
        inv_idx_hu:       wp.array(dtype=wp.int32),      # type: ignore  (m,)
        inv_idx_hl:       wp.array(dtype=wp.int32),      # type: ignore  (m,)
        inv_idx_xu:       wp.array(dtype=wp.int32),      # type: ignore  (n,)
        inv_idx_xl:       wp.array(dtype=wp.int32),      # type: ignore  (n,)
        zu_minus_zl:      wp.array2d(dtype=wp.float64),  # type: ignore  (B, m) output
        zbu_minus_zbl:    wp.array2d(dtype=wp.float64),  # type: ignore  (B, n) output
    ):
        m_static = wp.static(m)
        b, i = wp.tid()

        if i < m_static:
            val = wp.float64(0.0)
            k_hu = inv_idx_hu[i]
            if k_hu >= 0:
                val = val + z_u[b, k_hu]
            k_hl = inv_idx_hl[i]
            if k_hl >= 0:
                val = val - z_l[b, k_hl]
            zu_minus_zl[b, i] = val
        else:
            j = i - m_static
            val = wp.float64(0.0)
            k_xu = inv_idx_xu[j]
            if k_xu >= 0:
                val = val + z_bu[b, k_xu]
            k_xl = inv_idx_xl[j]
            if k_xl >= 0:
                val = val - z_bl[b, k_xl]
            zbu_minus_zbl[b, j] = x_b_scaling[b, j] * val

    return prepare_zu_minus_zl_and_zbu_minus_zbl_kernel


def create_update_rho_delta_with_ineq_kernel(n: int, num_duals: int):
    """Fused adaptive-regularization update for the inequality-constrained path.
    """
    @wp.kernel
    def update_rho_delta_with_ineq_kernel(
        info_dual_res:         wp.array(dtype=wp.float64),   # type: ignore
        info_prev_dual_res:    wp.array(dtype=wp.float64),   # type: ignore
        info_dual_res_rel:     wp.array(dtype=wp.float64),   # type: ignore
        info_dual_prox_inf:    wp.array(dtype=wp.float64),   # type: ignore
        info_primal_res:       wp.array(dtype=wp.float64),   # type: ignore
        info_prev_primal_res:  wp.array(dtype=wp.float64),   # type: ignore
        info_primal_res_rel:   wp.array(dtype=wp.float64),   # type: ignore
        info_primal_prox_inf:  wp.array(dtype=wp.float64),   # type: ignore
        info_reg_limit:        wp.array(dtype=wp.float64),   # type: ignore
        info_rho:              wp.array(dtype=wp.float64),   # type: ignore
        info_delta:            wp.array(dtype=wp.float64),   # type: ignore
        info_no_primal_update: wp.array(dtype=wp.int32),     # type: ignore  in-out (B,)
        info_no_dual_update:   wp.array(dtype=wp.int32),     # type: ignore  in-out (B,)
        result_x:              wp.array2d(dtype=wp.float64), # type: ignore
        prox_x:                wp.array2d(dtype=wp.float64), # type: ignore
        result_duals:          wp.array2d(dtype=wp.float64), # type: ignore
        prox_duals:            wp.array2d(dtype=wp.float64), # type: ignore
        settings_eps_abs:              wp.float64,
        settings_eps_rel:              wp.float64,
        settings_reg_finetune_lower:   wp.float64,
        settings_infeas_thresh:        wp.float64,
        current_iter:                  wp.int32,
    ):
        b, i = wp.tid()
        n_static = wp.static(n)
        num_duals_static = wp.static(num_duals)
        iter_under_5 = (current_iter < wp.int32(5))

        dual_improved = (
            (info_dual_res[b] < wp.float64(0.95) * info_prev_dual_res[b])
            or (info_dual_res[b] < settings_eps_abs)
            or (info_dual_res_rel[b] < settings_eps_rel)
            or ((info_rho[b] == settings_reg_finetune_lower) and (info_dual_prox_inf[b] < settings_infeas_thresh))
        )
        primal_improved = (
            (info_primal_res[b] < wp.float64(0.95) * info_prev_primal_res[b])
            or (info_primal_res[b] < settings_eps_abs)
            or (info_primal_res_rel[b] < settings_eps_rel)
            or ((info_delta[b] == settings_reg_finetune_lower) and (info_primal_prox_inf[b] < settings_infeas_thresh))
        )

        if i == 0:
            old_rho = info_rho[b]
            rho_fast = wp.max(info_reg_limit[b], wp.float64(0.1) * old_rho)
            rho_slow = wp.max(info_reg_limit[b], wp.float64(0.5) * old_rho)
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
            delta_fast = wp.max(info_reg_limit[b], wp.float64(0.1) * old_delta)
            delta_slow = wp.max(info_reg_limit[b], wp.float64(0.5) * old_delta)
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

        if i < n_static:
            prox_x[b, i] = wp.where(dual_improved, result_x[b, i], prox_x[b, i])
        elif i < n_static + num_duals_static:
            t = i - n_static
            prox_duals[b, t] = wp.where(primal_improved, result_duals[b, t], prox_duals[b, t])

    return update_rho_delta_with_ineq_kernel


def create_update_rho_delta_without_ineq_kernel(n: int, p: int):
    """Fused adaptive-regularization update for the equality-only path.
    """
    @wp.kernel
    def update_rho_delta_without_ineq_kernel(
        info_dual_res:         wp.array(dtype=wp.float64),   # type: ignore
        info_prev_dual_res:    wp.array(dtype=wp.float64),   # type: ignore
        info_dual_res_rel:     wp.array(dtype=wp.float64),   # type: ignore
        info_dual_prox_inf:    wp.array(dtype=wp.float64),   # type: ignore
        info_primal_res:       wp.array(dtype=wp.float64),   # type: ignore
        info_prev_primal_res:  wp.array(dtype=wp.float64),   # type: ignore
        info_primal_res_rel:   wp.array(dtype=wp.float64),   # type: ignore
        info_primal_prox_inf:  wp.array(dtype=wp.float64),   # type: ignore
        info_reg_limit:        wp.array(dtype=wp.float64),   # type: ignore
        info_rho:              wp.array(dtype=wp.float64),   # type: ignore
        info_delta:            wp.array(dtype=wp.float64),   # type: ignore
        info_no_primal_update: wp.array(dtype=wp.int32),     # type: ignore
        info_no_dual_update:   wp.array(dtype=wp.int32),     # type: ignore
        result_x:              wp.array2d(dtype=wp.float64), # type: ignore
        prox_x:                wp.array2d(dtype=wp.float64), # type: ignore
        result_y:              wp.array2d(dtype=wp.float64), # type: ignore
        prox_y:                wp.array2d(dtype=wp.float64), # type: ignore
        settings_eps_abs:              wp.float64,
        settings_eps_rel:              wp.float64,
        settings_infeas_thresh:        wp.float64,
        current_iter:                  wp.int32,
    ):
        b, i = wp.tid()
        n_static = wp.static(n)
        p_static = wp.static(p)
        iter_under_5 = (current_iter < wp.int32(5))

        dual_improved = (
            (info_dual_res[b] < wp.float64(0.95) * info_prev_dual_res[b])
            or (info_dual_res[b] < settings_eps_abs)
            or (info_dual_res_rel[b] < settings_eps_rel)
        )
        primal_improved = (
            (info_primal_res[b] < wp.float64(0.95) * info_prev_primal_res[b])
            or (info_primal_res[b] < settings_eps_abs)
            or (info_primal_res_rel[b] < settings_eps_rel)
        )

        if i == 0:
            old_rho = info_rho[b]
            rho_fast = wp.max(info_reg_limit[b], wp.float64(0.1) * old_rho)
            rho_slow = wp.max(info_reg_limit[b], wp.float64(0.5) * old_rho)
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
            delta_fast = wp.max(info_reg_limit[b], wp.float64(0.1) * old_delta)
            delta_slow = wp.max(info_reg_limit[b], wp.float64(0.5) * old_delta)
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


def create_boundary_shift_kernel(num_hl: int, num_hu: int, num_xl: int, num_xu: int):
    """Per-element ``z`` boundary shift to avoid division-by-zero in the IPM.

        if z_hl[b, k] < eps: z_hl[b, k] += eps    (k in [0, num_hl))
        if z_hu[b, k] < eps: z_hu[b, k] += eps    (k in [0, num_hu))
        if z_bl[b, k] < eps: z_bl[b, k] += eps    (k in [0, num_xl))
        if z_bu[b, k] < eps: z_bu[b, k] += eps    (k in [0, num_xu))
    """
    # IEEE 754 float64 machine epsilon (== np.finfo(np.float64).eps)
    EPS_F64 = wp.constant(wp.float64(2.220446049250313e-16))
    
    @wp.kernel
    def boundary_shift_kernel(
        z_hl:  wp.array2d(dtype=wp.float64),  # type: ignore  (B, num_hl)
        z_hu:  wp.array2d(dtype=wp.float64),  # type: ignore  (B, num_hu)
        z_bl: wp.array2d(dtype=wp.float64),   # type: ignore  (B, num_xl)
        z_bu: wp.array2d(dtype=wp.float64),   # type: ignore  (B, num_xu)
    ):
        b, t = wp.tid()
        n_hl = wp.static(num_hl)
        n_hu = wp.static(num_hu)
        n_xl = wp.static(num_xl)
        n_xu = wp.static(num_xu)

        if t < n_hl:
            if z_hl[b, t] < EPS_F64:
                z_hl[b, t] = z_hl[b, t] + EPS_F64
        elif t < n_hl + n_hu:
            i = t - n_hl
            if z_hu[b, i] < EPS_F64:
                z_hu[b, i] = z_hu[b, i] + EPS_F64
        elif t < n_hl + n_hu + n_xl:
            i = t - n_hl - n_hu
            if z_bl[b, i] < EPS_F64:
                z_bl[b, i] = z_bl[b, i] + EPS_F64
        elif t < n_hl + n_hu + n_xl + n_xu:
            i = t - n_hl - n_hu - n_xl
            if z_bu[b, i] < EPS_F64:
                z_bu[b, i] = z_bu[b, i] + EPS_F64
        else:
            return

    return boundary_shift_kernel
