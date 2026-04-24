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
