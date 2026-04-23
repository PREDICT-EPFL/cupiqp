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