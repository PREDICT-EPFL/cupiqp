import warp as wp


@wp.func
def _ruiz_limit_scaling(d: wp.float64, min_scaling: wp.float64, max_scaling: wp.float64) -> wp.float64:
    if d < min_scaling:
        return wp.float64(1.0)
    if d > max_scaling:
        return max_scaling
    return d

def create_clamp_and_rsqrt_kernel(n: int, p: int, m: int,
                              min_scaling: float, max_scaling: float):
    """Fused the following cupy chain:

        delta_iter[delta_iter < self.min_scaling] = 1.0
        cp.minimum(delta_iter, self.max_scaling, out=delta_iter)
        cp.sqrt(d_iter, out=d_iter)
        cp.reciprocal(d_iter, out=d_iter)

    with one launch dispatched ``(B, n+p+m)``. Each thread also do the same for delta_b_iter
    ``delta_b_iter[b, k]`` if ``k < n``.
    """
    low = float(min_scaling)
    high = float(max_scaling)

    @wp.kernel
    def clamp_rsqrt_kernel(
        delta_iter:   wp.array2d(dtype=wp.float64),   # type: ignore  (B, n+p+m)
        delta_b_iter: wp.array2d(dtype=wp.float64),   # type: ignore  (B, n)
    ):
        b, k = wp.tid()
        delta = delta_iter[b, k]
        delta = _ruiz_limit_scaling(delta, wp.float64(low), wp.float64(high))
        delta_iter[b, k] = wp.float64(1.0) / wp.sqrt(delta)

        if k < wp.static(n):
            delta_b = delta_b_iter[b, k]
            delta_b = _ruiz_limit_scaling(delta_b, wp.float64(low), wp.float64(high))
            delta_b_iter[b, k] = wp.float64(1.0) / wp.sqrt(delta_b)

    return clamp_rsqrt_kernel


def create_calc_scaling_inv_and_scale_bounds_kernel(n: int, p: int, m: int):
    """Fuse the kernel for storing the inverse of scaling factors and scaling the bounds.

    Replaces the cupy/warp chain

        cp.reciprocal(delta,        out=delta_inv)
        cp.reciprocal(delta_b,      out=delta_b_inv)
        cp.reciprocal(cost_scaling, out=cost_scaling_inv)
        data._b    *= d_y                        # if p > 0
        data._h_l  *= d_z;  data._h_u *= d_z      # if m > 0
        data._x_l[:, idx_xl] *= delta_b[:, idx_xl]
        data._x_u[:, idx_xu] *= delta_b[:, idx_xu]

    with one launch dispatched ``(B, n+p+m+1)``.  Index k fans out:
      - k ∈ [0, n):            delta_inv[b, k]; delta_b_inv[b, k];
                               x_l[b, k] *= delta_b[b, k];
                               x_u[b, k] *= delta_b[b, k]
      - k ∈ [n, n+p):          delta_inv[b, k]; b_vec[b, k-n] *= delta[b, k]
      - k ∈ [n+p, n+p+m):      delta_inv[b, k]; h_l[b, k-n-p] *= delta[b, k];
                               h_u[b, k-n-p] *= delta[b, k]
      - k == n+p+m:            cost_scaling_inv[b]

    Unconditional scaling of ``x_l``/``x_u`` is safe: at unbounded indices,
    ``delta_b[b, i] == 1.0`` exactly (d_b_iter = rsqrt(1.0) = 1.0 every
    iteration), so the ±PIQP_INF sentinels are preserved bit-exactly.
    """
    @wp.kernel
    def finalize_and_scale_bounds_kernel(
        delta:            wp.array2d(dtype=wp.float64),  # type: ignore  (B, n+p+m)
        delta_inv:        wp.array2d(dtype=wp.float64),  # type: ignore  (B, n+p+m)
        delta_b:          wp.array2d(dtype=wp.float64),  # type: ignore  (B, n)
        delta_b_inv:      wp.array2d(dtype=wp.float64),  # type: ignore  (B, n)
        cost_scaling:     wp.array(dtype=wp.float64),    # type: ignore  (B,)
        cost_scaling_inv: wp.array(dtype=wp.float64),    # type: ignore  (B,)
        data_b:           wp.array2d(dtype=wp.float64),  # type: ignore  (B, p) — to be scaled in-place
        data_h_l:         wp.array2d(dtype=wp.float64),  # type: ignore  (B, m) — to be scaled in-place
        data_h_u:         wp.array2d(dtype=wp.float64),  # type: ignore  (B, m) — to be scaled in-place
        data_x_l:         wp.array2d(dtype=wp.float64),  # type: ignore  (B, n) — to be scaled in-place
        data_x_u:         wp.array2d(dtype=wp.float64),  # type: ignore  (B, n) — to be scaled in-place
    ):
        b, k = wp.tid()
        n_static = wp.static(n)
        np_static = wp.static(n + p)
        npm_static = wp.static(n + p + m)

        if k < n_static:
            delta_inv[b, k] = wp.float64(1.0) / delta[b, k]
            delta_b_inv[b, k] = wp.float64(1.0) / delta_b[b, k]
            # scale x_l and x_u inplace
            data_x_l[b, k] = data_x_l[b, k] * delta_b[b, k]
            data_x_u[b, k] = data_x_u[b, k] * delta_b[b, k]
        elif k < np_static:
            delta_inv[b, k] = wp.float64(1.0) / delta[b, k]
            # scale rhs of equality constraints inplace
            data_b[b, k - n_static] = data_b[b, k - n_static] * delta[b, k]
        elif k < npm_static:
            delta_inv[b, k] = wp.float64(1.0) / delta[b, k]
            # scale rhs of inequality constraints inplace
            jm = k - np_static
            data_h_l[b, jm] = data_h_l[b, jm] * delta[b, k]
            data_h_u[b, jm] = data_h_u[b, jm] * delta[b, k]
        elif k < npm_static + 1:
            # compute inverse of cost scaling
            cost_scaling_inv[b] = wp.float64(1.0) / cost_scaling[b]
        else:
            return

    return finalize_and_scale_bounds_kernel


def create_accumulate_deltas_kernel(n: int, p: int, m: int):
    """Fused per-iteration state update.

    Replaces the 4-launch cupy chain

        x_b_scaling *= delta_b_iter * d_x        # 2 launches
        delta       *= delta_iter                 # 1
        delta_b     *= delta_b_iter               # 1

    with one launch dispatched ``(B, n+p+m)``.  For ``k < n`` each thread
    also updates ``x_b_scaling`` and ``delta_b`` from ``delta_b_iter``.
    """
    @wp.kernel
    def accumulate_deltas_kernel(
        delta:        wp.array2d(dtype=wp.float64),  # type: ignore  (B, n+p+m) in-out
        delta_b:      wp.array2d(dtype=wp.float64),  # type: ignore  (B, n)     in-out
        x_b_scaling:  wp.array2d(dtype=wp.float64),  # type: ignore  (B, n)     in-out
        delta_iter:   wp.array2d(dtype=wp.float64),  # type: ignore  (B, n+p+m) input
        delta_b_iter: wp.array2d(dtype=wp.float64),  # type: ignore  (B, n)     input
    ):
        b, k = wp.tid()
        di = delta_iter[b, k]
        delta[b, k] = delta[b, k] * di
        if k < wp.static(n):
            dbi = delta_b_iter[b, k]
            delta_b[b, k] = delta_b[b, k] * dbi
            x_b_scaling[b, k] = x_b_scaling[b, k] * dbi * di

    return accumulate_deltas_kernel


@wp.func
def _diff_with_one(d: wp.float64) -> wp.float64:
    return wp.abs(wp.float64(1.0) - d)


def create_ruiz_conv_check_kernel(n: int, p: int, m: int):
    """Tile-based per-batch convergence reduction.

    Replaces the 4-launch cupy chain

        c1 = float(cp.max(cp.abs(1 - delta_iter)))   # 2 launches + sync
        c2 = float(cp.max(cp.abs(1 - delta_b_iter))) # 2 launches + sync
        conv = max(c1, c2)

    with one launch dispatched ``wp.launch_tiled(dim=[B])``. Writes per-batch
    max of ``|1 - delta_iter|`` into ``conv_buf[2*b]`` and per-batch max of
    ``|1 - delta_b_iter|`` into ``conv_buf[2*b + 1]``. Host then does one
    ``float(cp.max(conv_buf))`` + one sync.
    """
    total = n + p + m

    @wp.kernel
    def conv_check_kernel(
        delta_iter:   wp.array2d(dtype=wp.float64),   # type: ignore  (B, n+p+m)
        delta_b_iter: wp.array2d(dtype=wp.float64),   # type: ignore  (B, n)
        conv_buf:     wp.array(dtype=wp.float64),     # type: ignore  (2B,) output
    ):
        b, _ = wp.tid()

        d_tile = wp.tile_load(delta_iter[b], shape=total)
        d_abs = wp.tile_map(_diff_with_one, d_tile)
        max_d = wp.tile_max(d_abs)
        wp.tile_store(conv_buf, max_d, offset=2 * b)

        db_tile = wp.tile_load(delta_b_iter[b], shape=n)
        db_abs = wp.tile_map(_diff_with_one, db_tile)
        max_db = wp.tile_max(db_abs)
        wp.tile_store(conv_buf, max_db, offset=2 * b + 1)

    return conv_check_kernel


