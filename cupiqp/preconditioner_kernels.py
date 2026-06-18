import warp as wp
from .utils import to_warp_dtype


def create_clamp_and_rsqrt_kernel(n: int, p: int, m: int,
                              min_scaling: float, max_scaling: float, dtype=wp.float64):
    dtype = to_warp_dtype(dtype)
    @wp.func
    def _ruiz_limit_scaling(d: dtype, min_scaling: dtype, max_scaling: dtype) -> dtype:
        if d < min_scaling:
            return dtype(1.0)
        if d > max_scaling:
            return max_scaling
        return d

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
        delta_iter:   wp.array2d(dtype=dtype),   # type: ignore  (B, n+p+m)
        delta_b_iter: wp.array2d(dtype=dtype),   # type: ignore  (B, n)
    ):
        b, k = wp.tid()
        delta = delta_iter[b, k]
        delta = _ruiz_limit_scaling(delta, dtype(low), dtype(high))
        delta_iter[b, k] = dtype(1.0) / wp.sqrt(delta)

        if k < wp.static(n):
            delta_b = delta_b_iter[b, k]
            delta_b = _ruiz_limit_scaling(delta_b, dtype(low), dtype(high))
            delta_b_iter[b, k] = dtype(1.0) / wp.sqrt(delta_b)

    return clamp_rsqrt_kernel


def create_calc_scaling_inv_and_scale_bounds_kernel(
    n: int, p: int, m: int,
    num_hl: int, num_hu: int, num_xl: int, num_xu: int,
dtype=wp.float64):
    dtype = to_warp_dtype(dtype)
    """Fuse the kernel for storing the inverse of scaling factors and scaling the bounds.

    Replaces the cupy/warp chain

        cp.reciprocal(delta,        out=delta_inv)
        cp.reciprocal(delta_b,      out=delta_b_inv)
        cp.reciprocal(cost_scaling, out=cost_scaling_inv)
        data._b    *= d_y                        # if p > 0
        data._h_l  *= d_z;  data._h_u *= d_z      # if m > 0
        data._x_l *= delta_b
        data._x_u *= delta_b

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
        delta:            wp.array2d(dtype=dtype),  # type: ignore  (B, n+p+m)
        delta_inv:        wp.array2d(dtype=dtype),  # type: ignore  (B, n+p+m)
        delta_b:          wp.array2d(dtype=dtype),  # type: ignore  (B, n)
        delta_b_inv:      wp.array2d(dtype=dtype),  # type: ignore  (B, n)
        cost_scaling:     wp.array(dtype=dtype),    # type: ignore  (B,)
        cost_scaling_inv: wp.array(dtype=dtype),    # type: ignore  (B,)
        data_b:           wp.array2d(dtype=dtype),  # type: ignore  (B, p) — to be scaled in-place
        data_h_l:         wp.array2d(dtype=dtype),  # type: ignore  (B, m) — to be scaled in-place
        data_h_u:         wp.array2d(dtype=dtype),  # type: ignore  (B, m) — to be scaled in-place
        data_x_l:         wp.array2d(dtype=dtype),  # type: ignore  (B, n) — to be scaled in-place
        data_x_u:         wp.array2d(dtype=dtype),  # type: ignore  (B, n) — to be scaled in-place
        dual_res_unscale_factor:   wp.array2d(dtype=dtype),  # type: ignore  (B, n)          output
        primal_res_unscale_factor: wp.array2d(dtype=dtype),  # type: ignore  (B, num_duals)  output
    ):
        b, k = wp.tid()
        n_static = wp.static(n)
        np_static = wp.static(n + p)
        npm_static = wp.static(n + p + m)
        tail_start = wp.static(n + p + m + 1)
        off_zl_end  = wp.static(p)
        off_zu_end  = wp.static(p + num_hl)
        off_zbl_end = wp.static(p + num_hl + num_hu)
        off_zbu_end = wp.static(p + num_hl + num_hu + num_xl)
        num_duals   = wp.static(p + num_hl + num_hu + num_xl + num_xu)

        if k < n_static:
            delta_inv[b, k] = dtype(1.0) / delta[b, k]
            delta_b_inv[b, k] = dtype(1.0) / delta_b[b, k]
            # scale x_l and x_u inplace
            data_x_l[b, k] = data_x_l[b, k] * delta_b[b, k]
            data_x_u[b, k] = data_x_u[b, k] * delta_b[b, k]
            # dual_res_unscale_factor = cost_scaling_inv * delta_inv on x-block
            dual_res_unscale_factor[b, k] = dtype(1.0) / (cost_scaling[b] * delta[b, k])
        elif k < np_static:
            delta_inv[b, k] = dtype(1.0) / delta[b, k]
            # scale rhs of equality constraints inplace
            data_b[b, k - n_static] = data_b[b, k - n_static] * delta[b, k]
        elif k < npm_static:
            delta_inv[b, k] = dtype(1.0) / delta[b, k]
            # scale rhs of inequality constraints inplace
            jm = k - np_static
            data_h_l[b, jm] = data_h_l[b, jm] * delta[b, k]
            data_h_u[b, jm] = data_h_u[b, jm] * delta[b, k]
        elif k < npm_static + 1:
            # compute inverse of cost scaling
            cost_scaling_inv[b] = dtype(1.0) / cost_scaling[b]
        elif k < tail_start + num_duals:
            # primal_res_unscale_factor[b, j], packed in _dual_buffer order
            # [y | z_l | z_u | z_bl | z_bu]. The inequality/box duals are
            # full-length (identity index map), so each segment maps to a
            # contiguous slice of delta / delta_b. Read-only inputs (delta,
            # delta_b) — no read-after-write hazard within this kernel.
            j = k - tail_start
            if j < off_zl_end:
                primal_res_unscale_factor[b, j] = dtype(1.0) / delta[b, wp.static(n) + j]
            elif j < off_zu_end:
                idx = j - wp.static(p)
                primal_res_unscale_factor[b, j] = dtype(1.0) / delta[b, wp.static(n + p) + idx]
            elif j < off_zbl_end:
                idx = j - wp.static(p + num_hl)
                primal_res_unscale_factor[b, j] = dtype(1.0) / delta[b, wp.static(n + p) + idx]
            elif j < off_zbu_end:
                idx = j - wp.static(p + num_hl + num_hu)
                primal_res_unscale_factor[b, j] = dtype(1.0) / delta_b[b, idx]
            else:
                idx = j - wp.static(p + num_hl + num_hu + num_xl)
                primal_res_unscale_factor[b, j] = dtype(1.0) / delta_b[b, idx]
        else:
            return

    return finalize_and_scale_bounds_kernel


def create_scale_bounds_kernel(n: int, p: int, m: int, dtype=wp.float64):
    dtype = to_warp_dtype(dtype)
    """Sentinel-safe in-place forward bound scaling, single launch (B, n+p+m).

    Perform:

        b   *= d_y                                 # if p > 0
        h_l *= d_z;  h_u *= d_z                    # if m > 0
        x_l *= delta_b                             # full-length box bounds
        x_u *= delta_b
    """
    @wp.kernel
    def scale_bounds_kernel(
        delta:    wp.array2d(dtype=dtype),  # type: ignore  (B, n+p+m)
        delta_b:  wp.array2d(dtype=dtype),  # type: ignore  (B, n)
        data_b:   wp.array2d(dtype=dtype),  # type: ignore  (B, p)
        data_h_l: wp.array2d(dtype=dtype),  # type: ignore  (B, m)
        data_h_u: wp.array2d(dtype=dtype),  # type: ignore  (B, m)
        data_x_l: wp.array2d(dtype=dtype),  # type: ignore  (B, n)
        data_x_u: wp.array2d(dtype=dtype),  # type: ignore  (B, n)
    ):
        b, k = wp.tid()
        n_static = wp.static(n)
        np_static = wp.static(n + p)
        npm_static = wp.static(n + p + m)
        if k < n_static:
            dx = delta_b[b, k]
            data_x_l[b, k] = data_x_l[b, k] * dx
            data_x_u[b, k] = data_x_u[b, k] * dx
        elif k < np_static:
            jp = k - n_static
            data_b[b, jp] = data_b[b, jp] * delta[b, k]
        elif k < npm_static:
            jm = k - np_static
            dz = delta[b, k]
            data_h_l[b, jm] = data_h_l[b, jm] * dz
            data_h_u[b, jm] = data_h_u[b, jm] * dz
        else:
            return

    return scale_bounds_kernel


def create_unscale_bounds_kernel(n: int, p: int, m: int, dtype=wp.float64):
    dtype = to_warp_dtype(dtype)
    """Sentinel-safe in-place inverse bound scaling, single launch (B, n+p+m).

    Mirrors ``create_scale_bounds_kernel`` but multiplies by the stored
    inverse factors. Kept as a separate kernel (rather than reusing the
    forward one with inverse arguments) so the call site reads in the
    natural direction.

    Perform:

        b   *= d_y_inv                                 # if p > 0
        h_l *= d_z_inv;  h_u *= d_z_inv                # if m > 0
        x_l *= delta_b_inv                             # full-length box bounds
        x_u *= delta_b_inv

    """
    @wp.kernel
    def unscale_bounds_kernel(
        delta_inv:   wp.array2d(dtype=dtype),  # type: ignore  (B, n+p+m)
        delta_b_inv: wp.array2d(dtype=dtype),  # type: ignore  (B, n)
        data_b:      wp.array2d(dtype=dtype),  # type: ignore  (B, p)
        data_h_l:    wp.array2d(dtype=dtype),  # type: ignore  (B, m)
        data_h_u:    wp.array2d(dtype=dtype),  # type: ignore  (B, m)
        data_x_l:    wp.array2d(dtype=dtype),  # type: ignore  (B, n)
        data_x_u:    wp.array2d(dtype=dtype),  # type: ignore  (B, n)
    ):
        b, k = wp.tid()
        n_static = wp.static(n)
        np_static = wp.static(n + p)
        npm_static = wp.static(n + p + m)
        if k < n_static:
            dx_inv = delta_b_inv[b, k]
            data_x_l[b, k] = data_x_l[b, k] * dx_inv
            data_x_u[b, k] = data_x_u[b, k] * dx_inv
        elif k < np_static:
            jp = k - n_static
            data_b[b, jp] = data_b[b, jp] * delta_inv[b, k]
        elif k < npm_static:
            jm = k - np_static
            dz_inv = delta_inv[b, k]
            data_h_l[b, jm] = data_h_l[b, jm] * dz_inv
            data_h_u[b, jm] = data_h_u[b, jm] * dz_inv
        else:
            return

    return unscale_bounds_kernel


def create_accumulate_deltas_kernel(n: int, p: int, m: int, dtype=wp.float64):
    dtype = to_warp_dtype(dtype)
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
        delta:        wp.array2d(dtype=dtype),  # type: ignore  (B, n+p+m) in-out
        delta_b:      wp.array2d(dtype=dtype),  # type: ignore  (B, n)     in-out
        x_b_scaling:  wp.array2d(dtype=dtype),  # type: ignore  (B, n)     in-out
        delta_iter:   wp.array2d(dtype=dtype),  # type: ignore  (B, n+p+m) input
        delta_b_iter: wp.array2d(dtype=dtype),  # type: ignore  (B, n)     input
    ):
        b, k = wp.tid()
        di = delta_iter[b, k]
        delta[b, k] = delta[b, k] * di
        if k < wp.static(n):
            dbi = delta_b_iter[b, k]
            delta_b[b, k] = delta_b[b, k] * dbi
            x_b_scaling[b, k] = x_b_scaling[b, k] * dbi * di

    return accumulate_deltas_kernel


def create_ruiz_conv_check_kernel(n: int, p: int, m: int, dtype=wp.float64):
    dtype = to_warp_dtype(dtype)
    @wp.func
    def _diff_with_one(d: dtype) -> dtype:
        return wp.abs(dtype(1.0) - d)

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
        delta_iter:   wp.array2d(dtype=dtype),   # type: ignore  (B, n+p+m)
        delta_b_iter: wp.array2d(dtype=dtype),   # type: ignore  (B, n)
        conv_buf:     wp.array(dtype=dtype),     # type: ignore  (2B,) output
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


def create_compute_constraints_rhs_inf_norm_unscaled_kernel(
    n: int, p: int, m: int,
    num_hl: int, num_hu: int, num_xl: int, num_xu: int,
dtype=wp.float64):
    dtype = to_warp_dtype(dtype)
    """Tile-based per-batch unscaled-RHS inf-norm reduction.

        Perform:

        out.fill(0.0)
        if p > 0:        max(out, max_axis1(|delta_inv_y * b|))
        if num_hu > 0:   max(out, max_axis1(|finite_mask_hu * delta_inv_z * h_u|))
        if num_hl > 0:   max(out, max_axis1(|finite_mask_hl * delta_inv_z * h_l|))
        if num_xu > 0:   max(out, max_axis1(|finite_mask_xu * delta_b_inv * x_u|))
        if num_xl > 0:   max(out, max_axis1(|finite_mask_xl * delta_b_inv * x_l|))
    """
    @wp.func
    def finite_value(v: dtype, mask: dtype) -> dtype:
        return wp.where(mask > dtype(0.5), v, dtype(0.0))

    @wp.kernel
    def kernel(
        delta_inv:   wp.array2d(dtype=dtype),  # type: ignore  (B, n+p+m)
        delta_b_inv: wp.array2d(dtype=dtype),  # type: ignore  (B, n)
        data_b:      wp.array2d(dtype=dtype),  # type: ignore  (B, p)
        data_h_l:    wp.array2d(dtype=dtype),  # type: ignore  (B, m)
        data_h_u:    wp.array2d(dtype=dtype),  # type: ignore  (B, m)
        data_x_l:    wp.array2d(dtype=dtype),  # type: ignore  (B, n)
        data_x_u:    wp.array2d(dtype=dtype),  # type: ignore  (B, n)
        finite_mask_hl:   wp.array2d(dtype=dtype),  # type: ignore  (B, m)
        finite_mask_hu:   wp.array2d(dtype=dtype),  # type: ignore  (B, m)
        finite_mask_xl:   wp.array2d(dtype=dtype),  # type: ignore  (B, n)
        finite_mask_xu:   wp.array2d(dtype=dtype),  # type: ignore  (B, n)
        out:         wp.array(dtype=dtype),    # type: ignore  (B,)  output
    ):
        b, i = wp.tid()

        m_val = dtype(0.0)

        # Eq: contiguous slice — no gather.
        if wp.static(p > 0):
            b_tile = wp.tile_load(data_b[b], shape=p)
            dy_tile = wp.tile_load(delta_inv[b], shape=p, offset=n)
            m_eq = wp.tile_extract(
                wp.tile_max(wp.tile_map(wp.abs, b_tile * dy_tile)), 0,
            )
            m_val = wp.max(m_val, m_eq)

        # h_u: full-length inequality upper bounds, masked before use.
        if wp.static(num_hu > 0):
            finite_mask_hu_tile = wp.tile_load(finite_mask_hu[b], shape=num_hu)
            hu_raw_tile = wp.tile_load(data_h_u[b], shape=num_hu)
            hu_tile = wp.tile_map(finite_value, hu_raw_tile, finite_mask_hu_tile)
            dz_hu_tile = wp.tile_load(delta_inv[b], shape=num_hu, offset=(n + p))
            m_hu = wp.tile_extract(
                wp.tile_max(wp.tile_map(wp.abs, hu_tile * dz_hu_tile)), 0,
            )
            m_val = wp.max(m_val, m_hu)

        # h_l: full-length inequality lower bounds, masked before use.
        if wp.static(num_hl > 0):
            finite_mask_hl_tile = wp.tile_load(finite_mask_hl[b], shape=num_hl)
            hl_raw_tile = wp.tile_load(data_h_l[b], shape=num_hl)
            hl_tile = wp.tile_map(finite_value, hl_raw_tile, finite_mask_hl_tile)
            dz_hl_tile = wp.tile_load(delta_inv[b], shape=num_hl, offset=(n + p))
            m_hl = wp.tile_extract(
                wp.tile_max(wp.tile_map(wp.abs, hl_tile * dz_hl_tile)), 0,
            )
            m_val = wp.max(m_val, m_hl)

        # x_u: full-length box upper bounds, masked before use.
        if wp.static(num_xu > 0):
            finite_mask_xu_tile = wp.tile_load(finite_mask_xu[b], shape=num_xu)
            xu_raw_tile = wp.tile_load(data_x_u[b], shape=num_xu)
            xu_tile = wp.tile_map(finite_value, xu_raw_tile, finite_mask_xu_tile)
            db_xu_tile = wp.tile_load(delta_b_inv[b], shape=num_xu)
            m_xu = wp.tile_extract(
                wp.tile_max(wp.tile_map(wp.abs, xu_tile * db_xu_tile)), 0,
            )
            m_val = wp.max(m_val, m_xu)

        # x_l: full-length box lower bounds, masked before use.
        if wp.static(num_xl > 0):
            finite_mask_xl_tile = wp.tile_load(finite_mask_xl[b], shape=num_xl)
            xl_raw_tile = wp.tile_load(data_x_l[b], shape=num_xl)
            xl_tile = wp.tile_map(finite_value, xl_raw_tile, finite_mask_xl_tile)
            db_xl_tile = wp.tile_load(delta_b_inv[b], shape=num_xl)
            m_xl = wp.tile_extract(
                wp.tile_max(wp.tile_map(wp.abs, xl_tile * db_xl_tile)), 0,
            )
            m_val = wp.max(m_val, m_xl)

        # All threads in the tile see the same scalar m_val (tile_extract
        # broadcasts); single-thread store avoids redundant writes.
        if i == 0:
            out[b] = m_val

    return kernel


