import functools

import warp as wp
from .utils import to_warp_dtype


wp.set_module_options({"enable_backward": False})


@functools.lru_cache(maxsize=None)
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


@functools.lru_cache(maxsize=None)
def create_calc_scaling_inv_and_scale_bounds_kernel(
    n: int, p: int, m: int,
    num_hl: int, num_hu: int, num_xl: int, num_xu: int,
    has_h_l: bool, has_h_u: bool,
    has_x_l: bool, has_x_u: bool,
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
            # scale x_l and x_u inplace (only for box blocks that exist)
            if wp.static(has_x_l):
                data_x_l[b, k] = data_x_l[b, k] * delta_b[b, k]
            if wp.static(has_x_u):
                data_x_u[b, k] = data_x_u[b, k] * delta_b[b, k]
            # dual_res_unscale_factor = cost_scaling_inv * delta_inv on x-block
            dual_res_unscale_factor[b, k] = dtype(1.0) / (cost_scaling[b] * delta[b, k])
        elif k < np_static:
            delta_inv[b, k] = dtype(1.0) / delta[b, k]
            # scale rhs of equality constraints inplace
            data_b[b, k - n_static] = data_b[b, k - n_static] * delta[b, k]
        elif k < npm_static:
            delta_inv[b, k] = dtype(1.0) / delta[b, k]
            # scale rhs of inequality constraints inplace (only for present sides)
            jm = k - np_static
            if wp.static(has_h_l):
                data_h_l[b, jm] = data_h_l[b, jm] * delta[b, k]
            if wp.static(has_h_u):
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


@functools.lru_cache(maxsize=None)
def create_scale_bounds_kernel(n: int, p: int, m: int, has_h_l: bool, has_h_u: bool, has_x_l: bool, has_x_u: bool, dtype=wp.float64):
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
            if wp.static(has_x_l):
                data_x_l[b, k] = data_x_l[b, k] * dx
            if wp.static(has_x_u):
                data_x_u[b, k] = data_x_u[b, k] * dx
        elif k < np_static:
            jp = k - n_static
            data_b[b, jp] = data_b[b, jp] * delta[b, k]
        elif k < npm_static:
            jm = k - np_static
            dz = delta[b, k]
            if wp.static(has_h_l):
                data_h_l[b, jm] = data_h_l[b, jm] * dz
            if wp.static(has_h_u):
                data_h_u[b, jm] = data_h_u[b, jm] * dz
        else:
            return

    return scale_bounds_kernel


@functools.lru_cache(maxsize=None)
def create_unscale_bounds_kernel(n: int, p: int, m: int, has_h_l: bool, has_h_u: bool, has_x_l: bool, has_x_u: bool, dtype=wp.float64):
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
            if wp.static(has_x_l):
                data_x_l[b, k] = data_x_l[b, k] * dx_inv
            if wp.static(has_x_u):
                data_x_u[b, k] = data_x_u[b, k] * dx_inv
        elif k < np_static:
            jp = k - n_static
            data_b[b, jp] = data_b[b, jp] * delta_inv[b, k]
        elif k < npm_static:
            jm = k - np_static
            dz_inv = delta_inv[b, k]
            if wp.static(has_h_l):
                data_h_l[b, jm] = data_h_l[b, jm] * dz_inv
            if wp.static(has_h_u):
                data_h_u[b, jm] = data_h_u[b, jm] * dz_inv
        else:
            return

    return unscale_bounds_kernel


@functools.lru_cache(maxsize=None)
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


# Threads per batch entry in the block-reduction kernels below; each thread
# strides over its row, so the kernels are independent of the problem width.
REDUCTION_BLOCK_DIM = 256


@functools.lru_cache(maxsize=None)
def create_ruiz_conv_check_kernel(dtype=wp.float64):
    dtype = to_warp_dtype(dtype)
    """Per-batch Ruiz convergence measure, reduced over the whole batch.

        conv = max_b max( max_k |1 - delta_iter[b, k]|, max_k |1 - delta_b_iter[b, k]| )

    ``conv_out`` is a one-element atomic-max accumulator; zero it before the
    launch and read it back once. Dispatch with
    ``wp.launch_tiled(dim=[B], block_dim=REDUCTION_BLOCK_DIM)``.
    """
    @wp.kernel
    def conv_check_kernel(
        delta_iter:   wp.array2d(dtype=dtype),   # type: ignore  (B, n+p+m)
        delta_b_iter: wp.array2d(dtype=dtype),   # type: ignore  (B, n)
        conv_out:     wp.array(dtype=dtype),     # type: ignore  (1,) output
    ):
        b, i = wp.tid()
        bd = wp.block_dim()
        v = dtype(0.0)
        for k in range(i, delta_iter.shape[1], bd):
            v = wp.max(v, wp.abs(dtype(1.0) - delta_iter[b, k]))
        for k in range(i, delta_b_iter.shape[1], bd):
            v = wp.max(v, wp.abs(dtype(1.0) - delta_b_iter[b, k]))
        red = wp.tile_max(wp.tile(v))
        if i == 0:
            wp.atomic_max(conv_out, 0, red[0])

    return conv_check_kernel


@functools.lru_cache(maxsize=None)
def create_compute_constraints_rhs_inf_norm_unscaled_kernel(dtype=wp.float64):
    dtype = to_warp_dtype(dtype)
    """Per-batch inf-norm of the user-space (unscaled) constraint right-hand sides.

        out[b] = max( max_k |delta_inv_y * b|,
                      max_k |finite_mask_hu * delta_inv_z * h_u|,
                      max_k |finite_mask_hl * delta_inv_z * h_l|,
                      max_k |finite_mask_xu * delta_b_inv * x_u|,
                      max_k |finite_mask_xl * delta_b_inv * x_l| )

    over the blocks that exist (an absent block is ``(B, 0)``). Dispatch with
    ``wp.launch_tiled(dim=[B], block_dim=REDUCTION_BLOCK_DIM)``.
    """
    @wp.func
    def finite_value(v: dtype, mask: dtype) -> dtype:    # type: ignore
        return wp.where(mask > dtype(0.5), v, dtype(0.0))

    @wp.kernel
    def kernel(
        delta_inv:   wp.array2d(dtype=dtype),  # type: ignore  (B, n+p+m)
        delta_b_inv: wp.array2d(dtype=dtype),  # type: ignore  (B, n)
        data_b:      wp.array2d(dtype=dtype),  # type: ignore  (B, p)
        data_h_l:    wp.array2d(dtype=dtype),  # type: ignore  (B, num_hl)
        data_h_u:    wp.array2d(dtype=dtype),  # type: ignore  (B, num_hu)
        data_x_l:    wp.array2d(dtype=dtype),  # type: ignore  (B, num_xl)
        data_x_u:    wp.array2d(dtype=dtype),  # type: ignore  (B, num_xu)
        finite_mask_hl:   wp.array2d(dtype=dtype),  # type: ignore  (B, num_hl)
        finite_mask_hu:   wp.array2d(dtype=dtype),  # type: ignore  (B, num_hu)
        finite_mask_xl:   wp.array2d(dtype=dtype),  # type: ignore  (B, num_xl)
        finite_mask_xu:   wp.array2d(dtype=dtype),  # type: ignore  (B, num_xu)
        out:         wp.array(dtype=dtype),    # type: ignore  (B,)  output
    ):
        b, i = wp.tid()
        bd = wp.block_dim()
        n = delta_b_inv.shape[1]
        p = data_b.shape[1]
        v = dtype(0.0)
        for k in range(i, p, bd):
            v = wp.max(v, wp.abs(data_b[b, k] * delta_inv[b, n + k]))
        for k in range(i, data_h_u.shape[1], bd):
            v = wp.max(v, wp.abs(finite_value(data_h_u[b, k], finite_mask_hu[b, k]) * delta_inv[b, n + p + k]))
        for k in range(i, data_h_l.shape[1], bd):
            v = wp.max(v, wp.abs(finite_value(data_h_l[b, k], finite_mask_hl[b, k]) * delta_inv[b, n + p + k]))
        for k in range(i, data_x_u.shape[1], bd):
            v = wp.max(v, wp.abs(finite_value(data_x_u[b, k], finite_mask_xu[b, k]) * delta_b_inv[b, k]))
        for k in range(i, data_x_l.shape[1], bd):
            v = wp.max(v, wp.abs(finite_value(data_x_l[b, k], finite_mask_xl[b, k]) * delta_b_inv[b, k]))
        red = wp.tile_max(wp.tile(v))
        if i == 0:
            out[b] = red[0]

    return kernel


@functools.lru_cache(maxsize=None)
def create_unscale_solution_kernel(dtype=wp.float64):
    dtype = to_warp_dtype(dtype)
    """Map a scaled IPM iterate back to original coordinates, in place.

    One launch over ``dim=(B, n_primal + n_dual)`` covering both contiguous
    variable buffers ``[x | s_l | s_u | s_bl | s_bu]`` and
    ``[y | z_l | z_u | z_bl | z_bu]`` (an omitted block has width 0):

        x    *= delta_x                     s_l, s_u   *= delta_inv_z
        y    *= c_inv * delta_y             s_bl, s_bu *= delta_b_inv
        z_l, z_u   *= c_inv * delta_z
        z_bl, z_bu *= c_inv * delta_b
    """
    @wp.kernel
    def unscale_solution_kernel(
        x:    wp.array2d(dtype=dtype),   # type: ignore  (B, n)
        s_l:  wp.array2d(dtype=dtype),   # type: ignore  (B, num_hl)
        s_u:  wp.array2d(dtype=dtype),   # type: ignore  (B, num_hu)
        s_bl: wp.array2d(dtype=dtype),   # type: ignore  (B, num_xl)
        s_bu: wp.array2d(dtype=dtype),   # type: ignore  (B, num_xu)
        y:    wp.array2d(dtype=dtype),   # type: ignore  (B, p)
        z_l:  wp.array2d(dtype=dtype),   # type: ignore  (B, num_hl)
        z_u:  wp.array2d(dtype=dtype),   # type: ignore  (B, num_hu)
        z_bl: wp.array2d(dtype=dtype),   # type: ignore  (B, num_xl)
        z_bu: wp.array2d(dtype=dtype),   # type: ignore  (B, num_xu)
        delta:            wp.array2d(dtype=dtype),  # type: ignore  (B, n+p+m)
        delta_inv:        wp.array2d(dtype=dtype),  # type: ignore  (B, n+p+m)
        delta_b:          wp.array2d(dtype=dtype),  # type: ignore  (B, n)
        delta_b_inv:      wp.array2d(dtype=dtype),  # type: ignore  (B, n)
        cost_scaling_inv: wp.array(dtype=dtype),    # type: ignore  (B,)
    ):
        b, t = wp.tid()
        n = x.shape[1]
        p = y.shape[1]
        num_hl = s_l.shape[1]
        num_hu = s_u.shape[1]
        num_xl = s_bl.shape[1]
        num_xu = s_bu.shape[1]
        c_inv = cost_scaling_inv[b]

        end_x = n
        end_sl = end_x + num_hl
        end_su = end_sl + num_hu
        end_sbl = end_su + num_xl
        end_sbu = end_sbl + num_xu
        end_y = end_sbu + p
        end_zl = end_y + num_hl
        end_zu = end_zl + num_hu
        end_zbl = end_zu + num_xl
        end_zbu = end_zbl + num_xu

        if t < end_x:
            x[b, t] = x[b, t] * delta[b, t]
        elif t < end_sl:
            k = t - end_x
            s_l[b, k] = s_l[b, k] * delta_inv[b, n + p + k]
        elif t < end_su:
            k = t - end_sl
            s_u[b, k] = s_u[b, k] * delta_inv[b, n + p + k]
        elif t < end_sbl:
            k = t - end_su
            s_bl[b, k] = s_bl[b, k] * delta_b_inv[b, k]
        elif t < end_sbu:
            k = t - end_sbl
            s_bu[b, k] = s_bu[b, k] * delta_b_inv[b, k]
        elif t < end_y:
            k = t - end_sbu
            y[b, k] = y[b, k] * delta[b, n + k] * c_inv
        elif t < end_zl:
            k = t - end_y
            z_l[b, k] = z_l[b, k] * delta[b, n + p + k] * c_inv
        elif t < end_zu:
            k = t - end_zl
            z_u[b, k] = z_u[b, k] * delta[b, n + p + k] * c_inv
        elif t < end_zbl:
            k = t - end_zu
            z_bl[b, k] = z_bl[b, k] * delta_b[b, k] * c_inv
        elif t < end_zbu:
            k = t - end_zbl
            z_bu[b, k] = z_bu[b, k] * delta_b[b, k] * c_inv

    return unscale_solution_kernel
