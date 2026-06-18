import warp as wp

from .utils import to_warp_dtype
from .typedef import PIQP_INF


def create_finite_bound_masks_kernel(dtype=wp.float64):
    """Build all per-batch finite-bound masks in a single GPU pass.

    For each problem in the batch the kernel marks, with 1.0 (finite) or 0.0
    (+/-inf), which inequality and box bounds are active, writing:

    * ``finite_mask_all`` -- the packed ``(B, 2m+2n)`` mask laid out as
      ``[finite_mask_hl(m) | finite_mask_hu(m) | finite_mask_xl(n) | finite_mask_xu(n)]`` (the four
      per-class masks are views into this buffer);
    * ``active_G_row`` ``(B, m)`` -- 1.0 where an inequality row has any finite
      bound (lower or upper);
    * ``active_x_bound`` ``(B, n)`` -- 1.0 where a variable has any finite box
      bound;
    * ``num_finite_bounds`` ``(B,)`` -- the per-problem count of finite bounds
      (the divisor used for mu/sigma).

    A bound is finite iff ``-PIQP_INF < value < PIQP_INF``. One thread handles
    one batch element. Launch with ``dim=(B,)``.
    """
    dtype = to_warp_dtype(dtype)

    @wp.kernel
    def finite_bound_masks_kernel(
        h_l: wp.array2d(dtype=dtype),               # type: ignore  (B, m)
        h_u: wp.array2d(dtype=dtype),               # type: ignore  (B, m)
        x_l: wp.array2d(dtype=dtype),               # type: ignore  (B, n)
        x_u: wp.array2d(dtype=dtype),               # type: ignore  (B, n)
        finite_mask_all: wp.array2d(dtype=dtype),        # type: ignore  (B, 2m+2n)
        active_G_row: wp.array2d(dtype=dtype),      # type: ignore  (B, m)
        active_x_bound: wp.array2d(dtype=dtype),    # type: ignore  (B, n)
        num_finite_bounds: wp.array(dtype=dtype),   # type: ignore  (B,)
    ):
        b = wp.tid()
        m = h_l.shape[1]
        n = x_l.shape[1]
        one = dtype(1.0)
        zero = dtype(0.0)
        neg_inf = dtype(-PIQP_INF)
        pos_inf = dtype(PIQP_INF)

        count = zero
        for i in range(m):
            fl = wp.where(h_l[b, i] > neg_inf, one, zero)
            fu = wp.where(h_u[b, i] < pos_inf, one, zero)
            finite_mask_all[b, i] = fl          # finite_mask_hl
            finite_mask_all[b, m + i] = fu      # finite_mask_hu
            active_G_row[b, i] = wp.max(fl, fu)
            count += fl + fu
        for j in range(n):
            fl = wp.where(x_l[b, j] > neg_inf, one, zero)
            fu = wp.where(x_u[b, j] < pos_inf, one, zero)
            finite_mask_all[b, 2 * m + j] = fl          # finite_mask_xl
            finite_mask_all[b, 2 * m + n + j] = fu      # finite_mask_xu
            active_x_bound[b, j] = wp.max(fl, fu)
            count += fl + fu
        num_finite_bounds[b] = count

    return finite_bound_masks_kernel
