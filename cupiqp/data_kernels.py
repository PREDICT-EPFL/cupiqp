import warp as wp

from .utils import to_warp_dtype
from .typedef import PIQP_INF


def create_finite_bound_masks_kernel(
    has_h_l: bool, has_h_u: bool, has_x_l: bool, has_x_u: bool, dtype=wp.float64,
    ):
    """Build all per-batch finite-bound masks in a single GPU pass.

    The kernel is specialized to the block presence flags ``has_h_l`` /
    ``has_h_u`` / ``has_x_l`` / ``has_x_u``, which are fixed at setup(): a block
    that was omitted occupies no storage and is never read or written.

    For each problem in the batch the kernel marks, with 1.0 (finite) or 0.0
    (+/-inf), which inequality and box bounds are active, writing:

    * ``finite_mask_all`` -- the packed ``(B, num_ineq)`` mask laid out as
      ``[finite_mask_hl(num_hl) | finite_mask_hu(num_hu) | finite_mask_xl(num_xl) | finite_mask_xu(num_xu)]``,
      where each width is ``m`` (for h blocks) or ``n`` (for x blocks) if the
      block is present and ``0`` otherwise (the four per-class masks are views
      into this buffer);
    * ``active_G_row`` ``(B, m)`` -- 1.0 where an inequality row has any finite
      bound (lower or upper); all-zero when both inequality blocks are absent;
    * ``active_x_bound`` ``(B, n)`` -- 1.0 where a variable has any finite box
      bound; all-zero when both box blocks are absent;
    * ``num_finite_bounds`` ``(B,)`` -- the per-problem count of finite bounds
      (the divisor used for mu/sigma).

    A bound is finite iff ``-PIQP_INF < value < PIQP_INF``. One thread handles
    one batch element. Launch with ``dim=(B,)``.
    """
    dtype = to_warp_dtype(dtype)

    @wp.kernel
    def finite_bound_masks_kernel(
        h_l: wp.array2d(dtype=dtype),               # type: ignore  (B, num_hl)
        h_u: wp.array2d(dtype=dtype),               # type: ignore  (B, num_hu)
        x_l: wp.array2d(dtype=dtype),               # type: ignore  (B, num_xl)
        x_u: wp.array2d(dtype=dtype),               # type: ignore  (B, num_xu)
        finite_mask_all: wp.array2d(dtype=dtype),   # type: ignore  (B, num_ineq)
        active_G_row: wp.array2d(dtype=dtype),      # type: ignore  (B, m)
        active_x_bound: wp.array2d(dtype=dtype),    # type: ignore  (B, n)
        num_finite_bounds: wp.array(dtype=dtype),   # type: ignore  (B,)
    ):
        b = wp.tid()
        # never derive m / n from h_l / x_l: an omitted block is (B, 0).
        m = active_G_row.shape[1]
        n = active_x_bound.shape[1]
        one = dtype(1.0)
        zero = dtype(0.0)
        neg_inf = dtype(-PIQP_INF)
        pos_inf = dtype(PIQP_INF)

        # Running offsets in the packed [hl? | hu? | xl? | xu?] layout. An
        # omitted block has zero width, so the following block slides up.
        off_hl = 0
        off_hu = wp.static(int(has_h_l)) * m
        off_xl = wp.static(int(has_h_l) + int(has_h_u)) * m

        count = zero
        for i in range(m):
            fl = zero
            fu = zero
            if wp.static(has_h_l):
                fl = wp.where(h_l[b, i] > neg_inf, one, zero)
                finite_mask_all[b, off_hl + i] = fl     # finite_mask_hl
                count += fl
            if wp.static(has_h_u):
                fu = wp.where(h_u[b, i] < pos_inf, one, zero)
                finite_mask_all[b, off_hu + i] = fu     # finite_mask_hu
                count += fu
            active_G_row[b, i] = wp.max(fl, fu)

        off_xu = off_xl + wp.static(int(has_x_l)) * n
        for j in range(n):
            fl = zero
            fu = zero
            if wp.static(has_x_l):
                fl = wp.where(x_l[b, j] > neg_inf, one, zero)
                finite_mask_all[b, off_xl + j] = fl     # finite_mask_xl
                count += fl
            if wp.static(has_x_u):
                fu = wp.where(x_u[b, j] < pos_inf, one, zero)
                finite_mask_all[b, off_xu + j] = fu     # finite_mask_xu
                count += fu
            active_x_bound[b, j] = wp.max(fl, fu)
        num_finite_bounds[b] = count

    return finite_bound_masks_kernel
