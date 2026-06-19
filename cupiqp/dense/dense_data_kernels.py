import warp as wp

from ..utils import to_warp_dtype
from ..typedef import PIQP_INF


def create_bound_set_check_kernel(dtype=wp.float64):
    """Fused in-place bound update + finite/infinite-structure check.

    For each entry of a bound vector (``h_l``, ``h_u``, ``x_l`` or ``x_u``) the
    kernel, in a single GPU pass:

    * checks that the new value's finite/infinite status matches the pattern
      fixed at ``setup()`` (``expected[i]`` is 1 if entry ``i`` must stay
      finite), incrementing ``err`` whenever it differs -- so callers can
      validate on-device without a host reduction; and
    * writes the new value into ``out``, except on *disabled* rows (both bounds
      infinite, ``free[i] == 1``) where it writes the ``benign`` finite value
      instead. This mirrors ``_disable_inf_constraints`` and stops a raw
      ``+/-inf`` from re-entering a row that is part of the active set.

    ``sign`` selects the bound direction: ``+1`` for upper bounds (finite iff
    ``v < PIQP_INF``), ``-1`` for lower bounds (finite iff ``v > -PIQP_INF``,
    i.e. ``-v < PIQP_INF``). ``commit`` gates the write: pass ``0`` for a
    check-only pass (leaves ``out`` untouched, so a rejected update does not
    modify the data) and ``1`` to also assign. Launch with ``dim=(B, k)``.
    """
    wp_dtype = to_warp_dtype(dtype)
    inf = float(PIQP_INF)

    @wp.kernel
    def bound_set_check_kernel(
        value: wp.array2d(dtype=wp_dtype),   # type: ignore   (B, k) new values
        expected: wp.array(dtype=wp.int32),  # type: ignore   (k,)  1 if finite expected
        free: wp.array(dtype=wp.int32),      # type: ignore   (k,)  1 if disabled (both-inf) row
        sign: wp_dtype,                       # type: ignore   +1 upper, -1 lower
        benign: wp_dtype,                     # type: ignore   value written on disabled rows
        commit: wp.int32,                     # type: ignore   1 = also write out, 0 = check only
        out: wp.array2d(dtype=wp_dtype),     # type: ignore   (B, k) destination
        err: wp.array(dtype=wp.int32),       # type: ignore   (1,)  violation counter
    ):
        b, i = wp.tid()
        v = value[b, i]

        is_finite = wp.int32(0)
        if sign * v < wp_dtype(inf):
            is_finite = wp.int32(1)
        if is_finite != expected[i]:
            wp.atomic_add(err, 0, 1)

        if commit == wp.int32(1):
            if free[i] == wp.int32(1):
                out[b, i] = benign
            else:
                out[b, i] = v

    return bound_set_check_kernel
