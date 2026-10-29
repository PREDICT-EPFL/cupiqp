import functools

import warp as wp
from ..utils import to_warp_dtype


wp.set_module_options({"enable_backward": False})


@functools.lru_cache(maxsize=None)
def create_update_kkt_kernel(n: int, p: int, m: int, dtype=wp.float64):
    dtype = to_warp_dtype(dtype)
    """Fused warp kernel of the part of ``DenseKKTSolver.update_and_factor_kkt``
    before the ``G^T diag(z_reg_inv) G`` product.

    One launch builds, in parallel:
      - ``delta_inv[b]         = 1 / delta[b]``                      (per-batch scalar)
      - ``kkt_mat[b, i, j]     = P[b, i, j]
                                  + (i == j) * x_reg[b, i]
                                  + AtA[b, i, j] / delta[b]``
      - ``z_reg_inv_out[b, k]  = z_reg_inv[b, k]``  (cached copy, k in [0, m))

    Here ``z_reg_inv`` is the condensed inequality row weight ``w_l + w_u``. A
    zero weight is valid and means the full-length row is inactive.
    """
    n_n_total  = n * n
    total_dim  = n_n_total + m

    @wp.kernel
    def update_kkt_kernel(
        # Inputs
        data_P:    wp.array3d(dtype=dtype),   # type: ignore  (B, n, n)
        data_AtA:  wp.array3d(dtype=dtype),   # type: ignore  (B, n, n)  (zero-shape if p == 0)
        delta:     wp.array(dtype=dtype),     # type: ignore  (B,)
        x_reg:     wp.array2d(dtype=dtype),   # type: ignore  (B, n)
        z_reg_inv: wp.array2d(dtype=dtype),   # type: ignore  (B, m)
        # Outputs
        delta_inv:        wp.array(dtype=dtype),    # type: ignore  (B,)
        z_reg_inv_out:    wp.array2d(dtype=dtype),  # type: ignore  (B, m), copy z_reg_inv to here
        kkt_mat:          wp.array3d(dtype=dtype),  # type: ignore  (B, n, n)
    ):
        b, t = wp.tid()
        n_static    = wp.static(n)
        n_n_static  = wp.static(n_n_total)

        # Per-batch scalar (one writer per batch).
        if t == 0:
            delta_inv[b] = dtype(1.0) / delta[b]

        if t < n_n_static:
            # kkt_mat = P + diag(x_reg) + 1/delta*AtA
            i = t // n_static
            j = t % n_static
            v = data_P[b, i, j]
            if i == j:
                v = v + x_reg[b, i]
            if wp.static(p > 0):
                v = v + (dtype(1.0) / delta[b]) * data_AtA[b, i, j]
            kkt_mat[b, i, j] = v
        else:
            # Cache the row weight for solve()/recovery.
            k = t - n_n_static
            z_reg_inv_out[b, k] = z_reg_inv[b, k]

    return update_kkt_kernel, total_dim


@functools.lru_cache(maxsize=None)
def create_solve_pre_cholesky_kernel(p: int, m: int, dtype=wp.float64):
    dtype = to_warp_dtype(dtype)
    """Fused post-matvec / pre-Cholesky assembly of the right-hand side
    ``delta_x``::

        delta_x[b, i] = rhs_x[b, i]
                        + (p>0 ? delta_inv[b] * work_n_AT[b, i] : 0)
                        + (m>0 ? work_n_GT[b, i]                : 0)
    """
    @wp.kernel
    def solve_pre_cholesky_kernel(
        rhs_x:       wp.array2d(dtype=dtype),  # type: ignore  (B, n)
        delta_inv:   wp.array(dtype=dtype),    # type: ignore  (B,)
        work_n_AT:   wp.array2d(dtype=dtype),  # type: ignore  (B, n)  (zero-shape if p == 0)
        work_n_GT:   wp.array2d(dtype=dtype),  # type: ignore  (B, n)  (zero-shape if m == 0)
        delta_x:     wp.array2d(dtype=dtype),  # type: ignore  (B, n)  output
    ):
        b, i = wp.tid()
        v = rhs_x[b, i]
        if wp.static(p > 0):
            v = v + delta_inv[b] * work_n_AT[b, i]
        if wp.static(m > 0):
            v = v + work_n_GT[b, i]
        delta_x[b, i] = v

    return solve_pre_cholesky_kernel


@functools.lru_cache(maxsize=None)
def create_solve_post_cholesky_kernel(p: int, m: int, dtype=wp.float64):
    dtype = to_warp_dtype(dtype)
    """Fused finalization after the Cholesky solve and the A / G matvecs::

        delta_y[b, i] = (delta_y[b, i] - rhs_y[b, i]) * delta_inv[b, 0]    (i in [0, p))
        delta_z[b, k] = (delta_z[b, k] - rhs_z[b, k]) * z_reg_inv[b, k]    (k in [0, m))

    """
    @wp.kernel
    def solve_post_cholesky_kernel(
        rhs_y:     wp.array2d(dtype=dtype),  # type: ignore  (B, p)
        rhs_z:     wp.array2d(dtype=dtype),  # type: ignore  (B, m)
        delta_inv: wp.array(dtype=dtype),    # type: ignore  (B,)
        z_reg_inv: wp.array2d(dtype=dtype),  # type: ignore  (B, m)
        delta_y:   wp.array2d(dtype=dtype),  # type: ignore  (B, p)  in-out
        delta_z:   wp.array2d(dtype=dtype),  # type: ignore  (B, m)  in-out
    ):
        b, t = wp.tid()
        p_static = wp.static(p)
        m_static = wp.static(m)
        if t < p_static:
            delta_y[b, t] = (delta_y[b, t] - rhs_y[b, t]) * delta_inv[b]
        elif t < p_static + m_static:
            k = t - p_static
            delta_z[b, k] = (delta_z[b, k] - rhs_z[b, k]) * z_reg_inv[b, k]

    return solve_post_cholesky_kernel
