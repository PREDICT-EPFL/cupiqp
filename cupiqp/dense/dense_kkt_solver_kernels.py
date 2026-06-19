import warp as wp
from ..utils import to_warp_dtype


def create_update_kkt_kernel(n: int, p: int, m: int, dtype=wp.float64):
    dtype = to_warp_dtype(dtype)
    """Fused warp kernel replacing the cupy ops in ``DenseKKTSolver.update_kkt``
    (everything that ran before the trailing ``syrk``).

    One launch builds, in parallel:
      - ``delta_inv[b, 0]      = 1 / delta[b]``                      (per-batch scalar)
      - ``kkt_mat[b, i, j]     = P[b, i, j]
                                  + (i == j) * x_reg[b, i]
                                  + AtA[b, i, j] / delta[b]``
      - ``z_reg_inv_out[b, k]  = z_reg_inv[b, k]``  (cached copy, k in [0, m))
      - ``z_reg_inv_sqrt[b, k] = sqrt(z_reg_inv[b, k])``
      - ``G_scaled[b, i, j]    = sqrt(z_reg_inv[b, i]) * G[b, i, j]``  (sqrt recomputed
                                  per (i, j) thread to avoid cross-thread sync)

    Here ``z_reg_inv`` is the condensed inequality row weight ``w_l + w_u``. The
    condensed dense assembly needs the weight directly (the explicit augmented
    diagonal ``z_reg = 1 / weight`` is not used here). A zero weight is valid and
    means the full-length row is inactive, so it contributes neither to
    ``G_scaled`` nor to ``G.T @ diag(z_reg_inv) @ G``.

    """
    n_n_total  = n * n
    m_n_total  = m * n
    total_dim  = n_n_total + m + m_n_total

    @wp.kernel
    def update_kkt_kernel(
        # Inputs
        data_P:    wp.array3d(dtype=dtype),   # type: ignore  (B, n, n)
        data_AtA:  wp.array3d(dtype=dtype),   # type: ignore  (B, n, n)  (zero-shape if p == 0)
        data_G:    wp.array3d(dtype=dtype),   # type: ignore  (B, m, n)  (zero-shape if m == 0)
        delta:     wp.array(dtype=dtype),     # type: ignore  (B,)
        x_reg:     wp.array2d(dtype=dtype),   # type: ignore  (B, n)
        z_reg_inv: wp.array2d(dtype=dtype),   # type: ignore  (B, m)
        # Outputs
        delta_inv:        wp.array(dtype=dtype),    # type: ignore  (B,)
        z_reg_inv_out:    wp.array2d(dtype=dtype),  # type: ignore  (B, m), copy z_reg_inv to here
        z_reg_inv_sqrt:   wp.array2d(dtype=dtype),  # type: ignore  (B, m)
        kkt_mat:          wp.array3d(dtype=dtype),  # type: ignore  (B, n, n)
        G_scaled:         wp.array3d(dtype=dtype),  # type: ignore  (B, m, n)
    ):
        b, t = wp.tid()
        n_static    = wp.static(n)
        m_static    = wp.static(m)
        n_n_static  = wp.static(n_n_total)
        m_n_static  = wp.static(m_n_total)

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
        elif t < n_n_static + m_static:
            # Cache the row weight and its sqrt for solve()/recovery.
            # NOTE: z_reg_inv contains zeros for inactive rows s.t. -inf <= G[i] * x <= +inf
            k = t - n_n_static
            w = z_reg_inv[b, k]
            z_reg_inv_out[b, k] = w
            z_reg_inv_sqrt[b, k] = wp.sqrt(w)
        elif t < n_n_static + m_static + m_n_static:
            # G_scaled = sqrt(weight[b, i]) * G[b, i, j]; inactive rows (weight 0)
            # contribute nothing to G.T @ diag(z_reg_inv) @ G.
            k = t - n_n_static - m_static
            i = k // n_static
            j = k % n_static
            G_scaled[b, i, j] = wp.sqrt(z_reg_inv[b, i]) * data_G[b, i, j]
        else:
            return

    return update_kkt_kernel, total_dim


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
