import warp as wp


def create_update_kkt_kernel(n: int, p: int, m: int):
    """Fused warp kernel replacing the cupy ops in ``DenseKKTSolver.update_kkt``
    (everything that ran before the trailing ``syrk``).

    One launch builds, in parallel:
      - ``delta_inv[b, 0]      = 1 / delta[b]``                      (per-batch scalar)
      - ``kkt_mat[b, i, j]     = P[b, i, j]
                                  + (i == j) * x_reg[b, i]
                                  + AtA[b, i, j] / delta[b]``
      - ``z_reg_inv[b, k]      = 1 / z_reg[b, k]``                   (k in [0, m))
      - ``z_reg_inv_sqrt[b, k] = sqrt(1 / z_reg[b, k])``
      - ``G_scaled[b, i, j]    = sqrt(1 / z_reg[b, i]) * G[b, i, j]``  (z-sqrt recomputed
                                  per (i, j) thread to avoid cross-thread sync)

    """
    n_n_total  = n * n
    m_n_total  = m * n
    total_dim  = n_n_total + m + m_n_total

    @wp.kernel
    def update_kkt_kernel(
        # Inputs
        data_P:    wp.array3d(dtype=wp.float64),   # type: ignore  (B, n, n)
        data_AtA:  wp.array3d(dtype=wp.float64),   # type: ignore  (B, n, n)  (zero-shape if p == 0)
        data_G:    wp.array3d(dtype=wp.float64),   # type: ignore  (B, m, n)  (zero-shape if m == 0)
        delta:     wp.array(dtype=wp.float64),     # type: ignore  (B,)
        x_reg:     wp.array2d(dtype=wp.float64),   # type: ignore  (B, n)
        z_reg:     wp.array2d(dtype=wp.float64),   # type: ignore  (B, m)
        # Outputs
        delta_inv:        wp.array2d(dtype=wp.float64),  # type: ignore  (B, 1)
        z_reg_inv:        wp.array2d(dtype=wp.float64),  # type: ignore  (B, m)
        z_reg_inv_sqrt:   wp.array2d(dtype=wp.float64),  # type: ignore  (B, m)
        kkt_mat:          wp.array3d(dtype=wp.float64),  # type: ignore  (B, n, n)
        G_scaled:         wp.array3d(dtype=wp.float64),  # type: ignore  (B, m, n)
    ):
        b, t = wp.tid()
        n_static    = wp.static(n)
        m_static    = wp.static(m)
        n_n_static  = wp.static(n_n_total)
        m_n_static  = wp.static(m_n_total)

        # Per-batch scalar (one writer per batch).
        if t == 0:
            delta_inv[b, 0] = wp.float64(1.0) / delta[b]

        if t < n_n_static:
            # Region 1: kkt_mat = P + diag(x_reg) + 1/delta*AtA
            i = t // n_static
            j = t % n_static
            v = data_P[b, i, j]
            if i == j:
                v = v + x_reg[b, i]
            if wp.static(p > 0):
                v = v + (wp.float64(1.0) / delta[b]) * data_AtA[b, i, j]
            kkt_mat[b, i, j] = v
        elif t < n_n_static + m_static:
            # Region 2: z_reg_inv & z_reg_inv_sqrt.
            k = t - n_n_static
            zinv = wp.float64(1.0) / z_reg[b, k]
            z_reg_inv[b, k] = zinv
            z_reg_inv_sqrt[b, k] = wp.sqrt(zinv)
        elif t < n_n_static + m_static + m_n_static:
            # Region 3: G_scaled = z_reg_inv_sqrt[b, i] * G[b, i, j].
            # Recompute the per-row sqrt locally to avoid the in-kernel
            # producer-consumer dependency on Region B's writes.
            k = t - n_n_static - m_static
            i = k // n_static
            j = k % n_static
            G_scaled[b, i, j] = wp.sqrt(wp.float64(1.0) / z_reg[b, i]) * data_G[b, i, j]
        else:
            return

    return update_kkt_kernel, total_dim

