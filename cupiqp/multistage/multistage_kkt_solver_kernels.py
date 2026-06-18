import warp as wp
from ..utils import to_warp_dtype


def create_update_kkt_kernel(num_blocks: int, block_size: int,
                             p: int, m: int, rows_of_G: int, dtype=wp.float64):
    dtype = to_warp_dtype(dtype)
    """Fused warp kernel that builds the entire condensed KKT matrix
    and also writes ``delta_inv`` and a cached copy of ``z_reg_inv``.

    Perform the followings:

        delta_inv     <- 1 / delta                     (cupy)
        x_reg copy    <- ipm-provided x_reg            (cupy)
        z_reg_inv copy <- ipm-provided z_reg_inv         (cupy)
        KKT_D <- 0,  KKT_E <- 0
        KKT  <- KKT + P                                (block_tridiag_gead)
        KKT_D <- KKT_D + diag(x_reg)                   (block_tridiag_diaad)
        KKT  <- KKT + (1/delta) * A^T A                (block_tridiag_gead)
        KKT  <- KKT + G^T diag(z_reg_inv) G             (weighted block SYRK)
    """
    @wp.kernel
    def update_kkt_kernel(
        # ---- inputs ----
        P_D:        wp.array4d(dtype=dtype),  # type: ignore   (B, N, d, d)
        P_E:        wp.array4d(dtype=dtype),  # type: ignore   (B, N-1, d, d)
        x_reg:      wp.array2d(dtype=dtype),  # type: ignore   (B, N*d)
        AtA_D:      wp.array4d(dtype=dtype),  # type: ignore   (B, N, d, d)    if p>0
        AtA_E:      wp.array4d(dtype=dtype),  # type: ignore   (B, N-1, d, d)  if p>0
        delta:      wp.array(dtype=dtype),    # type: ignore   (B,)            raw delta
        G_D:        wp.array4d(dtype=dtype),  # type: ignore   (B, N, rg, d)   if m>0
        G_E:        wp.array4d(dtype=dtype),  # type: ignore   (B, N, rg, d)   if m>0
        z_reg_inv:   wp.array2d(dtype=dtype),  # type: ignore   (B, (N+1)*rg)  if m>0
        # ---- outputs ----
        KKT_D:      wp.array4d(dtype=dtype),  # type: ignore   (B, N, d, d)
        KKT_E:      wp.array4d(dtype=dtype),  # type: ignore   (B, N-1, d, d)
        delta_inv:  wp.array(dtype=dtype),    # type: ignore   (B,)            output
        z_reg_inv_out: wp.array2d(dtype=dtype),  # type: ignore   (B, (N+1)*rg)   output, if m>0
    ):
        b, k, i, j = wp.tid()
        N_static = wp.static(num_blocks)
        d_static = wp.static(block_size)
        rows_G_static = wp.static(rows_of_G)

        delta_inv_b = dtype(1.0) / delta[b]

        # ---- write delta_inv ----
        if k == 0 and i == 0 and j == 0:
            delta_inv[b] = delta_inv_b

        # ---- write z_reg_inv (designated writer per (b, idx), for solve()) ----
        # Threads (b, k, i, 0) for k in [0, N+1), i in [0, rg) cover all
        # (N+1)*rg = m elements. Trailing j > 0 / i >= rg threads skip.
        if wp.static(m > 0):
            if k <= N_static and i < rows_G_static and j == 0:
                z_reg_inv_out[b, k * rows_G_static + i] = z_reg_inv[b, k * rows_G_static + i]

        # ---- diagonal block element (k in [0, N)) ----
        if k < N_static:
            v_D = P_D[b, k, i, j]
            if i == j:
                v_D = v_D + x_reg[b, k * d_static + i]
            if wp.static(p > 0):
                v_D = v_D + delta_inv_b * AtA_D[b, k, i, j]
            if wp.static(m > 0):
                acc_D = dtype(0.0)
                for q in range(rows_G_static):
                    w_dk = z_reg_inv[b, k * rows_G_static + q]
                    w_ek = z_reg_inv[b, (k + 1) * rows_G_static + q]
                    acc_D = acc_D + w_dk * G_D[b, k, q, i] * G_D[b, k, q, j]
                    acc_D = acc_D + w_ek * G_E[b, k, q, i] * G_E[b, k, q, j]
                v_D = v_D + acc_D
            KKT_D[b, k, i, j] = v_D

        # ---- off-diagonal block element (only k < N-1) ----
        if k < N_static - 1:
            v_E = P_E[b, k, i, j]
            if wp.static(p > 0):
                v_E = v_E + delta_inv_b * AtA_E[b, k, i, j]
            if wp.static(m > 0):
                acc_E = dtype(0.0)
                for q in range(rows_G_static):
                    w_kp1 = z_reg_inv[b, (k + 1) * rows_G_static + q]
                    acc_E = acc_E + w_kp1 * G_D[b, k + 1, q, i] * G_E[b, k, q, j]
                v_E = v_E + acc_E
            KKT_E[b, k, i, j] = v_E

    return update_kkt_kernel
