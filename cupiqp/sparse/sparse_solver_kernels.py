import warp as wp
from ..utils import to_warp_dtype


def create_sparse_data_gradients_kernel(
    nnz_P: int, nnz_A: int, nnz_G: int,
    p: int, m: int, n: int, num_xu: int,
dtype=wp.float64):
    dtype = to_warp_dtype(dtype)
    
    @wp.kernel
    def sparse_data_gradients_kernel(
        # Inputs — user-space lambdas (active sizes) and full-layout scatters.
        lam_x:        wp.array2d(dtype=dtype),  # type: ignore (B, n)
        lam_y:        wp.array2d(dtype=dtype),  # type: ignore (B, p)
        lam_zu_full:  wp.array2d(dtype=dtype),  # type: ignore (B, m)
        lam_zl_full:  wp.array2d(dtype=dtype),  # type: ignore (B, m)
        lam_zbu_full: wp.array2d(dtype=dtype),  # type: ignore (B, n)
        zu_full:      wp.array2d(dtype=dtype),  # type: ignore (B, m)
        zl_full:      wp.array2d(dtype=dtype),  # type: ignore (B, m)
        # Primal / dual solution at the optimum (user space).
        res_x:     wp.array2d(dtype=dtype),  # type: ignore (B, n)
        res_y:     wp.array2d(dtype=dtype),  # type: ignore (B, p)
        # CSR row/col indices for each matrix's nnz positions.
        p_rows:    wp.array(dtype=wp.int32),  # type: ignore (nnz_P,)
        p_indices: wp.array(dtype=wp.int32),  # type: ignore (nnz_P,)
        a_rows:    wp.array(dtype=wp.int32),  # type: ignore (nnz_A,)
        a_indices: wp.array(dtype=wp.int32),  # type: ignore (nnz_A,)
        g_rows:    wp.array(dtype=wp.int32),  # type: ignore (nnz_G,)
        g_indices: wp.array(dtype=wp.int32),  # type: ignore (nnz_G,)
        # Outputs.
        dP_values: wp.array2d(dtype=dtype),  # type: ignore (B, nnz_P)
        dA_values: wp.array2d(dtype=dtype),  # type: ignore (B, nnz_A)
        dG_values: wp.array2d(dtype=dtype),  # type: ignore (B, nnz_G)
        db:        wp.array2d(dtype=dtype),  # type: ignore (B, p)
        dh_u:      wp.array2d(dtype=dtype),  # type: ignore (B, m)
        dx_u:      wp.array2d(dtype=dtype),  # type: ignore (B, num_xu)
    ):
        b, t = wp.tid()
        nnzP_s = wp.static(nnz_P)
        nnzA_s = wp.static(nnz_A)
        nnzG_s = wp.static(nnz_G)
        p_s    = wp.static(p)
        m_s    = wp.static(m)
        n_s    = wp.static(n)
        num_xu_s = wp.static(num_xu)

        end_dP   = nnzP_s
        end_dA   = end_dP + nnzA_s
        end_dG   = end_dA + nnzG_s
        end_db   = end_dG + p_s
        end_dh_u = end_db + m_s
        end_dx_u = end_dh_u + num_xu_s

        if t < end_dP:
            k = t
            i = p_rows[k]
            j = p_indices[k]
            dP_values[b, k] = dtype(0.5) * (
                lam_x[b, i] * res_x[b, j] + res_x[b, i] * lam_x[b, j]
            )

        elif t < end_dA:
            k = t - end_dP
            i = a_rows[k]
            j = a_indices[k]
            dA_values[b, k] = res_y[b, i] * lam_x[b, j] + lam_y[b, i] * res_x[b, j]

        elif t < end_dG:
            k = t - end_dA
            i = g_rows[k]
            j = g_indices[k]
            dG_values[b, k] = (
                (zu_full[b, i] - zl_full[b, i]) * lam_x[b, j]
                + (lam_zu_full[b, i] - lam_zl_full[b, i]) * res_x[b, j]
            )

        elif t < end_db:
            k = t - end_dG
            db[b, k] = -lam_y[b, k]

        elif t < end_dh_u:
            k = t - end_db
            dh_u[b, k] = -lam_zu_full[b, k]

        elif t < end_dx_u:
            j = t - end_dh_u
            dx_u[b, j] = -lam_zbu_full[b, j]

        else:
            return

    return sparse_data_gradients_kernel
