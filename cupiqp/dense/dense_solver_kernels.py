import warp as wp
from ..utils import to_warp_dtype


def create_dense_data_gradients_kernel(n: int, p: int, m: int, num_xu: int, dtype=wp.float64):
    dtype = to_warp_dtype(dtype)

    @wp.kernel
    def dense_data_gradients_kernel(
        # Inputs — user-space lambdas (active sizes) and full-layout scatters.
        lam_x:        wp.array2d(dtype=dtype),  # type: ignore (B, n)
        lam_y:        wp.array2d(dtype=dtype),  # type: ignore (B, p)
        lam_zu_full:  wp.array2d(dtype=dtype),  # type: ignore (B, m)
        lam_zl_full:  wp.array2d(dtype=dtype),  # type: ignore (B, m)
        lam_zbu_full: wp.array2d(dtype=dtype),  # type: ignore (B, n)
        zu_full:      wp.array2d(dtype=dtype),  # type: ignore (B, m)
        zl_full:      wp.array2d(dtype=dtype),  # type: ignore (B, m)
        # Primal / dual solution at the optimum (user space).
        res_x: wp.array2d(dtype=dtype),  # type: ignore (B, n)
        res_y: wp.array2d(dtype=dtype),  # type: ignore (B, p)
        # Outputs.
        dP:   wp.array3d(dtype=dtype),  # type: ignore (B, n, n)
        dA:   wp.array3d(dtype=dtype),  # type: ignore (B, p, n)
        dG:   wp.array3d(dtype=dtype),  # type: ignore (B, m, n)
        db:   wp.array2d(dtype=dtype),  # type: ignore (B, p)
        dh_u: wp.array2d(dtype=dtype),  # type: ignore (B, m)
        dx_u: wp.array2d(dtype=dtype),  # type: ignore (B, num_xu)
    ):
        b, t = wp.tid()
        n_s = wp.static(n)
        p_s = wp.static(p)
        m_s = wp.static(m)
        num_xu_s = wp.static(num_xu)

        end_dP   = n_s * n_s
        end_dA   = end_dP + p_s * n_s
        end_dG   = end_dA + m_s * n_s
        end_db   = end_dG + p_s
        end_dh_u = end_db + m_s
        end_dx_u = end_dh_u + num_xu_s

        if t < end_dP:
            # dP[b, i, j] = 0.5 (λ_x[b,i]·x[b,j] + x[b,i]·λ_x[b,j])
            i = t // n_s
            j = t %  n_s
            dP[b, i, j] = dtype(0.5) * (
                lam_x[b, i] * res_x[b, j] + res_x[b, i] * lam_x[b, j]
            )

        elif t < end_dA:
            # dA[b, k, i] = y[b,k]·λ_x[b,i] + λ_y[b,k]·x[b,i]
            idx = t - end_dP
            k = idx // n_s
            i = idx %  n_s
            dA[b, k, i] = res_y[b, k] * lam_x[b, i] + lam_y[b, k] * res_x[b, i]

        elif t < end_dG:
            # dG[b, k, i] = (zu^full-zl^full)[b,k]·λ_x[b,i] + (λ_zu^full-λ_zl^full)[b,k]·x[b,i]
            idx = t - end_dA
            k = idx // n_s
            i = idx %  n_s
            dG[b, k, i] = (
                (zu_full[b, k] - zl_full[b, k]) * lam_x[b, i]
                + (lam_zu_full[b, k] - lam_zl_full[b, k]) * res_x[b, i]
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

    return dense_data_gradients_kernel
