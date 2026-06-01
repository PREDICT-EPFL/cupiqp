import warp as wp
from ..utils import to_warp_dtype


def create_multistage_data_gradients_kernel(
    N: int, d: int,
    N_a: int, r_a: int,
    N_g: int, r_g: int,
    p: int, m: int, n: int,
    dtype=wp.float64
    ):
    dtype = to_warp_dtype(dtype)
    r"""Fused assembly of multistage matrix + vector gradients.

    Single warp launch that writes:

    * ``dP_diag`` (``B, N, d, d``)   — symmetric outer product on each
      stage's diagonal block.
    * ``dP_offdiag_lower`` (``B, N-1, d, d``) — accumulated outer
      product for each stored lower off-diagonal block and its implicit
      transposed upper block.
    * ``dA_D, dA_E`` (``B, N_a, r_a, d``) — bidiag outer products for
      ``A`` at the diagonal and sub-diagonal block positions.
    * ``dG_D, dG_E`` (``B, N_g, r_g, d``) — same for ``G``.
    * ``dc, db, dh_u, dh_l, dx_u, dx_l`` — vector grads, written as
      flat ``(B, k)`` arrays that alias the underlying ``BlockVec.data``
      buffers via DLPack reshape.

    Mapping conventions (all in flat user-space layout):

    * Stage ``k`` of ``x`` occupies flat indices ``[k*d, (k+1)*d)``.
    * Stage ``k`` row-block of ``A x`` occupies ``[k*r_a, (k+1)*r_a)``;
      ``A`` has ``N`` D-blocks at (k, k) and ``N`` E-blocks at (k+1, k).
    * Same for ``G`` with ``r_g``.

    Absent constraint groups (``p == 0`` ⇒ ``N_a = r_a = 0``;
    ``m == 0`` ⇒ ``N_g = r_g = 0``) collapse their sub-ranges to size
    0 — callers must still pass valid (possibly empty) wp.array
    arguments for those inputs/outputs.
    """
    # Pre-compute compile-time region sizes / cumulative offsets.
    N_off = max(N - 1, 0)
    sz_dP_diag    = N * d * d
    sz_dP_offdiag = N_off * d * d
    sz_dA_D       = N_a * r_a * d
    sz_dA_E       = N_a * r_a * d
    sz_dG_D       = N_g * r_g * d
    sz_dG_E       = N_g * r_g * d

    end_dP_diag    = sz_dP_diag
    end_dP_offdiag = end_dP_diag    + sz_dP_offdiag
    end_dA_D       = end_dP_offdiag + sz_dA_D
    end_dA_E       = end_dA_D       + sz_dA_E
    end_dG_D       = end_dA_E       + sz_dG_D
    end_dG_E       = end_dG_D       + sz_dG_E
    end_dc         = end_dG_E       + n
    end_db         = end_dc         + p
    end_dh_u       = end_db         + m
    end_dh_l       = end_dh_u       + m
    end_dx_u       = end_dh_l       + n
    end_dx_l       = end_dx_u       + n

    @wp.kernel
    def multistage_data_gradients_kernel(
        # Inputs — user-space lambdas (active sizes) and full-layout scatters.
        lam_x:        wp.array2d(dtype=dtype),  # type: ignore (B, n)
        lam_y:        wp.array2d(dtype=dtype),  # type: ignore (B, p)
        lam_zu_full:  wp.array2d(dtype=dtype),  # type: ignore (B, m)
        lam_zl_full:  wp.array2d(dtype=dtype),  # type: ignore (B, m)
        lam_zbu_full: wp.array2d(dtype=dtype),  # type: ignore (B, n)
        lam_zbl_full: wp.array2d(dtype=dtype),  # type: ignore (B, n)
        zu_full:      wp.array2d(dtype=dtype),  # type: ignore (B, m)
        zl_full:      wp.array2d(dtype=dtype),  # type: ignore (B, m)
        # Primal / dual solution at the optimum (user space, flat).
        res_x: wp.array2d(dtype=dtype),  # type: ignore (B, n)
        res_y: wp.array2d(dtype=dtype),  # type: ignore (B, p)
        # Outputs — block-structured matrix grads.
        dP_diag:           wp.array4d(dtype=dtype),  # type: ignore (B, N, d, d)
        dP_offdiag_lower:  wp.array4d(dtype=dtype),  # type: ignore (B, N-1, d, d)
        dA_D:              wp.array4d(dtype=dtype),  # type: ignore (B, N_a, r_a, d)
        dA_E:              wp.array4d(dtype=dtype),  # type: ignore (B, N_a, r_a, d)
        dG_D:              wp.array4d(dtype=dtype),  # type: ignore (B, N_g, r_g, d)
        dG_E:              wp.array4d(dtype=dtype),  # type: ignore (B, N_g, r_g, d)
        # Outputs — flat (B, k) vector grads (aliased to BlockVec.data).
        dc:   wp.array2d(dtype=dtype),  # type: ignore (B, n)
        db:   wp.array2d(dtype=dtype),  # type: ignore (B, p)
        dh_u: wp.array2d(dtype=dtype),  # type: ignore (B, m)
        dh_l: wp.array2d(dtype=dtype),  # type: ignore (B, m)
        dx_u: wp.array2d(dtype=dtype),  # type: ignore (B, n)
        dx_l: wp.array2d(dtype=dtype),  # type: ignore (B, n)
    ):
        b, t = wp.tid()
        d_s   = wp.static(d)
        ra_s  = wp.static(r_a)
        rg_s  = wp.static(r_g)
        dd_s  = wp.static(d * d)
        rad_s = wp.static(r_a * d)
        rgd_s = wp.static(r_g * d)

        e0  = wp.static(end_dP_diag)
        e1  = wp.static(end_dP_offdiag)
        e2  = wp.static(end_dA_D)
        e3  = wp.static(end_dA_E)
        e4  = wp.static(end_dG_D)
        e5  = wp.static(end_dG_E)
        e6  = wp.static(end_dc)
        e7  = wp.static(end_db)
        e8  = wp.static(end_dh_u)
        e9  = wp.static(end_dh_l)
        e10 = wp.static(end_dx_u)
        e11 = wp.static(end_dx_l)

        if t < e0:
            # dP_diag[b, k, i, j] = ½ (λ_x[k*d+i]·x[k*d+j] + x[k*d+i]·λ_x[k*d+j])
            k = t // dd_s
            r = t %  dd_s
            i = r // d_s
            j = r %  d_s
            dP_diag[b, k, i, j] = dtype(0.5) * (
                lam_x[b, k * d_s + i] * res_x[b, k * d_s + j]
                + res_x[b, k * d_s + i] * lam_x[b, k * d_s + j]
            )

        elif t < e1:
            # A stored lower off-diagonal block also parameterizes its
            # implicit transposed upper block, so both contributions add.
            idx = t - e0
            k = idx // dd_s
            r = idx %  dd_s
            i = r // d_s
            j = r %  d_s
            dP_offdiag_lower[b, k, i, j] = (
                lam_x[b, (k + 1) * d_s + i] * res_x[b, k * d_s + j]
                + res_x[b, (k + 1) * d_s + i] * lam_x[b, k * d_s + j]
            )

        elif t < e2:
            # dA_D[b, k, i, j] = y[k*r_a+i]·λ_x[k*d+j] + λ_y[k*r_a+i]·x[k*d+j]
            idx = t - e1
            k = idx // rad_s
            r = idx %  rad_s
            i = r // d_s
            j = r %  d_s
            dA_D[b, k, i, j] = (
                res_y[b, k * ra_s + i] * lam_x[b, k * d_s + j]
                + lam_y[b, k * ra_s + i] * res_x[b, k * d_s + j]
            )

        elif t < e3:
            # dA_E[b, k, i, j] = y[(k+1)*r_a+i]·λ_x[k*d+j] + λ_y[(k+1)*r_a+i]·x[k*d+j]
            idx = t - e2
            k = idx // rad_s
            r = idx %  rad_s
            i = r // d_s
            j = r %  d_s
            dA_E[b, k, i, j] = (
                res_y[b, (k + 1) * ra_s + i] * lam_x[b, k * d_s + j]
                + lam_y[b, (k + 1) * ra_s + i] * res_x[b, k * d_s + j]
            )

        elif t < e4:
            # dG_D[b, k, i, j]
            idx = t - e3
            k = idx // rgd_s
            r = idx %  rgd_s
            i = r // d_s
            j = r %  d_s
            dG_D[b, k, i, j] = (
                (zu_full[b, k * rg_s + i] - zl_full[b, k * rg_s + i]) * lam_x[b, k * d_s + j]
                + (lam_zu_full[b, k * rg_s + i] - lam_zl_full[b, k * rg_s + i]) * res_x[b, k * d_s + j]
            )

        elif t < e5:
            # dG_E[b, k, i, j]
            idx = t - e4
            k = idx // rgd_s
            r = idx %  rgd_s
            i = r // d_s
            j = r %  d_s
            dG_E[b, k, i, j] = (
                (zu_full[b, (k + 1) * rg_s + i] - zl_full[b, (k + 1) * rg_s + i]) * lam_x[b, k * d_s + j]
                + (lam_zu_full[b, (k + 1) * rg_s + i] - lam_zl_full[b, (k + 1) * rg_s + i]) * res_x[b, k * d_s + j]
            )

        elif t < e6:
            k = t - e5
            dc[b, k] = lam_x[b, k]

        elif t < e7:
            k = t - e6
            db[b, k] = -lam_y[b, k]

        elif t < e8:
            k = t - e7
            dh_u[b, k] = -lam_zu_full[b, k]

        elif t < e9:
            k = t - e8
            dh_l[b, k] = lam_zl_full[b, k]

        elif t < e10:
            k = t - e9
            dx_u[b, k] = -lam_zbu_full[b, k]

        elif t < e11:
            k = t - e10
            dx_l[b, k] = lam_zbl_full[b, k]

        else:
            return

    return multistage_data_gradients_kernel
