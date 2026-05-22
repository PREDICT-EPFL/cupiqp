"""Warp kernel factories for the multistage-backend Ruiz preconditioner.

Each batched block matrix is represented directly as warp 4D arrays — no
DLPack bridging is needed inside the kernels:

    P :   diag_blocks.data           (B, N, d, d)      symmetric block-tridiag
          off_diag_blocks_lower.data (B, N-1, d, d)    upper = lower^T
    A,G : D                          (B, N, r, d)      block lower-bidiagonal
          E                          (B, N, r, d)      sub-diagonal blocks

For absent A or G, the call sites pass small dummy 4D buffers and rely on
``wp.static(rows_A > 0)`` / ``wp.static(rows_G > 0)`` guards to dead-code-eliminate
all access at codegen.
"""

import warp as wp
from ..utils import to_warp_dtype


def create_multistage_scale_matrices_kernel(N: int, d: int, rows_A: int, rows_G: int, dtype=wp.float64):
    dtype = to_warp_dtype(dtype)
    """Single fused kernel for ``scale_matrices``.

    Per ``(b, k, i, j)`` thread (grid (B, N, max(d, rows_A, rows_G), d)):
        if i < d   : P_D[b,k,i,j] *= d_x[b, k*d+i] * d_x[b, k*d+j] * cf
                     if k < N-1 : P_E[b,k,i,j] *= d_x[b, (k+1)*d+i] * d_x[b, k*d+j] * cf
                     if i == 0  : c[b, k*d+j] *= d_x[b, k*d+j] * cf
        if i < rows_A : A_D[b,k,i,j] *= d_y[b, k*rows_A+i]     * d_x[b, k*d+j]
                     A_E[b,k,i,j] *= d_y[b, (k+1)*rows_A+i] * d_x[b, k*d+j]
        if i < rows_G : G_D[b,k,i,j] *= d_z[b, k*rows_G+i]     * d_x[b, k*d+j]
                     G_E[b,k,i,j] *= d_z[b, (k+1)*rows_G+i] * d_x[b, k*d+j]

    All shape constants ``(N, d, rows_A, rows_G)`` are baked in via ``wp.static``;
    when ``rows_A == 0`` or ``rows_G == 0`` the corresponding branches are
    dead-code-eliminated and the dummy A/G arrays passed by the caller are
    never touched.
    """
    @wp.kernel
    def multistage_scale_matrices_kernel(
        P_D:         wp.array4d(dtype=dtype),  # type: ignore  (B, N, d, d) in-out
        P_E:         wp.array4d(dtype=dtype),  # type: ignore  (B, N-1, d, d) in-out
        A_D:         wp.array4d(dtype=dtype),  # type: ignore  (B, N, rows_A, d) in-out
        A_E:         wp.array4d(dtype=dtype),  # type: ignore  (B, N, rows_A, d) in-out
        G_D:         wp.array4d(dtype=dtype),  # type: ignore  (B, N, rows_G, d) in-out
        G_E:         wp.array4d(dtype=dtype),  # type: ignore  (B, N, rows_G, d) in-out
        c:           wp.array2d(dtype=dtype),  # type: ignore  (B, n) in-out
        d_x:         wp.array2d(dtype=dtype),  # type: ignore  (B, n)
        d_y:         wp.array2d(dtype=dtype),  # type: ignore  (B, p)
        d_z:         wp.array2d(dtype=dtype),  # type: ignore  (B, m)
        cost_factor: wp.array(dtype=dtype),    # type: ignore  (B,)
    ):
        b, k, i, j = wp.tid()
        cf = cost_factor[b]
        d_x_kj = d_x[b, k * wp.static(d) + j]

        if i < wp.static(d):
            d_x_ki = d_x[b, k * wp.static(d) + i]
            P_D[b, k, i, j] = P_D[b, k, i, j] * d_x_ki * d_x_kj * cf
            if k < wp.static(N - 1):
                d_x_kp1_i = d_x[b, (k + 1) * wp.static(d) + i]
                P_E[b, k, i, j] = P_E[b, k, i, j] * d_x_kp1_i * d_x_kj * cf
            if i == 0:
                c[b, k * wp.static(d) + j] = c[b, k * wp.static(d) + j] * d_x_kj * cf

        if wp.static(rows_A > 0):
            if i < wp.static(rows_A):
                d_y_ki = d_y[b, k * wp.static(rows_A) + i]
                d_y_kp1_i = d_y[b, (k + 1) * wp.static(rows_A) + i]
                A_D[b, k, i, j] = A_D[b, k, i, j] * d_y_ki * d_x_kj
                A_E[b, k, i, j] = A_E[b, k, i, j] * d_y_kp1_i * d_x_kj

        if wp.static(rows_G > 0):
            if i < wp.static(rows_G):
                d_z_ki = d_z[b, k * wp.static(rows_G) + i]
                d_z_kp1_i = d_z[b, (k + 1) * wp.static(rows_G) + i]
                G_D[b, k, i, j] = G_D[b, k, i, j] * d_z_ki * d_x_kj
                G_E[b, k, i, j] = G_E[b, k, i, j] * d_z_kp1_i * d_x_kj

    return multistage_scale_matrices_kernel


def create_multistage_compute_kkt_norms_kernel(N: int, d: int, rows_A: int, rows_G: int, dtype=wp.float64):
    dtype = to_warp_dtype(dtype)
    """Single fused kernel for ``compute_kkt_norms``.

    Per ``(b, j)`` thread (grid (B, n+p+m)) — j addresses one slot of d_iter:
        x-block  j ∈ [0, n)         row inf-norm of P (symmetric)
                                    + col inf-norm of A and G at col j
                                    + max with x_b_scaling[b, j]
        y-block  j ∈ [n, n+p)       row inf-norm of A
        z-block  j ∈ [n+p, n+p+m)   row inf-norm of G

    Each output slot has a unique writer thread — no atomics. All shape
    constants are static; the inner d / rows_A / rows_G loops unroll at codegen.
    """
    n = N * d
    p = (N + 1) * rows_A
    m = (N + 1) * rows_G

    @wp.kernel
    def multistage_compute_kkt_norms_kernel(
        P_D:         wp.array4d(dtype=dtype),  # type: ignore  (B, N, d, d)
        P_E:         wp.array4d(dtype=dtype),  # type: ignore  (B, N-1, d, d)
        A_D:         wp.array4d(dtype=dtype),  # type: ignore  (B, N, rows_A, d)
        A_E:         wp.array4d(dtype=dtype),  # type: ignore  (B, N, rows_A, d)
        G_D:         wp.array4d(dtype=dtype),  # type: ignore  (B, N, rows_G, d)
        G_E:         wp.array4d(dtype=dtype),  # type: ignore  (B, N, rows_G, d)
        x_b_scaling: wp.array2d(dtype=dtype),  # type: ignore  (B, n)
        d_iter:      wp.array2d(dtype=dtype),  # type: ignore  (B, n+p+m) out
        d_b_iter:    wp.array2d(dtype=dtype),  # type: ignore  (B, n)     out
    ):
        b, j = wp.tid()
        n_static = wp.static(n)
        np_static = wp.static(n + p)
        npm_static = wp.static(n + p + m)
        N_static = wp.static(N)
        d_static = wp.static(d)

        if j < n_static:
            # x-block: row of full symmetric P + cols of A and G + x_b_scaling
            k_blk = j // d_static
            i = j - k_blk * d_static

            v = dtype(0.0)
            # P diag block: row i of P_D[k_blk]
            for col in range(d_static):
                v = wp.max(v, wp.abs(P_D[b, k_blk, i, col]))
            # Lower off-diag (block (k_blk, k_blk-1)): P_E[k_blk-1] row i
            if k_blk > 0:
                for col in range(d_static):
                    v = wp.max(v, wp.abs(P_E[b, k_blk - 1, i, col]))
            # Upper off-diag (= P_E[k_blk] transposed): |P_E[k_blk, col, i]|
            if k_blk < N_static - 1:
                for col in range(d_static):
                    v = wp.max(v, wp.abs(P_E[b, k_blk, col, i]))

            # A col j: D[k_blk][:, i] and E[k_blk][:, i]
            if wp.static(rows_A > 0):
                for row in range(wp.static(rows_A)):
                    v = wp.max(v, wp.abs(A_D[b, k_blk, row, i]))
                    v = wp.max(v, wp.abs(A_E[b, k_blk, row, i]))

            # G col j: D[k_blk][:, i] and E[k_blk][:, i]
            if wp.static(rows_G > 0):
                for row in range(wp.static(rows_G)):
                    v = wp.max(v, wp.abs(G_D[b, k_blk, row, i]))
                    v = wp.max(v, wp.abs(G_E[b, k_blk, row, i]))

            xbs = x_b_scaling[b, j]
            v = wp.max(v, xbs)
            d_iter[b, j] = v
            d_b_iter[b, j] = xbs

        elif j < np_static:
            # y-block: row inf-norm of A
            if wp.static(rows_A > 0):
                jp = j - n_static
                k_blk = jp // wp.static(rows_A)
                i = jp - k_blk * wp.static(rows_A)

                v = dtype(0.0)
                if k_blk < N_static:
                    for col in range(d_static):
                        v = wp.max(v, wp.abs(A_D[b, k_blk, i, col]))
                if k_blk > 0:
                    for col in range(d_static):
                        v = wp.max(v, wp.abs(A_E[b, k_blk - 1, i, col]))
                d_iter[b, j] = v

        elif j < npm_static:
            # z-block: row inf-norm of G
            if wp.static(rows_G > 0):
                jm = j - np_static
                k_blk = jm // wp.static(rows_G)
                i = jm - k_blk * wp.static(rows_G)

                v = dtype(0.0)
                if k_blk < N_static:
                    for col in range(d_static):
                        v = wp.max(v, wp.abs(G_D[b, k_blk, i, col]))
                if k_blk > 0:
                    for col in range(d_static):
                        v = wp.max(v, wp.abs(G_E[b, k_blk - 1, i, col]))
                d_iter[b, j] = v

    return multistage_compute_kkt_norms_kernel
