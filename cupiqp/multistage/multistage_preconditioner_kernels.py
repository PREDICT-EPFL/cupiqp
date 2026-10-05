"""Warp kernel factories for the multistage-backend Ruiz preconditioner.

Each batched block matrix is represented directly as warp 4D arrays — no
DLPack bridging is needed inside the kernels:

    P :   D   (B, N, d, d)      symmetric block-tridiag
          E   (B, N-1, d, d)    upper = lower^T
    A,G : D   (B, N, r, d)      block lower-bidiagonal
          E   (B, N, r, d)      sub-diagonal blocks

For absent A or G, the call sites pass small dummy 4D buffers and rely on
``wp.static(A_rows_per_block > 0)`` / ``wp.static(G_rows_per_block > 0)`` guards to dead-code-eliminate
all access at codegen.
"""

import warp as wp
from ..utils import to_warp_dtype


def create_multistage_scale_matrices_kernel(N: int, d: int, A_rows_per_block: int, G_rows_per_block: int, dtype=wp.float64):
    dtype = to_warp_dtype(dtype)
    """Single fused kernel for ``scale_matrices``.

    Per ``(b, k, i, j)`` thread (grid (B, N, max(d, A_rows_per_block, G_rows_per_block), d)):
        if i < d   : P_D[b,k,i,j] *= d_x[b, k*d+i] * d_x[b, k*d+j] * cf
                     if k < N-1 : P_E[b,k,i,j] *= d_x[b, (k+1)*d+i] * d_x[b, k*d+j] * cf
                     if i == 0  : c[b, k*d+j] *= d_x[b, k*d+j] * cf
        if i < A_rows_per_block : A_D[b,k,i,j] *= d_y[b, k*A_rows_per_block+i]     * d_x[b, k*d+j]
                     A_E[b,k,i,j] *= d_y[b, (k+1)*A_rows_per_block+i] * d_x[b, k*d+j]
        if i < G_rows_per_block : G_D[b,k,i,j] *= d_z[b, k*G_rows_per_block+i]     * d_x[b, k*d+j]
                     G_E[b,k,i,j] *= d_z[b, (k+1)*G_rows_per_block+i] * d_x[b, k*d+j]

    All shape constants ``(N, d, A_rows_per_block, G_rows_per_block)`` are baked in via ``wp.static``;
    when ``A_rows_per_block == 0`` or ``G_rows_per_block == 0`` the corresponding branches are
    dead-code-eliminated and the dummy A/G arrays passed by the caller are
    never touched.
    """
    @wp.kernel
    def multistage_scale_matrices_kernel(
        P_D:         wp.array4d(dtype=dtype),  # type: ignore  (B, N, d, d) in-out
        P_E:         wp.array4d(dtype=dtype),  # type: ignore  (B, N-1, d, d) in-out
        A_D:         wp.array4d(dtype=dtype),  # type: ignore  (B, N, A_rows_per_block, d) in-out
        A_E:         wp.array4d(dtype=dtype),  # type: ignore  (B, N, A_rows_per_block, d) in-out
        G_D:         wp.array4d(dtype=dtype),  # type: ignore  (B, N, G_rows_per_block, d) in-out
        G_E:         wp.array4d(dtype=dtype),  # type: ignore  (B, N, G_rows_per_block, d) in-out
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

        if wp.static(A_rows_per_block > 0):
            if i < wp.static(A_rows_per_block):
                d_y_ki = d_y[b, k * wp.static(A_rows_per_block) + i]
                d_y_kp1_i = d_y[b, (k + 1) * wp.static(A_rows_per_block) + i]
                A_D[b, k, i, j] = A_D[b, k, i, j] * d_y_ki * d_x_kj
                A_E[b, k, i, j] = A_E[b, k, i, j] * d_y_kp1_i * d_x_kj

        if wp.static(G_rows_per_block > 0):
            if i < wp.static(G_rows_per_block):
                d_z_ki = d_z[b, k * wp.static(G_rows_per_block) + i]
                d_z_kp1_i = d_z[b, (k + 1) * wp.static(G_rows_per_block) + i]
                G_D[b, k, i, j] = G_D[b, k, i, j] * d_z_ki * d_x_kj
                G_E[b, k, i, j] = G_E[b, k, i, j] * d_z_kp1_i * d_x_kj

    return multistage_scale_matrices_kernel


def create_multistage_compute_kkt_norms_kernel(N: int, d: int, A_rows_per_block: int, G_rows_per_block: int, dtype=wp.float64):
    dtype = to_warp_dtype(dtype)
    """Single fused kernel for ``compute_kkt_norms``.

    Per ``(b, j)`` thread (grid (B, n+p+m)) — j addresses one slot of d_iter:
        x-block  j ∈ [0, n)         row inf-norm of P (symmetric)
                                    + col inf-norm of A and G at col j
                                    + max with x_b_scaling[b, j]
        y-block  j ∈ [n, n+p)       row inf-norm of A
        z-block  j ∈ [n+p, n+p+m)   row inf-norm of G

    Each output slot has a unique writer thread — no atomics. All shape
    constants are static; the inner d / A_rows_per_block / G_rows_per_block loops unroll at codegen.
    """
    n = N * d
    p = (N + 1) * A_rows_per_block
    m = (N + 1) * G_rows_per_block

    @wp.kernel
    def multistage_compute_kkt_norms_kernel(
        P_D:         wp.array4d(dtype=dtype),  # type: ignore  (B, N, d, d)
        P_E:         wp.array4d(dtype=dtype),  # type: ignore  (B, N-1, d, d)
        A_D:         wp.array4d(dtype=dtype),  # type: ignore  (B, N, A_rows_per_block, d)
        A_E:         wp.array4d(dtype=dtype),  # type: ignore  (B, N, A_rows_per_block, d)
        G_D:         wp.array4d(dtype=dtype),  # type: ignore  (B, N, G_rows_per_block, d)
        G_E:         wp.array4d(dtype=dtype),  # type: ignore  (B, N, G_rows_per_block, d)
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
            if wp.static(A_rows_per_block > 0):
                for row in range(wp.static(A_rows_per_block)):
                    v = wp.max(v, wp.abs(A_D[b, k_blk, row, i]))
                    v = wp.max(v, wp.abs(A_E[b, k_blk, row, i]))

            # G col j: D[k_blk][:, i] and E[k_blk][:, i]
            if wp.static(G_rows_per_block > 0):
                for row in range(wp.static(G_rows_per_block)):
                    v = wp.max(v, wp.abs(G_D[b, k_blk, row, i]))
                    v = wp.max(v, wp.abs(G_E[b, k_blk, row, i]))

            xbs = x_b_scaling[b, j]
            v = wp.max(v, xbs)
            d_iter[b, j] = v
            d_b_iter[b, j] = xbs

        elif j < np_static:
            # y-block: row inf-norm of A
            if wp.static(A_rows_per_block > 0):
                jp = j - n_static
                k_blk = jp // wp.static(A_rows_per_block)
                i = jp - k_blk * wp.static(A_rows_per_block)

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
            if wp.static(G_rows_per_block > 0):
                jm = j - np_static
                k_blk = jm // wp.static(G_rows_per_block)
                i = jm - k_blk * wp.static(G_rows_per_block)

                v = dtype(0.0)
                if k_blk < N_static:
                    for col in range(d_static):
                        v = wp.max(v, wp.abs(G_D[b, k_blk, i, col]))
                if k_blk > 0:
                    for col in range(d_static):
                        v = wp.max(v, wp.abs(G_E[b, k_blk - 1, i, col]))
                d_iter[b, j] = v

    return multistage_compute_kkt_norms_kernel


def create_multistage_P_col_norms_kernel(N: int, d: int, dtype=wp.float64):
    dtype = to_warp_dtype(dtype)
    """Column inf-norms of the symmetric block-tridiagonal P, per variable.

    Per ``(b, j)`` thread (grid (B, n)), with ``j = k*d + i``::

        out[b, j] = max( max_col |P_D[b, k, i, col]|,
                         max_col |P_E[b, k-1, i, col]|   (k > 0,   lower block (k, k-1))
                         max_row |P_E[b, k, row, i]|     (k < N-1, upper block (k, k+1) = E_k^T) )

    the same P contribution as ``create_multistage_compute_kkt_norms_kernel``.
    """
    @wp.kernel
    def multistage_P_col_norms_kernel(
        P_D: wp.array4d(dtype=dtype),  # type: ignore  (B, N, d, d)
        P_E: wp.array4d(dtype=dtype),  # type: ignore  (B, N-1, d, d)
        out: wp.array2d(dtype=dtype),  # type: ignore  (B, n)
    ):
        b, j = wp.tid()
        N_static = wp.static(N)
        d_static = wp.static(d)
        k_blk = j // d_static
        i = j - k_blk * d_static
        v = dtype(0.0)
        for col in range(d_static):
            v = wp.max(v, wp.abs(P_D[b, k_blk, i, col]))
        if k_blk > 0:
            for col in range(d_static):
                v = wp.max(v, wp.abs(P_E[b, k_blk - 1, i, col]))
        if k_blk < N_static - 1:
            for col in range(d_static):
                v = wp.max(v, wp.abs(P_E[b, k_blk, col, i]))
        out[b, j] = v

    return multistage_P_col_norms_kernel


def create_gamma_from_norms_kernel(min_scaling: float, max_scaling: float, dtype=wp.float64):
    dtype = to_warp_dtype(dtype)
    """Per-batch Ruiz cost-scaling factor from column norms and the linear cost.

        g = mean_j(col_norms[b, j]);  g = limit(g)
        g = max(g, max_j |c[b, j]|);  g = limit(g)
        gamma[b] = 1 / g

    where ``limit`` resets values below ``min_scaling`` to 1 and clamps at
    ``max_scaling``. One thread per batch entry; launch with ``dim=(B,)``.
    """
    lo = float(min_scaling)
    hi = float(max_scaling)

    @wp.func
    def _ruiz_limit_scaling(v: dtype, min_scaling: dtype, max_scaling: dtype) -> dtype:  # type: ignore
        if v < min_scaling:
            return dtype(1.0)
        if v > max_scaling:
            return max_scaling
        return v

    @wp.kernel
    def gamma_from_norms_kernel(
        col_norms: wp.array2d(dtype=dtype),  # type: ignore  (B, n)
        c:         wp.array2d(dtype=dtype),  # type: ignore  (B, n)
        gamma:     wp.array(dtype=dtype),    # type: ignore  (B,) output
    ):
        b = wp.tid()
        n = col_norms.shape[1]
        total = dtype(0.0)
        cn = dtype(0.0)
        for j in range(n):
            total = total + col_norms[b, j]
            cn = wp.max(cn, wp.abs(c[b, j]))
        g = total / dtype(n)
        g = _ruiz_limit_scaling(g, dtype(lo), dtype(hi))
        g = wp.max(g, cn)
        g = _ruiz_limit_scaling(g, dtype(lo), dtype(hi))
        gamma[b] = dtype(1.0) / g

    return gamma_from_norms_kernel
