"""Warp kernel factories for the sparse-backend Ruiz preconditioner.

The kernels work directly off the CSR triple ``(data, indices, indptr)`` —
no per-nz row-index buffer is materialized; each row-scanning thread is a
``(batch, row)`` pair and walks ``indptr[row] : indptr[row + 1]`` to find
the nz range owned by that row.
"""

import warp as wp


def create_sparse_scale_matrices_kernel(n: int, p: int, m: int):
    """Single fused kernel for ``scale_matrices``: applies row+col Ruiz scaling
    to P, A, G and the linear cost c, plus an optional batchwise cost-scaling
    factor on (P, c), all in one launch.

    Grid (B, R) with ``R = max(n, p, m)``. Per ``(b, r)`` thread:
        if r < n : for k in [P_indptr[r], P_indptr[r+1]):
                       P_data[b, k] *= d_x[b, r] * d_x[b, P_indices[k]] * cf
                   c[b, r] *= d_x[b, r] * cf
        if r < p : for k in [A_indptr[r], A_indptr[r+1]):
                       A_data[b, k] *= d_y[b, r] * d_x[b, A_indices[k]]
        if r < m : for k in [G_indptr[r], G_indptr[r+1]):
                       G_data[b, k] *= d_z[b, r] * d_x[b, G_indices[k]]

    The ``(p, m)`` static guards are baked in via ``wp.static`` so absent
    branches are dead-code-eliminated at codegen — no runtime check.
    Caller passes a (B,)-array of ones as ``cost_factor`` when no cost scaling
    should apply.
    """
    @wp.kernel
    def sparse_scale_matrices_kernel(
        P_data:      wp.array2d(dtype=wp.float64),  # type: ignore  (B, nnz_P) in-out
        P_indptr:    wp.array(dtype=wp.int32),      # type: ignore  (n+1,)
        P_indices:   wp.array(dtype=wp.int32),      # type: ignore  (nnz_P,)
        A_data:      wp.array2d(dtype=wp.float64),  # type: ignore  (B, nnz_A) in-out
        A_indptr:    wp.array(dtype=wp.int32),      # type: ignore  (p+1,)
        A_indices:   wp.array(dtype=wp.int32),      # type: ignore  (nnz_A,)
        G_data:      wp.array2d(dtype=wp.float64),  # type: ignore  (B, nnz_G) in-out
        G_indptr:    wp.array(dtype=wp.int32),      # type: ignore  (m+1,)
        G_indices:   wp.array(dtype=wp.int32),      # type: ignore  (nnz_G,)
        c:           wp.array2d(dtype=wp.float64),  # type: ignore  (B, n) in-out
        d_x:         wp.array2d(dtype=wp.float64),  # type: ignore  (B, n)
        d_y:         wp.array2d(dtype=wp.float64),  # type: ignore  (B, p)
        d_z:         wp.array2d(dtype=wp.float64),  # type: ignore  (B, m)
        cost_factor: wp.array(dtype=wp.float64),    # type: ignore  (B,)
    ):
        b, r = wp.tid()

        if r < wp.static(n):
            d_xr = d_x[b, r]
            start = P_indptr[r]
            end = P_indptr[r + 1]
            for k in range(start, end):
                cc = P_indices[k]
                P_data[b, k] = P_data[b, k] * d_xr * d_x[b, cc] * cost_factor[b]
            c[b, r] = c[b, r] * d_xr * cost_factor[b]

        if wp.static(p > 0):
            if r < wp.static(p):
                d_yr = d_y[b, r]
                start_a = A_indptr[r]
                end_a = A_indptr[r + 1]
                for k in range(start_a, end_a):
                    cc = A_indices[k]
                    A_data[b, k] = A_data[b, k] * d_yr * d_x[b, cc]

        if wp.static(m > 0):
            if r < wp.static(m):
                d_zr = d_z[b, r]
                start_g = G_indptr[r]
                end_g = G_indptr[r + 1]
                for k in range(start_g, end_g):
                    cc = G_indices[k]
                    G_data[b, k] = G_data[b, k] * d_zr * d_x[b, cc]

    return sparse_scale_matrices_kernel


def create_sparse_compute_kkt_norms_kernel(n: int, p: int, m: int):
    """Two fused kernels backing ``compute_kkt_norms``: row scans for
    ``[P; A; G]`` and the per-block ``x_b_scaling`` integration in one pass,
    then a per-nz col scatter that atomically maxes A and G column
    contributions into the x-block of ``d_iter``.

    Returns ``(sparse_compute_row_inf_norm_kernel,
              sparse_compute_col_inf_norm_kernel)``.

    Row kernel — grid (B, max(n, p, m)). Per ``(b, j)``:
        if j < n :  v = max over P_indptr[j]..P_indptr[j+1] of |P_data|
                    v = max(v, x_b_scaling[b, j])
                    d_iter[b, j]   = v
                    d_b_iter[b, j] = x_b_scaling[b, j]
        if j < p :  d_iter[b, n + j]      = max over A row j of |A_data|
        if j < m :  d_iter[b, n + p + j]  = max over G row j of |G_data|

    Col kernel — grid (B, max(nnz_A, nnz_G)). Per ``(b, k)``:
        if k < nnz_A : atomic_max(d_iter[b, A_indices[k]], |A_data[b, k]|)
        if k < nnz_G : atomic_max(d_iter[b, G_indices[k]], |G_data[b, k]|)

    The col kernel is run *after* the row kernel so the plain writes to
    ``d_iter[:, :n]`` are already in place; the atomic_max then folds in
    the A/G column contributions on top, matching the ``cp.maximum.at``
    semantics of the cupy fallback.

    The ``(p, m)`` static guards are baked in via ``wp.static`` so absent
    branches are dead-code-eliminated at codegen.
    """
    @wp.kernel
    def sparse_compute_row_inf_norm_kernel(
        P_data:      wp.array2d(dtype=wp.float64),  # type: ignore  (B, nnz_P)
        P_indptr:    wp.array(dtype=wp.int32),      # type: ignore  (n+1,)
        A_data:      wp.array2d(dtype=wp.float64),  # type: ignore  (B, nnz_A)
        A_indptr:    wp.array(dtype=wp.int32),      # type: ignore  (p+1,)
        G_data:      wp.array2d(dtype=wp.float64),  # type: ignore  (B, nnz_G)
        G_indptr:    wp.array(dtype=wp.int32),      # type: ignore  (m+1,)
        x_b_scaling: wp.array2d(dtype=wp.float64),  # type: ignore  (B, n)
        d_iter:      wp.array2d(dtype=wp.float64),  # type: ignore  (B, n+p+m) out
        d_b_iter:    wp.array2d(dtype=wp.float64),  # type: ignore  (B, n)     out
    ):
        b, j = wp.tid()

        if j < wp.static(n):
            v = wp.float64(0.0)
            start = P_indptr[j]
            end = P_indptr[j + 1]
            for k in range(start, end):
                v = wp.max(v, wp.abs(P_data[b, k]))
            xbs = x_b_scaling[b, j]
            v = wp.max(v, xbs)
            d_iter[b, j] = v
            d_b_iter[b, j] = xbs

        if wp.static(p > 0):
            if j < wp.static(p):
                v = wp.float64(0.0)
                start = A_indptr[j]
                end = A_indptr[j + 1]
                for k in range(start, end):
                    v = wp.max(v, wp.abs(A_data[b, k]))
                d_iter[b, wp.static(n) + j] = v

        if wp.static(m > 0):
            if j < wp.static(m):
                v = wp.float64(0.0)
                start = G_indptr[j]
                end = G_indptr[j + 1]
                for k in range(start, end):
                    v = wp.max(v, wp.abs(G_data[b, k]))
                d_iter[b, wp.static(n + p) + j] = v

    @wp.kernel
    def sparse_compute_col_inf_norm_kernel(
        A_data:    wp.array2d(dtype=wp.float64),  # type: ignore  (B, nnz_A)
        A_indices: wp.array(dtype=wp.int32),      # type: ignore  (nnz_A,)
        G_data:    wp.array2d(dtype=wp.float64),  # type: ignore  (B, nnz_G)
        G_indices: wp.array(dtype=wp.int32),      # type: ignore  (nnz_G,)
        d_iter:    wp.array2d(dtype=wp.float64),  # type: ignore  (B, n+p+m) in-out
    ):
        b, k = wp.tid()
        if wp.static(p > 0):
            if k < A_data.shape[1]:
                cc = A_indices[k]
                wp.atomic_max(d_iter, b, cc, wp.abs(A_data[b, k]))
        if wp.static(m > 0):
            if k < G_data.shape[1]:
                cc = G_indices[k]
                wp.atomic_max(d_iter, b, cc, wp.abs(G_data[b, k]))

    return sparse_compute_row_inf_norm_kernel, sparse_compute_col_inf_norm_kernel
