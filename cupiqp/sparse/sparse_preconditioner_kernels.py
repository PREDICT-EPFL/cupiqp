"""Warp kernel factories for the sparse-backend Ruiz preconditioner.

The kernels work directly off the CSR triple ``(data, indices, indptr)`` —
no per-nz row-index buffer is materialized; each row-scanning thread is a
``(batch, row)`` pair and walks ``indptr[row] : indptr[row + 1]`` to find
the nz range owned by that row.
"""

import warp as wp
from ..utils import to_warp_dtype


def create_sparse_scale_matrices_kernel(n: int, p: int, m: int, dtype=wp.float64):
    dtype = to_warp_dtype(dtype)
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
        P_data:      wp.array2d(dtype=dtype),  # type: ignore  (B, nnz_P) in-out
        P_indptr:    wp.array(dtype=wp.int32),      # type: ignore  (n+1,)
        P_indices:   wp.array(dtype=wp.int32),      # type: ignore  (nnz_P,)
        A_data:      wp.array2d(dtype=dtype),  # type: ignore  (B, nnz_A) in-out
        A_indptr:    wp.array(dtype=wp.int32),      # type: ignore  (p+1,)
        A_indices:   wp.array(dtype=wp.int32),      # type: ignore  (nnz_A,)
        G_data:      wp.array2d(dtype=dtype),  # type: ignore  (B, nnz_G) in-out
        G_indptr:    wp.array(dtype=wp.int32),      # type: ignore  (m+1,)
        G_indices:   wp.array(dtype=wp.int32),      # type: ignore  (nnz_G,)
        c:           wp.array2d(dtype=dtype),  # type: ignore  (B, n) in-out
        d_x:         wp.array2d(dtype=dtype),  # type: ignore  (B, n)
        d_y:         wp.array2d(dtype=dtype),  # type: ignore  (B, p)
        d_z:         wp.array2d(dtype=dtype),  # type: ignore  (B, m)
        cost_factor: wp.array(dtype=dtype),    # type: ignore  (B,)
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


def create_sparse_compute_kkt_norms_kernel(n: int, p: int, m: int, dtype=wp.float64):
    dtype = to_warp_dtype(dtype)
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
        P_data:      wp.array2d(dtype=dtype),  # type: ignore  (B, nnz_P)
        P_indptr:    wp.array(dtype=wp.int32),      # type: ignore  (n+1,)
        A_data:      wp.array2d(dtype=dtype),  # type: ignore  (B, nnz_A)
        A_indptr:    wp.array(dtype=wp.int32),      # type: ignore  (p+1,)
        G_data:      wp.array2d(dtype=dtype),  # type: ignore  (B, nnz_G)
        G_indptr:    wp.array(dtype=wp.int32),      # type: ignore  (m+1,)
        x_b_scaling: wp.array2d(dtype=dtype),  # type: ignore  (B, n)
        d_iter:      wp.array2d(dtype=dtype),  # type: ignore  (B, n+p+m) out
        d_b_iter:    wp.array2d(dtype=dtype),  # type: ignore  (B, n)     out
    ):
        b, j = wp.tid()

        if j < wp.static(n):
            v = dtype(0.0)
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
                v = dtype(0.0)
                start = A_indptr[j]
                end = A_indptr[j + 1]
                for k in range(start, end):
                    v = wp.max(v, wp.abs(A_data[b, k]))
                d_iter[b, wp.static(n) + j] = v

        if wp.static(m > 0):
            if j < wp.static(m):
                v = dtype(0.0)
                start = G_indptr[j]
                end = G_indptr[j + 1]
                for k in range(start, end):
                    v = wp.max(v, wp.abs(G_data[b, k]))
                d_iter[b, wp.static(n + p) + j] = v

    @wp.kernel
    def sparse_compute_col_inf_norm_kernel(
        A_data:    wp.array2d(dtype=dtype),  # type: ignore  (B, nnz_A)
        A_indices: wp.array(dtype=wp.int32),      # type: ignore  (nnz_A,)
        G_data:    wp.array2d(dtype=dtype),  # type: ignore  (B, nnz_G)
        G_indices: wp.array(dtype=wp.int32),      # type: ignore  (nnz_G,)
        d_iter:    wp.array2d(dtype=dtype),  # type: ignore  (B, n+p+m) in-out
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


def create_sparse_P_norms_kernel(dtype=wp.float64):
    dtype = to_warp_dtype(dtype)
    """Per-index inf-norms of P treating the stored triangle symmetrically:
    ``norms[b, i] = max(max_j |P[b, i, j]|, max_j |P[b, j, i]|)`` over the
    stored entries. ``norms`` must be zeroed first. Launch with
    ``dim=(B, nnz_P)``."""

    @wp.kernel
    def sparse_P_norms_kernel(
        P_data:    wp.array2d(dtype=dtype),   # type: ignore  (B, nnz_P)
        P_rows:    wp.array(dtype=wp.int32),  # type: ignore  (nnz_P,)
        P_indices: wp.array(dtype=wp.int32),  # type: ignore  (nnz_P,)
        norms:     wp.array2d(dtype=dtype),   # type: ignore  (B, n) in-out
    ):
        b, k = wp.tid()
        v = wp.abs(P_data[b, k])
        wp.atomic_max(norms, b, P_rows[k], v)
        wp.atomic_max(norms, b, P_indices[k], v)

    return sparse_P_norms_kernel


def create_sparse_compute_gamma_kernel(min_scaling: float, max_scaling: float, dtype=wp.float64):
    dtype = to_warp_dtype(dtype)
    """Per-problem cost-scaling factor from the P column norms and c:

        g = clip(mean_i(norms[b, i]), min, max)
        g = clip(max(g, max_i |c[b, i]|), min, max)
        gamma[b] = 1 / g

    Dispatch with ``wp.launch_tiled(dim=[B], block_dim=REDUCTION_BLOCK_DIM)``."""
    lo = float(min_scaling)
    hi = float(max_scaling)

    @wp.kernel
    def sparse_compute_gamma_kernel(
        norms: wp.array2d(dtype=dtype),  # type: ignore  (B, n)
        c:     wp.array2d(dtype=dtype),  # type: ignore  (B, n)
        gamma: wp.array(dtype=dtype),    # type: ignore  (B,) output
    ):
        b, i = wp.tid()
        n = norms.shape[1]
        acc_sum = dtype(0.0)
        acc_max = dtype(0.0)
        for k in range(i, n, wp.block_dim()):
            acc_sum += norms[b, k]
            acc_max = wp.max(acc_max, wp.abs(c[b, k]))
        total = wp.tile_sum(wp.tile(acc_sum))
        c_max = wp.tile_max(wp.tile(acc_max))
        if i == 0:
            g = total[0] / dtype(n)
            g = wp.clamp(g, dtype(lo), dtype(hi))
            g = wp.max(g, c_max[0])
            g = wp.clamp(g, dtype(lo), dtype(hi))
            gamma[b] = dtype(1.0) / g

    return sparse_compute_gamma_kernel


def create_sparse_apply_gamma_kernel(dtype=wp.float64):
    dtype = to_warp_dtype(dtype)
    """Scale P values and c by ``gamma[b]`` and accumulate it into the cost
    scaling. Launch with ``dim=(B, max(nnz_P, n))``."""

    @wp.kernel
    def sparse_apply_gamma_kernel(
        P_data:       wp.array2d(dtype=dtype),  # type: ignore  (B, nnz_P) in-out
        c:            wp.array2d(dtype=dtype),  # type: ignore  (B, n) in-out
        cost_scaling: wp.array(dtype=dtype),    # type: ignore  (B,) in-out
        gamma:        wp.array(dtype=dtype),    # type: ignore  (B,)
    ):
        b, i = wp.tid()
        g = gamma[b]
        if i < P_data.shape[1]:
            P_data[b, i] = P_data[b, i] * g
        if i < c.shape[1]:
            c[b, i] = c[b, i] * g
        if i == 0:
            cost_scaling[b] = cost_scaling[b] * g

    return sparse_apply_gamma_kernel
