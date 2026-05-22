"""Warp kernel factories for the dense-backend Ruiz preconditioner.

These kernels back ``DenseRuizEquilibration`` (``dense_preconditioner.py``).
Tier-A (backend-agnostic) Ruiz kernels live in
``cupiqp/preconditioner_kernels.py``.

Kernels defined here:
    - create_dense_compute_kkt_norms_kernel   (row/col inf-norms of [P; A; G])
    - create_dense_scale_P_c_kernel           (P and c row/col + cost scaling)
    - create_dense_scale_rect_kernel          (A, G row/col scaling)
    - create_dense_compute_gamma_kernel       (cost-scaling factor gamma)
    - create_dense_apply_gamma_kernel         (apply gamma to P, c, cost_scaling)
"""

import warp as wp
from ..utils import to_warp_dtype


def create_dense_compute_kkt_norms_kernel(n: int, p: int, m: int, dtype=wp.float64):
    dtype = to_warp_dtype(dtype)
    """Fused KKT row/col inf-norm computation for the dense backend.
    """
    @wp.kernel
    def dense_compute_kkt_norms_kernel(
        P:           wp.array3d(dtype=dtype),   # type: ignore  (B, n, n)
        A:           wp.array3d(dtype=dtype),   # type: ignore  (B, p, n) — may be (B,0,n)
        G:           wp.array3d(dtype=dtype),   # type: ignore  (B, m, n) — may be (B,0,n)
        x_b_scaling: wp.array2d(dtype=dtype),   # type: ignore  (B, n)
        d_iter:      wp.array2d(dtype=dtype),   # type: ignore  (B, n+p+m) output
        d_b_iter:    wp.array2d(dtype=dtype),   # type: ignore  (B, n) output
    ):
        b, j = wp.tid()
        n_static = wp.static(n)
        p_static = wp.static(p)
        m_static = wp.static(m)

        if j < n_static:
            v = dtype(0.0)
            for k in range(wp.static(n)):
                v = wp.max(v, wp.abs(P[b, j, k]))
            if wp.static(p > 0):
                for k in range(wp.static(p)):
                    v = wp.max(v, wp.abs(A[b, k, j]))
            if wp.static(m > 0):
                for k in range(wp.static(m)):
                    v = wp.max(v, wp.abs(G[b, k, j]))
            xbs = x_b_scaling[b, j]
            v = wp.max(v, xbs)
            d_iter[b, j] = v
            d_b_iter[b, j] = xbs
        elif j < n_static + p_static:
            jp = j - n_static
            v = dtype(0.0)
            for k in range(wp.static(n)):
                v = wp.max(v, wp.abs(A[b, jp, k]))
            d_iter[b, j] = v
        elif j < n_static + p_static + m_static:
            jm = j - n_static - p_static
            v = dtype(0.0)
            for k in range(wp.static(n)):
                v = wp.max(v, wp.abs(G[b, jm, k]))
            d_iter[b, j] = v
        else:
            return

    return dense_compute_kkt_norms_kernel


def create_dense_scale_P_and_c_kernel(n: int, dtype=wp.float64):
    dtype = to_warp_dtype(dtype)
    """Fused row+col scaling for P (symmetric) and c, plus optional cost factor.

        P[b, i, j] *= d_x[b, i] * d_x[b, j] * cost_scaling_factor[b]
        c[b, i]    *= d_x[b, i]             * cost_scaling_factor[b]    (one thread per (b, i))

    When no cost-scaling should apply, pass a (B,)-array of ones as
    ``cost_scaling_factor`` (the ``self._cost_factor_ones`` buffer).
    """
    @wp.kernel
    def dense_scale_P_and_c_kernel(
        P:        wp.array3d(dtype=dtype),   # type: ignore  (B, n, n) in-out
        c:        wp.array2d(dtype=dtype),   # type: ignore  (B, n)    in-out
        d_x:      wp.array2d(dtype=dtype),   # type: ignore  (B, n)
        cost_scaling_factor: wp.array(dtype=dtype),     # type: ignore  (B,) — ones if no cost factor
    ):
        b, i, j = wp.tid()
        cf = cost_scaling_factor[b]
        P[b, i, j] = P[b, i, j] * d_x[b, i] * d_x[b, j] * cf
        if j == 0:
            c[b, i] = c[b, i] * d_x[b, i] * cf

    return dense_scale_P_and_c_kernel


def create_dense_scale_A_or_G_kernel(rows: int, cols: int, dtype=wp.float64):
    dtype = to_warp_dtype(dtype)
    """Fused row+col scaling for a dense rectangular (B, rows, cols) matrix.

        M[b, i, j] *= d_row[b, i] * d_col[b, j]

    Used once for A (rows=p, cols=n) and once for G (rows=m, cols=n).
    """
    @wp.kernel
    def dense_scale_A_or_G_kernel(
        M:     wp.array3d(dtype=dtype),   # type: ignore  (B, rows, cols) in-out
        d_row: wp.array2d(dtype=dtype),   # type: ignore  (B, rows)
        d_col: wp.array2d(dtype=dtype),   # type: ignore  (B, cols)
    ):
        b, i, j = wp.tid()
        M[b, i, j] = M[b, i, j] * d_row[b, i] * d_col[b, j]

    return dense_scale_A_or_G_kernel


def create_dense_compute_gamma_kernel(n: int,
                                      min_scaling: float, max_scaling: float, dtype=wp.float64):
    dtype = to_warp_dtype(dtype)
    """Per-batch computation of Ruiz cost-scaling factor gamma.

    For each batch b:
        row_col_max[i] = max(
            max_{j >= i} |P[b, i, j]|,
            max_{j <= i} |P[b, j, i]|,
        )                                       # upper-triangular semantics
        g = mean_i(row_col_max[i])
        g = limit(g, min, max)
        g = max(g, max_i |c[b, i]|)
        g = limit(g, min, max)
        gamma[b] = 1 / g
    """
    lo = float(min_scaling)
    hi = float(max_scaling)

    @wp.func
    def _ruiz_limit_scaling(d: dtype, min_scaling: dtype, max_scaling: dtype) -> dtype:
        if d < min_scaling:
            return dtype(1.0)
        if d > max_scaling:
            return max_scaling
        return d

    @wp.kernel
    def dense_compute_gamma_kernel(
        P:     wp.array3d(dtype=dtype),   # type: ignore  (B, n, n)
        c:     wp.array2d(dtype=dtype),   # type: ignore  (B, n)
        gamma: wp.array(dtype=dtype),     # type: ignore  (B,) output
    ):
        b = wp.tid()
        total = dtype(0.0)
        for i in range(wp.static(n)):
            row_max = dtype(0.0)
            col_max = dtype(0.0)
            for j in range(wp.static(n)):
                if j >= i:
                    row_max = wp.max(row_max, wp.abs(P[b, i, j]))
                if j <= i:
                    col_max = wp.max(col_max, wp.abs(P[b, j, i]))
            total = total + wp.max(row_max, col_max)
        g = total / dtype(wp.static(n))
        g = _ruiz_limit_scaling(g, dtype(lo), dtype(hi))

        cn = dtype(0.0)
        for i in range(wp.static(n)):
            cn = wp.max(cn, wp.abs(c[b, i]))
        g = wp.max(g, cn)
        g = _ruiz_limit_scaling(g, dtype(lo), dtype(hi))
        gamma[b] = dtype(1.0) / g

    return dense_compute_gamma_kernel


def create_dense_apply_gamma_kernel(n: int, dtype=wp.float64):
    dtype = to_warp_dtype(dtype)
    """Apply computed gamma to P, c, and accumulate into cost_scaling.

        data._P *= gamma[:, None, None]
        data._c *= gamma[:, None]
        self._cost_scaling *= gamma
    """
    @wp.kernel
    def dense_apply_gamma_kernel(
        P:            wp.array3d(dtype=dtype),   # type: ignore  (B, n, n)
        c:            wp.array2d(dtype=dtype),   # type: ignore  (B, n)
        cost_scaling: wp.array(dtype=dtype),     # type: ignore  (B,)
        gamma:        wp.array(dtype=dtype),     # type: ignore  (B,)
    ):
        b, i, j = wp.tid()
        g = gamma[b]
        P[b, i, j] = P[b, i, j] * g
        if j == 0:
            c[b, i] = c[b, i] * g
            if i == 0:
                cost_scaling[b] = cost_scaling[b] * g

    return dense_apply_gamma_kernel
