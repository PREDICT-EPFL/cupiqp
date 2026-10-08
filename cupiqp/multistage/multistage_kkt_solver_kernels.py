from typing import Any

import functools

import warp as wp
from ..utils import to_warp_dtype


wp.set_module_options({"enable_backward": False})


@functools.lru_cache(maxsize=None)
def create_update_kkt_kernel(num_blocks: int, block_size: int,
                             p: int, m: int, G_rows_per_block: int, dtype=wp.float64):
    """Build the fused kernel that assembles the condensed multistage KKT matrix.

    For each problem ``b`` of the batch the kernel computes the
    block-tridiagonal matrix

        KKT = P + diag(x_reg) + (1 / delta) * A^T A + G^T diag(z_reg_inv) G

    and writes its ``N`` diagonal blocks to ``KKT_D`` and its ``N - 1`` lower
    off-diagonal blocks to ``KKT_E`` (``KKT_E[k]`` is block ``(k + 1, k)``).
    Every entry is written exactly once, so the outputs need no zeroing.
    ``A^T A`` is not formed here: it is read from the precomputed blocks
    ``AtA_D`` / ``AtA_E``. ``G`` is block-bidiagonal with ``N + 1`` block rows of
    ``G_rows_per_block`` rows each; block row ``k`` holds ``G_D[k]`` on stage
    ``k`` (if ``k < N``) and ``G_E[k - 1]`` on stage ``k - 1`` (if ``k > 0``), so

        KKT_D[k] += G_D[k]^T W_k G_D[k] + G_E[k]^T W_{k+1} G_E[k]
        KKT_E[k] += G_D[k+1]^T W_{k+1} G_E[k]

    where ``W_k`` is the diagonal of ``z_reg_inv`` for block row ``k``.

    Alongside the matrix, the kernel also writes ``delta_inv[b] = 1 / delta[b]``
    and copies ``z_reg_inv`` to ``z_reg_inv_out`` for the later solve.

    Launch with ``dim=(B, N + 1, d, d)``: thread ``(b, k, i, j)`` computes entry
    ``(i, j)`` of ``KKT_D[k]`` (``k < N``) and of ``KKT_E[k]`` (``k < N - 1``).
    The extra ``k = N`` slice exists only to copy the last block row of
    ``z_reg_inv``. Thread ``(b, k, i, 0)`` copies rows ``i, i + d, i + 2d, ...``
    of block row ``k``, so ``G_rows_per_block`` may exceed ``d``.

    Args:
        num_blocks: number of stages ``N`` (diagonal blocks of the KKT matrix).
        block_size: stage size ``d``.
        p, m: total number of equality / inequality rows; the A and G terms
            are compiled out when the count is 0.
        G_rows_per_block: rows per block row of G. Pass any positive value
            (e.g. 1) when ``m == 0``; it is then unused.
        dtype: ``wp.float32`` or ``wp.float64``.
    """
    dtype = to_warp_dtype(dtype)

    @wp.kernel
    def update_kkt_kernel(
        # ---- inputs ----
        P_D:        wp.array4d(dtype=dtype),  # type: ignore   (B, N, d, d)
        P_E:        wp.array4d(dtype=dtype),  # type: ignore   (B, N-1, d, d)
        x_reg:      wp.array2d(dtype=dtype),  # type: ignore   (B, N*d)
        AtA_D:      wp.array4d(dtype=dtype),  # type: ignore   (B, N, d, d)    if p>0
        AtA_E:      wp.array4d(dtype=dtype),  # type: ignore   (B, N-1, d, d)  if p>0
        delta:      wp.array(dtype=dtype),    # type: ignore   (B,)            raw delta
        G_D:        wp.array4d(dtype=dtype),  # type: ignore   (B, N, rg, d)   if m>0, rg means rows of per block of G
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
        G_rows_per_block_static = wp.static(G_rows_per_block)

        delta_inv_b = dtype(1.0) / delta[b]

        # ---- write delta_inv ----
        if k == 0 and i == 0 and j == 0:
            delta_inv[b] = delta_inv_b

        # ---- write z_reg_inv (designated writer per (b, idx), for solve()) ----
        # Threads (b, k, i, 0) for k in [0, N+1), i in [0, d) cover all
        # (N+1)*rg = m elements, thread i copying rows i, i + d, i + 2d, ...
        # of block k, so rg may exceed d. Trailing j > 0 threads skip.
        if wp.static(m > 0):
            if k <= N_static and j == 0:
                for q in range(i, G_rows_per_block_static, d_static):
                    z_reg_inv_out[b, k * G_rows_per_block_static + q] = z_reg_inv[b, k * G_rows_per_block_static + q]
        # ---- diagonal block element (k in [0, N)) ----
        if k < N_static:
            v_D = P_D[b, k, i, j]
            if i == j:
                v_D = v_D + x_reg[b, k * d_static + i]
            if wp.static(p > 0):
                v_D = v_D + delta_inv_b * AtA_D[b, k, i, j]
            if wp.static(m > 0):
                acc_D = dtype(0.0)
                for q in range(G_rows_per_block_static):
                    w_dk = z_reg_inv[b, k * G_rows_per_block_static + q]
                    w_ek = z_reg_inv[b, (k + 1) * G_rows_per_block_static + q]
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
                for q in range(G_rows_per_block_static):
                    w_kp1 = z_reg_inv[b, (k + 1) * G_rows_per_block_static + q]
                    acc_E = acc_E + w_kp1 * G_D[b, k + 1, q, i] * G_E[b, k, q, j]
                v_E = v_E + acc_E
            KKT_E[b, k, i, j] = v_E

    return update_kkt_kernel


@functools.lru_cache(maxsize=None)
def create_add_scaled_rows_kernel(dtype=wp.float64):
    """``out[b, :] += scale[b] * x[b, :]`` on ``(B, n)`` buffers with a per-problem scale."""
    dtype = to_warp_dtype(dtype)

    @wp.kernel
    def add_scaled_rows(out: wp.array2d(dtype=dtype), scale: wp.array(dtype=dtype), x: wp.array2d(dtype=dtype)):  # type: ignore
        i, j = wp.tid()
        out[i, j] = out[i, j] + scale[i] * x[i, j]

    return add_scaled_rows


@functools.lru_cache(maxsize=None)
def create_sub_scale_rows_kernel(dtype=wp.float64):
    """``out[b, :] = (out[b, :] - x[b, :]) * scale[b]`` on ``(B, n)`` buffers with a per-problem scale."""
    dtype = to_warp_dtype(dtype)

    @wp.kernel
    def sub_scale_rows(out: wp.array2d(dtype=dtype), x: wp.array2d(dtype=dtype), scale: wp.array(dtype=dtype)):  # type: ignore
        i, j = wp.tid()
        out[i, j] = (out[i, j] - x[i, j]) * scale[i]

    return sub_scale_rows


@wp.func
def sub_mul(o: Any, x: Any, w: Any):
    # element-wise (o - x) * w, applied with wp.map. A Warp function with
    # generic argument types works for float32 and float64 and, unlike a plain
    # Python function, is not re-parsed on every wp.map call.
    return (o - x) * w


@functools.lru_cache(maxsize=None)
def create_has_nan_rows_kernel(dtype=wp.float64):
    """Sets ``flag[b] = 1`` for every row ``b`` of ``a`` that contains a NaN;
    rows without NaN are left untouched, so clear ``flag`` first. Launch
    with ``dim=a.shape``. Used to detect, per problem, a failed block
    Cholesky factorization (the block factorization writes NaN where a
    pivot block is not positive definite)."""
    dtype = to_warp_dtype(dtype)

    @wp.kernel
    def has_nan_rows(a: wp.array2d(dtype=dtype), flag: wp.array(dtype=wp.int32)):  # type: ignore
        b, i = wp.tid()
        if wp.isnan(a[b, i]):
            flag[b] = wp.int32(1)

    return has_nan_rows
