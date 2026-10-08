"""Warp tile kernels of the batched dense Cholesky factorization and solve.

One CUDA block per matrix; the matrix size ``n`` is a compile-time constant, so
each size gets its own kernel (own module, compiled once per size and dtype).
Only the row-major upper triangle of every matrix is read: ``A = U^T U`` with
``U`` upper triangular, which is the triangle the dense KKT assembly writes.
"""
import functools

import warp as wp


@functools.lru_cache(maxsize=None)
def create_tile_cholesky_factor_kernel(n: int, dtype=wp.float64):
    """In-place ``A[b] = U^T U`` for every ``b``; ``U`` overwrites the upper triangle.

    ``status[b]`` is set to 1 when matrix ``b`` is not numerically positive
    definite (a non-positive or non-finite pivot) and to 0 otherwise.
    Dispatch with ``wp.launch_tiled(..., dim=[B])``.
    """

    @wp.kernel(module="unique", enable_backward=False)
    def tile_cholesky_factor_kernel(
        A: wp.array3d(dtype=dtype),           # type: ignore  (B, n, n)
        status: wp.array(dtype=wp.int32),     # type: ignore  (B,)
    ):
        b = wp.tid()
        a = wp.tile_load(A[b], shape=(n, n))
        wp.tile_cholesky_inplace(a, fill_mode="upper")
        wp.tile_store(A[b], a)
        failed = wp.int32(0)
        for i in range(n):
            d = a[i, i]
            if not (d > dtype(0.0) and wp.isfinite(d)):
                failed = wp.int32(1)
        status[b] = failed

    return tile_cholesky_factor_kernel


@functools.lru_cache(maxsize=None)
def create_tile_cholesky_solve_kernel(n: int, dtype=wp.float64):
    """In-place ``x[b] = A[b]^{-1} x[b]`` with the factors of
    ``create_tile_cholesky_factor_kernel``. Dispatch with
    ``wp.launch_tiled(..., dim=[B])``."""

    @wp.kernel(module="unique", enable_backward=False)
    def tile_cholesky_solve_kernel(
        U: wp.array3d(dtype=dtype),   # type: ignore  (B, n, n), factors
        x: wp.array2d(dtype=dtype),   # type: ignore  (B, n), right-hand sides, overwritten
    ):
        b = wp.tid()
        u = wp.tile_load(U[b], shape=(n, n))
        y = wp.tile_load(x[b], shape=(n,))
        wp.tile_cholesky_solve_inplace(u, y, fill_mode="upper")
        wp.tile_store(x[b], y)

    return tile_cholesky_solve_kernel
