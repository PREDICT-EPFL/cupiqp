"""Warp kernels of the small batched matrix products in ``dense_matmul.py``."""
import functools

import warp as wp
from ..utils import to_warp_dtype


wp.set_module_options({"enable_backward": False})


@functools.lru_cache(maxsize=None)
def create_batched_gemv_kernel(transpose: bool, dtype=wp.float64):
    """Batched matrix-vector product for small matrices, one thread per output entry.

    - ``transpose=False``: ``z[b, i] = alpha * sum_k mat[b, i, k] * x[b, k]``
    - ``transpose=True`` : ``z[b, j] = alpha * sum_k mat[b, k, j] * x[b, k]``

    ``mat`` is ``(B, rows, cols)``; ``x`` and ``z`` are ``(B, k)`` arrays and may
    be row-strided views. ``z`` is overwritten, never read. Dispatch with
    ``wp.launch(..., dim=(B, len_z))``.
    """
    dtype = to_warp_dtype(dtype)

    @wp.kernel
    def batched_gemv_kernel(
        mat:   wp.array3d(dtype=dtype),  # type: ignore  (B, rows, cols)
        x:     wp.array2d(dtype=dtype),  # type: ignore  (B, cols), or (B, rows) if transpose
        alpha: dtype,                    # type: ignore  scale of the product
        z:     wp.array2d(dtype=dtype),  # type: ignore  (B, rows), or (B, cols) if transpose
    ):
        b, i = wp.tid()
        acc = dtype(0.0)
        if wp.static(transpose):
            for k in range(mat.shape[1]):
                acc += mat[b, k, i] * x[b, k]
        else:
            for k in range(mat.shape[2]):
                acc += mat[b, i, k] * x[b, k]
        z[b, i] = alpha * acc

    return batched_gemv_kernel


@functools.lru_cache(maxsize=None)
def create_batched_syrk_kernel(accumulate: bool, dtype=wp.float64):
    """Batched ``C = A^T A`` (or ``C += A^T A`` with ``accumulate``) for small
    matrices, one thread per entry of ``C``.

    Only the upper triangle (``i <= j``) of the row-major ``C`` is written:
    the triangle the dense Cholesky solvers read (cuSOLVER FILL_MODE_LOWER
    in its column-major view). The other triangle is left untouched.

    ``A`` is ``(B, k, n)`` and ``C`` is ``(B, n, n)``. Without ``accumulate``,
    ``C`` is overwritten and never read, so it may hold uninitialized values.
    Dispatch with ``wp.launch(..., dim=(B, n, n))``.
    """
    dtype = to_warp_dtype(dtype)

    @wp.kernel
    def batched_syrk_kernel(
        A: wp.array3d(dtype=dtype),  # type: ignore  (B, k, n)
        C: wp.array3d(dtype=dtype),  # type: ignore  (B, n, n)
    ):
        b, i, j = wp.tid()
        if i > j:
            return
        acc = dtype(0.0)
        for k in range(A.shape[1]):
            acc += A[b, k, i] * A[b, k, j]
        if wp.static(accumulate):
            C[b, i, j] = C[b, i, j] + acc
        else:
            C[b, i, j] = acc

    return batched_syrk_kernel
