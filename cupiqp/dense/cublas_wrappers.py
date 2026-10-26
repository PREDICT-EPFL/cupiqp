"""Graph-safe cuBLAS wrappers via nvmath-python bindings.

Both single-precision (``s*``) and double-precision (``d*``) variants
are exposed as separate public functions -- each hardcoded to one
precision and using the matching ``ctypes.c_float`` / ``ctypes.c_double``
alpha/beta scalar type. Callers pick the right variant once based on
their dtype (e.g. ``DenseKKTSolver`` does a lazy import in ``__init__``)
so there is zero per-call dispatch overhead.

Array arguments are C-contiguous (row-major) Warp arrays of the wrapper's
dtype. Batched operands may have an arbitrary batch stride (row-strided
views). Only the address, shape and batch stride are read; nothing is
validated: the solver checks user data once at setup/update.
"""

import ctypes
from typing import Union

import warp as wp
from nvmath.bindings import cublas


# ---------------------------------------------------------------------------
# Constants (cuBLAS C enum values -- kept as ints for backward compat with
# call sites that imported them by name).
# ---------------------------------------------------------------------------
OP_N = 0            # CUBLAS_OP_N  (non-transpose)
OP_T = 1            # CUBLAS_OP_T  (transpose)
FILL_LOWER = 0      # CUBLAS_FILL_MODE_LOWER
FILL_UPPER = 1      # CUBLAS_FILL_MODE_UPPER
SIDE_RIGHT = 1      # CUBLAS_SIDE_RIGHT
POINTER_HOST = 0    # CUBLAS_POINTER_MODE_HOST
POINTER_DEVICE = 1  # CUBLAS_POINTER_MODE_DEVICE


# ---------------------------------------------------------------------------
# Handle/stream management -- independent of dtype
# ---------------------------------------------------------------------------
def cublas_create_handle() -> int:
    """Create a new cuBLAS handle (thread-safe, independent of CuPy's shared handle)."""
    return cublas.create()


def cublas_destroy_handle(handle: int) -> None:
    """Destroy a cuBLAS handle created by :func:`cublas_create_handle`."""
    cublas.destroy(handle)


def cublas_set_stream(handle: int, cuda_stream: int) -> None:
    """Associate a CUDA stream with the cuBLAS handle."""
    cublas.set_stream(handle, cuda_stream)


def set_pointer_mode(handle: int, mode: int) -> None:
    """Set cuBLAS pointer mode (``POINTER_HOST`` or ``POINTER_DEVICE``)."""
    cublas.set_pointer_mode(handle, mode)


# ---------------------------------------------------------------------------
# Layout helpers. cuBLAS is column-major, so it sees each row-major matrix
# as its transpose with ``ld = cols``; the wrappers below flip the operation
# flags accordingly.
# ---------------------------------------------------------------------------
def _gemv_layout(mat: wp.array2d, transa: bool) -> tuple:
    """``(ptr, m, n, lda, op)`` for a row-major ``(rows, cols)`` matrix."""
    rows, cols = int(mat.shape[0]), int(mat.shape[1])
    op = OP_T if not transa else OP_N
    return mat.ptr, cols, rows, max(cols, 1), op


def _batched_matrix(a: wp.array3d) -> tuple:
    """``(ptr, rows, cols, ld, batch_stride, batch)`` of a row-major ``(B, rows, cols)`` array (elements)."""
    rows, cols = int(a.shape[1]), int(a.shape[2])
    itemsize = wp.types.type_size_in_bytes(a.dtype)
    return a.ptr, rows, cols, max(cols, 1), a.strides[0] // itemsize, int(a.shape[0])


def _batched_vector(x: wp.array2d) -> tuple:
    """``(ptr, rows, cols, ld, batch_stride, batch)`` of a ``(B, k)`` vector batch, described to cuBLAS as ``(B, k, 1)`` column matrices."""
    itemsize = wp.types.type_size_in_bytes(x.dtype)
    return x.ptr, int(x.shape[1]), 1, 1, x.strides[0] // itemsize, int(x.shape[0])


def _gemm_strided_args(A: tuple, B: tuple, C: tuple, transa: bool, transb: bool) -> tuple:
    """cuBLAS arguments for the batched ``C_i = op(A_i) op(B_i)`` on row-major
    operands (each a ``(ptr, rows, cols, ld, batch_stride, batch)`` tuple).

    Row-major ``C = op(A) op(B)`` is column-major ``C^T = op(B)^T op(A)^T``,
    so the operands are swapped and cuBLAS' ``m, n, k`` are ``C``'s cols,
    rows and the contraction length.
    """
    a_ptr, rA, cA, lda, sA, batch = A
    b_ptr, rB, cB, ldb, sB, _ = B
    c_ptr, _, _, ldc, sC, _ = C
    op_a_cm = OP_N if not transa else OP_T
    op_b_cm = OP_N if not transb else OP_T
    m_blas, k_blas = (cB, rB) if not transb else (rB, cB)
    n_blas = rA if not transa else cA
    return (op_b_cm, op_a_cm, m_blas, n_blas, k_blas,
            b_ptr, ldb, sB, a_ptr, lda, sA, c_ptr, ldc, sC, batch)


# ===========================================================================
# Double-precision (float64) wrappers
# ===========================================================================
def dgemv(handle: int, mat: wp.array2d, x: wp.array1d, y: wp.array1d,
          transa: bool = False, alpha: float = 1.0, beta: float = 0.0) -> None:
    """``y = alpha * op(mat) * x + beta * y`` for float64 arrays."""
    ptr, m, n, lda, op = _gemv_layout(mat, transa)
    _alpha = ctypes.c_double(alpha)
    _beta = ctypes.c_double(beta)
    cublas.dgemv(
        handle, op, m, n,
        ctypes.addressof(_alpha), ptr, lda,
        x.ptr, 1,
        ctypes.addressof(_beta), y.ptr, 1,
    )


def dcopy(handle: int, n: int, x_ptr: int, incx: int, y_ptr: int, incy: int) -> None:
    """``y = x`` (float64 buffers)."""
    cublas.dcopy(handle, n, x_ptr, incx, y_ptr, incy)


def daxpy(handle: int, n: int, alpha: Union[float, int], x_ptr: int, incx: int, y_ptr: int, incy: int) -> None:
    """``y = alpha * x + y`` (float64). ``alpha`` is a host float or a
    device pointer; the latter triggers POINTER_DEVICE mode."""
    if isinstance(alpha, float):
        _alpha = ctypes.c_double(alpha)
        cublas.daxpy(handle, n, ctypes.addressof(_alpha), x_ptr, incx, y_ptr, incy)
    else:
        cublas.set_pointer_mode(handle, POINTER_DEVICE)
        cublas.daxpy(handle, n, alpha, x_ptr, incx, y_ptr, incy)
        cublas.set_pointer_mode(handle, POINTER_HOST)


def dsyrk(handle: int, uplo: int, trans: int, n: int, k: int, alpha: float, a_ptr: int, lda: int,
          beta: float, c_ptr: int, ldc: int) -> None:
    """``C = alpha * op(A) * op(A)^T + beta * C`` (float64)."""
    _alpha = ctypes.c_double(alpha)
    _beta = ctypes.c_double(beta)
    cublas.dsyrk(
        handle, uplo, trans, n, k,
        ctypes.addressof(_alpha), a_ptr, lda,
        ctypes.addressof(_beta), c_ptr, ldc,
    )


def ddgmm(handle: int, mode: int, m: int, n: int, a_ptr: int, lda: int, x_ptr: int, incx: int,
          c_ptr: int, ldc: int) -> None:
    """``C = diag(x) * A`` or ``A * diag(x)`` (float64)."""
    cublas.ddgmm(handle, mode, m, n, a_ptr, lda, x_ptr, incx, c_ptr, ldc)


def ddot(handle: int, n: int, x_ptr: int, incx: int, y_ptr: int, incy: int, result_ptr: int) -> None:
    """``result = x^T * y`` (float64, result is a device pointer)."""
    cublas.ddot(handle, n, x_ptr, incx, y_ptr, incy, result_ptr)


def _dgemm_strided(handle: int, A: tuple, B: tuple, C: tuple,
                   transa: bool, transb: bool, alpha: float, beta: float) -> None:
    (op_b_cm, op_a_cm, m_blas, n_blas, k_blas,
     b_ptr, ldb, sB, a_ptr, lda, sA, c_ptr, ldc, sC, batch) = _gemm_strided_args(A, B, C, transa, transb)
    _alpha = ctypes.c_double(alpha)
    _beta = ctypes.c_double(beta)
    cublas.dgemm_strided_batched(
        handle, op_b_cm, op_a_cm, m_blas, n_blas, k_blas,
        ctypes.addressof(_alpha), b_ptr, ldb, sB, a_ptr, lda, sA,
        ctypes.addressof(_beta), c_ptr, ldc, sC, batch,
    )


def dgemm_strided_batched(handle: int, A: wp.array3d, B: wp.array3d, C: wp.array3d,
                          transa: bool = False, transb: bool = False,
                          alpha: float = 1.0, beta: float = 0.0) -> None:
    r"""Batched GEMM on row-major 3-D float64 arrays: ``C_i = alpha * op(A_i) op(B_i) + beta * C_i``."""
    _dgemm_strided(handle, _batched_matrix(A), _batched_matrix(B), _batched_matrix(C),
                   transa, transb, alpha, beta)


def dgemv_strided_batched(handle: int, mat: wp.array3d, x: wp.array2d, y: wp.array2d,
                          transa: bool = False, alpha: float = 1.0, beta: float = 0.0) -> None:
    r"""Batched matrix-vector product via strided batched GEMM (float64)."""
    _dgemm_strided(handle, _batched_matrix(mat), _batched_vector(x), _batched_vector(y),
                   transa, False, alpha, beta)


# ===========================================================================
# Single-precision (float32) wrappers
# ===========================================================================
def sgemv(handle: int, mat: wp.array2d, x: wp.array1d, y: wp.array1d,
          transa: bool = False, alpha: float = 1.0, beta: float = 0.0) -> None:
    """``y = alpha * op(mat) * x + beta * y`` for float32 arrays."""
    ptr, m, n, lda, op = _gemv_layout(mat, transa)
    _alpha = ctypes.c_float(alpha)
    _beta = ctypes.c_float(beta)
    cublas.sgemv(
        handle, op, m, n,
        ctypes.addressof(_alpha), ptr, lda,
        x.ptr, 1,
        ctypes.addressof(_beta), y.ptr, 1,
    )


def scopy(handle: int, n: int, x_ptr: int, incx: int, y_ptr: int, incy: int) -> None:
    """``y = x`` (float32 buffers)."""
    cublas.scopy(handle, n, x_ptr, incx, y_ptr, incy)


def saxpy(handle: int, n: int, alpha: Union[float, int], x_ptr: int, incx: int, y_ptr: int, incy: int) -> None:
    """``y = alpha * x + y`` (float32). ``alpha`` is a host float or a
    device pointer; the latter triggers POINTER_DEVICE mode."""
    if isinstance(alpha, float):
        _alpha = ctypes.c_float(alpha)
        cublas.saxpy(handle, n, ctypes.addressof(_alpha), x_ptr, incx, y_ptr, incy)
    else:
        cublas.set_pointer_mode(handle, POINTER_DEVICE)
        cublas.saxpy(handle, n, alpha, x_ptr, incx, y_ptr, incy)
        cublas.set_pointer_mode(handle, POINTER_HOST)


def ssyrk(handle: int, uplo: int, trans: int, n: int, k: int, alpha: float, a_ptr: int, lda: int,
          beta: float, c_ptr: int, ldc: int) -> None:
    """``C = alpha * op(A) * op(A)^T + beta * C`` (float32)."""
    _alpha = ctypes.c_float(alpha)
    _beta = ctypes.c_float(beta)
    cublas.ssyrk(
        handle, uplo, trans, n, k,
        ctypes.addressof(_alpha), a_ptr, lda,
        ctypes.addressof(_beta), c_ptr, ldc,
    )


def sdgmm(handle: int, mode: int, m: int, n: int, a_ptr: int, lda: int, x_ptr: int, incx: int,
          c_ptr: int, ldc: int) -> None:
    """``C = diag(x) * A`` or ``A * diag(x)`` (float32)."""
    cublas.sdgmm(handle, mode, m, n, a_ptr, lda, x_ptr, incx, c_ptr, ldc)


def sdot(handle: int, n: int, x_ptr: int, incx: int, y_ptr: int, incy: int, result_ptr: int) -> None:
    """``result = x^T * y`` (float32, result is a device pointer)."""
    cublas.sdot(handle, n, x_ptr, incx, y_ptr, incy, result_ptr)


def _sgemm_strided(handle: int, A: tuple, B: tuple, C: tuple,
                   transa: bool, transb: bool, alpha: float, beta: float) -> None:
    (op_b_cm, op_a_cm, m_blas, n_blas, k_blas,
     b_ptr, ldb, sB, a_ptr, lda, sA, c_ptr, ldc, sC, batch) = _gemm_strided_args(A, B, C, transa, transb)
    _alpha = ctypes.c_float(alpha)
    _beta = ctypes.c_float(beta)
    cublas.sgemm_strided_batched(
        handle, op_b_cm, op_a_cm, m_blas, n_blas, k_blas,
        ctypes.addressof(_alpha), b_ptr, ldb, sB, a_ptr, lda, sA,
        ctypes.addressof(_beta), c_ptr, ldc, sC, batch,
    )


def sgemm_strided_batched(handle: int, A: wp.array3d, B: wp.array3d, C: wp.array3d,
                          transa: bool = False, transb: bool = False,
                          alpha: float = 1.0, beta: float = 0.0) -> None:
    r"""Batched GEMM on row-major 3-D float32 arrays: ``C_i = alpha * op(A_i) op(B_i) + beta * C_i``."""
    _sgemm_strided(handle, _batched_matrix(A), _batched_matrix(B), _batched_matrix(C),
                   transa, transb, alpha, beta)


def sgemv_strided_batched(handle: int, mat: wp.array3d, x: wp.array2d, y: wp.array2d,
                          transa: bool = False, alpha: float = 1.0, beta: float = 0.0) -> None:
    r"""Batched matrix-vector product via strided batched GEMM (float32)."""
    _sgemm_strided(handle, _batched_matrix(mat), _batched_vector(x), _batched_vector(y),
                   transa, False, alpha, beta)
