"""Graph-safe cuBLAS wrappers via nvmath-python bindings.

Both single-precision (``s*``) and double-precision (``d*``) variants
are exposed as separate public functions — each hardcoded to one
precision and using the matching ``ctypes.c_float`` / ``ctypes.c_double``
alpha/beta scalar type. Callers pick the right variant once based on
their dtype (e.g. ``DenseKKTSolver`` does a lazy import in ``__init__``)
so there is zero per-call dispatch overhead.
"""

import ctypes

from nvmath.bindings import cublas


# ---------------------------------------------------------------------------
# Constants (cuBLAS C enum values — kept as ints for backward compat with
# call sites that imported them by name).
# ---------------------------------------------------------------------------
OP_N = 0            # CUBLAS_OP_N  (non-transpose)
OP_T = 1            # CUBLAS_OP_T  (transpose)
FILL_UPPER = 1      # CUBLAS_FILL_MODE_UPPER
SIDE_RIGHT = 1      # CUBLAS_SIDE_RIGHT
POINTER_HOST = 0    # CUBLAS_POINTER_MODE_HOST
POINTER_DEVICE = 1  # CUBLAS_POINTER_MODE_DEVICE


# ---------------------------------------------------------------------------
# Handle/stream management — independent of dtype
# ---------------------------------------------------------------------------
def cublas_create_handle():
    """Create a new cuBLAS handle (thread-safe, independent of CuPy's shared handle)."""
    return cublas.create()


def cublas_destroy_handle(handle):
    """Destroy a cuBLAS handle created by :func:`cublas_create_handle`."""
    cublas.destroy(handle)


def cublas_set_stream(handle, cuda_stream):
    """Associate a CUDA stream with the cuBLAS handle."""
    cublas.set_stream(handle, cuda_stream)


def set_pointer_mode(handle, mode):
    """Set cuBLAS pointer mode (``POINTER_HOST`` or ``POINTER_DEVICE``)."""
    cublas.set_pointer_mode(handle, mode)


# ---------------------------------------------------------------------------
# Helper used by the two GEMV wrappers to compute (m, n, lda, op) from
# a matrix's contiguity and the user's transpose flag.
# ---------------------------------------------------------------------------
def _gemv_layout(mat, transa):
    rows, cols = mat.shape
    if mat.flags["F_CONTIGUOUS"]:
        m, n, lda = rows, cols, rows
        op = OP_N if not transa else OP_T
    else:
        m, n, lda = cols, rows, cols
        op = OP_T if not transa else OP_N
    return m, n, lda, op


# ---------------------------------------------------------------------------
# Helper used by the two GEMM-strided wrappers to compute the cuBLAS
# (column-major) dimensions/strides from row-major (Python) layout.
# Returns the tuple of arguments threaded to cuBLAS.
# ---------------------------------------------------------------------------
def _gemm_strided_layout(A, B, C, transa, transb):
    batch = A.shape[0]
    rA, cA = A.shape[1], A.shape[2]
    rB, cB = B.shape[1], B.shape[2]

    op_a_cm = OP_N if not transa else OP_T
    op_b_cm = OP_N if not transb else OP_T

    if not transb:
        m_blas, k_blas = cB, rB
    else:
        m_blas, k_blas = rB, cB

    if not transa:
        n_blas = rA
    else:
        n_blas = cA

    lda_blas = cB
    ldb_blas = cA
    ldc_blas = C.shape[2]

    # Strides in *elements* — itemsize is dtype-dependent (4 for f32, 8 for f64)
    strideA_blas = B.strides[0] // B.itemsize
    strideB_blas = A.strides[0] // A.itemsize
    strideC_blas = C.strides[0] // C.itemsize

    return (op_b_cm, op_a_cm, m_blas, n_blas, k_blas,
            lda_blas, strideA_blas, ldb_blas, strideB_blas,
            ldc_blas, strideC_blas, batch)


# ===========================================================================
# Double-precision (float64) wrappers
# ===========================================================================
def dgemv(handle, mat, x, y, transa=False, alpha=1.0, beta=0.0):
    """``y = alpha * op(mat) * x + beta * y`` for float64 arrays."""
    m, n, lda, op = _gemv_layout(mat, transa)
    _alpha = ctypes.c_double(alpha)
    _beta = ctypes.c_double(beta)
    cublas.dgemv(
        handle, op, m, n,
        ctypes.addressof(_alpha), mat.data.ptr, lda,
        x.data.ptr, 1,
        ctypes.addressof(_beta), y.data.ptr, 1,
    )


def dcopy(handle, n, x_ptr, incx, y_ptr, incy):
    """``y = x`` (float64 buffers)."""
    cublas.dcopy(handle, n, x_ptr, incx, y_ptr, incy)


def daxpy(handle, n, alpha, x_ptr, incx, y_ptr, incy):
    """``y = alpha * x + y`` (float64). ``alpha`` is a host float or a
    device pointer; the latter triggers POINTER_DEVICE mode."""
    if isinstance(alpha, float):
        _alpha = ctypes.c_double(alpha)
        cublas.daxpy(handle, n, ctypes.addressof(_alpha), x_ptr, incx, y_ptr, incy)
    else:
        cublas.set_pointer_mode(handle, POINTER_DEVICE)
        cublas.daxpy(handle, n, alpha, x_ptr, incx, y_ptr, incy)
        cublas.set_pointer_mode(handle, POINTER_HOST)


def dsyrk(handle, uplo, trans, n, k, alpha, a_ptr, lda, beta, c_ptr, ldc):
    """``C = alpha * op(A) * op(A)^T + beta * C`` (float64)."""
    _alpha = ctypes.c_double(alpha)
    _beta = ctypes.c_double(beta)
    cublas.dsyrk(
        handle, uplo, trans, n, k,
        ctypes.addressof(_alpha), a_ptr, lda,
        ctypes.addressof(_beta), c_ptr, ldc,
    )


def ddgmm(handle, mode, m, n, a_ptr, lda, x_ptr, incx, c_ptr, ldc):
    """``C = diag(x) * A`` or ``A * diag(x)`` (float64)."""
    cublas.ddgmm(handle, mode, m, n, a_ptr, lda, x_ptr, incx, c_ptr, ldc)


def ddot(handle, n, x_ptr, incx, y_ptr, incy, result_ptr):
    """``result = x^T * y`` (float64, result is a device pointer)."""
    cublas.ddot(handle, n, x_ptr, incx, y_ptr, incy, result_ptr)


def dgemm_strided_batched(handle, A, B, C,
                          transa=False, transb=False,
                          alpha=1.0, beta=0.0):
    r"""Batched GEMM on C-contiguous 3-D float64 arrays.

    .. math::

        C_i = \alpha\;\mathrm{op}(A_i)\;B_i + \beta\;C_i
        \qquad i = 0 \ldots \text{batch}-1
    """
    (op_b_cm, op_a_cm, m_blas, n_blas, k_blas,
     lda_blas, strideA_blas, ldb_blas, strideB_blas,
     ldc_blas, strideC_blas, batch) = _gemm_strided_layout(A, B, C, transa, transb)

    _alpha = ctypes.c_double(alpha)
    _beta = ctypes.c_double(beta)
    cublas.dgemm_strided_batched(
        handle,
        op_b_cm, op_a_cm,
        m_blas, n_blas, k_blas,
        ctypes.addressof(_alpha),
        B.data.ptr, lda_blas, strideA_blas,
        A.data.ptr, ldb_blas, strideB_blas,
        ctypes.addressof(_beta),
        C.data.ptr, ldc_blas, strideC_blas,
        batch,
    )


def dgemv_strided_batched(handle, mat, x, y,
                          transa=False, alpha=1.0, beta=0.0):
    r"""Batched matrix-vector product via strided batched GEMM (float64)."""
    batch = mat.shape[0]
    x3 = x.reshape(batch, x.shape[1], 1)
    y3 = y.reshape(batch, y.shape[1], 1)
    dgemm_strided_batched(handle, mat, x3, y3,
                          transa=transa, alpha=alpha, beta=beta)


# ===========================================================================
# Single-precision (float32) wrappers
# ===========================================================================
def sgemv(handle, mat, x, y, transa=False, alpha=1.0, beta=0.0):
    """``y = alpha * op(mat) * x + beta * y`` for float32 arrays."""
    m, n, lda, op = _gemv_layout(mat, transa)
    _alpha = ctypes.c_float(alpha)
    _beta = ctypes.c_float(beta)
    cublas.sgemv(
        handle, op, m, n,
        ctypes.addressof(_alpha), mat.data.ptr, lda,
        x.data.ptr, 1,
        ctypes.addressof(_beta), y.data.ptr, 1,
    )


def scopy(handle, n, x_ptr, incx, y_ptr, incy):
    """``y = x`` (float32 buffers)."""
    cublas.scopy(handle, n, x_ptr, incx, y_ptr, incy)


def saxpy(handle, n, alpha, x_ptr, incx, y_ptr, incy):
    """``y = alpha * x + y`` (float32). ``alpha`` is a host float or a
    device pointer; the latter triggers POINTER_DEVICE mode."""
    if isinstance(alpha, float):
        _alpha = ctypes.c_float(alpha)
        cublas.saxpy(handle, n, ctypes.addressof(_alpha), x_ptr, incx, y_ptr, incy)
    else:
        cublas.set_pointer_mode(handle, POINTER_DEVICE)
        cublas.saxpy(handle, n, alpha, x_ptr, incx, y_ptr, incy)
        cublas.set_pointer_mode(handle, POINTER_HOST)


def ssyrk(handle, uplo, trans, n, k, alpha, a_ptr, lda, beta, c_ptr, ldc):
    """``C = alpha * op(A) * op(A)^T + beta * C`` (float32)."""
    _alpha = ctypes.c_float(alpha)
    _beta = ctypes.c_float(beta)
    cublas.ssyrk(
        handle, uplo, trans, n, k,
        ctypes.addressof(_alpha), a_ptr, lda,
        ctypes.addressof(_beta), c_ptr, ldc,
    )


def sdgmm(handle, mode, m, n, a_ptr, lda, x_ptr, incx, c_ptr, ldc):
    """``C = diag(x) * A`` or ``A * diag(x)`` (float32)."""
    cublas.sdgmm(handle, mode, m, n, a_ptr, lda, x_ptr, incx, c_ptr, ldc)


def sdot(handle, n, x_ptr, incx, y_ptr, incy, result_ptr):
    """``result = x^T * y`` (float32, result is a device pointer)."""
    cublas.sdot(handle, n, x_ptr, incx, y_ptr, incy, result_ptr)


def sgemm_strided_batched(handle, A, B, C,
                          transa=False, transb=False,
                          alpha=1.0, beta=0.0):
    r"""Batched GEMM on C-contiguous 3-D float32 arrays."""
    (op_b_cm, op_a_cm, m_blas, n_blas, k_blas,
     lda_blas, strideA_blas, ldb_blas, strideB_blas,
     ldc_blas, strideC_blas, batch) = _gemm_strided_layout(A, B, C, transa, transb)

    _alpha = ctypes.c_float(alpha)
    _beta = ctypes.c_float(beta)
    cublas.sgemm_strided_batched(
        handle,
        op_b_cm, op_a_cm,
        m_blas, n_blas, k_blas,
        ctypes.addressof(_alpha),
        B.data.ptr, lda_blas, strideA_blas,
        A.data.ptr, ldb_blas, strideB_blas,
        ctypes.addressof(_beta),
        C.data.ptr, ldc_blas, strideC_blas,
        batch,
    )


def sgemv_strided_batched(handle, mat, x, y,
                          transa=False, alpha=1.0, beta=0.0):
    r"""Batched matrix-vector product via strided batched GEMM (float32)."""
    batch = mat.shape[0]
    x3 = x.reshape(batch, x.shape[1], 1)
    y3 = y.reshape(batch, y.shape[1], 1)
    sgemm_strided_batched(handle, mat, x3, y3,
                          transa=transa, alpha=alpha, beta=beta)
