"""Graph-safe cuBLAS wrappers via ctypes.

Loads the cuBLAS shared library once at module level and provides thin
Python wrappers around the C functions. No ``check_status`` is called
after any cuBLAS call, making all functions safe for CUDA stream capture
(``check_status`` picks up stale CUDA errors from Warp's legacy-stream
memory frees during graph capture on a blocking stream).

Both single-precision (``s*``) and double-precision (``d*``) variants
are exposed as separate public functions — each hardcoded to one
precision and using the matching ``ctypes.c_float`` / ``ctypes.c_double``
alpha/beta scalar type. Callers pick the right variant once based on
their dtype (e.g. ``DenseKKTSolver`` does a lazy import in ``__init__``)
so there is zero per-call dispatch overhead.
"""

import ctypes
import ctypes.util


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OP_N = 0            # CUBLAS_OP_N  (non-transpose)
OP_T = 1            # CUBLAS_OP_T  (transpose)
FILL_UPPER = 1      # CUBLAS_FILL_MODE_UPPER
SIDE_RIGHT = 1      # CUBLAS_SIDE_RIGHT
POINTER_HOST = 0    # CUBLAS_POINTER_MODE_HOST
POINTER_DEVICE = 1  # CUBLAS_POINTER_MODE_DEVICE


# ---------------------------------------------------------------------------
# Load cuBLAS shared library once (module level)
# ---------------------------------------------------------------------------
def _load_cublas_lib() -> ctypes.CDLL:
    """Load the cuBLAS shared library via ctypes."""
    try:
        import cupy.cuda.runtime as rt
        major = rt.runtimeGetVersion() // 1000  # e.g. 12040 -> 12
        versioned = f"libcublas.so.{major}"
        try:
            return ctypes.CDLL(versioned)
        except OSError:
            pass
    except Exception:
        pass

    for name in ("libcublas.so", "cublas"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    lib_path = ctypes.util.find_library("cublas")
    if lib_path:
        return ctypes.CDLL(lib_path)
    raise RuntimeError("Could not find cuBLAS shared library")


_lib = _load_cublas_lib()


# ---------------------------------------------------------------------------
# Bind both single- (S) and double-precision (D) symbols. Argument
# types are identical between S and D — only the alpha/beta scalar
# buffer width differs (handled in each wrapper).
# ---------------------------------------------------------------------------
def _bind_gemv(sym):
    sym.restype = ctypes.c_int
    sym.argtypes = [
        ctypes.c_void_p,  # handle
        ctypes.c_int,     # trans
        ctypes.c_int,     # m
        ctypes.c_int,     # n
        ctypes.c_void_p,  # alpha  (pointer)
        ctypes.c_void_p,  # A
        ctypes.c_int,     # lda
        ctypes.c_void_p,  # x
        ctypes.c_int,     # incx
        ctypes.c_void_p,  # beta   (pointer)
        ctypes.c_void_p,  # y
        ctypes.c_int,     # incy
    ]
    return sym


def _bind_copy(sym):
    sym.restype = ctypes.c_int
    sym.argtypes = [
        ctypes.c_void_p,  # handle
        ctypes.c_int,     # n
        ctypes.c_void_p,  # x
        ctypes.c_int,     # incx
        ctypes.c_void_p,  # y
        ctypes.c_int,     # incy
    ]
    return sym


def _bind_axpy(sym):
    sym.restype = ctypes.c_int
    sym.argtypes = [
        ctypes.c_void_p,  # handle
        ctypes.c_int,     # n
        ctypes.c_void_p,  # alpha  (pointer)
        ctypes.c_void_p,  # x
        ctypes.c_int,     # incx
        ctypes.c_void_p,  # y
        ctypes.c_int,     # incy
    ]
    return sym


def _bind_syrk(sym):
    sym.restype = ctypes.c_int
    sym.argtypes = [
        ctypes.c_void_p,  # handle
        ctypes.c_int,     # uplo
        ctypes.c_int,     # trans
        ctypes.c_int,     # n
        ctypes.c_int,     # k
        ctypes.c_void_p,  # alpha  (pointer)
        ctypes.c_void_p,  # A
        ctypes.c_int,     # lda
        ctypes.c_void_p,  # beta   (pointer)
        ctypes.c_void_p,  # C
        ctypes.c_int,     # ldc
    ]
    return sym


def _bind_dgmm(sym):
    sym.restype = ctypes.c_int
    sym.argtypes = [
        ctypes.c_void_p,  # handle
        ctypes.c_int,     # mode
        ctypes.c_int,     # m
        ctypes.c_int,     # n
        ctypes.c_void_p,  # A
        ctypes.c_int,     # lda
        ctypes.c_void_p,  # x
        ctypes.c_int,     # incx
        ctypes.c_void_p,  # C
        ctypes.c_int,     # ldc
    ]
    return sym


def _bind_dot(sym):
    sym.restype = ctypes.c_int
    sym.argtypes = [
        ctypes.c_void_p,  # handle
        ctypes.c_int,     # n
        ctypes.c_void_p,  # x
        ctypes.c_int,     # incx
        ctypes.c_void_p,  # y
        ctypes.c_int,     # incy
        ctypes.c_void_p,  # result (pointer)
    ]
    return sym


def _bind_gemm_strided(sym):
    sym.restype = ctypes.c_int
    sym.argtypes = [
        ctypes.c_void_p,     # handle
        ctypes.c_int,        # transa
        ctypes.c_int,        # transb
        ctypes.c_int,        # m
        ctypes.c_int,        # n
        ctypes.c_int,        # k
        ctypes.c_void_p,     # alpha (host pointer)
        ctypes.c_void_p,     # A
        ctypes.c_int,        # lda
        ctypes.c_longlong,   # strideA
        ctypes.c_void_p,     # B
        ctypes.c_int,        # ldb
        ctypes.c_longlong,   # strideB
        ctypes.c_void_p,     # beta (host pointer)
        ctypes.c_void_p,     # C
        ctypes.c_int,        # ldc
        ctypes.c_longlong,   # strideC
        ctypes.c_int,        # batchCount
    ]
    return sym


_dgemv_sym = _bind_gemv(_lib.cublasDgemv_v2)
_sgemv_sym = _bind_gemv(_lib.cublasSgemv_v2)

_dcopy_sym = _bind_copy(_lib.cublasDcopy_v2)
_scopy_sym = _bind_copy(_lib.cublasScopy_v2)

_daxpy_sym = _bind_axpy(_lib.cublasDaxpy_v2)
_saxpy_sym = _bind_axpy(_lib.cublasSaxpy_v2)

_dsyrk_sym = _bind_syrk(_lib.cublasDsyrk_v2)
_ssyrk_sym = _bind_syrk(_lib.cublasSsyrk_v2)

_ddgmm_sym = _bind_dgmm(_lib.cublasDdgmm)
_sdgmm_sym = _bind_dgmm(_lib.cublasSdgmm)

_ddot_sym = _bind_dot(_lib.cublasDdot_v2)
_sdot_sym = _bind_dot(_lib.cublasSdot_v2)

_dgemm_strided_sym = _bind_gemm_strided(_lib.cublasDgemmStridedBatched)
_sgemm_strided_sym = _bind_gemm_strided(_lib.cublasSgemmStridedBatched)


# ---------------------------------------------------------------------------
# Handle/stream management — independent of dtype
# ---------------------------------------------------------------------------
_set_pointer_mode = _lib.cublasSetPointerMode_v2
_set_pointer_mode.restype = ctypes.c_int
_set_pointer_mode.argtypes = [ctypes.c_void_p, ctypes.c_int]

_set_stream = _lib.cublasSetStream_v2
_set_stream.restype = ctypes.c_int
_set_stream.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

_create = _lib.cublasCreate_v2
_create.restype = ctypes.c_int
_create.argtypes = [ctypes.POINTER(ctypes.c_void_p)]

_destroy = _lib.cublasDestroy_v2
_destroy.restype = ctypes.c_int
_destroy.argtypes = [ctypes.c_void_p]


def cublas_create_handle():
    """Create a new cuBLAS handle (thread-safe, independent of CuPy's shared handle)."""
    handle = ctypes.c_void_p()
    status = _create(ctypes.byref(handle))
    if status != 0:
        raise RuntimeError(f"cublasCreate failed with status {status}")
    return handle.value


def cublas_destroy_handle(handle):
    """Destroy a cuBLAS handle created by :func:`cublas_create_handle`."""
    status = _destroy(handle)
    if status != 0:
        raise RuntimeError(f"cublasDestroy failed with status {status}")


def cublas_set_stream(handle, cuda_stream):
    """Associate a CUDA stream with the cuBLAS handle."""
    status = _set_stream(handle, cuda_stream)
    if status != 0:
        raise RuntimeError(f"cublasSetStream failed with status {status}")


def set_pointer_mode(handle, mode):
    """Set cuBLAS pointer mode (``POINTER_HOST`` or ``POINTER_DEVICE``)."""
    _set_pointer_mode(handle, mode)


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
    _dgemv_sym(
        handle, op, m, n,
        ctypes.addressof(_alpha), mat.data.ptr, lda,
        x.data.ptr, 1,
        ctypes.addressof(_beta), y.data.ptr, 1,
    )


def dcopy(handle, n, x_ptr, incx, y_ptr, incy):
    """``y = x`` (float64 buffers)."""
    _dcopy_sym(handle, n, x_ptr, incx, y_ptr, incy)


def daxpy(handle, n, alpha, x_ptr, incx, y_ptr, incy):
    """``y = alpha * x + y`` (float64). ``alpha`` is a host float or a
    device pointer; the latter triggers POINTER_DEVICE mode."""
    if isinstance(alpha, float):
        _alpha = ctypes.c_double(alpha)
        _daxpy_sym(handle, n, ctypes.addressof(_alpha), x_ptr, incx, y_ptr, incy)
    else:
        _set_pointer_mode(handle, POINTER_DEVICE)
        _daxpy_sym(handle, n, alpha, x_ptr, incx, y_ptr, incy)
        _set_pointer_mode(handle, POINTER_HOST)


def dsyrk(handle, uplo, trans, n, k, alpha, a_ptr, lda, beta, c_ptr, ldc):
    """``C = alpha * op(A) * op(A)^T + beta * C`` (float64)."""
    _alpha = ctypes.c_double(alpha)
    _beta = ctypes.c_double(beta)
    _dsyrk_sym(
        handle, uplo, trans, n, k,
        ctypes.addressof(_alpha), a_ptr, lda,
        ctypes.addressof(_beta), c_ptr, ldc,
    )


def ddgmm(handle, mode, m, n, a_ptr, lda, x_ptr, incx, c_ptr, ldc):
    """``C = diag(x) * A`` or ``A * diag(x)`` (float64)."""
    _ddgmm_sym(handle, mode, m, n, a_ptr, lda, x_ptr, incx, c_ptr, ldc)


def ddot(handle, n, x_ptr, incx, y_ptr, incy, result_ptr):
    """``result = x^T * y`` (float64, result is a device pointer)."""
    _ddot_sym(handle, n, x_ptr, incx, y_ptr, incy, result_ptr)


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
    _dgemm_strided_sym(
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
    _sgemv_sym(
        handle, op, m, n,
        ctypes.addressof(_alpha), mat.data.ptr, lda,
        x.data.ptr, 1,
        ctypes.addressof(_beta), y.data.ptr, 1,
    )


def scopy(handle, n, x_ptr, incx, y_ptr, incy):
    """``y = x`` (float32 buffers)."""
    _scopy_sym(handle, n, x_ptr, incx, y_ptr, incy)


def saxpy(handle, n, alpha, x_ptr, incx, y_ptr, incy):
    """``y = alpha * x + y`` (float32). ``alpha`` is a host float or a
    device pointer; the latter triggers POINTER_DEVICE mode."""
    if isinstance(alpha, float):
        _alpha = ctypes.c_float(alpha)
        _saxpy_sym(handle, n, ctypes.addressof(_alpha), x_ptr, incx, y_ptr, incy)
    else:
        _set_pointer_mode(handle, POINTER_DEVICE)
        _saxpy_sym(handle, n, alpha, x_ptr, incx, y_ptr, incy)
        _set_pointer_mode(handle, POINTER_HOST)


def ssyrk(handle, uplo, trans, n, k, alpha, a_ptr, lda, beta, c_ptr, ldc):
    """``C = alpha * op(A) * op(A)^T + beta * C`` (float32)."""
    _alpha = ctypes.c_float(alpha)
    _beta = ctypes.c_float(beta)
    _ssyrk_sym(
        handle, uplo, trans, n, k,
        ctypes.addressof(_alpha), a_ptr, lda,
        ctypes.addressof(_beta), c_ptr, ldc,
    )


def sdgmm(handle, mode, m, n, a_ptr, lda, x_ptr, incx, c_ptr, ldc):
    """``C = diag(x) * A`` or ``A * diag(x)`` (float32)."""
    _sdgmm_sym(handle, mode, m, n, a_ptr, lda, x_ptr, incx, c_ptr, ldc)


def sdot(handle, n, x_ptr, incx, y_ptr, incy, result_ptr):
    """``result = x^T * y`` (float32, result is a device pointer)."""
    _sdot_sym(handle, n, x_ptr, incx, y_ptr, incy, result_ptr)


def sgemm_strided_batched(handle, A, B, C,
                          transa=False, transb=False,
                          alpha=1.0, beta=0.0):
    r"""Batched GEMM on C-contiguous 3-D float32 arrays."""
    (op_b_cm, op_a_cm, m_blas, n_blas, k_blas,
     lda_blas, strideA_blas, ldb_blas, strideB_blas,
     ldc_blas, strideC_blas, batch) = _gemm_strided_layout(A, B, C, transa, transb)

    _alpha = ctypes.c_float(alpha)
    _beta = ctypes.c_float(beta)
    _sgemm_strided_sym(
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
