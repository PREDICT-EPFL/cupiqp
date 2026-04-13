"""Graph-safe cuBLAS wrappers via ctypes.

Loads the cuBLAS shared library once at module level and provides thin
Python wrappers around the C functions.  No ``check_status`` is called
after any cuBLAS call, making all functions safe for CUDA stream capture
(``check_status`` picks up stale CUDA errors from Warp's legacy-stream
memory frees during graph capture on a blocking stream).

The ``_v2`` symbol suffix (e.g. ``cublasDgemv_v2``) only appears in the
module-level ctypes setup; every wrapper function uses a clean name.
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
    # Try the version matching the active CUDA runtime first.
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

    # Fallback: unversioned symlink, then find_library.
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
# ctypes function signatures  (_v2 symbols aliased to clean names)
# ---------------------------------------------------------------------------
_dgemv = _lib.cublasDgemv_v2
_dgemv.restype = ctypes.c_int
_dgemv.argtypes = [
    ctypes.c_void_p,  # handle
    ctypes.c_int,     # trans
    ctypes.c_int,     # m
    ctypes.c_int,     # n
    ctypes.c_void_p,  # alpha  (pointer to double)
    ctypes.c_void_p,  # A      (device pointer)
    ctypes.c_int,     # lda
    ctypes.c_void_p,  # x      (device pointer)
    ctypes.c_int,     # incx
    ctypes.c_void_p,  # beta   (pointer to double)
    ctypes.c_void_p,  # y      (device pointer)
    ctypes.c_int,     # incy
]

_dcopy = _lib.cublasDcopy_v2
_dcopy.restype = ctypes.c_int
_dcopy.argtypes = [
    ctypes.c_void_p,  # handle
    ctypes.c_int,     # n
    ctypes.c_void_p,  # x
    ctypes.c_int,     # incx
    ctypes.c_void_p,  # y
    ctypes.c_int,     # incy
]

_daxpy = _lib.cublasDaxpy_v2
_daxpy.restype = ctypes.c_int
_daxpy.argtypes = [
    ctypes.c_void_p,  # handle
    ctypes.c_int,     # n
    ctypes.c_void_p,  # alpha  (pointer to double)
    ctypes.c_void_p,  # x
    ctypes.c_int,     # incx
    ctypes.c_void_p,  # y
    ctypes.c_int,     # incy
]

_dsyrk = _lib.cublasDsyrk_v2
_dsyrk.restype = ctypes.c_int
_dsyrk.argtypes = [
    ctypes.c_void_p,  # handle
    ctypes.c_int,     # uplo
    ctypes.c_int,     # trans
    ctypes.c_int,     # n
    ctypes.c_int,     # k
    ctypes.c_void_p,  # alpha  (pointer to double)
    ctypes.c_void_p,  # A
    ctypes.c_int,     # lda
    ctypes.c_void_p,  # beta   (pointer to double)
    ctypes.c_void_p,  # C
    ctypes.c_int,     # ldc
]

_ddgmm = _lib.cublasDdgmm
_ddgmm.restype = ctypes.c_int
_ddgmm.argtypes = [
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

_ddot = _lib.cublasDdot_v2
_ddot.restype = ctypes.c_int
_ddot.argtypes = [
    ctypes.c_void_p,  # handle
    ctypes.c_int,     # n
    ctypes.c_void_p,  # x
    ctypes.c_int,     # incx
    ctypes.c_void_p,  # y
    ctypes.c_int,     # incy
    ctypes.c_void_p,  # result (pointer to double)
]

_dgemm_strided_batched = _lib.cublasDgemmStridedBatched
_dgemm_strided_batched.restype = ctypes.c_int
_dgemm_strided_batched.argtypes = [
    ctypes.c_void_p,     # handle
    ctypes.c_int,        # transa
    ctypes.c_int,        # transb
    ctypes.c_int,        # m
    ctypes.c_int,        # n
    ctypes.c_int,        # k
    ctypes.c_void_p,     # alpha  (host pointer to double)
    ctypes.c_void_p,     # A      (device pointer)
    ctypes.c_int,        # lda
    ctypes.c_longlong,   # strideA
    ctypes.c_void_p,     # B      (device pointer)
    ctypes.c_int,        # ldb
    ctypes.c_longlong,   # strideB
    ctypes.c_void_p,     # beta   (host pointer to double)
    ctypes.c_void_p,     # C      (device pointer)
    ctypes.c_int,        # ldc
    ctypes.c_longlong,   # strideC
    ctypes.c_int,        # batchCount
]

_set_pointer_mode = _lib.cublasSetPointerMode_v2
_set_pointer_mode.restype = ctypes.c_int
_set_pointer_mode.argtypes = [
    ctypes.c_void_p,  # handle
    ctypes.c_int,     # mode
]

_set_stream = _lib.cublasSetStream_v2
_set_stream.restype = ctypes.c_int
_set_stream.argtypes = [
    ctypes.c_void_p,  # handle
    ctypes.c_void_p,  # stream (cudaStream_t)
]

_create = _lib.cublasCreate_v2
_create.restype = ctypes.c_int
_create.argtypes = [ctypes.POINTER(ctypes.c_void_p)]

_destroy = _lib.cublasDestroy_v2
_destroy.restype = ctypes.c_int
_destroy.argtypes = [ctypes.c_void_p]


# ---------------------------------------------------------------------------
# Wrapper functions
# ---------------------------------------------------------------------------
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
    """Associate a CUDA stream with the cuBLAS handle.

    All subsequent cuBLAS calls on *handle* will be submitted to *stream_ptr*.
    This is a host-side state change (no graph node is created), so it is safe
    to call before/after ``stream.begin_capture()``.
    """
    status = _set_stream(handle, cuda_stream)
    if status != 0:
        raise RuntimeError(f"cublasSetStream failed with status {status}")
    
def dgemv(handle, mat, x, y, transa=False, alpha=1.0, beta=0.0):
    """``y = alpha * op(mat) * x + beta * y``  (CUDA graph safe).

    Parameters
    ----------
    handle : int
        cuBLAS handle (from ``cp.cuda.Device().cublas_handle``).
    mat : cp.ndarray
        2-D device array, dtype float64, C- or F-contiguous.
    x, y : cp.ndarray
        Input / output vectors (1-D, float64).
    transa : bool
        If False, compute ``mat @ x``.  If True, compute ``mat.T @ x``.
    alpha, beta : float
        Host scalars baked into the graph node at capture time.
    """
    rows, cols = mat.shape
    if mat.flags["F_CONTIGUOUS"]:
        m, n, lda = rows, cols, rows
        op = OP_N if not transa else OP_T
    else:
        m, n, lda = cols, rows, cols
        op = OP_T if not transa else OP_N

    _alpha = ctypes.c_double(alpha)
    _beta = ctypes.c_double(beta)

    _dgemv(
        handle, op, m, n,
        ctypes.addressof(_alpha), mat.data.ptr, lda,
        x.data.ptr, 1,
        ctypes.addressof(_beta), y.data.ptr, 1,
    )


def dcopy(handle, n, x_ptr, incx, y_ptr, incy):
    """``y = x``  (element-wise copy)."""
    _dcopy(handle, n, x_ptr, incx, y_ptr, incy)


def daxpy(handle, n, alpha, x_ptr, incx, y_ptr, incy):
    """``y = alpha * x + y``.

    Parameters
    ----------
    alpha : float or int
        float → host scalar (read immediately, baked into graph at capture).
        int   → device pointer (switches to POINTER_DEVICE, then restores).
    """
    if isinstance(alpha, float):
        _alpha = ctypes.c_double(alpha)
        _daxpy(handle, n, ctypes.addressof(_alpha), x_ptr, incx, y_ptr, incy)
    else:
        _set_pointer_mode(handle, POINTER_DEVICE)
        _daxpy(handle, n, alpha, x_ptr, incx, y_ptr, incy)
        _set_pointer_mode(handle, POINTER_HOST)


def dsyrk(handle, uplo, trans, n, k, alpha, a_ptr, lda, beta, c_ptr, ldc):
    """``C = alpha * op(A) * op(A)^T + beta * C``  (host scalars)."""
    _alpha = ctypes.c_double(alpha)
    _beta = ctypes.c_double(beta)
    _dsyrk(
        handle, uplo, trans, n, k,
        ctypes.addressof(_alpha), a_ptr, lda,
        ctypes.addressof(_beta), c_ptr, ldc,
    )


def ddgmm(handle, mode, m, n, a_ptr, lda, x_ptr, incx, c_ptr, ldc):
    """``C = diag(x) * A``  or  ``A * diag(x)``."""
    _ddgmm(handle, mode, m, n, a_ptr, lda, x_ptr, incx, c_ptr, ldc)


def ddot(handle, n, x_ptr, incx, y_ptr, incy, result_ptr):
    """``result = x^T * y``  (result is a device pointer)."""
    _ddot(handle, n, x_ptr, incx, y_ptr, incy, result_ptr)


def dgemm_strided_batched(handle, A, B, C,
                          transa=False, transb=False,
                          alpha=1.0, beta=0.0):
    r"""Batched GEMM on C-contiguous 3-D arrays.

    .. math::

        C_i = \alpha\;\mathrm{op}(A_i)\;B_i + \beta\;C_i
        \qquad i = 0 \ldots \text{batch}-1

    where :math:`\mathrm{op}(X) = X` when *trans\*=False* and
    :math:`X^\top` when *trans\*=True*.

    Parameters
    ----------
    handle : int
        cuBLAS handle.
    A : cp.ndarray
        Shape ``(batch, rA, cA)``, C-contiguous, float64.
    B : cp.ndarray
        Shape ``(batch, rB, cB)``, C-contiguous, float64.
    C : cp.ndarray
        Shape ``(batch, rC, cC)``, C-contiguous, float64.  Modified **in-place**.
    transa, transb : bool
        Whether to transpose A / B before multiplication.
    alpha, beta : float
        Host scalars (baked into a CUDA graph at capture time).
    """
    batch = A.shape[0]
    rA, cA = A.shape[1], A.shape[2]
    rB, cB = B.shape[1], B.shape[2]

    # ------------------------------------------------------------------
    # Row-major (Python) → column-major (cuBLAS) mapping
    #
    # A C-contiguous (r, c) matrix is stored identically to a
    # column-major (c, r) matrix, i.e. the cuBLAS view is the
    # transpose.
    #
    # We want:  C = alpha * opA(A) @ opB(B) + beta * C
    # Taking transposes:
    #   C^T = alpha * opB(B)^T @ opA(A)^T + beta * C^T
    #       = alpha * op_b_cm(B_cm) @ op_a_cm(A_cm) + beta * C_cm
    #
    # where op_x_cm maps the Python flag to the cuBLAS op applied to
    # the transposed (column-major) matrix:
    #   transa=False → opA(A)^T = A^T = A_cm  → OP_N on A_cm
    #   transa=True  → opA(A)^T = A   = A_cm^T → OP_T on A_cm
    # ------------------------------------------------------------------
    op_a_cm = OP_N if not transa else OP_T
    op_b_cm = OP_N if not transb else OP_T

    # cuBLAS dgemm:  C_blas = alpha * op(A_blas) @ op(B_blas) + beta * C_blas
    # We identify:  A_blas = B_cm,  B_blas = A_cm  (swapped)
    #               op on A_blas = op_b_cm,  op on B_blas = op_a_cm

    # Dimensions of op(A_blas) = op_b_cm(B_cm):
    #   B_cm is (cB, rB);  OP_N → (cB, rB);  OP_T → (rB, cB)
    if not transb:
        m_blas, k_blas = cB, rB
    else:
        m_blas, k_blas = rB, cB

    # Dimensions of op(B_blas) = op_a_cm(A_cm):
    #   A_cm is (cA, rA);  OP_N → (cA, rA);  OP_T → (rA, cA)
    if not transa:
        n_blas = rA
    else:
        n_blas = cA

    # Leading dimensions (column-major storage rows)
    lda_blas = cB   # B_cm has cB leading rows
    ldb_blas = cA   # A_cm has cA leading rows
    ldc_blas = C.shape[2]  # C_cm has cC leading rows

    # NOTE: use actual array strides to support non-contiguous views
    # strides[0] means how many bytes to move to the next axis of this array (here [0] means batch). For double, itemsize is always 8 bytes 
    strideA_blas = B.strides[0] // B.itemsize
    strideB_blas = A.strides[0] // A.itemsize
    strideC_blas = C.strides[0] // C.itemsize

    _alpha = ctypes.c_double(alpha)
    _beta = ctypes.c_double(beta)

    _dgemm_strided_batched(
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
    r"""Batched matrix-vector product via strided batched GEMM.

    .. math::

        y_i = \alpha\;\mathrm{op}(M_i)\;x_i + \beta\;y_i

    Parameters
    ----------
    handle : int
        cuBLAS handle.
    mat : cp.ndarray
        Shape ``(batch, rows, cols)``, C-contiguous, float64.
    x : cp.ndarray
        Shape ``(batch, k)``, C-contiguous, float64.
    y : cp.ndarray
        Shape ``(batch, m)``, C-contiguous, float64.  Modified **in-place**.
    transa : bool
        If *False*, compute ``mat @ x``.  If *True*, compute ``mat.T @ x``.
    alpha, beta : float
        Host scalars.
    """
    batch = mat.shape[0]
    x3 = x.reshape(batch, x.shape[1], 1)
    y3 = y.reshape(batch, y.shape[1], 1)
    dgemm_strided_batched(handle, mat, x3, y3,
                          transa=transa, alpha=alpha, beta=beta)


def set_pointer_mode(handle, mode):
    """Set cuBLAS pointer mode (``POINTER_HOST`` or ``POINTER_DEVICE``)."""
    _set_pointer_mode(handle, mode)

