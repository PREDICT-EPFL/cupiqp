"""Graph-safe cuSPARSE SpMV wrapper.

CuPy's ``cupyx.cusparse.spmv`` allocates a workspace buffer on every call,
which breaks CUDA graph capture.  This module provides a thin wrapper that
pre-allocates all cuSPARSE descriptors and the workspace buffer once at
construction time so that ``__call__`` only invokes ``cusparseSpMV`` -- no
allocation, no host-device sync -- making it safe for stream capture and 
also more efficient.

The ``__call__`` method uses **ctypes** to invoke the cuSPARSE C API
directly, bypassing CuPy's Python-level ``_setStream`` guard which raises
``NotImplementedError`` when called during CUDA stream capture.
"""

import ctypes
import ctypes.util
import cupy as cp
import cupyx.scipy.sparse as cpsp
from cupy.cuda import cusparse, device
import cupy.cuda.runtime as rt

# ---------------------------------------------------------------------------
# Load cuSPARSE shared library once (module level) for graph-safe direct calls
# ---------------------------------------------------------------------------
def _load_cusparse_lib():
    """Load the cuSPARSE shared library via ctypes."""
    for name in ("libcusparse.so.12", "libcusparse.so", "cusparse"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    lib_path = ctypes.util.find_library("cusparse")
    if lib_path:
        return ctypes.CDLL(lib_path)
    raise RuntimeError("Could not find cuSPARSE shared library")


_cusparse_lib = _load_cusparse_lib()

# cusparseStatus_t cusparseSpMV(handle, opA, alpha, matA, vecX,
#                                beta, vecY, computeType, alg, externalBuffer)
_cusparse_lib.cusparseSpMV.restype = ctypes.c_int
_cusparse_lib.cusparseSpMV.argtypes = [
    ctypes.c_void_p,  # handle
    ctypes.c_int,     # opA
    ctypes.c_void_p,  # alpha
    ctypes.c_void_p,  # matA
    ctypes.c_void_p,  # vecX
    ctypes.c_void_p,  # beta
    ctypes.c_void_p,  # vecY
    ctypes.c_int,     # computeType
    ctypes.c_int,     # alg
    ctypes.c_void_p,  # externalBuffer
]

# cusparseStatus_t cusparseDnVecSetValues(dnVecDescr, values)
_cusparse_lib.cusparseDnVecSetValues.restype = ctypes.c_int
_cusparse_lib.cusparseDnVecSetValues.argtypes = [
    ctypes.c_void_p,  # dnVecDescr
    ctypes.c_void_p,  # values
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _idx_type(arr: cp.ndarray) -> int:
    """Return the cuSPARSE index-type constant matching *arr*'s dtype."""
    return (
        cusparse.CUSPARSE_INDEX_32I
        if arr.dtype == cp.int32
        else cusparse.CUSPARSE_INDEX_64I
    )


def _create_mat_desc(mat, compute_type: int) -> int:
    """Create a cuSPARSE sparse-matrix descriptor for *mat*.

    Supports CSR, CSC, and COO formats.  The descriptor reuses the matrix's
    existing device memory; no copy is made.
    """
    rows, cols = mat.shape
    if isinstance(mat, cpsp.csr_matrix):
        return cusparse.createCsr(
            rows, cols, mat.nnz,
            mat.indptr.data.ptr,
            mat.indices.data.ptr,
            mat.data.data.ptr,
            _idx_type(mat.indptr),
            _idx_type(mat.indices),
            cusparse.CUSPARSE_INDEX_BASE_ZERO,
            compute_type,
        )
    elif isinstance(mat, cpsp.csc_matrix):
        return cusparse.createCsc(
            rows, cols, mat.nnz,
            mat.indptr.data.ptr,
            mat.indices.data.ptr,
            mat.data.data.ptr,
            _idx_type(mat.indptr),
            _idx_type(mat.indices),
            cusparse.CUSPARSE_INDEX_BASE_ZERO,
            compute_type,
        )
    elif isinstance(mat, cpsp.coo_matrix):
        return cusparse.createCoo(
            rows, cols, mat.nnz,
            mat.row.data.ptr,
            mat.col.data.ptr,
            mat.data.data.ptr,
            _idx_type(mat.row),
            cusparse.CUSPARSE_INDEX_BASE_ZERO,
            compute_type,
        )
    else:
        raise TypeError(
            f"Unsupported sparse matrix type: {type(mat).__name__}. "
            "Expected csr_matrix, csc_matrix, or coo_matrix."
        )



class SparseMatVecProduct:
    """CUDA-graph-safe sparse matrix-vector product: ``y = alpha * op(A) * x + beta * y``.

    Parameters
    ----------
    mat : cupyx.scipy.sparse.csr_matrix | csc_matrix | coo_matrix
        The sparse matrix.  Must stay alive and its data buffers must not be
        reallocated for the lifetime of this object.
    transa : bool
        If ``True``, compute ``A^T * x`` instead of ``A * x``.

    Notes
    -----
    ``alpha`` and ``beta`` are supplied at call time.  If this op is captured
    inside a CUDA graph, both scalars are baked into the captured node and
    must remain the same on every replay; re-capture if they need to change.

    cuSPARSE does not support ``CUSPARSE_OPERATION_TRANSPOSE`` for COO matrices.
    Convert to CSR or CSC first if you need ``transa=True`` with a COO matrix.
    """

    def __init__(
        self,
        mat: cpsp.spmatrix,
        transa: bool = False,
    ):
        self._handle = device.get_cusparse_handle()

        # ---- operation ----
        self._op = (
            cusparse.CUSPARSE_OPERATION_TRANSPOSE
            if transa
            else cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE
        )

        # ---- data type ----
        self._compute_type = rt.CUDA_R_64F

        # ---- sparse matrix descriptor (reuses mat's existing device memory) ----
        self._mat_desc = _create_mat_desc(mat, self._compute_type)

        # ---- dense vector descriptors ----
        if transa:
            x_size, y_size = mat.shape[0], mat.shape[1]
        else:
            x_size, y_size = mat.shape[1], mat.shape[0]

        dummy_x = cp.empty(x_size, dtype=cp.float64)
        dummy_y = cp.empty(y_size, dtype=cp.float64)
        self._x_desc = cusparse.createDnVec(x_size, dummy_x.data.ptr, self._compute_type)
        self._y_desc = cusparse.createDnVec(y_size, dummy_y.data.ptr, self._compute_type)

        # ---- workspace buffer (allocated once; size is independent of scalars) ----
        self._alg = cusparse.CUSPARSE_MV_ALG_DEFAULT
        _alpha_placeholder = ctypes.c_double(1.0)
        _beta_placeholder = ctypes.c_double(0.0)
        buf_size = cusparse.spMV_bufferSize(
            self._handle,
            self._op,
            ctypes.addressof(_alpha_placeholder),
            self._mat_desc,
            self._x_desc,
            ctypes.addressof(_beta_placeholder),
            self._y_desc,
            self._compute_type,
            self._alg,
        )
        self._buffer = cp.empty(max(buf_size, 1), dtype=cp.uint8)

        # Keep mat alive so its device buffers aren't freed.
        self._mat_ref = mat

    # ------------------------------------------------------------------
    def __call__(
        self,
        x: cp.ndarray,
        y: cp.ndarray,
        alpha: float = 1.0,
        beta: float = 0.0,
    ) -> None:
        """Execute ``y = alpha * op(A) * x + beta * y``.

        Safe to call inside ``stream.begin_capture() / end_capture()``.

        ``cusparseDnVecSetValues`` is a pure host-side descriptor update
        (no kernel launch), so updating the pointers before the SpMV kernel
        is recorded is safe during stream capture.

        Parameters
        ----------
        x : cp.ndarray
            Input vector.
        y : cp.ndarray
            Output vector (updated in-place).
        alpha : float
            Scalar multiplier for ``op(A) * x``.  Defaults to ``1.0``.
        beta : float
            Scalar multiplier for the initial value of ``y``.  Defaults to ``0.0``.
        """
        _alpha = ctypes.c_double(alpha)
        _beta = ctypes.c_double(beta)

        _cusparse_lib.cusparseDnVecSetValues(self._x_desc, x.data.ptr)
        _cusparse_lib.cusparseDnVecSetValues(self._y_desc, y.data.ptr)

        status = _cusparse_lib.cusparseSpMV(
            self._handle,
            self._op,
            ctypes.addressof(_alpha),
            self._mat_desc,
            self._x_desc,
            ctypes.addressof(_beta),
            self._y_desc,
            self._compute_type,
            self._alg,
            self._buffer.data.ptr,
        )
        if status != 0:
            raise RuntimeError(f"cusparseSpMV failed with status {status}")

    # ------------------------------------------------------------------
    def __del__(self):
        try:
            cusparse.destroyDnVec(self._x_desc)
        except Exception:
            pass
        try:
            cusparse.destroyDnVec(self._y_desc)
        except Exception:
            pass
        try:
            cusparse.destroySpMat(self._mat_desc)
        except Exception:
            pass


