"""Graph-safe cuSPARSE SpMV wrapper with native batching.

CuPy's ``cupyx.cusparse.spmv`` allocates a workspace buffer on every call,
which breaks CUDA graph capture.  This module provides a thin wrapper that
pre-allocates all cuSPARSE descriptors and the workspace buffer once at
construction time so that ``__call__`` only invokes ``cusparseSpMV`` -- no
allocation, no host-device sync -- making it safe for stream capture and
also more efficient.

Batching (B > 1):

When a *list* of B CSR matrices (sharing the same sparsity pattern) is
passed, a single block-diagonal CSR matrix is built internally::

    [ A_0              ]   [ x_0 ]   [ y_0 ]
    [      A_1         ] * [ x_1 ] = [ y_1 ]
    [           ...    ]   [ ... ]   [ ... ]
    [              A_B ]   [ x_B ]   [ y_B ]

A single ``cusparseSpMV`` call on the stacked vectors processes all B
problems simultaneously.  The individual matrices' ``.data`` attributes
are reassigned to contiguous views of a packed buffer so that later
in-place modifications (e.g. preconditioner scaling) are automatically
visible to the cuSPARSE descriptor.

For B = 1 (a single CSR matrix) the original, SpMV is directly called
"""

import ctypes
import ctypes.util
from typing import Sequence
import cupy as cp
import cupyx.scipy.sparse as cpsp
from cupy.cuda import cusparse
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

# cusparseStatus_t cusparseSetStream(cusparseHandle_t handle, cudaStream_t streamId)
_cusparse_lib.cusparseSetStream.restype = ctypes.c_int
_cusparse_lib.cusparseSetStream.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

# cusparseStatus_t cusparseCreate(cusparseHandle_t *handle)
_cusparse_lib.cusparseCreate.restype = ctypes.c_int
_cusparse_lib.cusparseCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]

# cusparseStatus_t cusparseDestroy(cusparseHandle_t handle)
_cusparse_lib.cusparseDestroy.restype = ctypes.c_int
_cusparse_lib.cusparseDestroy.argtypes = [ctypes.c_void_p]


def _create_cusparse_handle():
    handle = ctypes.c_void_p()
    status = _cusparse_lib.cusparseCreate(ctypes.byref(handle))
    if status != 0:
        raise RuntimeError(f"cusparseCreate failed with status {status}")
    return handle.value


def _destroy_cusparse_handle(handle):
    status = _cusparse_lib.cusparseDestroy(handle)
    if status != 0:
        raise RuntimeError(f"cusparseDestroy failed with status {status}")


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
    """CUDA-graph-safe sparse matrix-vector product.

    Computes ``y = alpha * op(A) * x + beta * y``.

    Parameters
    ----------
    mat : csr_matrix | csc_matrix | coo_matrix | list[csr_matrix]
        A single sparse matrix (B = 1), or a list of B CSR matrices that
        share the same sparsity pattern (B > 1).
    transa : bool
        If ``True``, compute ``A^T * x`` instead of ``A * x``.

    Notes
    -----
    For B > 1 the individual matrices' ``.data`` attributes are reassigned
    to contiguous views of an internal packed buffer.  Any subsequent
    in-place modification of those arrays (e.g. ``mat.data *= scale``)
    is automatically visible to the cuSPARSE descriptor.
    """

    def __init__(
        self,
        mat: Sequence[cpsp.csr_matrix],
        transa: bool = False,
    ):
        assert isinstance(mat, Sequence) and len(mat) > 0
        assert isinstance(mat[0], cpsp.csr_matrix)
        self._batch_size = len(mat)
        if self._batch_size == 1:
            self._init_single(mat[0], transa)
        else:
            self._init_batched(list(mat), transa)

    def _init_single(self, mat, transa):
        self._cusparse_handle = _create_cusparse_handle()

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
            self._cusparse_handle,
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

    def _init_batched(self, mats, transa):
        B = self._batch_size
        template = mats[0]
        rows, cols = template.shape
        nnz = template.nnz

        # -- pack matrix data contiguously: [data_0 | data_1 | ... ] ----
        # Check whether data is already packed (e.g. by SparseData).
        if self._is_data_packed(mats, nnz):
            # Reuse existing contiguous buffer
            self._packed_data = cp.ndarray(
                B * nnz, dtype=cp.float64,
                memptr=cp.cuda.MemoryPointer(mats[0].data.data.mem, mats[0].data.data.ptr - mats[0].data.data.mem.ptr),
            )
        else:
            self._packed_data = cp.empty(B * nnz, dtype=cp.float64)
            for b in range(B):
                view = self._packed_data[b * nnz : (b + 1) * nnz]
                view[:] = mats[b].data
                mats[b].data = view

        # -- build block-diagonal structure via bmat --------------------
        # bmat gives us correct indptr/indices; we replace .data with our
        # packed buffer so in-place modifications stay visible.
        from cupyx.scipy.sparse import bmat as sp_bmat
        blocks = [[None] * B for _ in range(B)]
        for b in range(B):
            blocks[b][b] = mats[b]
        self._block_diag = sp_bmat(blocks, format='csr', dtype=cp.float64)
        # Wire the packed data into the block-diagonal CSR
        self._block_diag.data = self._packed_data

        # -- cuSPARSE setup ---------------------------------------------
        self._cusparse_handle = _create_cusparse_handle()
        self._compute_type = rt.CUDA_R_64F
        self._op = (
            cusparse.CUSPARSE_OPERATION_TRANSPOSE
            if transa
            else cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE
        )
        self._alg = cusparse.CUSPARSE_MV_ALG_DEFAULT

        self._mat_desc = _create_mat_desc(self._block_diag, self._compute_type)

        # Dense vector descriptors sized for the stacked vectors
        if transa:
            x_size, y_size = B * rows, B * cols
        else:
            x_size, y_size = B * cols, B * rows

        dummy_x = cp.empty(x_size, dtype=cp.float64)
        dummy_y = cp.empty(y_size, dtype=cp.float64)
        self._x_desc = cusparse.createDnVec(x_size, dummy_x.data.ptr, self._compute_type)
        self._y_desc = cusparse.createDnVec(y_size, dummy_y.data.ptr, self._compute_type)

        # Workspace
        _alpha_placeholder = ctypes.c_double(1.0)
        _beta_placeholder = ctypes.c_double(0.0)
        buf_size = cusparse.spMV_bufferSize(
            self._cusparse_handle,
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

        # Keep refs alive
        self._mat_ref = mats

    @staticmethod
    def _is_data_packed(mats, nnz):
        """Return True if the matrices' data arrays already form a contiguous
        packed buffer with stride ``nnz * 8`` bytes between consecutive blocks."""
        if len(mats) < 2:
            return True
        stride = nnz * 8  # bytes
        ptr0 = mats[0].data.data.ptr
        for b in range(1, len(mats)):
            if mats[b].data.data.ptr != ptr0 + b * stride:
                return False
        return True

    def __call__(
        self,
        x: cp.ndarray,
        y: cp.ndarray,
        alpha: float = 1.0,
        beta: float = 0.0,
        stream_ptr: int = None,
    ) -> None:
        """Execute ``y = alpha * op(A) * x + beta * y``.

        *x* and *y* are ``(B, k)`` arrays stacked contiguously in memory.
        """
        if stream_ptr is None:
            stream_ptr = cp.cuda.get_current_stream().ptr
        _cusparse_lib.cusparseSetStream(self._cusparse_handle, stream_ptr)

        _alpha = ctypes.c_double(alpha)
        _beta = ctypes.c_double(beta)

        _cusparse_lib.cusparseDnVecSetValues(self._x_desc, x.data.ptr)
        _cusparse_lib.cusparseDnVecSetValues(self._y_desc, y.data.ptr)

        status = _cusparse_lib.cusparseSpMV(
            self._cusparse_handle,
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

        handle = getattr(self, "_cusparse_handle", None)
        if handle is not None:
            try:
                _destroy_cusparse_handle(handle)
            except Exception:
                pass
