import ctypes
import ctypes.util
from typing import Union
import cupy as cp
from cupyx.scipy.sparse import spmatrix, csr_matrix, csc_matrix, coo_matrix
from cupy.cuda import cusparse
import cupy.cuda.runtime as rt

from .batched_csr import BatchedCsrMatrix

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




class SingleSparseMatVecProduct:
    """Graph-capture-safe sparse matrix-vector product for a single matrix.

    Computes ``y = alpha * op(A) * x + beta * y`` where ``A`` is a cupy
    sparse matrix in CSR, CSC, or COO format and ``op`` is either identity
    or transpose (``transa``).

    Why a custom wrapper: ``cupyx.cusparse.spmv`` allocates its workspace
    buffer on every call, which (a) prevents CUDA graph capture and (b)
    adds per-call overhead. This class pre-allocates the cuSPARSE handle,
    the sparse- and dense-vector descriptors, and the workspace buffer at
    construction time. ``__call__`` then only rebinds the dense-vector
    pointers and dispatches ``cusparseSpMV`` via a direct ctypes call —
    no device allocation, no host-device sync, safe for stream capture.

    Lifetime: the sparse matrix ``mat`` is kept alive through ``self._mat``
    and the descriptor reuses its device buffers. In-place updates to
    ``mat.data`` are therefore visible on subsequent calls as long as the
    underlying allocation is not replaced (e.g. ``mat.data[:] = ...`` is
    safe; ``mat.data = new_array`` is not).

    Parameters
    ----------
    mat : csr_matrix | csc_matrix | coo_matrix
        The sparse matrix. Device buffers are reused (no copy).
    transa : bool, default False
        If ``True``, compute ``A^T @ x`` instead of ``A @ x``.

    Call signature
    --------------
    ``op(x, y, alpha=1.0, beta=0.0, stream_ptr=None)``
        ``x`` is a 1-D float64 cupy array of length ``cols`` (or ``rows``
        if ``transa``); ``y`` is a 1-D float64 cupy array of matching
        output length. Both are used directly (no staging).
    """

    def __init__(
        self,
        mat: Union[csr_matrix, csc_matrix, coo_matrix],
        transa: bool = False,
    ):
        if not isinstance(mat, (csr_matrix, csc_matrix, coo_matrix)):
            raise TypeError(
                f"mat must be a csr_matrix, csc_matrix, or coo_matrix; "
                f"got {type(mat).__name__}."
            )
        self._mat = mat
        self._transa = transa
        self._setup_cusparse()
        
    def _setup_cusparse(self):

        self._cusparse_handle = _create_cusparse_handle()

        # ---- operation ----
        self._op = (
            cusparse.CUSPARSE_OPERATION_TRANSPOSE
            if self._transa
            else cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE
        )

        # ---- data type ----
        self._compute_type = rt.CUDA_R_64F

        # ---- sparse matrix descriptor (reuses mat's existing device memory) ----
        self._mat_desc = self._create_mat_desc(self._mat, self._compute_type)

        # ---- dense vector descriptors ----
        if self._transa:
            x_size, y_size = self._mat.shape[0], self._mat.shape[1]
        else:
            x_size, y_size = self._mat.shape[1], self._mat.shape[0]

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


    @staticmethod
    def _create_mat_desc(mat: spmatrix, compute_type: int) -> int:
        """Create a cuSPARSE sparse-matrix descriptor for *mat*.

        Supports CSR, CSC, and COO formats.  The descriptor reuses the matrix's
        existing device memory; no copy is made.
        """
        rows, cols = mat.shape
        if isinstance(mat, csr_matrix):
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
        elif isinstance(mat, csc_matrix):
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
        elif isinstance(mat, coo_matrix):
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

    def __call__(
        self,
        x: cp.ndarray,  # shape (n,)
        y: cp.ndarray,  # shape (m,)
        alpha: float = 1.0,
        beta: float = 0.0,
        stream_ptr: int = None,
    ) -> None:
        """Execute ``y = alpha * op(A) * x + beta * y``. """
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



class BatchedSparseMatVecProduct:
    """Graph-capture-safe batched sparse matrix-vector product.

    Computes ``y[b] = alpha * op(A[b]) @ x[b] + beta * y[b]`` for every
    ``b`` in ``range(B)`` with a *single* ``cusparseSpMV`` call on an
    internally constructed block-diagonal CSR::

        [ A_0                  ]   [ x_0   ]   [ y_0   ]
        [     A_1              ] @ [ x_1   ] = [ y_1   ]
        [           ...        ]   [  ...  ]   [  ...  ]
        [               A_B-1  ]   [ x_B-1 ]   [ y_B-1 ]

    The stacked operator is ``(B*m, B*n)`` with ``B*nnz`` nonzeros. Because
    all batches share the same sparsity pattern, the block-diagonal's
    ``indptr`` and ``indices`` are built via vectorized cupy broadcasting —
    no Python loop over batches — and the block-diagonal's values buffer is
    a zero-copy ``reshape`` of ``mats.data``. Any in-place update to
    ``mats`` (``mats[b] = ...``, ``mats.data *= scale``, etc.) is therefore
    visible on the next call without rebuilding anything.

    Graph safety mirrors ``SingleSparseMatVecProduct``: every cuSPARSE
    descriptor and the workspace buffer are allocated once in ``__init__``;
    ``__call__`` only rebinds dense-vector pointers and dispatches
    ``cusparseSpMV`` via a direct ctypes call.

    Parameters
    ----------
    mats : BatchedCsrMatrix
        The batched sparse matrices. All batches share the sparsity
        pattern; ``mats.data`` is a packed ``(B, nnz)`` float64 buffer.
    transa : bool, default False
        If ``True``, compute ``A[b]^T @ x[b]`` instead of ``A[b] @ x[b]``.

    Call signature
    --------------
    ``op(x, y, alpha=1.0, beta=0.0, stream_ptr=None)``
        ``x`` / ``y`` are float64 arrays of shape ``(B, k_x)`` / ``(B, k_y)``.
        C-contiguous arrays are passed to cuSPARSE zero-copy. Non-contiguous
        arrays (e.g. column slices of a wider ``(B, K)`` buffer) are staged
        through pre-allocated internal buffers via ``cudaMemcpy2DAsync``;
        when ``beta != 0`` the current ``y`` values are also staged in
        before the call so cuSPARSE reads them correctly.
    """

    def __init__(
        self,
        mats: BatchedCsrMatrix,
        transa: bool = False,
    ):
        self._batch_size = mats.batch_size
        self._mats = mats
        self._transa = transa

        self._setup_big_diag_matrix()
        self._setup_cusparse()

    def _setup_big_diag_matrix(self):
        B, rows, cols, nnz = self._mats.batch_size, self._mats.rows, self._mats.cols, self._mats.nnz
        indptr = self._mats.indptr       # (rows+1,)
        indices = self._mats.indices     # (nnz,)

        # ---- block-diagonal indptr ((B*rows + 1,)): big[b*rows+r] = b*nnz + indptr[r]
        batch_nnz = cp.arange(B, dtype=indptr.dtype) * nnz
        big_mat_indptr = cp.empty(B * rows + 1, dtype=indptr.dtype)
        big_mat_indptr[:B * rows].reshape(B, rows)[:] = (
            indptr[:-1][None, :] + batch_nnz[:, None]
        )
        big_mat_indptr[-1] = B * nnz

        # ---- block-diagonal indices ((B*nnz,)): big[b*nnz+k] = indices[k] + b*cols
        batch_col = cp.arange(B, dtype=indices.dtype) * cols
        big_mat_indices = (indices[None, :] + batch_col[:, None]).reshape(-1)

        # ---- block-diagonal values: a zero-copy view into mats.data
        big_mat_data = self._mats.data.reshape(-1)

        self._block_diag_mat = csr_matrix(
            (big_mat_data, big_mat_indices, big_mat_indptr),
            shape=(B * rows, B * cols),
        )

    def _setup_cusparse(self):
        B, rows, cols, nnz = self._mats.batch_size, self._mats.rows, self._mats.cols, self._mats.nnz

        # ---- cuSPARSE setup ------------------------------------------
        self._cusparse_handle = _create_cusparse_handle()
        self._compute_type = rt.CUDA_R_64F
        self._op = (
            cusparse.CUSPARSE_OPERATION_TRANSPOSE
            if self._transa
            else cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE
        )
        self._alg = cusparse.CUSPARSE_MV_ALG_DEFAULT
        self._mat_desc = SingleSparseMatVecProduct._create_mat_desc(
            self._block_diag_mat, self._compute_type,
        )

        # Per-problem vector lengths (before stacking).
        if self._transa:
            self._x_vec_len, self._y_vec_len = rows, cols
        else:
            self._x_vec_len, self._y_vec_len = cols, rows

        # Persistent contiguous staging buffers. Used when the caller passes
        # non-contiguous (B, k) views (e.g. column slices of a wider buffer).
        x_size = B * self._x_vec_len
        y_size = B * self._y_vec_len
        self._x_buf = cp.empty(x_size, dtype=cp.float64)
        self._y_buf = cp.empty(y_size, dtype=cp.float64)
        self._x_desc = cusparse.createDnVec(x_size, self._x_buf.data.ptr, self._compute_type)
        self._y_desc = cusparse.createDnVec(y_size, self._y_buf.data.ptr, self._compute_type)

        # Workspace size is independent of the alpha/beta scalars.
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
            cusparse.CUSPARSE_MV_ALG_DEFAULT,
        )
        self._buffer = cp.empty(max(buf_size, 1), dtype=cp.uint8)


    def __call__(
        self,
        x: cp.ndarray,
        y: cp.ndarray,
        alpha: float = 1.0,
        beta: float = 0.0,
        stream_ptr: int = None,
    ) -> None:
        """Execute ``y = alpha * op(A) * x + beta * y``.

        ``x`` and ``y`` are ``(B, k)`` float64 arrays. C-contiguous arrays
        are bound to cuSPARSE zero-copy. Non-contiguous arrays (e.g. column
        slices of a wider buffer) are staged through pre-allocated internal
        buffers via ``cudaMemcpy2DAsync``; when ``beta != 0`` the current y
        values are also staged in before the call so cuSPARSE reads them.
        """
        if stream_ptr is None:
            stream_ptr = cp.cuda.get_current_stream().ptr
        _cusparse_lib.cusparseSetStream(self._cusparse_handle, stream_ptr)

        _alpha = ctypes.c_double(alpha)
        _beta = ctypes.c_double(beta)

        B = self._batch_size
        xk, yk = self._x_vec_len, self._y_vec_len
        itemsize = 8  # float64
        x_contig = x.flags['C_CONTIGUOUS']
        y_contig = y.flags['C_CONTIGUOUS']

        if x_contig:
            _cusparse_lib.cusparseDnVecSetValues(self._x_desc, x.data.ptr)
        else:
            # Copy x (non-contig) into the contiguous internal x_buf so the
            # descriptor's expected layout matches memory. One D->D 2D copy.
            rt.memcpy2DAsync(
                self._x_buf.data.ptr,  # dst: start of internal packed buffer
                xk * itemsize,         # dpitch: bytes between dst rows (tight)
                x.data.ptr,            # src: caller's (possibly strided) buffer
                x.strides[0],          # spitch: bytes between src rows (row stride)
                xk * itemsize,         # width: bytes copied per row (xk float64s)
                B,                     # height: number of rows (batches)
                3,                     # kind: cudaMemcpyDeviceToDevice
                stream_ptr,            # stream
            )
            _cusparse_lib.cusparseDnVecSetValues(self._x_desc, self._x_buf.data.ptr)

        if y_contig:
            _cusparse_lib.cusparseDnVecSetValues(self._y_desc, y.data.ptr)
        else:
            if beta != 0.0:
                # cuSPARSE reads y before writing when beta != 0, so we must
                # stage the caller's current y values into y_buf first;
                # otherwise the read would see stale contents left over from
                # a previous call.
                rt.memcpy2DAsync(
                    self._y_buf.data.ptr,  # dst: internal packed y buffer
                    yk * itemsize,         # dpitch: tight yk*8 bytes per row
                    y.data.ptr,            # src: caller's y buffer
                    y.strides[0],          # spitch: caller's row stride
                    yk * itemsize,         # width: bytes copied per row
                    B,                     # height: number of batches
                    3,                     # kind: cudaMemcpyDeviceToDevice
                    stream_ptr,            # stream
                )
            _cusparse_lib.cusparseDnVecSetValues(self._y_desc, self._y_buf.data.ptr)

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

        if not y_contig:
            # Copy the SpMV result back out: internal packed y_buf -> caller's
            # strided y. Same shape as the input stage, direction reversed.
            rt.memcpy2DAsync(
                y.data.ptr,            # dst: caller's (strided) y buffer
                y.strides[0],          # dpitch: caller's row stride in bytes
                self._y_buf.data.ptr,  # src: internal packed y buffer
                yk * itemsize,         # spitch: tight yk*8 bytes per row
                yk * itemsize,         # width: bytes copied per row
                B,                     # height: number of batches
                3,                     # kind: cudaMemcpyDeviceToDevice
                stream_ptr,            # stream
            )

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
