import ctypes
from typing import Union
import cupy as cp
from cupyx.scipy.sparse import spmatrix, csr_matrix, csc_matrix, coo_matrix
import cupy.cuda.runtime as rt
from nvmath.bindings import cusparse

from .batched_csr import UniformBatchedCsrMatrix


# ---------------------------------------------------------------------------
# Graph-safe cuSPARSE entry points via nvmath-python.
# ---------------------------------------------------------------------------
def _create_cusparse_handle():
    return cusparse.create()


def _destroy_cusparse_handle(handle):
    cusparse.destroy(handle)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _idx_type(arr: cp.ndarray) -> int:
    """Return the cuSPARSE index-type constant matching *arr*'s dtype."""
    if arr.dtype == cp.int32:
        return cusparse.IndexType.INDEX_32I
    if arr.dtype == cp.int64:
        return cusparse.IndexType.INDEX_64I
    raise TypeError(
        "Sparse matvec indices must have dtype int32 or int64; "
        f"got {arr.dtype}."
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
    pointers and dispatches ``cusparseSpMV`` via ``nvmath.bindings.cusparse``
    — no device allocation, no host-device sync, safe for stream capture.

    Lifetime: the sparse matrix ``mat`` is kept alive through ``self._mat``.
    The descriptor owns copies of its index buffers but reuses ``mat.data``.
    In-place value updates are therefore visible on subsequent calls as long
    as that value allocation is not replaced (e.g. ``mat.data[:] = ...`` is
    safe; ``mat.data = new_array`` is not).

    Parameters
    ----------
    mat : csr_matrix | csc_matrix | coo_matrix
        The sparse matrix. Values are shared; sparse structure is copied once.
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
            cusparse.Operation.TRANSPOSE
            if self._transa
            else cusparse.Operation.NON_TRANSPOSE
        )

        # ---- data type — pick to match the matrix's element dtype ----
        mat_dtype = self._mat.dtype
        if mat_dtype == cp.float32:
            self._compute_type = rt.CUDA_R_32F
            self._np_dtype = cp.float32
        elif mat_dtype == cp.float64:
            self._compute_type = rt.CUDA_R_64F
            self._np_dtype = cp.float64
        else:
            raise TypeError(
                f"Sparse matvec supports float32 and float64 matrix data only; "
                f"got dtype {mat_dtype}."
            )

        # Own descriptor structure while sharing values for in-place updates.
        # Mutating caller indices cannot invalidate the native access pattern.
        self._owned_structure = self._copy_structure(self._mat)
        self._mat_desc = self._create_mat_desc(
            self._mat, self._compute_type, structure=self._owned_structure,
        )
        self._mat_data_ptr = self._mat.data.data.ptr

        # ---- dense vector descriptors ----
        if self._transa:
            x_size, y_size = self._mat.shape[0], self._mat.shape[1]
        else:
            x_size, y_size = self._mat.shape[1], self._mat.shape[0]
        # Frozen at setup; checked against caller-supplied x, y in __call__.
        self._x_size = x_size
        self._y_size = y_size

        dummy_x = cp.empty(x_size, dtype=self._np_dtype)
        dummy_y = cp.empty(y_size, dtype=self._np_dtype)
        self._x_desc = cusparse.create_dn_vec(x_size, dummy_x.data.ptr, self._compute_type)
        self._y_desc = cusparse.create_dn_vec(y_size, dummy_y.data.ptr, self._compute_type)

        # ---- workspace buffer (allocated once; size is independent of scalars) ----
        self._alg = cusparse.SpMVAlg.DEFAULT
        # Alpha/beta scalar type must match compute_type (32F → float, 64F → double).
        self._c_scalar = ctypes.c_float if self._np_dtype == cp.float32 else ctypes.c_double
        _alpha_placeholder = self._c_scalar(1.0)
        _beta_placeholder = self._c_scalar(0.0)
        buf_size = cusparse.sp_mv_buffer_size(
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
    def _copy_structure(mat: spmatrix) -> tuple:
        """Copy and retain the index buffers used by a cuSPARSE descriptor."""
        if isinstance(mat, (csr_matrix, csc_matrix)):
            return (mat.indptr.copy(), mat.indices.copy())
        if isinstance(mat, coo_matrix):
            return (mat.row.copy(), mat.col.copy())
        return ()  # unreachable: _create_mat_desc already rejected the type

    @staticmethod
    def _create_mat_desc(
        mat: spmatrix, compute_type: int, structure: tuple = None,
    ) -> int:
        """Create a cuSPARSE sparse-matrix descriptor for *mat*.

        Supports CSR, CSC, and COO formats. Values always reuse ``mat.data``.
        If *structure* is provided, its retained index buffers are used
        instead of the matrix's caller-owned sparse structure.
        """
        rows, cols = mat.shape
        if isinstance(mat, csr_matrix):
            indptr, indices = structure or (mat.indptr, mat.indices)
            return cusparse.create_csr(
                rows, cols, mat.nnz,
                indptr.data.ptr,
                indices.data.ptr,
                mat.data.data.ptr,
                _idx_type(indptr),
                _idx_type(indices),
                cusparse.IndexBase.ZERO,
                compute_type,
            )
        elif isinstance(mat, csc_matrix):
            indptr, indices = structure or (mat.indptr, mat.indices)
            return cusparse.create_csc(
                rows, cols, mat.nnz,
                indptr.data.ptr,
                indices.data.ptr,
                mat.data.data.ptr,
                _idx_type(indptr),
                _idx_type(indices),
                cusparse.IndexBase.ZERO,
                compute_type,
            )
        elif isinstance(mat, coo_matrix):
            row, col = structure or (mat.row, mat.col)
            row_type = _idx_type(row)
            if _idx_type(col) != row_type:
                raise TypeError(
                    "COO row and column indices must use the same dtype."
                )
            return cusparse.create_coo(
                rows, cols, mat.nnz,
                row.data.ptr,
                col.data.ptr,
                mat.data.data.ptr,
                row_type,
                cusparse.IndexBase.ZERO,
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
        # Validate against what cuSPARSE actually needs: a contiguous run
        # of the right number of elements with the right dtype. We don't
        # constrain ndim — callers may pass either ``(n,)`` or ``(1, n)``,
        # both of which present the same flat byte layout to cuSPARSE.
        if x.size != self._x_size or x.dtype != self._np_dtype or not x.flags.c_contiguous:
            raise ValueError(
                f"x: expected {self._x_size} contiguous elements of dtype "
                f"{self._np_dtype}, got size {x.size} dtype {x.dtype} "
                f"c_contiguous={x.flags.c_contiguous}"
            )
        if y.size != self._y_size or y.dtype != self._np_dtype or not y.flags.c_contiguous:
            raise ValueError(
                f"y: expected {self._y_size} contiguous elements of dtype "
                f"{self._np_dtype}, got size {y.size} dtype {y.dtype} "
                f"c_contiguous={y.flags.c_contiguous}"
            )
        if self._mat.data.data.ptr != self._mat_data_ptr:
            raise ValueError(
                "Sparse matrix value buffer has been replaced since construction; "
                "construct a new operator. In-place value mutation remains safe."
            )

        if stream_ptr is None:
            stream_ptr = cp.cuda.get_current_stream().ptr
        cusparse.set_stream(self._cusparse_handle, stream_ptr)

        _alpha = self._c_scalar(alpha)
        _beta = self._c_scalar(beta)
        cusparse.dn_vec_set_values(self._x_desc, x.data.ptr)
        cusparse.dn_vec_set_values(self._y_desc, y.data.ptr)

        cusparse.sp_mv(
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

    def __del__(self):
        try:
            cusparse.destroy_dn_vec(self._x_desc)
        except Exception:
            pass
        try:
            cusparse.destroy_dn_vec(self._y_desc)
        except Exception:
            pass
        try:
            cusparse.destroy_sp_mat(self._mat_desc)
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
    ``cusparseSpMV`` via ``nvmath.bindings.cusparse``.

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
        mats: UniformBatchedCsrMatrix,
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
        # data type — pick to match the matrix's element dtype.
        # See SingleSparseMatVecProduct for why the silent fallback is unsafe.
        mat_dtype = self._block_diag_mat.dtype
        if mat_dtype == cp.float32:
            self._compute_type = rt.CUDA_R_32F
            self._np_dtype = cp.float32
            self._c_scalar = ctypes.c_float
        elif mat_dtype == cp.float64:
            self._compute_type = rt.CUDA_R_64F
            self._np_dtype = cp.float64
            self._c_scalar = ctypes.c_double
        else:
            raise TypeError(
                f"Batched sparse matvec supports float32 and float64 matrix "
                f"data only; got dtype {mat_dtype}."
            )
        self._op = (
            cusparse.Operation.TRANSPOSE
            if self._transa
            else cusparse.Operation.NON_TRANSPOSE
        )
        self._alg = cusparse.SpMVAlg.DEFAULT
        self._mat_desc = SingleSparseMatVecProduct._create_mat_desc(
            self._block_diag_mat, self._compute_type,
        )
        # Pin the descriptor's value-buffer pointer. ``_block_diag_mat.data``
        # is a zero-copy reshape of ``self._mats.data``; if the caller does
        # ``mats.data = new_array`` the reshape view goes stale and so does
        # the descriptor. We compare current ``self._mats.data.data.ptr``
        # against this snapshot in __call__ and raise if it has changed.
        self._mats_data_ptr = self._mats.data.data.ptr

        # Per-problem vector lengths (before stacking).
        if self._transa:
            self._x_vec_len, self._y_vec_len = rows, cols
        else:
            self._x_vec_len, self._y_vec_len = cols, rows

        # Persistent contiguous staging buffers. Used when the caller passes
        # non-contiguous (B, k) views (e.g. column slices of a wider buffer).
        x_size = B * self._x_vec_len
        y_size = B * self._y_vec_len
        self._x_buf = cp.empty(x_size, dtype=self._np_dtype)
        self._y_buf = cp.empty(y_size, dtype=self._np_dtype)
        self._x_desc = cusparse.create_dn_vec(x_size, self._x_buf.data.ptr, self._compute_type)
        self._y_desc = cusparse.create_dn_vec(y_size, self._y_buf.data.ptr, self._compute_type)

        # Workspace size is independent of the alpha/beta scalars.
        _alpha_placeholder = self._c_scalar(1.0)
        _beta_placeholder = self._c_scalar(0.0)
        buf_size = cusparse.sp_mv_buffer_size(
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
        B = self._batch_size
        xk, yk = self._x_vec_len, self._y_vec_len
        itemsize = self._x_buf.itemsize  # 4 for f32, 8 for f64

        # Validate caller's arrays. cuSPARSE only needs the right total
        # element count + dtype; we don't constrain ndim on the contiguous
        # fast path. For the non-contiguous staging path we additionally
        # require strict (B, k) shape with stride[-1] == itemsize, and
        # stride[0] >= row_bytes — the last guard rules out negative pitches
        # (e.g. x[::-1]) and overlapping rows (as_strided tricks), both of
        # which would make cudaMemcpy2DAsync read unintended memory.
        x_contig = x.flags['C_CONTIGUOUS']
        y_contig = y.flags['C_CONTIGUOUS']
        x_row_bytes = xk * itemsize
        y_row_bytes = yk * itemsize
        if x.size != B * xk or x.dtype != self._np_dtype:
            raise ValueError(
                f"x: expected {B * xk} elements of dtype {self._np_dtype}, "
                f"got size {x.size} dtype {x.dtype}"
            )
        if y.size != B * yk or y.dtype != self._np_dtype:
            raise ValueError(
                f"y: expected {B * yk} elements of dtype {self._np_dtype}, "
                f"got size {y.size} dtype {y.dtype}"
            )
        if not x_contig:
            if (x.shape != (B, xk) or x.strides[-1] != itemsize
                    or x.strides[0] < x_row_bytes):
                raise ValueError(
                    f"non-contiguous x must have shape ({B}, {xk}), "
                    f"stride[-1] == {itemsize} bytes, and stride[0] >= "
                    f"{x_row_bytes} bytes (non-overlapping, forward rows); "
                    f"got shape {x.shape} strides {x.strides}"
                )
        if not y_contig:
            if (y.shape != (B, yk) or y.strides[-1] != itemsize
                    or y.strides[0] < y_row_bytes):
                raise ValueError(
                    f"non-contiguous y must have shape ({B}, {yk}), "
                    f"stride[-1] == {itemsize} bytes, and stride[0] >= "
                    f"{y_row_bytes} bytes (non-overlapping, forward rows); "
                    f"got shape {y.shape} strides {y.strides}"
                )
        if self._mats.data.data.ptr != self._mats_data_ptr:
            raise ValueError(
                "Batched sparse matrix value buffer (``mats.data``) has been "
                "replaced since construction. The cuSPARSE descriptor still "
                "references the original buffer; construct a new "
                "BatchedSparseMatVecProduct with the new matrix. In-place "
                "mutation (``mats.data[:] = ...``) remains safe."
            )

        if stream_ptr is None:
            stream_ptr = cp.cuda.get_current_stream().ptr
        cusparse.set_stream(self._cusparse_handle, stream_ptr)

        _alpha = self._c_scalar(alpha)
        _beta = self._c_scalar(beta)

        if x_contig:
            cusparse.dn_vec_set_values(self._x_desc, x.data.ptr)
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
            cusparse.dn_vec_set_values(self._x_desc, self._x_buf.data.ptr)

        if y_contig:
            cusparse.dn_vec_set_values(self._y_desc, y.data.ptr)
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
            cusparse.dn_vec_set_values(self._y_desc, self._y_buf.data.ptr)

        cusparse.sp_mv(
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
            cusparse.destroy_dn_vec(self._x_desc)
        except Exception:
            pass
        try:
            cusparse.destroy_dn_vec(self._y_desc)
        except Exception:
            pass
        try:
            cusparse.destroy_sp_mat(self._mat_desc)
        except Exception:
            pass

        handle = getattr(self, "_cusparse_handle", None)
        if handle is not None:
            try:
                _destroy_cusparse_handle(handle)
            except Exception:
                pass
