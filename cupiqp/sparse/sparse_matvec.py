"""Graph-safe cuSPARSE sparse matrix-vector product over Warp CSR storage.

The operator takes a :class:`UniformBatchedCsrMatrix` and keeps every
cuSPARSE resource (handle, descriptors, workspace) from construction on, so a
call only rebinds the dense-vector pointers and dispatches ``cusparseSpMV``
through ``nvmath.bindings.cusparse``: no allocation, no host sync, safe to
record in a CUDA graph. Calls run on Warp's current stream. cupy is used
only while building the block-diagonal pattern for a batch of several.
"""
import ctypes

import cupy as cp
import warp as wp
from nvmath.bindings import cusparse

from .batched_csr import UniformBatchedCsrMatrix, to_wp_int32


# cudaDataType_t values for the two supported value types.
_CUDA_R_64F = 1
_CUDA_R_32F = 2


def _value_type(dtype):
    """``(cuda_data_type, ctypes scalar)`` for a Warp value dtype."""
    if dtype is wp.float64:
        return _CUDA_R_64F, ctypes.c_double
    if dtype is wp.float32:
        return _CUDA_R_32F, ctypes.c_float
    raise TypeError(f"Sparse matvec supports wp.float32 and wp.float64 only; got {dtype}.")


def _ptr(arr: wp.array) -> int:
    """Device address of a Warp array; ``0`` for an array without elements."""
    return arr.ptr if arr.ptr is not None else 0


def _csr_descriptor(rows: int, cols: int, nnz: int, indptr: wp.array, indices: wp.array,
                    data: wp.array, compute_type: int) -> int:
    return cusparse.create_csr(
        rows, cols, nnz, _ptr(indptr), _ptr(indices), _ptr(data),
        cusparse.IndexType.INDEX_32I, cusparse.IndexType.INDEX_32I,
        cusparse.IndexBase.ZERO, compute_type
    )


class SparseMatVecProduct:
    """``y[b] = alpha * op(A[b]) @ x[b] + beta * y[b]`` for every matrix of a
    :class:`UniformBatchedCsrMatrix`, with a single ``cusparseSpMV``.

    ``op`` is the identity or the transpose (``transa``). For ``B > 1`` the
    call runs on a block-diagonal operator whose pattern is built once from
    the shared pattern::

        [ A_0                  ]   [ x_0   ]   [ y_0   ]
        [     A_1              ] @ [ x_1   ] = [ y_1   ]
        [           ...        ]   [  ...  ]   [  ...  ]
        [               A_B-1  ]   [ x_B-1 ]   [ y_B-1 ]

    and whose values are the ``(B, nnz)`` buffer of ``mats`` seen flat; for
    ``B = 1`` the matrix pattern is used as is. In-place updates of
    ``mats.data`` are seen at the next call.

    ``x`` and ``y`` are ``(B, k)`` Warp arrays. Arrays whose rows are packed
    (C-contiguous, or a single row) are bound to cuSPARSE directly;
    row-strided views (for instance column slices of a wider ``(B, K)``
    buffer) are staged through internal contiguous buffers with ``wp.copy``,
    reading the current ``y`` in first when ``beta != 0``.
    """

    def __init__(self, mats: UniformBatchedCsrMatrix, transa: bool = False):
        self._mats = mats
        self._transa = transa
        self._batch_size = B = mats.batch_size
        rows, cols, nnz = mats.rows, mats.cols, mats.nnz
        self._compute_type, self._c_scalar = _value_type(mats.dtype)

        if B == 1:
            indptr, indices = mats.indptr, mats.indices
        else:
            # Block-diagonal pattern (cupy at setup): big_indptr[b*rows + r] = b*nnz + indptr[r],
            # big_indices[b*nnz + k] = indices[k] + b*cols.
            indptr_cp = cp.asarray(mats.indptr)
            indices_cp = cp.asarray(mats.indices)
            big_indptr = cp.empty(B * rows + 1, dtype=cp.int32)
            big_indptr[:B * rows] = (indptr_cp[:-1][None, :] + (cp.arange(B, dtype=cp.int32) * nnz)[:, None]).reshape(-1)
            big_indptr[-1] = B * nnz
            big_indices = (indices_cp[None, :] + (cp.arange(B, dtype=cp.int32) * cols)[:, None]).reshape(-1)
            indptr = to_wp_int32(big_indptr, mats.device)
            indices = to_wp_int32(big_indices, mats.device)
        self._indptr, self._indices = indptr, indices   # kept alive for the descriptor
        self._mat_desc = _csr_descriptor(B * rows, B * cols, B * nnz, indptr, indices,
                                         mats.data, self._compute_type)

        self._x_len, self._y_len = (rows, cols) if transa else (cols, rows)
        self._cusparse_handle = cusparse.create()
        self._op = cusparse.Operation.TRANSPOSE if transa else cusparse.Operation.NON_TRANSPOSE
        self._alg = cusparse.SpMVAlg.DEFAULT
        # Contiguous staging buffers for row-strided x / y views; they also
        # give the vector descriptors a valid address at creation.
        self._x_buf = wp.empty((B, self._x_len), dtype=mats.dtype, device=mats.device)
        self._y_buf = wp.empty((B, self._y_len), dtype=mats.dtype, device=mats.device)
        self._x_desc = cusparse.create_dn_vec(B * self._x_len, _ptr(self._x_buf), self._compute_type)
        self._y_desc = cusparse.create_dn_vec(B * self._y_len, _ptr(self._y_buf), self._compute_type)

        # Workspace size does not depend on the scalars.
        alpha = self._c_scalar(1.0)
        beta = self._c_scalar(0.0)
        buf_size = cusparse.sp_mv_buffer_size(
            self._cusparse_handle, self._op, ctypes.addressof(alpha), self._mat_desc,
            self._x_desc, ctypes.addressof(beta), self._y_desc, self._compute_type, self._alg
        )
        self._buffer = wp.empty(max(buf_size, 1), dtype=wp.uint8, device=mats.device)

    @staticmethod
    def _packed(v: wp.array) -> bool:
        """True if the rows of ``v`` are consecutive in memory (one row always is)."""
        return v.is_contiguous or int(v.shape[0]) == 1

    def __call__(self, x: wp.array, y: wp.array, alpha: float = 1.0, beta: float = 0.0) -> None:
        """Execute ``y[b] = alpha * op(A[b]) @ x[b] + beta * y[b]`` on the current stream."""
        if self._packed(x):
            x_ptr = x.ptr
        else:
            wp.copy(self._x_buf, x)
            x_ptr = self._x_buf.ptr

        y_packed = self._packed(y)
        if y_packed:
            y_ptr = y.ptr
        else:
            if beta != 0.0:
                # cuSPARSE reads y before writing it when beta != 0.
                wp.copy(self._y_buf, y)
            y_ptr = self._y_buf.ptr

        cusparse.set_stream(self._cusparse_handle, wp.get_stream("cuda").cuda_stream)
        _alpha = self._c_scalar(alpha)
        _beta = self._c_scalar(beta)
        cusparse.dn_vec_set_values(self._x_desc, x_ptr)
        cusparse.dn_vec_set_values(self._y_desc, y_ptr)
        cusparse.sp_mv(
            self._cusparse_handle, self._op, ctypes.addressof(_alpha), self._mat_desc,
            self._x_desc, ctypes.addressof(_beta), self._y_desc, self._compute_type,
            self._alg, self._buffer.ptr
        )

        if not y_packed:
            wp.copy(y, self._y_buf)

    def __del__(self):
        for name, destroy in (("_x_desc", cusparse.destroy_dn_vec),
                              ("_y_desc", cusparse.destroy_dn_vec),
                              ("_mat_desc", cusparse.destroy_sp_mat),
                              ("_cusparse_handle", cusparse.destroy)):
            handle = getattr(self, name, None)
            if handle is not None:
                try:
                    destroy(handle)
                except Exception:
                    pass
