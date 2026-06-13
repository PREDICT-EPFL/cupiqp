from typing import Optional, Sequence, Tuple, Union
import cupy as cp
from cupyx.scipy.sparse import csr_matrix



class UniformBatchedCsrMatrix:
    """A batch of CSR matrices that share one sparsity pattern.

    A batched extension of cupy's ``cupyx.scipy.sparse.csr_matrix``: where a
    single CSR matrix stores ``indptr``, ``indices``, and a 1-D ``data`` array
    of length ``nnz``, this type stores **one shared** ``indptr`` / ``indices``
    pair plus a **2-D** ``data`` buffer of shape ``(batch_size, nnz)`` - the
    values of all matrices stacked along a leading batch axis. Every
    matrix in the batch therefore has the *same* nonzero structure and differs
    only in its values.

    This is the storage cuPIQP uses internally for batched sparse problems, and
    the preferred input for solving a batch with ``SparseSolver``: because the
    values are contiguous with a uniform per-matrix stride, batched sparse
    linear-algebra routines can sweep the whole batch with no copy (unlike a
    Python list of separate ``csr_matrix`` objects, which must be copied into
    this layout first).


    Parameters
    ----------
    batch_size : int
        Number of matrices in the batch, ``B`` (must be positive).
    indices, indptr : sequence of int
        The shared CSR column-index and row-pointer arrays - the single
        sparsity pattern used by every matrix in the batch.
    data : cupy.ndarray
        Values of shape ``(batch_size, nnz)``: row ``i`` holds the nonzeros of
        the ``i``-th matrix, laid out against the shared ``indices`` /
        ``indptr``.
    shape : tuple of (int, int), optional
        Dense ``(rows, cols)`` shape of each matrix; inferred from the pattern
        when omitted.
    dtype : data-type, default: ``cupy.float64``
        Value dtype.

    Attributes
    ----------
    batch_size, nnz, rows, cols, shape :
        Batch size ``B``, nonzeros per matrix, and the shared dense shape.
    indices, indptr : cupy.ndarray
        The shared CSR sparsity pattern.
    data : cupy.ndarray
        The ``(batch_size, nnz)`` values buffer.
    """
    def __init__(
        self,
        batch_size: int,
        indices: Sequence[int],
        indptr: Sequence[int],
        data: cp.ndarray,
        shape: Optional[Tuple[int, int]] = None,
        dtype=cp.float64,
    ):
        if not batch_size > 0:
            raise ValueError("batch_size must be a positive integer.")

        self._dtype = dtype
        data_cp = cp.asarray(data, dtype=dtype)
        try:
            if shape is None:
                self._template_matrix = csr_matrix((data_cp[0], indices, indptr))
            else:
                self._template_matrix = csr_matrix(
                    (data_cp[0], indices, indptr), shape=shape,
                )
        except Exception as e:
            raise ValueError(
                "Invalid indices, indptr, and data combination."
            ) from e

        self._batch_size = batch_size
        self._indices = self._template_matrix.indices
        self._indptr = self._template_matrix.indptr
        self._nnz = self._template_matrix.nnz

        if data_cp.shape != (batch_size, self._nnz):
            raise ValueError(
                f"data must have shape ({batch_size}, {self._nnz}), got {data_cp.shape}."
            )
        self.data = cp.empty((batch_size, self._nnz), dtype=dtype)
        self.data[:] = data_cp

    # TODO: consider not allocate self.data but directly point to the provided data. Also, change update_data() as a setter to point to new data
    def update_data(self, new_data: cp.ndarray) -> None:
        """Overwrite the internal values buffer in place.

        ``new_data`` must be a cupy array of shape ``(batch_size, nnz)``.
        Copies element-wise into ``self.data`` — preserves the buffer's
        device pointer, so any cuSPARSE descriptor built on it stays valid.
        """
        new_data_cp = cp.asarray(new_data, dtype=self._dtype)
        if new_data_cp.shape != (self._batch_size, self._nnz):
            raise ValueError(
                f"new_data must have shape ({self._batch_size}, {self._nnz}); "
                f"got {new_data_cp.shape}."
            )
        self.data[:] = new_data_cp

    @staticmethod
    def is_torch_sparse_csr_tensor(obj) -> bool:
        """Return whether *obj* is a Torch CSR tensor."""
        if not (hasattr(obj, "layout") and hasattr(obj, "crow_indices")
                and hasattr(obj, "values")):
            return False
        try:
            import torch
        except ImportError:
            return False
        return isinstance(obj, torch.Tensor) and obj.layout == torch.sparse_csr

    @classmethod
    def from_input(
        cls,
        matrix,
        dtype=cp.float64,
        validate_shared_sparsity: bool = True,
    ) -> "UniformBatchedCsrMatrix":
        """Normalize one accepted sparse input form to uniform batched CSR."""
        if isinstance(matrix, cls):
            return cls(
                batch_size=matrix.batch_size,
                indices=matrix.indices,
                indptr=matrix.indptr,
                data=matrix.data,
                shape=(matrix.rows, matrix.cols),
                dtype=dtype,
            )
        if cls.is_torch_sparse_csr_tensor(matrix):
            return cls.from_torch_sparse_csr_tensor(
                matrix,
                dtype=dtype,
                validate_shared_sparsity=validate_shared_sparsity,
            )
        if isinstance(matrix, (list, tuple)):
            return cls.from_cupy_csr_matrix_sequence(
                matrix,
                dtype=dtype,
                validate_shared_sparsity=validate_shared_sparsity,
            )
        return cls.from_cupy_csr_matrix(matrix, dtype=dtype)

    @classmethod
    def from_cupy_csr_matrix(cls, matrix, batch_size: int = 1, dtype=cp.float64) -> "UniformBatchedCsrMatrix":
        """Build a uniform batched CSR matrix from a single CuPy CSR (or convertible) input.

        The sparsity pattern (``indices``/``indptr``) is stored once and shared across the
        whole batch; the values are replicated into ``batch_size`` identical rows. The
        default ``batch_size=1`` wraps the input as a one-element batch.
        """
        matrix = csr_matrix(matrix, dtype=dtype)
        return cls(
            batch_size=batch_size,
            indices=matrix.indices,
            indptr=matrix.indptr,
            data=matrix.data.reshape(1, -1) if batch_size == 1 else cp.tile(matrix.data, (batch_size, 1)),
            shape=matrix.shape,
            dtype=dtype,
        )

    @classmethod
    def from_cupy_csr_matrix_sequence(
        cls,
        matrices: Sequence[csr_matrix],
        dtype=cp.float64,
        validate_shared_sparsity: bool = True,
    ) -> "UniformBatchedCsrMatrix":
        """Build from a non-empty list or tuple of uniform CuPy CSR matrices."""
        if not isinstance(matrices, (list, tuple)) or len(matrices) == 0:
            raise ValueError("matrices must be a non-empty list or tuple of CuPy csr matrices.")
        matrices = [csr_matrix(matrix, dtype=dtype) for matrix in matrices]
        template = matrices[0]
        if validate_shared_sparsity:
            cls._require_uniform_sparsity(matrices)
        data = (
            cp.stack([matrix.data for matrix in matrices])
            if template.nnz > 0 else cp.empty((len(matrices), 0), dtype=dtype)
            )
        return cls(
            batch_size=len(matrices),
            indices=template.indices,
            indptr=template.indptr,
            data=data,
            shape=template.shape,
            dtype=dtype,
        )
    
    @staticmethod
    def _require_uniform_sparsity(matrices: Sequence[csr_matrix]) -> None:
        """Raise unless all CSR matrices share one shape and structure."""
        template = matrices[0]
        error = "All matrices must share the same CSR sparsity pattern."
        if any(
            matrix.shape != template.shape or matrix.nnz != template.nnz
            for matrix in matrices[1:]
        ):
            raise ValueError(error)
        if len(matrices) == 1:
            return

        indices = cp.stack([matrix.indices for matrix in matrices[1:]])
        indptr = cp.stack([matrix.indptr for matrix in matrices[1:]])
        same = (
            cp.array_equal(indices, cp.broadcast_to(template.indices, indices.shape))
            & cp.array_equal(indptr, cp.broadcast_to(template.indptr, indptr.shape))
        )
        if not bool(same):
            raise ValueError(error)

    @classmethod
    def empty(
        cls, batch_size: int, rows: int, cols: int, dtype=cp.float64,
    ) -> "UniformBatchedCsrMatrix":
        """Build an empty uniform CSR matrix batch with a declared shape."""
        return cls(
            batch_size=batch_size,
            indices=cp.empty(0, dtype=cp.int32),
            indptr=cp.zeros(rows + 1, dtype=cp.int32),
            data=cp.empty((batch_size, 0), dtype=dtype),
            shape=(rows, cols),
            dtype=dtype,
        )

    @classmethod
    def from_torch_sparse_csr_tensor(
        cls, tensor, dtype=cp.float64, validate_shared_sparsity: bool = True,
    ) -> "UniformBatchedCsrMatrix":
        """Build a uniform batched CSR matrix from a torch CSR tensor.

        ``tensor`` may be 2-D with shape ``(M, N)`` or 3-D with shape
        ``(B, M, N)``. For 3-D inputs, every batch must share one CSR
        pattern; by default this is checked before using the first batch's
        ``indptr`` / ``indices`` as the shared structure.

        Device-side input buffers are imported via DLPack before values are
        copied into solver-owned storage.
        """
        import torch  # local import to avoid a hard dependency on torch

        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected a torch.Tensor, got {type(tensor).__name__}.")
        if tensor.layout != torch.sparse_csr:
            raise ValueError(f"Expected torch.sparse_csr layout, got {tensor.layout}.")
        if tensor.dim() not in (2, 3):
            raise ValueError(
                "Expected a sparse CSR tensor of shape (M, N) or (B, M, N); "
                f"got shape {tuple(tensor.shape)}."
            )
        if not tensor.is_cuda:
            raise ValueError("tensor must reside on a CUDA device.")

        if tensor.dim() == 2:
            rows, cols = int(tensor.shape[0]), int(tensor.shape[1])
            indptr = cp.from_dlpack(tensor.crow_indices().contiguous()).astype(
                cp.int32, copy=False,
            )
            indices = cp.from_dlpack(tensor.col_indices().contiguous()).astype(
                cp.int32, copy=False,
            )
            values = cp.from_dlpack(tensor.values().contiguous())
            data = values.reshape(1, -1)
            return cls(
                batch_size=1,
                indices=indices,
                indptr=indptr,
                data=data,
                shape=(rows, cols),
                dtype=dtype,
            )

        B, rows, cols = (
            int(tensor.shape[0]), int(tensor.shape[1]), int(tensor.shape[2])
        )
        if B < 1:
            raise ValueError(
                "A batched sparse CSR tensor must contain at least one matrix."
            )
        crow = tensor.crow_indices()   # (B, M+1)
        col = tensor.col_indices()     # (B, nnz)
        values = tensor.values()       # (B, nnz)

        if validate_shared_sparsity and B > 1:
            crow_all = cp.from_dlpack(crow.contiguous())
            col_all = cp.from_dlpack(col.contiguous())
            same = (
                cp.array_equal(crow_all, cp.broadcast_to(crow_all[0], crow_all.shape))
                & cp.array_equal(col_all, cp.broadcast_to(col_all[0], col_all.shape))
            )
            if not bool(same):
                raise ValueError(
                    "All batch matrices must share the same CSR sparsity pattern."
                )

        # Zero-copy views into torch device memory, then dtype-cast as needed.
        # Slicing crow[0] / col[0] picks the shared per-batch pattern; those
        # rows are contiguous in the underlying 2-D buffer so DLPack works
        # without a copy.
        indptr = cp.from_dlpack(crow[0].contiguous()).astype(cp.int32, copy=False)
        indices = cp.from_dlpack(col[0].contiguous()).astype(cp.int32, copy=False)
        data = cp.from_dlpack(values.contiguous())

        return cls(
            batch_size=B,
            indices=indices,
            indptr=indptr,
            data=data,
            shape=(rows, cols),
            dtype=dtype,
        )

    @property
    def batch_size(self) -> int: return self._batch_size

    @property
    def nnz(self) -> int: return self._nnz

    @property
    def indices(self): return self._indices

    @property
    def indptr(self): return self._indptr

    @property
    def rows(self): return self._template_matrix.shape[0]

    @property
    def cols(self): return self._template_matrix.shape[1]

    @property
    def shape(self) -> Tuple[int]: 
        return (self.batch_size, self._template_matrix.shape[0], self._template_matrix.shape[1])

    def __getitem__(self, key: int) -> csr_matrix:
        return csr_matrix(
            (self.data[key], self._indices, self._indptr),
            shape=self._template_matrix.shape,
        )

    def diagonal(self) -> cp.ndarray:
        """Batched analogue of ``cupyx.scipy.sparse.csr_matrix.diagonal``.

        Returns a ``(batch_size, min(rows, cols))`` cupy float64 array whose
        ``[b, i]`` entry is the value at position ``(i, i)`` of the ``b``-th
        matrix, or ``0`` if that entry is not structurally stored.

        Fully vectorized — no Python loop over batches. The row/column
        mapping depends only on the shared sparsity pattern, so it is
        computed lazily on first call and cached for reuse.
        """
        var_idx, csr_idx = self._diag_index_map()
        k = min(self._template_matrix.shape)
        out = cp.zeros((self._batch_size, k), dtype=self.data.dtype)
        if var_idx.size > 0:
            out[:, var_idx] = self.data[:, csr_idx]
        return out

    def _diag_index_map(self):
        """Cached ``(var_idx, csr_idx)`` locating stored diagonal entries.

        ``var_idx[j]`` is the row/col index of the j-th stored diagonal;
        ``csr_idx[j]`` is its column in ``self.data``. Rows with no stored
        diagonal entry are simply absent from both arrays (``diagonal()``
        leaves those slots at 0). Computed on-device via ``cp.searchsorted``
        + a mask — no Python loop over rows.
        """
        cached = getattr(self, "_diag_idx_cache", None)
        if cached is not None:
            return cached
        if self._nnz == 0:
            empty = cp.empty(0, dtype=cp.int32)
            self._diag_idx_cache = (empty, empty)
            return self._diag_idx_cache
        idx_dtype = self._indptr.dtype
        # Row index of each CSR entry (indptr is non-decreasing).
        rows = cp.searchsorted(
            self._indptr[1:], cp.arange(self._nnz, dtype=idx_dtype),
            side='right',
        )
        is_diag = rows == self._indices
        csr_idx = cp.where(is_diag)[0].astype(cp.int32)
        var_idx = rows[csr_idx].astype(cp.int32)
        self._diag_idx_cache = (var_idx, csr_idx)
        return self._diag_idx_cache

    def __setitem__(self, key: int, value: Union[csr_matrix, cp.ndarray]):
        """Overwrite the ``key``-th matrix. ``value`` is either a csr_matrix
        sharing the template's sparsity, or a 1D cupy array of the nnz values."""
        if isinstance(value, csr_matrix):
            if value.shape != self._template_matrix.shape:
                raise ValueError(
                    f"Shape mismatch. Expected {self._template_matrix.shape}, "
                    f"got {value.shape}."
                )
            if value.nnz != self._nnz:
                raise ValueError(
                    f"nnz mismatch. Expected {self._nnz}, got {value.nnz}. "
                    "Input must share the template's sparsity pattern."
                )
            self.data[key] = value.data
        elif isinstance(value, cp.ndarray):
            if value.shape != (self._nnz,):
                raise ValueError(
                    f"Expected cupy.ndarray of shape ({self._nnz},), "
                    f"got {value.shape}."
                )
            self.data[key] = value
        else:
            raise TypeError(
                f"Unsupported value type {type(value).__name__}; expected "
                "cupyx.scipy.sparse.csr_matrix or cupy.ndarray."
            )
