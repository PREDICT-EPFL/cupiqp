from typing import Optional, Tuple

import cupy as cp
import numpy as np
import warp as wp
from cupyx.scipy.sparse import csr_matrix

from ..utils import to_warp_dtype


class UniformBatchedCsrMatrix:
    """A batch of CSR matrices that share one sparsity pattern.

    A batched extension of ``cupyx.scipy.sparse.csr_matrix``: where a single
    CSR matrix stores ``indptr``, ``indices`` and a 1-D ``data`` array of
    length ``nnz``, this type stores **one shared** ``indptr`` / ``indices``
    pair plus a **2-D** ``data`` buffer of shape ``(batch_size, nnz)`` - the
    values of all matrices stacked along a leading batch axis. Every matrix
    in the batch therefore has the same nonzero structure and differs only
    in its values.

    The storage is Warp arrays owned by this object: ``indptr`` and
    ``indices`` are ``int32`` and ``data`` has the value dtype. cuSPARSE and
    cuDSS read them through their device pointers and the solver's Warp
    kernels index them directly. cupy is used only while constructing the
    pattern; ``mat[b]`` returns a zero-copy ``csr_matrix`` view of one matrix
    for pattern algebra at setup time.

    Parameters
    ----------
    batch_size : int
        Number of matrices in the batch, ``B`` (must be positive).
    indptr, indices : int array
        The shared CSR row-pointer and column-index arrays - the single
        sparsity pattern used by every matrix in the batch. Any host or GPU
        integer array; copied into ``int32`` Warp arrays.
    data : GPU array
        Values of shape ``(batch_size, nnz)``, row ``i`` holding the nonzeros
        of the ``i``-th matrix in the order of ``indices``, or ``(nnz,)`` for
        the same values in every matrix. Copied into the owned buffer.
    shape : tuple of (int, int)
        Dense ``(rows, cols)`` shape of each matrix.
    dtype : Warp scalar type, default: ``wp.float64``
        Value dtype (``wp.float32`` or ``wp.float64``).
    device : str, default: ``"cuda"``
        Warp device of the storage.

    Attributes
    ----------
    batch_size, nnz, rows, cols, shape :
        Batch size ``B``, nonzeros per matrix, and the shared dense shape.
    indptr, indices : warp.array of int32
        The shared CSR sparsity pattern.
    data : warp.array2d
        The ``(batch_size, nnz)`` values buffer.
    """
    def __init__(
        self,
        batch_size: int,
        indptr,
        indices,
        data,
        shape: Tuple[int, int],
        dtype=wp.float64,
        device: str = "cuda"
    ):
        if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        batch_size = int(batch_size)
        rows, cols = int(shape[0]), int(shape[1])
        dtype = to_warp_dtype(dtype)

        # The pattern is validated and normalized with cupy (setup time only).
        indptr_cp = cp.asarray(indptr, dtype=cp.int32).ravel()
        indices_cp = cp.asarray(indices, dtype=cp.int32).ravel()
        nnz = int(indices_cp.size)
        if int(indptr_cp.size) != rows + 1:
            raise ValueError(
                f"indptr must have length rows + 1 = {rows + 1}; got {int(indptr_cp.size)}."
            )
        data_cp = cp.asarray(data, dtype=wp.dtype_to_numpy(dtype))
        if data_cp.shape == (nnz,):
            data_cp = cp.broadcast_to(data_cp, (batch_size, nnz))
        if data_cp.shape != (batch_size, nnz):
            raise ValueError(
                f"data must have shape ({batch_size}, {nnz}) or ({nnz},), got {data_cp.shape}."
            )

        self._batch_size = batch_size
        self._nnz = nnz
        self._rows = rows
        self._cols = cols
        self._dtype = dtype
        self._device = device
        self._indptr = to_wp_int32(indptr_cp, device)
        self._indices = to_wp_int32(indices_cp, device)
        self._data = wp.empty((batch_size, nnz), dtype=dtype, device=device)
        if nnz > 0:
            wp.copy(self._data, wp.array(data_cp, dtype=dtype, copy=False))
        self._row_indices: Optional[wp.array] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def nnz(self) -> int:
        return self._nnz

    @property
    def rows(self) -> int:
        return self._rows

    @property
    def cols(self) -> int:
        return self._cols

    @property
    def shape(self) -> Tuple[int, int, int]:
        return (self._batch_size, self._rows, self._cols)

    @property
    def dtype(self):
        """Value dtype, a Warp scalar type."""
        return self._dtype

    @property
    def device(self) -> str:
        return self._device

    @property
    def indptr(self) -> wp.array:
        """Shared CSR row pointers, ``(rows + 1,)`` int32."""
        return self._indptr

    @property
    def indices(self) -> wp.array:
        """Shared CSR column indices, ``(nnz,)`` int32."""
        return self._indices

    @property
    def data(self) -> wp.array:
        """Values, ``(batch_size, nnz)``."""
        return self._data

    @property
    def row_indices(self) -> wp.array:
        """Row index of every stored entry, ``(nnz,)`` int32 (the COO row array).

        Depends only on the pattern, so it is computed once, with cupy, on
        first use.
        """
        if self._row_indices is None:
            if self._nnz == 0:
                self._row_indices = wp.zeros(0, dtype=wp.int32, device=self._device)
            else:
                indptr_cp = cp.asarray(self._indptr)
                rows = cp.searchsorted(indptr_cp[1:], cp.arange(self._nnz, dtype=cp.int32), side="right")
                self._row_indices = to_wp_int32(rows, self._device)
        return self._row_indices

    def __getitem__(self, key: int) -> csr_matrix:
        """Zero-copy ``csr_matrix`` view of the ``key``-th matrix (cupy views of
        the Warp buffers), for pattern algebra and library setup calls."""
        return csr_matrix(
            (cp.asarray(self._data)[key], cp.asarray(self._indices), cp.asarray(self._indptr)),
            shape=(self._rows, self._cols)
        )

    @classmethod
    def empty(
        cls, batch_size: int, rows: int, cols: int, dtype=wp.float64, device: str = "cuda"
    ) -> "UniformBatchedCsrMatrix":
        """Build a batch of ``(rows, cols)`` matrices with no stored entries."""
        return cls(
            batch_size=batch_size,
            indptr=np.zeros(rows + 1, dtype=np.int32),
            indices=np.zeros(0, dtype=np.int32),
            data=cp.empty((batch_size, 0), dtype=wp.dtype_to_numpy(to_warp_dtype(dtype))),
            shape=(rows, cols),
            dtype=dtype,
            device=device
        )


def to_wp_int32(values, device: str = "cuda") -> wp.array:
    """Owned ``int32`` Warp copy of an integer array (host or GPU); a
    zero-length input gives an empty array instead of a null-pointer view."""
    values_cp = cp.asarray(values, dtype=cp.int32).ravel()
    if values_cp.size == 0:
        return wp.zeros(0, dtype=wp.int32, device=device)
    return wp.array(cp.ascontiguousarray(values_cp), dtype=wp.int32, device=device)
