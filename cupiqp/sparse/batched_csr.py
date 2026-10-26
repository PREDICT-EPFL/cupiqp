from typing import Optional, Tuple, Union

import numpy as np
import warp as wp
from scipy.sparse import csr_matrix

from ..utils import to_warp_dtype, is_cuda_array, as_warp_array, batch_broadcast_view
from .csr_helpers import csr_row_indices


class UniformBatchedCsrMatrix:
    """A batch of CSR matrices that share one sparsity pattern.

    A batched extension of ``scipy.sparse.csr_matrix``: where a single
    CSR matrix stores ``indptr``, ``indices`` and a 1-D ``data`` array of
    length ``nnz``, this type stores **one shared** ``indptr`` / ``indices``
    pair plus a **2-D** ``data`` buffer of shape ``(batch_size, nnz)`` - the
    values of all matrices stacked along a leading batch axis. Every matrix
    in the batch therefore has the same nonzero structure and differs only
    in its values.

    The storage is Warp arrays owned by this object: ``indptr`` and
    ``indices`` are ``int32`` and ``data`` has the value dtype. cuSPARSE and
    cuDSS read them through their device pointers and the solver's Warp
    kernels index them directly. The setup-time pattern algebra (KKT
    assembly, index maps) is done on the host with numpy / scipy:
    ``pattern()`` downloads the pattern when called.

    Parameters
    ----------
    batch_size : int
        Number of matrices in the batch, ``B`` (must be positive).
    indptr, indices : numpy.ndarray or GPU array (warp, cupy, ...) of integers
        The shared CSR row-pointer and column-index arrays - the single
        sparsity pattern used by every matrix in the batch. Any 1-D host or
        GPU integer array; copied once into ``int32`` Warp arrays.
    data : numpy.ndarray or GPU array (warp, cupy, ...)
        Values of shape ``(batch_size, nnz)``, row ``i`` holding the nonzeros
        of the ``i``-th matrix in the order of ``indices``, or ``(nnz,)`` for
        the same values in every matrix. A GPU array must already have the
        value dtype; a host array is converted. Copied into the owned buffer.
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
        The shared CSR sparsity pattern, on the device.
    data : warp.array2d
        The ``(batch_size, nnz)`` values buffer, on the device.
    """
    def __init__(
        self,
        batch_size: int,
        indptr: Union[np.ndarray, wp.array],
        indices: Union[np.ndarray, wp.array],
        data: wp.array,
        shape: Tuple[int, int],
        dtype: Union[type[wp.float32], type[wp.float64]] = wp.float64,
        device: str = "cuda"
    ) -> None:
        if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        batch_size = int(batch_size)
        rows, cols = int(shape[0]), int(shape[1])
        dtype = to_warp_dtype(dtype)

        indptr_wp = to_wp_int32(indptr, device)
        indices_wp = to_wp_int32(indices, device)
        nnz = int(indices_wp.shape[0])
        if int(indptr_wp.shape[0]) != rows + 1:
            raise ValueError(
                f"indptr must have length rows + 1 = {rows + 1}; got {int(indptr_wp.shape[0])}."
            )
        if is_cuda_array(data) or isinstance(data, wp.array):
            data_wp = as_warp_array(data, "data", dtype)
        else:
            data_wp = wp.array(np.ascontiguousarray(data, dtype=wp.dtype_to_numpy(dtype)), dtype=dtype, device=device)
        if tuple(data_wp.shape) == (nnz,):
            data_wp = batch_broadcast_view(data_wp, batch_size)
        if tuple(data_wp.shape) != (batch_size, nnz):
            raise ValueError(
                f"data must have shape ({batch_size}, {nnz}) or ({nnz},), got {tuple(data_wp.shape)}."
            )

        self._batch_size = batch_size
        self._nnz = nnz
        self._rows = rows
        self._cols = cols
        self._dtype = dtype
        self._device = device
        self._indptr = indptr_wp
        self._indices = indices_wp
        self._data = wp.empty((batch_size, nnz), dtype=dtype, device=device)
        if nnz > 0:
            wp.copy(self._data, data_wp)
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
    def device(self):
        return self._device

    @property
    def indptr(self) -> wp.array:
        """Shared CSR row pointers, ``(rows + 1,)`` int32, on the device."""
        return self._indptr

    @property
    def indices(self) -> wp.array:
        """Shared CSR column indices, ``(nnz,)`` int32, on the device."""
        return self._indices

    @property
    def data(self) -> wp.array:
        """Values, ``(batch_size, nnz)`` in ``dtype``, on the device."""
        return self._data

    @property
    def row_indices(self) -> wp.array:
        """Row index of every stored entry, ``(nnz,)`` int32 on the device (the COO row array).

        Depends only on the pattern, so it is computed once, on the host, on
        first use.
        """
        if self._row_indices is None:
            self._row_indices = to_wp_int32(csr_row_indices(self._indptr.numpy()), self._device)
        return self._row_indices

    def pattern(self) -> csr_matrix:
        """Host ``scipy.sparse.csr_matrix`` of the shared sparsity pattern,
        with every stored entry set to one (explicit zeros of the values
        stay stored entries here). Downloads the pattern: for setup-time
        use only."""
        return csr_matrix(
            (np.ones(self._nnz), self._host_indices(), self._indptr.numpy()),
            shape=(self._rows, self._cols)
        )

    def _host_indices(self) -> np.ndarray:
        """Host copy of ``indices`` (a zero-length Warp array has no data to download)."""
        return self._indices.numpy() if self._nnz > 0 else np.zeros(0, dtype=np.int32)

    @classmethod
    def empty(
        cls, batch_size: int, rows: int, cols: int, dtype: Union[type[wp.float32], type[wp.float64]] = wp.float64,
        device: str = "cuda"
    ) -> "UniformBatchedCsrMatrix":
        """Build a batch of ``(rows, cols)`` matrices with no stored entries."""
        return cls(
            batch_size=batch_size,
            indptr=np.zeros(rows + 1, dtype=np.int32),
            indices=np.zeros(0, dtype=np.int32),
            data=np.empty((batch_size, 0), dtype=wp.dtype_to_numpy(to_warp_dtype(dtype))),
            shape=(rows, cols),
            dtype=dtype,
            device=device
        )


def to_wp_int32(values: Union[np.ndarray, wp.array], device: str = "cuda") -> wp.array:
    """Owned 1-D ``int32`` Warp copy of an integer array (host or GPU); a
    zero-length input gives an empty array instead of a null-pointer view.

    A 1-D ``int32`` GPU array is copied on the device; anything else goes
    through the host once.
    """
    if isinstance(values, wp.array) or is_cuda_array(values):
        arr = as_warp_array(values)
        if arr.ndim == 1 and arr.dtype == wp.int32:
            if arr.shape[0] == 0:
                return wp.zeros(0, dtype=wp.int32, device=device)
            out = wp.empty(arr.shape[0], dtype=wp.int32, device=device)
            wp.copy(out, arr)
            return out
        values = arr.numpy()
    values_np = np.ascontiguousarray(values, dtype=np.int32).ravel()
    if values_np.size == 0:
        return wp.zeros(0, dtype=wp.int32, device=device)
    return wp.array(values_np, dtype=wp.int32, device=device)
