import warp as wp
from ..utils import to_warp_dtype

# ----------------------------------------------------------------------
# Block container classes always carry a leading batch axis. batch_size
# defaults to 1 for single-QP inputs.
# ----------------------------------------------------------------------
class BlockTridiagMat:
    """Symmetric block-tridiagonal matrix, batched.

    Layout::

        D   shape (B, N, d, d)     - diagonal blocks
        E   shape (B, N-1, d, d)   - lower-diagonal blocks

    The matrix is symmetric; the upper off-diagonal is implicitly the
    transpose of the lower one. batch_size defaults to 1.

    diag_blocks and off_diag_blocks_lower may be assigned a Warp array or
    any CUDA array implementing __cuda_array_interface__ (CuPy, dense CUDA
    PyTorch, JAX, Numba, ...). Non-Warp arrays are wrapped zero-copy; shape
    and dtype must match the construction arguments.
    """
    def __init__(self, num_diag_blocks: int, block_size: int,
                 dtype=wp.float64, device="cuda", batch_size: int = 1):
        self.block_size = block_size
        self._device = device
        self._dtype = to_warp_dtype(dtype)
        self._D_shape = (batch_size, num_diag_blocks, block_size, block_size)
        self._E_shape = (batch_size, num_diag_blocks - 1, block_size, block_size)
        self.D = wp.zeros(self._D_shape, dtype=self._dtype, device=device)
        self.E = wp.zeros(self._E_shape, dtype=self._dtype, device=device)

    @property
    def D(self) -> wp.array:
        """Diagonal blocks with shape (batch_size, N, block_size, block_size)."""
        return self._D

    @D.setter
    def D(self, value):
        arr = wp.from_dlpack(value, dtype=self._dtype)
        if tuple(arr.shape) != self._D_shape:
            raise ValueError(
                f"diag_blocks has shape {tuple(arr.shape)}, "
                f"expected {self._D_shape}"
            )
        self._D = arr

    @property
    def E(self) -> wp.array:
        """Lower off-diagonal blocks with shape (batch_size, N-1, d, d)."""
        return self._E

    @E.setter
    def E(self, value):
        arr = wp.from_dlpack(value, dtype=self._dtype)
        if tuple(arr.shape) != self._E_shape:
            raise ValueError(
                f"off_diag_blocks_lower has shape {tuple(arr.shape)}, "
                f"expected {self._E_shape}"
            )
        self._E = arr

    @property
    def batch_size(self):
        return self.D.shape[0]

    @property
    def num_diag_blocks(self):
        return self.D.shape[1]

    @property
    def rows(self):
        return self.num_diag_blocks * self.D.shape[2]

    @property
    def cols(self):
        return self.num_diag_blocks * self.D.shape[3]

    def clone(self) -> "BlockTridiagMat":
        """Return a deep copy with independent diag / off-diag Warp buffers."""
        new = BlockTridiagMat.__new__(BlockTridiagMat)
        new.block_size = self.block_size
        new._device = self._device
        new._dtype = self._dtype
        new._D_shape = self._D_shape
        new._E_shape = self._E_shape
        new._D = wp.clone(self.D)
        new._E = wp.clone(self.E)
        return new


class BlockBidiagMat:
    """Block lower-bidiagonal matrix, batched.

    Stores A or G of the multistage problem with shape::

        D shape (B, N, rows_of_blocks, cols_of_blocks)
        E shape (B, N, rows_of_blocks, cols_of_blocks)

    Logical structure (per batch)::

        A =
        [ D0
          E0  D1
              E1  D2
                  ...
                  E_{N-2} D_{N-1}
                          E_{N-1} ]

    ``D`` and ``E`` may be assigned a warp array or any CUDA array
    implementing ``__cuda_array_interface__`` (cupy, dense CUDA torch,
    jax, numba, ...). Non-warp arrays are wrapped zero-copy, so the
    container aliases the assigned buffer; shape and dtype must match
    the construction arguments.
    """
    def __init__(self, rows_of_blocks: int, cols_of_blocks: int, N: int,
                 dtype=wp.float64, device="cuda", batch_size: int = 1):
        self.N = N
        self.cols_of_blocks = cols_of_blocks
        self.rows_of_blocks = rows_of_blocks
        self._device = device
        self._dtype = to_warp_dtype(dtype)
        self._shape = (batch_size, N, rows_of_blocks, cols_of_blocks)
        self._D = wp.zeros(self._shape, dtype=self._dtype, device=device)
        self._E = wp.zeros(self._shape, dtype=self._dtype, device=device)

    @property
    def D(self) -> wp.array:
        """Diagonal blocks of shape ``(batch_size, N, rows_of_blocks, cols_of_blocks)``."""
        return self._D

    @D.setter
    def D(self, value):
        arr = wp.from_dlpack(value, dtype=self._dtype)
        if tuple(arr.shape) != self._shape:
            raise ValueError(f"D has shape {tuple(arr.shape)}, expected {self._shape}")
        self._D = arr

    @property
    def E(self) -> wp.array:
        """Sub-diagonal blocks of shape ``(batch_size, N, rows_of_blocks, cols_of_blocks)``."""
        return self._E

    @E.setter
    def E(self, value):
        arr = wp.from_dlpack(value, dtype=self._dtype)
        if tuple(arr.shape) != self._shape:
            raise ValueError(f"E has shape {tuple(arr.shape)}, expected {self._shape}")
        self._E = arr

    @property
    def batch_size(self):
        return self.D.shape[0]

    def clone(self) -> "BlockBidiagMat":
        """Return a deep copy with independent ``D`` / ``E`` warp buffers."""
        new = BlockBidiagMat.__new__(BlockBidiagMat)
        new.N = self.N
        new.cols_of_blocks = self.cols_of_blocks
        new.rows_of_blocks = self.rows_of_blocks
        new._device = self._device
        new._dtype = self._dtype
        new._shape = self._shape
        new._D = wp.clone(self._D)
        new._E = wp.clone(self._E)
        return new


class BlockVec:
    """Block vector, batched. ``data`` shape ``(B, num_blocks, rows)``.

    ``data`` may be assigned a warp array or any CUDA array implementing
    ``__cuda_array_interface__`` (cupy, dense CUDA torch, jax, numba, ...).
    Non-warp arrays are wrapped zero-copy, so the container aliases the
    assigned buffer; shape and dtype must match the construction arguments.
    """
    def __init__(self, num_blocks: int, rows: int,
                 dtype=wp.float64, device="cuda", batch_size: int = 1):
        self.num_blocks = num_blocks
        self.rows = rows
        self._device = device
        self._dtype = to_warp_dtype(dtype)
        self._shape = (batch_size, num_blocks, rows)
        self._data = wp.zeros(self._shape, dtype=self._dtype, device=device)

    @property
    def data(self) -> wp.array:
        """Vector storage of shape ``(batch_size, num_blocks, rows)``."""
        return self._data

    @data.setter
    def data(self, value):
        arr = wp.from_dlpack(value, dtype=self._dtype)
        if tuple(arr.shape) != self._shape:
            raise ValueError(f"data has shape {tuple(arr.shape)}, expected {self._shape}")
        self._data = arr

    @property
    def batch_size(self):
        return self.data.shape[0]

    def clone(self) -> "BlockVec":
        """Return a deep copy with an independent ``data`` warp buffer."""
        new = BlockVec.__new__(BlockVec)
        new.num_blocks = self.num_blocks
        new.rows = self.rows
        new._device = self._device
        new._dtype = self._dtype
        new._shape = self._shape
        new._data = wp.clone(self.data)
        return new
