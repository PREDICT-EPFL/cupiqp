from typing import Protocol

import warp as wp


Vector = wp.array  # 1D array
Matrix = wp.array2d  # 2D array

PIQP_INF = 1e20


class CudaArray(Protocol):
    """A GPU-resident dense array exposing the CUDA Array Interface.

    cuPIQP accepts any object satisfying this protocol wherever a dense GPU
    vector or matrix is expected -- e.g. a ``cupy.ndarray``, a dense CUDA
    ``torch.Tensor``, a CUDA JAX array, or a Numba device array. CPU arrays
    do not expose this interface and are rejected (cuPIQP never silently
    copies host data to the device); see :func:`cupiqp.utils.is_cuda_array`.
    """

    @property
    def __cuda_array_interface__(self) -> dict: ...