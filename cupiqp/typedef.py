from typing import Protocol

import warp as wp


PIQP_INF = 1e20

# Solver status codes. ``cupiqp.results.Status`` wraps these values for users;
# the device termination kernels write them as plain ``int32`` so that the
# per-problem status can be decided on the GPU without a host round trip.
STATUS_UNSOLVED = -1
STATUS_SOLVED = 0
STATUS_MAX_ITER_REACHED = 1
STATUS_PRIMAL_INFEASIBLE = 2
STATUS_DUAL_INFEASIBLE = 3
STATUS_NUMERICAL_ISSUES = 4


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