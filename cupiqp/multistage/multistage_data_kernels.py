import functools

import warp as wp

from ..utils import to_warp_dtype


wp.set_module_options({"enable_backward": False})


@functools.lru_cache(maxsize=None)
def create_broadcast_copy_kernels(dtype=wp.float64):
    """Build the kernels that copy one problem's data into every batch entry.

    Returns ``(broadcast_3d_to_4d, broadcast_2d_to_3d)``:

    * ``broadcast_3d_to_4d(src (N, r, c), dst (B, N, r, c))`` -- matrix
      blocks. Launch with ``dim=dst.shape``.
    * ``broadcast_2d_to_3d(src (N, r), dst (B, N, r))`` -- block vectors.
      Launch with ``dim=dst.shape``.

    Both write ``dst[b, ...] = src[...]`` for every batch entry ``b``.
    """
    dtype = to_warp_dtype(dtype)

    @wp.kernel
    def broadcast_3d_to_4d(
        src: wp.array3d(dtype=dtype),  # type: ignore   (N, r, c)
        dst: wp.array4d(dtype=dtype),  # type: ignore   (B, N, r, c)
    ):
        b, k, i, j = wp.tid()
        dst[b, k, i, j] = src[k, i, j]

    @wp.kernel
    def broadcast_2d_to_3d(
        src: wp.array2d(dtype=dtype),  # type: ignore   (N, r)
        dst: wp.array3d(dtype=dtype),  # type: ignore   (B, N, r)
    ):
        b, k, i = wp.tid()
        dst[b, k, i] = src[k, i]

    return broadcast_3d_to_4d, broadcast_2d_to_3d
