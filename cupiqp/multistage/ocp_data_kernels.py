"""Warp kernels that write HPIPM-style OCP fields into the multistage block storage.

Every kernel writes one stage block of a batched buffer. A single-problem
value is passed as a zero-stride batch view, so the same kernels serve
unbatched (broadcast) and batched writes.
"""
import functools

import warp as wp

from ..utils import to_warp_dtype


@functools.lru_cache(maxsize=None)
def create_ocp_data_kernels(dtype=wp.float64):
    """Kernels writing OCP fields into ``(B, num_blocks, ...)`` Warp buffers.

    Returns a dict:

    * ``set_block``: ``dst[b, k, row0 + i, col0 + j] = scale * src[b, i, j]``
      (or ``src[b, j, i]`` when ``transpose``); launch ``dim=(B, rows, cols)``
      of the written sub-block.
    * ``set_vec``: ``dst[b, k, col0 + i] = scale * src[b, i]``; launch
      ``dim=(B, len)``.
    * ``set_vec_cols``: ``dst[b, k, cols[i]] = src[b, i]``; launch ``dim=(B, len(cols))``.
    * ``fill_cols``: ``dst[b, k, cols[i]] = value`` for ``k < num_stages``;
      launch ``dim=(B, num_stages, len(cols))``.
    * ``init_coupling_diag``: identity in block row 0 and ``-I`` in the
      following block rows of the ``A`` diagonal blocks (``E_k = I`` default);
      launch ``dim=(B, num_blocks, nx)``.
    """
    dtype = to_warp_dtype(dtype)

    @wp.kernel
    def set_block(
        dst: wp.array4d(dtype=dtype),   # type: ignore  (B, num_blocks, rows, cols)
        k: wp.int32,
        row0: wp.int32,
        col0: wp.int32,
        src: wp.array3d(dtype=dtype),   # type: ignore  (B, r, c) or (B, c, r) if transpose
        scale: dtype,                   # type: ignore
        transpose: wp.int32
    ):
        b, i, j = wp.tid()
        if transpose == wp.int32(1):
            v = src[b, j, i]
        else:
            v = src[b, i, j]
        dst[b, k, row0 + i, col0 + j] = scale * v

    @wp.kernel
    def set_vec(
        dst: wp.array3d(dtype=dtype),   # type: ignore  (B, num_blocks, rows)
        k: wp.int32,
        col0: wp.int32,
        src: wp.array2d(dtype=dtype),   # type: ignore  (B, len)
        scale: dtype,                   # type: ignore
    ):
        b, i = wp.tid()
        dst[b, k, col0 + i] = scale * src[b, i]

    @wp.kernel
    def set_vec_cols(
        dst: wp.array3d(dtype=dtype),   # type: ignore  (B, num_blocks, rows)
        k: wp.int32,
        cols: wp.array(dtype=wp.int32), # type: ignore  (len,)
        src: wp.array2d(dtype=dtype),   # type: ignore  (B, len)
    ):
        b, i = wp.tid()
        dst[b, k, cols[i]] = src[b, i]

    @wp.kernel
    def fill_cols(
        dst: wp.array3d(dtype=dtype),   # type: ignore  (B, num_blocks, rows)
        cols: wp.array(dtype=wp.int32), # type: ignore  (len,)
        value: dtype,                   # type: ignore
    ):
        b, k, i = wp.tid()
        dst[b, k, cols[i]] = value

    @wp.kernel
    def init_coupling_diag(
        A_diag: wp.array4d(dtype=dtype),  # type: ignore  (B, num_blocks, nx, d)
    ):
        b, k, i = wp.tid()
        if k == 0:
            A_diag[b, k, i, i] = dtype(1.0)
        else:
            A_diag[b, k, i, i] = dtype(-1.0)

    return {
        "set_block": set_block,
        "set_vec": set_vec,
        "set_vec_cols": set_vec_cols,
        "fill_cols": fill_cols,
        "init_coupling_diag": init_coupling_diag,
    }
