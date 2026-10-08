"""Batched dense matrix products of the dense KKT solver.

Each class handles one matrix shape and chooses its implementation once, at
construction: a one-thread-per-output-entry Warp kernel for small matrices,
cuBLAS otherwise. All products run on Warp's current stream, which the solver
sets to its own stream.
"""
from typing import Optional

import warp as wp

from ..utils import device_ptr
from .cublas_wrappers import (
    FILL_LOWER,
    OP_N,
    cublas_create_handle,
    cublas_destroy_handle,
    cublas_set_stream,
    dgemm_strided_batched,
    dgemv,
    dgemv_strided_batched,
    dsyrk,
    sgemm_strided_batched,
    sgemv,
    sgemv_strided_batched,
    ssyrk,
)
from .dense_matmul_kernels import create_batched_gemv_kernel, create_batched_syrk_kernel


# Largest matrix (rows x cols) whose products run as Warp kernels instead of
# cuBLAS. Below these sizes cuBLAS's smallest batched kernels compute a 32 x 32
# tile per problem, mostly wasted: on an RTX 5090 a one-thread-per-entry Warp
# kernel is up to 145x (A^T A) and 15x (matrix-vector) faster. Above them
# cuBLAS's tiled kernels win.
WARP_MATMUL_MAX_COLS = 16
WARP_MATMUL_MAX_ROWS = 32


def _use_warp(rows: int, cols: int) -> bool:
    """Whether the products with a (rows, cols) matrix run as Warp kernels."""
    return rows == 0 or (rows <= WARP_MATMUL_MAX_ROWS and cols <= WARP_MATMUL_MAX_COLS)


class CublasHandle:
    """A cuBLAS handle shared by several products, created on first use.

    The handle is bound to Warp's current stream when it is created, and
    destroyed when the last product that holds this object is released.
    """

    def __init__(self):
        self._handle: Optional[int] = None

    def get(self) -> int:
        if self._handle is None:
            self._handle = cublas_create_handle()
            cublas_set_stream(self._handle, wp.get_stream("cuda").cuda_stream)
        return self._handle

    def __del__(self):
        if self._handle is not None:
            try:
                cublas_destroy_handle(self._handle)
            except Exception:
                pass


class DenseGemv:
    """Batched matrix-vector products with matrices of shape ``(B, rows, cols)``.

    ``gemv(M, x, z)`` computes ``z = alpha * M @ x`` and
    ``gemv(M, x, z, transpose=True)`` computes ``z = alpha * M^T @ x``. ``x``
    and ``z`` are ``(B, k)`` arrays and may be row-strided views; ``z`` is
    overwritten.
    """

    def __init__(self, batch_size: int, rows: int, cols: int, dtype, cublas: CublasHandle):
        self._batch_size = batch_size
        self._rows = rows
        self._cols = cols
        self._dtype = dtype
        self._use_warp = _use_warp(rows, cols)
        if self._use_warp:
            self._kernel = create_batched_gemv_kernel(False, dtype)
            self._kernel_t = create_batched_gemv_kernel(True, dtype)
        else:
            self._cublas = cublas
            self._handle = cublas.get()
            if batch_size > 1:
                self._cublas_gemv = sgemv_strided_batched if dtype is wp.float32 else dgemv_strided_batched
            else:
                self._cublas_gemv = sgemv if dtype is wp.float32 else dgemv

    def __call__(self, mat: wp.array3d, x: wp.array2d, z: wp.array2d,
                 transpose: bool = False, alpha: float = 1.0) -> None:
        if self._use_warp:
            wp.launch(
                kernel=self._kernel_t if transpose else self._kernel,
                dim=(self._batch_size, self._cols if transpose else self._rows),
                inputs=[mat, x, self._dtype(alpha), z],
                device=mat.device,
            )
        elif self._batch_size > 1:
            self._cublas_gemv(self._handle, mat, x, z, transa=transpose, alpha=alpha, beta=0.0)
        else:
            self._cublas_gemv(self._handle, mat[0], x[0], z[0], transa=transpose, alpha=alpha, beta=0.0)


class DenseSyrk:
    """Batched ``C = A^T A`` for matrices ``A`` of shape ``(B, rows, cols)``.

    Only the row-major upper triangle of ``C`` (``i <= j``, shape
    ``(B, cols, cols)``) is guaranteed: the triangle the dense Cholesky solvers
    read. ``syrk(A, C)`` overwrites it; ``syrk(A, C, accumulate=True)`` adds
    ``A^T A`` to it.
    """

    def __init__(self, batch_size: int, rows: int, cols: int, dtype, cublas: CublasHandle):
        self._batch_size = batch_size
        self._rows = rows
        self._cols = cols
        self._use_warp = _use_warp(rows, cols)
        if self._use_warp:
            self._kernel = create_batched_syrk_kernel(False, dtype)
            self._kernel_add = create_batched_syrk_kernel(True, dtype)
        else:
            self._cublas = cublas
            self._handle = cublas.get()
            if batch_size > 1:
                # No batched syrk in cuBLAS: a strided-batched gemm fills both triangles.
                self._gemm = sgemm_strided_batched if dtype is wp.float32 else dgemm_strided_batched
            else:
                self._syrk = ssyrk if dtype is wp.float32 else dsyrk

    def __call__(self, A: wp.array3d, C: wp.array3d, accumulate: bool = False) -> None:
        beta = 1.0 if accumulate else 0.0
        if self._use_warp:
            wp.launch(
                kernel=self._kernel_add if accumulate else self._kernel,
                dim=(self._batch_size, self._cols, self._cols),
                inputs=[A, C],
                device=A.device,
            )
        elif self._batch_size > 1:
            self._gemm(self._handle, A, A, C, transa=True, transb=False, alpha=1.0, beta=beta)
        else:
            # Column-major FILL_LOWER of (A^T A) is the row-major upper triangle.
            self._syrk(self._handle, FILL_LOWER, OP_N, self._cols, self._rows,
                       1.0, device_ptr(A), self._cols, beta, device_ptr(C), self._cols)
