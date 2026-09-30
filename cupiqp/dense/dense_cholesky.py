"""In-place dense Cholesky factorization and solves through cuSOLVER.

Both classes work on C-contiguous Warp arrays of the solver dtype: the
condensed KKT matrices ``(n, n)`` / ``(B, n, n)`` and right-hand sides
``(n,)`` / ``(B, n)`` (rows contiguous, batch stride free). Nothing here
checks layout or dtype; the caller guarantees both. The cuSOLVER handle is
bound once, to the Warp stream that is current when the object is created.
"""
import warp as wp
from nvmath.bindings import cublas, cusolverDn


@wp.kernel
def _fill_ptrs_kernel(base: wp.int64, stride: wp.int64,
                      out: wp.array(dtype=wp.int64)):  # type: ignore
    # out[i] = base + i * stride : device pointer to each matrix/row in a
    # batch, computed from the base address and outer batch stride (bytes).
    i = wp.tid()
    out[i] = base + wp.int64(i) * stride


# ---------------------------------------------------------------------------
# Handle/stream management via nvmath-python.
# ---------------------------------------------------------------------------
def cusolver_create_handle():
    """Create a new cuSOLVER dense handle."""
    return cusolverDn.create()


def cusolver_destroy_handle(handle):
    """Destroy a cuSOLVER dense handle."""
    cusolverDn.destroy(handle)


def cusolver_set_stream(handle, stream_ptr):
    """Associate a CUDA stream with the cuSOLVER handle."""
    cusolverDn.set_stream(handle, stream_ptr)


class CholeskyInplaceSolver:
    """Perform in-place dense Cholesky factorization and solves using cuSOLVER.

    Code inspired by cupy.linalg.cholesky implementation, but adapted for repeated use
    on the same size matrix without repeated allocations. The matrix is C-contiguous,
    which cuSOLVER (column-major) sees as the upper triangle of its transpose.
    """
    def __init__(self, n: int, dtype=wp.float64):
        self.n = n
        self._dtype = dtype
        self._stream = wp.get_stream("cuda")
        self._cusolver_handle = cusolver_create_handle()
        cusolver_set_stream(self._cusolver_handle, self._stream.cuda_stream)
        self._uplo = cublas.FillMode.UPPER

        if dtype is wp.float32:
            self._potrf = cusolverDn.spotrf
            self._potrs = cusolverDn.spotrs
            buffer_func = cusolverDn.spotrf_buffer_size
        elif dtype is wp.float64:
            self._potrf = cusolverDn.dpotrf
            self._potrs = cusolverDn.dpotrs
            buffer_func = cusolverDn.dpotrf_buffer_size
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

        self._dev_info = wp.zeros(1, dtype=wp.int32, device="cuda")
        self._dev_info_host = wp.zeros(1, dtype=wp.int32, device="cpu", pinned=True)
        self._buffersize = buffer_func(self._cusolver_handle, self._uplo, n, 0, n)
        self._workspace = wp.empty(max(self._buffersize, 1), dtype=dtype, device="cuda")

        self._factor_ptr = None
        self._ctx_A = None  # holds reference to A

    def __del__(self):
        handle = getattr(self, "_cusolver_handle", None)
        if handle is not None:
            try:
                cusolver_destroy_handle(handle)
            except Exception:
                pass

    def factorize(self, A: wp.array) -> bool:
        """In-place Cholesky factorization of the C-contiguous ``(n, n)`` matrix ``A``."""
        # Keep A alive!
        self._ctx_A = A
        self._factor_ptr = A.ptr

        self._potrf(
            self._cusolver_handle,
            self._uplo,
            self.n,
            self._factor_ptr,
            self.n,
            self._workspace.ptr,
            self._buffersize,
            self._dev_info.ptr
        )

        # dev_info == 0 indicates success (one D2H copy and sync)
        wp.copy(self._dev_info_host, self._dev_info, stream=self._stream)
        wp.synchronize_stream(self._stream)
        return bool(self._dev_info_host.numpy()[0] == 0)

    def solve(self, B: wp.array):
        """Combined forward and backward substitution to solve Ax = B in place, B of shape ``(n,)``."""
        if self._factor_ptr is None:
            raise RuntimeError("You must call factorize() before solve().")

        self._potrs(
            self._cusolver_handle,
            self._uplo,
            self.n,
            1,  # nrhs
            self._factor_ptr,
            self.n,
            B.ptr,
            self.n,
            self._dev_info.ptr
        )


class BatchedCholeskyInplaceSolver:
    """Batched in-place dense Cholesky factorization and solve using cuSOLVER.

    Uses ``cusolverDnDpotrfBatched`` / ``cusolverDnDpotrsBatched`` to
    process all B matrices in a single kernel launch.

    The batched API takes a device array of pointers (one per matrix).
    Pointer arrays are cached and rebuilt when the base address changes
    or when the outer batch stride changes.

    Parameters
    ----------
    n : int
        Matrix dimension (each matrix is *n x n*).
    batch_size : int
        Number of matrices in the batch.
    dtype : wp.float32 or wp.float64
        Element type (default ``wp.float64``).
    """

    def __init__(self, n: int, batch_size: int, dtype=wp.float64):
        self._n = n
        self._batch_size = batch_size
        self._dtype = dtype
        self._stream = wp.get_stream("cuda")
        self._cusolver_handle = cusolver_create_handle()
        cusolver_set_stream(self._cusolver_handle, self._stream.cuda_stream)
        self._uplo = cublas.FillMode.UPPER  # C-contiguous -> upper in col-major

        if dtype is wp.float32:
            self._potrf_batched = cusolverDn.spotrf_batched
            self._potrs_batched = cusolverDn.spotrs_batched
        elif dtype is wp.float64:
            self._potrf_batched = cusolverDn.dpotrf_batched
            self._potrs_batched = cusolverDn.dpotrs_batched
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

        self._dev_info = wp.zeros(batch_size, dtype=wp.int32, device="cuda")
        self._dev_info_host = wp.zeros(batch_size, dtype=wp.int32, device="cpu", pinned=True)
        self._dev_info_potrs = wp.zeros(1, dtype=wp.int32, device="cuda")

        # Preallocated device pointer arrays and their source-layout keys.
        self._A_ptrs = wp.zeros(batch_size, dtype=wp.int64, device="cuda")  # store the pointer to each batch
        self._A_ptr_key = None
        self._B_ptrs = wp.zeros(batch_size, dtype=wp.int64, device="cuda")
        self._B_ptr_key = None

        self._ctx_A = None
        self._ctx_B = None
        self._factorized = False

    def _ensure_ptrs(self, arr: wp.array, ptrs: wp.array,
                     cached_key: tuple) -> tuple:
        """Rebuild pointers if the base address or batch stride changed.

        Returns the current source-layout key for caching.
        """
        key = (arr.ptr, arr.strides[0])
        if key != cached_key:
            wp.launch(
                _fill_ptrs_kernel,
                dim=self._batch_size,
                inputs=[wp.int64(arr.ptr), wp.int64(arr.strides[0])],
                outputs=[ptrs],
                device="cuda",
                stream=self._stream,
            )
        return key

    def __del__(self):
        handle = getattr(self, "_cusolver_handle", None)
        if handle is not None:
            try:
                cusolver_destroy_handle(handle)
            except Exception:
                pass

    @property
    def n(self) -> int:
        return self._n

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def factorize(self, A: wp.array) -> bool:
        """In-place Cholesky factorization of all B matrices.

        Parameters
        ----------
        A : wp.array, shape ``(batch_size, n, n)``
            Overwritten with Cholesky factors. Each ``(n, n)`` matrix is
            C-contiguous; the outer batch stride may be arbitrary (the
            per-matrix pointer kernel uses ``strides[0]`` directly).
        """
        self._A_ptr_key = self._ensure_ptrs(A, self._A_ptrs, self._A_ptr_key)
        self._ctx_A = A

        self._potrf_batched(
            self._cusolver_handle,
            self._uplo,
            self.n,
            self._A_ptrs.ptr,
            self.n,
            self._dev_info.ptr,
            self.batch_size,
        )
        self._factorized = True

        # dev_info == 0 for every matrix indicates success (one D2H copy and sync)
        wp.copy(self._dev_info_host, self._dev_info, stream=self._stream)
        wp.synchronize_stream(self._stream)
        return bool((self._dev_info_host.numpy() == 0).all())

    def solve(self, B: wp.array):
        """In-place Cholesky solve.

        Parameters
        ----------
        B : wp.array, shape ``(batch_size, n)``, row-contiguous with arbitrary outer batch stride
            Overwritten with the solution. Single RHS per batch only
            (``potrsBatched`` is invoked with ``nrhs = 1``).
        """
        if not self._factorized:
            raise RuntimeError("You must call factorize() before solve().")

        self._B_ptr_key = self._ensure_ptrs(B, self._B_ptrs, self._B_ptr_key)
        self._ctx_B = B

        self._potrs_batched(
            self._cusolver_handle,
            self._uplo,
            self.n,
            1,  # nrhs
            self._A_ptrs.ptr,
            self.n,
            self._B_ptrs.ptr,
            self.n,
            self._dev_info_potrs.ptr,
            self.batch_size,
        )
