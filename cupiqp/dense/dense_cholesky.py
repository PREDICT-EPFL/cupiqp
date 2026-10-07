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

        self._factor_status = wp.zeros(1, dtype=wp.int32, device="cuda")
        self._buffersize = buffer_func(self._cusolver_handle, self._uplo, n, 0, n)
        self._workspace = wp.empty(max(self._buffersize, 1), dtype=dtype, device="cuda")

        self._solve_info = wp.zeros(1, dtype=wp.int32, device="cuda")  # potrs argument check, unused
        self._factor_ptr = None
        self._ctx_A = None  # holds reference to A

    def __del__(self):
        handle = getattr(self, "_cusolver_handle", None)
        if handle is not None:
            try:
                cusolver_destroy_handle(handle)
            except Exception:
                pass

    @property
    def factor_status(self) -> wp.array:
        """``(1,)`` int32 device array: cuSOLVER's ``info`` of the last
        ``factorize()``; zero means the factorization succeeded. Written on
        the device, never read on the host by this class."""
        return self._factor_status

    def factorize(self, A: wp.array) -> None:
        """In-place Cholesky factorization of the C-contiguous ``(n, n)``
        matrix ``A``, launched on the current stream without waiting for it;
        the outcome is left in ``factor_status``."""
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
            self._factor_status.ptr
        )

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
            self._solve_info.ptr
        )


class BatchedCholeskyInplaceSolver:
    """Batched in-place dense Cholesky factorization and solve using cuSOLVER.

    Uses ``cusolverDnDpotrfBatched`` / ``cusolverDnDpotrsBatched`` to
    process all B matrices in a single kernel launch.

    The batched API takes a device array of pointers (one per matrix),
    rebuilt by a small kernel on every call so the class keeps no host-side
    state that a CUDA graph replay could invalidate.

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
        # cuSOLVER's potrfBatched uses lower internally. If we choose UPPER,
        # it makes a copy, which introduces some overhead
        # Internally: potrfBatch_upper2lower -> potrf_cta_lower_batch -> potrfBatch_lower2upper
        self._uplo = cublas.FillMode.LOWER

        if dtype is wp.float32:
            self._potrf_batched = cusolverDn.spotrf_batched
            self._potrs_batched = cusolverDn.spotrs_batched
        elif dtype is wp.float64:
            self._potrf_batched = cusolverDn.dpotrf_batched
            self._potrs_batched = cusolverDn.dpotrs_batched
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

        self._factor_status = wp.zeros(batch_size, dtype=wp.int32, device="cuda")
        self._solve_info = wp.zeros(1, dtype=wp.int32, device="cuda")  # potrsBatched argument check, unused

        # Preallocated device pointer arrays (one pointer per matrix / rhs row).
        # They are refilled on every factorize() / solve() rather than cached
        # against a host-side record of the last buffer: a CUDA graph replay
        # rewrites them on the device without the host knowing, so any such
        # record goes stale and a later uncaptured call would skip a needed
        # refill (the initial guess solved into the step buffer that way).
        self._A_ptrs = wp.zeros(batch_size, dtype=wp.int64, device="cuda")
        self._B_ptrs = wp.zeros(batch_size, dtype=wp.int64, device="cuda")

        self._ctx_A = None
        self._ctx_B = None
        self._factorized = False

    def _fill_ptrs(self, arr: wp.array, ptrs: wp.array) -> None:
        """Write the device address of every matrix / row of ``arr`` into ``ptrs``
        (one tiny kernel, from the base address and the outer batch stride)."""
        wp.launch(
            _fill_ptrs_kernel,
            dim=self._batch_size,
            inputs=[wp.int64(arr.ptr), wp.int64(arr.strides[0])],
            outputs=[ptrs],
            device="cuda",
            stream=self._stream,
        )

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

    @property
    def factor_status(self) -> wp.array:
        """``(batch_size,)`` int32 device array: cuSOLVER's per-matrix
        ``info`` of the last ``factorize()``; zero where the factorization
        of that matrix succeeded. Written on the device, never read on the
        host by this class."""
        return self._factor_status

    def factorize(self, A: wp.array) -> None:
        """In-place Cholesky factorization of all B matrices, launched on the
        current stream without waiting for it.

        Parameters
        ----------
        A : wp.array, shape ``(batch_size, n, n)``
            Overwritten with Cholesky factors. Each ``(n, n)`` matrix is
            C-contiguous; the outer batch stride may be arbitrary (the
            per-matrix pointer kernel uses ``strides[0]`` directly).

        The per-matrix outcome is left in ``factor_status``; nothing is read
        on the host.
        """
        self._fill_ptrs(A, self._A_ptrs)
        self._ctx_A = A

        self._potrf_batched(
            self._cusolver_handle,
            self._uplo,
            self.n,
            self._A_ptrs.ptr,
            self.n,
            self._factor_status.ptr,
            self.batch_size,
        )
        self._factorized = True

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

        self._fill_ptrs(B, self._B_ptrs)
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
            self._solve_info.ptr,
            self.batch_size,
        )
