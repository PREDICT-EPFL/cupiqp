import cupy as cp
from cupy_backends.cuda.libs import cublas, cusolver
from nvmath.bindings import cusolverDn


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
    
    Code insipred by cupy.linalg.cholesky implementation, but adapted for repeated use
    on the same size matrix without repeated allocations.
    """
    def __init__(self, n: int, dtype = cp.float64):
        self.n = n
        self._dtype = cp.dtype(dtype).char
        self._cusolver_handle = cusolver_create_handle()
        
        if self._dtype == 'f':
            self._potrf = cusolver.spotrf
            self._potrs = cusolver.spotrs
            buffer_func = cusolver.spotrf_bufferSize
        elif self._dtype == 'd':
            self._potrf = cusolver.dpotrf
            self._potrs = cusolver.dpotrs
            buffer_func = cusolver.dpotrf_bufferSize
        elif self._dtype == 'F':
            self._potrf = cusolver.cpotrf
            self._potrs = cusolver.cpotrs
            buffer_func = cusolver.cpotrf_bufferSize
        elif self._dtype == 'D':
            self._potrf = cusolver.zpotrf
            self._potrs = cusolver.zpotrs
            buffer_func = cusolver.zpotrf_bufferSize
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

        self._dev_info = cp.empty(1, dtype=cp.int32)
        self._buffersize = buffer_func(self._cusolver_handle, cublas.CUBLAS_FILL_MODE_UPPER, n, 0, n)
        self._workspace = cp.empty(self._buffersize, dtype=self._dtype)

        self._factor_ptr = None
        self._uplo = None
        self._ctx_A = None # holds reference to A

    def __del__(self):
        handle = getattr(self, "_cusolver_handle", None)
        if handle is not None:
            try:
                cusolver_destroy_handle(handle)
            except Exception:
                pass

    def factorize(self, A: cp.ndarray) -> bool:
        # point the cusolver handle at CuPy's current stream
        cusolver_set_stream(self._cusolver_handle, cp.cuda.get_current_stream().ptr)
        if not cp.dtype(A.dtype).char == self._dtype:
            raise TypeError(f"Input matrix dtype {A.dtype} does not match solver dtype {self._dtype}.")
        if A.shape[0] != self.n or A.shape[1] != self.n:
            raise ValueError(f"Shape mismatch. Expected ({self.n}, {self.n}), got {A.shape}")
        
        # Layout Detection
        if A.flags.f_contiguous:
            self._uplo = cublas.CUBLAS_FILL_MODE_LOWER
        elif A.flags.c_contiguous:
            self._uplo = cublas.CUBLAS_FILL_MODE_UPPER
        else:
            raise ValueError("Matrix A must be contiguous (C or F order).")

        # Keep A alive!
        self._ctx_A = A
        self._factor_ptr = A.data.ptr
        
        self._potrf(
            self._cusolver_handle,
            self._uplo,
            self.n,
            self._factor_ptr,
            self.n,
            self._workspace.data.ptr,
            self._buffersize,
            self._dev_info.data.ptr
        )

        factorization_success = cp.all(self._dev_info == 0)  # dev_info == 0 indicates success
        return factorization_success

    def solve(self, B: cp.ndarray):
        """Combined forward and backward substitution to solve Ax = B."""
        cusolver_set_stream(self._cusolver_handle, cp.cuda.get_current_stream().ptr)
        if self._factor_ptr is None:
            raise RuntimeError("You must call factorize() before solve().")

        # Handle dimensions
        if B.ndim == 1:
            nrhs = 1
            ldb = self.n
        else:
            if B.shape[0] != self.n:
                raise ValueError(f"RHS dimension mismatch. Expected rows {self.n}, got {B.shape[0]}")
            nrhs = B.shape[1]
            ldb = self.n

        if not (B.flags.c_contiguous or B.flags.f_contiguous):
             raise ValueError("RHS B must be contiguous.")

        self._potrs(
            self._cusolver_handle,
            self._uplo,
            self.n,
            nrhs,
            self._factor_ptr,
            self.n,
            B.data.ptr,
            ldb,
            self._dev_info.data.ptr
        )

class BatchedCholeskyInplaceSolver:
    """Batched in-place dense Cholesky factorization and solve using cuSOLVER.

    Uses ``cusolverDnDpotrfBatched`` / ``cusolverDnDpotrsBatched`` to
    process all B matrices in a single kernel launch.

    The batched API takes a device array of pointers (one per matrix).
    Pointer arrays are cached and rebuilt only when the base address changes.

    Parameters
    ----------
    n : int
        Matrix dimension (each matrix is *n × n*).
    batch_size : int
        Number of matrices in the batch.
    dtype : dtype
        Element type (default ``float64``).
    """

    def __init__(self, n: int, batch_size: int, dtype=cp.float64):
        self._n = n
        self._batch_size = batch_size
        self._dtype = cp.dtype(dtype).char
        self._cusolver_handle = cusolver_create_handle()
        self._uplo = cublas.CUBLAS_FILL_MODE_UPPER  # C-contiguous → upper in col-major

        if self._dtype == 'f':
            self._potrf_batched = cusolver.spotrfBatched
            self._potrs_batched = cusolver.spotrsBatched
        elif self._dtype == 'd':
            self._potrf_batched = cusolver.dpotrfBatched
            self._potrs_batched = cusolver.dpotrsBatched
        elif self._dtype == 'F':
            self._potrf_batched = cusolver.cpotrfBatched
            self._potrs_batched = cusolver.cpotrsBatched
        elif self._dtype == 'D':
            self._potrf_batched = cusolver.zpotrfBatched
            self._potrs_batched = cusolver.zpotrsBatched
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

        self._dev_info = cp.empty(batch_size, dtype=cp.int32)
        self._dev_info_potrs = cp.empty(1, dtype=cp.int32)

        # Preallocated device pointer arrays and base addresses
        self._A_ptrs = cp.empty(batch_size, dtype=cp.int64)  # store the pointer to each batch
        self._A_base_ptr = 0                                 # store the pointer to the whole batch
        self._B_ptrs = cp.empty(batch_size, dtype=cp.int64)
        self._B_base_ptr = 0

        self._ctx_A = None
        self._ctx_B = None
        self._factorized = False

    def _ensure_ptrs(self, arr: cp.ndarray, ptrs: cp.ndarray,
                     cached_ptr: int) -> int:
        """Rebuild a device pointer array if the base address changed.

        Returns the (possibly updated) base pointer for caching.
        """
        if arr.data.ptr != cached_ptr:
            _fill_ptrs_kernel = cp.ElementwiseKernel(
                'int64 base, int64 stride',
                'int64 out',
                'out = base + i * stride',
                'fill_ptrs',
            )
            _fill_ptrs_kernel(cp.int64(arr.data.ptr),
                              cp.int64(arr.strides[0]), ptrs)
        return arr.data.ptr

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

    def factorize(self, A: cp.ndarray) -> bool:
        """In-place Cholesky factorization of all B matrices.

        Parameters
        ----------
        A : cp.ndarray, shape ``(batch_size, n, n)``
            Overwritten with Cholesky factors.
        """
        cusolver_set_stream(self._cusolver_handle,
                            cp.cuda.get_current_stream().ptr)
        self._A_base_ptr = self._ensure_ptrs(A, self._A_ptrs, self._A_base_ptr)
        self._ctx_A = A

        self._potrf_batched(
            self._cusolver_handle,
            self._uplo,
            self.n,
            self._A_ptrs.data.ptr,
            self.n,
            self._dev_info.data.ptr,
            self.batch_size,
        )
        self._factorized = True
        return bool(cp.all(self._dev_info == 0))

    def solve(self, B: cp.ndarray):
        """In-place Cholesky solve.

        Parameters
        ----------
        B : cp.ndarray, shape ``(batch_size, n)``
            Overwritten with the solution.
        """
        cusolver_set_stream(self._cusolver_handle,
                            cp.cuda.get_current_stream().ptr)
        if not self._factorized:
            raise RuntimeError("You must call factorize() before solve().")

        self._B_base_ptr = self._ensure_ptrs(B, self._B_ptrs, self._B_base_ptr)
        self._ctx_B = B

        self._potrs_batched(
            self._cusolver_handle,
            self._uplo,
            self.n,
            1,  # nrhs
            self._A_ptrs.data.ptr,
            self.n,
            self._B_ptrs.data.ptr,
            self.n,
            self._dev_info_potrs.data.ptr,
            self.batch_size,
        )
