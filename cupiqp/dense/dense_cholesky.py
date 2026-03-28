import ctypes
import ctypes.util
import cupy as cp
from cupy_backends.cuda.libs import cublas, cusolver


# ---------------------------------------------------------------------------
# Load cuSOLVER shared library for handle management
# ---------------------------------------------------------------------------
def _load_cusolver_lib() -> ctypes.CDLL:
    try:
        import cupy.cuda.runtime as rt
        major = rt.runtimeGetVersion() // 1000
        try:
            return ctypes.CDLL(f"libcusolver.so.{major}")
        except OSError:
            pass
    except Exception:
        pass
    for name in ("libcusolver.so", "cusolver"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    lib_path = ctypes.util.find_library("cusolver")
    if lib_path:
        return ctypes.CDLL(lib_path)
    raise RuntimeError("Could not find cuSOLVER shared library")


_cusolver_lib = _load_cusolver_lib()

_cusolver_dn_create = _cusolver_lib.cusolverDnCreate
_cusolver_dn_create.restype = ctypes.c_int
_cusolver_dn_create.argtypes = [ctypes.POINTER(ctypes.c_void_p)]

_cusolver_dn_destroy = _cusolver_lib.cusolverDnDestroy
_cusolver_dn_destroy.restype = ctypes.c_int
_cusolver_dn_destroy.argtypes = [ctypes.c_void_p]

_cusolver_dn_set_stream = _cusolver_lib.cusolverDnSetStream
_cusolver_dn_set_stream.restype = ctypes.c_int
_cusolver_dn_set_stream.argtypes = [ctypes.c_void_p, ctypes.c_void_p]


def cusolver_create_handle():
    """Create a new cuSOLVER dense handle."""
    handle = ctypes.c_void_p()
    status = _cusolver_dn_create(ctypes.byref(handle))
    if status != 0:
        raise RuntimeError(f"cusolverDnCreate failed with status {status}")
    return handle.value


def cusolver_destroy_handle(handle):
    """Destroy a cuSOLVER dense handle."""
    status = _cusolver_dn_destroy(handle)
    if status != 0:
        raise RuntimeError(f"cusolverDnDestroy failed with status {status}")


def cusolver_set_stream(handle, stream_ptr):
    """Associate a CUDA stream with the cuSOLVER handle."""
    status = _cusolver_dn_set_stream(handle, stream_ptr)
    if status != 0:
        raise RuntimeError(f"cusolverDnSetStream failed with status {status}")


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

