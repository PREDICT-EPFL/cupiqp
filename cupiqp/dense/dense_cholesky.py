import ctypes
import ctypes.util
import torch

from .cublas_wrappers import FILL_UPPER


# ---------------------------------------------------------------------------
# Load cuSOLVER shared library for handle management
# ---------------------------------------------------------------------------
def _load_cusolver_lib() -> ctypes.CDLL:
    try:
        cuda_version = torch.version.cuda
        if cuda_version:
            major = int(cuda_version.split('.')[0])
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

# cusolverDnDpotrf
_cusolver_dn_dpotrf_buffer_size = _cusolver_lib.cusolverDnDpotrf_bufferSize
_cusolver_dn_dpotrf_buffer_size.restype = ctypes.c_int
_cusolver_dn_dpotrf_buffer_size.argtypes = [
    ctypes.c_void_p,  # handle
    ctypes.c_int,     # uplo
    ctypes.c_int,     # n
    ctypes.c_void_p,  # A
    ctypes.c_int,     # lda
    ctypes.POINTER(ctypes.c_int),  # lwork
]

_cusolver_dn_dpotrf = _cusolver_lib.cusolverDnDpotrf
_cusolver_dn_dpotrf.restype = ctypes.c_int
_cusolver_dn_dpotrf.argtypes = [
    ctypes.c_void_p,  # handle
    ctypes.c_int,     # uplo
    ctypes.c_int,     # n
    ctypes.c_void_p,  # A
    ctypes.c_int,     # lda
    ctypes.c_void_p,  # workspace
    ctypes.c_int,     # lwork
    ctypes.c_void_p,  # devInfo
]

# cusolverDnDpotrs
_cusolver_dn_dpotrs = _cusolver_lib.cusolverDnDpotrs
_cusolver_dn_dpotrs.restype = ctypes.c_int
_cusolver_dn_dpotrs.argtypes = [
    ctypes.c_void_p,  # handle
    ctypes.c_int,     # uplo
    ctypes.c_int,     # n
    ctypes.c_int,     # nrhs
    ctypes.c_void_p,  # A
    ctypes.c_int,     # lda
    ctypes.c_void_p,  # B
    ctypes.c_int,     # ldb
    ctypes.c_void_p,  # devInfo
]

# cuSOLVER fill mode constants
CUBLAS_FILL_MODE_LOWER = 0
CUBLAS_FILL_MODE_UPPER = 1


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

    Code inspired by cupy.linalg.cholesky implementation, but adapted for repeated use
    on the same size matrix without repeated allocations.
    """
    def __init__(self, n: int, dtype=torch.float64):
        self.n = n
        self._cusolver_handle = cusolver_create_handle()

        # Get buffer size
        lwork = ctypes.c_int(0)
        _cusolver_dn_dpotrf_buffer_size(
            self._cusolver_handle, CUBLAS_FILL_MODE_UPPER, n, 0, n, ctypes.byref(lwork))

        self._buffersize = lwork.value
        self._workspace = torch.empty(self._buffersize, dtype=torch.float64, device='cuda')
        self._dev_info = torch.empty(1, dtype=torch.int32, device='cuda')

        self._factor_ptr = None
        self._uplo = None
        self._ctx_A = None  # holds reference to A

    def __del__(self):
        handle = getattr(self, "_cusolver_handle", None)
        if handle is not None:
            try:
                cusolver_destroy_handle(handle)
            except Exception:
                pass

    def factorize(self, A: torch.Tensor) -> bool:
        # point the cusolver handle at torch's current stream
        cusolver_set_stream(self._cusolver_handle, torch.cuda.current_stream().cuda_stream)

        if A.dtype != torch.float64:
            raise TypeError(f"Input matrix dtype {A.dtype} does not match solver dtype float64.")
        if A.shape[0] != self.n or A.shape[1] != self.n:
            raise ValueError(f"Shape mismatch. Expected ({self.n}, {self.n}), got {A.shape}")

        # Layout Detection
        # For C-contiguous (row-major), cuSOLVER sees it as column-major upper = row-major lower
        if A.stride(0) == 1:
            # F-contiguous
            self._uplo = CUBLAS_FILL_MODE_LOWER
        elif A.stride(1) == 1:
            # C-contiguous
            self._uplo = CUBLAS_FILL_MODE_UPPER
        else:
            raise ValueError("Matrix A must be contiguous (C or F order).")

        # Keep A alive!
        self._ctx_A = A
        self._factor_ptr = A.data_ptr()

        _cusolver_dn_dpotrf(
            self._cusolver_handle,
            self._uplo,
            self.n,
            self._factor_ptr,
            self.n,
            self._workspace.data_ptr(),
            self._buffersize,
            self._dev_info.data_ptr()
        )

        factorization_success = torch.all(self._dev_info == 0)  # dev_info == 0 indicates success
        return factorization_success

    def solve(self, B: torch.Tensor):
        """Combined forward and backward substitution to solve Ax = B."""
        cusolver_set_stream(self._cusolver_handle, torch.cuda.current_stream().cuda_stream)
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

        if not B.is_contiguous():
            raise ValueError("RHS B must be contiguous.")

        _cusolver_dn_dpotrs(
            self._cusolver_handle,
            self._uplo,
            self.n,
            nrhs,
            self._factor_ptr,
            self.n,
            B.data_ptr(),
            ldb,
            self._dev_info.data_ptr()
        )
