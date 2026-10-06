import os, importlib.util
from typing import Optional, Union
from abc import ABC, abstractmethod
import numpy as np
import warp as wp
import nvtx
from cuda.bindings import runtime as cudart
from nvmath.bindings import cudss

from .batched_csr import UniformBatchedCsrMatrix



class SparseDirectSolver(ABC):
    """Abstract base for sparse direct solvers - natively supports batching.

    ``matrix`` is a :class:`UniformBatchedCsrMatrix` (``B >= 1``): one CSR
    structure with a packed per-batch values buffer, solved through cuDSS
    uniform batching. The right-hand side and solution are Warp arrays of
    shape ``(B, dim)`` owned by the solver.
    """

    def __init__(self, matrix: UniformBatchedCsrMatrix):
        if matrix.rows != matrix.cols:
            raise ValueError("All matrices must be square.")
        self._batch_size = matrix.batch_size
        self._dim = matrix.rows
        self._mat = matrix

        # NOTE: the direct solver holds pointers into the matrix buffers
        # for in-place factorization and solves. Callers may update the
        # values in place between solves; the buffers are never reallocated.
        # rhs/sol must match the matrix dtype - cuDSS rejects a dtype mismatch.
        self._rhs = wp.empty((self._batch_size, self._dim), dtype=matrix.dtype, device=matrix.device)
        self._sol = wp.empty((self._batch_size, self._dim), dtype=matrix.dtype, device=matrix.device)
        # Per-problem outcome of the last factor(): 0 = success, nonzero = failed.
        self._factor_status = wp.zeros(self._batch_size, dtype=wp.int32, device=matrix.device)

    @property
    def factor_status(self) -> wp.array:
        """``(B,)`` int32 device array written by ``factor()``: zero where the
        factorization of that problem succeeded, nonzero where it failed."""
        return self._factor_status

    @nvtx.annotate("SparseDirectSolver::plan")
    @abstractmethod
    def plan(self, cuda_stream: int) -> bool:
        """Precompute reordering and symbolic factorization"""
        pass

    @nvtx.annotate("SparseDirectSolver::factor")
    @abstractmethod
    def factor(self, cuda_stream: int) -> None:
        """Numerical factorization of the matrix; the per-problem outcome is
        written to ``factor_status``. Call after plan() and before solve()."""
        pass

    @nvtx.annotate("SparseDirectSolver::solve")
    @abstractmethod
    def solve(self, cuda_stream: int):
        """Solve the linear system for the given right-hand side."""
        pass

    def __del__(self):
        """Ensure resources are freed when the solver is garbage collected."""
        pass

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def rhs(self) -> wp.array:
        """The ``(B, dim)`` right-hand side buffer, filled in place before ``solve()``."""
        return self._rhs

    @property
    def sol(self) -> wp.array:
        """The ``(B, dim)`` solution buffer, valid after ``solve()``."""
        return self._sol


class CudssSparseDirectSolver(SparseDirectSolver):
    """cuDSS through its raw bindings, on the Warp buffers directly.

    The matrix descriptor points at the packed ``(B, nnz)`` values of
    ``matrix`` and the dense descriptors at the ``(B, dim)`` ``rhs`` / ``sol``
    buffers; cuDSS uniform batching (``UBATCH_SIZE``) walks the B problems
    behind them. Nothing is copied or allocated per call.
    """

    def __init__(self, matrix: UniformBatchedCsrMatrix, use_deterministic_mode: bool = False) -> None:
        super().__init__(matrix)
        _CUDSS_VALUE_TYPE = {wp.float32: cudss.DataType.R_32F, wp.float64: cudss.DataType.R_64F}
        value_type = _CUDSS_VALUE_TYPE[matrix.dtype]
        index_type = cudss.DataType.R_32I

        self._cudss_handle = cudss.create()
        self._cudss_config = cudss.config_create()
        self._cudss_data = cudss.data_create(self._cudss_handle)

        # Symmetric matrix, only the lower triangle is read. row_end = 0 means
        # standard CSR (row r ends where row r + 1 starts).
        self._cudss_a = cudss.matrix_create_csr(
            self._dim, self._dim, matrix.nnz,
            matrix.indptr.ptr, 0, matrix.indices.ptr, matrix.data.ptr,
            index_type, index_type, value_type,
            cudss.MatrixType.SYMMETRIC, cudss.MatrixViewType.LOWER, cudss.IndexBase.ZERO
        )
        # One column-major right-hand side / solution per problem.
        self._cudss_b = cudss.matrix_create_dn(
            self._dim, 1, self._dim, self._rhs.ptr, value_type, cudss.Layout.COL_MAJOR)
        self._cudss_x = cudss.matrix_create_dn(
            self._dim, 1, self._dim, self._sol.ptr, value_type, cudss.Layout.COL_MAJOR)

        # The multithreading layer speeds up the host part of the analysis.
        mt_lib = self._find_cudss_mt_lib() or os.getenv("CUDSS_THREADING_LIB")
        if mt_lib is not None and os.path.isfile(mt_lib):
            cudss.set_threading_layer(self._cudss_handle, mt_lib)

        # Workspace from the device's default memory pool (stream-ordered
        # cudaMallocAsync) instead of cuDSS's default synchronous cudaMalloc.
        err, pool = cudart.cudaDeviceGetDefaultMemPool(wp.get_device(matrix.device).ordinal)
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaDeviceGetDefaultMemPool failed: {err}")
        cudss.set_async_workspace_allocator(self._cudss_handle, int(pool))

        if self._batch_size > 1:
            self._config_set(cudss.ConfigParam.UBATCH_SIZE, self._batch_size)
        self._config_set(cudss.ConfigParam.HYBRID_MEMORY_MODE, 0)  # equivalent to ExecutionCUDA in nvmath's DirectSolver wrapper
        self._config_set(cudss.ConfigParam.USE_CUDA_REGISTER_MEMORY, 1)
        self._config_set(cudss.ConfigParam.REORDERING_ALG, cudss.ReorderingAlg.DEFAULT)
        self._config_set(cudss.ConfigParam.USE_SUPERPANELS, 0)
        self._config_set(cudss.ConfigParam.IR_N_STEPS, 0)  # NOTE: iterative refinement steps, to be tuned
        # cudss has IR_TOL, but not implemented yet according to https://docs.nvidia.com/cuda/cudss/types.html#c.cudssConfigParam_t.CUDSS_CONFIG_IR_TOL
        # NOTE: pivoting settings left at the cuDSS defaults; possible tuning knobs:
        # self._config_set(cudss.ConfigParam.PIVOT_TYPE, cudss.PivotType.PIVOT_GLOBAL_COL)
        # self._config_set(cudss.ConfigParam.PIVOT_THRESHOLD, 1.0)
        # self._config_set(cudss.ConfigParam.PIVOT_EPSILON, 1e-12)
        # Bit-wise reproducible results across runs; slower kernels.
        if use_deterministic_mode:
            self._config_set(cudss.ConfigParam.DETERMINISTIC_MODE, 1)

    def _config_set(self, param: cudss.ConfigParam, value: Union[int, float]) -> None:
        buf = np.array([value], dtype=cudss.get_config_param_dtype(param))
        cudss.config_set(self._cudss_config, param, buf.ctypes.data, buf.itemsize)

    def __del__(self) -> None:
        # Matrices and data before the handle that created them.
        for name, destroy in (("_cudss_a", cudss.matrix_destroy),
                              ("_cudss_b", cudss.matrix_destroy),
                              ("_cudss_x", cudss.matrix_destroy)):
            ptr = getattr(self, name, None)
            if ptr:
                try:
                    destroy(ptr)
                except Exception:
                    pass
                setattr(self, name, None)
        handle = getattr(self, "_cudss_handle", None)
        if getattr(self, "_cudss_data", None) and handle:
            try:
                cudss.data_destroy(handle, self._cudss_data)
            except Exception:
                pass
            self._cudss_data = None
        if getattr(self, "_cudss_config", None):
            try:
                cudss.config_destroy(self._cudss_config)
            except Exception:
                pass
            self._cudss_config = None
        if handle:
            try:
                cudss.destroy(handle)
            except Exception:
                pass
            self._cudss_handle = None

    def _execute(self, phase: cudss.Phase, cuda_stream: int) -> None:
        cudss.set_stream(self._cudss_handle, cuda_stream)
        cudss.execute(
            self._cudss_handle, phase,
            self._cudss_config, self._cudss_data,
            self._cudss_a, self._cudss_x, self._cudss_b
        )

    @nvtx.annotate("CudssSparseDirectSolver::plan")
    def plan(self, cuda_stream: int) -> bool:
        """Reordering and symbolic factorization (cuDSS ``ANALYSIS`` phase)."""
        try:
            self._execute(cudss.Phase.ANALYSIS, cuda_stream)
        except Exception as e:
            print(f"Planning failed: {e}")
            return False

        return True

    @nvtx.annotate("CudssSparseDirectSolver::factor")
    def factor(self, cuda_stream: int) -> None:
        """Numerical factorization (cuDSS ``FACTORIZATION`` phase), launched on
        the given stream without waiting for it.

        IMPORTANT: the factorization is ALWAYS ASSUMED TO SUCCEED. Nothing is
        checked and ``factor_status`` stays zero for every problem, because
        cuDSS reports its factorization outcome only through blocking host
        queries (see the comment below). Consequently the solver never retries
        a sparse factorization with raised regularization. A factorization
        that does fail shows up per problem only through the finite check of
        the solution (``solve_status``), since a zero pivot gives a non-finite
        solve; a factorization that is merely inaccurate is not detected.
        """
        # Earlier versions checked the outcome on the host after factorize():
        #
        #   if self._batch_size > 1:
        #       return fac_info.info == 0
        #   # NOTE: this causes a D2H synchronization, which can be inefficient.
        #   # More importantly, this prevents us from capturing cuda graphs.
        #   if fac_info.info != 0:
        #       return False
        #   # For ExecuteCUDA, check the diagonal entries of the factorization to
        #   # detect potential numerical issues. If any diagonal entry is very
        #   # small, it may indicate the matrix is close to singular or
        #   # indefinite, which can lead to very inaccurate results in
        #   # subsequent computations.
        #   # For ExecuteHybrid we cannot do this because fac_info.diag are
        #   # always all zeros.
        #   if isinstance(self._cudss_solver.execution_options, ExecutionCUDA):
        #       # NOTE: the threshold here may need to be tuned based on the problem
        #       if np.any(np.abs(fac_info.diag) < 1e-12):
        #           return False
        #
        # Why it was dropped (measured with cuDSS 0.8):
        # - CUDSS_DATA_INFO and CUDSS_DATA_DIAG are read with cudssDataGet,
        #   which waits for all work queued on the stream, even when DIAG is
        #   written to a device buffer, and fails inside a stream capture. Any
        #   such check rules out CUDA graphs.
        # - Under uniform batching DIAG returns the diagonal of one matrix only
        #   (dim values, not B * dim), so it cannot give a per-problem status.
        # - The reorderings the documentation lists for DIAG (COLAMD,
        #   BTF_COLAMD) are rejected for symmetric matrices; DIAG does work
        #   with DEFAULT, AMD, NESTED_DISSECTION and NONE.
        # - An absolute pivot threshold misfires on these KKT matrices: the
        #   -delta * I block has pivots of size delta by design (down to
        #   reg_finetune_lower_limit = 1e-13), so 1e-12 would reject healthy,
        #   quasi-definite factorizations after regularization finetuning.
        self._execute(cudss.Phase.FACTORIZATION, cuda_stream)

    @nvtx.annotate("CudssSparseDirectSolver::solve")
    def solve(self, cuda_stream: int) -> None:
        """Solve phase, launched on the given stream. It captures into a CUDA graph, but not
        into a conditional graph node: cuDSS issues small host-to-device
        copies from pageable memory inside the solve phase, which conditional
        node bodies reject (verified with cuDSS 0.8 for single, uniform-batch
        and explicit-batch matrices). The factorization phase has no such
        copies."""
        # The solution descriptor points at self._sol: cuDSS writes directly
        # into it, with no allocation and no copy.
        self._execute(cudss.Phase.SOLVE, cuda_stream)

    @staticmethod
    def _find_cudss_mt_lib() -> Optional[str]:
        """Auto-discover the cuDSS multithreading layer library.

        Searches across CUDA version packages (nvidia.cu11, nvidia.cu12, nvidia.cu13, ...)
        since the package name depends on the installed CUDA toolkit version.
        """
        for cuda_version in range(13, 10, -1):  # try 13, 12, 11
            spec = importlib.util.find_spec(f"nvidia.cu{cuda_version}")
            if spec is None:
                continue
            # nvidia.cuXX is a namespace package (no __init__.py), so spec.origin is None.
            # Use submodule_search_locations to find the package directory instead.
            search_paths = spec.submodule_search_locations
            if search_paths:
                for base in search_paths:
                    lib = os.path.join(base, "lib", "libcudss_mtlayer_gomp.so.0")
                    if os.path.isfile(lib):
                        return lib
        return None
