import os, importlib.util
from abc import ABC, abstractmethod
import numpy as np
import cupy as cp
import warp as wp
import nvtx
from nvmath.sparse.advanced import (
    DirectSolver,
    DirectSolverOptions,
    DirectSolverMatrixType,
    DirectSolverMatrixViewType,
    DirectSolverReorderingAlg,
    ExecutionHybrid,
    ExecutionCUDA
)
from nvmath.bindings import cudss as cudss_bindings

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
    def __init__(self, matrix: UniformBatchedCsrMatrix, use_deterministic_mode: bool = False, **cudss_kwargs):
        super().__init__(matrix)
        batch_size = self._batch_size

        # setup cuDSS solver
        opts = DirectSolverOptions(
            sparse_system_type=DirectSolverMatrixType.SYMMETRIC,
            sparse_system_view=DirectSolverMatrixViewType.LOWER,  # NOTE: only take the lower triangular part of the matrix
            multithreading_lib=self._find_cudss_mt_lib()
        )
        exe = ExecutionCUDA()  # Optional: ExecutionCUDA(). NOTE: hybrid mode seems more numerically stable

        # nvmath builds its descriptors from cupy objects: a csr_matrix view of
        # the first matrix and the first rhs row, both zero-copy views of the
        # Warp buffers. cuDSS uniform batching then walks the packed
        # (B, nnz) / (B, dim) storage behind them.
        self._cudss_solver = DirectSolver(
            a=matrix[0],
            b=cp.asarray(self._rhs)[0],
            options=opts,
            execution=exe,
            stream=wp.get_stream("cuda").cuda_stream
        )
        if batch_size > 1:
            ubatch_size = np.array([batch_size], dtype=np.int32)
            cudss_bindings.config_set(
                self._cudss_solver.config_ptr,
                cudss_bindings.ConfigParam.UBATCH_SIZE,
                ubatch_size.ctypes.data,
                ubatch_size.dtype.itemsize
            )
        self._cudss_solver.plan_config.reordering_algorithm = DirectSolverReorderingAlg.DEFAULT
        self._cudss_solver.plan_config.use_superpanels = 0
        self._cudss_solver.solution_config.ir_num_steps = 0  # NOTE: iterative refinement steps, to be tuned
        # cudss has IR_TOL, but not implemented yet according to https://docs.nvidia.com/cuda/cudss/types.html#c.cudssConfigParam_t.CUDSS_CONFIG_IR_TOL
        # self._cudss_solver.plan_config.pivot_type = cudss_bindings.PivotType.PIVOT_COL
        # self._cudss_solver.plan_config.pivot_threshold = 1.0
        # self._cudss_solver.factorization_config.pivot_eps = 1e-12

        # Enable deterministic mode for bit-wise reproducible results across runs.
        # Uses slower kernels but guarantees identical factorization every time.
        # Not exposed by nvmath's high-level API, so we call config_set directly.
        if use_deterministic_mode:
            _det_flag = np.ones(1, dtype=np.int32)
            cudss_bindings.config_set(
                self._cudss_solver.config_ptr,
                cudss_bindings.ConfigParam.DETERMINISTIC_MODE,
                _det_flag.ctypes.data,
                _det_flag.dtype.itemsize
            )

        # Use raw cuDSS handles for direct execute() calls in solve(),
        # bypassing nvmath's _allocate_batched_result overhead.
        self._cudss_handle = self._cudss_solver.handle
        self._cudss_config = self._cudss_solver.config_ptr
        self._cudss_data   = self._cudss_solver.data_ptr
        self._cudss_a      = self._cudss_solver.a_ptr
        self._cudss_x      = self._cudss_solver.x_ptr
        self._cudss_b      = self._cudss_solver.b_ptr

        # Uniform batching advances within these packed parent buffers; bind
        # them explicitly rather than relying on first-row view pointers.
        cudss_bindings.matrix_set_csr_pointers(
            self._cudss_a,
            self._mat.indptr.ptr,
            0,
            self._mat.indices.ptr,
            self._mat.data.ptr
        )
        cudss_bindings.matrix_set_values(self._cudss_b, self._rhs.ptr)
        cudss_bindings.matrix_set_values(self._cudss_x, self._sol.ptr)

    def __del__(self):
        cudss_solver = getattr(self, "_cudss_solver", None)
        if cudss_solver is not None:
            try:
                cudss_solver.free()
            except Exception:
                pass
            self._cudss_solver = None

    @nvtx.annotate("CudssSparseDirectSolver::plan")
    def plan(self, cuda_stream: int) -> bool:
        try:
            plan_info = self._cudss_solver.plan(stream=cuda_stream)
        except Exception as e:
            print(f"Planning failed: {e}")
            return False

        return True

    @nvtx.annotate("CudssSparseDirectSolver::factor")
    def factor(self, cuda_stream: int) -> None:
        # cuDSS reports its factorization information on the host only, so
        # this synchronizes once and writes the batch-wide outcome to every
        # problem of the device status.
        self._factor_status.fill_(0 if self._factor_ok(cuda_stream) else 1)

    def _factor_ok(self, cuda_stream: int) -> bool:
        try:
            fac_info = self._cudss_solver.factorize(stream=cuda_stream)

            if self._batch_size > 1:
                return fac_info.info == 0

            # NOTE: this causes a D2H synchronization, which can be inefficient. More importantly, this prevents us from capturing cuda graphs.
            if fac_info.info != 0:
                return False

            # For ExecuteCUDA, check the diagonal entries of the factorization to detect potential numerical issues.
            # If any diagonal entry is very small, it may indicate the matrix is close to singular or indefinite,
            # which can lead to very inaccurate results in subsequent computations.
            # For ExecuteHybrid we cannot do this because fac_info.diag are always all zeros.
            if isinstance(self._cudss_solver.execution_options, ExecutionCUDA):
                if np.any(np.abs(fac_info.diag) < 1e-12):  # NOTE: the threshold here may need to be tuned based on the problem
                    return False

        except Exception as e:
            print(f"Factorization failed: {e}")
            return False

        return True

    @nvtx.annotate("CudssSparseDirectSolver::solve")
    def solve(self, cuda_stream: int) -> None:
        # Bypass nvmath's solve() which allocates B fresh arrays every call.
        # x_ptr already points at self._sol (set once in __init__), so cuDSS
        # writes directly into our buffer - zero allocation, zero copy.
        cudss_bindings.set_stream(self._cudss_handle, cuda_stream)
        cudss_bindings.execute(
            self._cudss_handle, cudss_bindings.Phase.SOLVE,
            self._cudss_config, self._cudss_data,
            self._cudss_a, self._cudss_x, self._cudss_b
        )

    @staticmethod
    def _find_cudss_mt_lib():
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
