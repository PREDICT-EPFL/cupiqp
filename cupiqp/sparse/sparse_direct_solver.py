import os, importlib.util
from abc import ABC, abstractmethod
from typing import Union
import numpy as np
import cupy as cp
from cupyx.scipy.sparse import csr_matrix
import nvtx
from nvmath.sparse.advanced import (
    DirectSolver,
    DirectSolverAlgType,
    DirectSolverOptions,
    DirectSolverMatrixType,
    DirectSolverMatrixViewType,
    ExecutionHybrid,
    ExecutionCUDA,
)
from nvmath.bindings import cudss as cudss_bindings

from .batched_csr import BatchedCsrMatrix


class SparseDirectSolver(ABC):
    """Abstract base for sparse direct solvers — natively supports batching.

    ``matrix`` is either

    * a ``cupyx.scipy.sparse.csr_matrix`` (B = 1) — dispatched to nvmath's
      non-batched ``DirectSolver``; or
    * a ``BatchedCsrMatrix`` (B ≥ 1) — dispatched to nvmath's explicit-
      batching ``DirectSolver`` by materializing per-batch ``csr_matrix``
      views on top of the batched values buffer.
    """

    def __init__(self, matrix: Union[csr_matrix, BatchedCsrMatrix]):
        if isinstance(matrix, csr_matrix):
            if matrix.shape[0] != matrix.shape[1]:
                raise ValueError("Matrix must be square.")
            self._batch_size = 1
            self._dim = matrix.shape[0]
            self._mat = matrix
            self._mat_list = None
        elif isinstance(matrix, BatchedCsrMatrix):
            if matrix.rows != matrix.cols:
                raise ValueError("All matrices must be square.")
            self._batch_size = matrix.batch_size
            self._dim = matrix.rows
            self._mat = matrix
            # Per-batch csr_matrix views required by nvmath's batched API.
            # Each view shares indices/indptr with the BatchedCsrMatrix and
            # its .data points into the shared (B, nnz) values buffer, so
            # in-place updates via matrix.data[:, k] = ... are visible to
            # the solver automatically.
            self._mat_list = [matrix[b] for b in range(self._batch_size)]
        else:
            raise TypeError(
                "matrix must be a csr_matrix or BatchedCsrMatrix; got "
                f"{type(matrix).__name__}."
            )

        # NOTE: the direct solver holds pointers into the matrix buffers
        # for in-place factorization and solves. Callers may update the
        # values in place between solves but must not reallocate the
        # underlying buffer (e.g. swap BatchedCsrMatrix.data for a new array).
        self._rhs = cp.empty((self._batch_size, self._dim), dtype=cp.float64)
        self._sol = cp.empty((self._batch_size, self._dim), dtype=cp.float64)

    @nvtx.annotate("SparseDirectSolver::plan")
    @abstractmethod
    def plan(self, cuda_stream: int) -> bool:
        """Precompute reordering and symbolic factorization"""
        pass

    @nvtx.annotate("SparseDirectSolver::factor")
    @abstractmethod
    def factor(self, cuda_stream: int) -> bool:
        """Numerical factorization of the matrix. Should be called after plan() and before solve()."""
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
    def rhs(self) -> cp.ndarray:
        """Expose the (first) right-hand side vector for in-place updates before calling solve()."""
        return self._rhs

    @property
    def sol(self) -> cp.ndarray:
        """Expose the (first) solution vector after calling solve()."""
        return self._sol
    

class CudssSparseDirectSolver(SparseDirectSolver):
    def __init__(self, matrix: Union[csr_matrix, BatchedCsrMatrix], use_deterministic_mode: bool = False, **cudss_kwargs):
        super().__init__(matrix)
        batch_size = self._batch_size
        self._rhs_list = [self._rhs[i] for i in range(self._rhs.shape[0])]  # list of views
        self._sol_list = [self._sol[i] for i in range(self._sol.shape[0])]  # list of views

        # setup cuDSS solver
        opts = DirectSolverOptions(
            sparse_system_type=DirectSolverMatrixType.SYMMETRIC,
            sparse_system_view=DirectSolverMatrixViewType.LOWER,  # NOTE: only take the lower triangular part of the matrix
            multithreading_lib=self._find_cudss_mt_lib()
        )

        # NOTE: ExecutionHybrid() seems to be more efficient in some cases I tested. It triggers the cudss::factorize_v3_ker
        # while ExecutionCUDA() triggers cudss::superpanel_update_ker, which takes much long time
        # However, use_superpanels on/off also effects this. We explicitly disable superpanels here since
        # we find that with it disabled it's more efficient for large problems
        # However, with ExecuteHybrid(), fac_info.diag are always all zeros so we cannot check the quality of factorization.

        if batch_size == 1:
            exe = ExecutionCUDA()  # NOTE: hybrid mode seems more efficient on some big problems
            # Non-batched nvmath DirectSolver. ``self._mat`` is the raw
            # csr_matrix when the user passed one directly, or None +
            # self._mat_list[0] when a BatchedCsrMatrix with B=1 was passed.
            a_single = self._mat if self._mat_list is None else self._mat_list[0]
            self._cudss_solver = DirectSolver(
                a=a_single,
                b=self._rhs[0],
                options=opts,
                execution=exe,
                stream=cp.cuda.get_current_stream().ptr,
            )
        else:
            exe = ExecutionCUDA()
            # Explicit-batching nvmath DirectSolver. ``self._mat_list`` is a
            # list of csr_matrix views on top of the BatchedCsrMatrix.
            self._cudss_solver = DirectSolver(
                a=self._mat_list,
                b=self._rhs_list,
                options=opts,
                execution=exe,
                stream=cp.cuda.get_current_stream().ptr,
            )
        self._cudss_solver.plan_config.reordering_algorithm = DirectSolverAlgType.ALG_DEFAULT
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
                _det_flag.dtype.itemsize,
            )

        # Use raw cuDSS handles for direct execute() calls in solve(),
        # bypassing nvmath's _allocate_batched_result overhead.
        self._cudss_handle = self._cudss_solver.handle
        self._cudss_config = self._cudss_solver.config_ptr
        self._cudss_data   = self._cudss_solver.data_ptr
        self._cudss_a      = self._cudss_solver.a_ptr
        self._cudss_x      = self._cudss_solver.x_ptr
        self._cudss_b      = self._cudss_solver.b_ptr

        # Point x-descriptor at our pre-allocated _sol buffer (done once).
        if batch_size == 1:
            cudss_bindings.matrix_set_values(self._cudss_x, self._sol.data.ptr)
        else:
            row_bytes = self._dim * 8
            ptrs = np.array([self._sol.data.ptr + i * row_bytes
                             for i in range(batch_size)], dtype=np.uint64)
            self._sol_ptrs_dev = cp.array(ptrs)  # prevent GC — descriptor holds this pointer
            cudss_bindings.matrix_set_batch_values(
                self._cudss_x, self._sol_ptrs_dev.data.ptr)

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
            # cp.cuda.get_current_stream().synchronize()
        except Exception as e:
            print(f"Planning failed: {e}")
            return False

        return True

    @nvtx.annotate("CudssSparseDirectSolver::factor")
    def factor(self, cuda_stream: int) -> bool:
        try:
            fac_info = self._cudss_solver.factorize(stream=cuda_stream)
            # cp.cuda.get_current_stream().synchronize()

            if self._batch_size > 1:
                # explicit batching returns a tuple of FactorizationInfo
                if isinstance(fac_info, tuple):
                    return all(fi.info == 0 for fi in fac_info)
                return fac_info.info == 0

            # NOTE: this causes a D2H synchronization, which can be inefficient. More importantly, this prevents us from capturing cuda graphs.
            if fac_info.info != 0:
                # print(f"Factorization failed with info={fac_info.info}")
                return False

            # For ExecuteCUDA, check the diagonal entries of the factorization to detect potential numerical issues. 
            # If any diagonal entry is very small, it may indicate the matrix is close to singular or indefinite, 
            # which can lead to very inaccurate results in subsequent computations. 
            # For ExecuteHybrid we cannot do this because fac_info.diag are always all zeros.
            if isinstance(self._cudss_solver.execution_options, ExecutionCUDA):
                if np.any(np.abs(fac_info.diag) < 1e-12):  # NOTE: the threshold here may need to be tuned based on the problem
                    # print(f"\033[94mFactorization warning: small diagonal entries detected (min diag={fac_info.diag.min():.2e}). Matrix may be close to singular.\033[0m")
                    # print(f"\033[94mFactorization info: {fac_info.info}, inertia: {fac_info.inertia}, "f"min/max diag: {np.abs(fac_info.diag).min():.2e}/{np.abs(fac_info.diag).max():.2e}\033[0m")
                    return False

        except Exception as e:
            print(f"Factorization failed: {e}")
            return False

        return True

    @nvtx.annotate("CudssSparseDirectSolver::solve")
    def solve(self,
              cuda_stream: int,
              iterative_refinement: bool = False,
              ir_abs_tol: float = 1e-12,
              ir_rel_tol: float = 1e-12,
              ir_max_iter: int = 10,
              ir_min_improvement_rate: float = 5.0
              ) -> None:
        # Bypass nvmath's solve() which allocates B fresh arrays every call.
        # x_ptr already points at self._sol (set once in __init__), so cuDSS
        # writes directly into our buffer — zero allocation, zero copy.
        cudss_bindings.set_stream(self._cudss_handle, cuda_stream)
        cudss_bindings.execute(
            self._cudss_handle, cudss_bindings.Phase.SOLVE,
            self._cudss_config, self._cudss_data,
            self._cudss_a, self._cudss_x, self._cudss_b,
        )

        if iterative_refinement:
            raise NotImplementedError("Iterative refinement in CudssSparseDirectSolver is not implemented yet.")

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
