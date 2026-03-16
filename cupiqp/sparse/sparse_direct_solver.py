import os, importlib.util
from abc import ABC, abstractmethod
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


class SparseDirectSolver(ABC):
    def __init__(self, matrix: csr_matrix):
        if not isinstance(matrix, csr_matrix):
            raise ValueError("Input matrix must be a csr_matrix.")
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Input matrix must be square.")
        self._mat = matrix  # NOTE: should make sure memory of this matrix always exists and is not re-allocated, since the direct solver holds a pointer to the matrix memory for in-place factorization and solves. We can update the values of the matrix for each iteration, but should not re-allocate a new matrix.
        self._dim = matrix.shape[0]
        self._rhs = cp.empty(self._dim, dtype=cp.float64)
        self._sol = cp.empty(self._dim, dtype=cp.float64)
    
    @nvtx.annotate("SparseDirectSolver::plan")
    @abstractmethod
    def plan(self) -> bool:
        """Precompute reordering and symbolic factorization"""
        pass

    @nvtx.annotate("SparseDirectSolver::factor")
    @abstractmethod
    def factor(self) -> bool:
        """Numerical factorization of the matrix. Should be called after plan() and before solve()."""
        pass

    @nvtx.annotate("SparseDirectSolver::solve")
    @abstractmethod
    def solve(self):
        """Solve the linear system for the given right-hand side."""
        pass

    @property
    def mat(self):
        """Expose the matrix for in-place updates before calling factor()."""
        return self._mat
    
    @property
    def rhs(self):
        """Expose the right-hand side vector for in-place updates before calling solve()."""
        return self._rhs
    
    @property
    def sol(self):
        """Expose the solution vector after calling solve()."""
        return self._sol


class CudssSparseDirectSolver(SparseDirectSolver):
    def __init__(self, matrix: csr_matrix, **cudss_kwargs):
        super().__init__(matrix)

        # setup cuDSS solver
        opts = DirectSolverOptions(
            sparse_system_type=DirectSolverMatrixType.SYMMETRIC,
            sparse_system_view=DirectSolverMatrixViewType.FULL,
            multithreading_lib=self._find_cudss_mt_lib()
        )
        exe = ExecutionHybrid()  # allow both CPU and GPU execution. Optional: ExecutionCUDA(). NOTE: hybrid mode seems more numerically stable
        
        self._cudss_solver = DirectSolver(
            a=self._mat,
            b=self._rhs,
            options=opts,
            execution=exe,
            stream=cp.cuda.get_current_stream().ptr,
            )
        self._cudss_solver.plan_config.reordering_algorithm = DirectSolverAlgType.ALG_DEFAULT
        self._cudss_solver.solution_config.ir_num_steps = 5  # NOTE: iterative refinement steps, to be tuned
        # cudss has IR_TOL, but not implemented yet according to https://docs.nvidia.com/cuda/cudss/types.html#c.cudssConfigParam_t.CUDSS_CONFIG_IR_TOL

    def __del__(self):
        cudss_solver = getattr(self, "_cudss_solver", None)
        if cudss_solver is not None:
            try:
                cudss_solver.free()
            except Exception:
                pass
            self._cudss_solver = None

    def plan(self) -> bool:
        try:
            plan_info = self._cudss_solver.plan(stream=cp.cuda.get_current_stream().ptr)
            cp.cuda.get_current_stream().synchronize()
        except Exception as e:
            print(f"Planning failed: {e}")
            return False
        
        return True

    def factor(self) -> bool:
        try:
            cp.cuda.Device().synchronize()
            fac_info = self._cudss_solver.factorize(stream=cp.cuda.get_current_stream().ptr)
            cp.cuda.Device().synchronize()

            # NOTE: this causes a D2H synchronization, which can be inefficient. More importantly, this prevents us from capturing cuda graphs.
            if fac_info.info != 0:
                print(f"Factorization failed with info={fac_info.info}")
                return False

        except Exception as e:
            print(f"Factorization failed: {e}")
            return False

        return True
    
    def solve(self):
        cp.cuda.Device().synchronize()  # ensure any previous GPU work is done before solve
        self._sol[:] = self._cudss_solver.solve(stream=cp.cuda.get_current_stream().ptr)
        cp.cuda.Device().synchronize()  # ensure any previous GPU work is done before solve

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
    

        