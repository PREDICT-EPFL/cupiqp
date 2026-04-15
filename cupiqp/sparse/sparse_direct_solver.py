import os, importlib.util
from abc import ABC, abstractmethod
from typing import Sequence, Union, List
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

from .sparse_matvec import SparseMatVecProduct


class SparseDirectSolver(ABC):
    """Abstract base for sparse direct solvers — natively supports batching.

    Accepts a single CSR matrix (B = 1) or a list of B CSR matrices sharing
    the same sparsity pattern.  For B = 1 the original single-matrix attributes
    (``_mat``, ``_rhs``, ``_sol``) are preserved for backward compatibility.
    """

    def __init__(self, matrix: Sequence[csr_matrix]):
        assert isinstance(matrix, Sequence) and len(matrix) > 0
        self._batch_size = len(matrix)
        self._mat = list(matrix)
        for m in self._mat:
            if not isinstance(m, csr_matrix):
                raise ValueError("All matrices must be csr_matrix.")
            if m.shape[0] != m.shape[1]:
                raise ValueError("All matrices must be square.")

        self._dim = self._mat[0].shape[0]

        # NOTE: should make sure memory of each matrix always exists and is not
        # re-allocated, since the direct solver holds a pointer to the matrix
        # memory for in-place factorization and solves. We can update the values
        # of the matrix for each iteration, but should not re-allocate a new matrix.
        # self._rhs = [cp.empty((self._dim, ), dtype=cp.float64) for _ in range(self._batch_size)]
        # self._sol = [cp.empty((self._dim, ), dtype=cp.float64) for _ in range(self._batch_size)]
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
    def __init__(self, matrix: Union[csr_matrix, Sequence[csr_matrix]], use_deterministic_mode: bool = False, **cudss_kwargs):
        super().__init__(matrix)
        batch_size = self._batch_size
        self._rhs_list = [self._rhs[i] for i in range(self._rhs.shape[0])]  # list of views
        self._sol_list = [self._sol[i] for i in range(self._sol.shape[0])]  # list of views

        # setup cuDSS solver
        opts = DirectSolverOptions(
            sparse_system_type=DirectSolverMatrixType.SYMMETRIC,
            sparse_system_view=DirectSolverMatrixViewType.FULL,
            multithreading_lib=self._find_cudss_mt_lib()
        )
        exe = ExecutionCUDA()  # Optional: ExecutionCUDA(). NOTE: hybrid mode seems more numerically stable

        if batch_size == 1:
            # Non-batched nvmath DirectSolver
            self._cudss_solver = DirectSolver(
                a=self._mat[0],
                b=self._rhs[0],
                options=opts,
                execution=exe,
                stream=cp.cuda.get_current_stream().ptr,
            )
        else:
            # Explicit-batching nvmath DirectSolver
            self._cudss_solver = DirectSolver(
                a=self._mat,
                b=self._rhs_list,
                options=opts,
                execution=exe,
                stream=cp.cuda.get_current_stream().ptr,
            )
        self._cudss_solver.plan_config.reordering_algorithm = DirectSolverAlgType.ALG_DEFAULT
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

        # # Iterative refinement helpers (B = 1 only for now)
        # if batch_size == 1:
        #     self._mat_vec_prod = SparseMatVecProduct(self._mat[0], transa=False)
        #     self._res = cp.empty_like(self._rhs)
        #     self._rhs_saved = cp.empty_like(self._rhs)
        #     self._sol_ir = cp.empty_like(self._sol)

    def __del__(self):
        cudss_solver = getattr(self, "_cudss_solver", None)
        if cudss_solver is not None:
            try:
                cudss_solver.free()
            except Exception:
                pass
            self._cudss_solver = None

    def plan(self, cuda_stream: int) -> bool:
        try:
            plan_info = self._cudss_solver.plan(stream=cuda_stream)
            # cp.cuda.get_current_stream().synchronize()
        except Exception as e:
            print(f"Planning failed: {e}")
            return False

        return True

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

    def solve(self,
              cuda_stream: int,
              iterative_refinement: bool = False, 
              ir_abs_tol: float = 1e-12, 
              ir_rel_tol: float = 1e-12,
              ir_max_iter: int = 10, 
              ir_min_improvement_rate: float = 5.0
              ) -> None:       
        if self._batch_size == 1:
            self._sol[0][:] = self._cudss_solver.solve(stream=cuda_stream)
        else:
            # Explicit batching: sol is a list/tuple of B solution arrays
            # TODO: how can we avoid malloc here?
            sol = self._cudss_solver.solve(stream=cuda_stream)
            for b in range(self._batch_size):
                self._sol[b][:] = sol[b]

        if iterative_refinement:
            raise NotImplementedError("Iterative refinement in CudssSparseDirectSolver is not implemented yet.")
            # self.iterative_refinement(cuda_stream, ir_abs_tol, ir_rel_tol, ir_max_iter, ir_min_improvement_rate)

    # @nvtx.annotate("SparseDirectSolver::iterative_refinement")
    # def iterative_refinement(self, cuda_stream: int, abs_tol: float, rel_tol: float, max_iter: int, min_improvement_rate: float) -> None:
    #     VERBOSE = False
    #     USE_RHS_NORM = False  # True: tol = atol + rtol * ||b||,  False: tol = atol + rtol * ||r_0||
    #     # TODO: the iterative refinement here and the one in KKTSystem are actually the same thing. Should find a cleaner way to organize the code to avoid this confusion.

    #     self._rhs_saved[:] = self._rhs

    #     prev_res_norm = float('inf')

    #     if USE_RHS_NORM:
    #         rel_tol = 1e-16
    #         rhs_norm = float(cp.max(cp.abs(self._rhs_saved)))
    #         tol = abs_tol + rel_tol * rhs_norm
    #         if VERBOSE:
    #             print(f"      IR: ||b||={rhs_norm:.4e}, "
    #                   f"tol={tol:.4e} (atol={abs_tol:.1e} + rtol={rel_tol:.1e} * ||b||)")

    #     for itr in range(max_iter):
    #         # residual: res = b - A*x
    #         self._mat_vec_prod(x=self._sol, y=self._res)
    #         cp.subtract(self._rhs_saved, self._res, out=self._res)

    #         res_norm = float(cp.max(cp.abs(self._res)))

    #         if not USE_RHS_NORM and itr == 0:
    #             tol = abs_tol + rel_tol * res_norm
    #             if VERBOSE:
    #                 print(f"      IR: ||r_0||={res_norm:.4e}, "
    #                       f"tol={tol:.4e} (atol={abs_tol:.1e} + rtol={rel_tol:.1e} * ||r_0||)")

    #         improvement = prev_res_norm / max(res_norm, 1e-300)

    #         if VERBOSE:
    #             print(f"      IR iter {itr}: ||r||={res_norm:.4e}  improvement={improvement:.1f}x")

    #         # Check convergence
    #         if res_norm < tol:
    #             if VERBOSE:
    #                 print(f"      IR converged after {itr} iteration(s).")
    #             break

    #         # Check improvement rate — stop if not converging fast enough
    #         if itr > 0 and improvement < min_improvement_rate:
    #             if VERBOSE:
    #                 print(f"      IR stalled after {itr} iteration(s) "
    #                       f"(improvement {improvement:.1f}x < {min_improvement_rate:.1f}x threshold).")
    #             break

    #         prev_res_norm = res_norm

    #         # Solve for correction: A*dx = res
    #         self._rhs[:] = self._res
    #         self._sol_ir[:] = self._cudss_solver.solve(stream=cuda_stream)
    #         # cp.cuda.get_current_stream().synchronize()

    #         self._sol += self._sol_ir

    #     else:
    #         if VERBOSE:
    #             print(f"      IR reached max iterations ({max_iter}) "
    #               f"without convergence. Final ||r||={res_norm:.4e}, tol={tol:.4e}.")

    #     # Restore original RHS
    #     self._rhs[:] = self._rhs_saved

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
