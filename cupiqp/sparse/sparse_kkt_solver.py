from typing import Optional
import os
import importlib.util
import cupy as cp
from cupyx.scipy.sparse import csr_matrix, diags, bmat
from nvmath.sparse.advanced import (
    DirectSolver,
    DirectSolverAlgType,
    DirectSolverOptions,
    DirectSolverMatrixType,
    DirectSolverMatrixViewType,
    ExecutionHybrid,
    ExecutionCUDA,
)
from nvmath.bindings.cudss import PivotType
import nvtx

from ..kkt_solver import KKTSolverBase
from .sparse_data import SparseData
from .sparse_matvec import SparseMatVecProduct


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

_CUDSS_MT_LIB = _find_cudss_mt_lib()

class SparseKKTSolver(KKTSolverBase):
    """
    Sparse KKT solver with LDLT factorization.
    """
    def __init__(self, data: SparseData):
        super().__init__()
        self._kkt_mat = self._initialize_kkt_csr(data.P, data.A, data.G)
        self._rhs = cp.zeros(self._kkt_mat.shape[0], dtype=cp.float64)
        self._sol = cp.zeros(self._kkt_mat.shape[0], dtype=cp.float64)

        # pre-compute diagonal indices for efficient in-place updates
        self._diag_x_indices = cp.empty(data.n, dtype=cp.int32)
        self._diag_y_indices = cp.empty(data.p, dtype=cp.int32)
        self._diag_z_indices = cp.empty(data.m, dtype=cp.int32)
        self._find_diagonal_indices()

        # setup spmv operator for evaluating P, A, G matvecs
        self._spmv_P = SparseMatVecProduct(data.P, transa=False)
        self._spmv_A = SparseMatVecProduct(data.A, transa=False)
        self._spmv_AT = SparseMatVecProduct(data.A, transa=True)
        self._spmv_G = SparseMatVecProduct(data.G, transa=False)
        self._spmv_GT = SparseMatVecProduct(data.G, transa=True)

        # setup cuDSS solver
        # TODO: can change this to LOWER if only update lower part of KKT
        opts = DirectSolverOptions(sparse_system_type=DirectSolverMatrixType.SYMMETRIC,
                                   sparse_system_view=DirectSolverMatrixViewType.FULL,
                                   multithreading_lib=_CUDSS_MT_LIB)
        exe = ExecutionHybrid()  # allow both CPU and GPU execution. Optional: ExecutionCUDA()
        self._ldlt_solver = DirectSolver(a=self._kkt_mat, b=self._rhs, options=opts, execution=exe)
        self._ldlt_solver.plan_config.reordering_algorithm = DirectSolverAlgType.ALG_DEFAULT
        # self._ldlt_solver.plan_config.pivot_type = PivotType.PIVOT_NONE  # ! set to no pivoting, but seems don't work since changing pivot.eps still makes a difference
        # self._ldlt_solver.factorization_config.pivot_eps = 1e-10
        self._ldlt_solver.solution_config.ir_num_steps = 1  # ! iterative refinement steps, to be tuned
        
        with nvtx.annotate("SparseKKTSolver::cudss_plan"):
            plan_info = self._ldlt_solver.plan()  # precompute reordering and symbolic factorization
            cp.cuda.get_current_stream().synchronize()

        self._stream_cp = cp.cuda.get_current_stream()

    def __del__(self):
        ldlt_solver = getattr(self, "_ldlt_solver", None)
        if ldlt_solver is not None:
            try:
                ldlt_solver.free()
            except Exception:
                pass
            self._ldlt_solver = None

    @staticmethod
    def _initialize_kkt_csr(P: csr_matrix, A: Optional[csr_matrix] = None, G: Optional[csr_matrix] = None) -> csr_matrix:
        """
        Initialize the KKT matrix based on the sparsity of P, A, G.

        This builds a CSR matrix with a fixed sparsity pattern suitable for repeated
        numeric refactorizations. We intentionally insert identity diagonals into each
        diagonal block so later updates can use setdiag() without changing structure.
        """
        P = P.tocsr()
        n = P.shape[0]

        p = 0 if A is None else int(A.shape[0])
        m = 0 if G is None else int(G.shape[0])

        # Sparse diagonal placeholders (avoid cp.diag / cp.eye which create dense matrices)
        # Keep a diagonal entry present in each block so setdiag() won't change sparsity.
        P_diag_abs_max = cp.max(cp.abs(P.diagonal()))  # ensure diagonal exists
        
        In = diags(2 * P_diag_abs_max * cp.ones(n, dtype=cp.float64), 0, shape=(n, n), format="csr")  # make sure the diagonal entries of P+In are non-zero
        Ip = diags(cp.ones(p, dtype=cp.float64), 0, shape=(p, p), format="csr") if p else None
        Im = diags(cp.ones(m, dtype=cp.float64), 0, shape=(m, m), format="csr") if m else None
        kkt = bmat([
                [P+In, A.T,  G.T],
                [A,    Ip,   None],
                [G,    None, Im],
            ], format="csr", dtype=cp.float64
            )
        return kkt
    
    def _find_diagonal_indices(self) -> None:
        """
        Find the positions of diagonal elements in the CSR data array.
        Returns a 1D array of indices into mat.data where diagonal elements are located.
        """
        dim = self._kkt_mat.shape[0]
        diag_idx = cp.empty(dim, dtype=cp.int32)
        for i in range(dim):
            row_start = int(self._kkt_mat.indptr[i])
            row_end = int(self._kkt_mat.indptr[i + 1])
            # find where column == i in this row
            for j in range(row_start, row_end):
                if int(self._kkt_mat.indices[j]) == i:
                    diag_idx[i] = j
                    break

        n, p, m = self._diag_x_indices.size, self._diag_y_indices.size, self._diag_z_indices.size
        self._diag_x_indices = diag_idx[:n]
        self._diag_y_indices = diag_idx[n : n+p]
        self._diag_z_indices = diag_idx[n+p : n+p+m]
    
    @nvtx.annotate("SparseKKTSolver::update_kkt")
    def update_kkt(self, data: SparseData, delta: cp.ndarray, x_reg: cp.ndarray, z_reg: cp.ndarray) -> None:
        self._kkt_mat.data[self._diag_x_indices] = data.P.diagonal()
        self._kkt_mat.data[self._diag_x_indices] += x_reg        
        self._kkt_mat.data[self._diag_y_indices] = -delta
        self._kkt_mat.data[self._diag_z_indices] = -z_reg
    
    @nvtx.annotate("SparseKKTSolver::factor")
    def factor(self) -> bool:
        try:
            with nvtx.annotate("SparseKKTSolver::cudss_factorize"):
                fac_info = self._ldlt_solver.factorize()
                # surface async device-side failures here so info is meaningful now
                self._stream_cp.synchronize()

            # # TODO: this causes a D2H synchronization, which can be inefficient.
            # TODO: more importantly, this prevents us from capturing cuda graphs. We give up checking the factorization success for now.
            # if fac_info.info != 0:
            #     return False
            
            # check inertia (causes D2H synchronization, inefficient)
            # if fac_info.inertia[0] != data.n or fac_info.inertia[1] != data.p + data.m:
            #     return False

        except Exception as e:
            print(f"Factorization failed: {e}")
            return False

        return True
    
    @nvtx.annotate("SparseKKTSolver::solve")
    def solve(self, data: SparseData, rhs_x: cp.ndarray, rhs_y: cp.ndarray, rhs_z: cp.ndarray, delta_x: cp.ndarray, delta_y: cp.ndarray, delta_z: cp.ndarray):
        """
        Solve the KKT system using the factorized KKT matrix.
        """
        # ! cp.cuda.runtime.memcpyAsync has lower launch overhead than multiple small cp.copyto() calls
        # self._rhs <= [rhs_x, rhs_y, rhs_z]
        cp.cuda.runtime.memcpyAsync(self._rhs.data.ptr, rhs_x.data.ptr, data.n * 8, 1, self._stream_cp.ptr)
        cp.cuda.runtime.memcpyAsync(self._rhs.data.ptr + data.n * 8, rhs_y.data.ptr, data.p * 8, 1, self._stream_cp.ptr)
        cp.cuda.runtime.memcpyAsync(self._rhs.data.ptr + (data.n+data.p) * 8, rhs_z.data.ptr, data.m * 8, 1, self._stream_cp.ptr)

        # update RHS in-place to reuse factorization results. See here: https://docs.nvidia.com/cuda/nvmath-python/0.6.0/host-apis/sparse/generated/nvmath.sparse.advanced.DirectSolver.html.
        # Also see: https://github.com/NVIDIA/nvmath-python/blob/main/examples/sparse/advanced/direct_solver/example05_reset_operands.py
        with nvtx.annotate("SparseKKTSolver::cudss_solve"):
            self._stream_cp.synchronize()
            self._sol[:] = self._ldlt_solver.solve()
            self._stream_cp.synchronize()

        # [delta_x, delta_y, delta_z] <= self._sol
        cp.cuda.runtime.memcpyAsync(delta_x.data.ptr, self._sol.data.ptr, data.n * 8, 1, self._stream_cp.ptr)
        cp.cuda.runtime.memcpyAsync(delta_y.data.ptr, self._sol.data.ptr + data.n * 8, data.p * 8, 1, self._stream_cp.ptr)
        cp.cuda.runtime.memcpyAsync(delta_z.data.ptr, self._sol.data.ptr + (data.n+data.p) * 8, data.m * 8, 1, self._stream_cp.ptr)

    @nvtx.annotate("SparseKKTSolver::eval_P_x")
    def eval_P_x(self, data: SparseData, alpha: float, x: cp.ndarray, z: cp.ndarray):
        self._spmv_P(x, z, alpha=alpha, beta=0.0)
    
    @nvtx.annotate("SparseKKTSolver::eval_A_xn")
    def eval_A_xn(self, data: SparseData, alpha_n: float, xn: cp.ndarray, zn: cp.ndarray):
        self._spmv_A(xn, zn, alpha=alpha_n, beta=0.0)

    @nvtx.annotate("SparseKKTSolver::eval_AT_xt")
    def eval_AT_xt(self, data: SparseData, alpha_t: float, xt: cp.ndarray, zt: cp.ndarray):
        self._spmv_AT(xt, zt, alpha=alpha_t, beta=0.0)

    @nvtx.annotate("SparseKKTSolver::eval_G_xn")
    def eval_G_xn(self, data: SparseData, alpha_n: float, xn: cp.ndarray, zn: cp.ndarray):
        self._spmv_G(xn, zn, alpha=alpha_n, beta=0.0)

    @nvtx.annotate("SparseKKTSolver::eval_GT_xt")
    def eval_GT_xt(self, data: SparseData, alpha_t: float, xt: cp.ndarray, zt: cp.ndarray):
        self._spmv_GT(xt, zt, alpha=alpha_t, beta=0.0)

