from typing import Optional
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

class SparseKKTSolver(KKTSolverBase):
    """
    Sparse KKT solver with LDLT factorization.
    """
    def __init__(self, data: SparseData):
        super().__init__()
        # self._delta = cp.nan
        # self._x_reg = cp.zeros(data.n)
        # self._z_reg = cp.zeros(data.m)

        self._kkt_mat = self._initialize_kkt_csr(data.P, data.A, data.G)

        self._rhs = cp.zeros(self._kkt_mat.shape[0], dtype=cp.float64)
        self._sol = cp.zeros(self._kkt_mat.shape[0], dtype=cp.float64)

        # pre-compute diagonal indices for efficient in-place updates
        self._diag_x_indices = cp.empty(data.n, dtype=cp.int32)
        self._diag_y_indices = cp.empty(data.p, dtype=cp.int32)
        self._diag_z_indices = cp.empty(data.m, dtype=cp.int32)
        self._find_diagonal_indices()

        # setup cuDSS solver
        # TODO: can change this to LOWER if only update lower part of KKT
        opts = DirectSolverOptions(sparse_system_type=DirectSolverMatrixType.SYMMETRIC, 
                                   sparse_system_view=DirectSolverMatrixViewType.FULL)
        exe = ExecutionHybrid()  # allow both CPU and GPU execution. Optional: ExecutionCUDA()
        self._ldlt_solver = DirectSolver(a=self._kkt_mat, b=self._rhs, options=opts, execution=exe)
        self._ldlt_solver.plan_config.reordering_algorithm = DirectSolverAlgType.ALG_DEFAULT
        # self._ldlt_solver.plan_config.pivot_type = PivotType.PIVOT_NONE  # ! set to no pivoting, but seems don't work since changing pivot.eps still makes a difference
        # self._ldlt_solver.factorization_config.pivot_eps = 1e-10
        self._ldlt_solver.solution_config.ir_num_steps = 10  # ! iterative refinement steps, to be tuned
        
        with nvtx.annotate("SparseKKTSolver::cudss_plan"):
            plan_info = self._ldlt_solver.plan()  # precompute reordering and symbolic factorization
            cp.cuda.get_current_stream().synchronize()

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
    
    @nvtx.annotate("SparseKKTSolver::_update_kkt")
    def _update_kkt(self, data: SparseData, delta: float, x_reg: cp.ndarray, z_reg: cp.ndarray) -> None:
        self._kkt_mat.data[self._diag_x_indices] = data.P.diagonal()
        self._kkt_mat.data[self._diag_x_indices] += x_reg        
        self._kkt_mat.data[self._diag_y_indices] = -delta
        self._kkt_mat.data[self._diag_z_indices] = -z_reg
    
    @nvtx.annotate("SparseKKTSolver::update_scalings_and_factor")
    def update_scalings_and_factor(self, data: SparseData, delta: float, x_reg: cp.ndarray, z_reg: cp.ndarray) -> bool:
        # self._delta = delta
        # self._x_reg[:] = x_reg
        # self._z_reg[:] = z_reg

        self._update_kkt(data, delta, x_reg, z_reg)

        try:
            with nvtx.annotate("SparseKKTSolver::cudss_factorize"):
                fac_info = self._ldlt_solver.factorize()
                # surface async device-side failures here so info is meaningful now
                cp.cuda.get_current_stream().synchronize()

            if fac_info.info != 0:
                return False
            
            # check inertia
            if fac_info.inertia[0] != data.n or fac_info.inertia[1] != data.p + data.m:
                return False

        except Exception as e:
            print(f"Factorization failed: {e}")
            return False

        return True
    
    @nvtx.annotate("SparseKKTSolver::solve")
    def solve(self, data: SparseData, rhs_x: cp.ndarray, rhs_y: cp.ndarray, rhs_z: cp.ndarray, delta_x: cp.ndarray, delta_y: cp.ndarray, delta_z: cp.ndarray):
        """
        Solve the KKT system using the factorized KKT matrix.
        """
        n, p, m = data.n, data.p, data.m
        self._rhs[:n] = rhs_x
        self._rhs[n:n+p] = rhs_y
        self._rhs[n+p:n+p+m] = rhs_z

        # update RHS in-place to reuse factorization results. See here: https://docs.nvidia.com/cuda/nvmath-python/0.6.0/host-apis/sparse/generated/nvmath.sparse.advanced.DirectSolver.html.
        # Also see: https://github.com/NVIDIA/nvmath-python/blob/main/examples/sparse/advanced/direct_solver/example05_reset_operands.py
        with nvtx.annotate("SparseKKTSolver::cudss_solve"):
            cp.cuda.get_current_stream().synchronize()
            self._sol[:] = self._ldlt_solver.solve()
            cp.cuda.get_current_stream().synchronize()

        cp.copyto(delta_x, self._sol[:n])
        cp.copyto(delta_y, self._sol[n:n+p])
        cp.copyto(delta_z, self._sol[n+p:n+p+m])
    

    @nvtx.annotate("SparseKKTSolver::eval_P_x")
    def eval_P_x(self, data: SparseData, alpha: float, x: cp.ndarray, z: cp.ndarray):
        z[:] = data.P @ x * alpha
    
    @nvtx.annotate("SparseKKTSolver::eval_A_xn_and_AT_xt")
    def eval_A_xn_and_AT_xt(self, data: SparseData, alpha_n: float, xn: cp.ndarray, alpha_t: float, xt: cp.ndarray, zn: cp.ndarray, zt: cp.ndarray):
        zn[:] = (data.A @ xn) * alpha_n
        zt[:] = (data.A.T @ xt) * alpha_t
    
    @nvtx.annotate("SparseKKTSolver::eval_G_xn_and_GT_xt")
    def eval_G_xn_and_GT_xt(self, data: SparseData, alpha_n: float, xn: cp.ndarray, alpha_t: float, xt: cp.ndarray, zn: cp.ndarray, zt: cp.ndarray):
        zn[:] = (data.G @ xn) * alpha_n
        zt[:] = (data.G.T @ xt) * alpha_t