from typing import Optional
import cupy as cp
from cupyx.scipy.sparse import csr_matrix, diags, bmat
from nvmath.sparse.advanced import DirectSolver, DirectSolverAlgType

from ..kkt_solver import KKTSolverBase
from .sparse_data import SparseData

class SparseKKTSolver(KKTSolverBase):
    """
    Sparse KKT solver with LDLT factorization.
    """
    def __init__(self, data: SparseData):
        super().__init__()
        self._delta = cp.nan
        self._x_reg = cp.zeros(data.n)
        self._z_reg = cp.zeros(data.m)

        self._kkt_mat = self._initialize_kkt_csr(data.P, data.A, data.G)

        self._rhs = cp.zeros(self._kkt_mat.shape[0], dtype=cp.float64)
        self._sol = cp.zeros(self._kkt_mat.shape[0], dtype=cp.float64)

        # pre-compute diagonal indices for efficient in-place updates
        self._diag_indices = self._find_diagonal_indices(self._kkt_mat)

        # NOTE: do NOT use a context manager here; it will close the solver at the end
        # of __init__, making subsequent factorize() calls fail.
        self._ldlt_solver = DirectSolver(
            self._kkt_mat,
            self._rhs
        )
        config = self._ldlt_solver.plan_config
        config.reordering_algorithm = DirectSolverAlgType.ALG_1
        self._ldlt_solver.plan()  # precompute reordering and symbolic factorization

    def __del__(self):
        # Best-effort cleanup.
        solver = getattr(self, "_solver", None)
        close = getattr(solver, "close", None)
        if callable(close):
            close()

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
    
    @staticmethod
    def _find_diagonal_indices(mat: csr_matrix) -> cp.ndarray:
        """
        Find the positions of diagonal elements in the CSR data array.
        Returns a 1D array of indices into mat.data where diagonal elements are located.
        """
        n = mat.shape[0]
        diag_idx = cp.empty(n, dtype=cp.int32)
        
        for i in range(n):
            row_start = int(mat.indptr[i])
            row_end = int(mat.indptr[i + 1])
            # find where column == i in this row
            for j in range(row_start, row_end):
                if int(mat.indices[j]) == i:
                    diag_idx[i] = j
                    break
        return diag_idx
    
    def _update_kkt(self, data: SparseData, delta: float, x_reg: cp.ndarray, z_reg: cp.ndarray) -> None:
        # IMPORTANT:
        # - cupyx CSR does not support efficient block assignment like self._kkt_mat[:n, :n] = ...
        # - cp.diag/cp.eye create dense matrices and are very costly
        # - DirectSolver reuses the symbolic factorization only if sparsity pattern is unchanged
        # - setdiag() creates NEW buffers, breaking DirectSolver's reference to the matrix
        #
        # We built the KKT with diagonal placeholders. Now we update ONLY the diagonal IN-PLACE
        # by directly modifying the CSR data array using pre-computed indices.
        n, p, m = data.n, data.p, data.m
        diag_top = data.P.diagonal() + x_reg
        diag_mid = -delta * cp.ones(p, dtype=cp.float64)
        diag_bot = -z_reg
        full_diag = cp.concatenate([diag_top, diag_mid, diag_bot])
        self._kkt_mat.data[self._diag_indices] = full_diag
    

    def update_scalings_and_factor(self, data: SparseData, delta: float, x_reg: cp.ndarray, z_reg: cp.ndarray) -> bool:
        self._delta = delta
        self._x_reg = x_reg
        self._z_reg = z_reg

        self._update_kkt(data, delta, x_reg, z_reg)

        try:
            self._ldlt_solver.factorize()
        except Exception as e:
            print(f"Factorization failed: {e}")
            return False

        return True
    
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
        self._sol[:] = self._ldlt_solver.solve()

        assert self._sol.dtype == cp.float64
        assert cp.allclose(self._kkt_mat @ self._sol, self._rhs)

        delta_x[:] = self._sol[:n]
        delta_y[:] = self._sol[n:n+p]
        delta_z[:] = self._sol[n+p:n+p+m]
    

    def eval_P_x(self, data: SparseData, alpha: float, x: cp.ndarray, z: cp.ndarray):
        z[:] = data.P @ x * alpha
    
    def eval_A_xn_and_AT_xt(self, data: SparseData, alpha_n: float, xn: cp.ndarray, alpha_t: float, xt: cp.ndarray, zn: cp.ndarray, zt: cp.ndarray):
        zn[:] = (data.A @ xn) * alpha_n
        zt[:] = (data.A.T @ xt) * alpha_t
    
    def eval_G_xn_and_GT_xt(self, data: SparseData, alpha_n: float, xn: cp.ndarray, alpha_t: float, xt: cp.ndarray, zn: cp.ndarray, zt: cp.ndarray):
        zn[:] = (data.G @ xn) * alpha_n
        zt[:] = (data.G.T @ xt) * alpha_t