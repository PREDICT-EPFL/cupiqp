from abc import ABC, abstractmethod
import numpy as np
from scipy.linalg import solve_triangular, cholesky
import qdldl
import scipy.sparse as sp

from .typedef import Vector, Matrix
from .data import Data
from .kkt_fwd import KKTUpdateOptions


class KKTSolverBase(ABC):
    """Represent the system with the following form:
    [P+x_reg     A^T      G^T    ] [Delta_x] = [rhs_x]  
    [A         -delta*I     0    ] [Delta_y] = [rhs_y]
    [G           0     -(z_reg)  ] [Delta_z] = [rhs_z]

    where W = diag(s_i/z_i) for i=1,...,m
    Actually we should note the delta*I as regularization term since it's not necessarily delta*I if there are contributions from the box constriants, which are denoted as reg_x and reg_z.
    """
    def __init__(self, data: Data):
        self._x_reg = np.nan * np.ones(data.n)
        self._z_reg_inv = np.nan * np.ones(data.m)

    @abstractmethod
    def update_scalings_and_factor(self, data: Data, delta: float, x_reg: np.ndarray, z_reg: np.ndarray) -> bool:
        pass

    @abstractmethod
    def solve(self, data: Data, rhs_x, rhs_y, rhs_z):
        pass

    def eval_P_x(self, data: Data, alpha: float, x):
        """
        Evaluate alpha * P * x
        """
        # TODO: in PIQP cpp code the data stores upper triangular part of P only, so we need to adjust accordingly
        return alpha * (data.P @ x)

    def eval_A_xn_and_AT_xt(self, data: Data, alpha_n: float, alpha_t: float, xn, xt):
        """
        Evaluate Ax and A^T xt with scaling factors alpha_n and alpha_t:
        zn = alpha_n * A * xn, zt = alpha_t * A^T * xt
        """
        zn = alpha_n * (data.A @ xn)
        zt = alpha_t * (data.A.T @ xt)
        return zn, zt
    
    def eval_G_xn_and_GT_xt(self, data: Data, alpha_n: float, alpha_t: float, xn, xt):
        """
        Evaluate Gx and G^T xt with scaling factors alpha_n and alpha_t:
        zn = alpha_n * G * xn, zt = alpha_t * G^T * xt
        """
        zn = alpha_n * (data.G @ xn)
        zt = alpha_t * (data.G.T @ xt)
        return zn, zt


class DenseKKTSolver(KKTSolverBase):
    """
    Dense KKT solver.
    
    It eliminate Delta_y and Delta_z to form the following system:
    P + x_reg + 1/delta*A^T*A + G^T*(z_reg)^-1*G) Delta_x = rhs_x + 1/delta*A^T*rhs_y + G^T*diag((z_reg)^-1)*rhs_z

    x_reg and z_reg are both diagonal, so we only need to store their diagonals.

    Then we can solve for Delta_y and Delta_z accordingly.
    """
    def __init__(self, data: Data):
        super().__init__(data)

        self._delta = np.nan
        self._x_reg = np.nan * np.ones(data.n)  # used to store the diag(P) + reg_x, i.e., [... P_ii + reg_x_i ...]
        self._z_reg_inv = np.nan * np.ones(data.m)  # used to store the inverse of diag(W+delta*I), i.e., [... (s_i/z_i + delta)^-1 ...]
        
        self._kkt_mat = np.zeros((data.n, data.n))  # Placeholder for KKT matrix

        self._AtA = data.A.T @ data.A if data.p > 0 else np.zeros((0, 0))

    def update_data(self, data: Data, update_options: KKTUpdateOptions):
        if update_options == KKTUpdateOptions.KKT_UPDATE_A and data.p > 0:
            self._AtA = data.A.T @ data.A

    def _update_kkt(self, data: Data, x_reg: np.ndarray) -> None:
        """
        Compute the KKT matrix:
        KKT = P + x_reg + 1/delta*A^T*A + G^T*(z_reg)^-1*G
        """
        # ! can do these only for lower diagonal part
        self._kkt_mat = data.P + np.diag(x_reg)
        
        if data.p > 0:
            self._kkt_mat += (1.0 / self._delta) * self._AtA

        if data.m > 0:
            # self._kkt_mat += data.G.T @ np.diag(self._z_reg_inv) @ data.G
            self._kkt_mat += data.G.T @ (self._z_reg_inv[:, np.newaxis] * data.G)
    
    def update_scalings_and_factor(self, data: Data, delta: float, x_reg: np.ndarray, z_reg: np.ndarray) -> bool:
        """
        z_reg = [delta+s_i/z_i for i in 1, ..., m]
        """
        self._delta = delta
        self._z_reg_inv = 1.0 / z_reg
        self._update_kkt(data, x_reg)
        try:
            self._kkt_mat = cholesky(self._kkt_mat, lower=True, overwrite_a=False, check_finite=True)  # ? overwrite_a seems not making a difference?
            return True
        except np.linalg.LinAlgError:
            return False
        
    def solve(self, data: Data, rhs_x, rhs_y, rhs_z):
        """
        Solve the KKT system using the factorized KKT matrix.
        """
        # Solve KKT * dx = rhs_x + 1/delta*A^T*rhs_y + G^T*diag(z_reg_inv)*rhs_z
        lhs_x_tmp = np.array(rhs_x).flatten()
        if data.p > 0:
            lhs_x_tmp += (data.A.T @ rhs_y) / self._delta
        # if data.m > 0:
        if data.num_hl > 0 or data.num_hu > 0:
            lhs_x_tmp += data.G.T @ (self._z_reg_inv * rhs_z)
        # Solve L * L^T * dx = effective_rhs
        y = solve_triangular(self._kkt_mat, lhs_x_tmp, lower=True, overwrite_b=False)  # TODO: inplace
        lhs_x = solve_triangular(self._kkt_mat.T, y, lower=False, overwrite_b=False)  # TODO: inplace

        # dy = 1/delta * (A * dx - r_y)
        lhs_y = 1/self._delta * (data.A @ lhs_x - rhs_y) if data.p > 0 else np.array([])

        # dz = (W+delta*I)^-1 * (G * dx - r_z)
        lhs_z = np.diag(self._z_reg_inv) @ (data.G @ lhs_x - rhs_z) if data.m > 0 else np.array([])

        return lhs_x, lhs_y, lhs_z


class SparseKKTSolver(KKTSolverBase):
    """
    Sparse KKT solver.
    """
    def __init__(self, data: Data):
        super().__init__(data)
        self._delta = np.nan
        self._x_reg = np.nan * np.ones(data.n)  # used to store the diag(P) + reg_x, i.e., [... P_ii + reg_x_i ...]
        self._z_reg = np.nan * np.ones(data.m)  # used to store the inverse of diag(W+delta*I), i.e., [... (s_i/z_i + delta)^-1 ...]
        self._kkt_mat = self._initialize_kkt_csr(data.P, data.A, data.G)  # placeholder for KKT matrix
        self._ldlt_solver = qdldl.Solver(self._kkt_mat)

    @staticmethod
    def _initialize_kkt_csr(P: sp.csr_matrix, A: sp.csr_matrix, G: sp.csr_matrix) -> sp.csr_matrix:
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
        P_diag_abs_max = np.max(np.abs(P.diagonal()))  # ensure diagonal exists
        
        In = sp.diags(2 * P_diag_abs_max * np.ones(n, dtype=np.float64), 0, shape=(n, n), format="csr")  # make sure the diagonal entries of P+In are non-zero
        Ip = -sp.diags(np.ones(p, dtype=np.float64), 0, shape=(p, p), format="csr") if p else None
        Im = -sp.diags(np.ones(m, dtype=np.float64), 0, shape=(m, m), format="csr") if m else None
        kkt = sp.bmat([
                [P+In, A.T,  G.T],
                [A,    Ip,   None],
                [G,    None, Im],
            ], format="csr", dtype=np.float64
            )
        return kkt


    def update_scalings_and_factor(self, data: Data, delta: float, x_reg: np.ndarray, z_reg: np.ndarray) -> bool:
        self._delta = delta
        self._x_reg[:] = x_reg
        self._z_reg[:] = z_reg

        n, p, m = data.n, data.p, data.m
        
        # Top-left block: P + diag(x_reg)
        P_reg = data.P + sp.diags(self._x_reg, format='csc')
        
        # Middle diagonal block: -delta * I
        delta_I = -self._delta * sp.eye(p, format='csc') if p > 0 else sp.csc_matrix((0, 0))
        
        # Bottom-right diagonal block: -diag(1/z_reg)
        z_inv_diag = -sp.diags(self._z_reg, format='csc') if m > 0 else sp.csc_matrix((0, 0))
        
        self._kkt_mat = sp.bmat([
                [P_reg, data.A.T, data.G.T],
                [data.A, delta_I, None],
                [data.G, None, z_inv_diag]
            ], format='csc')
        # ldl(self._kkt_mat)  # ! for now just check if it's factorable
        try:
            self._ldlt_solver.update(self._kkt_mat)
            return True
        except Exception as e:
            print(f"Sparse ldlt solver failed to factorize the KKT matrix: {e}")
            return False

    def solve(self, data: Data, rhs_x, rhs_y, rhs_z):
        rhs = np.concatenate([rhs_x, rhs_y, rhs_z])
        lhs = self._ldlt_solver.solve(rhs)
        delta_x = lhs[:len(rhs_x)]
        delta_y = lhs[len(rhs_x):len(rhs_x)+len(rhs_y)]
        delta_z = lhs[len(rhs_x)+len(rhs_y):]
        return delta_x, delta_y, delta_z
