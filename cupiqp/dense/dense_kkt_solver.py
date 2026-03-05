import numpy as np
import cupy as cp
import warp as wp
from cupy_backends.cuda.libs import cublas
from cupy.cublas import gemv
import nvtx

from ..kkt_solver import KKTSolverBase, KKTUpdateOptions
from .dense_data import DenseData
from .dense_cholesky import CholeskyInplaceSolver


def create_update_dense_kkt_P_A_parts_kernel(n: int, p: int):
    """Create kernel specialized for specific problem dimension n"""
    
    @wp.kernel
    def update_kkt_P_A_parts_kernel(
        P: wp.array2d(dtype=wp.float64),
        AtA: wp.array2d(dtype=wp.float64),
        x_reg: wp.array1d(dtype=wp.float64), 
        delta: wp.float64,
        KKT: wp.array2d(dtype=wp.float64)
    ):
        tid = wp.tid()
        n_static = wp.static(n)
        p_static = wp.static(p)
        if tid < n_static * n_static:
            r = tid // n_static
            c = tid % n_static
            val = P[r, c]
            if r == c:
                val += x_reg[r]
            if p_static > 0:
                val += wp.float64(1.0) / delta * AtA[r, c]
            KKT[r, c] = val        
    
    return update_kkt_P_A_parts_kernel


class DenseKKTSolver(KKTSolverBase):
    """
    Dense KKT solver.
    
    It eliminate Delta_y and Delta_z to form the following system:
    P + x_reg + 1/delta*A^T*A + G^T*(z_reg)^-1*G) Delta_x = rhs_x + 1/delta*A^T*rhs_y + G^T*diag((z_reg)^-1)*rhs_z

    x_reg and z_reg are both diagonal, so we only need to store their diagonals.

    Then we can solve for Delta_y and Delta_z accordingly.
    """
    def __init__(self, data: DenseData):
        super().__init__()

        self._delta = cp.nan
        self._x_reg = cp.empty(data.n)
        self._z_reg_inv = cp.empty(data.m)
        self._z_reg_inv_sqrt = cp.empty(data.m) 
        
        self._kkt_mat = cp.empty((data.n, data.n))

        self._AtA = data.A.T @ data.A if data.p > 0 else cp.zeros((0, 0))
        self._G_scaled = cp.zeros_like(data.G) if data.m > 0 else cp.zeros((0, 0))  # store diag(z_reg_inv_sqrt) * G

        self._cholesky_solver = CholeskyInplaceSolver(data.n, dtype=cp.float64)

        self._cublas_handle = cp.cuda.Device().cublas_handle
        
        # cuBLAS default mode is HOST. It reads alpha/beta from CPU RAM.
        self._delta_inv = np.array([0.0], dtype=np.float64)
        self._one_val   = np.array([1.0], dtype=np.float64)
        self._delta_inv_ptr = self._delta_inv.ctypes.data
        self._one_ptr   = self._one_val.ctypes.data


        self._update_kkt_P_A_parts_kernel = create_update_dense_kkt_P_A_parts_kernel(data.n, data.p)
    
        # Pre-compile the kernel
        wp.load_module(device="cuda")

    def update_data(self, data: DenseData, update_options: KKTUpdateOptions):
        if update_options == KKTUpdateOptions.KKT_UPDATE_A and data.p > 0:
            self._AtA[:, :] = data.A.T @ data.A

    @nvtx.annotate("DenseKKTSolver::_update_kkt")
    def _update_kkt(self, data: DenseData, x_reg: cp.ndarray) -> None:
        """
        Efficiently assemble Lower Triangular KKT Matrix.
        """
        n = data.n
        wp.launch(
            self._update_kkt_P_A_parts_kernel,
            dim = n*n,
            inputs = [
                data.P,
                self._AtA,
                x_reg,
                self._delta,
                self._kkt_mat,
            ],
            device = "cuda"
        )

        # # Alternatively, use cuBLAS to do the updates
        # self._kkt_mat[:] = data.P
        # self._kkt_mat.flat[::n + 1] += x_reg
        # # add A term: KKT += 1/delta * AtA
        # if data.p > 0:
        #     self._delta_inv[0] = 1.0 / self._delta
        #     cublas.daxpy(
        #         self._cublas_handle,
        #         self._AtA.size,            
        #         self._delta_inv_ptr,           # points to CPU memory
        #         self._AtA.data.ptr,        
        #         1,                         
        #         self._kkt_mat.data.ptr,    
        #         1                          
        #     )

        # add G term: KKT += G^T * diag(z_reg_inv) * G
        if data.m > 0:
            cp.sqrt(self._z_reg_inv, out=self._z_reg_inv_sqrt)
            cp.multiply(data.G, self._z_reg_inv_sqrt[:, None], out=self._G_scaled)
            cublas.dsyrk(
                self._cublas_handle,
                cublas.CUBLAS_FILL_MODE_UPPER, 
                cublas.CUBLAS_OP_N,            
                n,
                data.m,                        
                self._one_ptr,                 # points to CPU memory
                self._G_scaled.data.ptr,       
                n,                         
                self._one_ptr,                 # points to CPU memory
                self._kkt_mat.data.ptr,        
                n                         
            )

    @nvtx.annotate("DenseKKTSolver::update_scalings_and_factor")
    def update_scalings_and_factor(self, data: DenseData, delta: float, x_reg: cp.ndarray, z_reg: cp.ndarray) -> bool:
        self._delta = delta
        cp.reciprocal(z_reg, out=self._z_reg_inv)
        self._update_kkt(data, x_reg)
        # try:
        #     with cupyx.errstate(linalg='raise'):  # raise exception on factorization failure
        #         self._kkt_mat[:, :] = cp.linalg.cholesky(self._kkt_mat)
        #         cp.cuda.get_current_stream().synchronize()
        #     return True
        # except cp.linalg.LinAlgError:
        #     return False
        return self._cholesky_solver.factorize(self._kkt_mat)
    
    @nvtx.annotate("DenseKKTSolver::solve")
    def solve(self, data: DenseData, rhs_x: cp.ndarray, rhs_y: cp.ndarray, rhs_z: cp.ndarray, delta_x: cp.ndarray, delta_y: cp.ndarray, delta_z: cp.ndarray):
        """
        Solve the KKT system using the factorized KKT matrix.
        """
        # solve KKT * dx = rhs_x + 1/delta*A^T*rhs_y + G^T*diag(z_reg_inv)*rhs_z
        delta_x[:] = rhs_x
        if data.p > 0:
            delta_x += 1/self._delta * data.A.T @ rhs_y
        if data.m > 0:
             delta_x += data.G.T @ (self._z_reg_inv * rhs_z)    
        
        self._cholesky_solver.solve(delta_x)

        # recover delta_y and delta_z
        # dy = 1/delta * (A * dx - r_y)
        delta_y[:] = data.A @ delta_x
        delta_y -= rhs_y
        delta_y /= self._delta
        # dz = (W+delta*I)^-1 * (G * dx - r_z)
        delta_z[:] = data.G @ delta_x
        delta_z -= rhs_z
        delta_z *= self._z_reg_inv

    @nvtx.annotate("DenseKKTSolver::eval_P_x")
    def eval_P_x(self, data: DenseData, alpha: float, x: cp.ndarray, z: cp.ndarray):
        gemv(transa='N', alpha=alpha, a=data.P, x=x, beta=0.0, y=z)
    
    @nvtx.annotate("DenseKKTSolver::eval_G_xn_and_GT_xt")
    def eval_A_xn_and_AT_xt(self, data: DenseData, alpha_n: float, xn: cp.ndarray, alpha_t: float, xt: cp.ndarray, zn: cp.ndarray, zt: cp.ndarray):
        gemv(transa='N', alpha=alpha_n, a=data.A, x=xn, beta=0.0, y=zn)
        gemv(transa='T', alpha=alpha_t, a=data.A, x=xt, beta=0.0, y=zt)
    
    @nvtx.annotate("DenseKKTSolver::eval_G_xn_and_GT_xt")
    def eval_G_xn_and_GT_xt(self, data: DenseData, alpha_n: float, xn: cp.ndarray, alpha_t: float, xt: cp.ndarray, zn: cp.ndarray, zt: cp.ndarray):
        gemv(transa='N', alpha=alpha_n, a=data.G, x=xn, beta=0.0, y=zn)
        gemv(transa='T', alpha=alpha_t, a=data.G, x=xt, beta=0.0, y=zt)

    @nvtx.annotate("DenseKKTSolver::eval_G_xn")
    def eval_G_xn(self, data: DenseData, alpha_n: float, xn: cp.ndarray, zn: cp.ndarray):
        gemv(transa='N', alpha=alpha_n, a=data.G, x=xn, beta=0.0, y=zn)