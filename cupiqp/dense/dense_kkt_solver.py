import cupy as cp
import nvtx

from ..kkt_solver import KKTSolverBase, KKTUpdateOptions
from .dense_data import DenseData
from .dense_cholesky import CholeskyInplaceSolver
from .cublas_wrappers import (
    dgemv, dcopy, daxpy, dsyrk, ddgmm, set_stream,
    OP_N, FILL_UPPER, SIDE_RIGHT,
)


class DenseKKTSolver(KKTSolverBase):
    """
    Dense KKT solver.

    Eliminates Delta_y and Delta_z to form:
    (P + diag(x_reg) + (1/delta)*A^T*A + G^T*diag(z_reg^{-1})*G) Delta_x = rhs

    Uses direct cuBLAS calls instead of high-level cupy operations for CUDA graph compatibility.
    """
    def __init__(self, data: DenseData):
        super().__init__()

        n, p, m = data.n, data.p, data.m

        # Pre-allocated workspace
        self._delta_inv = cp.empty(1, dtype=cp.float64)
        self._z_reg_inv = cp.empty(m, dtype=cp.float64) if m > 0 else cp.empty(0, dtype=cp.float64)
        self._z_reg_inv_sqrt = cp.empty(m, dtype=cp.float64) if m > 0 else cp.empty(0, dtype=cp.float64)

        self._kkt_mat = cp.empty((n, n), dtype=cp.float64)
        self._AtA = data.A.T @ data.A if p > 0 else cp.zeros((0, 0), dtype=cp.float64)
        self._G_scaled = cp.zeros_like(data.G) if m > 0 else cp.zeros((0, 0), dtype=cp.float64)

        self._cholesky_solver = CholeskyInplaceSolver(n, dtype=cp.float64)

        self._cublas_handle = cp.cuda.Device().cublas_handle

    def _sync_cublas_stream(self):
        """Point the cuBLAS handle at cupy's current stream.

        This ensures cuBLAS operations follow the ``with stream:`` context
        (critical for CUDA graph capture on a non-default stream).
        """
        set_stream(self._cublas_handle, cp.cuda.get_current_stream().ptr)

    def update_data(self, data: DenseData, update_options: KKTUpdateOptions):
        if update_options == KKTUpdateOptions.KKT_UPDATE_A and data.p > 0:
            self._AtA[:, :] = data.A.T @ data.A

    @nvtx.annotate("DenseKKTSolver::update_kkt")
    def update_kkt(self, data: DenseData, delta: cp.ndarray, x_reg: cp.ndarray, z_reg: cp.ndarray) -> None:
        """Assemble KKT matrix using direct cuBLAS calls (CUDA graph safe).
        
        Set cuBLAS handle to current cupy stream to ensure these operations are recorded into the graph when called within a cupy's ``with stream:`` context.
        """
        self._sync_cublas_stream()
        n = data.n
        handle = self._cublas_handle

        # Store delta reference for solve; compute 1/delta on device
        self._delta = delta
        cp.reciprocal(delta, out=self._delta_inv)

        # KKT = P  (flat copy of n*n elements)
        dcopy(handle, n * n, data.P.data.ptr, 1, self._kkt_mat.data.ptr, 1)

        # KKT_diag += x_reg  (add to diagonal with stride n+1 for C-contiguous)
        daxpy(handle, n, 1.0, x_reg.data.ptr, 1, self._kkt_mat.data.ptr, n + 1)

        # KKT += (1/delta) * AtA  (device pointer for delta_inv scalar)
        if data.p > 0:
            daxpy(handle, n * n, self._delta_inv.data.ptr,
                  self._AtA.data.ptr, 1, self._kkt_mat.data.ptr, 1)

        # KKT += G^T * diag(z_reg_inv) * G  =  G_scaled^T * G_scaled
        if data.m > 0:
            cp.reciprocal(z_reg, out=self._z_reg_inv)
            cp.sqrt(self._z_reg_inv, out=self._z_reg_inv_sqrt)

            # G_scaled = diag(z_reg_inv_sqrt) * G  via cublasDdgmm.
            ddgmm(handle, SIDE_RIGHT, n, data.m,
                  data.G.data.ptr, n, self._z_reg_inv_sqrt.data.ptr, 1,
                  self._G_scaled.data.ptr, n)
            
            # dsyrk: KKT += 1.0 * A * A^T + 1.0 * KKT  (where A = cuBLAS view of G_scaled)
            dsyrk(handle, FILL_UPPER, OP_N, n, data.m,
                  1.0, self._G_scaled.data.ptr, n,
                  1.0, self._kkt_mat.data.ptr, n)

    @nvtx.annotate("DenseKKTSolver::factor")
    def factor(self) -> bool:
        return self._cholesky_solver.factorize(self._kkt_mat)

    @nvtx.annotate("DenseKKTSolver::solve")
    def solve(self, data: DenseData, rhs_x: cp.ndarray, rhs_y: cp.ndarray, rhs_z: cp.ndarray,
              delta_x: cp.ndarray, delta_y: cp.ndarray, delta_z: cp.ndarray):
        """Solve the reduced KKT system and recover delta_y, delta_z.
        
        Set cublas handle to current cupy stream to ensure these operations are recorded into the graph when called within a cupy's ``with stream:`` context.
        """
        self._sync_cublas_stream()
        handle = self._cublas_handle
        n = data.n

        # delta_x = rhs_x
        dcopy(handle, n, rhs_x.data.ptr, 1, delta_x.data.ptr, 1)

        # delta_x += (1/delta) * A^T * rhs_y
        if data.p > 0:
            self.eval_AT_xt(data, 1.0, rhs_y, delta_y)
            daxpy(handle, n, self._delta_inv.data.ptr,
                  delta_y.data.ptr, 1, delta_x.data.ptr, 1)

        # delta_x += G^T * (z_reg_inv * rhs_z)
        if data.m > 0:
            cp.multiply(self._z_reg_inv, rhs_z, out=delta_z)
            dgemv(handle, data.G, delta_z, delta_x, transa=True, alpha=1.0, beta=1.0)

        self._cholesky_solver.solve(delta_x)

        # delta_y = (A * delta_x - rhs_y) / delta
        if data.p > 0:
            self.eval_A_xn(data, 1.0, delta_x, delta_y)
            delta_y -= rhs_y
            delta_y /= self._delta

        # delta_z = z_reg_inv * (G * delta_x - rhs_z)
        if data.m > 0:
            self.eval_G_xn(data, 1.0, delta_x, delta_z)
            delta_z -= rhs_z
            delta_z *= self._z_reg_inv

    @nvtx.annotate("DenseKKTSolver::eval_P_x")
    def eval_P_x(self, data: DenseData, alpha: float, x: cp.ndarray, z: cp.ndarray):
        self._sync_cublas_stream()
        dgemv(self._cublas_handle, data.P, x, z, alpha=alpha, beta=0.0)

    @nvtx.annotate("DenseKKTSolver::eval_A_xn")
    def eval_A_xn(self, data: DenseData, alpha_n: float, xn: cp.ndarray, zn: cp.ndarray):
        self._sync_cublas_stream()
        dgemv(self._cublas_handle, data.A, xn, zn, alpha=alpha_n, beta=0.0)

    @nvtx.annotate("DenseKKTSolver::eval_AT_xt")
    def eval_AT_xt(self, data: DenseData, alpha_t: float, xt: cp.ndarray, zt: cp.ndarray):
        self._sync_cublas_stream()
        dgemv(self._cublas_handle, data.A, xt, zt, transa=True, alpha=alpha_t, beta=0.0)

    @nvtx.annotate("DenseKKTSolver::eval_G_xn")
    def eval_G_xn(self, data: DenseData, alpha_n: float, xn: cp.ndarray, zn: cp.ndarray):
        self._sync_cublas_stream()
        dgemv(self._cublas_handle, data.G, xn, zn, alpha=alpha_n, beta=0.0)

    @nvtx.annotate("DenseKKTSolver::eval_GT_xt")
    def eval_GT_xt(self, data: DenseData, alpha_t: float, xt: cp.ndarray, zt: cp.ndarray):
        self._sync_cublas_stream()
        dgemv(self._cublas_handle, data.G, xt, zt, transa=True, alpha=alpha_t, beta=0.0)
