import torch
import nvtx

from ..kkt_solver import KKTSolverBase
from .dense_data import DenseData
from .dense_cholesky import CholeskyInplaceSolver
from .cublas_wrappers import (
    dgemv, dcopy, daxpy, dsyrk, ddgmm, cublas_set_stream,
    cublas_create_handle, cublas_destroy_handle,
    OP_N, FILL_UPPER, SIDE_RIGHT,
)


class DenseKKTSolver(KKTSolverBase):
    """
    Dense KKT solver.

    Eliminates Delta_y and Delta_z to form:
    (P + diag(x_reg) + (1/delta)*A^T*A + G^T*diag(z_reg^{-1})*G) Delta_x = rhs

    Uses direct cuBLAS calls instead of high-level operations for CUDA graph compatibility.
    """
    def __init__(self, data: DenseData):
        super().__init__()

        n, p, m = data.n, data.p, data.m

        # Pre-allocated workspace
        self._delta_inv = torch.empty(1, dtype=torch.float64, device='cuda')
        self._z_reg_inv = torch.empty(m, dtype=torch.float64, device='cuda') if m > 0 else torch.empty(0, dtype=torch.float64, device='cuda')
        self._z_reg_inv_sqrt = torch.empty(m, dtype=torch.float64, device='cuda') if m > 0 else torch.empty(0, dtype=torch.float64, device='cuda')

        self._kkt_mat = torch.empty(n, n, dtype=torch.float64, device='cuda')
        self._AtA = torch.empty(n, n, dtype=torch.float64, device='cuda') if p > 0 else torch.zeros(0, 0, dtype=torch.float64, device='cuda')
        self._G_scaled = torch.zeros_like(data.G) if m > 0 else torch.zeros(0, 0, dtype=torch.float64, device='cuda')

        self._cholesky_solver = CholeskyInplaceSolver(n, dtype=torch.float64)
        self._cublas_handle = cublas_create_handle()

        if p > 0:
            self._compute_AtA(data)

    def __del__(self):
        handle = getattr(self, "_cublas_handle", None)
        if handle is not None:
            try:
                cublas_destroy_handle(handle)
            except Exception:
                pass

    def _compute_AtA(self, data: DenseData):
        """Compute AtA = A^T * A via cuBLAS dsyrk to reduce overhead.

        For C-contiguous A of shape (p, n):
        - cuBLAS sees column-major layout as an nxp matrix (call it A_cm)
        - dsyrk with OP_N computes C = A_cm * A_cm^T = A^T * A  (nxn)
        """
        cublas_set_stream(self._cublas_handle, torch.cuda.current_stream().cuda_stream)
        n, p = data.n, data.p
        dsyrk(self._cublas_handle, FILL_UPPER, OP_N, n, p,
              1.0, data.A.data_ptr(), n,
              0.0, self._AtA.data_ptr(), n)

    def update_data(self, data: DenseData, update_P: bool, update_A: bool, update_G: bool):
        if update_A and data.p > 0:
            self._compute_AtA(data)

    @nvtx.annotate("DenseKKTSolver::update_kkt")
    def update_kkt(self, data: DenseData, delta: torch.Tensor, x_reg: torch.Tensor, z_reg: torch.Tensor) -> None:
        """Assemble KKT matrix using direct cuBLAS calls (CUDA graph safe).

        Set cuBLAS handle to current torch stream to ensure these operations are recorded into the graph.
        """
        cublas_set_stream(self._cublas_handle, torch.cuda.current_stream().cuda_stream)
        n = data.n
        handle = self._cublas_handle

        # Store delta reference for solve; compute 1/delta on device
        self._delta = delta
        torch.reciprocal(delta, out=self._delta_inv)

        # KKT = P  (flat copy of n*n elements)
        dcopy(handle, n * n, data.P.data_ptr(), 1, self._kkt_mat.data_ptr(), 1)

        # KKT_diag += x_reg  (add to diagonal with stride n+1 for C-contiguous)
        daxpy(handle, n, 1.0, x_reg.data_ptr(), 1, self._kkt_mat.data_ptr(), n + 1)

        # KKT += (1/delta) * AtA  (device pointer for delta_inv scalar)
        if data.p > 0:
            daxpy(handle, n * n, self._delta_inv.data_ptr(),
                  self._AtA.data_ptr(), 1, self._kkt_mat.data_ptr(), 1)

        # KKT += G^T * diag(z_reg_inv) * G  =  G_scaled^T * G_scaled
        if data.m > 0:
            torch.reciprocal(z_reg, out=self._z_reg_inv)
            torch.sqrt(self._z_reg_inv, out=self._z_reg_inv_sqrt)

            # G_scaled = diag(z_reg_inv_sqrt) * G  via cublasDdgmm.
            ddgmm(handle, SIDE_RIGHT, n, data.m,
                  data.G.data_ptr(), n, self._z_reg_inv_sqrt.data_ptr(), 1,
                  self._G_scaled.data_ptr(), n)

            # dsyrk: KKT += 1.0 * A * A^T + 1.0 * KKT  (where A = cuBLAS view of G_scaled)
            dsyrk(handle, FILL_UPPER, OP_N, n, data.m,
                  1.0, self._G_scaled.data_ptr(), n,
                  1.0, self._kkt_mat.data_ptr(), n)

    @nvtx.annotate("DenseKKTSolver::factor")
    def factor(self) -> bool:
        return self._cholesky_solver.factorize(self._kkt_mat)

    @nvtx.annotate("DenseKKTSolver::solve")
    def solve(self, data: DenseData, rhs_x: torch.Tensor, rhs_y: torch.Tensor, rhs_z: torch.Tensor,
              delta_x: torch.Tensor, delta_y: torch.Tensor, delta_z: torch.Tensor):
        """Solve the reduced KKT system and recover delta_y, delta_z.

        Set cublas handle to current torch stream to ensure these operations are recorded into the graph.
        """
        cublas_set_stream(self._cublas_handle, torch.cuda.current_stream().cuda_stream)
        handle = self._cublas_handle
        n = data.n

        # delta_x = rhs_x
        dcopy(handle, n, rhs_x.data_ptr(), 1, delta_x.data_ptr(), 1)

        # delta_x += (1/delta) * A^T * rhs_y
        if data.p > 0:
            self.eval_AT_xt(data, 1.0, rhs_y, delta_y)
            daxpy(handle, n, self._delta_inv.data_ptr(),
                  delta_y.data_ptr(), 1, delta_x.data_ptr(), 1)

        # delta_x += G^T * (z_reg_inv * rhs_z)
        if data.m > 0:
            torch.mul(self._z_reg_inv, rhs_z, out=delta_z)
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
    def eval_P_x(self, data: DenseData, alpha: float, x: torch.Tensor, z: torch.Tensor):
        cublas_set_stream(self._cublas_handle, torch.cuda.current_stream().cuda_stream)
        dgemv(self._cublas_handle, data.P, x, z, alpha=alpha, beta=0.0)

    @nvtx.annotate("DenseKKTSolver::eval_A_xn")
    def eval_A_xn(self, data: DenseData, alpha_n: float, xn: torch.Tensor, zn: torch.Tensor):
        cublas_set_stream(self._cublas_handle, torch.cuda.current_stream().cuda_stream)
        dgemv(self._cublas_handle, data.A, xn, zn, alpha=alpha_n, beta=0.0)

    @nvtx.annotate("DenseKKTSolver::eval_AT_xt")
    def eval_AT_xt(self, data: DenseData, alpha_t: float, xt: torch.Tensor, zt: torch.Tensor):
        cublas_set_stream(self._cublas_handle, torch.cuda.current_stream().cuda_stream)
        dgemv(self._cublas_handle, data.A, xt, zt, transa=True, alpha=alpha_t, beta=0.0)

    @nvtx.annotate("DenseKKTSolver::eval_G_xn")
    def eval_G_xn(self, data: DenseData, alpha_n: float, xn: torch.Tensor, zn: torch.Tensor):
        cublas_set_stream(self._cublas_handle, torch.cuda.current_stream().cuda_stream)
        dgemv(self._cublas_handle, data.G, xn, zn, alpha=alpha_n, beta=0.0)

    @nvtx.annotate("DenseKKTSolver::eval_GT_xt")
    def eval_GT_xt(self, data: DenseData, alpha_t: float, xt: torch.Tensor, zt: torch.Tensor):
        cublas_set_stream(self._cublas_handle, torch.cuda.current_stream().cuda_stream)
        dgemv(self._cublas_handle, data.G, xt, zt, transa=True, alpha=alpha_t, beta=0.0)
