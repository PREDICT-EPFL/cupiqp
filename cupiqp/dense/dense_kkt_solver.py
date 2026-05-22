import cupy as cp
import nvtx
import warp as wp

from ..kkt_solver import KKTSolverBase
from .dense_data import DenseData
from .dense_cholesky import CholeskyInplaceSolver, BatchedCholeskyInplaceSolver
from .dense_kkt_solver_kernels import (
    create_update_kkt_kernel,
    create_solve_pre_cholesky_kernel,
    create_solve_post_cholesky_kernel,
)
from .cublas_wrappers import (
    array_ptr,
    cublas_set_stream, cublas_create_handle, cublas_destroy_handle,
    dgemv, dsyrk, cublas_set_stream,
    dgemm_strided_batched, dgemv_strided_batched,
)


class DenseKKTSolver(KKTSolverBase):
    """
    Dense KKT solver.

    Eliminates Delta_y and Delta_z to form:
    (P + diag(x_reg) + (1/delta)*A^T*A + G^T*diag(z_reg^{-1})*G) Delta_x = rhs
    """
    def __init__(self, data: DenseData):
        super().__init__()

        n, p, m = data.n, data.p, data.m
        B = data.batch_size
        self._batch_size = B
        self._device = data._device
        self._dtype = data._dtype

        # Pre-allocated workspace — all (B, ...) shapes
        self._delta = wp.empty(B, dtype=self._dtype, device=self._device)
        self._delta_inv = wp.empty(B, dtype=self._dtype, device=self._device)
        self._z_reg_inv = wp.empty((B, m if m > 0 else 0), dtype=self._dtype, device=self._device)
        self._z_reg_inv_sqrt = wp.empty((B, m if m > 0 else 0), dtype=self._dtype, device=self._device)
        self._kkt_mat = wp.empty((B, n, n), dtype=self._dtype, device=self._device)
        self._AtA = (wp.empty((B, n, n), dtype=self._dtype, device=self._device) if p > 0
                     else wp.zeros((B, 0, 0), dtype=self._dtype, device=self._device))
        self._G_scaled = (wp.empty((B, m, n), dtype=self._dtype, device=self._device) if m > 0
                          else wp.zeros((B, 0, 0), dtype=self._dtype, device=self._device))
        self._work_n_AT = wp.empty((B, n if p > 0 else 0), dtype=self._dtype, device=self._device)
        self._work_n_GT = wp.empty((B, n if m > 0 else 0), dtype=self._dtype, device=self._device)

        self._update_kkt_kernel, self._update_kkt_kernel_launch_dim = create_update_kkt_kernel(n, p, m)
        self._solve_pre_cholesky_kernel = create_solve_pre_cholesky_kernel(p, m)
        self._solve_post_cholesky_kernel = create_solve_post_cholesky_kernel(p, m)

        self._cublas_handle = cublas_create_handle()

        if B > 1:
            self._cholesky_solver = BatchedCholeskyInplaceSolver(n, B, cp.float64)
            self._gemv = lambda handle, mat, x, y, transa, alpha, beta: \
                dgemv_strided_batched(handle, mat, x, y, transa=transa, alpha=alpha, beta=beta)
            self._syrk = lambda handle, A, C, alpha, beta: dgemm_strided_batched(
                handle, A, A, C, transa=True, transb=False, alpha=alpha, beta=beta)
        else:
            self._cholesky_solver = CholeskyInplaceSolver(n, cp.float64)
            self._gemv = lambda handle, mat, x, y, transa=False, alpha=1.0, beta=0.0: \
                dgemv(handle, mat[0], x[0], y[0], transa=transa, alpha=alpha, beta=beta)
            # dsyrk exploits symmetry; A is (1, k, n), C is (1, n, n)
            self._syrk = lambda handle, A, C, alpha, beta: dsyrk(
                handle, 1, 0, A.shape[-1], A.shape[-2],  # FILL_UPPER, OP_N, n, k
                alpha, array_ptr(A), A.shape[-1], beta, array_ptr(C), C.shape[-1])

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
        """Compute AtA = A^T * A."""
        cublas_set_stream(self._cublas_handle, cp.cuda.get_current_stream().ptr)
        self._syrk(self._cublas_handle, data.A, self._AtA, 1.0, 0.0)

    def update_data(self, data: DenseData, update_P: bool, update_A: bool, update_G: bool):
        if update_A and data.p > 0:
            self._compute_AtA(data)

    @nvtx.annotate("DenseKKTSolver::update_kkt")
    def update_kkt(self, data: DenseData, delta: cp.ndarray, x_reg: cp.ndarray, z_reg: cp.ndarray) -> None:
        """Assemble KKT matrix using batched cuBLAS calls (CUDA graph safe)."""
        cublas_set_stream(self._cublas_handle, cp.cuda.get_current_stream().ptr)

        USE_WARP_IMPL = True
        if USE_WARP_IMPL:
            # compute kkt = P + diag(x_reg) + 1/delta*AtA, as well as delta_inv, G_scaled, z_reg_inv, z_reg_inv_sqrt
            wp.launch(
                kernel=self._update_kkt_kernel,
                dim=(self._batch_size, self._update_kkt_kernel_launch_dim),
                inputs=[
                    data.P, self._AtA, data.G, delta, x_reg, z_reg,
                    self._delta_inv, self._z_reg_inv, self._z_reg_inv_sqrt,
                    self._kkt_mat, self._G_scaled,
                ],
                device="cuda",
                stream=wp.Stream(cuda_stream=cp.cuda.get_current_stream().ptr),
            )
            if data.m > 0:
                self._syrk(self._cublas_handle, self._G_scaled, self._kkt_mat, 1.0, 1.0)
        
        else:
            # --- cupy fallback ---
            n = data.n

            self._delta[:] = delta
            cp.reciprocal(self._delta, out=self._delta_inv)

            self._kkt_mat[:] = data.P
            # NOTE:
            # Diagonal write via a stride-(n+1) view of the flattened (B, n*n)
            # buffer, equivalent to ``self._kkt_mat[:, idx, idx] += x_reg`` but
            # without ``cupy_prepare_array_indexing``. Fancy indexing here would
            # allocate an internal index-scratch buffer from CuPy's mempool;
            # captured into the _update_reg_and_kkt CUDA graph, that scratch
            # pointer goes stale once the mempool is perturbed by other allocators
            # (JAX/RMM, torch caching), and replay crashes with
            # CUDA_ERROR_ILLEGAL_ADDRESS.
            self._kkt_mat.reshape(self._batch_size, -1)[:, ::n + 1] += x_reg

            if data.p > 0:
                self._kkt_mat += self._delta_inv[:, None] * self._AtA

            if data.m > 0:
                cp.reciprocal(z_reg, out=self._z_reg_inv)
                cp.sqrt(self._z_reg_inv, out=self._z_reg_inv_sqrt)
                cp.multiply(self._z_reg_inv_sqrt[:, :, None], data.G, out=self._G_scaled)
                self._syrk(self._cublas_handle, self._G_scaled, self._kkt_mat, 1.0, 1.0)

    @nvtx.annotate("DenseKKTSolver::factor")
    def factor(self) -> bool:
        # B=1: CholeskyInplaceSolver expects (n, n); B>1: BatchedCholeskyInplaceSolver expects (B, n, n)
        return bool(self._cholesky_solver.factorize(self._kkt_mat[0] if self._batch_size == 1 else self._kkt_mat))

    @nvtx.annotate("DenseKKTSolver::solve")
    def solve(self, data: DenseData, rhs_x: cp.ndarray, rhs_y: cp.ndarray, rhs_z: cp.ndarray,
              delta_x: cp.ndarray, delta_y: cp.ndarray, delta_z: cp.ndarray):
        """Solve the reduced KKT system and recover delta_y, delta_z."""
        cublas_set_stream(self._cublas_handle, cp.cuda.get_current_stream().ptr)
        wp_stream = wp.Stream(cuda_stream=cp.cuda.get_current_stream().ptr)

        n, p, m = data.n, data.p, data.m
        B = self._batch_size

        # work_n_AT = A^T @ rhs_y
        if p > 0:
            self.eval_AT_xt(data, 1.0, rhs_y, self._work_n_AT)
        # work_n_GT = G^T @ delta_z
        if m > 0:
            # delta_z = z_reg_inv * rhs_z   
            cp.multiply(self._z_reg_inv, rhs_z, out=delta_z)
            self.eval_GT_xt(data, 1.0, delta_z, self._work_n_GT)

        wp.launch(
            kernel=self._solve_pre_cholesky_kernel,
            dim=(B, n),
            inputs=[rhs_x, self._delta_inv,
                    self._work_n_AT, self._work_n_GT, delta_x],
            device="cuda", stream=wp_stream,
        )

        # B=1: CholeskyInplaceSolver expects 1D/2D; B>1: BatchedCholeskyInplaceSolver expects (B, n).
        self._cholesky_solver.solve(delta_x[0] if B == 1 else delta_x)

        # Recover delta_y = (A @ delta_x - rhs_y) / delta
        if p > 0:
            self.eval_A_xn(data, 1.0, delta_x, delta_y)
        # Recover delta_z = (G @ delta_x - rhs_z) * z_reg_inv
        if m > 0:
            self.eval_G_xn(data, 1.0, delta_x, delta_z)

        if p > 0 or m > 0:
            wp.launch(
                kernel=self._solve_post_cholesky_kernel,
                dim=(B, p+m),
                inputs=[rhs_y, rhs_z, self._delta_inv, self._z_reg_inv,
                        delta_y, delta_z],
                device="cuda", stream=wp_stream,
            )

    @nvtx.annotate("DenseKKTSolver::eval_P_x")
    def eval_P_x(self, data: DenseData, alpha: float, x: cp.ndarray, z: cp.ndarray):
        cublas_set_stream(self._cublas_handle, cp.cuda.get_current_stream().ptr)
        self._gemv(self._cublas_handle, data.P, x, z, transa=False, alpha=alpha, beta=0.0)

    @nvtx.annotate("DenseKKTSolver::eval_A_xn")
    def eval_A_xn(self, data: DenseData, alpha_n: float, xn: cp.ndarray, zn: cp.ndarray):
        cublas_set_stream(self._cublas_handle, cp.cuda.get_current_stream().ptr)
        self._gemv(self._cublas_handle, data.A, xn, zn, transa=False, alpha=alpha_n, beta=0.0)

    @nvtx.annotate("DenseKKTSolver::eval_AT_xt")
    def eval_AT_xt(self, data: DenseData, alpha_t: float, xt: cp.ndarray, zt: cp.ndarray):
        cublas_set_stream(self._cublas_handle, cp.cuda.get_current_stream().ptr)
        self._gemv(self._cublas_handle, data.A, xt, zt, transa=True, alpha=alpha_t, beta=0.0)

    @nvtx.annotate("DenseKKTSolver::eval_G_xn")
    def eval_G_xn(self, data: DenseData, alpha_n: float, xn: cp.ndarray, zn: cp.ndarray):
        cublas_set_stream(self._cublas_handle, cp.cuda.get_current_stream().ptr)
        self._gemv(self._cublas_handle, data.G, xn, zn, transa=False, alpha=alpha_n, beta=0.0)

    @nvtx.annotate("DenseKKTSolver::eval_GT_xt")
    def eval_GT_xt(self, data: DenseData, alpha_t: float, xt: cp.ndarray, zt: cp.ndarray):
        cublas_set_stream(self._cublas_handle, cp.cuda.get_current_stream().ptr)
        self._gemv(self._cublas_handle, data.G, xt, zt, transa=True, alpha=alpha_t, beta=0.0)
