import numpy as np
import warp as wp
import cupy as cp
import cupyx.scipy.sparse as cpsp
import sys
sys.path.append('./')
sys.path.append('../')
import unittest

from cupiqp.multistage.multistage_utils import create_csr_add_btd_kernel, create_block_syrk_kernel, create_weighted_block_syrk_kernel


def build_dense_bidiag(D_np: np.ndarray, E_np: np.ndarray) -> np.ndarray:
    """
    Build the dense (N+1)*m x N*c block lower-bidiagonal matrix A.

    D_np: (N, m, c)   diagonal blocks
    E_np: (N, m, c)   sub-diagonal blocks (includes E_{N-1})
    """
    N, m, c = D_np.shape
    A = np.zeros(((N + 1) * m, N * c), dtype=np.float64)
    for k in range(N):
        A[k * m:(k + 1) * m, k * c:(k + 1) * c] = D_np[k]
        A[(k + 1) * m:(k + 2) * m, k * c:(k + 1) * c] = E_np[k]
    return A


class TestMultistageUtils(unittest.TestCase):
    def setUp(self):
        wp.init()
        self.block_size = 20
        self.num_blocks = 40

        rng = np.random.default_rng(42)
        
        # Create numpy reference data
        self.D_np = rng.standard_normal((self.num_blocks, self.block_size, self.block_size))
        self.L_np = rng.standard_normal((self.num_blocks - 1, self.block_size, self.block_size))

        self._csr_add_to_btd_kernel = create_csr_add_btd_kernel(self.num_blocks, self.block_size, dtype=wp.float64)
        
        # Reconstruction logic: BlockTridiag -> CSR
        D_cp = cp.array(self.D_np)
        L_cp = cp.array(self.L_np)
        
        blocks = [[None for _ in range(self.num_blocks)] for _ in range(self.num_blocks)]
        for i in range(self.num_blocks):
            blocks[i][i] = cpsp.csr_matrix(D_cp[i]) # Diag
            if i < self.num_blocks - 1:
                blocks[i+1][i] = cpsp.csr_matrix(L_cp[i]) # Lower at (i+1, i)
                
        self.A_csr = cpsp.bmat(blocks, format='csr', dtype=cp.float64)


    def test_csr_add_btd(self):
        diag_blocks = wp.from_numpy(np.random.randn(self.num_blocks, self.block_size, self.block_size), dtype=wp.float64, device="cuda")
        offdiag_blocks = wp.from_numpy(np.random.randn(self.num_blocks - 1, self.block_size, self.block_size), dtype=wp.float64, device="cuda")
        alpha = np.random.randn()

        D_benchmark = diag_blocks.numpy() + alpha * self.D_np
        L_benchmark = offdiag_blocks.numpy() + alpha * self.L_np
        wp.launch(
            kernel=self._csr_add_to_btd_kernel,
            dim=self.A_csr.shape[0],
            inputs=[
                alpha,
                self.A_csr.indptr, self.A_csr.indices, self.A_csr.data,
                diag_blocks,
                offdiag_blocks,
            ],
            device="cuda"
        )
        D_reconstructed = diag_blocks.numpy()
        L_reconstructed = offdiag_blocks.numpy()
        
        self.assertTrue(np.allclose(D_reconstructed, D_benchmark, atol=1e-8))
        self.assertTrue(np.allclose(L_reconstructed, L_benchmark, atol=1e-8))

    def test_block_syrk(self):
        N, m, c = 5, 32, 12
        alpha, beta = 2.5, -0.3

        rng = np.random.default_rng(42)
        D_np = rng.standard_normal((N, m, c))
        E_np = rng.standard_normal((N, m, c))
        C_D_init = rng.standard_normal((N, c, c))
        C_E_init = rng.standard_normal((N - 1, c, c))

        A_dense = build_dense_bidiag(D_np, E_np)
        ATA = A_dense.T @ A_dense

        C_D_ref = np.empty_like(C_D_init)
        C_E_ref = np.empty_like(C_E_init)
        for k in range(N):
            C_D_ref[k] = alpha * ATA[k * c:(k + 1) * c, k * c:(k + 1) * c] + beta * C_D_init[k]
        for k in range(N - 1):
            C_E_ref[k] = alpha * ATA[(k + 1) * c:(k + 2) * c, k * c:(k + 1) * c] + beta * C_E_init[k]

        device = "cuda"
        kernel = create_block_syrk_kernel(N, m, c)

        A_D = wp.from_numpy(D_np, dtype=wp.float64, device=device)
        A_E = wp.from_numpy(E_np, dtype=wp.float64, device=device)
        C_D = wp.from_numpy(C_D_init.copy(), dtype=wp.float64, device=device)
        C_E = wp.from_numpy(C_E_init.copy(), dtype=wp.float64, device=device)

        wp.launch(kernel, dim=(N, c, c),
                  inputs=[alpha, A_D, A_E, beta, C_D, C_E], device=device)

        np.testing.assert_allclose(C_D.numpy(), C_D_ref, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(C_E.numpy(), C_E_ref, rtol=1e-10, atol=1e-10)

    def test_weighted_block_syrk(self):
        N, m, c = 5, 32, 12
        alpha, beta = 2.5, -0.3

        rng = np.random.default_rng(42)
        D_np = rng.standard_normal((N, m, c))
        E_np = rng.standard_normal((N, m, c))
        w_np = rng.standard_normal((N + 1) * m)
        C_D_init = rng.standard_normal((N, c, c))
        C_E_init = rng.standard_normal((N - 1, c, c))

        # Reference: C = alpha * A^T diag(w) A + beta * C
        A_dense = build_dense_bidiag(D_np, E_np)
        ATwA = A_dense.T @ np.diag(w_np) @ A_dense

        C_D_ref = np.empty_like(C_D_init)
        C_E_ref = np.empty_like(C_E_init)
        for k in range(N):
            C_D_ref[k] = alpha * ATwA[k * c:(k + 1) * c, k * c:(k + 1) * c] + beta * C_D_init[k]
        for k in range(N - 1):
            C_E_ref[k] = alpha * ATwA[(k + 1) * c:(k + 2) * c, k * c:(k + 1) * c] + beta * C_E_init[k]

        device = "cuda"
        kernel = create_weighted_block_syrk_kernel(N, m, c)

        A_D = wp.from_numpy(D_np, dtype=wp.float64, device=device)
        A_E = wp.from_numpy(E_np, dtype=wp.float64, device=device)
        w = wp.from_numpy(w_np, dtype=wp.float64, device=device)
        C_D = wp.from_numpy(C_D_init.copy(), dtype=wp.float64, device=device)
        C_E = wp.from_numpy(C_E_init.copy(), dtype=wp.float64, device=device)

        wp.launch(kernel, dim=(N, c, c),
                  inputs=[alpha, A_D, A_E, w, beta, C_D, C_E], device=device)

        np.testing.assert_allclose(C_D.numpy(), C_D_ref, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(C_E.numpy(), C_E_ref, rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    unittest.main()