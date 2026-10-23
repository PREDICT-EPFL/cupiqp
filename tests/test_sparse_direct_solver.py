"""The two cuDSS batch layouts (uniform batching and block-diagonal stacking)
factor and solve a batch of symmetric systems the same way."""
import numpy as np
import pytest
import warp as wp
from scipy.sparse import csr_matrix

from cupiqp.sparse.batched_csr import UniformBatchedCsrMatrix
from cupiqp.sparse.sparse_direct_solver import CudssUbatchSolver, CudssStackedSolver


def _system(B, n, seed):
    """``B`` symmetric positive definite matrices on one sparse pattern (full
    symmetric storage) and the dense matrices they represent."""
    rng = np.random.default_rng(seed)
    mask = np.eye(n, dtype=bool)
    mask[0, :] = mask[:, 0] = True
    mask |= np.eye(n, k=1, dtype=bool) | np.eye(n, k=-1, dtype=bool)
    R = rng.standard_normal((B, n, n)) * 0.3
    R = 0.5 * (R + R.transpose(0, 2, 1)) * mask
    idx = np.arange(n)
    R[:, idx, idx] = np.abs(R).sum(axis=2) + 0.5 + rng.random((B, n))
    pattern = csr_matrix(mask.astype(float))
    pattern.sort_indices()
    rows = np.repeat(np.arange(n), np.diff(pattern.indptr))
    return pattern, R[:, rows, pattern.indices], R


@pytest.mark.parametrize("solver_cls", [CudssUbatchSolver, CudssStackedSolver])
@pytest.mark.parametrize("B", [1, 5])
def test_factor_solve_and_refactor(solver_cls, B):
    n = 9
    pattern, values, dense = _system(B, n, seed=0)
    mat = UniformBatchedCsrMatrix(
        batch_size=B, indptr=pattern.indptr, indices=pattern.indices,
        data=wp.array(values, dtype=wp.float64, device="cuda"), shape=(n, n),
        dtype=wp.float64, device="cuda")
    solver = solver_cls(mat)
    stream = wp.get_stream("cuda").cuda_stream
    assert solver.plan(cuda_stream=stream)

    rng = np.random.default_rng(1)
    for scale in (1.0, 2.0):  # second round: values updated in place
        if scale != 1.0:
            wp.copy(mat.data, wp.array(values * scale, dtype=wp.float64, device="cuda"))
        rhs = rng.standard_normal((B, n))
        wp.copy(solver.rhs, wp.array(rhs, dtype=wp.float64, device="cuda"))
        solver.factor(cuda_stream=stream)
        solver.solve(cuda_stream=stream)
        expected = np.linalg.solve(scale * dense, rhs[..., None])[..., 0]
        np.testing.assert_allclose(solver.sol.numpy(), expected, rtol=1e-8, atol=1e-10)
