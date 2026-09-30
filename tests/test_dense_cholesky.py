"""Tests for CholeskyInplaceSolver in both single and batched modes.

Every numerical test is parameterized over dtype so the float64 and
float32 code paths (cuSOLVER ``d*potrf/potrs`` vs ``s*potrf/potrs``)
get identical shape, batch, and contiguity coverage.
"""
import pytest
import numpy as np
import cupy as cp
import warp as wp

from cupiqp.dense.dense_cholesky import CholeskyInplaceSolver, BatchedCholeskyInplaceSolver


# ---------------------------------------------------------------------------
# dtype dispatch and tolerances
# ---------------------------------------------------------------------------
DTYPES = [wp.float64, wp.float32]


def _np(dtype):
    return np.float64 if dtype is wp.float64 else np.float32


def _w(a):
    """Zero-copy Warp view: the Cholesky classes take Warp arrays only."""
    return wp.array(a, copy=False)


def _atol(dtype):
    return 1e-10 if dtype == np.float64 else 1e-4


def _make_spd(n, seed=42, dtype=np.float64):
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n))
    return (M @ M.T + n * np.eye(n)).astype(dtype)


def _make_spd_batch(batch_size, n, seed=42, dtype=np.float64):
    rng = np.random.default_rng(seed)
    As = []
    for _ in range(batch_size):
        M = rng.standard_normal((n, n))
        As.append(M @ M.T + n * np.eye(n))
    return np.stack(As).astype(dtype)


# ======================================================================
# Single mode
# ======================================================================
class TestSingleMode:

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_factorize_succeeds(self, dtype):
        n = 6
        A = cp.asarray(_make_spd(n, dtype=_np(dtype)))
        solver = CholeskyInplaceSolver(n, dtype=dtype)
        assert solver.factorize(_w(A))

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("n", [1, 2, 4, 8, 16, 32, 64, 128, 256])
    def test_factorize_and_solve(self, n, dtype):
        rng = np.random.default_rng(n)
        A_orig = _make_spd(n, seed=n, dtype=_np(dtype))
        b = rng.standard_normal(n).astype(_np(dtype))
        A_work = cp.asarray(A_orig.copy())
        x = cp.asarray(b.copy())

        solver = CholeskyInplaceSolver(n, dtype=dtype)
        assert solver.factorize(_w(A_work))
        solver.solve(_w(x))
        np.testing.assert_allclose(
            A_orig @ cp.asnumpy(x), b, atol=_atol(_np(dtype)), err_msg=f"n={n}",
        )

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_multiple_rhs(self, dtype):
        n = 5
        rng = np.random.default_rng(42)
        A_orig = _make_spd(n, dtype=_np(dtype))
        A_work = cp.asarray(A_orig.copy())

        solver = CholeskyInplaceSolver(n, dtype=dtype)
        solver.factorize(_w(A_work))

        for _ in range(3):
            b = rng.standard_normal(n).astype(_np(dtype))
            x = cp.asarray(b.copy())
            solver.solve(_w(x))
            np.testing.assert_allclose(A_orig @ cp.asnumpy(x), b, atol=_atol(_np(dtype)))


# ======================================================================
# Batched mode
# ======================================================================
class TestBatchedMode:

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_factorize_succeeds(self, dtype):
        B, n = 4, 6
        A = cp.asarray(_make_spd_batch(B, n, dtype=_np(dtype)))
        solver = BatchedCholeskyInplaceSolver(n, B, dtype=dtype)
        assert solver.factorize(_w(A))

    # (B, n) matrix exercises: batch=1 (smallest batched case), small
    # batches at various n, and a large-batch case. Fused from the former
    # test_factorize_and_solve / test_batch_size_one / test_various_sizes
    # / test_large_batch — they were all the same "build batched SPD,
    # factor, solve, check per batch" with different (B, n).
    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("B,n", [
        (1, 4),
        (5, 8),
        (8, 1), (8, 2), (8, 4), (8, 8), (8, 12), (8, 16), (8, 24), (8, 32),
        (128, 6),
    ])
    def test_factorize_and_solve(self, B, n, dtype):
        rng = np.random.default_rng(B * 100 + n)
        A_orig = _make_spd_batch(B, n, seed=B * 100 + n, dtype=_np(dtype))
        b = rng.standard_normal((B, n)).astype(_np(dtype))
        A_work = cp.asarray(A_orig.copy())
        x = cp.asarray(b.copy())

        solver = BatchedCholeskyInplaceSolver(n, B, dtype=dtype)
        assert solver.factorize(_w(A_work))
        solver.solve(_w(x))

        for i in range(B):
            np.testing.assert_allclose(
                A_orig[i] @ cp.asnumpy(x[i]), b[i], atol=_atol(_np(dtype)),
                err_msg=f"B={B}, n={n}, batch={i}")

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_multiple_solves_same_factorization(self, dtype):
        B, n = 3, 5
        rng = np.random.default_rng(42)
        A_orig = _make_spd_batch(B, n, dtype=_np(dtype))
        A_work = cp.asarray(A_orig.copy())

        solver = BatchedCholeskyInplaceSolver(n, B, dtype=dtype)
        solver.factorize(_w(A_work))

        for _ in range(3):
            b = rng.standard_normal((B, n)).astype(_np(dtype))
            x = cp.asarray(b.copy())
            solver.solve(_w(x))
            for i in range(B):
                np.testing.assert_allclose(
                    A_orig[i] @ cp.asnumpy(x[i]), b[i], atol=_atol(_np(dtype)))

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_non_contiguous_rhs(self, dtype):
        """Batched solve with non-contiguous RHS (view of larger buffer)."""
        B, n = 3, 4
        rng = np.random.default_rng(42)
        A_orig = _make_spd_batch(B, n, dtype=_np(dtype))
        b = rng.standard_normal((B, n)).astype(_np(dtype))
        A_work = cp.asarray(A_orig.copy())

        buf = cp.zeros((B, n + 5), dtype=_np(dtype))
        buf[:, :n] = cp.asarray(b)
        x_view = buf[:, :n]  # non-contiguous outer stride
        assert not x_view.flags['C_CONTIGUOUS']

        solver = BatchedCholeskyInplaceSolver(n, B, dtype=dtype)
        solver.factorize(_w(A_work))
        solver.solve(_w(x_view))

        for i in range(B):
            np.testing.assert_allclose(
                A_orig[i] @ cp.asnumpy(x_view[i]), b[i], atol=_atol(_np(dtype)),
                err_msg=f"batch {i}")


# ======================================================================
# Cross-mode: batched B=1 vs single should give same result
# ======================================================================
class TestCrossMode:

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_single_vs_batched_b1(self, dtype):
        n = 6
        rng = np.random.default_rng(42)
        A_np = _make_spd(n, dtype=_np(dtype))
        b_np = rng.standard_normal(n).astype(_np(dtype))

        # Single mode
        A1 = cp.asarray(A_np.copy())
        x1 = cp.asarray(b_np.copy())
        s1 = CholeskyInplaceSolver(n, dtype=dtype)
        s1.factorize(_w(A1))
        s1.solve(_w(x1))

        # Batched mode B=1
        A2 = cp.asarray(A_np.copy().reshape(1, n, n))
        x2 = cp.asarray(b_np.copy().reshape(1, n))
        s2 = BatchedCholeskyInplaceSolver(n, 1, dtype=dtype)
        s2.factorize(_w(A2))
        s2.solve(_w(x2))

        np.testing.assert_allclose(cp.asnumpy(x1), cp.asnumpy(x2[0]), atol=_atol(_np(dtype)))
