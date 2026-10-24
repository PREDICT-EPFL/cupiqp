"""Tests for CholeskyInplaceSolver in both single and batched modes; the
batched solver is tested on both its Warp tile and cuSOLVER implementations.

Every numerical test is parameterized over dtype so the float64 and
float32 code paths (cuSOLVER ``d*potrf/potrs`` vs ``s*potrf/potrs``)
get identical shape, batch, and contiguity coverage.
"""
import pytest
import numpy as np
import warp as wp

from cupiqp.dense.dense_cholesky import CholeskyInplaceSolver, BatchedCholeskyInplaceSolver
from tests.helpers import to_wp


# ---------------------------------------------------------------------------
# dtype dispatch and tolerances
# ---------------------------------------------------------------------------
DTYPES = [wp.float64, wp.float32]
# Implementations of BatchedCholeskyInplaceSolver, both tested at every size.
BACKENDS = ["warp", "cusolver"]


def _np(dtype):
    return np.float64 if dtype is wp.float64 else np.float32


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
        A = to_wp(_make_spd(n, dtype=_np(dtype)))
        solver = CholeskyInplaceSolver(n, dtype=dtype)
        solver.factorize(A)
        assert not solver.factor_status.numpy().any()

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("n", [1, 2, 4, 8, 16, 32, 64, 128, 256])
    def test_factorize_and_solve(self, n, dtype):
        rng = np.random.default_rng(n)
        A_orig = _make_spd(n, seed=n, dtype=_np(dtype))
        b = rng.standard_normal(n).astype(_np(dtype))
        A_work = to_wp(A_orig)
        x = to_wp(b)

        solver = CholeskyInplaceSolver(n, dtype=dtype)
        solver.factorize(A_work)
        assert not solver.factor_status.numpy().any()
        solver.solve(x)
        np.testing.assert_allclose(
            A_orig @ x.numpy(), b, atol=_atol(_np(dtype)), err_msg=f"n={n}",
        )

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_multiple_rhs(self, dtype):
        n = 5
        rng = np.random.default_rng(42)
        A_orig = _make_spd(n, dtype=_np(dtype))
        A_work = to_wp(A_orig)

        solver = CholeskyInplaceSolver(n, dtype=dtype)
        solver.factorize(A_work)

        for _ in range(3):
            b = rng.standard_normal(n).astype(_np(dtype))
            x = to_wp(b)
            solver.solve(x)
            np.testing.assert_allclose(A_orig @ x.numpy(), b, atol=_atol(_np(dtype)))


# ======================================================================
# Batched mode
# ======================================================================
class TestBatchedMode:

    @pytest.mark.parametrize("backend", BACKENDS)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_factorize_succeeds(self, dtype, backend):
        B, n = 4, 6
        A = to_wp(_make_spd_batch(B, n, dtype=_np(dtype)))
        solver = BatchedCholeskyInplaceSolver(n, B, dtype=dtype, backend=backend)
        solver.factorize(A)
        assert not solver.factor_status.numpy().any()

    # (B, n) matrix exercises: batch=1 (smallest batched case), small
    # batches at various n, and a large-batch case. Fused from the former
    # test_factorize_and_solve / test_batch_size_one / test_various_sizes
    # / test_large_batch - they were all the same "build batched SPD,
    # factor, solve, check per batch" with different (B, n).
    @pytest.mark.parametrize("backend", BACKENDS)
    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("B,n", [
        (1, 4),
        (5, 8),
        (8, 1), (8, 2), (8, 4), (8, 8), (8, 12), (8, 16), (8, 24), (8, 32),
        (8, 48), (8, 64),
        (128, 6),
    ])
    def test_factorize_and_solve(self, B, n, dtype, backend):
        rng = np.random.default_rng(B * 100 + n)
        A_orig = _make_spd_batch(B, n, seed=B * 100 + n, dtype=_np(dtype))
        b = rng.standard_normal((B, n)).astype(_np(dtype))
        A_work = to_wp(A_orig)
        x = to_wp(b)

        solver = BatchedCholeskyInplaceSolver(n, B, dtype=dtype, backend=backend)
        solver.factorize(A_work)
        assert not solver.factor_status.numpy().any()
        solver.solve(x)

        for i in range(B):
            np.testing.assert_allclose(
                A_orig[i] @ x.numpy()[i], b[i], atol=_atol(_np(dtype)),
                err_msg=f"B={B}, n={n}, batch={i}")

    @pytest.mark.parametrize("backend", BACKENDS)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_multiple_solves_same_factorization(self, dtype, backend):
        B, n = 3, 5
        rng = np.random.default_rng(42)
        A_orig = _make_spd_batch(B, n, dtype=_np(dtype))
        A_work = to_wp(A_orig)

        solver = BatchedCholeskyInplaceSolver(n, B, dtype=dtype, backend=backend)
        solver.factorize(A_work)

        for _ in range(3):
            b = rng.standard_normal((B, n)).astype(_np(dtype))
            x = to_wp(b)
            solver.solve(x)
            for i in range(B):
                np.testing.assert_allclose(
                    A_orig[i] @ x.numpy()[i], b[i], atol=_atol(_np(dtype)))

    @pytest.mark.parametrize("backend", BACKENDS)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_non_contiguous_rhs(self, dtype, backend):
        """Batched solve with non-contiguous RHS (view of larger buffer)."""
        B, n = 3, 4
        rng = np.random.default_rng(42)
        A_orig = _make_spd_batch(B, n, dtype=_np(dtype))
        b = rng.standard_normal((B, n)).astype(_np(dtype))
        A_work = to_wp(A_orig)

        buf = np.zeros((B, n + 5), dtype=_np(dtype))
        buf[:, :n] = b
        x_view = to_wp(buf)[:, :n]  # non-contiguous outer stride
        assert not x_view.is_contiguous

        solver = BatchedCholeskyInplaceSolver(n, B, dtype=dtype, backend=backend)
        solver.factorize(A_work)
        solver.solve(x_view)

        for i in range(B):
            np.testing.assert_allclose(
                A_orig[i] @ x_view.numpy()[i], b[i], atol=_atol(_np(dtype)),
                err_msg=f"batch {i}")


# ======================================================================
# Cross-mode: batched B=1 vs single should give same result
# ======================================================================

    @pytest.mark.parametrize("backend", BACKENDS)
    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("n", [5, 32])
    def test_reads_only_upper_triangle(self, n, dtype, backend):
        """The KKT assembly writes only the row-major upper triangle; the rest
        may hold anything."""
        B = 4
        rng = np.random.default_rng(7)
        A_orig = _make_spd_batch(B, n, seed=7, dtype=_np(dtype))
        A_nan = A_orig.copy()
        A_nan[:, np.tril_indices(n, -1)[0], np.tril_indices(n, -1)[1]] = np.nan
        b = rng.standard_normal((B, n)).astype(_np(dtype))
        A_work, x = to_wp(A_nan), to_wp(b)

        solver = BatchedCholeskyInplaceSolver(n, B, dtype=dtype, backend=backend)
        solver.factorize(A_work)
        solver.solve(x)
        assert not solver.factor_status.numpy().any()
        for i in range(B):
            np.testing.assert_allclose(A_orig[i] @ x.numpy()[i], b[i],
                                       atol=_atol(_np(dtype)) * n, err_msg=f"batch {i}")

    @pytest.mark.parametrize("backend", BACKENDS)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_status_flags_only_the_indefinite_matrix(self, dtype, backend):
        B, n, bad = 5, 6, 2
        rng = np.random.default_rng(3)
        A_orig = _make_spd_batch(B, n, seed=3, dtype=_np(dtype))
        A_orig[bad] -= 3 * n * np.eye(n, dtype=_np(dtype))      # indefinite
        b = rng.standard_normal((B, n)).astype(_np(dtype))
        A_work, x = to_wp(A_orig), to_wp(b)

        solver = BatchedCholeskyInplaceSolver(n, B, dtype=dtype, backend=backend)
        solver.factorize(A_work)
        status = solver.factor_status.numpy()
        assert status[bad] != 0
        assert not np.delete(status, bad).any()
        solver.solve(x)
        for i in range(B):
            if i != bad:
                np.testing.assert_allclose(A_orig[i] @ x.numpy()[i], b[i],
                                           atol=_atol(_np(dtype)), err_msg=f"batch {i}")


class TestCrossMode:

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_single_vs_batched_b1(self, dtype):
        n = 6
        rng = np.random.default_rng(42)
        A_np = _make_spd(n, dtype=_np(dtype))
        b_np = rng.standard_normal(n).astype(_np(dtype))

        # Single mode
        A1 = to_wp(A_np)
        x1 = to_wp(b_np)
        s1 = CholeskyInplaceSolver(n, dtype=dtype)
        s1.factorize(A1)
        s1.solve(x1)

        # Batched mode B=1
        A2 = to_wp(A_np.reshape(1, n, n))
        x2 = to_wp(b_np.reshape(1, n))
        s2 = BatchedCholeskyInplaceSolver(n, 1, dtype=dtype)
        s2.factorize(A2)
        s2.solve(x2)

        np.testing.assert_allclose(x1.numpy(), x2.numpy()[0], atol=_atol(_np(dtype)))
