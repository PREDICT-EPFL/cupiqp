"""Tests for the dgemv / sgemv (and their strided-batched siblings)
cuBLAS wrappers.

Every test is parameterized over dtype so the float64 (``d*``) and
float32 (``s*``) wrappers exercise identical shape, layout, scalar,
and graph-capture coverage. Tolerances are dtype-specific (see ``_atol``).
"""
import cupy as cp
import numpy as np
import pytest

from cupiqp.dense.cublas_wrappers import (
    dgemv,
    sgemv,
    dgemv_strided_batched,
    sgemv_strided_batched,
    dgemm_strided_batched,
    sgemm_strided_batched,
)


# ---------------------------------------------------------------------------
# dtype <-> wrapper / tolerance dispatch
# ---------------------------------------------------------------------------
DTYPES = [cp.float64, cp.float32]


def _gemv(dtype):
    return dgemv if dtype == cp.float64 else sgemv


def _gemv_batched(dtype):
    return dgemv_strided_batched if dtype == cp.float64 else sgemv_strided_batched


def _gemm_batched(dtype):
    return dgemm_strided_batched if dtype == cp.float64 else sgemm_strided_batched


def _atol(dtype):
    return 1e-12 if dtype == cp.float64 else 1e-4


def _rtol(dtype):
    return 1e-7 if dtype == cp.float64 else 1e-4


@pytest.fixture
def handle():
    return cp.cuda.Device().cublas_handle


def _random_dense(m, n, order="C", dtype=cp.float64, seed=42):
    rng = np.random.default_rng(seed)
    return cp.array(rng.standard_normal((m, n)), dtype=dtype, order=order)


# Mixed shape coverage: tiny / square / tall / wide / asymmetric / large.
SHAPES = [
    (1, 1),
    (1, 8),
    (8, 1),
    (3, 3),
    (4, 5),
    (5, 4),
    (16, 16),
    (32, 32),
    (128, 128),
    (200, 32),     # tall
    (32, 200),     # wide
    (37, 53),      # non-power-of-two
    (512, 512),    # larger square
    (1024, 256),   # large tall
]


# ---------------------------------------------------------------------------
# Shape × order × dtype matrix
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("m,n", SHAPES)
def test_basic_gemv(handle, m, n, order, dtype):
    A = _random_dense(m, n, order=order, dtype=dtype, seed=m * 1000 + n)
    rng = np.random.default_rng(m * 7 + n)
    x = cp.asarray(rng.standard_normal(n), dtype=dtype)
    y = cp.zeros(m, dtype=dtype)

    _gemv(dtype)(handle, A, x, y)
    cp.cuda.get_current_stream().synchronize()
    cp.testing.assert_allclose(y, A @ x, atol=_atol(dtype), rtol=_rtol(dtype))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("m,n", SHAPES)
def test_transpose(handle, m, n, order, dtype):
    A = _random_dense(m, n, order=order, dtype=dtype, seed=m * 31 + n)
    rng = np.random.default_rng(m * 13 + n + 1)
    x = cp.asarray(rng.standard_normal(m), dtype=dtype)
    y = cp.zeros(n, dtype=dtype)

    _gemv(dtype)(handle, A, x, y, transa=True)
    cp.cuda.get_current_stream().synchronize()
    cp.testing.assert_allclose(y, A.T @ x, atol=_atol(dtype), rtol=_rtol(dtype))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("m,n", SHAPES)
def test_alpha_beta(handle, m, n, order, dtype):
    A = _random_dense(m, n, order=order, dtype=dtype, seed=m * 17 + n + 2)
    rng = np.random.default_rng(m * 19 + n + 3)
    x = cp.asarray(rng.standard_normal(n), dtype=dtype)
    y = cp.asarray(rng.standard_normal(m), dtype=dtype)
    y_before = y.copy()

    _gemv(dtype)(handle, A, x, y, alpha=2.0, beta=0.5)
    cp.cuda.get_current_stream().synchronize()
    cp.testing.assert_allclose(
        y, 2.0 * (A @ x) + 0.5 * y_before, atol=_atol(dtype), rtol=_rtol(dtype),
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("m,n", SHAPES)
def test_alpha_beta_transpose(handle, m, n, order, dtype):
    A = _random_dense(m, n, order=order, dtype=dtype, seed=m * 23 + n + 4)
    rng = np.random.default_rng(m * 29 + n + 5)
    x = cp.asarray(rng.standard_normal(m), dtype=dtype)
    y = cp.asarray(rng.standard_normal(n), dtype=dtype)
    y_before = y.copy()

    _gemv(dtype)(handle, A, x, y, transa=True, alpha=-1.5, beta=2.0)
    cp.cuda.get_current_stream().synchronize()
    cp.testing.assert_allclose(
        y, -1.5 * (A.T @ x) + 2.0 * y_before, atol=_atol(dtype), rtol=_rtol(dtype),
    )


# ---------------------------------------------------------------------------
# Scalar-edge behaviour
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("m,n", [(8, 5), (5, 8), (32, 32)])
def test_alpha_zero_beta_one_is_identity(handle, m, n, dtype):
    """alpha=0, beta=1 must leave y unchanged regardless of A and x."""
    A = _random_dense(m, n, dtype=dtype, seed=m + n)
    x = cp.asarray(np.random.default_rng(11).standard_normal(n), dtype=dtype)
    y = cp.asarray(np.random.default_rng(12).standard_normal(m), dtype=dtype)
    y_before = y.copy()

    _gemv(dtype)(handle, A, x, y, alpha=0.0, beta=1.0)
    cp.cuda.get_current_stream().synchronize()
    cp.testing.assert_allclose(y, y_before, atol=_atol(dtype), rtol=_rtol(dtype))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("m,n", [(8, 5), (5, 8), (32, 32)])
def test_beta_zero_overwrites_garbage(handle, m, n, dtype):
    """beta=0 must overwrite y completely, even if it held NaN before."""
    A = _random_dense(m, n, dtype=dtype, seed=m * 3 + n)
    x = cp.asarray(np.random.default_rng(13).standard_normal(n), dtype=dtype)
    y = cp.full(m, cp.nan, dtype=dtype)

    _gemv(dtype)(handle, A, x, y, alpha=1.0, beta=0.0)
    cp.cuda.get_current_stream().synchronize()
    cp.testing.assert_allclose(y, A @ x, atol=_atol(dtype), rtol=_rtol(dtype))


# ---------------------------------------------------------------------------
# Misc small / sanity cases
# ---------------------------------------------------------------------------
# Note: ``test_inplace_buffer_update`` (gemv at (3, 3)) and
# ``test_different_buffers`` (gemv at (3, 4) with two ``x``) were dropped —
# both were subsumed by ``test_basic_gemv`` which already covers shape (3, 3)
# and the symmetric small rectangular shapes via ``SHAPES``.
@pytest.mark.parametrize("dtype", DTYPES)
def test_reuse_different_scalars(handle, dtype):
    A = _random_dense(4, 4, dtype=dtype)
    x = cp.arange(4, dtype=dtype)
    y = cp.zeros(4, dtype=dtype)

    _gemv(dtype)(handle, A, x, y, alpha=1.0, beta=0.0)
    cp.cuda.get_current_stream().synchronize()
    y1 = y.copy()
    cp.testing.assert_allclose(y1, A @ x, atol=_atol(dtype), rtol=_rtol(dtype))

    _gemv(dtype)(handle, A, x, y, alpha=3.0, beta=1.0)
    cp.cuda.get_current_stream().synchronize()
    cp.testing.assert_allclose(
        y, 3.0 * (A @ x) + y1, atol=_atol(dtype), rtol=_rtol(dtype),
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n", [1, 2, 8, 64, 256])
def test_identity(handle, n, dtype):
    A = cp.eye(n, dtype=dtype)
    x = cp.arange(n, dtype=dtype)
    y = cp.zeros(n, dtype=dtype)

    _gemv(dtype)(handle, A, x, y)
    cp.cuda.get_current_stream().synchronize()
    cp.testing.assert_allclose(y, x, atol=_atol(dtype), rtol=_rtol(dtype))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n", [3, 16, 64])
def test_symmetric_matrix(handle, n, dtype):
    rng = np.random.default_rng(123 + n)
    H = rng.standard_normal((n, n))
    A = cp.array(H + H.T, dtype=dtype)
    x = cp.asarray(rng.standard_normal(n), dtype=dtype)
    y = cp.zeros(n, dtype=dtype)
    y_T = cp.zeros(n, dtype=dtype)

    _gemv(dtype)(handle, A, x, y)
    _gemv(dtype)(handle, A, x, y_T, transa=True)
    cp.cuda.get_current_stream().synchronize()
    cp.testing.assert_allclose(y, A @ x, atol=_atol(dtype), rtol=_rtol(dtype))
    cp.testing.assert_allclose(y_T, y, atol=_atol(dtype), rtol=_rtol(dtype))


@pytest.mark.parametrize("dtype", DTYPES)
def test_scalar_matrix(handle, dtype):
    A = cp.asarray([[3.0]], dtype=dtype)
    x = cp.asarray([2.0], dtype=dtype)
    y = cp.zeros(1, dtype=dtype)
    _gemv(dtype)(handle, A, x, y)
    cp.cuda.get_current_stream().synchronize()
    cp.testing.assert_allclose(
        y, cp.asarray([6.0], dtype=dtype), atol=_atol(dtype), rtol=_rtol(dtype),
    )


# ---------------------------------------------------------------------------
# CUDA graph capture
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("transa", [False, True])
@pytest.mark.parametrize("order", ["C", "F"])
def test_cuda_graph_capture(handle, transa, order, dtype):
    A = _random_dense(4, 5, order=order, dtype=dtype)
    x = cp.ones(4 if transa else 5, dtype=dtype)
    y = cp.zeros(5 if transa else 4, dtype=dtype)

    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        stream.begin_capture()
        _gemv(dtype)(handle, A, x, y, transa=transa, alpha=1.0, beta=0.0)
        graph = stream.end_capture()

    graph.launch(stream)
    stream.synchronize()
    expected = A.T @ x if transa else A @ x
    cp.testing.assert_allclose(y, expected, atol=_atol(dtype), rtol=_rtol(dtype))


# ===========================================================================
# Batched gemv (dgemv_strided_batched / sgemv_strided_batched)
#
# The wrapper takes C-contiguous ``(batch, m, n)`` matrices and
# ``(batch, n)`` / ``(batch, m)`` vectors. F-order is not supported by the
# wrapper, so order is not parameterized here.
# ===========================================================================
BATCH_SHAPES = [
    (1, 1, 1),
    (1, 8, 5),
    (4, 3, 3),
    (4, 4, 5),
    (4, 5, 4),
    (8, 16, 16),
    (8, 32, 32),
    (16, 7, 11),    # non-power-of-two
    (32, 8, 24),    # rectangular
    (64, 16, 16),
    (128, 6, 6),    # large batch, tiny matrix
    (2, 1100, 1100),
]


def _random_batched(batch, m, n, dtype=cp.float64, seed=0):
    rng = np.random.default_rng(seed)
    return cp.asarray(rng.standard_normal((batch, m, n)), dtype=dtype)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("batch,m,n", BATCH_SHAPES)
def test_batched_gemv_basic(handle, batch, m, n, dtype):
    A = _random_batched(batch, m, n, dtype=dtype, seed=batch * 31 + m * 7 + n)
    rng = np.random.default_rng(batch + m * 3 + n * 5)
    x = cp.asarray(rng.standard_normal((batch, n)), dtype=dtype)
    y = cp.zeros((batch, m), dtype=dtype)

    _gemv_batched(dtype)(handle, A, x, y)
    cp.cuda.get_current_stream().synchronize()
    expected = cp.einsum("bij,bj->bi", A, x)
    cp.testing.assert_allclose(y, expected, atol=_atol(dtype), rtol=_rtol(dtype))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("batch,m,n", BATCH_SHAPES)
def test_batched_gemv_transpose(handle, batch, m, n, dtype):
    A = _random_batched(batch, m, n, dtype=dtype, seed=batch * 41 + m * 11 + n)
    rng = np.random.default_rng(batch * 2 + m * 3 + n * 7)
    x = cp.asarray(rng.standard_normal((batch, m)), dtype=dtype)
    y = cp.zeros((batch, n), dtype=dtype)

    _gemv_batched(dtype)(handle, A, x, y, transa=True)
    cp.cuda.get_current_stream().synchronize()
    expected = cp.einsum("bij,bi->bj", A, x)
    cp.testing.assert_allclose(y, expected, atol=_atol(dtype), rtol=_rtol(dtype))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("batch,m,n", [(4, 5, 4), (8, 16, 16), (32, 8, 12)])
def test_batched_gemv_alpha_beta(handle, batch, m, n, dtype):
    A = _random_batched(batch, m, n, dtype=dtype, seed=batch * 17 + m + n)
    rng = np.random.default_rng(batch * 19 + m + n + 1)
    x = cp.asarray(rng.standard_normal((batch, n)), dtype=dtype)
    y = cp.asarray(rng.standard_normal((batch, m)), dtype=dtype)
    y_before = y.copy()

    _gemv_batched(dtype)(handle, A, x, y, alpha=2.0, beta=0.5)
    cp.cuda.get_current_stream().synchronize()
    expected = 2.0 * cp.einsum("bij,bj->bi", A, x) + 0.5 * y_before
    cp.testing.assert_allclose(y, expected, atol=_atol(dtype), rtol=_rtol(dtype))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("batch,m,n", [(4, 5, 4), (8, 16, 16), (32, 8, 12)])
def test_batched_gemv_alpha_beta_transpose(handle, batch, m, n, dtype):
    A = _random_batched(batch, m, n, dtype=dtype, seed=batch * 23 + m + n)
    rng = np.random.default_rng(batch * 29 + m + n + 2)
    x = cp.asarray(rng.standard_normal((batch, m)), dtype=dtype)
    y = cp.asarray(rng.standard_normal((batch, n)), dtype=dtype)
    y_before = y.copy()

    _gemv_batched(dtype)(handle, A, x, y, transa=True, alpha=-1.5, beta=2.0)
    cp.cuda.get_current_stream().synchronize()
    expected = -1.5 * cp.einsum("bij,bi->bj", A, x) + 2.0 * y_before
    cp.testing.assert_allclose(y, expected, atol=_atol(dtype), rtol=_rtol(dtype))


@pytest.mark.parametrize("dtype", DTYPES)
def test_batched_gemv_matches_per_sample_gemv(handle, dtype):
    """Batched call must agree element-wise with calling gemv per batch."""
    batch, m, n = 6, 7, 5
    A = _random_batched(batch, m, n, dtype=dtype, seed=99)
    rng = np.random.default_rng(100)
    x = cp.asarray(rng.standard_normal((batch, n)), dtype=dtype)
    y_batched = cp.zeros((batch, m), dtype=dtype)
    y_loop = cp.zeros((batch, m), dtype=dtype)

    _gemv_batched(dtype)(handle, A, x, y_batched, alpha=1.3, beta=0.0)
    for i in range(batch):
        _gemv(dtype)(handle, A[i], x[i], y_loop[i], alpha=1.3, beta=0.0)
    cp.cuda.get_current_stream().synchronize()
    cp.testing.assert_allclose(
        y_batched, y_loop, atol=_atol(dtype), rtol=_rtol(dtype),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_batched_gemv_batch_size_one(handle, dtype):
    """``batch=1`` must produce the same result as the unbatched gemv."""
    m, n = 8, 5
    A = _random_batched(1, m, n, dtype=dtype, seed=7)
    rng = np.random.default_rng(8)
    x_b = cp.asarray(rng.standard_normal((1, n)), dtype=dtype)
    y_b = cp.zeros((1, m), dtype=dtype)

    _gemv_batched(dtype)(handle, A, x_b, y_b)

    y_single = cp.zeros(m, dtype=dtype)
    _gemv(dtype)(handle, A[0], x_b[0], y_single)
    cp.cuda.get_current_stream().synchronize()
    cp.testing.assert_allclose(
        y_b[0], y_single, atol=_atol(dtype), rtol=_rtol(dtype),
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("transa", [False, True])
def test_batched_gemv_cuda_graph_capture(handle, transa, dtype):
    batch, m, n = 4, 6, 5
    A = _random_batched(batch, m, n, dtype=dtype, seed=51)
    x = cp.ones((batch, m if transa else n), dtype=dtype)
    y = cp.zeros((batch, n if transa else m), dtype=dtype)

    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        stream.begin_capture()
        _gemv_batched(dtype)(handle, A, x, y, transa=transa, alpha=1.0, beta=0.0)
        graph = stream.end_capture()

    graph.launch(stream)
    stream.synchronize()
    if transa:
        expected = cp.einsum("bij,bi->bj", A, x)
    else:
        expected = cp.einsum("bij,bj->bi", A, x)
    cp.testing.assert_allclose(y, expected, atol=_atol(dtype), rtol=_rtol(dtype))


# ===========================================================================
# Batched gemm (dgemm_strided_batched / sgemm_strided_batched)
# ===========================================================================
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("batch,m,k,n", [
    (1, 4, 5, 3),
    (4, 8, 6, 7),
    (8, 16, 16, 16),
    (16, 5, 11, 9),
    (32, 6, 6, 6),
])
def test_batched_gemm_basic(handle, batch, m, k, n, dtype):
    A = _random_batched(batch, m, k, dtype=dtype, seed=batch * 37 + m + k + n)
    B = _random_batched(batch, k, n, dtype=dtype, seed=batch * 41 + m + k + n + 1)
    C = cp.zeros((batch, m, n), dtype=dtype)

    _gemm_batched(dtype)(handle, A, B, C)
    cp.cuda.get_current_stream().synchronize()
    expected = cp.einsum("bij,bjk->bik", A, B)
    cp.testing.assert_allclose(C, expected, atol=_atol(dtype), rtol=_rtol(dtype))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("transa,transb", [
    (False, True),
    (True, False),
    (True, True),
])
def test_batched_gemm_transpose_flags(handle, transa, transb, dtype):
    batch, m, k, n = 6, 5, 7, 4
    A_rows, A_cols = (k, m) if transa else (m, k)
    B_rows, B_cols = (n, k) if transb else (k, n)
    A = _random_batched(batch, A_rows, A_cols, dtype=dtype, seed=71)
    B = _random_batched(batch, B_rows, B_cols, dtype=dtype, seed=72)
    C = cp.zeros((batch, m, n), dtype=dtype)

    _gemm_batched(dtype)(handle, A, B, C, transa=transa, transb=transb)
    cp.cuda.get_current_stream().synchronize()

    A_eff = A.transpose(0, 2, 1) if transa else A
    B_eff = B.transpose(0, 2, 1) if transb else B
    expected = cp.einsum("bij,bjk->bik", A_eff, B_eff)
    cp.testing.assert_allclose(C, expected, atol=_atol(dtype), rtol=_rtol(dtype))


@pytest.mark.parametrize("dtype", DTYPES)
def test_batched_gemm_alpha_beta(handle, dtype):
    batch, m, k, n = 4, 5, 6, 7
    A = _random_batched(batch, m, k, dtype=dtype, seed=81)
    B = _random_batched(batch, k, n, dtype=dtype, seed=82)
    C = _random_batched(batch, m, n, dtype=dtype, seed=83)
    C_before = C.copy()

    _gemm_batched(dtype)(handle, A, B, C, alpha=2.0, beta=0.5)
    cp.cuda.get_current_stream().synchronize()
    expected = 2.0 * cp.einsum("bij,bjk->bik", A, B) + 0.5 * C_before
    cp.testing.assert_allclose(C, expected, atol=_atol(dtype), rtol=_rtol(dtype))
