"""Tests for SparseMatVecProduct, with B = 1 and B > 1.

The operators take Warp arrays; the matrices and vectors are built with
NumPy / SciPy on the host and the reference values are computed there.
``.numpy()`` reads wait for the Warp stream the operators run on.
"""
import numpy as np
import pytest
import scipy.sparse as sp_cpu
import warp as wp

from cupiqp.sparse.batched_csr import UniformBatchedCsrMatrix
from cupiqp.sparse.sparse_matvec import SparseMatVecProduct
from tests.helpers import DEVICE, csr_triple, to_wp


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------
def _random_csr(
    m: int, n: int, density: float = 0.3, seed: int = 42,
) -> tuple[UniformBatchedCsrMatrix, sp_cpu.csr_matrix]:
    """Random single-matrix batch plus the host CSR it was built from."""
    rng = np.random.default_rng(seed)
    A_dense = rng.standard_normal((m, n))
    A_dense[rng.random((m, n)) >= density] = 0.0
    A = sp_cpu.csr_matrix(A_dense)
    A.sort_indices()
    return UniformBatchedCsrMatrix(1, *csr_triple(A), shape=A.shape), A


def _random_batched_csr(
    B: int, m: int, n: int, density: float = 0.5, seed: int = 42,
) -> tuple[UniformBatchedCsrMatrix, sp_cpu.csr_matrix, np.ndarray]:
    """Random ``UniformBatchedCsrMatrix``, its host pattern, and the host
    (B, nnz) values backing it."""
    rng = np.random.default_rng(seed)
    template = sp_cpu.random(m, n, density=density, format='csr',
                             random_state=rng, dtype=np.float64)
    template.sort_indices()
    values = rng.standard_normal((B, template.nnz))
    indptr, indices, _ = csr_triple(template)
    return (
        UniformBatchedCsrMatrix(
            batch_size=B,
            indptr=indptr,
            indices=indices,
            data=to_wp(values),
            shape=template.shape,
        ),
        template,
        values,
    )


def _batched_ref(
    template: sp_cpu.csr_matrix,
    values: np.ndarray,
    x: np.ndarray,
    transpose: bool = False,
) -> np.ndarray:
    """Per-batch CPU reference: ``y[b] = A[b] @ x[b]`` (or ``A[b].T @ x[b]``)."""
    B = values.shape[0]
    out_dim = template.shape[1] if transpose else template.shape[0]
    y = np.empty((B, out_dim), dtype=np.float64)
    for b in range(B):
        Ab = template.copy()
        Ab.data[:] = values[b]
        y[b] = (Ab.T @ x[b]) if transpose else (Ab @ x[b])
    return y


# Mixed shape coverage: tiny / square / tall / wide / asymmetric / large.
SHAPES: list[tuple[int, int]] = [
    (1, 1),
    (1, 8),
    (8, 1),
    (3, 3),
    (4, 5),
    (5, 4),
    (16, 16),
    (128, 128),
    (200, 32),     # tall
    (32, 200),     # wide
    (37, 53),      # non-power-of-two
    (512, 512),    # larger square
    (1024, 256),   # large tall
]

# A smaller grid for batched correctness - the batched path stacks B copies
# block-diagonally so the dimensions multiply by B and "large square" hurts.
BATCHED_SHAPES: list[tuple[int, int]] = [
    (1, 1),
    (3, 3),
    (4, 5),
    (5, 4),
    (16, 16),
    (37, 53),
    (200, 32),
    (32, 200),
]


# ===========================================================================
# One-matrix batches (B = 1)
# ===========================================================================

@pytest.mark.parametrize("density", [0.05, 0.3, 0.9])
@pytest.mark.parametrize("m,n", SHAPES)
def test_basic_spmv(m: int, n: int, density: float) -> None:
    mat, A = _random_csr(m, n, density=density, seed=m * 1000 + n)
    rng = np.random.default_rng(m * 7 + n)
    x = rng.standard_normal(n)
    y = to_wp(np.zeros(m))

    op = SparseMatVecProduct(mat)
    op(to_wp(x), y, alpha=1.0, beta=0.0)
    np.testing.assert_allclose(y.numpy(), A.toarray() @ x, atol=1e-12)


@pytest.mark.parametrize("density", [0.05, 0.3, 0.9])
@pytest.mark.parametrize("m,n", SHAPES)
def test_transpose(m: int, n: int, density: float) -> None:
    mat, A = _random_csr(m, n, density=density, seed=m * 31 + n)
    rng = np.random.default_rng(m * 13 + n + 1)
    x = rng.standard_normal(m)
    y = to_wp(np.zeros(n))

    op = SparseMatVecProduct(mat, transa=True)
    op(to_wp(x), y, alpha=1.0, beta=0.0)
    np.testing.assert_allclose(y.numpy(), A.toarray().T @ x, atol=1e-12)


@pytest.mark.parametrize("m,n", SHAPES)
def test_alpha_beta(m: int, n: int) -> None:
    mat, A = _random_csr(m, n, seed=m * 17 + n + 2)
    rng = np.random.default_rng(m * 19 + n + 3)
    x = rng.standard_normal(n)
    y_before = rng.standard_normal(m)
    y = to_wp(y_before)

    op = SparseMatVecProduct(mat)
    op(to_wp(x), y, alpha=2.0, beta=0.5)
    np.testing.assert_allclose(
        y.numpy(), 2.0 * (A.toarray() @ x) + 0.5 * y_before, atol=1e-12,
    )


@pytest.mark.parametrize("m,n", SHAPES)
def test_alpha_beta_transpose(m: int, n: int) -> None:
    mat, A = _random_csr(m, n, seed=m * 23 + n + 4)
    rng = np.random.default_rng(m * 29 + n + 5)
    x = rng.standard_normal(m)
    y_before = rng.standard_normal(n)
    y = to_wp(y_before)

    op = SparseMatVecProduct(mat, transa=True)
    op(to_wp(x), y, alpha=-1.5, beta=2.0)
    np.testing.assert_allclose(
        y.numpy(), -1.5 * (A.toarray().T @ x) + 2.0 * y_before, atol=1e-12,
    )


def test_one_row_view_accepted() -> None:
    """A ``(1, k)`` row view of a wider buffer is accepted for x and y."""
    mat, A = _random_csr(4, 5, seed=3)
    big_x_h = np.random.default_rng(4).standard_normal((1, 9))
    big_x, big_y = to_wp(big_x_h), to_wp(np.zeros((1, 7)))
    x_view, y_view = big_x[:, 2:7], big_y[:, 1:5]

    op = SparseMatVecProduct(mat)
    op(x_view, y_view)
    out = big_y.numpy()
    np.testing.assert_allclose(out[0, 1:5], A.toarray() @ big_x_h[0, 2:7], atol=1e-12)
    np.testing.assert_allclose(out[0, 0], 0.0)
    np.testing.assert_allclose(out[0, 5:], 0.0)


def test_different_buffers() -> None:
    mat, A = _random_csr(3, 4)
    op = SparseMatVecProduct(mat)
    for x in (np.ones(4), np.arange(4, dtype=np.float64)):
        y = to_wp(np.zeros(3))
        op(to_wp(x), y)
        np.testing.assert_allclose(y.numpy(), A.toarray() @ x, atol=1e-12)


def test_reuse_different_scalars() -> None:
    mat, A = _random_csr(4, 4)
    x = np.arange(4, dtype=np.float64)
    x_d, y = to_wp(x), to_wp(np.zeros(4))

    op = SparseMatVecProduct(mat)
    op(x_d, y, alpha=1.0, beta=0.0)
    y1 = y.numpy()
    np.testing.assert_allclose(y1, A.toarray() @ x, atol=1e-12)

    op(x_d, y, alpha=3.0, beta=1.0)
    np.testing.assert_allclose(y.numpy(), 3.0 * (A.toarray() @ x) + y1, atol=1e-12)


@pytest.mark.parametrize("n", [1, 2, 8, 64, 256])
def test_identity(n: int) -> None:
    A = sp_cpu.eye(n, dtype=np.float64, format="csr")
    mat = UniformBatchedCsrMatrix(1, *csr_triple(A), shape=A.shape)
    x = np.arange(n, dtype=np.float64)
    y = to_wp(np.zeros(n))

    op = SparseMatVecProduct(mat)
    op(to_wp(x), y)
    np.testing.assert_allclose(y.numpy(), x, atol=1e-14)


def test_in_place_value_update_is_seen() -> None:
    """The operator reads ``mat.data`` at every call."""
    mat, A = _random_csr(4, 5, seed=11)
    x = np.random.default_rng(12).standard_normal(5)
    y = to_wp(np.zeros(4))
    op = SparseMatVecProduct(mat)

    new_vals = np.random.default_rng(13).standard_normal(mat.nnz)
    wp.copy(mat.data, to_wp(new_vals.reshape(1, -1)))
    op(to_wp(x), y)
    A2 = A.copy()
    A2.data[:] = new_vals
    np.testing.assert_allclose(y.numpy(), A2.toarray() @ x, atol=1e-12)


def test_cuda_graph_capture() -> None:
    mat, A = _random_csr(4, 4)
    x = np.ones(4)
    xw, yw = to_wp(x), to_wp(np.zeros(4))

    op = SparseMatVecProduct(mat)
    stream = wp.Stream("cuda")
    with wp.ScopedStream(stream):
        with wp.ScopedCapture(stream=stream) as capture:
            op(xw, yw, alpha=1.0, beta=0.0)
        wp.capture_launch(capture.graph, stream=stream)
    wp.synchronize_stream(stream)
    np.testing.assert_allclose(yw.numpy(), A.toarray() @ x, atol=1e-12)


def test_destructor_no_error() -> None:
    mat, _ = _random_csr(2, 2)
    op = SparseMatVecProduct(mat)
    del op  # must not raise


# ===========================================================================
# Batches of several (B > 1)
# ===========================================================================

@pytest.mark.parametrize("B", [1, 2, 5, 16])
@pytest.mark.parametrize("m,n", BATCHED_SHAPES)
def test_batched_basic_contiguous(B: int, m: int, n: int) -> None:
    bmat, tpl, vals = _random_batched_csr(B, m, n, density=0.4, seed=B * 100 + m * 10 + n)
    op = SparseMatVecProduct(bmat, transa=False)

    x = np.random.default_rng(B + m + n).standard_normal((B, n))
    out = to_wp(np.zeros((B, m)))
    op(to_wp(x), out)
    np.testing.assert_allclose(
        out.numpy(), _batched_ref(tpl, vals, x), atol=1e-12,
    )


@pytest.mark.parametrize("B", [1, 2, 5, 16])
@pytest.mark.parametrize("m,n", BATCHED_SHAPES)
def test_batched_transpose(B: int, m: int, n: int) -> None:
    bmat, tpl, vals = _random_batched_csr(B, m, n, density=0.5, seed=B * 200 + m * 7 + n)
    op = SparseMatVecProduct(bmat, transa=True)

    x = np.random.default_rng(B * 3 + m + n).standard_normal((B, m))
    out = to_wp(np.zeros((B, n)))
    op(to_wp(x), out)
    np.testing.assert_allclose(
        out.numpy(), _batched_ref(tpl, vals, x, transpose=True), atol=1e-12,
    )


@pytest.mark.parametrize("B", [1, 2, 5, 16])
@pytest.mark.parametrize("m,n", BATCHED_SHAPES)
def test_batched_alpha_beta(B: int, m: int, n: int) -> None:
    bmat, tpl, vals = _random_batched_csr(B, m, n, density=0.5, seed=B * 300 + m * 11 + n)
    op = SparseMatVecProduct(bmat)

    rng = np.random.default_rng(B * 5 + m + n + 1)
    x = rng.standard_normal((B, n))
    y_init = rng.standard_normal((B, m))
    y = to_wp(y_init)

    alpha, beta = 2.0, 3.0
    op(to_wp(x), y, alpha=alpha, beta=beta)
    expected = alpha * _batched_ref(tpl, vals, x) + beta * y_init
    np.testing.assert_allclose(y.numpy(), expected, atol=1e-12)


def test_batched_non_contiguous_x() -> None:
    B, m, n = 4, 5, 6
    K = 10
    col_start = 2
    bmat, tpl, vals = _random_batched_csr(B, m, n, seed=60)
    op = SparseMatVecProduct(bmat)

    big = np.random.default_rng(62).standard_normal((B, K))
    x_view = to_wp(big)[:, col_start:col_start + n]
    assert not x_view.is_contiguous

    out = to_wp(np.zeros((B, m)))
    op(x_view, out)
    np.testing.assert_allclose(
        out.numpy(), _batched_ref(tpl, vals, big[:, col_start:col_start + n]), atol=1e-12,
    )


def test_batched_non_contiguous_out() -> None:
    B, m, n = 3, 4, 5
    K = 9
    col_start = 3
    bmat, tpl, vals = _random_batched_csr(B, m, n, seed=63)
    op = SparseMatVecProduct(bmat)

    x = np.random.default_rng(65).standard_normal((B, n))
    big_out = to_wp(np.full((B, K), -7.0))  # sentinel outside the window
    out_view = big_out[:, col_start:col_start + m]
    assert not out_view.is_contiguous

    op(to_wp(x), out_view)
    out = big_out.numpy()
    np.testing.assert_allclose(
        out[:, col_start:col_start + m], _batched_ref(tpl, vals, x), atol=1e-12,
    )
    # Columns outside the target window are untouched.
    np.testing.assert_allclose(out[:, :col_start], -7.0)
    np.testing.assert_allclose(out[:, col_start + m:], -7.0)


def test_batched_non_contiguous_out_with_beta() -> None:
    """With ``beta != 0`` the current values of a strided y are read first."""
    B, m, n = 3, 4, 5
    bmat, tpl, vals = _random_batched_csr(B, m, n, seed=66)
    op = SparseMatVecProduct(bmat)

    rng = np.random.default_rng(67)
    x = rng.standard_normal((B, n))
    big_out_h = rng.standard_normal((B, 9))
    out_view = to_wp(big_out_h)[:, 2:2 + m]
    y_init = big_out_h[:, 2:2 + m].copy()

    op(to_wp(x), out_view, alpha=1.0, beta=0.5)
    np.testing.assert_allclose(
        out_view.numpy(), _batched_ref(tpl, vals, x) + 0.5 * y_init, atol=1e-12,
    )


def test_batched_multiple_calls_different_x() -> None:
    B, m, n = 3, 4, 5
    bmat, tpl, vals = _random_batched_csr(B, m, n, seed=70)
    op = SparseMatVecProduct(bmat)

    x1 = np.random.default_rng(72).standard_normal((B, n))
    x2 = np.random.default_rng(73).standard_normal((B, n))
    out = wp.empty((B, m), dtype=wp.float64, device=DEVICE)

    op(to_wp(x1), out)
    np.testing.assert_allclose(
        out.numpy(), _batched_ref(tpl, vals, x1), atol=1e-12,
    )
    op(to_wp(x2), out)
    np.testing.assert_allclose(
        out.numpy(), _batched_ref(tpl, vals, x2), atol=1e-12,
    )


def test_batched_tracks_in_place_value_update() -> None:
    """After copying new values into ``bmat.data``, the SpMV uses them."""
    B, m, n = 3, 4, 5
    bmat, tpl, _ = _random_batched_csr(B, m, n, seed=74)
    op = SparseMatVecProduct(bmat)

    x = np.random.default_rng(76).standard_normal((B, n))
    x_d = to_wp(x)
    out = wp.empty((B, m), dtype=wp.float64, device=DEVICE)
    op(x_d, out)

    new_vals = np.random.default_rng(77).standard_normal((B, tpl.nnz))
    wp.copy(bmat.data, to_wp(new_vals))
    op(x_d, out)
    np.testing.assert_allclose(
        out.numpy(), _batched_ref(tpl, new_vals, x), atol=1e-12,
    )
