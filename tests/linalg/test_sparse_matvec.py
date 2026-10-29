"""Tests for SparseMatVecProduct, with B = 1 and B > 1.

The operators take Warp arrays; the matrices and vectors are built with
NumPy / SciPy on the host, rounded to the test dtype (float64 / float32), and
the reference values are computed there in float64.
``.numpy()`` reads wait for the Warp stream the operators run on.
"""
import numpy as np
import pytest
import scipy.sparse as sp
import warp as wp

from cupiqp.sparse.batched_csr import UniformBatchedCsrMatrix
from cupiqp.sparse.sparse_matvec import SparseMatVecProduct
from tests.helpers import DEVICE, DTYPES, csr_triple, host, rounded, to_wp
from tests.tolerances import MATVEC_TOL


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------
def _random_csr(
    m: int, n: int, dtype, density: float = 0.3, seed: int = 42,
) -> tuple[UniformBatchedCsrMatrix, sp.csr_matrix]:
    """Random single-matrix batch plus the host CSR it was built from."""
    rng = np.random.default_rng(seed)
    A_dense = rounded(rng.standard_normal((m, n)), dtype)
    A_dense[rng.random((m, n)) >= density] = 0.0
    A = sp.csr_matrix(A_dense)
    A.sort_indices()
    triple = csr_triple(A, wp.dtype_to_numpy(dtype))
    return UniformBatchedCsrMatrix(1, *triple, shape=A.shape, dtype=dtype), A


def _random_batched_csr(
    B: int, m: int, n: int, dtype, density: float = 0.5, seed: int = 42,
) -> tuple[UniformBatchedCsrMatrix, sp.csr_matrix, np.ndarray]:
    """Random ``UniformBatchedCsrMatrix``, its host pattern, and the host
    (B, nnz) values backing it."""
    rng = np.random.default_rng(seed)
    template = sp.random(m, n, density=density, format='csr',
                             random_state=rng, dtype=np.float64)
    template.sort_indices()
    values = rounded(rng.standard_normal((B, template.nnz)), dtype)
    indptr, indices, _ = csr_triple(template)
    return (
        UniformBatchedCsrMatrix(
            batch_size=B,
            indptr=indptr,
            indices=indices,
            data=to_wp(values, dtype),
            shape=template.shape,
            dtype=dtype,
        ),
        template,
        values,
    )


def _batched_ref(
    template: sp.csr_matrix,
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


# Tolerances of the products, per dtype.

# Mixed shape coverage: tiny / square / tall / wide / asymmetric / large.
SHAPES: list[tuple[int, int]] = [
    (1, 1),
    (1, 8),
    (8, 1),
    (200, 32),     # tall
    (32, 200),     # wide
    (37, 53),      # non-power-of-two
    (1024, 256),   # large tall
]

# (m, n, density): every shape at a medium density, and the sparsity sweep
# on the non-power-of-two shape.
DENSITY_CASES = [(m, n, 0.3) for m, n in SHAPES] + [(37, 53, 0.05), (37, 53, 0.9)]

# A smaller grid for batched correctness - the batched path stacks B copies
# block-diagonally so the dimensions multiply by B and "large square" hurts.
BATCHED_SHAPES: list[tuple[int, int]] = [
    (1, 1),
    (4, 5),
    (37, 53),
    (200, 32),
    (32, 200),
]


# ===========================================================================
# One-matrix batches (B = 1)
# ===========================================================================

@pytest.mark.parametrize("m,n,density", DENSITY_CASES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_basic_spmv(m: int, n: int, density: float, dtype) -> None:
    mat, A = _random_csr(m, n, dtype, density=density, seed=m * 1000 + n)
    rng = np.random.default_rng(m * 7 + n)
    x = rounded(rng.standard_normal(n), dtype)
    y = to_wp(np.zeros(m), dtype)

    op = SparseMatVecProduct(mat)
    op(to_wp(x, dtype), y, alpha=1.0, beta=0.0)
    np.testing.assert_allclose(host(y), A.toarray() @ x, **MATVEC_TOL[dtype])


@pytest.mark.parametrize("m,n,density", DENSITY_CASES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_transpose(m: int, n: int, density: float, dtype) -> None:
    mat, A = _random_csr(m, n, dtype, density=density, seed=m * 31 + n)
    rng = np.random.default_rng(m * 13 + n + 1)
    x = rounded(rng.standard_normal(m), dtype)
    y = to_wp(np.zeros(n), dtype)

    op = SparseMatVecProduct(mat, transa=True)
    op(to_wp(x, dtype), y, alpha=1.0, beta=0.0)
    np.testing.assert_allclose(host(y), A.toarray().T @ x, **MATVEC_TOL[dtype])


@pytest.mark.parametrize("m,n", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_alpha_beta(m: int, n: int, dtype) -> None:
    mat, A = _random_csr(m, n, dtype, seed=m * 17 + n + 2)
    rng = np.random.default_rng(m * 19 + n + 3)
    x = rounded(rng.standard_normal(n), dtype)
    y_before = rounded(rng.standard_normal(m), dtype)
    y = to_wp(y_before, dtype)

    op = SparseMatVecProduct(mat)
    op(to_wp(x, dtype), y, alpha=2.0, beta=0.5)
    np.testing.assert_allclose(
        host(y), 2.0 * (A.toarray() @ x) + 0.5 * y_before, **MATVEC_TOL[dtype],
    )


@pytest.mark.parametrize("m,n", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_alpha_beta_transpose(m: int, n: int, dtype) -> None:
    mat, A = _random_csr(m, n, dtype, seed=m * 23 + n + 4)
    rng = np.random.default_rng(m * 29 + n + 5)
    x = rounded(rng.standard_normal(m), dtype)
    y_before = rounded(rng.standard_normal(n), dtype)
    y = to_wp(y_before, dtype)

    op = SparseMatVecProduct(mat, transa=True)
    op(to_wp(x, dtype), y, alpha=-1.5, beta=2.0)
    np.testing.assert_allclose(
        host(y), -1.5 * (A.toarray().T @ x) + 2.0 * y_before, **MATVEC_TOL[dtype],
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_one_row_view_accepted(dtype) -> None:
    """A ``(1, k)`` row view of a wider buffer is accepted for x and y."""
    mat, A = _random_csr(4, 5, dtype, seed=3)
    big_x_h = rounded(np.random.default_rng(4).standard_normal((1, 9)), dtype)
    big_x, big_y = to_wp(big_x_h, dtype), to_wp(np.zeros((1, 7)), dtype)
    x_view, y_view = big_x[:, 2:7], big_y[:, 1:5]

    op = SparseMatVecProduct(mat)
    op(x_view, y_view)
    out = host(big_y)
    np.testing.assert_allclose(out[0, 1:5], A.toarray() @ big_x_h[0, 2:7], **MATVEC_TOL[dtype])
    np.testing.assert_allclose(out[0, 0], 0.0)
    np.testing.assert_allclose(out[0, 5:], 0.0)


@pytest.mark.parametrize("dtype", DTYPES)
def test_different_buffers(dtype) -> None:
    mat, A = _random_csr(3, 4, dtype)
    op = SparseMatVecProduct(mat)
    for x in (np.ones(4), np.arange(4, dtype=np.float64)):
        y = to_wp(np.zeros(3), dtype)
        op(to_wp(x, dtype), y)
        np.testing.assert_allclose(host(y), A.toarray() @ x, **MATVEC_TOL[dtype])


@pytest.mark.parametrize("dtype", DTYPES)
def test_reuse_different_scalars(dtype) -> None:
    mat, A = _random_csr(4, 4, dtype)
    x = np.arange(4, dtype=np.float64)
    x_d, y = to_wp(x, dtype), to_wp(np.zeros(4), dtype)

    op = SparseMatVecProduct(mat)
    op(x_d, y, alpha=1.0, beta=0.0)
    y1 = host(y)
    np.testing.assert_allclose(y1, A.toarray() @ x, **MATVEC_TOL[dtype])

    op(x_d, y, alpha=3.0, beta=1.0)
    np.testing.assert_allclose(host(y), 3.0 * (A.toarray() @ x) + y1, **MATVEC_TOL[dtype])


@pytest.mark.parametrize("n", [1, 2, 8, 64, 256])
@pytest.mark.parametrize("dtype", DTYPES)
def test_identity(n: int, dtype) -> None:
    A = sp.eye(n, dtype=np.float64, format="csr")
    mat = UniformBatchedCsrMatrix(1, *csr_triple(A, wp.dtype_to_numpy(dtype)), shape=A.shape, dtype=dtype)
    x = np.arange(n, dtype=np.float64)
    y = to_wp(np.zeros(n), dtype)

    op = SparseMatVecProduct(mat)
    op(to_wp(x, dtype), y)
    # Identity: every product has one nonzero term, so it is exact.
    np.testing.assert_allclose(host(y), x, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", DTYPES)
def test_in_place_value_update_is_seen(dtype) -> None:
    """The operator reads ``mat.data`` at every call."""
    mat, A = _random_csr(4, 5, dtype, seed=11)
    x = rounded(np.random.default_rng(12).standard_normal(5), dtype)
    y = to_wp(np.zeros(4), dtype)
    op = SparseMatVecProduct(mat)

    new_vals = rounded(np.random.default_rng(13).standard_normal(mat.nnz), dtype)
    wp.copy(mat.data, to_wp(new_vals.reshape(1, -1), dtype))
    op(to_wp(x, dtype), y)
    A2 = A.copy()
    A2.data[:] = new_vals
    np.testing.assert_allclose(host(y), A2.toarray() @ x, **MATVEC_TOL[dtype])


@pytest.mark.parametrize("dtype", DTYPES)
def test_cuda_graph_capture(dtype) -> None:
    mat, A = _random_csr(4, 4, dtype)
    x = np.ones(4)
    xw, yw = to_wp(x, dtype), to_wp(np.zeros(4), dtype)

    op = SparseMatVecProduct(mat)
    stream = wp.Stream("cuda")
    with wp.ScopedStream(stream):
        with wp.ScopedCapture(stream=stream) as capture:
            op(xw, yw, alpha=1.0, beta=0.0)
        wp.capture_launch(capture.graph, stream=stream)
    wp.synchronize_stream(stream)
    np.testing.assert_allclose(host(yw), A.toarray() @ x, **MATVEC_TOL[dtype])


@pytest.mark.parametrize("dtype", DTYPES)
def test_destructor_no_error(dtype) -> None:
    mat, _ = _random_csr(2, 2, dtype)
    op = SparseMatVecProduct(mat)
    del op  # must not raise


# ===========================================================================
# Batches of several (B > 1)
# ===========================================================================

@pytest.mark.parametrize("B", [1, 5])
@pytest.mark.parametrize("m,n", BATCHED_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_batched_basic_contiguous(B: int, m: int, n: int, dtype) -> None:
    bmat, tpl, vals = _random_batched_csr(B, m, n, dtype, density=0.4, seed=B * 100 + m * 10 + n)
    op = SparseMatVecProduct(bmat, transa=False)

    x = rounded(np.random.default_rng(B + m + n).standard_normal((B, n)), dtype)
    out = to_wp(np.zeros((B, m)), dtype)
    op(to_wp(x, dtype), out)
    np.testing.assert_allclose(
        host(out), _batched_ref(tpl, vals, x), **MATVEC_TOL[dtype],
    )


@pytest.mark.parametrize("B", [1, 5])
@pytest.mark.parametrize("m,n", BATCHED_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_batched_transpose(B: int, m: int, n: int, dtype) -> None:
    bmat, tpl, vals = _random_batched_csr(B, m, n, dtype, density=0.5, seed=B * 200 + m * 7 + n)
    op = SparseMatVecProduct(bmat, transa=True)

    x = rounded(np.random.default_rng(B * 3 + m + n).standard_normal((B, m)), dtype)
    out = to_wp(np.zeros((B, n)), dtype)
    op(to_wp(x, dtype), out)
    np.testing.assert_allclose(
        host(out), _batched_ref(tpl, vals, x, transpose=True), **MATVEC_TOL[dtype],
    )


@pytest.mark.parametrize("B", [1, 5])
@pytest.mark.parametrize("m,n", BATCHED_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_batched_alpha_beta(B: int, m: int, n: int, dtype) -> None:
    bmat, tpl, vals = _random_batched_csr(B, m, n, dtype, density=0.5, seed=B * 300 + m * 11 + n)
    op = SparseMatVecProduct(bmat)

    rng = np.random.default_rng(B * 5 + m + n + 1)
    x = rounded(rng.standard_normal((B, n)), dtype)
    y_init = rounded(rng.standard_normal((B, m)), dtype)
    y = to_wp(y_init, dtype)

    alpha, beta = 2.0, 3.0
    op(to_wp(x, dtype), y, alpha=alpha, beta=beta)
    expected = alpha * _batched_ref(tpl, vals, x) + beta * y_init
    np.testing.assert_allclose(host(y), expected, **MATVEC_TOL[dtype])


@pytest.mark.parametrize("dtype", DTYPES)
def test_batched_non_contiguous_x(dtype) -> None:
    B, m, n = 4, 5, 6
    K = 10
    col_start = 2
    bmat, tpl, vals = _random_batched_csr(B, m, n, dtype, seed=60)
    op = SparseMatVecProduct(bmat)

    big = rounded(np.random.default_rng(62).standard_normal((B, K)), dtype)
    x_view = to_wp(big, dtype)[:, col_start:col_start + n]
    assert not x_view.is_contiguous

    out = to_wp(np.zeros((B, m)), dtype)
    op(x_view, out)
    np.testing.assert_allclose(
        host(out), _batched_ref(tpl, vals, big[:, col_start:col_start + n]), **MATVEC_TOL[dtype],
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_batched_non_contiguous_out(dtype) -> None:
    B, m, n = 3, 4, 5
    K = 9
    col_start = 3
    bmat, tpl, vals = _random_batched_csr(B, m, n, dtype, seed=63)
    op = SparseMatVecProduct(bmat)

    x = rounded(np.random.default_rng(65).standard_normal((B, n)), dtype)
    big_out = to_wp(np.full((B, K), -7.0), dtype)  # sentinel outside the window
    out_view = big_out[:, col_start:col_start + m]
    assert not out_view.is_contiguous

    op(to_wp(x, dtype), out_view)
    out = host(big_out)
    np.testing.assert_allclose(
        out[:, col_start:col_start + m], _batched_ref(tpl, vals, x), **MATVEC_TOL[dtype],
    )
    # Columns outside the target window are untouched.
    np.testing.assert_allclose(out[:, :col_start], -7.0)
    np.testing.assert_allclose(out[:, col_start + m:], -7.0)


@pytest.mark.parametrize("dtype", DTYPES)
def test_batched_non_contiguous_out_with_beta(dtype) -> None:
    """With ``beta != 0`` the current values of a strided y are read first."""
    B, m, n = 3, 4, 5
    bmat, tpl, vals = _random_batched_csr(B, m, n, dtype, seed=66)
    op = SparseMatVecProduct(bmat)

    rng = np.random.default_rng(67)
    x = rounded(rng.standard_normal((B, n)), dtype)
    big_out_h = rounded(rng.standard_normal((B, 9)), dtype)
    out_view = to_wp(big_out_h, dtype)[:, 2:2 + m]
    y_init = big_out_h[:, 2:2 + m].copy()

    op(to_wp(x, dtype), out_view, alpha=1.0, beta=0.5)
    np.testing.assert_allclose(
        host(out_view), _batched_ref(tpl, vals, x) + 0.5 * y_init, **MATVEC_TOL[dtype],
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_batched_multiple_calls_different_x(dtype) -> None:
    B, m, n = 3, 4, 5
    bmat, tpl, vals = _random_batched_csr(B, m, n, dtype, seed=70)
    op = SparseMatVecProduct(bmat)

    x1 = rounded(np.random.default_rng(72).standard_normal((B, n)), dtype)
    x2 = rounded(np.random.default_rng(73).standard_normal((B, n)), dtype)
    out = wp.empty((B, m), dtype=dtype, device=DEVICE)

    op(to_wp(x1, dtype), out)
    np.testing.assert_allclose(
        host(out), _batched_ref(tpl, vals, x1), **MATVEC_TOL[dtype],
    )
    op(to_wp(x2, dtype), out)
    np.testing.assert_allclose(
        host(out), _batched_ref(tpl, vals, x2), **MATVEC_TOL[dtype],
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_batched_tracks_in_place_value_update(dtype) -> None:
    """After copying new values into ``bmat.data``, the SpMV uses them."""
    B, m, n = 3, 4, 5
    bmat, tpl, _ = _random_batched_csr(B, m, n, dtype, seed=74)
    op = SparseMatVecProduct(bmat)

    x = rounded(np.random.default_rng(76).standard_normal((B, n)), dtype)
    x_d = to_wp(x, dtype)
    out = wp.empty((B, m), dtype=dtype, device=DEVICE)
    op(x_d, out)

    new_vals = rounded(np.random.default_rng(77).standard_normal((B, tpl.nnz)), dtype)
    wp.copy(bmat.data, to_wp(new_vals, dtype))
    op(x_d, out)
    np.testing.assert_allclose(
        host(out), _batched_ref(tpl, new_vals, x), **MATVEC_TOL[dtype],
    )
