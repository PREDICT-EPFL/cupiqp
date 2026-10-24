"""DenseGemv / DenseSyrk against numpy, on both the Warp and the cuBLAS path."""
import numpy as np
import pytest
import warp as wp

from cupiqp.dense.dense_matmul import (
    WARP_MATMUL_MAX_COLS,
    WARP_MATMUL_MAX_ROWS,
    CublasHandle,
    DenseGemv,
    DenseSyrk,
)
from tests.helpers import to_wp

DTYPES = [(wp.float64, np.float64, 1e-12), (wp.float32, np.float32, 1e-4)]
# (rows, cols): small enough for the Warp kernels, and large enough for cuBLAS.
SHAPES = {
    "warp": (WARP_MATMUL_MAX_ROWS // 2, WARP_MATMUL_MAX_COLS // 2),
    "cublas": (WARP_MATMUL_MAX_ROWS + 8, WARP_MATMUL_MAX_COLS + 4),
    "zero_rows": (0, WARP_MATMUL_MAX_COLS + 4),
}


@pytest.mark.parametrize("wp_dtype,np_dtype,tol", DTYPES)
@pytest.mark.parametrize("shape", sorted(SHAPES))
@pytest.mark.parametrize("B", [1, 3])
@pytest.mark.parametrize("transpose", [False, True])
def test_gemv(wp_dtype, np_dtype, tol, shape, B, transpose):
    rows, cols = SHAPES[shape]
    rng = np.random.default_rng(0)
    M = rng.standard_normal((B, rows, cols)).astype(np_dtype)
    k_in, k_out = (rows, cols) if transpose else (cols, rows)
    x = rng.standard_normal((B, k_in)).astype(np_dtype)
    alpha = -0.5
    gemv = DenseGemv(B, rows, cols, wp_dtype, CublasHandle())
    z = wp.full((B, k_out), 7.0, dtype=wp_dtype, device="cuda")     # overwritten
    gemv(to_wp(M, wp_dtype), to_wp(x, wp_dtype), z, transpose=transpose, alpha=alpha)
    op = np.swapaxes(M, 1, 2) if transpose else M
    np.testing.assert_allclose(z.numpy(), alpha * np.einsum("bij,bj->bi", op, x),
                               rtol=tol, atol=tol * max(1, k_in))


@pytest.mark.parametrize("shape", ["warp", "cublas"])
@pytest.mark.parametrize("B", [1, 3])
def test_gemv_row_strided_vectors(shape, B):
    """x and z may be column slices of wider arrays, as in the KKT solver."""
    rows, cols = SHAPES[shape]
    rng = np.random.default_rng(1)
    M = rng.standard_normal((B, rows, cols))
    x_wide = to_wp(rng.standard_normal((B, cols + 5)), wp.float64)
    z_wide = wp.zeros((B, rows + 4), dtype=wp.float64, device="cuda")
    x, z = x_wide[:, 2:2 + cols], z_wide[:, 1:1 + rows]
    DenseGemv(B, rows, cols, wp.float64, CublasHandle())(to_wp(M, wp.float64), x, z)
    expected = np.einsum("bij,bj->bi", M, x_wide.numpy()[:, 2:2 + cols])
    out = z_wide.numpy()
    np.testing.assert_allclose(out[:, 1:1 + rows], expected, rtol=1e-12, atol=1e-11)
    assert not out[:, 0].any() and not out[:, 1 + rows:].any()       # outside the slice untouched


@pytest.mark.parametrize("wp_dtype,np_dtype,tol", DTYPES)
@pytest.mark.parametrize("shape", sorted(SHAPES))
@pytest.mark.parametrize("B", [1, 3])
@pytest.mark.parametrize("accumulate", [False, True])
def test_syrk_upper_triangle(wp_dtype, np_dtype, tol, shape, B, accumulate):
    rows, cols = SHAPES[shape]
    rng = np.random.default_rng(2)
    A = rng.standard_normal((B, rows, cols)).astype(np_dtype)
    C0 = rng.standard_normal((B, cols, cols)).astype(np_dtype)
    C = to_wp(C0, wp_dtype)
    DenseSyrk(B, rows, cols, wp_dtype, CublasHandle())(to_wp(A, wp_dtype), C, accumulate=accumulate)
    expected = np.einsum("bki,bkj->bij", A, A) + (C0 if accumulate else 0.0)
    upper = np.triu(np.ones((cols, cols), dtype=bool))
    np.testing.assert_allclose(C.numpy()[:, upper], expected[:, upper],
                               rtol=tol, atol=tol * max(1, rows))

