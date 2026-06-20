"""Batched block-bidiag SYRK / weighted SYRK against a per-batch numpy reference.

Both kernels compute ``A^T (W) A`` for a lower block-bidiagonal ``A``
defined by per-stage ``(D_k, E_k)`` blocks, and store the result back into
the (block-tridiagonal) ``(C_D, C_E)`` containers as
``C := alpha * (A^T (W) A)_block + beta * C``.

Each test is parametrized over a matrix of ``(B, N, m, c)`` shapes that
covers: minimum ``N = 2`` (so the off-diagonal ``C_E`` block exists),
``m > c``, ``m == c``, ``m < c``, and larger batch sizes.
"""
import numpy as np
import pytest
import warp as wp

from cupiqp.multistage.multistage_utils import BlockTridiagMat
from cupiqp.multistage.multistage_utils_kernels import (
    create_block_syrk_kernel,
    create_weighted_block_syrk_kernel,
)


@pytest.fixture(autouse=True, scope="module")
def _warp_init():
    wp.init()


# ----------------------------------------------------------------------
# Reference: dense block-bidiag assembly + numpy SYRK.
# ----------------------------------------------------------------------

def _build_dense_bidiag(D_np: np.ndarray, E_np: np.ndarray) -> np.ndarray:
    """Build the dense (N+1)*m x N*c block lower-bidiagonal matrix."""
    N, m, c = D_np.shape
    A = np.zeros(((N + 1) * m, N * c), dtype=np.float64)
    for k in range(N):
        A[k * m:(k + 1) * m,           k * c:(k + 1) * c] = D_np[k]
        A[(k + 1) * m:(k + 2) * m,     k * c:(k + 1) * c] = E_np[k]
    return A


def _gen_blocks(B, N, m, c, seed):
    rng = np.random.default_rng(seed)
    D_np = rng.standard_normal((B, N, m, c))
    E_np = rng.standard_normal((B, N, m, c))
    C_D_init = rng.standard_normal((B, N, c, c))
    C_E_init = rng.standard_normal((B, N - 1, c, c))
    return D_np, E_np, C_D_init, C_E_init


def _block_syrk_ref(D_np, E_np, C_D_init, C_E_init, alpha, beta, weights=None):
    """Per-batch reference for ``alpha * (A^T (W) A) + beta * C``, sliced
    into the relevant block-diagonal and block-subdiagonal positions."""
    B, N, m, c = D_np.shape
    C_D_ref = np.empty_like(C_D_init)
    C_E_ref = np.empty_like(C_E_init)
    for b in range(B):
        A_dense = _build_dense_bidiag(D_np[b], E_np[b])
        if weights is None:
            M = A_dense.T @ A_dense
        else:
            M = A_dense.T @ np.diag(weights[b]) @ A_dense
        for k in range(N):
            C_D_ref[b, k] = alpha * M[k * c:(k + 1) * c, k * c:(k + 1) * c] + beta * C_D_init[b, k]
        for k in range(N - 1):
            C_E_ref[b, k] = alpha * M[(k + 1) * c:(k + 2) * c, k * c:(k + 1) * c] + beta * C_E_init[b, k]
    return C_D_ref, C_E_ref


# ----------------------------------------------------------------------
# Size matrix.
#
# Both kernels are shape-specialized via ``create_*_block_syrk_kernel(N, m, c)``
# (one Warp compile per unique (N, m, c) triple), so we keep the matrix
# moderate but cover the interesting axes.
# ----------------------------------------------------------------------

SYRK_SIZES = [
    pytest.param( 1, 2,  4,  2, id="B1-N2-m4-c2"),     # smallest valid (N=2 → C_E exists)
    pytest.param( 1, 3,  8,  4, id="B1-N3-m8-c4"),
    pytest.param( 4, 5, 32, 12, id="B4-N5-m32-c12"),   # original test case
    pytest.param( 8, 6, 16,  6, id="B8-N6-m16-c6"),
    pytest.param( 2, 4,  8,  8, id="B2-N4-m8-c8"),     # square blocks: m == c
    pytest.param( 1, 3,  4,  6, id="B1-N3-m4-c6"),     # wide blocks: m < c
    pytest.param(16, 4, 12,  6, id="B16-N4-m12-c6"),   # larger batch
]


# ----------------------------------------------------------------------
# Block container tests.
# ----------------------------------------------------------------------

def test_block_tridiag_stores_arrays_directly():
    B, N, d = 2, 3, 2
    diag_np = np.arange(B * N * d * d, dtype=np.float64).reshape(B, N, d, d)
    off_diag_np = np.arange(
        B * (N - 1) * d * d, dtype=np.float64,
    ).reshape(B, N - 1, d, d)

    matrix = BlockTridiagMat(
        num_diag_blocks=N, block_size=d, batch_size=B,
    )
    matrix.D = wp.from_numpy(
        diag_np, dtype=wp.float64, device="cuda",
    )
    matrix.E = wp.from_numpy(
        off_diag_np, dtype=wp.float64, device="cuda",
    )

    assert tuple(matrix.D.shape) == (B, N, d, d)
    assert tuple(matrix.E.shape) == (B, N - 1, d, d)
    np.testing.assert_array_equal(matrix.D.numpy(), diag_np)
    np.testing.assert_array_equal(
        matrix.E.numpy(), off_diag_np,
    )

    cloned = matrix.clone()
    assert cloned.D.ptr != matrix.D.ptr
    assert (
        cloned.E.ptr
        != matrix.E.ptr
    )
    np.testing.assert_array_equal(cloned.D.numpy(), diag_np)
    np.testing.assert_array_equal(
        cloned.E.numpy(), off_diag_np,
    )

    with pytest.raises(ValueError, match="diag_blocks has shape"):
        matrix.D = wp.zeros(
            (B, N - 1, d, d), dtype=wp.float64, device="cuda",
        )


# ----------------------------------------------------------------------
# Tests.
# ----------------------------------------------------------------------

@pytest.mark.parametrize("B,N,m,c", SYRK_SIZES)
def test_block_syrk(B, N, m, c):
    alpha, beta = 2.5, -0.3
    seed = B * 1000 + N * 100 + m * 10 + c

    D_np, E_np, C_D_init, C_E_init = _gen_blocks(B, N, m, c, seed=seed)
    C_D_ref, C_E_ref = _block_syrk_ref(D_np, E_np, C_D_init, C_E_init, alpha, beta)

    kernel = create_block_syrk_kernel(N, m, c)
    A_D = wp.from_numpy(D_np,           dtype=wp.float64, device="cuda")
    A_E = wp.from_numpy(E_np,           dtype=wp.float64, device="cuda")
    C_D = wp.from_numpy(C_D_init.copy(), dtype=wp.float64, device="cuda")
    C_E = wp.from_numpy(C_E_init.copy(), dtype=wp.float64, device="cuda")

    wp.launch(kernel, dim=(B, N, c, c),
              inputs=[alpha, A_D, A_E, beta, C_D, C_E], device="cuda")

    np.testing.assert_allclose(C_D.numpy(), C_D_ref, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(C_E.numpy(), C_E_ref, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("B,N,m,c", SYRK_SIZES)
def test_weighted_block_syrk(B, N, m, c):
    alpha, beta = 2.5, -0.3
    seed = B * 1000 + N * 100 + m * 10 + c + 7

    D_np, E_np, C_D_init, C_E_init = _gen_blocks(B, N, m, c, seed=seed)
    w_np = np.random.default_rng(seed + 4).standard_normal((B, (N + 1) * m))
    C_D_ref, C_E_ref = _block_syrk_ref(D_np, E_np, C_D_init, C_E_init,
                                       alpha, beta, weights=w_np)

    kernel = create_weighted_block_syrk_kernel(N, m, c)
    A_D = wp.from_numpy(D_np,           dtype=wp.float64, device="cuda")
    A_E = wp.from_numpy(E_np,           dtype=wp.float64, device="cuda")
    w   = wp.from_numpy(w_np,           dtype=wp.float64, device="cuda")
    C_D = wp.from_numpy(C_D_init.copy(), dtype=wp.float64, device="cuda")
    C_E = wp.from_numpy(C_E_init.copy(), dtype=wp.float64, device="cuda")

    wp.launch(kernel, dim=(B, N, c, c),
              inputs=[alpha, A_D, A_E, w, beta, C_D, C_E], device="cuda")

    np.testing.assert_allclose(C_D.numpy(), C_D_ref, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(C_E.numpy(), C_E_ref, rtol=1e-10, atol=1e-10)


# ----------------------------------------------------------------------
# Scalar-coefficient edge cases at a single size: alpha=0 (pure scale of C),
# beta=0 (overwrite, no accumulation), alpha=1 beta=1 (additive accumulate).
# These hit the kernel's coefficient handling without paying for many extra
# shape-specialized compiles.
# ----------------------------------------------------------------------

ALPHA_BETA_CASES = [
    pytest.param(0.0,  1.0, id="alpha0-beta1-(C-unchanged)"),
    pytest.param(1.0,  0.0, id="alpha1-beta0-(overwrite)"),
    pytest.param(1.0,  1.0, id="alpha1-beta1-(accumulate)"),
    pytest.param(2.5, -0.3, id="general"),
    pytest.param(-1.0, 2.0, id="negative-alpha"),
]


@pytest.mark.parametrize("alpha,beta", ALPHA_BETA_CASES)
def test_block_syrk_alpha_beta(alpha, beta):
    B, N, m, c = 2, 4, 8, 4
    D_np, E_np, C_D_init, C_E_init = _gen_blocks(B, N, m, c, seed=2024)
    C_D_ref, C_E_ref = _block_syrk_ref(D_np, E_np, C_D_init, C_E_init, alpha, beta)

    kernel = create_block_syrk_kernel(N, m, c)
    A_D = wp.from_numpy(D_np,           dtype=wp.float64, device="cuda")
    A_E = wp.from_numpy(E_np,           dtype=wp.float64, device="cuda")
    C_D = wp.from_numpy(C_D_init.copy(), dtype=wp.float64, device="cuda")
    C_E = wp.from_numpy(C_E_init.copy(), dtype=wp.float64, device="cuda")

    wp.launch(kernel, dim=(B, N, c, c),
              inputs=[alpha, A_D, A_E, beta, C_D, C_E], device="cuda")

    np.testing.assert_allclose(C_D.numpy(), C_D_ref, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(C_E.numpy(), C_E_ref, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("alpha,beta", ALPHA_BETA_CASES)
def test_weighted_block_syrk_alpha_beta(alpha, beta):
    B, N, m, c = 2, 4, 8, 4
    D_np, E_np, C_D_init, C_E_init = _gen_blocks(B, N, m, c, seed=2025)
    w_np = np.random.default_rng(2025 + 1).standard_normal((B, (N + 1) * m))
    C_D_ref, C_E_ref = _block_syrk_ref(D_np, E_np, C_D_init, C_E_init,
                                       alpha, beta, weights=w_np)

    kernel = create_weighted_block_syrk_kernel(N, m, c)
    A_D = wp.from_numpy(D_np,           dtype=wp.float64, device="cuda")
    A_E = wp.from_numpy(E_np,           dtype=wp.float64, device="cuda")
    w   = wp.from_numpy(w_np,           dtype=wp.float64, device="cuda")
    C_D = wp.from_numpy(C_D_init.copy(), dtype=wp.float64, device="cuda")
    C_E = wp.from_numpy(C_E_init.copy(), dtype=wp.float64, device="cuda")

    wp.launch(kernel, dim=(B, N, c, c),
              inputs=[alpha, A_D, A_E, w, beta, C_D, C_E], device="cuda")

    np.testing.assert_allclose(C_D.numpy(), C_D_ref, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(C_E.numpy(), C_E_ref, rtol=1e-10, atol=1e-10)
