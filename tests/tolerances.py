"""Named tolerance policies of the test suite.

Each policy is a dict from a float type to the ``atol`` / ``rtol`` keywords
of ``np.testing.assert_allclose``:

    np.testing.assert_allclose(actual, expected, **MATVEC_TOL[dtype])

A policy lives here when it names an error model several checks share. A
tolerance a single check needs beyond its policy stays inline at that check,
with a comment saying why.
"""
import numpy as np
import warp as wp

# Matrix-vector (and vector) products against the float64 NumPy product of
# the same values: the only error is the rounding of the accumulation.
MATVEC_TOL = {
    wp.float64: dict(atol=1e-12, rtol=1e-10),
    wp.float32: dict(atol=1e-5, rtol=1e-5),
}

# Dense GEMV / SYRK against einsum, per product term: the absolute error of
# a dot product grows with its length, so atol is scaled by the inner
# dimension at each check.
MATMUL_TOL = {
    wp.float64: dict(atol=1e-12, rtol=1e-12),
    wp.float32: dict(atol=1e-6, rtol=1e-6),
}

# Residual A x - b of a Cholesky solve of a well conditioned SPD matrix.
CHOLESKY_RESIDUAL_TOL = {
    wp.float64: dict(atol=1e-10, rtol=1e-10),
    wp.float32: dict(atol=1e-4, rtol=1e-4),
}

# A KKT solution against a float64 NumPy solve of the same (well
# conditioned, regularized) system: forward error of the factorization.
KKT_SOLVE_TOL = {
    wp.float64: dict(atol=1e-9, rtol=1e-9),
    wp.float32: dict(atol=1e-4, rtol=1e-4),
}

# The residual round trip K_c solve(rhs) = rhs through mul_condensed_kkt:
# backward error of the solve plus the rounding of the matvec.
KKT_RESIDUAL_TOL = {
    wp.float64: dict(atol=1e-8, rtol=1e-8),
    wp.float32: dict(atol=1e-4, rtol=1e-4),
}

# A framework (JAX) result against the core solver on the same problem: the
# same kernels run, so only reduction order may differ. Keyed by NumPy types,
# the dtypes the JAX tests use.
FRAMEWORK_PARITY_TOL = {
    np.float64: dict(atol=1e-10, rtol=1e-9),
    np.float32: dict(atol=1e-4, rtol=1e-4),
}
