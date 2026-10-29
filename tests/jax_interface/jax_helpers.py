"""Shared helpers of the JAX interface tests.

Importing this module configures JAX for the tests (no up-front GPU memory
reservation, 64-bit types) and skips the importing test module when JAX is
not installed; ``requires_gpu`` skips it when JAX has no CUDA device.
"""
import os

import numpy as np
import pytest
import scipy.sparse as sp

# JAX reserves most of the GPU memory for itself when its backend starts;
# the core solver then cannot create its cuSOLVER / cuDSS handles. Leave the
# memory to be allocated on demand (must be set before JAX initializes).
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)

from cupiqp.jax import QPSolution  # noqa: E402

from tests.helpers import map_arrays  # noqa: E402
from tests.tolerances import FRAMEWORK_PARITY_TOL  # noqa: E402

requires_gpu = pytest.mark.skipif(
    not any(d.platform == "gpu" for d in jax.devices()), reason="a CUDA JAX device is required"
)

FIELDS = QPSolution._fields
VARIABLE_FIELDS = ("x", "y", "z_l", "z_u", "z_bl", "z_bu", "s_l", "s_u", "s_bl", "s_bu")
DATA_FIELDS = ("P", "c", "A", "b", "G", "h_u", "h_l", "x_u", "x_l")


# ---------------------------------------------------------------------------
# Problems and conversion
# ---------------------------------------------------------------------------

def to_jax(qp: dict, dtype=np.float64) -> dict:
    """The problem dict as JAX arrays of ``dtype`` (``None`` fields dropped)."""
    converted = map_arrays(qp, lambda v: jnp.asarray(np.asarray(v, dtype=dtype)))
    return {k: v for k, v in converted.items() if v is not None}


def patterns(qp: dict) -> dict:
    """The ``(indptr, indices)`` pattern of every sparse matrix of ``qp``."""
    return {name: (M.indptr, M.indices) for name, M in qp.items() if sp.issparse(M)}


def values(qp: dict) -> dict:
    """JAX arguments of ``qp``: the stored values of each matrix and the vectors."""
    out = {}
    for name, value in qp.items():
        data = value.data if sp.issparse(value) else np.asarray(value, dtype=np.float64)
        out[name] = jnp.asarray(data)
    return out


# ---------------------------------------------------------------------------
# References and comparisons
# ---------------------------------------------------------------------------

def core_result(solver) -> dict:
    """Every public field of the last solve of a core solver, as ``(B, ...)``
    NumPy arrays named like the ``QPSolution`` fields."""
    result = solver.result
    out = {name: getattr(result, name).numpy() for name in VARIABLE_FIELDS}
    out["status"] = result.info.status.numpy()
    out["iterations"] = result.info.iter.numpy()
    out["primal_res"] = result.info.primal_res.numpy()
    out["dual_res"] = result.info.dual_res.numpy()
    return out


def assert_close(sol: QPSolution, ref: dict, dtype=np.float64, unbatched=False) -> None:
    """``sol`` agrees with the core reference ``ref`` to the ``dtype``
    tolerance; statuses and iteration counts agree exactly."""
    for name in FIELDS:
        got = np.asarray(getattr(sol, name))
        expected = ref[name][0] if unbatched else ref[name]
        assert got.shape == expected.shape, name
        if name in ("status", "iterations"):
            np.testing.assert_array_equal(got, expected, err_msg=name)
        elif got.size:
            np.testing.assert_allclose(got, expected, err_msg=name, **FRAMEWORK_PARITY_TOL[np.dtype(dtype).type])


def assert_bitwise(a: QPSolution, b: QPSolution) -> None:
    for name in FIELDS:
        assert np.array_equal(np.asarray(getattr(a, name)), np.asarray(getattr(b, name))), name


def directional_fd(f, args: dict, name: str, direction: np.ndarray, eps=1e-6) -> float:
    """Central finite difference of ``f(**args)`` along ``direction`` in ``args[name]``."""
    plus = dict(args, **{name: args[name] + eps * direction})
    minus = dict(args, **{name: args[name] - eps * direction})
    return (float(f(**plus)) - float(f(**minus))) / (2 * eps)
