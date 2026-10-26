"""backward() with cotangents on the slack variables, against central finite
differences, for every backend.

Regression: the transposed KKT solve recovered the box duals from the
forward slack elimination ``rhs.z - rhs.s / Z`` instead of the transposed
``rhs.z - (S / Z) rhs.s``, so a loss on ``s_bl`` / ``s_bu`` got wrong
gradients with respect to ``x_l`` / ``x_u`` (off by orders of magnitude at an
active bound). Losses without slack cotangents were not affected."""

import numpy as np
import pytest
import warp as wp

import cupiqp
from cupiqp import Status

from tests.helpers import csr_triple, random_qp, to_wp


SLACKS_AND_INPUTS = [
    ("s_bl", "x_l"), ("s_bu", "x_u"), ("s_l", "h_l"), ("s_u", "h_u"),
    ("s_bl", "c"), ("s_bu", "b"), ("s_bu", "x_l"),
]


def active_box_qp(seed=0, n=6, p=2, m=4) -> dict:
    """A random QP whose solution has an active lower and an active upper box
    bound (strictly complementary), so that the slack gradients at an active
    bound are exercised."""
    qp = random_qp(seed, n=n, p=p, m=m)
    P, c, A = qp["P"], qp["c"], qp["A"]
    kkt = np.block([[P, A.T], [A, np.zeros((p, p))]])
    x_star = np.linalg.solve(kkt, np.concatenate([-c, qp["b"]]))[:n]
    qp["x_u"][0] = x_star[0] - 0.2
    qp["x_l"][1] = x_star[1] + 0.2
    return qp


def make_solver(cls, qp, preconditioner_iter):
    solver = cls()
    solver.settings.enable_grad = True
    solver.settings.eps_abs = solver.settings.eps_rel = 1e-11
    solver.settings.preconditioner_iter = preconditioner_iter
    to_dev = csr_triple if cls is cupiqp.SparseSolver else to_wp
    solver.setup(**{k: (to_dev(v) if k in ("P", "A", "G") else to_wp(v)) for k, v in qp.items()})
    solver.solve()
    assert all(s == Status.CUPIQP_SOLVED for s in solver.result.info.to_host().status)
    return solver


@pytest.mark.parametrize("preconditioner_iter", [0, 10])
@pytest.mark.parametrize("cls", [cupiqp.DenseSolver, cupiqp.SparseSolver])
@pytest.mark.parametrize("field,name", SLACKS_AND_INPUTS)
def test_slack_cotangent_matches_finite_differences(cls, field, name, preconditioner_iter):
    qp = active_box_qp(3)
    rng = np.random.default_rng(hash((field, name)) % 2**32)
    solver = make_solver(cls, qp, preconditioner_iter)
    weights = rng.standard_normal(getattr(solver.result, field).shape)
    gradients = solver.backward(**{f"grad_{field}": to_wp(weights)})
    gradient = getattr(gradients, name)
    gradient = gradient.data.numpy() if hasattr(gradient, "indptr") else gradient.numpy()

    direction = rng.standard_normal(np.shape(qp[name]))
    eps = 1e-6

    def loss(values):
        perturbed = make_solver(cls, dict(qp, **{name: values}), preconditioner_iter)
        return float(np.sum(getattr(perturbed.result, field).numpy() * weights))

    numeric = (loss(qp[name] + eps * direction) - loss(qp[name] - eps * direction)) / (2 * eps)
    analytic = float(np.sum(gradient[0] * direction))
    np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-7)
