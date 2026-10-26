"""Finite-difference tests for the multistage solver backward pass."""

import numpy as np
import pytest

from cupiqp import MultistageSolver
from cupiqp.results import Status
from tests.helpers import to_wp


def _solve_two_stage_qp(
    offdiag: float, *, enable_grad: bool, preconditioner_iter: int
):
    P_diag = to_wp(np.array([[[2.0]], [[3.0]]]))   # (N=2, d=1, d=1)
    P_offdiag = to_wp(np.array([[[offdiag]]]))     # (N-1=1, 1, 1)
    c = to_wp(np.array([[1.0], [-0.5]]))           # (N, d)

    solver = MultistageSolver()
    solver.settings.enable_cuda_graph = False
    solver.settings.enable_grad = enable_grad
    solver.settings.preconditioner_iter = preconditioner_iter
    solver.settings.eps_abs = 1e-11
    solver.settings.eps_rel = 1e-11
    solver.setup(P=(P_diag, P_offdiag), c=c)
    solver.solve()
    assert solver.result.info.to_host().status == [Status.CUPIQP_SOLVED]
    return solver


@pytest.mark.parametrize("preconditioner_iter", [0, 10])
def test_offdiag_hessian_gradient_counts_implicit_transpose(preconditioner_iter):
    solver = _solve_two_stage_qp(
        0.25, enable_grad=True, preconditioner_iter=preconditioner_iter
    )
    grad = solver.backward(grad_x=to_wp(np.ones((1, 2), dtype=np.float64)))
    analytic = float(grad.P_offdiag.numpy()[0, 0, 0, 0])

    eps = 1e-5
    plus = _solve_two_stage_qp(
        0.25 + eps, enable_grad=False, preconditioner_iter=preconditioner_iter
    )
    minus = _solve_two_stage_qp(
        0.25 - eps, enable_grad=False, preconditioner_iter=preconditioner_iter
    )
    finite_diff = float((plus.result.x.numpy().sum() - minus.result.x.numpy().sum()) / (2.0 * eps))

    np.testing.assert_allclose(analytic, finite_diff, rtol=1e-5, atol=1e-7)
