"""Device-resident runtime settings.

The IPM kernels read tolerances, regularization limits, ``tau`` and
``max_iter`` from a GPU settings buffer (``DeviceSettings``) that ``solve()`` refreshes from
``solver.settings``. A scalar launch argument would be frozen into any CUDA
graph captured by the first solve; reading from the buffer keeps later solves
of the same solver following the current settings. These tests change
settings between two solves of one solver, with graphs enabled, and compare
with a fresh solver built directly with the changed settings.
"""
import numpy as np
import pytest
import warp as wp

from cupiqp import DenseSolver, MultistageSolver, Status
from tests.helpers import primal_infeasible, random_qp, stack_qps, to_wp, to_wp_dict


def _batch(np_dtype, seed=0, size=4):
    """Three well-posed problems plus one primal infeasible problem, whose
    stagnation exercises the infeasibility and finetune thresholds."""
    rng = np.random.default_rng(seed)
    qps = [random_qp(rng) for _ in range(size - 1)]
    qps.append(primal_infeasible(qps[0]))
    return to_wp_dict(stack_qps(qps), np_dtype)


def _apply(settings, changes):
    for name, value in changes.items():
        setattr(settings, name, value)


def _solve_fresh(dtype, data, changes):
    solver = DenseSolver(dtype=dtype)
    solver.settings.enable_cuda_graph = True
    _apply(solver.settings, changes)
    solver.setup(**data)
    solver.solve()
    return solver


def _snapshot(solver):
    info = solver.result.info
    h = info.to_host()
    return (
        h.status, h.iter,
        solver.result.x.numpy().copy(), info.rho.numpy().copy(), info.delta.numpy().copy(),
    )


def _assert_same(a, b):
    assert a[0] == b[0]
    np.testing.assert_array_equal(a[1], b[1])
    for x, y in zip(a[2:], b[2:]):
        np.testing.assert_array_equal(x, y)


DTYPES = [(wp.float64, np.float64), (wp.float32, np.float32)]

# Each case changes settings that reach a different kernel: termination and
# rho/delta tolerances, the finetune limit, the step damping, the iteration
# limit, and the initial regularization used by the reset kernel.
CHANGES = [
    pytest.param({"eps_abs": 1e-3, "eps_rel": 1e-3, "eps_duality_gap_abs": 1e-3,
                  "eps_duality_gap_rel": 1e-3}, id="tolerances"),
    pytest.param({"check_duality_gap": False}, id="no_gap_check"),
    pytest.param({"tau": 0.5}, id="tau"),
    pytest.param({"max_iter": 4}, id="max_iter"),
    pytest.param({"rho_init": 1e-2, "delta_init": 1e-1}, id="rho_delta_init"),
    pytest.param({"reg_lower_limit": 1e-6, "reg_finetune_lower_limit": 1e-8}, id="reg_limits"),
    pytest.param({"infeasibility_threshold": 1e6}, id="infeasibility_threshold"),
    pytest.param({"reg_finetune_primal_update_threshold": 0,
                  "reg_finetune_dual_update_threshold": 0}, id="finetune_thresholds"),
]


@pytest.mark.parametrize("dtype,np_dtype", DTYPES)
@pytest.mark.parametrize("changes", CHANGES)
def test_settings_change_between_solves(dtype, np_dtype, changes):
    data = _batch(np_dtype)
    solver = DenseSolver(dtype=dtype)
    solver.settings.enable_cuda_graph = True
    solver.setup(**data)
    solver.solve()
    first = _snapshot(solver)

    _apply(solver.settings, changes)
    solver.solve()
    second = _snapshot(solver)
    assert second[0] != first[0] or not np.array_equal(second[1], first[1]) or \
        not np.array_equal(second[2], first[2]), "the changed settings had no effect"

    _assert_same(second, _snapshot(_solve_fresh(dtype, data, changes)))


def test_settings_change_back_restores_result():
    data = _batch(np.float64)
    solver = DenseSolver()
    solver.settings.enable_cuda_graph = True
    solver.setup(**data)
    solver.solve()
    first = _snapshot(solver)

    solver.settings.tau = 0.5
    solver.settings.eps_abs = 1e-3
    solver.solve()
    assert not np.array_equal(_snapshot(solver)[1], first[1])

    solver.settings.tau = 0.99
    solver.settings.eps_abs = 1e-8
    solver.solve()
    _assert_same(_snapshot(solver), first)


def test_max_iter_change_is_respected():
    solver = DenseSolver()
    solver.setup(**_batch(np.float64))
    solver.solve()
    assert solver.result.info.to_host().status[:3] == [Status.CUPIQP_SOLVED] * 3

    solver.settings.max_iter = 3
    solver.solve()
    h = solver.result.info.to_host()
    assert h.status == [Status.CUPIQP_MAX_ITER_REACHED] * 4
    np.testing.assert_array_equal(h.iter, 3)  # iter == max_iter at the limit


def test_multistage_tau_change():
    """The multistage backend shares the step-length kernel and the config."""
    P_diag = to_wp([[[4.0, 0.5], [0.5, 3.0]], [[5.0, 0.0], [0.0, 2.0]]])
    P_offdiag = to_wp([[[0.3, 0.1], [0.0, 0.2]]])
    c = to_wp([[1.0, -1.0, 0.5, -0.5], [2.0, 1.0, -0.5, 0.0], [-1.0, 0.0, 1.0, 3.0]])
    x_l, x_u = to_wp(-np.ones(4)), to_wp(np.ones(4))
    solver = MultistageSolver()
    solver.setup(P=(P_diag, P_offdiag), c=c, x_l=x_l, x_u=x_u)
    solver.solve()
    assert solver.result.info.to_host().status == [Status.CUPIQP_SOLVED] * 3
    iters_default = solver.result.info.to_host().iter

    solver.settings.tau = 0.3
    solver.solve()
    assert solver.result.info.to_host().status == [Status.CUPIQP_SOLVED] * 3
    iters_damped = solver.result.info.to_host().iter
    assert iters_damped.sum() > iters_default.sum()

    fresh = MultistageSolver()
    fresh.settings.tau = 0.3
    fresh.setup(P=(P_diag, P_offdiag), c=c, x_l=x_l, x_u=x_u)
    fresh.solve()
    np.testing.assert_array_equal(fresh.result.info.to_host().iter, iters_damped)
    np.testing.assert_array_equal(fresh.result.x.numpy(), solver.result.x.numpy())
