"""Tests for cupiqp.torch.DenseSolverLayer: public contract, forward agreement
with the core solver, finite-difference checks of the implicit-diff backward
for every input field, and the pending-solve policy (max_pending_solves) for
rollouts."""

import gc

import numpy as np
import pytest
import torch
import warp as wp

import cupiqp
from cupiqp import Status
from cupiqp.torch import DenseSolverLayer, StructureMismatchError, TorchSolution


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")

DEVICE = "cuda"
DTYPE = torch.float64


def _t(values, dtype=DTYPE):
    return torch.tensor(values, dtype=dtype, device=DEVICE)


def _statuses(layer):
    """Host tuple of per-problem Status enums (explicit device-to-host read)."""
    return tuple(Status(int(v)) for v in layer.result.status.cpu().tolist())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _rand_qp(n=4, p=1, m=2, seed=0, with_bounds=True, dtype=DTYPE):
    """A random strictly convex QP with a non-empty interior, as torch tensors."""
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n))
    P = M @ M.T + n * np.eye(n)
    c = rng.standard_normal(n)
    A = rng.standard_normal((p, n)) if p > 0 else None
    b = rng.standard_normal(p) if p > 0 else None
    G = rng.standard_normal((m, n)) if m > 0 else None
    if m > 0:
        mid = rng.standard_normal(m)
        rad = 1.0 + rng.random(m)
        h_l, h_u = mid - rad, mid + rad
    else:
        h_l = h_u = None
    if with_bounds:
        mid = rng.standard_normal(n)
        rad = 2.0 + rng.random(n)
        x_l, x_u = mid - rad, mid + rad
    else:
        x_l = x_u = None

    def t(a):
        return None if a is None else torch.tensor(a, dtype=dtype, device=DEVICE)

    return dict(P=t(P), c=t(c), A=t(A), b=t(b), G=t(G), h_u=t(h_u), h_l=t(h_l),
                x_u=t(x_u), x_l=t(x_l))


def _box_qp(dtype=DTYPE):
    """Box QP with the interior solution x = (0.5, 0.3)."""
    return dict(
        P=torch.eye(2, dtype=dtype, device=DEVICE),
        c=_t([-0.5, -0.3], dtype),
        x_l=torch.zeros(2, dtype=dtype, device=DEVICE),
        x_u=torch.ones(2, dtype=dtype, device=DEVICE),
    )


BOX_X = [[0.5, 0.3]]


def _make_layer(preconditioner_iter=0, **settings):
    layer = DenseSolverLayer()
    layer.settings.preconditioner_iter = preconditioner_iter
    layer.settings.verbose = False
    layer.settings.eps_abs = 1e-11
    layer.settings.eps_rel = 1e-11
    for name, value in settings.items():
        setattr(layer.settings, name, value)
    return layer


def _solve(layer, qp):
    """Set up on first use, then solve with every tensor passed to the call so
    all of them can receive gradients."""
    if not layer.is_setup:
        layer.setup(**qp)
    layer(**qp)
    return layer.result


def _loss(sol: TorchSolution, kind: str) -> torch.Tensor:
    if kind == "x":
        return sol.x.sum()
    if kind == "x+y+zu":
        out = sol.x.sum()
        if sol.y is not None:
            out = out + sol.y.sum()
        if sol.z_u is not None:
            out = out + sol.z_u.sum()
        return out
    if kind == "x*su+y":
        out = sol.x.sum()
        if sol.s_u is not None:
            out = out + sol.x.sum() * sol.s_u.sum()
        if sol.y is not None:
            out = out + sol.y.sum()
        return out
    raise ValueError(kind)


def _analytic_grad(param, layer, qp, kind):
    param.requires_grad_(True)
    param.grad = None
    sol = _solve(layer, qp)
    _loss(sol, kind).backward()
    grad = param.grad.detach().clone()
    param.requires_grad_(False)
    param.grad = None
    return grad


def _fd_grad(param, layer, qp, kind, eps=1e-5):
    base = param.detach().clone()
    grad = torch.zeros_like(base)
    flat = grad.view(-1)
    for i in range(flat.numel()):
        bump = torch.zeros_like(base).view(-1)
        bump[i] = eps
        bump = bump.view(base.shape)
        param.data.copy_(base + bump)
        layer.reset()
        loss_plus = float(_loss(_solve(layer, qp), kind))
        param.data.copy_(base - bump)
        layer.reset()
        loss_minus = float(_loss(_solve(layer, qp), kind))
        flat[i] = (loss_plus - loss_minus) / (2 * eps)
    param.data.copy_(base)
    layer.reset()
    return grad


def _check_all_fields(qp, kind="x", note=""):
    layer = _make_layer()
    for name, param in qp.items():
        if param is None:
            continue
        analytic = _analytic_grad(param, layer, qp, kind)
        assert torch.isfinite(analytic).all(), f"{name}: non-finite gradient"
        fd = _fd_grad(param, layer, qp, kind)
        if name == "P":
            fd = 0.5 * (fd + fd.transpose(-1, -2))   # analytic P gradient is symmetrized
        assert torch.allclose(analytic, fd, atol=5e-4, rtol=5e-4), (
            f"{note}{kind}: {name} analytic vs FD mismatch\n"
            f"  analytic: {analytic}\n  fd      : {fd}\n"
            f"  max |err|: {(analytic - fd).abs().max()}"
        )


# ---------------------------------------------------------------------------
# Namespace and construction
# ---------------------------------------------------------------------------
def test_namespace_is_lazy_and_exposes_layers():
    import importlib
    module = importlib.import_module("cupiqp")
    assert module.torch.DenseSolverLayer is DenseSolverLayer
    assert isinstance(DenseSolverLayer(), torch.nn.Module)


def test_constructor_validates_dtype_and_stream():
    with pytest.raises(TypeError, match="dtype"):
        DenseSolverLayer(dtype=torch.float16)
    with pytest.raises(TypeError, match="stream"):
        DenseSolverLayer(stream="not a stream")


def test_settings_follow_layer_dtype():
    layer = DenseSolverLayer(dtype=torch.float32)
    assert layer.settings.enable_grad is True
    with pytest.raises(ValueError, match="dtype"):
        layer.settings = cupiqp.Settings()           # float64 settings
    layer.settings = cupiqp.Settings.for_dtype(wp.float32)
    assert layer.settings.dtype is wp.float32


def test_device_argument_and_stream_device():
    qp = _box_qp()
    layer = DenseSolverLayer(device="cuda")
    assert layer.device == torch.device("cuda", torch.cuda.current_device())
    layer.settings.verbose = False
    layer.setup(**qp)
    torch.testing.assert_close(layer(), _t(BOX_X))
    with pytest.raises(ValueError, match="CUDA"):
        DenseSolverLayer(device="cpu")
    stream = torch.cuda.Stream()
    layer = DenseSolverLayer(stream=stream)
    assert layer.device == stream.device


# ---------------------------------------------------------------------------
# Forward contract: always batched, views, no allocation
# ---------------------------------------------------------------------------
def test_forward_returns_batched_view_and_result():
    qp = _box_qp()
    layer = _make_layer()
    layer.setup(**qp)
    assert layer.batch_size == 1
    x = layer()
    assert isinstance(x, torch.Tensor) and x.is_cuda and x.dtype == DTYPE
    assert x.shape == (1, 2)
    torch.testing.assert_close(x, _t(BOX_X))
    assert x.data_ptr() == layer.result.x.data_ptr()        # a view, no copy
    assert _statuses(layer) == (Status.CUPIQP_SOLVED,)
    assert layer.result.status.shape == (1,)
    assert int(layer.result.iterations[0]) > 0
    assert layer.result.y is None and layer.result.z_u is None
    assert layer.result.z_bl.shape == (1, 2) and layer.result.s_bu.shape == (1, 2)


def test_forward_and_backward_allocate_nothing():
    qp = _box_qp()
    layer = _make_layer()
    c = qp["c"].clone().requires_grad_(True)
    layer.setup(**qp)
    layer(c=c).sum().backward()               # warm up: torch creates c.grad
    layer(**qp)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    x = layer(c=c)
    x.sum().backward()
    with torch.no_grad():
        layer(**qp)
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == before


def test_outputs_are_views_overwritten_by_the_next_call():
    qp = _box_qp()
    layer = _make_layer()
    layer.setup(**qp)
    x1 = layer()
    kept = x1.clone()
    x2 = layer(c=-qp["c"])
    torch.testing.assert_close(kept, _t(BOX_X))
    torch.testing.assert_close(x2, _t([[0.0, 0.0]]))
    assert x1.data_ptr() == x2.data_ptr()


def test_forward_matches_core_solver_batched():
    batch_size, n = 3, 4
    rng = np.random.default_rng(5)
    P = _t(np.stack([(lambda M: M @ M.T + n * np.eye(n))(rng.standard_normal((n, n)))
                     for _ in range(batch_size)]))
    c = _t(rng.standard_normal((batch_size, n)))
    layer = _make_layer()
    layer.setup(P, c)
    x = layer()
    assert layer.batch_size == batch_size and x.shape == (batch_size, n)
    assert layer.result.status.shape == (batch_size,)
    assert _statuses(layer) == (Status.CUPIQP_SOLVED,) * batch_size
    core = cupiqp.DenseSolver()
    core.settings = layer.settings
    core.setup(P, c)
    core.solve()
    torch.testing.assert_close(x, torch.from_dlpack(core.result.x))


def test_shared_matrix_with_batched_vector():
    n, batch_size = 3, 4
    P = torch.eye(n, dtype=DTYPE, device=DEVICE)
    c = torch.arange(batch_size * n, dtype=DTYPE, device=DEVICE).reshape(batch_size, n)
    layer = _make_layer()
    layer.setup(P, c)
    torch.testing.assert_close(layer(c=c), -c)
    torch.testing.assert_close(layer(c=c[0]), -c[0].expand(batch_size, n))


def test_float32():
    qp = _box_qp(dtype=torch.float32)
    layer = DenseSolverLayer(dtype=torch.float32)
    layer.settings.verbose = False
    layer.setup(**qp)
    x = layer()
    assert x.dtype == torch.float32
    torch.testing.assert_close(x, _t(BOX_X, torch.float32), atol=1e-4, rtol=0)


def test_non_default_torch_stream_constructor_and_current():
    qp = _box_qp()
    stream = torch.cuda.Stream()
    layer = DenseSolverLayer(stream=stream)
    layer.settings.verbose = False
    with torch.cuda.stream(stream):
        c = qp["c"] * 0.5                      # interior solution (0.25, 0.15)
        layer.setup(**qp)
        x = layer(c=c)
    stream.synchronize()
    torch.testing.assert_close(x, _t([[0.25, 0.15]]))
    layer2 = _make_layer()
    with torch.cuda.stream(torch.cuda.Stream()):
        layer2.setup(**qp)
        x2 = layer2()
    torch.cuda.synchronize()
    torch.testing.assert_close(x2, _t(BOX_X))


def test_result_clone_survives_next_call():
    qp = _box_qp()
    layer = _make_layer()
    layer.setup(**qp)
    layer()
    kept = layer.result.clone()
    layer(c=-qp["c"])
    torch.testing.assert_close(kept.x, _t(BOX_X))
    assert Status(int(kept.status[0])) == Status.CUPIQP_SOLVED


def test_non_converged_status_is_reported_not_raised():
    qp = _box_qp()
    layer = _make_layer(max_iter=1)
    layer.setup(**qp)
    layer()
    assert _statuses(layer) == (Status.CUPIQP_MAX_ITER_REACHED,)


# ---------------------------------------------------------------------------
# Structure and lifecycle errors
# ---------------------------------------------------------------------------
def test_forward_before_setup_raises():
    with pytest.raises(RuntimeError, match="setup"):
        _make_layer()()


def test_setup_twice_requires_reset():
    qp = _box_qp()
    layer = _make_layer()
    layer.setup(**qp)
    with pytest.raises(RuntimeError, match="reset"):
        layer.setup(**qp)
    layer.reset()
    assert not layer.is_setup
    layer.setup(**qp)
    torch.testing.assert_close(layer(), _t(BOX_X))


def test_structure_mismatch_errors():
    qp = _box_qp()
    layer = _make_layer()
    layer.setup(**qp)
    with pytest.raises(StructureMismatchError, match="shape"):
        layer(c=torch.zeros(3, dtype=DTYPE, device=DEVICE))
    with pytest.raises(StructureMismatchError, match="not given at setup"):
        layer(G=torch.eye(2, dtype=DTYPE, device=DEVICE), h_u=torch.ones(2, dtype=DTYPE, device=DEVICE))
    assert isinstance(StructureMismatchError("x"), ValueError)


def test_input_validation():
    qp = _box_qp()
    layer = _make_layer()
    with pytest.raises(ValueError, match="dtype"):
        layer.setup(qp["P"].float(), qp["c"])
    with pytest.raises(ValueError, match="cuda"):
        layer.setup(qp["P"].cpu(), qp["c"])
    with pytest.raises(TypeError, match="torch.Tensor"):
        layer.setup(np.eye(2), qp["c"])
    with pytest.raises(ValueError, match="square"):
        layer.setup(torch.ones(2, 3, dtype=DTYPE, device=DEVICE), qp["c"])


@pytest.mark.parametrize("bad", [0, -1, True, False, 2.0, "2", None])
def test_max_pending_solves_must_be_a_positive_integer(bad):
    qp = _box_qp()
    layer = _make_layer()
    with pytest.raises(ValueError, match="max_pending_solves must be a positive integer"):
        layer.setup(qp["P"], qp["c"], max_pending_solves=bad, per_call_fields=("c",))
    assert not layer.is_setup


def test_per_call_fields_rules():
    qp = _box_qp()
    layer = _make_layer()
    with pytest.raises(ValueError, match="per_call_fields is only used"):
        layer.setup(qp["P"], qp["c"], per_call_fields=("c",))          # capacity one
    with pytest.raises(ValueError, match="requires per_call_fields"):
        layer.setup(qp["P"], qp["c"], max_pending_solves=2)
    with pytest.raises(ValueError, match="per_call_fields must name"):
        layer.setup(qp["P"], qp["c"], max_pending_solves=2, per_call_fields=("G",))
    with pytest.raises(ValueError, match="per_call_fields must name"):
        layer.setup(qp["P"], qp["c"], max_pending_solves=2, per_call_fields=("q",))
    layer.settings.enable_grad = False
    with pytest.raises(ValueError, match="enable_grad"):
        layer.setup(qp["P"], qp["c"], max_pending_solves=2, per_call_fields=("c",))
    layer.settings.enable_grad = True
    layer.setup(qp["P"], qp["c"], max_pending_solves=2, per_call_fields=("c",))
    assert layer.max_pending_solves == 2


def test_removed_backward_mode_arguments_are_rejected():
    qp = _box_qp()
    layer = _make_layer()
    with pytest.raises(TypeError):
        layer.setup(qp["P"], qp["c"], backward_mode="refactorize")
    with pytest.raises(TypeError):
        layer.setup(qp["P"], qp["c"], checkpoint_capacity=2)


def test_capacity_one_pending_solve_blocks_further_calls():
    qp = _box_qp()
    layer = _make_layer()
    layer.setup(**qp)
    c = qp["c"].clone().requires_grad_(True)
    x = layer(c=c)
    with pytest.raises(RuntimeError, match="pending backward"):
        layer(c=c)
    with pytest.raises(RuntimeError, match="pending backward"):
        layer.reset()
    x.sum().backward()
    torch.testing.assert_close(c.grad, -torch.ones(2, dtype=DTYPE, device=DEVICE))
    layer(c=c).sum().backward()
    layer.reset()
    assert not layer.is_setup


def test_dropping_the_output_releases_the_lease():
    qp = _box_qp()
    layer = _make_layer()
    layer.setup(**qp)
    c = qp["c"].clone().requires_grad_(True)
    x = layer(c=c)
    loss = x.sum()
    with pytest.raises(RuntimeError, match="pending backward"):
        layer(c=c)
    del x, loss
    gc.collect()
    x = layer(c=c)
    x.sum().backward()
    torch.testing.assert_close(c.grad, -torch.ones(2, dtype=DTYPE, device=DEVICE))


def test_backward_twice_through_one_solve_raises():
    qp = _box_qp()
    layer = _make_layer()
    layer.setup(**qp)
    c = qp["c"].clone().requires_grad_(True)
    x = layer(c=c)
    x.sum().backward(retain_graph=True)
    with pytest.raises(RuntimeError, match="twice"):
        x.sum().backward()


def test_requires_grad_with_enable_grad_off_raises():
    qp = _box_qp()
    layer = _make_layer(enable_grad=False)
    layer.setup(**qp)
    c = qp["c"].clone().requires_grad_(True)
    with pytest.raises(RuntimeError, match="enable_grad"):
        layer(c=c)
    with torch.no_grad():
        layer(c=c)


def test_structural_settings_are_frozen_at_setup():
    """The core solver freezes these settings at setup() and rejects a change
    at the next call; the layer relies on it."""
    qp = _box_qp()
    layer = _make_layer()
    layer.setup(**qp)
    layer.settings.max_iter = 77                 # numerical: allowed
    layer()
    for name, changed in (("enable_grad", False),
                          ("gradient_smoothing", not layer.settings.gradient_smoothing),
                          ("preconditioner_iter", 7)):
        original = getattr(layer.settings, name)
        setattr(layer.settings, name, changed)
        with pytest.raises(RuntimeError, match=rf"\['{name}'\] changed after setup\(\)"):
            layer()
        with pytest.raises(RuntimeError, match="changed after setup"):
            layer(c=qp["c"])
        setattr(layer.settings, name, original)
    layer()                                      # restored: fine again
    c = qp["c"].clone().requires_grad_(True)
    layer.settings.gradient_smoothing = not layer.settings.gradient_smoothing
    with pytest.raises(RuntimeError, match="changed after setup"):
        layer(c=c)
    layer.settings.gradient_smoothing = not layer.settings.gradient_smoothing
    layer(c=c).sum().backward()                  # the failed call left no lease


def test_setup_rank_and_call_lifecycle_errors():
    qp = _rand_qp(n=4, p=1, m=2, seed=3)
    layer = _make_layer()
    with pytest.raises(RuntimeError, match="not set up"):
        layer(c=qp["c"])
    with pytest.raises(ValueError, match="'G' must be 2-D or 3-D"):
        layer.setup(qp["P"], qp["c"], G=qp["G"][0], h_u=qp["h_u"])
    with pytest.raises(ValueError, match="'h_u' must be 1-D or 2-D"):
        layer.setup(qp["P"], qp["c"], G=qp["G"], h_u=qp["h_u"].reshape(1, 1, 2))
    with pytest.raises(ValueError, match="'P' must be 2-D or 3-D"):
        layer.setup(qp["P"].reshape(1, 1, 4, 4), qp["c"])
    with pytest.raises(TypeError, match="'c' must be a dense"):
        layer.setup(qp["P"], qp["c"].to_sparse())
    assert not layer.is_setup
    batched_c = torch.stack([qp["c"], qp["c"] + 1.0])
    layer.setup(qp["P"], batched_c, G=qp["G"], h_u=qp["h_u"])
    assert layer.batch_size == 2
    with pytest.raises(StructureMismatchError, match=r"\(4,\) or \(2, 4\)"):
        layer(c=torch.zeros(3, 4, dtype=DTYPE, device=DEVICE))   # wrong batch size
    with pytest.raises(StructureMismatchError, match=r"\(2, 4\) or \(2, 2, 4\)"):
        layer(G=torch.zeros(2, 5, dtype=DTYPE, device=DEVICE))
    with pytest.raises(ValueError, match="dtype"):
        layer(c=qp["c"].float())
    with pytest.raises(ValueError, match="must be on cuda"):
        layer(c=qp["c"].cpu())
    layer(c=batched_c)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
def test_tensor_on_another_gpu_is_rejected():
    qp = _box_qp()
    layer = _make_layer()
    with pytest.raises(ValueError, match="must be on cuda:0"):
        layer.setup(qp["P"], qp["c"].to("cuda:1"))


# ---------------------------------------------------------------------------
# Gradients (reuse-factorization mode)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["x", "x+y+zu", "x*su+y"])
def test_backward_finite_diff_all_fields(kind):
    _check_all_fields(_rand_qp(n=4, p=1, m=2, seed=0), kind)


@pytest.mark.parametrize("drop", ["x_u", "x_l", "h_l", "h_u"])
def test_backward_one_sided(drop):
    qp = _rand_qp(n=4, p=1, m=3, seed=2)
    qp[drop] = None
    _check_all_fields(qp, "x", note=f"drop={drop} ")


def test_gradient_only_for_tensors_passed_to_the_call():
    qp = _box_qp()
    layer = _make_layer()
    c_setup = qp["c"].clone().requires_grad_(True)
    layer.setup(qp["P"], c_setup, x_l=qp["x_l"], x_u=qp["x_u"])
    assert layer().grad_fn is None
    c_call = qp["c"].clone().requires_grad_(True)
    layer(c=c_call).sum().backward()
    assert c_setup.grad is None
    torch.testing.assert_close(c_call.grad, -torch.ones(2, dtype=DTYPE, device=DEVICE))


def test_shared_input_gradient_is_summed_over_batch():
    n, batch_size = 3, 4
    rng = np.random.default_rng(11)
    M = rng.standard_normal((n, n))
    P = _t(M @ M.T + n * np.eye(n))
    c = _t(rng.standard_normal((batch_size, n)))
    layer = _make_layer()
    layer.setup(P, c)
    P_shared = P.clone().requires_grad_(True)
    layer(P=P_shared, c=c).sum().backward()
    grad_shared = P_shared.grad.clone()
    P_batched = P.expand(batch_size, n, n).clone().requires_grad_(True)
    layer(P=P_batched, c=c).sum().backward()
    assert P_batched.grad.shape == (batch_size, n, n)
    torch.testing.assert_close(grad_shared, P_batched.grad.sum(dim=0), atol=1e-8, rtol=1e-6)


def test_gradient_accumulates_across_calls():
    qp = _box_qp()
    layer = _make_layer()
    layer.setup(**qp)
    c = qp["c"].clone().requires_grad_(True)
    for _ in range(3):
        layer(c=c).sum().backward()
    torch.testing.assert_close(c.grad, -3.0 * torch.ones(2, dtype=DTYPE, device=DEVICE))


@pytest.mark.parametrize("mode", ["differentiate", "zero"])
def test_backward_proceeds_with_unsolved_problem_in_batch(mode):
    P = torch.eye(2, dtype=DTYPE, device=DEVICE)
    c = _t([[1.0, 1.0], [1.0, 1.0]]).requires_grad_(True)
    A = _t([[1.0, 0.0], [1.0, 0.0]])
    b = _t([[0.0, 0.0], [0.0, 1.0]])                 # problem 1 is infeasible
    layer = _make_layer(gradient_on_failure=mode)
    layer.setup(P, c.detach(), A=A, b=b)
    layer(c=c).sum().backward()
    statuses = _statuses(layer)
    assert statuses[0] == Status.CUPIQP_SOLVED and statuses[1] != Status.CUPIQP_SOLVED
    torch.testing.assert_close(c.grad[0], _t([0.0, -1.0]), atol=1e-6, rtol=0)
    if mode == "zero":
        assert torch.all(c.grad[1] == 0.0)


def test_training_loop():
    qp = _box_qp()
    layer = _make_layer()
    layer.setup(**qp)
    theta = torch.zeros(2, dtype=DTYPE, device=DEVICE, requires_grad=True)
    target = _t([[0.25, 0.75]])
    opt = torch.optim.SGD([theta], lr=0.5)
    for _ in range(40):
        opt.zero_grad()
        ((layer(c=theta) - target) ** 2).sum().backward()
        opt.step()
    with torch.no_grad():
        torch.testing.assert_close(layer(c=theta), target, atol=1e-3, rtol=0)


# ---------------------------------------------------------------------------
# Several pending solves (max_pending_solves > 1): rollouts
# ---------------------------------------------------------------------------
def _rollout_layer(T, per_call_fields=("c",), **settings):
    qp = _rand_qp(n=4, p=1, m=2, seed=7)
    layer = _make_layer(**settings)
    layer.setup(**qp, max_pending_solves=T, per_call_fields=per_call_fields)
    return layer, qp


def test_multi_pending_outputs_own_their_storage():
    layer, qp = _rollout_layer(3)
    c0 = qp["c"].clone().requires_grad_(True)
    c1 = (qp["c"] + 0.5).requires_grad_(True)
    x0 = layer(c=c0)
    kept0 = x0.detach().clone()
    x1 = layer(c=c1)
    assert x0.data_ptr() != x1.data_ptr()
    torch.testing.assert_close(x0.detach(), kept0)          # not overwritten
    assert not torch.allclose(x0.detach(), x1.detach())
    assert layer.result.x.data_ptr() == x1.data_ptr()
    (x0.sum() + x1.sum()).backward()
    assert c0.grad is not None and c1.grad is not None


@pytest.mark.parametrize("order", ["forward", "reverse"])
def test_multi_pending_rollout_matches_independent_layers(order):
    T = 4
    layer, qp = _rollout_layer(T, per_call_fields=("c", "h_u"))
    # Small perturbations: a larger shift of h_u moves this problem onto a
    # degenerate active set, where even the reference is far from finite differences.
    cs = [(qp["c"] + 0.1 * k).requires_grad_(True) for k in range(T)]
    hs = [(qp["h_u"] + 0.02 * k).requires_grad_(True) for k in range(T)]
    outs = [layer(c=cs[k], h_u=hs[k]) for k in range(T)]
    weights = [1.0, -2.0, 0.5, 3.0]
    terms = [w * o.sum() for w, o in zip(weights, outs)]
    if order == "reverse":
        terms = terms[::-1]
    sum(terms).backward()

    for k in range(T):
        ref = _make_layer()
        ref.setup(**qp)
        ck = cs[k].detach().clone().requires_grad_(True)
        hk = hs[k].detach().clone().requires_grad_(True)
        (weights[k] * ref(c=ck, h_u=hk).sum()).backward()
        torch.testing.assert_close(cs[k].grad, ck.grad, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(hs[k].grad, hk.grad, atol=1e-6, rtol=1e-6)


def test_multi_pending_matches_finite_differences():
    layer, qp = _rollout_layer(2)
    c0 = qp["c"].clone().requires_grad_(True)
    c1 = (qp["c"] + 0.4).requires_grad_(True)
    (layer(c=c0).sum() + 2.0 * layer(c=c1).sum()).backward()
    eps = 1e-5
    for param, weight in ((c0, 1.0), (c1, 2.0)):
        fd = torch.zeros_like(param)
        for i in range(param.numel()):
            d = torch.zeros_like(param); d.view(-1)[i] = eps
            with torch.no_grad():
                plus = layer(c=param.detach() + d).sum().item()
                minus = layer(c=param.detach() - d).sum().item()
            fd.view(-1)[i] = weight * (plus - minus) / (2 * eps)
        torch.testing.assert_close(param.grad, fd, atol=5e-4, rtol=5e-4)


def test_multi_pending_capacity_exhaustion_and_release():
    layer, qp = _rollout_layer(2)
    c = qp["c"].clone().requires_grad_(True)
    x0 = layer(c=c)
    x1 = layer(c=c)
    with pytest.raises(RuntimeError, match="max_pending_solves"):
        layer(c=c)
    with pytest.raises(RuntimeError, match="pending backward"):
        layer.reset()
    x0.sum().backward()                      # frees one saved state
    x2 = layer(c=c)
    del x1
    gc.collect()                             # an abandoned graph frees its saved state
    x3 = layer(c=c)
    (x2.sum() + x3.sum()).backward()
    layer.reset()


def test_multi_pending_exhaustion_raises_before_touching_the_solver():
    from unittest import mock
    layer, qp = _rollout_layer(2)
    cs = [(qp["c"] + 0.3 * k).requires_grad_(True) for k in range(2)]
    outs = [layer(c=ck) for ck in cs]
    kept = [o.detach().clone() for o in outs]
    with mock.patch.object(cupiqp.DenseSolver, "update", side_effect=AssertionError("update")), \
            mock.patch.object(cupiqp.DenseSolver, "solve", side_effect=AssertionError("solve")):
        with pytest.raises(RuntimeError, match="max_pending_solves"):
            layer(c=(qp["c"] + 1.0).requires_grad_(True))
    for out, ref in zip(outs, kept):
        torch.testing.assert_close(out.detach(), ref)
    sum(o.sum() for o in outs).backward()


def test_multi_pending_enforces_the_field_set():
    layer, qp = _rollout_layer(3, per_call_fields=("c", "h_u"))
    c = qp["c"].clone().requires_grad_(True)
    x0 = layer(c=c, h_u=qp["h_u"])
    with pytest.raises(ValueError, match="per_call_fields"):
        layer(c=c)                           # h_u missing
    with pytest.raises(ValueError, match="per_call_fields"):
        layer(c=c, h_u=qp["h_u"], b=qp["b"]) # b extra
    with torch.no_grad():
        with pytest.raises(ValueError, match="per_call_fields"):
            layer(c=c)                       # also for non-differentiable calls
    x0.sum().backward()
    with pytest.raises(ValueError, match="per_call_fields"):
        layer(c=c)                           # the set is fixed at setup, not per sequence
    strided = torch.empty(4, 2, dtype=DTYPE, device=DEVICE)[:, 0]   # shape (4,), stride 2
    strided.copy_(c.detach())
    assert not strided.is_contiguous()
    with pytest.raises(ValueError, match="contiguous"):
        layer(c=strided, h_u=qp["h_u"])


def test_multi_pending_failed_calls_do_not_leak_saved_states():
    from unittest import mock
    layer, qp = _rollout_layer(2)
    c = qp["c"].clone().requires_grad_(True)
    with pytest.raises(StructureMismatchError):
        layer(c=torch.zeros(3, dtype=DTYPE, device=DEVICE).requires_grad_(True))
    with mock.patch.object(cupiqp.DenseSolver, "solve", side_effect=RuntimeError("boom")):
        for _ in range(3):
            with pytest.raises(RuntimeError, match="boom"):
                layer(c=c)
    x = layer(c=c)
    with mock.patch.object(cupiqp.DenseSolver, "backward_from_state",
                           side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError, match="boom"):
            x.sum().backward()
    del x
    x0, x1 = layer(c=c), layer(c=c)          # both saved states are free again
    (x0.sum() + x1.sum()).backward()


def test_multi_pending_allocates_nothing_per_call_and_backward():
    layer, qp = _rollout_layer(3)
    c = qp["c"].clone().requires_grad_(True)
    (layer(c=c).sum() + layer(c=c).sum()).backward()        # warm up
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    (layer(c=c).sum() + layer(c=c).sum()).backward()
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == before


def test_multi_pending_with_preconditioner():
    T = 3
    qp = _rand_qp(n=4, p=1, m=2, seed=9)
    layer = _make_layer(preconditioner_iter=10)
    layer.setup(**qp, max_pending_solves=T, per_call_fields=("c",))
    cs = [(qp["c"] + 0.2 * k).requires_grad_(True) for k in range(T)]
    sum(layer(c=ck).sum() for ck in cs).backward()
    for k in range(T):
        ref = _make_layer(preconditioner_iter=10)
        ref.setup(**qp)
        ck = cs[k].detach().clone().requires_grad_(True)
        ref(c=ck).sum().backward()
        torch.testing.assert_close(cs[k].grad, ck.grad, atol=1e-6, rtol=1e-6)


def _spy(name):
    """Patch a DenseSolver method with a mock that still runs the original."""
    from unittest import mock
    return mock.patch.object(cupiqp.DenseSolver, name, autospec=True,
                             side_effect=getattr(cupiqp.DenseSolver, name))


@pytest.mark.parametrize("order", ["forward", "reverse"])
def test_multi_pending_backward_always_refactorizes(order):
    """Every backward restores and refactorizes, also the one of the most
    recent call, whose factorization is still in the core solver."""
    layer, qp = _rollout_layer(3)
    cs = [(qp["c"] + 0.2 * k).requires_grad_(True) for k in range(3)]
    outs = [layer(c=ck) for ck in cs]
    with _spy("backward") as reuse, _spy("backward_from_state") as refactorize:
        for k in (range(3) if order == "forward" else reversed(range(3))):
            outs[k].sum().backward()
    assert reuse.call_count == 0
    assert refactorize.call_count == 3
    # The most recent call alone, too.
    c_last = qp["c"].clone().requires_grad_(True)
    x = layer(c=c_last)
    with _spy("backward") as reuse, _spy("backward_from_state") as refactorize:
        x.sum().backward()
    assert reuse.call_count == 0 and refactorize.call_count == 1


def test_capacity_one_reuses_the_factorization():
    qp = _box_qp()
    layer = _make_layer()
    layer.setup(**qp)
    assert layer.max_pending_solves == 1
    c = qp["c"].clone().requires_grad_(True)
    x = layer(c=c)
    with _spy("backward") as reuse, _spy("backward_from_state") as refactorize:
        x.sum().backward()
    assert reuse.call_count == 1 and refactorize.call_count == 0


def _torch_bytes_of(make):
    """Torch allocator bytes held by the objects ``make()`` returns. The
    garbage collector is off meanwhile, so that it cannot free unrelated
    tensors inside the measured window."""
    gc.collect()
    gc.disable()
    try:
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        kept = make()
        torch.cuda.synchronize()
        used = torch.cuda.memory_allocated() - before
    finally:
        gc.enable()
    del kept
    return used


@pytest.mark.parametrize("capacity", [1, 2, 5])
def test_state_pool_allocates_exactly_the_capacity(capacity):
    """Capacity one allocates no state pool; a larger capacity allocates one
    stacked buffer per saved field, for exactly that many solves. The pool is
    the only Torch-allocated memory of setup() besides the shared-gradient
    buffers (the core solver allocates through Warp)."""
    qp = _rand_qp(n=4, p=1, m=2, seed=7)            # B = 1, every block present
    n, p, m = 4, 1, 2
    gradient_shapes = [(n, n), (n,), (p, n), (p,), (m, n), (m,), (m,), (n,), (n,)]

    def shared_gradient_buffers():
        return [torch.empty(s, dtype=DTYPE, device=DEVICE) for s in gradient_shapes]

    def pool(slots):
        widths = (n, p, m, m, n, n, m, m, n, n)       # x, y, z_*, s_* of this QP
        return ([torch.zeros((slots, 1, w), dtype=DTYPE, device=DEVICE) for w in widths]
                + [torch.zeros((slots, 1), dtype=DTYPE, device=DEVICE) for _ in range(2)]
                + [torch.zeros((slots, 1), dtype=torch.int32, device=DEVICE) for _ in range(2)])

    expected = _torch_bytes_of(shared_gradient_buffers)
    if capacity > 1:
        expected += _torch_bytes_of(lambda: pool(capacity))
        per_call = dict(per_call_fields=("c",))
    else:
        per_call = {}
    layer = _make_layer()
    used = _torch_bytes_of(lambda: layer.setup(**qp, max_pending_solves=capacity, **per_call))
    assert used == expected


def test_multi_pending_storage_is_stable_across_rounds():
    """The pool is fixed at setup: every round of pending calls hands out
    views of the same slot buffers, and backward allocates nothing new."""
    T = 3
    layer, qp = _rollout_layer(T)

    def one_round():
        cs = [(qp["c"] + 0.1 * k).requires_grad_(True) for k in range(T)]
        outs, statuses = [], []
        for ck in cs:
            outs.append(layer(c=ck))
            statuses.append(layer.result.status)
        pointers = ({o.data_ptr() for o in outs}, {s.data_ptr() for s in statuses})
        sum(o.sum() for o in outs).backward()
        return pointers

    first = one_round()
    assert len(first[0]) == T and len(first[1]) == T
    assert one_round() == first
    assert one_round() == first


def test_saved_state_keeps_solution_status_and_iterations():
    """A pending call's x, duals, status and iterations stay those of its own
    solve while later calls run."""
    T = 3
    layer, qp = _rollout_layer(T)
    cs = [(qp["c"] + 0.5 * k).requires_grad_(True) for k in range(T)]
    outs, kept = [], []
    for ck in cs:
        outs.append((layer(c=ck), layer.result))
        kept.append(layer.result.clone(detach=True))
    for (x, result), saved in zip(outs, kept):
        torch.testing.assert_close(x.detach(), saved.x, rtol=0, atol=0)
        torch.testing.assert_close(result.z_u.detach(), saved.z_u, rtol=0, atol=0)
        torch.testing.assert_close(result.status, saved.status, rtol=0, atol=0)
        torch.testing.assert_close(result.iterations, saved.iterations, rtol=0, atol=0)
    for k in range(T):                                   # the values are those of call k
        ref = _make_layer()
        ref.setup(**qp)
        with torch.no_grad():
            torch.testing.assert_close(ref(c=cs[k].detach()), kept[k].x, atol=1e-8, rtol=1e-8)
    sum(x.sum() for x, _ in outs).backward()


def test_multi_pending_no_grad_calls_do_not_count_and_do_not_disturb():
    T = 2
    layer, qp = _rollout_layer(T)
    cs = [(qp["c"] + 0.3 * k).requires_grad_(True) for k in range(T)]
    x0 = layer(c=cs[0])
    kept0 = x0.detach().clone()
    with torch.no_grad():
        for k in range(3):                   # more than the capacity
            layer(c=qp["c"] - 0.5 * (k + 1))
    plain = qp["c"] + 2.0                    # requires no grad: also free
    layer(c=plain)
    x1 = layer(c=cs[1])                      # the second and last saved state
    torch.testing.assert_close(x0.detach(), kept0)
    (x0.sum() + 2.0 * x1.sum()).backward()
    for k, weight in enumerate((1.0, 2.0)):
        ref = _make_layer()
        ref.setup(**qp)
        ck = cs[k].detach().clone().requires_grad_(True)
        (weight * ref(c=ck).sum()).backward()
        torch.testing.assert_close(cs[k].grad, ck.grad, atol=1e-6, rtol=1e-6)


def test_multi_pending_backward_twice_through_one_solve_raises():
    layer, qp = _rollout_layer(2)
    c = qp["c"].clone().requires_grad_(True)
    loss = layer(c=c).sum()
    loss.backward(retain_graph=True)
    with pytest.raises(RuntimeError, match="differentiated twice"):
        loss.backward()


# ---------------------------------------------------------------------------
# Lifecycle of the backward state (lease) owned by the autograd node
# ---------------------------------------------------------------------------
def test_capacity_one_rejects_no_grad_calls_while_pending():
    qp = _box_qp()
    layer = _make_layer()
    layer.setup(**qp)
    c = qp["c"].clone().requires_grad_(True)
    x = layer(c=c)
    with torch.no_grad():
        with pytest.raises(RuntimeError, match="pending backward"):
            layer(c=qp["c"])              # would overwrite the factorization
    torch.testing.assert_close(x.detach(), _t(BOX_X))   # untouched
    x.sum().backward()


def test_capacity_one_failed_forward_releases_the_lease():
    from unittest import mock
    qp = _box_qp()
    layer = _make_layer()
    layer.setup(**qp)
    c = qp["c"].clone().requires_grad_(True)
    with mock.patch.object(cupiqp.DenseSolver, "solve", side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError, match="boom"):
            layer(c=c)
    x = layer(c=c)                        # the lease was released by the failure
    x.sum().backward()


@pytest.mark.parametrize("max_pending_solves", [1, 3])
def test_dropped_layer_is_freed_by_reference_counting(max_pending_solves):
    """After setup, forward and backward, dropping the last reference frees
    the layer (and its core solver) at once, without the cyclic garbage
    collector, which could otherwise run during another solver's CUDA graph
    capture."""
    import weakref
    qp = _rand_qp(n=4, p=1, m=2, seed=7)
    gc.collect()
    gc.disable()
    try:
        layer = _make_layer()
        per_call = dict(per_call_fields=("c",)) if max_pending_solves > 1 else {}
        layer.setup(**qp, max_pending_solves=max_pending_solves, **per_call)
        c = qp["c"].clone().requires_grad_(True)
        layer(c=c).sum().backward()
        torch.cuda.synchronize()
        ref = weakref.ref(layer)
        del layer
        assert ref() is None, "a reference cycle keeps the dropped layer alive"
    finally:
        gc.enable()


def test_capacity_one_failed_backward_releases_the_lease():
    from unittest import mock
    qp = _box_qp()
    layer = _make_layer()
    layer.setup(**qp)
    c = qp["c"].clone().requires_grad_(True)
    x = layer(c=c)
    with mock.patch.object(cupiqp.DenseSolver, "backward", side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError, match="boom"):
            x.sum().backward()
    layer(c=c).sum().backward()           # not blocked by the failed backward


def test_status_and_iterations_are_not_differentiable_and_follow_the_saved_state():
    layer, qp = _rollout_layer(2)
    c = qp["c"].clone().requires_grad_(True)
    x0 = layer(c=c)
    s0, it0 = layer.result.status, layer.result.iterations
    assert not s0.requires_grad and not it0.requires_grad
    assert s0.dtype == torch.int32 and it0.dtype == torch.int32
    x1 = layer(c=c * 3.0)
    s1 = layer.result.status
    assert s0.data_ptr() != s1.data_ptr()   # each call's status lives in its saved state
    assert int(s0[0]) == Status.CUPIQP_SOLVED and int(s1[0]) == Status.CUPIQP_SOLVED
    (x0.sum() + x1.sum()).backward()


def test_inputs_without_grad_are_still_restored_for_the_backward():
    """With max_pending_solves > 1 every passed tensor is saved, also one that needs
    no gradient: it is part of the QP whose other input is differentiated."""
    T = 3
    layer, qp = _rollout_layer(T, per_call_fields=("c", "h_u"))
    cs = [(qp["c"] + 0.1 * k).requires_grad_(True) for k in range(T)]
    hs = [qp["h_u"] + 0.02 * k for k in range(T)]           # no grad
    sum(layer(c=cs[k], h_u=hs[k]).sum() for k in range(T)).backward()
    for k in range(T):
        ref = _make_layer()
        ref.setup(**qp)
        ck = cs[k].detach().clone().requires_grad_(True)
        ref(c=ck, h_u=hs[k]).sum().backward()
        torch.testing.assert_close(cs[k].grad, ck.grad, atol=1e-6, rtol=1e-6)
