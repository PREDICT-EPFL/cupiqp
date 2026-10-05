"""Tests for the high-level HPIPM-style OcpSolver (multistage backend)."""
import numpy as np
import cupy as cp
import pytest

import cupiqp
from cupiqp import OcpSolver, OcpData, DenseSolver, Status
from cupiqp.settings import Settings


def _solved(status):
    """solve() returns a per-problem list of Status; True iff all converged."""
    return all(st == Status.CUPIQP_SOLVED for st in status)


@pytest.fixture(autouse=True)
def _solver_test_isolation(monkeypatch):
    """Isolate the many solver instances these tests create in one process.

    Two measures, both standard for this backend and orthogonal to the solver
    logic under test:

    * Disable CUDA-graph capture (the existing batched tests do the same).
    * Disable CuPy memory pooling for the duration. Creating many short-lived
      solver instances back-to-back otherwise lets the pool hand a just-freed
      buffer to the next instance while a dlpack view still aliases it, which
      manifests as nondeterministic numerical failures. A single long-lived
      solver (the realistic MPC pattern) is unaffected.
    """
    orig = Settings.for_dtype.__func__

    def patched(cls, dtype):
        s = orig(cls, dtype)
        s.enable_cuda_graph = False
        return s

    monkeypatch.setattr(Settings, "for_dtype", classmethod(patched))
    # Release everything earlier test modules left in the pool before turning
    # it off, so no pooled block can be handed out while still aliased.
    import gc
    import warp as wp
    gc.collect()
    wp.synchronize()
    cp.cuda.runtime.deviceSynchronize()
    cp.get_default_memory_pool().free_all_blocks()
    cp.cuda.set_allocator(None)
    yield
    cp.cuda.set_allocator(cp.get_default_memory_pool().malloc)
    gc.collect()
    wp.synchronize()
    cp.cuda.runtime.deviceSynchronize()


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

def _double_integrator(dt=0.2):
    A = np.array([[1.0, dt], [0.0, 1.0]])
    B = np.array([[0.5 * dt * dt], [dt]])
    return A, B


def _solve_dense(N, nx, nu, Qs, Rs, Ss, qs, rs, As, Bs, Es, bs, x0,
                 Cs=None, Ds=None, lgs=None, ugs=None,
                 xl_stage=None, xu_stage=None):
    """Assemble the same OCP as a flat dense QP and solve with DenseSolver.

    Variable ordering matches OcpSolver: z = [y_0; ...; y_N], y_k = [x_k; u_k].
    Returns (x_traj (N+1, nx), u_traj (N, nu)).
    """
    d = nx + nu
    n = (N + 1) * d

    P = np.zeros((n, n))
    c = np.zeros(n)
    for k in range(N + 1):
        blk = np.zeros((d, d))
        blk[:nx, :nx] = Qs[k]
        blk[nx:, nx:] = Rs[k]
        blk[nx:, :nx] = Ss[k]
        blk[:nx, nx:] = Ss[k].T
        P[k * d:(k + 1) * d, k * d:(k + 1) * d] = blk
        c[k * d:k * d + nx] = qs[k]
        c[k * d + nx:(k + 1) * d] = rs[k]

    p = (N + 1) * nx
    Aeq = np.zeros((p, n))
    beq = np.zeros(p)
    Aeq[:nx, :nx] = np.eye(nx)
    beq[:nx] = x0
    for k in range(N):
        r0 = (k + 1) * nx
        Aeq[r0:r0 + nx, k * d:k * d + nx] = As[k]
        Aeq[r0:r0 + nx, k * d + nx:k * d + nx + nu] = Bs[k]
        Aeq[r0:r0 + nx, (k + 1) * d:(k + 1) * d + nx] = -Es[k]
        beq[r0:r0 + nx] = -bs[k]

    G = hl = hu = None
    if Cs is not None:
        ng = Cs[0].shape[0]
        m = (N + 1) * ng
        G = np.zeros((m, n))
        hl = np.full(m, -np.inf)
        hu = np.full(m, np.inf)
        for k in range(N + 1):
            r0 = k * ng
            G[r0:r0 + ng, k * d:k * d + nx] = Cs[k]
            G[r0:r0 + ng, k * d + nx:(k + 1) * d] = Ds[k]
            hl[r0:r0 + ng] = lgs[k]
            hu[r0:r0 + ng] = ugs[k]

    if xl_stage is None:
        xl = np.full(n, -np.inf)
        xu = np.full(n, np.inf)
    else:
        xl = np.asarray(xl_stage, float).reshape(n)
        xu = np.asarray(xu_stage, float).reshape(n)

    s = DenseSolver()
    s.settings.eps_abs = 1e-9
    s.settings.eps_rel = 1e-10
    s.settings.verbose = False
    kw = dict(P=cp.asarray(P), c=cp.asarray(c),
              A=cp.asarray(Aeq), b=cp.asarray(beq),
              x_l=cp.asarray(xl), x_u=cp.asarray(xu))
    if G is not None:
        kw.update(G=cp.asarray(G), h_l=cp.asarray(hl), h_u=cp.asarray(hu))
    s.setup(**kw)
    s.solve()
    assert s.result.info.to_host().status[0] == Status.CUPIQP_SOLVED
    z = cp.asnumpy(s.result.x[0]).reshape(N + 1, d)
    return z[:, :nx], z[:N, nx:]


def _fill_ocp(s, N, nx, nu, Qs, Rs, As, Bs, x0, Es=None, Qf=None):
    """Fill a standard (E defaults to I unless Es given) OCP into an OcpSolver."""
    for k in range(N):
        s.set("A", k, cp.asarray(As[k]))
        s.set("B", k, cp.asarray(Bs[k]))
        if Es is not None:
            s.set("E", k, cp.asarray(Es[k]))
        s.set("Q", k, cp.asarray(Qs[k]))
        s.set("R", k, cp.asarray(Rs[k]))
    s.set("Q", N, cp.asarray(Qf if Qf is not None else Qs[-1]))
    s.set("x0", 0, cp.asarray(x0))


# ----------------------------------------------------------------------
# tests
# ----------------------------------------------------------------------

def test_import_smoke():
    assert hasattr(cupiqp, "OcpSolver")
    assert cupiqp.OcpSolver is OcpSolver
    assert cupiqp.OcpData is OcpData


def test_idxbx_state_box():
    """idxbx bounds a specific state component; the bound is enforced."""
    N, nx, nu = 6, 2, 1
    A, B = _double_integrator()
    Q = np.diag([1.0, 0.1])
    R = np.array([[0.02]])
    Qf = np.diag([5.0, 1.0])
    x0 = np.array([1.0, 0.0])
    vmax = 0.3                       # cap the velocity (state component 1)

    s = OcpSolver()
    s.settings.eps_abs = 1e-9
    s.setup(N=N, nx=nx, nu=nu, idxbx=[1])      # box only on the velocity state
    _fill_ocp(s, N, nx, nu, [Q] * N, [R] * N, [A] * N, [B] * N, x0, Qf=Qf)
    for k in range(N + 1):
        s.set("lbx", k, cp.asarray([-vmax]))
        s.set("ubx", k, cp.asarray([vmax]))
    s.solve()
    assert _solved(s.result.info.to_host().status)

    x = cp.asnumpy(s.x_traj[0])
    np.testing.assert_allclose(x[0], x0, atol=1e-7)
    assert np.all(x[:, 1] <= vmax + 1e-6) and np.all(x[:, 1] >= -vmax - 1e-6)
    # omitting idxbx/idxbu => no box bounds at all
    assert OcpData(N=3, nx=2, nu=1).idxbx.size == 0


def test_matches_dense_box_constrained():
    N, nx, nu = 6, 2, 1
    A, B = _double_integrator()
    Q = np.diag([1.0, 0.1])
    R = np.array([[0.05]])
    Qf = np.diag([5.0, 1.0])
    x0 = np.array([1.0, 0.5])
    umax = 0.4

    Qs = [Q] * N + [Qf]
    Rs = [R] * N + [np.zeros((nu, nu))]
    Ss = [np.zeros((nu, nx))] * (N + 1)
    qs = [np.zeros(nx)] * (N + 1)
    rs = [np.zeros(nu)] * (N + 1)
    As, Bs, bs = [A] * N, [B] * N, [np.zeros(nx)] * N
    Es = [np.eye(nx)] * N

    # input box active at control stages 0..N-1; u_N is padding
    xl_stage = np.full((N + 1, nx + nu), np.inf) * -1.0
    xu_stage = np.full((N + 1, nx + nu), np.inf)
    xl_stage[:N, nx:] = -umax
    xu_stage[:N, nx:] = umax
    x_ref, u_ref = _solve_dense(N, nx, nu, Qs, Rs, Ss, qs, rs, As, Bs, Es, bs, x0,
                                xl_stage=xl_stage, xu_stage=xu_stage)

    s = OcpSolver()
    s.settings.eps_abs = 1e-9
    s.settings.eps_rel = 1e-10
    s.setup(N=N, nx=nx, nu=nu, idxbu=[0])
    _fill_ocp(s, N, nx, nu, Qs[:N], Rs[:N], As, Bs, x0, Qf=Qf)
    for k in range(N):
        s.set("lbu", k, cp.asarray([-umax]))
        s.set("ubu", k, cp.asarray([umax]))
    s.solve()
    status = s.result.info.to_host().status
    assert _solved(status)

    x = cp.asnumpy(s.x_traj[0])
    u = cp.asnumpy(s.u_traj[0])
    np.testing.assert_allclose(x, x_ref, atol=1e-5)
    np.testing.assert_allclose(u, u_ref, atol=1e-5)
    # initial condition honored, input limit respected
    np.testing.assert_allclose(x[0], x0, atol=1e-7)
    assert np.all(u <= umax + 1e-6) and np.all(u >= -umax - 1e-6)


def test_descriptor_E():
    N, nx, nu = 5, 2, 1
    A, B = _double_integrator()
    E = np.array([[1.0, 0.0], [0.0, 2.0]])   # invertible descriptor
    Einv = np.linalg.inv(E)
    Q = np.diag([1.0, 0.1])
    R = np.array([[0.1]])
    Qf = np.diag([4.0, 1.0])
    x0 = np.array([1.0, 0.0])

    # (a) descriptor form via set('E')
    s = OcpSolver()
    s.settings.eps_abs = 1e-9
    s.setup(N=N, nx=nx, nu=nu)
    _fill_ocp(s, N, nx, nu, [Q] * N, [R] * N, [A] * N, [B] * N, x0,
              Es=[E] * N, Qf=Qf)
    s.solve()
    assert _solved(s.result.info.to_host().status)
    x = cp.asnumpy(s.x_traj[0])
    u = cp.asnumpy(s.u_traj[0])

    # the returned trajectory satisfies E x_{k+1} = A x_k + B u_k
    for k in range(N):
        lhs = E @ x[k + 1]
        rhs = A @ x[k] + B @ u[k]
        np.testing.assert_allclose(lhs, rhs, atol=1e-6)

    # (b) equivalent explicit problem x_{k+1} = E^{-1}(A x_k + B u_k)
    s2 = OcpSolver()
    s2.settings.eps_abs = 1e-9
    s2.setup(N=N, nx=nx, nu=nu)
    _fill_ocp(s2, N, nx, nu, [Q] * N, [R] * N, [Einv @ A] * N, [Einv @ B] * N,
              x0, Qf=Qf)
    s2.solve()
    assert _solved(s2.result.info.to_host().status)
    np.testing.assert_allclose(x, cp.asnumpy(s2.x_traj[0]), atol=1e-5)
    np.testing.assert_allclose(u, cp.asnumpy(s2.u_traj[0]), atol=1e-5)


def test_general_inequality():
    N, nx, nu = 6, 2, 1
    A, B = _double_integrator()
    Q = np.diag([1.0, 0.1])
    R = np.array([[0.02]])
    Qf = np.diag([5.0, 1.0])
    x0 = np.array([1.0, 0.0])        # feasible w.r.t. the velocity cap at stage 0
    vmax = 0.3                       # velocity limit via general constraint (binds en route)

    # general constraint: -vmax <= [0,1] x_k + [0] u_k <= vmax
    C = np.array([[0.0, 1.0]])
    D = np.zeros((1, nu))
    Cs = [C] * (N + 1)
    Ds = [D] * (N + 1)
    lgs = [np.array([-vmax])] * (N + 1)
    ugs = [np.array([vmax])] * (N + 1)
    Qs = [Q] * N + [Qf]
    Rs = [R] * N + [np.zeros((nu, nu))]
    Ss = [np.zeros((nu, nx))] * (N + 1)
    qs = [np.zeros(nx)] * (N + 1)
    rs = [np.zeros(nu)] * (N + 1)
    As, Bs, bs, Es = [A] * N, [B] * N, [np.zeros(nx)] * N, [np.eye(nx)] * N
    x_ref, u_ref = _solve_dense(N, nx, nu, Qs, Rs, Ss, qs, rs, As, Bs, Es, bs, x0,
                                Cs=Cs, Ds=Ds, lgs=lgs, ugs=ugs)

    s = OcpSolver()
    s.settings.eps_abs = 1e-9
    s.setup(N=N, nx=nx, nu=nu, ng=1)
    _fill_ocp(s, N, nx, nu, Qs[:N], Rs[:N], As, Bs, x0, Qf=Qf)
    for k in range(N):
        s.set("C", k, cp.asarray(C))
        s.set("D", k, cp.asarray(D))
        s.set("lg", k, cp.asarray([-vmax]))
        s.set("ug", k, cp.asarray([vmax]))
    s.set("C", N, cp.asarray(C))
    s.set("lg", N, cp.asarray([-vmax]))
    s.set("ug", N, cp.asarray([vmax]))
    s.solve()
    assert _solved(s.result.info.to_host().status)

    x = cp.asnumpy(s.x_traj[0])
    u = cp.asnumpy(s.u_traj[0])
    np.testing.assert_allclose(x, x_ref, atol=1e-5)
    np.testing.assert_allclose(u, u_ref, atol=1e-5)
    assert np.all(x[:, 1] <= vmax + 1e-6) and np.all(x[:, 1] >= -vmax - 1e-6)


def test_batched_per_x0():
    N, nx, nu, Bsz = 5, 2, 1, 3
    A, B = _double_integrator()
    Q = np.diag([1.0, 0.1])
    R = np.array([[0.05]])
    Qf = np.diag([5.0, 1.0])
    x0s = np.array([[1.0, 0.0], [-1.0, 0.5], [0.3, -0.7]])

    s = OcpSolver()
    s.settings.eps_abs = 1e-9
    s.setup(N=N, nx=nx, nu=nu, idxbu=[0], batch_size=Bsz)
    for k in range(N):
        s.set("A", k, cp.asarray(A))          # broadcast across the batch
        s.set("B", k, cp.asarray(B))
        s.set("Q", k, cp.asarray(Q))
        s.set("R", k, cp.asarray(R))
        s.set("lbu", k, cp.asarray([-0.5]))
        s.set("ubu", k, cp.asarray([0.5]))
    s.set("Q", N, cp.asarray(Qf))
    s.set("x0", 0, cp.asarray(x0s))           # leading batch axis -> per-problem IC
    s.solve()
    statuses = s.result.info.to_host().status

    assert isinstance(statuses, list) and len(statuses) == Bsz
    assert all(st == Status.CUPIQP_SOLVED for st in statuses)
    x0_got = cp.asnumpy(s.get("x", 0))
    np.testing.assert_allclose(x0_got, x0s, atol=1e-6)

    # cross-check each batch member against a single-problem solve
    for i in range(Bsz):
        si = OcpSolver()
        si.settings.eps_abs = 1e-9
        si.setup(N=N, nx=nx, nu=nu, idxbu=[0])
        _fill_ocp(si, N, nx, nu, [Q] * N, [R] * N, [A] * N, [B] * N, x0s[i], Qf=Qf)
        for k in range(N):
            si.set("lbu", k, cp.asarray([-0.5]))
            si.set("ubu", k, cp.asarray([0.5]))
        si.solve()
        np.testing.assert_allclose(cp.asnumpy(s.x_traj[i]),
                                   cp.asnumpy(si.x_traj[0]), atol=1e-5)


def test_mpc_loop_update_only():
    N, nx, nu = 8, 2, 1
    A, B = _double_integrator()
    Q = np.diag([1.0, 0.1])
    R = np.array([[0.05]])
    Qf = np.diag([10.0, 5.0])
    umax = 0.5

    s = OcpSolver()
    s.settings.eps_abs = 1e-9
    s.setup(N=N, nx=nx, nu=nu, idxbu=[0])
    _fill_ocp(s, N, nx, nu, [Q] * N, [R] * N, [A] * N, [B] * N,
              np.zeros(nx), Qf=Qf)
    for k in range(N):
        s.set("lbu", k, cp.asarray([-umax]))
        s.set("ubu", k, cp.asarray([umax]))

    x = np.array([1.5, 0.0])
    norms = [float(np.linalg.norm(x))]
    for _ in range(25):
        s.set("x0", 0, cp.asarray(x))            # only the initial condition changes
        s.solve()
        assert _solved(s.result.info.to_host().status)
        u0 = float(cp.asnumpy(s.get("u", 0))[0, 0])
        assert -umax - 1e-6 <= u0 <= umax + 1e-6
        x = A @ x + (B @ np.array([u0]))
        norms.append(float(np.linalg.norm(x)))

    # closed loop drives the state toward the origin
    assert norms[-1] < 0.1 * norms[0]


def test_ocp_data_read_only_arrays():
    data = OcpData(N=2, nx=2, nu=1)
    assert set(data.arrays) == {
        "P", "c", "A", "b", "G", "h_l", "h_u", "x_l", "x_u"
    }
    # the mapping structure is immutable (buffers change only via set_field)
    with pytest.raises(TypeError):
        data.arrays["P"] = None


def test_matrix_update_between_solves():
    """Re-setting a cost matrix via set() between solves is re-flushed."""
    N, nx, nu = 5, 2, 1
    A, B = _double_integrator()
    Q = np.diag([1.0, 0.1])
    Qf = np.diag([5.0, 1.0])
    R_cheap = np.array([[0.01]])
    R_pricey = np.array([[10.0]])
    x0 = np.array([1.0, 0.0])

    s = OcpSolver()
    s.settings.eps_abs = 1e-9
    s.setup(N=N, nx=nx, nu=nu)
    _fill_ocp(s, N, nx, nu, [Q] * N, [R_cheap] * N, [A] * N, [B] * N, x0, Qf=Qf)
    s.solve()
    assert _solved(s.result.info.to_host().status)
    u_cheap = cp.asnumpy(s.u_traj[0])

    for k in range(N):
        s.set("R", k, cp.asarray(R_pricey))        # heavier input penalty (flags the P block)
    s.solve()
    assert _solved(s.result.info.to_host().status)
    u_pricey = cp.asnumpy(s.u_traj[0])

    # a much larger input penalty shrinks the control effort
    assert np.linalg.norm(u_pricey) < np.linalg.norm(u_cheap)


@pytest.mark.parametrize(
    "field,value",
    [
        ("R", np.ones((1, 1))),
        ("S", np.ones((1, 2))),
        ("r", np.ones(1)),
        ("D", np.ones((1, 1))),
        ("lbu", np.ones(1)),
        ("ubu", np.ones(1)),
    ],
)
def test_terminal_input_fields_rejected(field, value):
    data = OcpData(N=2, nx=2, nu=1, ng=1, idxbu=[0])
    with pytest.raises(ValueError, match="expected 0..1"):
        data.set_field(field, 2, value)


def test_field_shape_and_stage_validation():
    data = OcpData(N=2, nx=2, nu=1, batch_size=2)

    with pytest.raises(ValueError, match="has shape"):
        data.set_field("A", 0, cp.ones((2,)))          # unbatched, wrong shape

    with pytest.raises(ValueError, match="has shape"):
        data.set_field("A", 0, cp.ones((3, 2, 2)))     # wrong batch size

    with pytest.raises(TypeError, match="stage must be an integer"):
        data.set_field("A", 0.0, cp.eye(2))

    data.set_field("A", 0, cp.ones((2, 2, 2)))         # correct (batched) shape


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"N": 1.5, "nx": 2, "nu": 1}, AssertionError),     # non-integer dim
        ({"N": 1, "nx": 2, "nu": 1, "ng": -1}, AssertionError),
        ({"N": 1, "nx": 2, "nu": 1, "batch_size": 0}, AssertionError),
        ({"N": 1, "nx": 2, "nu": 1, "idxbu": [0.5]}, TypeError),   # from _normalize_idx
    ],
)
def test_dimension_validation(kwargs, error):
    with pytest.raises(error):
        OcpData(**kwargs)


def test_solution_lifecycle_guards():
    s = OcpSolver()
    s.setup(N=2, nx=2, nu=1)

    with pytest.raises(RuntimeError, match="call solve"):
        s.get("x", 0)
    with pytest.raises(RuntimeError, match="call solve"):
        _ = s.x_traj
    with pytest.raises(RuntimeError, match="call solve"):
        _ = s.u_traj
    with pytest.raises(RuntimeError, match="only be called once"):
        s.setup(N=2, nx=2, nu=1)


def test_set_invalidates_solution():
    N, nx, nu = 4, 2, 1
    A, B = _double_integrator()
    Q, R = np.diag([1.0, 0.1]), np.array([[0.05]])

    s = OcpSolver()
    s.setup(N=N, nx=nx, nu=nu)
    _fill_ocp(s, N, nx, nu, [Q] * N, [R] * N, [A] * N, [B] * N, np.array([1.0, 0.0]))
    s.solve()
    assert _solved(s.result.info.to_host().status)
    x_old = s.x_traj.numpy().copy()

    # a rejected write changes nothing: the solution stays readable
    with pytest.raises(ValueError, match="has shape"):
        s.set("Q", 0, cp.ones((3,)))
    np.testing.assert_array_equal(s.x_traj.numpy(), x_old)

    # a successful write makes the old solution unavailable until solve()
    s.set("x0", 0, cp.asarray([0.5, 0.0]))
    with pytest.raises(RuntimeError, match="call solve"):
        _ = s.x_traj
    with pytest.raises(RuntimeError, match="call solve"):
        s.get("u", 0)

    s.solve()
    assert _solved(s.result.info.to_host().status)
    assert not np.allclose(s.x_traj.numpy(), x_old)
