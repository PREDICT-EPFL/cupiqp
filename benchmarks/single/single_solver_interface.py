"""
Unified single-problem QP solver interface for benchmarking.

All solvers accept the same QP formulation:

    min  0.5 x^T P x + c^T x
    s.t. A x = b
         h_l <= G x <= h_u
         x_l <= x <= x_u

Each solver subclass declares ``device = 'cpu' | 'gpu'`` so callers can
filter by backend, pick the right timing strategy, and skip GPU solvers
when CUDA is unavailable.

Bundled solvers
    CPU : osqp, piqp-sparse, clarabel, qpalm, gurobi,
          hpipm   (OCP-structured; reads SingleQPData.ocp),
          cyqlone (OCP-structured; reads SingleQPData.ocp)
    GPU : cupiqp-sparse, cupiqp-dense, cupiqp-multistage,
          qoco-gpu (algebra='cuda'),
          cuclarabel (Clarabel.jl + CUDA.jl via juliacall),
          cuopt (NVIDIA cuOpt LP/QP),
          moreau-torch (moreau via PyTorch, cuDSS bundled)
"""

import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Union, ClassVar, List
import time

import numpy as np
import scipy.sparse as sp
import nvtx


SOLVER_COLORS = {
    "osqp":              "purple",
    "piqp-sparse":       "orange",
    "clarabel":          "yellow",
    "hpipm":             "olive",
    "qpalm":             "teal",
    "cyqlone":           "indigo",
    "gurobi":            "maroon",
    "qoco-gpu":          "green",
    "cupiqp-sparse":     "blue",
    "cupiqp-dense":      "cyan",
    "cupiqp-multistage": "navy",
    "cuclarabel":        "red",
    "cuopt":             "brown",
    "moreau-torch":      "magenta",
}


# ============================================================================
# Shared data / result types
# ============================================================================

SparseOrDense = Union[np.ndarray, sp.spmatrix]


@dataclass
class SingleQPData:
    """Single-problem QP data. Matrices may be dense numpy or scipy sparse.

    The flat fields (``P, c, A, b, G, h_*, x_*``) are the canonical form
    every solver in this interface understands.

    Optional ``ocp`` carries an :class:`benchmarks.problems.OCPProblem`
    instance for solvers that exploit OCP / multistage block structure
    (``hpipm``, ``cyqlone``, ``cupiqp-multistage``). The OCP-aware
    solvers consume its stagewise attributes (``A, B, Q, R, S, QN, x0,
    ul, uu, C, D, gl, gu, N, nx, nu, system.{nx, nu_max, nx_max}``) and
    — for ``cupiqp-multistage`` only — its block-structured
    ``ms_P, ms_c, ms_A, ms_b, ms_G, ms_h_u, ms_h_l, ms_x_u, ms_x_l``
    attributes built by the concrete OCPProblem subclass. Flat-QP
    solvers ignore this field.
    """
    P: SparseOrDense                              # (n, n)
    c: np.ndarray                                 # (n,)
    A: Optional[SparseOrDense] = None             # (p, n)
    b: Optional[np.ndarray] = None                # (p,)
    G: Optional[SparseOrDense] = None             # (m, n)
    h_l: Optional[np.ndarray] = None              # (m,)
    h_u: Optional[np.ndarray] = None              # (m,)
    x_l: Optional[np.ndarray] = None              # (n,)
    x_u: Optional[np.ndarray] = None              # (n,)
    # Optional reference to the OCPProblem that generated this QP, for
    # OCP-aware solvers that need the stagewise block matrices. Typed
    # as ``Optional[Any]`` rather than ``Optional[OCPProblem]`` to keep
    # the import lazy at module load time.
    ocp: Optional[Any] = None

    @property
    def n(self) -> int:
        return self.P.shape[0]

    @property
    def p(self) -> int:
        return self.A.shape[0] if self.A is not None else 0

    @property
    def m(self) -> int:
        return self.G.shape[0] if self.G is not None else 0


@dataclass
class SingleQPResult:
    """Unified result from a single QP solve."""
    x: np.ndarray
    setup_time_ms: float = 0.0
    solve_time_ms: float = 0.0
    solve_times_all: List[float] = field(default_factory=list)
    solved: bool = False
    n_iter: int = -1
    obj: float = float('nan')
    status: str = ''
    solver_name: str = ''
    device: str = 'cpu'

    @property
    def solve_time_std(self) -> float:
        return float(np.std(self.solve_times_all)) if self.solve_times_all else 0.0

    @property
    def solve_time_stderr(self) -> float:
        n = len(self.solve_times_all)
        return float(np.std(self.solve_times_all) / np.sqrt(n)) if n > 0 else 0.0


# ============================================================================
# Base class
# ============================================================================

class SingleQPSolver(ABC):
    """Base class for single-problem QP solvers.

    Lifecycle (mirrors ``BatchedQPSolver``):
        _prepare_data(data)  --  numpy → native conversion (NOT timed).
        setup()              --  solver-specific init (factor pre-conditions).
        solve()              --  the actual solve. GPU solvers must sync.
        _collect_result()    --  pull solution to host, build SingleQPResult.

    Subclass attributes
    -------------------
    device : ClassVar[str]
        ``'cpu'`` or ``'gpu'``. The benchmark harness uses this to skip GPU
        solvers when CUDA isn't available and (optionally) to switch timing
        strategies. Default is ``'cpu'``.
    """
    device: ClassVar[str] = 'cpu'

    def __init__(self, tol_abs: float = 1e-6, max_iter: int = 300):
        self.tol_abs = tol_abs
        self.max_iter = max_iter

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    def is_gpu(self) -> bool:
        return self.device == 'gpu'

    @abstractmethod
    def _prepare_data(self, data: SingleQPData) -> None:
        ...

    @abstractmethod
    def setup(self) -> None:
        ...

    @abstractmethod
    def solve(self) -> None:
        """Run the solver. GPU subclasses must end with a stream sync."""
        ...

    @abstractmethod
    def _collect_result(self) -> SingleQPResult:
        ...

    def benchmark(self, data: SingleQPData, n_repeats: int = 5) -> SingleQPResult:
        """Time setup once and ``n_repeats`` solves, return median.

        First solve after setup is a warm-up (factorization caches, JIT,
        etc.) and is excluded from ``solve_times_all``.
        """
        self._prepare_data(data)

        t0 = time.perf_counter()
        self.setup()
        t1 = time.perf_counter()
        setup_time_ms = (t1 - t0) * 1000.0

        # Warm-up
        self.solve()

        solve_times = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            self.solve()
            t1 = time.perf_counter()
            solve_times.append((t1 - t0) * 1000.0)

        result = self._collect_result()
        result.setup_time_ms = setup_time_ms
        result.solve_time_ms = float(np.median(solve_times))
        result.solve_times_all = solve_times
        result.solver_name = self.name
        result.device = self.device
        return result


# ----------------------------------------------------------------------------
# Helpers shared by several backends
# ----------------------------------------------------------------------------

def _maybe_dense(M: SparseOrDense) -> np.ndarray:
    return np.asarray(M.todense()) if sp.issparse(M) else np.asarray(M)


def _stack_box_constraints(data: SingleQPData):
    """Compose [A_eq; G; I] with bounds [b,b; h_l,h_u; x_l,x_u] for OSQP-style.

    Drops rows whose corresponding bound is +/-inf on the relevant side.
    Returns (A_stack_csc, l, u). All rows kept; OSQP handles +/-inf natively.
    """
    n = data.n
    rows, l_parts, u_parts = [], [], []

    if data.A is not None:
        rows.append(sp.csc_matrix(data.A))
        l_parts.append(np.asarray(data.b, dtype=np.float64))
        u_parts.append(np.asarray(data.b, dtype=np.float64))

    if data.G is not None:
        rows.append(sp.csc_matrix(data.G))
        m = data.G.shape[0]
        l_parts.append(np.asarray(data.h_l, dtype=np.float64) if data.h_l is not None
                       else np.full(m, -np.inf))
        u_parts.append(np.asarray(data.h_u, dtype=np.float64) if data.h_u is not None
                       else np.full(m, +np.inf))

    if data.x_l is not None or data.x_u is not None:
        rows.append(sp.eye(n, format='csc'))
        l_parts.append(np.asarray(data.x_l, dtype=np.float64) if data.x_l is not None
                       else np.full(n, -np.inf))
        u_parts.append(np.asarray(data.x_u, dtype=np.float64) if data.x_u is not None
                       else np.full(n, +np.inf))

    if rows:
        A_stack = sp.vstack(rows, format='csc')
        l = np.concatenate(l_parts)
        u = np.concatenate(u_parts)
    else:
        A_stack = sp.csc_matrix((0, n))
        l = np.zeros(0)
        u = np.zeros(0)
    return A_stack, l, u


# ============================================================================
# CPU SOLVERS
# ============================================================================

# ---------------------------------------------------------------------------
# OSQP
# ---------------------------------------------------------------------------

try:
    import osqp
    _OSQP_AVAILABLE = True
except ImportError as e:
    _OSQP_AVAILABLE = False
    print(f"[single_solver_interface] osqp unavailable: {e}", file=sys.stderr)


class OsqpSolver(SingleQPSolver):
    """OSQP (CPU). Format: min 0.5 x^T P x + q^T x  s.t.  l <= A x <= u."""
    device: ClassVar[str] = 'cpu'

    @property
    def name(self) -> str:
        return "osqp"

    def _prepare_data(self, data: SingleQPData) -> None:
        if not _OSQP_AVAILABLE:
            raise ImportError("osqp not installed")
        self._n = data.n
        self._P = sp.csc_matrix(data.P).astype(np.float64)
        self._q = np.asarray(data.c, dtype=np.float64)
        self._A_stack, self._l, self._u = _stack_box_constraints(data)

    @nvtx.annotate("osqp::setup", color=SOLVER_COLORS["osqp"])
    def setup(self) -> None:
        self._solver = osqp.OSQP()
        # OSQP is ADMM, not an IPM — the harness's IPM-tuned ``max_iter``
        # (often 250) starves it. Floor at 2000 iters and enable ``polish``
        # so a final direct-solve refinement recovers precision after the
        # ADMM iterate has bracketed the optimum.
        self._solver.setup(
            P=self._P, q=self._q,
            A=self._A_stack, l=self._l, u=self._u,
            eps_abs=self.tol_abs, eps_rel=self.tol_abs,
            max_iter=max(self.max_iter, 2000),
            polish=True,
            verbose=False,
        )

    @nvtx.annotate("osqp::solve", color=SOLVER_COLORS["osqp"])
    def solve(self) -> None:
        self._last = self._solver.solve()

    def _collect_result(self) -> SingleQPResult:
        r = self._last
        x = np.asarray(r.x) if r.x is not None else np.full(self._n, np.nan)
        status = str(r.info.status)
        return SingleQPResult(
            x=x[: self._n],
            n_iter=int(r.info.iter),
            obj=float(r.info.obj_val),
            status=status,
            solved=(status == 'solved'),
        )


# ---------------------------------------------------------------------------
# PIQP (CPU)
# ---------------------------------------------------------------------------

try:
    import piqp
    _PIQP_AVAILABLE = True
except ImportError as e:
    _PIQP_AVAILABLE = False
    print(f"[single_solver_interface] piqp unavailable: {e}", file=sys.stderr)


class PiqpSparseSolver(SingleQPSolver):
    """PIQP (sparse) on CPU. Format matches SingleQPData natively."""
    device: ClassVar[str] = 'cpu'

    @property
    def name(self) -> str:
        return "piqp-sparse"

    def _prepare_data(self, data: SingleQPData) -> None:
        if not _PIQP_AVAILABLE:
            raise ImportError("piqp not installed")
        self._P = sp.csc_matrix(data.P).astype(np.float64)
        self._A = sp.csc_matrix(data.A).astype(np.float64) if data.A is not None else None
        self._G = sp.csc_matrix(data.G).astype(np.float64) if data.G is not None else None

        def _arr(v):
            return None if v is None else np.asarray(v, dtype=np.float64)
        self._c = _arr(data.c)
        self._b = _arr(data.b)
        self._h_l = _arr(data.h_l)
        self._h_u = _arr(data.h_u)
        self._x_l = _arr(data.x_l)
        self._x_u = _arr(data.x_u)
        self._n = data.n

    @nvtx.annotate("piqp-sparse::setup", color=SOLVER_COLORS["piqp-sparse"])
    def setup(self) -> None:
        self._solver = piqp.SparseSolver()
        self._solver.settings.eps_abs = self.tol_abs
        self._solver.settings.max_iter = self.max_iter
        self._solver.settings.verbose = False
        self._solver.setup(
            P=self._P, c=self._c,
            A=self._A, b=self._b,
            G=self._G, h_l=self._h_l, h_u=self._h_u,
            x_l=self._x_l, x_u=self._x_u,
        )

    @nvtx.annotate("piqp-sparse::solve", color=SOLVER_COLORS["piqp-sparse"])
    def solve(self) -> None:
        self._last_status = self._solver.solve()

    def _collect_result(self) -> SingleQPResult:
        r = self._solver.result
        status = str(r.info.status)
        return SingleQPResult(
            x=np.asarray(r.x),
            n_iter=int(r.info.iter),
            obj=float(r.info.primal_obj),
            status=status,
            solved=status.endswith('SOLVED') or status.endswith('PIQP_SOLVED'),
        )


# ---------------------------------------------------------------------------
# Clarabel (CPU). Conic form:
#     min 0.5 x^T P x + q^T x  s.t.  A x + s = b,  s ∈ K
# K = ZeroCone(n_eq) × NonnegativeCone(n_ineq). We stack
#     [ A_eq;  G_+ ;  -G_- ;  +I_xu ;  -I_xl ]
# and drop rows whose corresponding bound is +/- inf.
# ---------------------------------------------------------------------------

try:
    import clarabel
    _CLARABEL_AVAILABLE = True
except ImportError as e:
    _CLARABEL_AVAILABLE = False
    print(f"[single_solver_interface] clarabel unavailable: {e}", file=sys.stderr)


class ClarabelSolver(SingleQPSolver):
    """Clarabel (CPU sparse). Builds zero + nonneg cone stack from SingleQPData."""
    device: ClassVar[str] = 'cpu'

    @property
    def name(self) -> str:
        return "clarabel"

    def _prepare_data(self, data: SingleQPData) -> None:
        if not _CLARABEL_AVAILABLE:
            raise ImportError("clarabel not installed")
        n = data.n

        zero_rows, zero_b = [], []
        nn_rows,   nn_b   = [], []

        if data.A is not None:
            zero_rows.append(sp.csc_matrix(data.A).astype(np.float64))
            zero_b.append(np.asarray(data.b, dtype=np.float64))

        if data.G is not None:
            G_csc = sp.csc_matrix(data.G).astype(np.float64)
            if data.h_u is not None:
                hu = np.asarray(data.h_u, dtype=np.float64)
                m = np.isfinite(hu)
                if m.any():
                    nn_rows.append(G_csc[m]); nn_b.append(hu[m])
            if data.h_l is not None:
                hl = np.asarray(data.h_l, dtype=np.float64)
                m = np.isfinite(hl)
                if m.any():
                    nn_rows.append(-G_csc[m]); nn_b.append(-hl[m])

        if data.x_u is not None:
            xu = np.asarray(data.x_u, dtype=np.float64)
            m = np.isfinite(xu)
            if m.any():
                nn_rows.append(sp.eye(n, format='csc').astype(np.float64)[m])
                nn_b.append(xu[m])
        if data.x_l is not None:
            xl = np.asarray(data.x_l, dtype=np.float64)
            m = np.isfinite(xl)
            if m.any():
                nn_rows.append(-sp.eye(n, format='csc').astype(np.float64)[m])
                nn_b.append(-xl[m])

        zero_csc = (sp.vstack(zero_rows, format='csc')
                    if zero_rows else sp.csc_matrix((0, n)))
        nn_csc = (sp.vstack(nn_rows, format='csc')
                  if nn_rows else sp.csc_matrix((0, n)))

        self._A = sp.vstack([zero_csc, nn_csc], format='csc')
        self._b = (np.concatenate(zero_b + nn_b)
                   if (zero_b or nn_b) else np.zeros(0))

        # Clarabel takes only the upper-triangular part of P.
        self._P = sp.triu(
            sp.csc_matrix(data.P).astype(np.float64), format='csc')
        self._q = np.asarray(data.c, dtype=np.float64)

        self._n_eq = int(zero_csc.shape[0])
        self._n_ineq = int(nn_csc.shape[0])
        self._n = n

    @nvtx.annotate("clarabel::setup", color=SOLVER_COLORS["clarabel"])
    def setup(self) -> None:
        cones = []
        if self._n_eq > 0:
            cones.append(clarabel.ZeroConeT(self._n_eq))
        if self._n_ineq > 0:
            cones.append(clarabel.NonnegativeConeT(self._n_ineq))

        settings = clarabel.DefaultSettings()
        settings.verbose = False
        settings.max_iter = self.max_iter
        # Tolerance names in clarabel-python; setattr with a try block so
        # an older binding without ``eps_abs`` still works (we'd fall back
        # to defaults).
        for attr in ('eps_abs', 'eps_rel'):
            try:
                setattr(settings, attr, self.tol_abs)
            except AttributeError:
                pass

        self._solver = clarabel.DefaultSolver(
            self._P, self._q, self._A, self._b, cones, settings)

    @nvtx.annotate("clarabel::solve", color=SOLVER_COLORS["clarabel"])
    def solve(self) -> None:
        self._last = self._solver.solve()

    def _collect_result(self) -> SingleQPResult:
        r = self._last
        status = str(r.status)
        return SingleQPResult(
            x=np.asarray(r.x)[: self._n],
            n_iter=int(r.iterations),
            obj=float(r.obj_val),
            status=status,
            solved=('solved' in status.lower()),
        )


# ---------------------------------------------------------------------------
# HPIPM (CPU, OCP-structured). Requires SingleQPData.ocp to carry the raw
# OCP problem object with ``ChainMassOCPProblem``'s attribute layout. For
# flat QPs (huber, portfolio, ...) this solver raises ValueError on
# ``_prepare_data`` and the benchmark harness records a clean FAILED entry.
# ---------------------------------------------------------------------------

try:
    from hpipm_python import (  # noqa: F401
        hpipm_ocp_qp_dim as _hpipm_dim,
        hpipm_ocp_qp as _hpipm_qp,
        hpipm_ocp_qp_sol as _hpipm_sol,
        hpipm_ocp_qp_solver as _hpipm_solver,
        hpipm_ocp_qp_solver_arg as _hpipm_arg,
    )
    _HPIPM_AVAILABLE = True
except ImportError as e:
    _HPIPM_AVAILABLE = False
    print(f"[single_solver_interface] hpipm unavailable: {e}", file=sys.stderr)


class HpipmSolver(SingleQPSolver):
    """HPIPM (CPU), native OCP QP interface.

    Consumes ``data.ocp`` directly. Assumes the OCP attribute layout used by
    :class:`examples.chain_mass_ocp.ChainMassOCPProblem`:
    augmented state ``nx = system.nx + system.nu``, input ``nu = system.nu``,
    stagewise dynamics ``(A, B)``, cost ``(Q, R, S, QN)``, initial state
    ``x0``, input bounds ``(ul, uu)``, general constraints ``(C, D, gl, gu)``,
    and box-style state bounds via ``system.nx_max / system.nu_max``.
    """
    device: ClassVar[str] = 'cpu'

    @property
    def name(self) -> str:
        return "hpipm"

    def _prepare_data(self, data: SingleQPData) -> None:
        if not _HPIPM_AVAILABLE:
            raise ImportError("hpipm_python not installed")
        ocp = data.ocp
        if ocp is None:
            raise ValueError(
                "hpipm requires SingleQPData.ocp with the raw OCP block "
                "structure (A, B, Q, R, S, QN, x0, ul, uu, C, D, gl, gu, "
                "N, nx, nu, system.{nx, nu_max, nx_max}). This problem "
                "doesn't provide it.")
        self._ocp = ocp
        self._n = data.n

    @nvtx.annotate("hpipm::setup", color=SOLVER_COLORS["hpipm"])
    def setup(self) -> None:
        ocp = self._ocp
        N = ocp.N
        nx_ocp = ocp.nx
        nu_ocp = ocp.nu
        nx_sys = ocp.system.nx
        ng = ocp.C.shape[0]

        # ---- Dimensions ----
        dim = _hpipm_dim(N)
        for k in range(N + 1):
            dim.set('nx', nx_ocp, k)
            dim.set('nu', nu_ocp if k < N else 0, k)
            dim.set('nbx', nx_ocp, k)
            dim.set('nbu', nu_ocp if k < N else 0, k)
            dim.set('ng', ng if 0 < k < N else 0, k)
            dim.set('ns', 0, k)

        # ---- QP data ----
        qp = _hpipm_qp(dim)
        for k in range(N):
            qp.set('A', ocp.A, k)
            qp.set('B', ocp.B, k)
            qp.set('Q', ocp.Q, k)
            qp.set('R', ocp.R, k)
            if np.any(ocp.S != 0):
                qp.set('S', ocp.S, k)

        # Terminal cost
        qp.set('Q', ocp.QN, N)

        Jbx = np.eye(nx_ocp)
        Jbu = np.eye(nu_ocp)
        for k in range(N + 1):
            if k == 0:
                # Pin the initial augmented state: [x0; 0_{nu}]
                x0_aug = np.zeros(nx_ocp)
                x0_aug[:nx_sys] = ocp.x0
                qp.set('Jbx', Jbx, k)
                qp.set('lbx', x0_aug, k)
                qp.set('ubx', x0_aug, k)
            else:
                lbx = np.concatenate(
                    [-ocp.system.nx_max * np.ones(nx_sys),
                     -ocp.system.nu_max * np.ones(nu_ocp)])
                ubx = np.concatenate(
                    [ocp.system.nx_max * np.ones(nx_sys),
                     ocp.system.nu_max * np.ones(nu_ocp)])
                qp.set('Jbx', Jbx, k)
                qp.set('lbx', lbx, k)
                qp.set('ubx', ubx, k)

            if k < N:
                qp.set('Jbu', Jbu, k)
                qp.set('lbu', ocp.ul, k)
                qp.set('ubu', ocp.uu, k)

            if 0 < k < N:
                qp.set('C', ocp.C, k)
                qp.set('D', ocp.D, k)
                qp.set('lg', ocp.gl, k)
                qp.set('ug', ocp.gu, k)

        arg = _hpipm_arg(dim, 'speed')
        arg.set('iter_max', self.max_iter)

        self._dim = dim
        self._qp = qp
        self._sol = _hpipm_sol(dim)
        self._solver = _hpipm_solver(dim, arg)

    @nvtx.annotate("hpipm::solve", color=SOLVER_COLORS["hpipm"])
    def solve(self) -> None:
        self._solver.solve(self._qp, self._sol)

    def _collect_result(self) -> SingleQPResult:
        ocp = self._ocp
        N = ocp.N
        sol = self._sol
        status = self._solver.get('status')

        # Rebuild flat z = [x0; u0; x1; u1; ...; x_N] and recompute
        # the objective from the trajectory.
        flat_parts = []
        obj = 0.0
        for k in range(N):
            xk = np.asarray(sol.get('x', k)).ravel()
            uk = np.asarray(sol.get('u', k)).ravel()
            obj += (0.5 * xk @ ocp.Q @ xk
                    + 0.5 * uk @ ocp.R @ uk
                    + uk @ (ocp.S @ xk))
            flat_parts.append(xk)
            flat_parts.append(uk)
        xN = np.asarray(sol.get('x', N)).ravel()
        obj += 0.5 * xN @ ocp.QN @ xN
        flat_parts.append(xN)
        x_flat = np.concatenate(flat_parts)[: self._n]

        # HPIPM status: 0=converged, 1=max_iter, 2=NaN res, 3=stop crit fail
        status_int = int(status)
        return SingleQPResult(
            x=x_flat,
            n_iter=-1,
            obj=float(obj),
            status=f"hpipm_status={status_int}",
            solved=(status_int == 0),
        )


# ---------------------------------------------------------------------------
# Gurobi (CPU). Commercial QP solver; barrier (interior-point) backend by
# default. We feed it through the vectorised ``addMVar`` /
# ``addMConstr`` API directly on scipy CSR — no Python-loop overhead
# unlike the cuOpt wrapper. Equalities, two-sided inequalities and box
# bounds map to native Gurobi constraints / variable bounds.
# ---------------------------------------------------------------------------

try:
    import gurobipy as _gurobipy
    _GUROBI_AVAILABLE = True
except ImportError as e:
    _GUROBI_AVAILABLE = False
    print(f"[single_solver_interface] gurobi unavailable: {e}", file=sys.stderr)


class GurobiSolver(SingleQPSolver):
    """Gurobi (CPU). Vectorised QP through ``addMVar`` / ``addMConstr``.

    Uses Gurobi's barrier method (``Method=2``, default for QP). Crossover
    is disabled (``Crossover=0``) so we get the pure interior-point time
    without the LP-simplex postprocessing — matching how OSQP / Clarabel
    are typically benchmarked.
    """
    device: ClassVar[str] = 'cpu'

    @property
    def name(self) -> str:
        return "gurobi"

    def _prepare_data(self, data: SingleQPData) -> None:
        if not _GUROBI_AVAILABLE:
            raise ImportError("gurobipy not installed")
        self._n = data.n
        self._P = sp.csr_matrix(data.P).astype(np.float64)
        # Symmetric P with explicit halving: Gurobi's QP form expects
        # ``x'Qx`` with no 1/2 factor, while ours is ``0.5 x'Px + c'x``.
        self._c = np.asarray(data.c, dtype=np.float64)

        if data.A is not None:
            self._A = sp.csr_matrix(data.A).astype(np.float64)
            self._b = np.asarray(data.b, dtype=np.float64)
        else:
            self._A = None
            self._b = None

        if data.G is not None:
            self._G = sp.csr_matrix(data.G).astype(np.float64)
            self._h_l = (np.asarray(data.h_l, dtype=np.float64)
                         if data.h_l is not None else None)
            self._h_u = (np.asarray(data.h_u, dtype=np.float64)
                         if data.h_u is not None else None)
        else:
            self._G = None
            self._h_l = None
            self._h_u = None

        # Gurobi accepts +/-GRB.INFINITY on variable bounds; numpy +/-inf
        # also works since gurobipy maps them.
        self._x_l = (np.asarray(data.x_l, dtype=np.float64)
                     if data.x_l is not None
                     else np.full(self._n, -np.inf))
        self._x_u = (np.asarray(data.x_u, dtype=np.float64)
                     if data.x_u is not None
                     else np.full(self._n, +np.inf))

    @nvtx.annotate("gurobi::setup", color=SOLVER_COLORS["gurobi"])
    def setup(self) -> None:
        env = _gurobipy.Env(empty=True)
        env.setParam('OutputFlag', 0)
        env.start()
        self._env = env
        m = _gurobipy.Model('qp', env=env)
        m.setParam('OutputFlag', 0)
        m.setParam('Method', 2)      # barrier
        m.setParam('Crossover', 0)   # no LP crossover after barrier
        m.setParam('BarConvTol', self.tol_abs)
        m.setParam('FeasibilityTol', max(self.tol_abs, 1e-9))
        m.setParam('OptimalityTol', max(self.tol_abs, 1e-9))
        m.setParam('BarIterLimit', self.max_iter)
        m.setParam('Threads', 1)     # match other CPU solvers

        x = m.addMVar(self._n, lb=self._x_l, ub=self._x_u, name='x')
        m.setObjective(0.5 * (x @ self._P @ x) + self._c @ x,
                       _gurobipy.GRB.MINIMIZE)
        if self._A is not None:
            m.addConstr(self._A @ x == self._b, name='eq')
        if self._G is not None:
            # Drop +/-inf rows directly via boolean masks before passing
            # to addConstr — Gurobi rejects inf RHS on inequalities.
            if self._h_u is not None:
                mask = np.isfinite(self._h_u)
                if mask.any():
                    G_u = self._G[mask]
                    m.addConstr(G_u @ x <= self._h_u[mask], name='ineq_u')
            if self._h_l is not None:
                mask = np.isfinite(self._h_l)
                if mask.any():
                    G_l = self._G[mask]
                    m.addConstr(G_l @ x >= self._h_l[mask], name='ineq_l')
        m.update()
        self._model = m
        self._x = x

    @nvtx.annotate("gurobi::solve", color=SOLVER_COLORS["gurobi"])
    def solve(self) -> None:
        # ``benchmark()`` invokes ``solve()`` several times (warmup +
        # n_repeats) on the SAME model. Gurobi caches the previous
        # optimum, so a second ``optimize()`` would return instantly
        # without doing any barrier iterations. ``reset()`` wipes the
        # solution + warm-start state so each timed solve does the full
        # work — matching how the other solver wrappers behave.
        self._model.reset()
        self._model.optimize()

    def _collect_result(self) -> SingleQPResult:
        m = self._model
        status = m.Status
        # Gurobi status codes: 2 = OPTIMAL, 3 = INFEASIBLE, 4 =
        # INF_OR_UNBD, 5 = UNBOUNDED, 13 = SUBOPTIMAL, 11 = ITERATION_LIMIT
        solved = (status == _gurobipy.GRB.OPTIMAL)
        status_names = {2: 'OPTIMAL', 3: 'INFEASIBLE', 4: 'INF_OR_UNBD',
                        5: 'UNBOUNDED', 11: 'ITERATION_LIMIT',
                        12: 'NUMERIC', 13: 'SUBOPTIMAL'}
        status_str = status_names.get(status, f'Gurobi status={status}')

        if solved:
            x = np.asarray(self._x.X, dtype=np.float64)
            obj = float(m.ObjVal)
        else:
            x = np.full(self._n, np.nan)
            obj = float('nan')

        # Use barrier iterations as the "iter" count (we disabled crossover
        # and forced ``Method=2``).
        try:
            n_iter = int(m.BarIterCount)
        except AttributeError:
            n_iter = int(getattr(m, 'IterCount', -1))

        return SingleQPResult(
            x=x, n_iter=n_iter, obj=obj,
            status=status_str, solved=solved,
        )


# ---------------------------------------------------------------------------
# QPALM (CPU). Augmented-Lagrangian QP. Same OSQP-style form:
#     min 0.5 x^T Q x + q^T x  s.t.  bmin <= A x <= bmax
# We reuse ``_stack_box_constraints`` to compose [A_eq; G; I].
# ---------------------------------------------------------------------------

try:
    import qpalm
    _QPALM_AVAILABLE = True
except ImportError as e:
    _QPALM_AVAILABLE = False
    print(f"[single_solver_interface] qpalm unavailable: {e}", file=sys.stderr)


class QpalmSolver(SingleQPSolver):
    """QPALM (CPU). Augmented-Lagrangian QP, OSQP-compatible interface."""
    device: ClassVar[str] = 'cpu'

    @property
    def name(self) -> str:
        return "qpalm"

    def _prepare_data(self, data: SingleQPData) -> None:
        if not _QPALM_AVAILABLE:
            raise ImportError("qpalm not installed (pip install qpalm)")
        self._n = data.n
        self._P = sp.csc_matrix(data.P).astype(np.float64)
        self._q = np.asarray(data.c, dtype=np.float64)
        self._A_stack, self._bmin, self._bmax = _stack_box_constraints(data)

    @nvtx.annotate("qpalm::setup", color=SOLVER_COLORS["qpalm"])
    def setup(self) -> None:
        qd = qpalm.Data(self._n, self._A_stack.shape[0])
        qd.Q = self._P
        qd.q = self._q
        qd.A = self._A_stack
        qd.bmin = self._bmin
        qd.bmax = self._bmax

        settings = qpalm.Settings()
        settings.verbose = False
        # Tolerance + iter knobs that QPALM exposes; protect with try/except
        # so an older binding without one of these still works.
        for attr in ('eps_abs', 'eps_rel'):
            try:
                setattr(settings, attr, self.tol_abs)
            except AttributeError:
                pass
        try:
            # QPALM is augmented-Lagrangian — needs more iters than an IPM.
            settings.max_iter = max(self.max_iter, 10_000)
        except AttributeError:
            pass

        self._solver = qpalm.Solver(qd, settings)

    @nvtx.annotate("qpalm::solve", color=SOLVER_COLORS["qpalm"])
    def solve(self) -> None:
        self._solver.solve()

    def _collect_result(self) -> SingleQPResult:
        sol = self._solver.solution
        info = self._solver.info
        x = np.asarray(sol.x)[: self._n]
        status = str(info.status)
        return SingleQPResult(
            x=x,
            n_iter=int(info.iter),
            obj=float(info.objective),
            status=status,
            solved=('solved' in status.lower()),
        )


# ---------------------------------------------------------------------------
# cyqlone QPALM-Cyqlone (CPU OCP solver). Augmented-state OCP form with
# uniform per-stage general constraints; consumes ``SingleQPData.ocp``
# directly (same OCP attribute layout as :class:`HpipmSolver`).
#
# cyqlone has no separate state/input box-bound facility — they must be
# encoded as general constraints. So we extend the per-stage constraint
# block with: rows I (nx, nx) for the augmented state, rows I (nu, nu)
# for the input, and the original (C, D) input-rate rows. Stage-0
# state-bound and input-rate rows are disabled via ±inf (x_0 is pinned by
# the rhs_eq mechanism; no u_{-1} exists).
# ---------------------------------------------------------------------------

try:
    import cyqlone as _cyqlone
    _CYQLONE_AVAILABLE = True
    # Always use the scalar variant. The simd4 / simd8 variants require
    # ``backend.processors`` and the horizon ``N`` to be multiples of the
    # SIMD width (see ``cyqlone.python.test.test_cyqlone_v2.SIMD_P_COMBOS``).
    # With our default ``processors=1`` and a generic ``N`` from the
    # benchmark sweep, only ``scalar`` is guaranteed to work. A SIMD
    # variant could be selected by the caller after profiling the problem.
    _CYQLONE_QPALM_CLS = _cyqlone.scalar.QPALM_Cyqlone
except ImportError as e:
    _CYQLONE_AVAILABLE = False
    print(f"[single_solver_interface] cyqlone unavailable: {e}", file=sys.stderr)


class CyqloneSolver(SingleQPSolver):
    """cyqlone QPALM-Cyqlone (CPU OCP solver).

    Requires ``SingleQPData.ocp`` with the OCP attribute layout used by
    :class:`examples.chain_mass_ocp.ChainMassOCPProblem` (same as
    :class:`HpipmSolver`). Builds an augmented-state OCP and encodes all
    state / input / input-rate bounds as general constraints (cyqlone has
    no separate box-bound facility).
    """
    device: ClassVar[str] = 'cpu'

    @property
    def name(self) -> str:
        return "cyqlone"

    def _prepare_data(self, data: SingleQPData) -> None:
        if not _CYQLONE_AVAILABLE:
            raise ImportError("cyqlone not installed")
        ocp = data.ocp
        if ocp is None:
            raise ValueError(
                "cyqlone requires SingleQPData.ocp with the raw OCP block "
                "structure (A, B, Q, R, S, QN, x0, ul, uu, C, D, gl, gu, "
                "N, nx, nu, system.{nx, nu_max, nx_max}).")

        N, nx, nu = ocp.N, ocp.nx, ocp.nu
        nx_sys = ocp.system.nx
        nx_max = ocp.system.nx_max
        nu_max = ocp.system.nu_max
        ny_orig = ocp.C.shape[0]

        # Extended per-stage general constraints: state-bound + input-bound
        # + original input-rate.
        ny_ext = nx + nu + ny_orig
        CD_block = np.zeros((ny_ext, nx + nu))
        CD_block[:nx, :nx] = np.eye(nx)
        CD_block[nx:nx + nu, nx:nx + nu] = np.eye(nu)
        CD_block[nx + nu:, :nx] = ocp.C
        CD_block[nx + nu:, nx:] = ocp.D
        CD = np.broadcast_to(CD_block, (N, ny_ext, nx + nu)).copy()
        # Terminal state bound.
        CN = np.eye(nx)
        ny_N = nx

        AB = np.broadcast_to(np.hstack([ocp.A, ocp.B]),
                             (N, nx, nx + nu)).copy()
        QRS_block = np.block([[ocp.Q, ocp.S.T], [ocp.S, ocp.R]])
        QRS = np.broadcast_to(QRS_block,
                              (N, nx + nu, nx + nu)).copy()
        QN = ocp.QN.copy()

        # x_0 pin: rhs_eq[:nx] = -x_0_aug
        x0_aug = np.zeros(nx)
        x0_aug[:nx_sys] = ocp.x0
        rhs_eq = np.zeros((N + 1) * nx)
        rhs_eq[:nx] = -x0_aug

        # Per-stage bound vectors.
        xlb_aug = np.concatenate([-nx_max * np.ones(nx_sys),
                                  -nu_max * np.ones(nu)])
        xub_aug = np.concatenate([+nx_max * np.ones(nx_sys),
                                  +nu_max * np.ones(nu)])
        rhs_lb = np.empty(N * ny_ext + ny_N)
        rhs_ub = np.empty(N * ny_ext + ny_N)
        for k in range(N):
            off = k * ny_ext
            # state-bound rows: disabled at k=0 (x_0 already pinned by rhs_eq).
            if k == 0:
                rhs_lb[off:off + nx] = -np.inf
                rhs_ub[off:off + nx] = +np.inf
            else:
                rhs_lb[off:off + nx] = xlb_aug
                rhs_ub[off:off + nx] = xub_aug
            # input-bound rows (active all stages).
            rhs_lb[off + nx:off + nx + nu] = ocp.ul
            rhs_ub[off + nx:off + nx + nu] = ocp.uu
            # input-rate rows: disabled at k=0 (no u_{-1}).
            if k == 0:
                rhs_lb[off + nx + nu:off + ny_ext] = -np.inf
                rhs_ub[off + nx + nu:off + ny_ext] = +np.inf
            else:
                rhs_lb[off + nx + nu:off + ny_ext] = ocp.gl
                rhs_ub[off + nx + nu:off + ny_ext] = ocp.gu
        # Terminal state bound.
        rhs_lb[N * ny_ext:N * ny_ext + ny_N] = xlb_aug
        rhs_ub[N * ny_ext:N * ny_ext + ny_N] = xub_aug

        # Linear cost. Chain-mass has c = 0; future problems with a non-zero
        # ``ocp.qr`` could read it directly from the OCP. Keep zero by default.
        qr = np.zeros(N * (nx + nu) + nx)

        self._cyq_ocp = _cyqlone.OCP(
            AB=np.asfortranarray(AB), CD=np.asfortranarray(CD),
            CN=np.asfortranarray(CN), QRS=np.asfortranarray(QRS),
            QN=np.asfortranarray(QN),
            qr=qr, rhs_eq=rhs_eq, rhs_lb=rhs_lb, rhs_ub=rhs_ub,
        )
        self._N, self._nx, self._nu = N, nx, nu
        self._nx_sys = nx_sys
        self._x0_sys = ocp.x0.copy()
        self._n = data.n
        self._P_flat = sp.csc_matrix(data.P).astype(np.float64)
        self._c_flat = np.asarray(data.c, dtype=np.float64)

    @nvtx.annotate("cyqlone::setup", color=SOLVER_COLORS["cyqlone"])
    def setup(self) -> None:
        backend = _cyqlone.CyQPALMBackendSettings()
        backend.processors = 1
        qset = _cyqlone.Settings()
        qset.verbose = False
        qset.tolerance = self.tol_abs
        qset.dual_tolerance = self.tol_abs
        qset.eq_constr_tolerance = self.tol_abs
        qset.max_outer_iter = max(self.max_iter, 50)
        self._solver = _CYQLONE_QPALM_CLS(self._cyq_ocp, backend, qset)

    @nvtx.annotate("cyqlone::solve", color=SOLVER_COLORS["cyqlone"])
    def solve(self) -> None:
        self._last_status = self._solver()

    def _collect_result(self) -> SingleQPResult:
        N, nx, nu, nx_sys = self._N, self._nx, self._nu, self._nx_sys
        sol = np.asarray(self._solver.solution)
        nux = nu + nx

        # Map cyqlone's augmented OCP solution (u_0, x_1_aug, u_1, ...,
        # x_N_aug) to SingleQPData z = [x_0_sys; u_0; x_1_sys; u_1; ...; x_N_sys].
        x_flat = np.empty(self._n)
        ofs = 0
        x_flat[ofs:ofs + nx_sys] = self._x0_sys
        ofs += nx_sys
        for k in range(N):
            x_flat[ofs:ofs + nu] = sol[k * nux:k * nux + nu]
            ofs += nu
            x_k1_aug = sol[k * nux + nu:(k + 1) * nux]
            x_flat[ofs:ofs + nx_sys] = x_k1_aug[:nx_sys]
            ofs += nx_sys

        obj = float(0.5 * x_flat @ (self._P_flat @ x_flat)
                    + self._c_flat @ x_flat)
        status = str(self._last_status)
        return SingleQPResult(
            x=x_flat,
            n_iter=int(self._solver.stats.outer_iter),
            obj=obj,
            status=status,
            solved=(self._last_status == _cyqlone.SolverStatus.Converged),
        )


# ============================================================================
# GPU SOLVERS
# ============================================================================

# ---------------------------------------------------------------------------
# qoco-gpu  (qoco with algebra='cuda'). Conic form:
#     min 0.5 x^T P x + c^T x  s.t.  A x = b,  h - G x in K.
# K is a Cartesian product of nonneg orthant (dim l) + SOCs. For QPs we use
# the orthant only; +/-inf bounds get dropped.
#
# NOTE: this requires the qoco wheel to have been built with CUDA support
# AND libcudss.so.1 to be on LD_LIBRARY_PATH at runtime.
# ---------------------------------------------------------------------------

try:
    import qoco
    from qoco.interface import algebra_available as _qoco_algebra_available
    _QOCO_AVAILABLE = True
    _QOCO_GPU_AVAILABLE = _qoco_algebra_available('cuda')
except ImportError as e:
    _QOCO_AVAILABLE = False
    _QOCO_GPU_AVAILABLE = False
    print(f"[single_solver_interface] qoco unavailable: {e}", file=sys.stderr)


class QocoSolver(SingleQPSolver):
    """qoco-gpu. Stacks G/h to a single nonneg-orthant cone block."""
    device: ClassVar[str] = 'gpu'
    _algebra: ClassVar[str] = 'cuda'  # set to 'builtin' for CPU qoco

    @property
    def name(self) -> str:
        return f"qoco-{self._algebra if self._algebra == 'cuda' else 'cpu'}".replace(
            "qoco-cuda", "qoco-gpu")

    def _prepare_data(self, data: SingleQPData) -> None:
        if not _QOCO_AVAILABLE:
            raise ImportError("qoco not installed")
        if self._algebra == 'cuda' and not _QOCO_GPU_AVAILABLE:
            raise ImportError("qoco was not built with CUDA support")
        n = data.n

        # Equality block
        if data.A is not None:
            self._A = sp.csc_matrix(data.A).astype(np.float64)
            self._b = np.asarray(data.b, dtype=np.float64)
            self._p = data.A.shape[0]
        else:
            self._A = None
            self._b = None
            self._p = 0

        # Inequality block: stack into one big nonneg-orthant cone
        # [ G x <= h_u  →  +G x <= h_u                  ]
        # [ G x >= h_l  →  -G x <= -h_l                 ]
        # [   x <= x_u  →  +I x <= x_u                  ]
        # [   x >= x_l  →  -I x <= -x_l                 ]
        # i.e. h - G_stack x ∈ R_+^m_total.
        # Drop rows where the bound is +/-inf — qoco doesn't accept inf RHS.
        rows, h_parts = [], []
        if data.G is not None:
            G_csc = sp.csc_matrix(data.G).astype(np.float64)
            if data.h_u is not None:
                hu = np.asarray(data.h_u, dtype=np.float64)
                mask = np.isfinite(hu)
                if mask.any():
                    rows.append(G_csc[mask])
                    h_parts.append(hu[mask])
            if data.h_l is not None:
                hl = np.asarray(data.h_l, dtype=np.float64)
                mask = np.isfinite(hl)
                if mask.any():
                    rows.append(-G_csc[mask])
                    h_parts.append(-hl[mask])
        if data.x_u is not None:
            xu = np.asarray(data.x_u, dtype=np.float64)
            mask = np.isfinite(xu)
            if mask.any():
                rows.append(sp.eye(n, format='csc').astype(np.float64)[mask])
                h_parts.append(xu[mask])
        if data.x_l is not None:
            xl = np.asarray(data.x_l, dtype=np.float64)
            mask = np.isfinite(xl)
            if mask.any():
                rows.append(-sp.eye(n, format='csc').astype(np.float64)[mask])
                h_parts.append(-xl[mask])

        if rows:
            self._G = sp.vstack(rows, format='csc')
            self._h = np.concatenate(h_parts)
        else:
            self._G = None
            self._h = None
        self._m = self._G.shape[0] if self._G is not None else 0

        # qoco expects only the upper-triangular part of P (passing the
        # full symmetric matrix returns ``Setup Error (Error Code 3)``).
        self._P = sp.triu(
            sp.csc_matrix(data.P).astype(np.float64), format='csc')
        self._c = np.asarray(data.c, dtype=np.float64)
        self._n = n

    @nvtx.annotate("qoco-gpu::setup", color=SOLVER_COLORS["qoco-gpu"])
    def setup(self) -> None:
        self._solver = qoco.QOCO(algebra=self._algebra)
        # qoco.setup is positional — see qoco-gpu-benchmarks/solvers.py.
        self._solver.setup(
            self._n, self._m, self._p,
            self._P, self._c,
            self._A, self._b,
            self._G, self._h,
            self._m,        # l: nonneg orthant dim (all rows)
            0,              # nsoc
            None,           # q (SOC dims)
            abstol=self.tol_abs, reltol=self.tol_abs,
            max_iters=self.max_iter,
            verbose=False,
        )

    @nvtx.annotate("qoco-gpu::solve", color=SOLVER_COLORS["qoco-gpu"])
    def solve(self) -> None:
        self._last = self._solver.solve()

    def _collect_result(self) -> SingleQPResult:
        r = self._last
        status = str(r.status)
        return SingleQPResult(
            x=np.asarray(r.x),
            n_iter=int(r.iters),
            obj=float(r.obj),
            status=status,
            solved=('solved' in status.lower()),
        )


# ---------------------------------------------------------------------------
# cuPIQP (GPU)
# ---------------------------------------------------------------------------

try:
    import cupy as cp
    from cupyx.scipy.sparse import csr_matrix as gpu_csr_matrix
    from cupiqp import (
        SparseLargeProblemSolver,
        DenseLargeProblemSolver,
        MultistageLargeProblemSolver,
        Status as CupiqpStatus,
    )
    _CUPIQP_AVAILABLE = True
except ImportError as e:
    _CUPIQP_AVAILABLE = False
    print(f"[single_solver_interface] cupiqp unavailable: {e}", file=sys.stderr)


class CupiqpSolverMixin(SingleQPSolver):
    """Shared logic for cuPIQP backends. Subclasses set ``_solver_cls``.

    Uses the ``*LargeProblemSolver`` variants from
    ``cupiqp.solver_large_problem``, which swap the warp tile-kernel inner
    loop for cupy axis-1 reductions — appropriate when ``max(n, p, m)`` is
    large enough that tile-kernel JIT compile time dominates first-solve
    latency. Numerically equivalent to the regular ``DenseSolver`` /
    ``SparseSolver`` classes.
    """
    device: ClassVar[str] = 'gpu'
    _solver_cls: ClassVar[type]
    _matrices_are_sparse: ClassVar[bool]

    def _prepare_data(self, data: SingleQPData) -> None:
        if not _CUPIQP_AVAILABLE:
            raise ImportError("cupiqp / cupy not installed")
        self._n = data.n

        kw: dict = {'c': cp.asarray(data.c, dtype=cp.float64)}

        if self._matrices_are_sparse:
            kw['P'] = gpu_csr_matrix(sp.csr_matrix(data.P).astype(np.float64))
            if data.A is not None:
                kw['A'] = gpu_csr_matrix(sp.csr_matrix(data.A).astype(np.float64))
                kw['b'] = cp.asarray(data.b, dtype=cp.float64)
            if data.G is not None:
                kw['G'] = gpu_csr_matrix(sp.csr_matrix(data.G).astype(np.float64))
                if data.h_l is not None:
                    kw['h_l'] = cp.asarray(data.h_l, dtype=cp.float64)
                if data.h_u is not None:
                    kw['h_u'] = cp.asarray(data.h_u, dtype=cp.float64)
        else:
            # dense_cholesky path: feed dense cupy arrays
            kw['P'] = cp.asarray(_maybe_dense(data.P), dtype=cp.float64)
            if data.A is not None:
                kw['A'] = cp.asarray(_maybe_dense(data.A), dtype=cp.float64)
                kw['b'] = cp.asarray(data.b, dtype=cp.float64)
            if data.G is not None:
                kw['G'] = cp.asarray(_maybe_dense(data.G), dtype=cp.float64)
                if data.h_l is not None:
                    kw['h_l'] = cp.asarray(data.h_l, dtype=cp.float64)
                if data.h_u is not None:
                    kw['h_u'] = cp.asarray(data.h_u, dtype=cp.float64)

        if data.x_l is not None:
            kw['x_l'] = cp.asarray(data.x_l, dtype=cp.float64)
        if data.x_u is not None:
            kw['x_u'] = cp.asarray(data.x_u, dtype=cp.float64)

        self._setup_kwargs = kw

    def setup(self) -> None:
        # ``kkt_solver`` is set in each concrete class's __init__; we don't
        # touch it here. NVTX name comes from ``self.name`` so the three
        # concrete subclasses (cupiqp-sparse / cupiqp-dense /
        # cupiqp-multistage) show up as distinct ranges in the timeline.
        with nvtx.annotate(f"{self.name}::setup",
                           color=SOLVER_COLORS.get(self.name, "gray")):
            self._solver = self._solver_cls()
            self._solver.settings.max_iter = self.max_iter
            self._solver.settings.eps_abs = self.tol_abs
            self._solver.settings.verbose = False
            with cp.cuda.Device(0):
                self._solver.setup(**self._setup_kwargs)

    def solve(self) -> None:
        with nvtx.annotate(f"{self.name}::solve",
                           color=SOLVER_COLORS.get(self.name, "gray")):
            with cp.cuda.Device(0):
                self._solver.solve()
                cp.cuda.Device(0).synchronize()

    def _collect_result(self) -> SingleQPResult:
        info = self._solver.result.info
        status_enum = info.status[0]   # Status enum (B=1)
        x_host = cp.asnumpy(self._solver.result.x[0])
        return SingleQPResult(
            x=x_host[: self._n],
            n_iter=int(info.iter[0]),
            obj=float(info.primal_obj[0]),
            status=str(status_enum),
            solved=(status_enum == CupiqpStatus.PIQP_SOLVED),
        )


class CupiqpSparseSolver(CupiqpSolverMixin):
    _solver_cls = SparseLargeProblemSolver if _CUPIQP_AVAILABLE else None
    _matrices_are_sparse = True

    @property
    def name(self) -> str:
        return "cupiqp-sparse"


class CupiqpDenseSolver(CupiqpSolverMixin):
    _solver_cls = DenseLargeProblemSolver if _CUPIQP_AVAILABLE else None
    _matrices_are_sparse = False

    @property
    def name(self) -> str:
        return "cupiqp-dense"


class CupiqpMultistageSolver(CupiqpSolverMixin):
    """cuPIQP with the multistage block-Cholesky KKT backend.

    Requires an :class:`benchmarks.problems.OCPProblem` instance attached
    to ``SingleQPData.ocp`` whose concrete subclass exposes the
    block-structured attributes ``ms_P, ms_c, ms_A, ms_b, ms_G, ms_h_u,
    ms_h_l, ms_x_u, ms_x_l`` (``BlockTridiagMat`` / ``BlockBidiagMat`` /
    ``BlockVec`` instances from
    :mod:`cupiqp.multistage.multistage_utils`). For problems without an
    ``ocp``, ``_prepare_data`` raises ``ValueError`` and the benchmark
    records a clean FAILED entry rather than crashing.
    """
    _solver_cls = MultistageLargeProblemSolver if _CUPIQP_AVAILABLE else None
    _matrices_are_sparse = False   # unused: we override _prepare_data

    @property
    def name(self) -> str:
        return "cupiqp-multistage"

    def _prepare_data(self, data: SingleQPData) -> None:
        if not _CUPIQP_AVAILABLE:
            raise ImportError("cupiqp / cupy not installed")
        ocp = data.ocp
        if ocp is None or not hasattr(ocp, 'ms_P'):
            raise ValueError(
                "cupiqp-multistage requires SingleQPData.ocp with "
                "block-structured ``ms_*`` attributes (P, c, A, b, G, "
                "h_u, h_l, x_u, x_l) typically built by an OCPProblem "
                "subclass such as ChainMassOCPProblem. This problem "
                "doesn't provide them.")
        self._n = data.n
        kw = {'P': ocp.ms_P, 'c': ocp.ms_c}
        for k in ('A', 'b', 'G', 'h_u', 'h_l', 'x_u', 'x_l'):
            v = getattr(ocp, f'ms_{k}', None)
            if v is not None:
                kw[k] = v
        self._setup_kwargs = kw


# ---------------------------------------------------------------------------
# cuClarabel  (Clarabel.jl + CUDA.jl via juliacall, direct_solve_method=:cudss)
#
# Mirrors qoco-org/qoco-gpu-benchmarks/solvers.py::solve_cuclarabel_direct:
# build a cupy CSR P and a cupy CSR A_combined = [A_eq; G_+; -G_-; +I; -I]
# (with bounds blocks added for finite x_l/x_u rows), pass the raw GPU
# pointers to Julia via Clarabel's PythonExt FFI helpers, and let
# Clarabel.jl call cuDSS for the KKT solve.
# ---------------------------------------------------------------------------

try:
    import juliacall  # noqa: F401  (only checking the bridge is importable)
    _CUCLARABEL_AVAILABLE = True
except ImportError as e:
    _CUCLARABEL_AVAILABLE = False
    print(f"[single_solver_interface] cuclarabel unavailable: {e}", file=sys.stderr)


class CuClarabelSolver(SingleQPSolver):
    """cuClarabel via Clarabel.jl + CUDA.jl, called from Python via juliacall.

    NOTE: requires Clarabel.jl + CUDA.jl + CUDA.CUSPARSE installed in the
    Julia environment, plus libcudss available at runtime.
    """
    device: ClassVar[str] = 'gpu'

    # Module-level Julia handle is cached so we only import Clarabel/CUDA
    # once per process.
    _jl_initialized: ClassVar[bool] = False
    _pyext = None

    @property
    def name(self) -> str:
        return "cuclarabel"

    @classmethod
    def _init_julia(cls):
        if cls._jl_initialized:
            return
        from juliacall import Main as jl
        jl.seval("using Clarabel, LinearAlgebra, SparseArrays")
        jl.seval("using CUDA, CUDA.CUSPARSE")
        cls._pyext = jl.Base.get_extension(jl.Clarabel, jl.Symbol("PythonExt"))
        if cls._pyext is None:
            raise RuntimeError(
                "Clarabel.PythonExt extension not loaded — make sure your "
                "Clarabel.jl version supports the Python CUDA bridge.")
        cls._jl_initialized = True

    def _prepare_data(self, data: SingleQPData) -> None:
        if not _CUCLARABEL_AVAILABLE:
            raise ImportError("juliacall not installed")
        if not _CUPIQP_AVAILABLE:
            # We rely on cupy + cupyx.scipy.sparse, both pulled in by the
            # cupiqp dep block; if that import failed, cupy isn't here.
            raise ImportError("cupy is required for cuclarabel")
        n = data.n

        # Compose Clarabel constraint blocks: Ax + s = b, s ∈ {Zero, Nonneg}
        zero_rows, zero_b = [], []
        nn_rows,   nn_b   = [], []

        if data.A is not None:
            zero_rows.append(sp.csr_matrix(data.A).astype(np.float64))
            zero_b.append(np.asarray(data.b, dtype=np.float64))

        if data.G is not None:
            G_csr = sp.csr_matrix(data.G).astype(np.float64)
            if data.h_u is not None:
                hu = np.asarray(data.h_u, dtype=np.float64)
                m = np.isfinite(hu)
                if m.any():
                    nn_rows.append(G_csr[m]); nn_b.append(hu[m])
            if data.h_l is not None:
                hl = np.asarray(data.h_l, dtype=np.float64)
                m = np.isfinite(hl)
                if m.any():
                    nn_rows.append(-G_csr[m]); nn_b.append(-hl[m])

        if data.x_u is not None:
            xu = np.asarray(data.x_u, dtype=np.float64)
            m = np.isfinite(xu)
            if m.any():
                nn_rows.append(sp.eye(n, format='csr').astype(np.float64)[m])
                nn_b.append(xu[m])
        if data.x_l is not None:
            xl = np.asarray(data.x_l, dtype=np.float64)
            m = np.isfinite(xl)
            if m.any():
                nn_rows.append(-sp.eye(n, format='csr').astype(np.float64)[m])
                nn_b.append(-xl[m])

        zero_csr = (sp.vstack(zero_rows, format='csr')
                    if zero_rows else sp.csr_matrix((0, n)))
        nn_csr = (sp.vstack(nn_rows, format='csr')
                  if nn_rows else sp.csr_matrix((0, n)))

        A_combined = sp.vstack([zero_csr, nn_csr], format='csr')
        b_combined = (np.concatenate(zero_b + nn_b)
                      if (zero_b or nn_b) else np.zeros(0))

        # Move everything to GPU once. Indices/indptr must be int32 for
        # CuSparseMatrixCSR; PythonExt expects raw pointers.
        P_csr = sp.csr_matrix(data.P).astype(np.float64)

        def _to_gpu_csr(M_csr):
            G = gpu_csr_matrix(M_csr)
            G.indices = G.indices.astype(cp.int32)
            G.indptr = G.indptr.astype(cp.int32)
            G.data = cp.ascontiguousarray(G.data, dtype=cp.float64)
            return G

        self._Pgpu = _to_gpu_csr(P_csr)
        self._Agpu = _to_gpu_csr(A_combined)
        self._qgpu = cp.ascontiguousarray(
            cp.asarray(data.c, dtype=cp.float64))
        self._bgpu = cp.ascontiguousarray(
            cp.asarray(b_combined, dtype=cp.float64))

        self._zero_dim = int(zero_csr.shape[0])
        self._nn_dim   = int(nn_csr.shape[0])
        self._n = n

    @nvtx.annotate("cuclarabel::setup", color=SOLVER_COLORS["cuclarabel"])
    def setup(self) -> None:
        self._init_julia()
        from juliacall import Main as jl
        pyext = self._pyext

        # Hand GPU pointers to Julia.
        if self._Pgpu.nnz != 0:
            jl.P = pyext.cupy_to_cucsrmat(
                jl.Float64,
                int(self._Pgpu.data.data.ptr),
                int(self._Pgpu.indices.data.ptr),
                int(self._Pgpu.indptr.data.ptr),
                self._Pgpu.shape[0], self._Pgpu.shape[1],
                self._Pgpu.nnz,
            )
        else:
            jl.seval(
                f"P = CuSparseMatrixCSR(sparse(Float64[], Float64[], "
                f"Float64[], {self._n}, {self._n}))"
            )

        jl.q = pyext.cupy_to_cuvector(
            jl.Float64, int(self._qgpu.data.ptr), self._qgpu.size)
        jl.A = pyext.cupy_to_cucsrmat(
            jl.Float64,
            int(self._Agpu.data.data.ptr),
            int(self._Agpu.indices.data.ptr),
            int(self._Agpu.indptr.data.ptr),
            self._Agpu.shape[0], self._Agpu.shape[1],
            self._Agpu.nnz,
        )
        jl.b = pyext.cupy_to_cuvector(
            jl.Float64, int(self._bgpu.data.ptr), self._bgpu.size)

        jl.seval("cones = Clarabel.SupportedCone[]")
        if self._zero_dim > 0:
            jl.seval(f"push!(cones, Clarabel.ZeroConeT({self._zero_dim}))")
        if self._nn_dim > 0:
            jl.seval(f"push!(cones, Clarabel.NonnegativeConeT({self._nn_dim}))")

        jl.seval(f"""
            settings = Clarabel.Settings(
                direct_solve_method = :cudss,
                tol_gap_abs = {self.tol_abs},
                tol_gap_rel = {self.tol_abs},
                tol_feas    = {self.tol_abs},
                max_iter    = {self.max_iter},
                verbose     = false,
            )
            solver = Clarabel.Solver(settings)
            solver = Clarabel.setup!(solver, P, q, A, b, cones)
        """)

    @nvtx.annotate("cuclarabel::solve", color=SOLVER_COLORS["cuclarabel"])
    def solve(self) -> None:
        from juliacall import Main as jl
        # Solve, then drain CUDA.jl's stream from the Julia side and the
        # CuPy stream from the Python side, both inside the timed window.
        #
        # Why both syncs are needed: Clarabel.jl + CUDA.jl issue kernels
        # on Julia's own CUDA stream (within the shared primary context).
        # ``Clarabel.solve!`` returns as soon as the kernels are queued —
        # not when they finish. CuPy's ``Device.synchronize()`` only
        # waits on streams CuPy tracks, which doesn't include Julia's
        # private stream, so the cuopt-side timing window can close
        # before the GPU is actually done. The result is a
        # ``solve_time_ms`` that underestimates the real cost (we've
        # observed factor-of-2 underreports on portfolio n=30000).
        # An explicit ``CUDA.synchronize()`` from Julia waits on its own
        # streams; the trailing CuPy sync remains for symmetry / safety.
        jl.seval("Clarabel.solve!(solver); CUDA.synchronize()")
        cp.cuda.Device(0).synchronize()

    def _collect_result(self) -> SingleQPResult:
        from juliacall import Main as jl
        x_jl = jl.seval("Array(solver.solution.x)")  # GPU → host vector
        x_np = np.asarray(x_jl)[: self._n]
        status = str(jl.seval("string(solver.solution.status)"))
        return SingleQPResult(
            x=x_np,
            n_iter=int(jl.seval("solver.solution.iterations")),
            obj=float(jl.seval("solver.solution.obj_val")),
            status=status,
            solved=('solved' in status.lower()),
        )


# ---------------------------------------------------------------------------
# cuOpt  (NVIDIA, GPU). LP/QP solver via cuopt.linear_programming.
#
# Construction is loop-heavy (one Python addConstraint per row), so setup
# can be slow for large m. solve() blocks until the GPU work is done.
# ---------------------------------------------------------------------------

# IMPORTANT: cuopt is **not** imported eagerly. Loading
# ``cuopt.linear_programming`` at module import time spins up RMM /
# cuDF / Arrow CUDA state that subsequently corrupts cupy allocations
# made by *other* solvers (cupiqp-sparse in particular) — the symptom
# is ``cudf::fatal_cuda_error / CUDA_ERROR_ILLEGAL_ADDRESS`` during a
# later cupiqp solve. By using ``importlib.util.find_spec`` for the
# availability probe and deferring the real import into ``CuoptSolver``
# methods, worker subprocesses that don't run cuOpt never pay this cost.
import importlib.util as _importlib_util
_CUOPT_AVAILABLE = _importlib_util.find_spec("cuopt") is not None
if not _CUOPT_AVAILABLE:
    print("[single_solver_interface] cuopt unavailable (package not found)",
          file=sys.stderr)


class CuoptSolver(SingleQPSolver):
    """NVIDIA cuOpt LP/QP solver."""
    device: ClassVar[str] = 'gpu'

    @property
    def name(self) -> str:
        return "cuopt"

    def _prepare_data(self, data: SingleQPData) -> None:
        if not _CUOPT_AVAILABLE:
            raise ImportError("cuopt not installed")
        # Lazy import — see the comment block above CuoptSolver. Pulling
        # cuopt into this process has heavy side effects on the CUDA
        # context, so we only do it for workers that actually run cuOpt.
        from cuopt.linear_programming.problem import (
            Problem as _CuoptProblem,
            QuadraticExpression as _CuoptQuadExpr,
            MINIMIZE as _CUOPT_MIN,
        )
        from cuopt.linear_programming.solver_settings import (
            SolverSettings as _CuoptSettings,
        )
        self._CuoptProblem = _CuoptProblem
        self._CuoptQuadExpr = _CuoptQuadExpr
        self._CUOPT_MIN = _CUOPT_MIN
        self._CuoptSettings = _CuoptSettings

        self._n = data.n

        # Cache CSR/COO host structures so setup() only builds the cuOpt
        # Problem object (no per-call sparse conversion).
        self._P_coo = sp.csr_matrix(data.P).astype(np.float64).tocoo()
        self._c = np.asarray(data.c, dtype=np.float64)

        if data.A is not None:
            self._A_csr = sp.csr_matrix(data.A).astype(np.float64)
            self._b = np.asarray(data.b, dtype=np.float64)
        else:
            self._A_csr = None
            self._b = None

        if data.G is not None:
            self._G_csr = sp.csr_matrix(data.G).astype(np.float64)
            self._h_l = (np.asarray(data.h_l, dtype=np.float64)
                         if data.h_l is not None else None)
            self._h_u = (np.asarray(data.h_u, dtype=np.float64)
                         if data.h_u is not None else None)
        else:
            self._G_csr = None
            self._h_l = None
            self._h_u = None

        self._x_l = (np.asarray(data.x_l, dtype=np.float64)
                     if data.x_l is not None else None)
        self._x_u = (np.asarray(data.x_u, dtype=np.float64)
                     if data.x_u is not None else None)

    @nvtx.annotate("cuopt::setup", color=SOLVER_COLORS["cuopt"])
    def setup(self) -> None:
        n = self._n
        prob = self._CuoptProblem("singleQP")

        # Variables with finite bounds; +/-inf -> +/-inf sentinel (NOT
        # None). cuOpt 26.2.0's presolver mishandles ``ub=None`` /
        # ``lb=None`` — the verbose log shows
        # ``Variable bounds range: [0e+00, 0e+00]`` and the solve exits
        # with ``Status=2`` (infeasible) even on a trivial 5-variable LP
        # like ``min sum(-x_i) s.t. sum(x)=1, x>=0``. Passing ``np.inf``
        # instead makes cuOpt treat the side as unbounded correctly.
        variables = []
        for i in range(n):
            lb = (float(self._x_l[i])
                  if (self._x_l is not None and np.isfinite(self._x_l[i]))
                  else -np.inf)
            ub = (float(self._x_u[i])
                  if (self._x_u is not None and np.isfinite(self._x_u[i]))
                  else np.inf)
            variables.append(prob.addVariable(lb=lb, ub=ub))

        # Objective:  0.5 x^T P x + c^T x
        Pc = self._P_coo
        qvars1 = [variables[int(i)] for i in Pc.row]
        qvars2 = [variables[int(j)] for j in Pc.col]
        qcoeffs = (0.5 * Pc.data).tolist()
        quad = self._CuoptQuadExpr(
            qvars1=qvars1, qvars2=qvars2, qcoefficients=qcoeffs,
            vars=variables, coefficients=self._c.tolist(),
        )
        prob.setObjective(quad, sense=self._CUOPT_MIN)

        # Equality constraints: A_eq x = b
        if self._A_csr is not None:
            A = self._A_csr
            b = self._b
            for i in range(A.shape[0]):
                s, e = A.indptr[i], A.indptr[i + 1]
                cols = A.indices[s:e]
                vals = A.data[s:e]
                expr = sum(float(v) * variables[int(j)]
                           for j, v in zip(cols, vals))
                prob.addConstraint(expr == float(b[i]))

        # Two-sided inequality constraints — cuOpt expects single-sided, so
        # we add a separate <= and >= per finite bound.
        if self._G_csr is not None:
            G = self._G_csr
            for i in range(G.shape[0]):
                s, e = G.indptr[i], G.indptr[i + 1]
                cols = G.indices[s:e]
                vals = G.data[s:e]
                expr = sum(float(v) * variables[int(j)]
                           for j, v in zip(cols, vals))
                if self._h_u is not None and np.isfinite(self._h_u[i]):
                    prob.addConstraint(expr <= float(self._h_u[i]))
                if self._h_l is not None and np.isfinite(self._h_l[i]):
                    prob.addConstraint(expr >= float(self._h_l[i]))

        self._prob = prob
        self._variables = variables
        self._settings = self._CuoptSettings()
        self._settings.set_optimality_tolerance(self.tol_abs)
        self._settings.set_parameter("log_to_console", "0")

    @nvtx.annotate("cuopt::solve", color=SOLVER_COLORS["cuopt"])
    def solve(self) -> None:
        self._prob.solve(self._settings)

    def _collect_result(self) -> SingleQPResult:
        prob = self._prob

        # 1. Recover the primal solution. The attribute differs across
        #    cuOpt versions (.X like Gurobi, or via prob.solution()).
        x_np = np.full(self._n, np.nan)
        try:
            x_np = np.array([float(v.X) for v in self._variables])
        except Exception:
            try:
                x_np = np.asarray(prob.solution(), dtype=np.float64)
            except Exception:
                pass

        # 2. ``prob.Status`` is cuOpt's authoritative outcome. ``ObjValue``
        #    is set to NaN by cuOpt when the solver bailed (numerical
        #    issues, infeasibility, etc.), so we can't infer solved-ness
        #    from "ObjValue is not None" alone.
        #
        #    cuOpt status codes (LP/QP):
        #        1 = Optimal,  2 = Infeasible,  3 = Unbounded,
        #        0 = no solution / unknown.
        status_code = getattr(prob, 'Status', None)
        obj_val = prob.ObjValue
        obj_finite = (obj_val is not None
                      and np.isfinite(float(obj_val)))

        # Note: we deliberately *don't* require ``x_np`` to be finite —
        # cuOpt sometimes returns Status=1 with ``ObjValue`` finite but
        # leaves ``v.X`` unpopulated for individual variables (returning
        # NaN for some entries). Those runs are still optimal from cuOpt's
        # perspective; downstream consumers should check ``solved`` /
        # ``status``, not the per-element finiteness of ``x``.
        # ``bool(...)`` so the result is JSON-serializable (a numpy/cython
        # scalar would fail ``json.dump``).
        solved = bool(status_code == 1 and obj_finite)

        # 3. Objective: prefer cuOpt's reported value when it's finite,
        #    else recompute from the primal as ``0.5 x^T P x + c^T x``.
        if obj_finite:
            obj = float(obj_val)
        elif bool(np.all(np.isfinite(x_np))):
            Px = self._P_coo.tocsr() @ x_np
            obj = float(0.5 * x_np @ Px + self._c @ x_np)
        else:
            obj = float('nan')

        # 4. Iteration count: cuOpt's Python API doesn't expose it as an
        #    attribute, but its verbose stdout prints "found in N
        #    iterations" on successful exit. Only worth running the
        #    extra verbose solve if cuOpt actually converged before.
        n_iter = (self._extract_iter_count_via_verbose_solve()
                  if solved else -1)

        return SingleQPResult(
            x=x_np,
            n_iter=n_iter,
            obj=obj,
            status=(f'OPTIMAL (status={status_code})' if solved
                    else f'FAILED (status={status_code})'),
            solved=solved,
        )

    def _extract_iter_count_via_verbose_solve(self) -> int:
        """Run one verbose solve, grep ``"found in N iterations"`` out of stdout.

        Stdout is duplicated at the file-descriptor level so we capture
        cuOpt's C-side log too (a plain ``contextlib.redirect_stdout``
        only catches Python-level prints). ``libc.fflush(NULL)`` is
        called before swapping the fd back so libc's stdio buffer drains
        into our temp file rather than into the parent's stdout pipe.
        Returns ``-1`` if the regex doesn't find a match.
        """
        import ctypes
        import re
        import tempfile

        verbose_settings = self._CuoptSettings()
        verbose_settings.set_optimality_tolerance(self.tol_abs)
        # leave log_to_console at its default so cuOpt prints the
        # iteration summary

        with tempfile.NamedTemporaryFile(
                mode='w', suffix='.txt', delete=False) as f:
            log_path = f.name

        libc = ctypes.CDLL(None)
        stdout_fd = os.dup(1)
        try:
            libc.fflush(None)               # drain libc stdio first
            with open(log_path, 'w') as f:
                os.dup2(f.fileno(), 1)
                self._prob.solve(verbose_settings)
                libc.fflush(None)           # drain cuOpt's writes
                os.dup2(stdout_fd, 1)
            with open(log_path) as f:
                log = f.read()
            match = re.search(r'found in (\d+) iterations', log)
            return int(match.group(1)) if match else -1
        finally:
            os.close(stdout_fd)
            try:
                os.unlink(log_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Moreau (PyTorch wrapper). Conic form same as cuclarabel/qoco-gpu:
#     A_eq → ZeroCone(p),  G_+ / -G_- / +I_xu / -I_xl → NonnegCone(l).
# Moreau bundles its own cuDSS so it doesn't depend on a system libcudss.
# We feed it as a "batched" problem with B=1 (moreau is natively batched).
# ---------------------------------------------------------------------------

try:
    import torch as _torch
    import moreau as _moreau
    from moreau.torch import Solver as _MoreauTorchSolver
    _MOREAU_TORCH_AVAILABLE = True
except ImportError as e:
    _MOREAU_TORCH_AVAILABLE = False
    print(f"[single_solver_interface] moreau-torch unavailable: {e}", file=sys.stderr)


class MoreauTorchSolver(SingleQPSolver):
    """Moreau IPM via the PyTorch wrapper (GPU, cuDSS-backed)."""
    device: ClassVar[str] = 'gpu'

    @property
    def name(self) -> str:
        return "moreau-torch"

    def _prepare_data(self, data: SingleQPData) -> None:
        if not _MOREAU_TORCH_AVAILABLE:
            raise ImportError("moreau / torch not installed")
        n = data.n

        # Same conic stacking pattern as cuclarabel: zero-cone block first,
        # nonneg-cone block second, +/-inf rows dropped.
        zero_rows, zero_b = [], []
        nn_rows, nn_b = [], []

        if data.A is not None:
            zero_rows.append(sp.csr_matrix(data.A).astype(np.float64))
            zero_b.append(np.asarray(data.b, dtype=np.float64))

        if data.G is not None:
            G_csr = sp.csr_matrix(data.G).astype(np.float64)
            if data.h_u is not None:
                hu = np.asarray(data.h_u, dtype=np.float64)
                m = np.isfinite(hu)
                if m.any():
                    nn_rows.append(G_csr[m]); nn_b.append(hu[m])
            if data.h_l is not None:
                hl = np.asarray(data.h_l, dtype=np.float64)
                m = np.isfinite(hl)
                if m.any():
                    nn_rows.append(-G_csr[m]); nn_b.append(-hl[m])

        if data.x_u is not None:
            xu = np.asarray(data.x_u, dtype=np.float64)
            m = np.isfinite(xu)
            if m.any():
                nn_rows.append(sp.eye(n, format='csr').astype(np.float64)[m])
                nn_b.append(xu[m])
        if data.x_l is not None:
            xl = np.asarray(data.x_l, dtype=np.float64)
            m = np.isfinite(xl)
            if m.any():
                nn_rows.append(-sp.eye(n, format='csr').astype(np.float64)[m])
                nn_b.append(-xl[m])

        zero_csr = (sp.vstack(zero_rows, format='csr')
                    if zero_rows else sp.csr_matrix((0, n)))
        nn_csr = (sp.vstack(nn_rows, format='csr')
                  if nn_rows else sp.csr_matrix((0, n)))
        A_cone = sp.vstack([zero_csr, nn_csr], format='csr')
        b_cone = (np.concatenate(zero_b + nn_b)
                  if (zero_b or nn_b) else np.zeros(0))

        self._P_sp = sp.csr_matrix(data.P).astype(np.float64)
        self._A_sp = A_cone
        self._num_zero = int(zero_csr.shape[0])
        self._num_nonneg = int(nn_csr.shape[0])
        self._n = n

        # Push values to CUDA up front so solve() doesn't pay H2D cost.
        # Moreau's torch wrapper expects batched inputs; we use B=1.
        self._P_vals = _torch.tensor(self._P_sp.data[None, :],
                                     dtype=_torch.float64, device='cuda')
        self._A_vals = _torch.tensor(self._A_sp.data[None, :],
                                     dtype=_torch.float64, device='cuda')
        self._q = _torch.tensor(np.asarray(data.c, dtype=np.float64)[None, :],
                                dtype=_torch.float64, device='cuda')
        self._b = _torch.tensor(b_cone[None, :],
                                dtype=_torch.float64, device='cuda')

    @nvtx.annotate("moreau-torch::setup", color=SOLVER_COLORS["moreau-torch"])
    def setup(self) -> None:
        cones = _moreau.Cones(num_zero_cones=self._num_zero,
                              num_nonneg_cones=self._num_nonneg)
        ipm_settings = _moreau.IPMSettings(
            direct_solve_method="cudss",
            tol_feas=self.tol_abs,
            tol_gap_abs=self.tol_abs,
            tol_gap_rel=self.tol_abs,
        )
        settings = _moreau.Settings(
            batch_size=1,
            max_iter=self.max_iter,
            enable_grad=False,
            ipm_settings=ipm_settings,
            device="cuda",
        )
        self._solver = _MoreauTorchSolver(
            n=self._n, m=self._A_sp.shape[0],
            P_row_offsets=_torch.tensor(self._P_sp.indptr, dtype=_torch.int32),
            P_col_indices=_torch.tensor(self._P_sp.indices, dtype=_torch.int32),
            A_row_offsets=_torch.tensor(self._A_sp.indptr, dtype=_torch.int32),
            A_col_indices=_torch.tensor(self._A_sp.indices, dtype=_torch.int32),
            cones=cones,
            settings=settings,
        )

    @nvtx.annotate("moreau-torch::solve", color=SOLVER_COLORS["moreau-torch"])
    def solve(self) -> None:
        with _torch.no_grad():
            self._last_sol = self._solver.solve(
                self._P_vals, self._A_vals, self._q, self._b)
        _torch.cuda.synchronize()

    def _collect_result(self) -> SingleQPResult:
        info = self._solver.info
        # info.status / .iterations are sometimes scalars when batch_size=1
        # and sometimes lists/arrays — handle both.
        st_field = info.status
        st = st_field[0] if hasattr(st_field, '__len__') else st_field
        it_field = info.iterations
        try:
            n_iter = int(it_field[0])
        except (TypeError, IndexError):
            n_iter = int(it_field)
        x_np = self._last_sol.x[0].cpu().numpy()
        x = x_np[: self._n]

        # Moreau's torch wrapper doesn't expose primal_obj on info, so we
        # compute it from the returned primal: 0.5 x^T P x + c^T x.
        c_host = self._q[0].detach().cpu().numpy()
        Px = self._P_sp @ x
        obj = float(0.5 * np.dot(x, Px) + np.dot(c_host, x))

        valid = {
            _moreau.SolverStatus.Solved,
            _moreau.SolverStatus.AlmostSolved,
        }
        return SingleQPResult(
            x=x,
            n_iter=n_iter,
            obj=obj,
            status=str(st),
            solved=(st in valid),
        )


# ============================================================================
# Registry helpers
# ============================================================================

ALL_SOLVERS: List[type] = [
    OsqpSolver,
    PiqpSparseSolver,
    ClarabelSolver,
    HpipmSolver,
    QpalmSolver,
    CyqloneSolver,
    GurobiSolver,
    CupiqpSparseSolver,
    CupiqpMultistageSolver,
    # CupiqpDenseSolver,
    QocoSolver,
    CuClarabelSolver,
    CuoptSolver,
    # MoreauTorchSolver,
]


def available_solvers(device: Optional[str] = None) -> List[type]:
    """Return solver classes whose dependencies are importable.

    Parameters
    ----------
    device : 'cpu' | 'gpu' | None
        If given, restricts to the matching device.
    """
    flags = {
        OsqpSolver: _OSQP_AVAILABLE,
        PiqpSparseSolver: _PIQP_AVAILABLE,
        ClarabelSolver: _CLARABEL_AVAILABLE,
        HpipmSolver: _HPIPM_AVAILABLE,
        QpalmSolver: _QPALM_AVAILABLE,
        CyqloneSolver: _CYQLONE_AVAILABLE,
        GurobiSolver: _GUROBI_AVAILABLE,
        QocoSolver: _QOCO_AVAILABLE and _QOCO_GPU_AVAILABLE,
        CupiqpSparseSolver: _CUPIQP_AVAILABLE,
        CupiqpMultistageSolver: _CUPIQP_AVAILABLE,
        # CupiqpDenseSolver: _CUPIQP_AVAILABLE,
        CuClarabelSolver: _CUCLARABEL_AVAILABLE and _CUPIQP_AVAILABLE,
        CuoptSolver: _CUOPT_AVAILABLE,
        MoreauTorchSolver: _MOREAU_TORCH_AVAILABLE,
    }
    out = [cls for cls in ALL_SOLVERS if flags.get(cls, False)]
    if device is not None:
        out = [cls for cls in out if cls.device == device]
    return out
