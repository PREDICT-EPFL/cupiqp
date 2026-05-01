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
    CPU : osqp, piqp-sparse
    GPU : cupiqp-sparse, cupiqp-dense,
          qoco-gpu (algebra='cuda'),
          cuclarabel (Clarabel.jl + CUDA.jl via juliacall),
          cuopt (NVIDIA cuOpt LP/QP),
          moreau-torch (moreau via PyTorch, cuDSS bundled)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Union, ClassVar, List
import time

import numpy as np
import scipy.sparse as sp


# ============================================================================
# Shared data / result types
# ============================================================================

SparseOrDense = Union[np.ndarray, sp.spmatrix]


@dataclass
class SingleQPData:
    """Single-problem QP data. Matrices may be dense numpy or scipy sparse."""
    P: SparseOrDense                              # (n, n)
    c: np.ndarray                                 # (n,)
    A: Optional[SparseOrDense] = None             # (p, n)
    b: Optional[np.ndarray] = None                # (p,)
    G: Optional[SparseOrDense] = None             # (m, n)
    h_l: Optional[np.ndarray] = None              # (m,)
    h_u: Optional[np.ndarray] = None              # (m,)
    x_l: Optional[np.ndarray] = None              # (n,)
    x_u: Optional[np.ndarray] = None              # (n,)

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
except ImportError:
    _OSQP_AVAILABLE = False


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

    def setup(self) -> None:
        self._solver = osqp.OSQP()
        self._solver.setup(
            P=self._P, q=self._q,
            A=self._A_stack, l=self._l, u=self._u,
            eps_abs=self.tol_abs, eps_rel=self.tol_abs,
            max_iter=self.max_iter,
            verbose=False,
        )

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
except ImportError:
    _PIQP_AVAILABLE = False


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
except ImportError:
    _QOCO_AVAILABLE = False
    _QOCO_GPU_AVAILABLE = False


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

        self._P = sp.csc_matrix(data.P).astype(np.float64)
        self._c = np.asarray(data.c, dtype=np.float64)
        self._n = n

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
    from cupiqp import SolverBase as CupiqpSolverBase
    from cupiqp import Status as CupiqpStatus
    _CUPIQP_AVAILABLE = True
except ImportError:
    _CUPIQP_AVAILABLE = False


class CupiqpSolverMixin(SingleQPSolver):
    """Shared logic for cuPIQP backends. Subclasses set ``_kkt_solver``."""
    device: ClassVar[str] = 'gpu'
    _kkt_solver: ClassVar[str]

    def _prepare_data(self, data: SingleQPData) -> None:
        if not _CUPIQP_AVAILABLE:
            raise ImportError("cupiqp / cupy not installed")
        self._n = data.n

        kw: dict = {'c': cp.asarray(data.c, dtype=cp.float64)}

        if self._kkt_solver == 'sparse_ldlt':
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
            # dense_cholesky: feed dense cupy arrays
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
        self._solver = CupiqpSolverBase()
        self._solver.settings.kkt_solver = self._kkt_solver
        self._solver.settings.max_iter = self.max_iter
        self._solver.settings.eps_abs = self.tol_abs
        self._solver.settings.verbose = False
        with cp.cuda.Device(0):
            self._solver.setup(**self._setup_kwargs)

    def solve(self) -> None:
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
    _kkt_solver: ClassVar[str] = 'sparse_ldlt'

    @property
    def name(self) -> str:
        return "cupiqp-sparse"


class CupiqpDenseSolver(CupiqpSolverMixin):
    _kkt_solver: ClassVar[str] = 'dense_cholesky'

    @property
    def name(self) -> str:
        return "cupiqp-dense"


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
except ImportError:
    _CUCLARABEL_AVAILABLE = False


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

    def solve(self) -> None:
        from juliacall import Main as jl
        jl.seval("Clarabel.solve!(solver)")
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

try:
    from cuopt.linear_programming.problem import (
        Problem as _CuoptProblem,
        QuadraticExpression as _CuoptQuadExpr,
        MINIMIZE as _CUOPT_MIN,
    )
    from cuopt.linear_programming.solver_settings import (
        SolverSettings as _CuoptSettings,
    )
    _CUOPT_AVAILABLE = True
except ImportError:
    _CUOPT_AVAILABLE = False


class CuoptSolver(SingleQPSolver):
    """NVIDIA cuOpt LP/QP solver."""
    device: ClassVar[str] = 'gpu'

    @property
    def name(self) -> str:
        return "cuopt"

    def _prepare_data(self, data: SingleQPData) -> None:
        if not _CUOPT_AVAILABLE:
            raise ImportError("cuopt not installed")
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

    def setup(self) -> None:
        n = self._n
        prob = _CuoptProblem("singleQP")

        # Variables with finite bounds; +/-inf -> None (unbounded).
        variables = []
        for i in range(n):
            lb = (float(self._x_l[i])
                  if (self._x_l is not None and np.isfinite(self._x_l[i]))
                  else None)
            ub = (float(self._x_u[i])
                  if (self._x_u is not None and np.isfinite(self._x_u[i]))
                  else None)
            variables.append(prob.addVariable(lb=lb, ub=ub))

        # Objective:  0.5 x^T P x + c^T x
        Pc = self._P_coo
        qvars1 = [variables[int(i)] for i in Pc.row]
        qvars2 = [variables[int(j)] for j in Pc.col]
        qcoeffs = (0.5 * Pc.data).tolist()
        quad = _CuoptQuadExpr(
            qvars1=qvars1, qvars2=qvars2, qcoefficients=qcoeffs,
            vars=variables, coefficients=self._c.tolist(),
        )
        prob.setObjective(quad, sense=_CUOPT_MIN)

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
        self._settings = _CuoptSettings()
        self._settings.set_optimality_tolerance(self.tol_abs)
        self._settings.set_parameter("log_to_console", "0")

    def solve(self) -> None:
        self._prob.solve(self._settings)

    def _collect_result(self) -> SingleQPResult:
        prob = self._prob
        # Try to pull primal values; the attribute differs across cuOpt
        # versions (.X like Gurobi, or via prob.solution). Default to NaN.
        x_np = np.full(self._n, np.nan)
        try:
            x_np = np.array([float(v.X) for v in self._variables])
        except Exception:
            try:
                x_np = np.asarray(prob.solution(), dtype=np.float64)
            except Exception:
                pass

        obj_val = prob.ObjValue
        solved = obj_val is not None
        return SingleQPResult(
            x=x_np,
            n_iter=-1,   # cuOpt doesn't expose iter count via Python API
            obj=float(obj_val) if solved else float('nan'),
            status='OPTIMAL' if solved else 'FAILED',
            solved=solved,
        )


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
except ImportError:
    _MOREAU_TORCH_AVAILABLE = False


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
    CupiqpSparseSolver,
    # CupiqpDenseSolver,
    # QocoSolver,
    CuClarabelSolver,
    CuoptSolver,
    MoreauTorchSolver,
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
        # QocoSolver: _QOCO_AVAILABLE and _QOCO_GPU_AVAILABLE,
        CupiqpSparseSolver: _CUPIQP_AVAILABLE,
        # CupiqpDenseSolver: _CUPIQP_AVAILABLE,
        CuClarabelSolver: _CUCLARABEL_AVAILABLE and _CUPIQP_AVAILABLE,
        CuoptSolver: _CUOPT_AVAILABLE,
        MoreauTorchSolver: _MOREAU_TORCH_AVAILABLE,
    }
    out = [cls for cls in ALL_SOLVERS if flags.get(cls, False)]
    if device is not None:
        out = [cls for cls in out if cls.device == device]
    return out
