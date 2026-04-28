"""
Unified batched QP solver interface for benchmarking.

All solvers accept the same QP formulation:

    min  0.5 x^T P x + c^T x
    s.t. A x = b
         h_l <= G x <= h_u
         x_l <= x <= x_u

Data shapes:
    P:   (B, n, n)
    c:   (B, n)
    A:   (B, p, n) or None
    b:   (B, p)    or None
    G:   (B, m, n) or None
    h_l: (B, m)    or None
    h_u: (B, m)    or None
    x_l: (B, n)    or None
    x_u: (B, n)    or None
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
from functools import partial
import time
import nvtx
import numpy as np
import cupy as cp
import torch
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, vmap


SOLVER_COLORS = {
    "cupiqp": "blue",
    "qpax":   "green",
    "qpth":   "red",
    "moreau": "yellow",
    "jaxopt": "magenta",
}




@dataclass
class BatchedQPData:
    """Batched QP data in numpy arrays."""
    P: np.ndarray       # (B, n, n)
    c: np.ndarray       # (B, n)
    A: Optional[np.ndarray] = None   # (B, p, n)
    b: Optional[np.ndarray] = None   # (B, p)
    G: Optional[np.ndarray] = None   # (B, m, n)
    h_l: Optional[np.ndarray] = None # (B, m)
    h_u: Optional[np.ndarray] = None # (B, m)
    x_l: Optional[np.ndarray] = None # (B, n)
    x_u: Optional[np.ndarray] = None # (B, n)

    @property
    def B(self) -> int:
        return self.P.shape[0]

    @property
    def n(self) -> int:
        return self.P.shape[1]

    @property
    def p(self) -> int:
        return self.A.shape[1] if self.A is not None else 0

    @property
    def m(self) -> int:
        return self.G.shape[1] if self.G is not None else 0


@dataclass
class BatchedQPResult:
    """Unified result from batched QP solve."""
    x: np.ndarray            # (B, n) primal solution
    setup_time_ms: float     # median wall-clock time for setup in ms
    solve_time_ms: float     # median wall-clock time for solve in ms
    solve_times_all: list    # all individual solve times in ms (excluding warmup)
    n_solved: int            # number of problems solved successfully
    total: int               # total number of problems
    index_unsolved: list[int]  # index of the failed problems (max itr reached or numerical error)
    solver_name: str         # name of the solver
    n_iter_max: int = -1     # max iteration count over the batch; -1 if solver does not expose it

    @property
    def solve_time_std(self) -> float:
        """Standard deviation of solve times in ms."""
        return float(np.std(self.solve_times_all)) if self.solve_times_all else 0.0

    @property
    def solve_time_stderr(self) -> float:
        """Standard error of the mean of solve times in ms."""
        n = len(self.solve_times_all)
        return float(np.std(self.solve_times_all) / np.sqrt(n)) if n > 0 else 0.0


class BatchedQPSolver(ABC):
    """Base class for batched QP solvers.

    Lifecycle:
        _prepare_data(data)  --  numpy → native conversion (NOT timed).
        setup()              --  solver-specific init (allocate, JIT, factor pre-conditions).
        solve()              --  the actual solve.

    Splitting ``_prepare_data`` from ``setup`` makes timing fair: every solver
    sees the input as numpy ``BatchedQPData``, but the cost of converting
    to its native representation (cupy / jax / torch / scipy.sparse) is
    counted as a one-off, not part of per-solve setup.
    """

    def __init__(self, tol_abs: float = 1e-6, max_iter: int = 300):
        self.tol_abs = tol_abs
        self.max_iter = max_iter

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def _prepare_data(self, data: BatchedQPData) -> None:
        """Convert numpy ``BatchedQPData`` to native arrays, store on ``self``.

        Called once before timing begins. Must populate whatever ``self.*``
        attributes ``setup()`` and ``solve()`` need.
        """
        ...

    @abstractmethod
    def setup(self) -> None:
        """Solver-specific setup using the prepared data already on ``self``."""
        ...

    @abstractmethod
    def solve(self) -> BatchedQPResult:
        """Solve the QP(s) and return the result."""
        ...

    def benchmark(self, data: BatchedQPData, n_repeats: int = 5) -> BatchedQPResult:
        """Time setup once and solve over n_repeats, return median.

        ``_prepare_data`` runs once before timing. ``setup()`` runs once
        (timed — includes any first-call JIT / symbolic-analysis cost).
        The first solve is a warm-up and excluded; the remaining
        ``n_repeats`` solves are timed.
        """
        self._prepare_data(data)

        t0 = time.perf_counter()
        self.setup()
        t1 = time.perf_counter()
        setup_time_ms = (t1 - t0) * 1000

        # Warm-up solve
        self.solve()

        solve_times = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            result = self.solve()
            t1 = time.perf_counter()
            solve_times.append((t1 - t0) * 1000)

        result.setup_time_ms = setup_time_ms
        result.solve_time_ms = float(np.median(solve_times))
        result.solve_times_all = solve_times
        return result


# ======================================================================
# cuPIQP
# ======================================================================

try:
    from cupiqp import SolverBase
    from cupiqp import Status as CupiqpStatus
    _CUPIQP_AVAILABLE = True
except ImportError:
    _CUPIQP_AVAILABLE = False

class CupiqpBatchedSolverBase(BatchedQPSolver):
    """Base class for cuPIQP batched solvers (GPU, CuPy).

    Subclasses must implement ``_to_native()`` and set ``_kkt_solver``.
    """
    _kkt_solver: str  # set by subclass

    @abstractmethod
    def _to_native(self, data: BatchedQPData) -> dict:
        """Convert BatchedQPData to kwargs for SolverBase.setup().

        Returns a dict with keys: P, c, and optionally A, b, G, h_l, h_u, x_l, x_u.
        """
        ...

    def _prepare_data(self, data: BatchedQPData) -> None:
        if not _CUPIQP_AVAILABLE:
            raise ImportError("cupiqp is required for CupiqpBatchedSolver")
        self._data = data
        self._setup_kwargs = self._to_native(data)

    @nvtx.annotate("cupiqp::setup", color=SOLVER_COLORS["cupiqp"])
    def setup(self) -> None:
        self._solver = SolverBase()
        self._solver.settings.kkt_solver = self._kkt_solver
        self._solver.settings.preconditioner_iter = 10
        self._solver.settings.max_iter = self.max_iter
        self._solver.settings.eps_abs = self.tol_abs
        self._solver.settings.verbose = False
        self._solver.setup(**self._setup_kwargs)

    @nvtx.annotate("cupiqp::solve", color=SOLVER_COLORS["cupiqp"])
    def solve(self) -> BatchedQPResult:
        self._solver.solve()
        cp.cuda.Device(0).synchronize()

        valid_statuses = {
            CupiqpStatus.PIQP_SOLVED.value,
            CupiqpStatus.PIQP_PRIMAL_INFEASIBLE.value,
            CupiqpStatus.PIQP_DUAL_INFEASIBLE.value,
        }
        idx_unsolved = [i for i, status_i in enumerate(self._solver.result.info._status_value) if status_i not in valid_statuses]
        n_solved = len(self._solver.result.info._status_value) - len(idx_unsolved)
        x = cp.asnumpy(self._solver.result.x)
        return BatchedQPResult(
            x=x, setup_time_ms=0, solve_time_ms=0, solve_times_all=[],  # filled by benchmark()
            n_solved=n_solved, total=self._data.B,
            solver_name=self.name,
            index_unsolved=idx_unsolved,
            n_iter_max=int(self._solver.result.info.iter.max()),
        )


class CupiqpDenseBatchedSolver(CupiqpBatchedSolverBase):
    """cuPIQP with dense Cholesky backend."""
    _kkt_solver = 'dense_cholesky'

    @property
    def name(self) -> str:
        return "cupiqp-dense"

    def _to_native(self, data: BatchedQPData) -> dict:
        return dict(
            P=cp.array(data.P), c=cp.array(data.c),
            A=cp.array(data.A) if data.A is not None else None,
            b=cp.array(data.b) if data.b is not None else None,
            G=cp.array(data.G) if data.G is not None else None,
            h_l=cp.array(data.h_l) if data.h_l is not None else None,
            h_u=cp.array(data.h_u) if data.h_u is not None else None,
            x_l=cp.array(data.x_l) if data.x_l is not None else None,
            x_u=cp.array(data.x_u) if data.x_u is not None else None,
        )


class CupiqpSparseBatchedSolver(CupiqpBatchedSolverBase):
    """cuPIQP with sparse direct backend."""
    _kkt_solver = 'sparse_ldlt'

    @property
    def name(self) -> str:
        return "cupiqp-sparse"

    def _to_native(self, data: BatchedQPData) -> dict:
        from scipy.sparse import csr_matrix as sp_csr
        B = data.B
        return dict(
            P=[sp_csr(data.P[i]) for i in range(B)],
            c=cp.array(data.c),
            A=[sp_csr(data.A[i]) for i in range(B)] if data.A is not None else None,
            b=cp.array(data.b) if data.b is not None else None,
            G=[sp_csr(data.G[i]) for i in range(B)] if data.G is not None else None,
            h_l=cp.array(data.h_l) if data.h_l is not None else None,
            h_u=cp.array(data.h_u) if data.h_u is not None else None,
            x_l=cp.array(data.x_l) if data.x_l is not None else None,
            x_u=cp.array(data.x_u) if data.x_u is not None else None,
        )


# ======================================================================
# qpax
# ======================================================================

try:
    import qpax
    _QPAX_AVAILABLE = True
except ImportError:
    _QPAX_AVAILABLE = False

class QpaxBatchedSolver(BatchedQPSolver):
    """qpax batched solver (GPU, JAX)."""

    @property
    def name(self) -> str:
        return "qpax"

    def _prepare_data(self, data: BatchedQPData) -> None:
        if not _QPAX_AVAILABLE:
            raise ImportError("qpax is required for QpaxBatchedSolver")
        self._data = data
        B, n, p, m = data.B, data.n, data.p, data.m

        self._Ps = jnp.array(data.P)
        self._cs = jnp.array(data.c)

        # qpax uses: Ax = b, Gx <= h (one-sided inequality)
        # Convert cuPIQP format to qpax format:
        #   equality: A, b passed directly
        #   h_l <= Gx <= h_u  →  Gx <= h_u  AND  -Gx <= -h_l  (stack both)
        #   x_l <= x <= x_u   →  -Ix <= -x_l AND  Ix <= x_u    (stack both)

        if p > 0:
            self._As = jnp.array(data.A)
            self._bs = jnp.array(data.b)
        else:
            self._As = jnp.zeros((B, 0, n))
            self._bs = jnp.zeros((B, 0))

        G_parts = []
        h_parts = []

        if data.G is not None and data.h_u is not None:
            G_parts.append(jnp.array(data.G))
            h_parts.append(jnp.array(data.h_u))
        if data.G is not None and data.h_l is not None:
            G_parts.append(-jnp.array(data.G))
            h_parts.append(-jnp.array(data.h_l))
        if data.x_u is not None:
            G_parts.append(jnp.tile(jnp.eye(n)[None], (B, 1, 1)))
            h_parts.append(jnp.array(data.x_u))
        if data.x_l is not None:
            G_parts.append(jnp.tile(-jnp.eye(n)[None], (B, 1, 1)))
            h_parts.append(-jnp.array(data.x_l))

        if G_parts:
            self._Gs = jnp.concatenate(G_parts, axis=1)
            self._hs = jnp.concatenate(h_parts, axis=1)
        else:
            self._Gs = jnp.zeros((B, 0, n))
            self._hs = jnp.zeros((B, 0))

    @nvtx.annotate("qpax::setup", color=SOLVER_COLORS["qpax"])
    def setup(self) -> None:
        solve_fn = partial(qpax.solve_qp, solver_tol=self.tol_abs, max_iter=self.max_iter)
        self._batch_solve = jit(vmap(solve_fn, in_axes=(0, 0, 0, 0, 0, 0)))

    @nvtx.annotate("qpax::solve", color=SOLVER_COLORS["qpax"])
    def solve(self) -> BatchedQPResult:
        xs, _, _, _, converged, pdip_iter = self._batch_solve(
            self._Ps, self._cs, self._As, self._bs, self._Gs, self._hs)
        xs.block_until_ready()

        n_solved = int(jnp.sum(converged))
        idx_unsolved = [i for i, converged_i in enumerate(converged) if not converged_i]
        return BatchedQPResult(
            x=np.array(xs), setup_time_ms=0, solve_time_ms=0, solve_times_all=[],
            n_solved=n_solved, total=self._data.B,
            solver_name=self.name,
            index_unsolved=idx_unsolved,
            n_iter_max=int(pdip_iter.max()),
        )


# ======================================================================
# qpth
# ======================================================================

try:
    from qpth.qp import QPFunction
    _QPTH_AVAILABLE = True
except ImportError:
    _QPTH_AVAILABLE = False

class QpthBatchedSolver(BatchedQPSolver):
    """qpth batched solver (GPU, PyTorch)."""

    @property
    def name(self) -> str:
        return "qpth"

    def _prepare_data(self, data: BatchedQPData) -> None:
        if not _QPTH_AVAILABLE:
            raise ImportError("qpth is required for QpthBatchedSolver")

        self._data = data
        B, n, p, m = data.B, data.n, data.p, data.m

        self._Q = torch.tensor(data.P, dtype=torch.float64, device='cuda')
        self._p = torch.tensor(data.c, dtype=torch.float64, device='cuda')

        # qpth uses: Ax = b, Gx <= h (one-sided)
        # Same conversion as qpax
        if p > 0:
            self._A = torch.tensor(data.A, dtype=torch.float64, device='cuda')
            self._b = torch.tensor(data.b, dtype=torch.float64, device='cuda')
        else:
            self._A = torch.empty(0, dtype=torch.float64, device='cuda')
            self._b = torch.empty(0, dtype=torch.float64, device='cuda')

        G_parts = []
        h_parts = []

        if data.G is not None and data.h_u is not None:
            G_parts.append(torch.tensor(data.G, dtype=torch.float64, device='cuda'))
            h_parts.append(torch.tensor(data.h_u, dtype=torch.float64, device='cuda'))
        if data.G is not None and data.h_l is not None:
            G_parts.append(-torch.tensor(data.G, dtype=torch.float64, device='cuda'))
            h_parts.append(-torch.tensor(data.h_l, dtype=torch.float64, device='cuda'))
        if data.x_u is not None:
            G_parts.append(torch.eye(n, dtype=torch.float64, device='cuda').unsqueeze(0).expand(B, -1, -1))
            h_parts.append(torch.tensor(data.x_u, dtype=torch.float64, device='cuda'))
        if data.x_l is not None:
            G_parts.append(-torch.eye(n, dtype=torch.float64, device='cuda').unsqueeze(0).expand(B, -1, -1))
            h_parts.append(-torch.tensor(data.x_l, dtype=torch.float64, device='cuda'))

        if G_parts:
            self._G = torch.cat(G_parts, dim=1)
            self._h = torch.cat(h_parts, dim=1)
        else:
            self._G = torch.empty(B, 0, n, dtype=torch.float64, device='cuda')
            self._h = torch.empty(B, 0, dtype=torch.float64, device='cuda')

    @nvtx.annotate("qpth::setup", color=SOLVER_COLORS["qpth"])
    def setup(self) -> None:
        self._qp_fn = QPFunction(verbose=0, maxIter=self.max_iter, eps=self.tol_abs, check_Q_spd=False)

    @nvtx.annotate("qpth::solve", color=SOLVER_COLORS["qpth"])
    def solve(self) -> BatchedQPResult:
        with torch.no_grad():
            x = self._qp_fn(self._Q, self._p, self._G, self._h, self._A, self._b)
        torch.cuda.synchronize()

        x_np = x.cpu().numpy()
        # qpth doesn't report per-problem convergence; check for NaN
        n_solved = -1  # NOTE: qpth does not return status

        return BatchedQPResult(
            x=x_np, setup_time_ms=0, solve_time_ms=0, solve_times_all=[],
            n_solved=n_solved, total=self._data.B,
            solver_name=self.name,
            index_unsolved=[],
        )


# ======================================================================
# moreau (GPU, PyTorch)
# ======================================================================

try:
    from moreau.torch import Solver as MoreauTorchSolver
    import moreau
    _MOREAU_AVAILABLE = True
except ImportError:
    _MOREAU_AVAILABLE = False

class MoreauBatchedSolver(BatchedQPSolver):
    """moreau batched conic solver (GPU, PyTorch).

    Converts QP to conic form:
        min 0.5 x^T P x + q^T x
        s.t.  A_cone x + s = b_cone,  s in K

    where K = {0}^p x R+^(num_ineq) for equality + inequality constraints.

    NOTE: The Moreau solver uses Sparse linear algebra by default!
    """

    @property
    def name(self) -> str:
        return "moreau"

    def _prepare_data(self, data: BatchedQPData) -> None:
        if not _MOREAU_AVAILABLE:
            raise ImportError("moreau is required for MoreauBatchedSolver")
        from scipy import sparse

        self._data = data
        B, n, p, m = data.B, data.n, data.p, data.m

        # Build conic constraint matrix A_cone and bounds b_cone
        # Row blocks: [A_eq; G (upper); -G (lower); I (x_upper); -I (x_lower)]
        A_parts = []
        b_parts_np = []
        num_zero = 0
        num_nonneg = 0

        if p > 0:
            A_parts.append(data.A[0])  # same sparsity for all problems
            b_parts_np.append(data.b)  # (B, p)
            num_zero = p

        if data.G is not None and data.h_u is not None:
            A_parts.append(data.G[0])
            b_parts_np.append(data.h_u)
            num_nonneg += m
        if data.G is not None and data.h_l is not None:
            A_parts.append(-data.G[0])
            b_parts_np.append(-data.h_l)
            num_nonneg += m
        if data.x_u is not None:
            A_parts.append(np.eye(n))
            b_parts_np.append(data.x_u)
            num_nonneg += n
        if data.x_l is not None:
            A_parts.append(-np.eye(n))
            b_parts_np.append(-data.x_l)
            num_nonneg += n

        if A_parts:
            A_cone_np = np.vstack(A_parts)
        else:
            A_cone_np = np.zeros((0, n))

        # P sparsity (same for all problems — use problem 0)
        P_np = data.P[0]
        self._P_sp = sparse.csr_matrix(P_np)
        self._A_sp = sparse.csr_matrix(A_cone_np)
        self._num_zero = num_zero
        self._num_nonneg = num_nonneg

        # Batched values: (B, nnz_P) and (B, nnz_A)
        P_nnz = self._P_sp.nnz
        self._P_vals = torch.zeros(B, P_nnz, dtype=torch.float64)
        for i in range(B):
            P_i = sparse.csr_matrix(data.P[i])
            self._P_vals[i] = torch.tensor(P_i.data, dtype=torch.float64)

        # A_cone is the same structure for all problems, but values differ
        # for equality and inequality bounds
        self._A_vals = torch.tensor(
            np.tile(self._A_sp.data[None, :], (B, 1)), dtype=torch.float64
        )

        self._q = torch.tensor(data.c, dtype=torch.float64)

        # b_cone: (B, total_constraints)
        if b_parts_np:
            self._b = torch.tensor(np.concatenate(b_parts_np, axis=1), dtype=torch.float64)
        else:
            self._b = torch.zeros(B, 0, dtype=torch.float64)

    @nvtx.annotate("moreau::setup", color=SOLVER_COLORS["moreau"])
    def setup(self) -> None:
        B, n = self._data.B, self._data.n

        cones = moreau.Cones(num_zero_cones=self._num_zero, num_nonneg_cones=self._num_nonneg)
        ipm_settings = moreau.IPMSettings(
            direct_solve_method="cudss",  # NOTE: if not set it to cudss, maybe it switches to CPU when batch size is small?
            tol_feas=self.tol_abs,
            cudss_ir_steps=0,  # NOTE: the default IR step is 2
            )
        settings = moreau.Settings(
            batch_size=B,  # NOTE: passing the batch_size seems to enhance the Moreau's perform a lot!
            max_iter=self.max_iter,
            enable_grad=False,
            ipm_settings=ipm_settings,
            device="cuda",
        )

        self._solver = MoreauTorchSolver(
            n=n, m=self._A_sp.shape[0],
            P_row_offsets=torch.tensor(self._P_sp.indptr, dtype=torch.int32),
            P_col_indices=torch.tensor(self._P_sp.indices, dtype=torch.int32),
            A_row_offsets=torch.tensor(self._A_sp.indptr, dtype=torch.int32),
            A_col_indices=torch.tensor(self._A_sp.indices, dtype=torch.int32),
            cones=cones,
            settings=settings,
        )

    @nvtx.annotate("moreau::solve", color=SOLVER_COLORS["moreau"])
    def solve(self) -> BatchedQPResult:
        with torch.no_grad():
            sol = self._solver.solve(self._P_vals, self._A_vals, self._q, self._b)
        torch.cuda.synchronize()

        valid_statuses = {
            moreau.SolverStatus.Solved,
            moreau.SolverStatus.PrimalInfeasible,
            moreau.SolverStatus.DualInfeasible,
            moreau.SolverStatus.AlmostSolved,
            moreau.SolverStatus.AlmostPrimalInfeasible,
            moreau.SolverStatus.AlmostDualInfeasible,
        }
        idx_unsolved = [i for i, status_i in enumerate(self._solver.info.status) if status_i not in valid_statuses]
        n_solved = len(self._solver.info.status) - len(idx_unsolved)
        x_np = sol.x.cpu().numpy()

        return BatchedQPResult(
            x=x_np, setup_time_ms=0, solve_time_ms=0, solve_times_all=[],
            n_solved=n_solved, total=self._data.B,
            solver_name=self.name,
            index_unsolved=idx_unsolved,
            n_iter_max=int(np.max(self._solver.info.iterations)),
        )


# ======================================================================
# jaxopt BoxOSQP
# ======================================================================

try:
    from jaxopt import BoxOSQP
    _JAXOPT_AVAILABLE = True
except ImportError:
    _JAXOPT_AVAILABLE = False

class JaxoptBatchedSolver(BatchedQPSolver):
    """jaxopt BoxOSQP batched solver (GPU, JAX)."""

    @property
    def name(self) -> str:
        return "jaxopt"

    def _prepare_data(self, data: BatchedQPData) -> None:
        if not _JAXOPT_AVAILABLE:
            raise ImportError("jaxopt is required for JaxoptBatchedSolver")
        self._data = data
        B, n, p, m = data.B, data.n, data.p, data.m

        self._Ps = jnp.array(data.P)
        self._cs = jnp.array(data.c)

        # BoxOSQP: params_ineq=(A, l, u) for l <= Ax <= u
        # Stack: [A_eq; G; I] with bounds [b,b; h_l,h_u; x_l,x_u]
        A_full_parts = []
        l_full_parts = []
        u_full_parts = []

        if p > 0:
            A_full_parts.append(jnp.array(data.A))
            l_full_parts.append(jnp.array(data.b))
            u_full_parts.append(jnp.array(data.b))  # equality: l = u = b

        if data.G is not None:
            A_full_parts.append(jnp.array(data.G))
            l_full_parts.append(jnp.array(data.h_l) if data.h_l is not None else jnp.full((B, m), -jnp.inf))
            u_full_parts.append(jnp.array(data.h_u) if data.h_u is not None else jnp.full((B, m), jnp.inf))

        x_l_jo = jnp.array(data.x_l) if data.x_l is not None else jnp.full((B, n), -jnp.inf)
        x_u_jo = jnp.array(data.x_u) if data.x_u is not None else jnp.full((B, n), jnp.inf)

        # Box bounds as identity rows
        I_n = jnp.tile(jnp.eye(n)[None], (B, 1, 1))
        A_full_parts.append(I_n)
        l_full_parts.append(x_l_jo)
        u_full_parts.append(x_u_jo)

        self._A_full = jnp.concatenate(A_full_parts, axis=1)
        self._l_full = jnp.concatenate(l_full_parts, axis=1)
        self._u_full = jnp.concatenate(u_full_parts, axis=1)

    @nvtx.annotate("jaxopt::setup", color=SOLVER_COLORS["jaxopt"])
    def setup(self) -> None:
        osqp = BoxOSQP(
            maxiter=self.max_iter,
            tol=self.tol_abs,
            jit=True,
            check_primal_dual_infeasability=True,
        )

        def solve_one(Q, c, A, l, u):
            sol = osqp.run(
                params_obj=(Q, c),
                params_ineq=(A, l, u),
            )
            return sol.params.primal[0]  # x

        self._batch_solve = jit(vmap(solve_one, in_axes=(0, 0, 0, 0, 0)))

    @nvtx.annotate("jaxopt::solve", color=SOLVER_COLORS["jaxopt"])
    def solve(self) -> BatchedQPResult:
        t0 = time.perf_counter()
        xs = self._batch_solve(
            self._Ps, self._cs, self._A_full, self._l_full, self._u_full)
        xs.block_until_ready()
        elapsed = time.perf_counter() - t0

        return BatchedQPResult(
            x=np.array(xs), setup_time_ms=0, solve_time_ms=elapsed * 1000, solve_times_all=[],
            n_solved=-1,  # NOTE: BoxOSQP doesn't report per-problem convergence easily
            total=self._data.B,
            solver_name=self.name,
            index_unsolved=[],
        )
