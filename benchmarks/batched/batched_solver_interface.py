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
from typing import Optional, Union
from functools import partial
import time
import nvtx
import numpy as np
import cupy as cp
import scipy.sparse as sp_cpu
import torch
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, vmap


SOLVER_COLORS = {
    "cupiqp":              "blue",
    "qpax":                "green",
    "qpth":                "red",
    "moreau-torch":        "yellow",
    "moreau-torch-hacked": "orange",
    "moreau-jax":          "purple",
    "jaxopt":              "magenta",
}


# ---------------------------------------------------------------------------
# Sparse batched matrix container
# ---------------------------------------------------------------------------
@dataclass
class SparseMatBatch:
    """A batch of sparse matrices sharing one structural sparsity pattern.

    ``pattern`` is a scipy CSR carrying ``indices`` and ``indptr`` (and a
    placeholder values array that is ignored — the real per-batch values
    live in ``values``). ``values`` has shape ``(B, nnz)`` and is aligned
    to ``pattern.indices`` / ``pattern.indptr``: ``values[b, k]`` is the
    nonzero at the same CSR slot as ``pattern.data[k]``.

    This mirrors the layout that ``UniformBatchedCsrMatrix`` uses on the
    device, so converting to/from the cupiqp form is a one-shot transfer
    with no Python loop over batches.
    """
    pattern: sp_cpu.csr_matrix
    values: np.ndarray   # (B, nnz)

    @property
    def B(self) -> int:
        return self.values.shape[0]

    @property
    def nnz(self) -> int:
        return self.pattern.nnz

    @property
    def shape(self) -> tuple[int, int]:
        return self.pattern.shape


# ---------------------------------------------------------------------------
# Batched QP data — dense and sparse storage variants
# ---------------------------------------------------------------------------
@dataclass
class DenseBatchedQPData:
    """Batched QP data with dense numpy storage.

    Use ``to_sparse()`` to convert to ``SparseBatchedQPData`` when feeding
    a sparse-natural solver.
    """
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

    def to_sparse(self) -> "SparseBatchedQPData":
        """Build a ``SparseBatchedQPData`` view: master CSR pattern from
        problem 0, per-batch values gathered by direct dense indexing."""
        return SparseBatchedQPData(
            P=_extract_sparse_batch(self.P),
            c=self.c,
            A=_extract_sparse_batch(self.A) if self.A is not None else None,
            b=self.b,
            G=_extract_sparse_batch(self.G) if self.G is not None else None,
            h_l=self.h_l, h_u=self.h_u,
            x_l=self.x_l, x_u=self.x_u,
        )


@dataclass
class SparseBatchedQPData:
    """Batched QP data with sparse matrix storage (shared sparsity per matrix).

    ``P``, ``A``, ``G`` are ``SparseMatBatch`` (pattern + per-batch values).
    Vectors stay dense — they have no useful sparsity.
    """
    P: SparseMatBatch                          # (n, n) pattern + (B, nnz) values
    c: np.ndarray                              # (B, n)
    A: Optional[SparseMatBatch] = None         # (p, n) pattern + (B, A_nnz) values
    b: Optional[np.ndarray] = None             # (B, p)
    G: Optional[SparseMatBatch] = None         # (m, n) pattern + (B, G_nnz) values
    h_l: Optional[np.ndarray] = None           # (B, m)
    h_u: Optional[np.ndarray] = None           # (B, m)
    x_l: Optional[np.ndarray] = None           # (B, n)
    x_u: Optional[np.ndarray] = None           # (B, n)

    @property
    def B(self) -> int:
        return self.P.B

    @property
    def n(self) -> int:
        return self.P.shape[1]

    @property
    def p(self) -> int:
        return self.A.shape[0] if self.A is not None else 0

    @property
    def m(self) -> int:
        return self.G.shape[0] if self.G is not None else 0

    def to_dense(self) -> "DenseBatchedQPData":
        """Materialize ``(B, m, n)`` dense ndarrays by scattering values
        back into zero-initialized buffers at the master CSR positions."""
        return DenseBatchedQPData(
            P=_densify_sparse_batch(self.P),
            c=self.c,
            A=_densify_sparse_batch(self.A) if self.A is not None else None,
            b=self.b,
            G=_densify_sparse_batch(self.G) if self.G is not None else None,
            h_l=self.h_l, h_u=self.h_u,
            x_l=self.x_l, x_u=self.x_u,
        )


def _extract_sparse_batch(arr_3d: np.ndarray) -> SparseMatBatch:
    """Build a ``SparseMatBatch`` from a ``(B, m, n)`` dense ndarray.

    The master sparsity pattern is the *union* of per-batch nonzero patterns:
    position ``(r, c)`` is included iff at least one batch element has a
    nonzero there. This is what scipy's ``csr_matrix(dense)`` would produce
    if you fed it the element-wise-OR of the absolute values — every
    nonzero in any batch is preserved, every shared structural zero (a
    position where *all* batches are zero) is dropped. Then per-batch
    values at the master positions are gathered in one vectorized step.

    Robustness: this avoids the "batch 0 happens to have a zero where batch
    1 has a nonzero" silent-corruption case. If you already have a known
    structural pattern (e.g. from a sparse problem builder), construct the
    ``SparseMatBatch`` directly instead of round-tripping through dense.
    """
    B, m, n = arr_3d.shape
    # (m, n) bool: True iff any batch has a nonzero at (r, c).
    mask = np.any(arr_3d != 0.0, axis=0)
    rows, cols = np.nonzero(mask)   # both already in row-major order
    # Build a canonical CSR (indices sorted within each row by construction).
    indptr = np.zeros(m + 1, dtype=np.int32)
    np.add.at(indptr, rows + 1, 1)
    indptr = np.cumsum(indptr).astype(np.int32)
    indices = cols.astype(np.int32)
    # The template's .data field is a placeholder. Real per-batch values
    # live in ``values`` below — consumers only read pattern.indices /
    # pattern.indptr / pattern.shape from the template.
    template = sp_cpu.csr_matrix(
        (np.zeros(len(cols), dtype=arr_3d.dtype), indices, indptr),
        shape=(m, n),
    )
    values = arr_3d[:, rows, cols]   # (B, nnz)
    return SparseMatBatch(pattern=template, values=values)


def _densify_sparse_batch(sp_mat: SparseMatBatch) -> np.ndarray:
    """Materialize a ``(B, m, n)`` dense ndarray from a ``SparseMatBatch``."""
    m, n = sp_mat.shape
    rows = np.repeat(
        np.arange(m, dtype=np.int64), np.diff(sp_mat.pattern.indptr),
    )
    cols = sp_mat.pattern.indices
    out = np.zeros((sp_mat.B, m, n), dtype=sp_mat.values.dtype)
    out[:, rows, cols] = sp_mat.values
    return out


# Backward-compatibility alias. The original ``BatchedQPData`` name was
# always dense-only; existing benchmark scripts construct via this alias.
# New sparse-native benchmarks should construct ``SparseBatchedQPData``
# directly.
BatchedQPData = DenseBatchedQPData


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
    def solve(self) -> None:
        """Run the solver kernels and end with a stream sync.

        Must NOT pull data to host or build a result object — those go in
        ``_collect_result()`` so they don't inflate the timing.
        """
        ...

    @abstractmethod
    def _collect_result(self) -> BatchedQPResult:
        """Pull the solution to host side and build the ``BatchedQPResult``.

        Called once after the timing loop. May freely D2H whatever it needs
        (primal x, status array, iter counter, etc.).
        """
        ...

    def benchmark(self, data: BatchedQPData, n_repeats: int = 5) -> BatchedQPResult:
        """Time setup once and solve over n_repeats, return median.

        ``_prepare_data`` runs once before timing. ``setup()`` runs once
        (timed — includes any first-call JIT / symbolic-analysis cost).
        The first solve is a warm-up and excluded; the remaining
        ``n_repeats`` solves are timed. ``_collect_result`` runs once at
        the end, outside the timed window.
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
            self.solve()
            t1 = time.perf_counter()
            solve_times.append((t1 - t0) * 1000)

        result = self._collect_result()
        result.setup_time_ms = setup_time_ms
        result.solve_time_ms = float(np.median(solve_times))
        result.solve_times_all = solve_times
        return result


# ======================================================================
# cuPIQP
# ======================================================================

try:
    from cupiqp import DenseSolver, SparseSolver, MultistageSolver
    from cupiqp import Status as CupiqpStatus
    from cupiqp.sparse.batched_csr import UniformBatchedCsrMatrix
    _CUPIQP_AVAILABLE = True
    _CUPIQP_BACKENDS = {
        'dense_cholesky': DenseSolver,
        'sparse_ldlt': SparseSolver,
        'multistage_block_cholesky': MultistageSolver,
    }
except ImportError:
    _CUPIQP_AVAILABLE = False
    _CUPIQP_BACKENDS = {}


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
        self._solver = _CUPIQP_BACKENDS[self._kkt_solver]()
        self._solver.settings.preconditioner_iter = 10
        self._solver.settings.max_iter = self.max_iter
        self._solver.settings.eps_abs = self.tol_abs
        self._solver.settings.verbose = False
        self._solver.setup(**self._setup_kwargs)

    @nvtx.annotate("cupiqp::solve", color=SOLVER_COLORS["cupiqp"])
    def solve(self) -> None:
        self._solver.solve()
        cp.cuda.Device(0).synchronize()

    def _collect_result(self) -> BatchedQPResult:
        valid_statuses = {
            CupiqpStatus.CUPIQP_SOLVED.value,
            CupiqpStatus.CUPIQP_PRIMAL_INFEASIBLE.value,
            CupiqpStatus.CUPIQP_DUAL_INFEASIBLE.value,
        }
        status_np = cp.asnumpy(self._solver.result.info._status_value)
        idx_unsolved = [i for i, s in enumerate(status_np) if s not in valid_statuses]
        n_solved = len(status_np) - len(idx_unsolved)
        return BatchedQPResult(
            x=cp.asnumpy(self._solver.result.x),
            setup_time_ms=0, solve_time_ms=0, solve_times_all=[],  # filled by benchmark()
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

    def _to_native(self, data: DenseBatchedQPData) -> dict:
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
    """cuPIQP with sparse direct backend.

    Consumes ``SparseBatchedQPData`` natively. If given a ``DenseBatchedQPData``
    it canonicalizes to sparse first via ``to_sparse()`` — runs outside the
    timed window in ``_prepare_data``, so the conversion cost is visible to
    the caller but not counted as solve time.
    """
    _kkt_solver = 'sparse_ldlt'

    @property
    def name(self) -> str:
        return "cupiqp-sparse"

    @staticmethod
    def _pack(sp_mat: SparseMatBatch) -> "UniformBatchedCsrMatrix":
        """Wrap a ``SparseMatBatch`` as a ``UniformBatchedCsrMatrix`` on device.

        Two H2D copies — indices/indptr and the ``(B, nnz)`` values block —
        no Python loop over batches, no per-batch CSR object construction.
        """
        return UniformBatchedCsrMatrix(
            batch_size=sp_mat.B,
            indices=cp.asarray(sp_mat.pattern.indices, dtype=cp.int32),
            indptr=cp.asarray(sp_mat.pattern.indptr, dtype=cp.int32),
            data=cp.asarray(sp_mat.values),
            shape=sp_mat.shape,
        )

    def _to_native(
        self, data: Union[DenseBatchedQPData, SparseBatchedQPData],
    ) -> dict:
        if isinstance(data, DenseBatchedQPData):
            data = data.to_sparse()
        return dict(
            P=self._pack(data.P),
            c=cp.array(data.c),
            A=self._pack(data.A) if data.A is not None else None,
            b=cp.array(data.b) if data.b is not None else None,
            G=self._pack(data.G) if data.G is not None else None,
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

        # qpax / qpth do not tolerate ±inf bounds — they propagate into NaN
        # through the IPM barrier. Filter rows where the *finite-bound*
        # pattern is identical across the batch by checking only row 0
        # (cuPIQP requires the structure to be uniform, so it's enough).
        def _finite_mask_for_upper(arr):
            return np.isfinite(arr[0])  # True where the upper bound is finite

        def _finite_mask_for_lower(arr):
            return np.isfinite(arr[0])

        if data.G is not None and data.h_u is not None:
            mask = _finite_mask_for_upper(data.h_u)
            if mask.any():
                G_parts.append(jnp.array(data.G[:, mask, :]))
                h_parts.append(jnp.array(data.h_u[:, mask]))
        if data.G is not None and data.h_l is not None:
            mask = _finite_mask_for_lower(data.h_l)
            if mask.any():
                G_parts.append(-jnp.array(data.G[:, mask, :]))
                h_parts.append(-jnp.array(data.h_l[:, mask]))
        if data.x_u is not None:
            mask = _finite_mask_for_upper(data.x_u)
            if mask.any():
                eye_rows = jnp.eye(n)[mask]            # (k, n)
                G_parts.append(jnp.tile(eye_rows[None], (B, 1, 1)))
                h_parts.append(jnp.array(data.x_u[:, mask]))
        if data.x_l is not None:
            mask = _finite_mask_for_lower(data.x_l)
            if mask.any():
                eye_rows = jnp.eye(n)[mask]
                G_parts.append(jnp.tile(-eye_rows[None], (B, 1, 1)))
                h_parts.append(-jnp.array(data.x_l[:, mask]))

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
    def solve(self) -> None:
        xs, _, _, _, converged, pdip_iter = self._batch_solve(
            self._Ps, self._cs, self._As, self._bs, self._Gs, self._hs)
        xs.block_until_ready()
        self._last_xs        = xs
        self._last_converged = converged
        self._last_iters     = pdip_iter

    def _collect_result(self) -> BatchedQPResult:
        converged_np = np.asarray(self._last_converged)
        return BatchedQPResult(
            x=np.asarray(self._last_xs),
            setup_time_ms=0, solve_time_ms=0, solve_times_all=[],
            n_solved=int(converged_np.sum()), total=self._data.B,
            solver_name=self.name,
            index_unsolved=[i for i, c in enumerate(converged_np) if not c],
            n_iter_max=int(self._last_iters.max()),
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

        # qpth's IPM barrier propagates ±inf into NaN — drop rows where
        # the bound is infinite (vacuous constraint). cuPIQP guarantees
        # uniform finite-bound structure across the batch, so checking
        # row 0 is sufficient.
        def _mask_finite(arr):
            return np.isfinite(arr[0])

        if data.G is not None and data.h_u is not None:
            mask = _mask_finite(data.h_u)
            if mask.any():
                G_parts.append(torch.tensor(data.G[:, mask, :], dtype=torch.float64, device='cuda'))
                h_parts.append(torch.tensor(data.h_u[:, mask], dtype=torch.float64, device='cuda'))
        if data.G is not None and data.h_l is not None:
            mask = _mask_finite(data.h_l)
            if mask.any():
                G_parts.append(-torch.tensor(data.G[:, mask, :], dtype=torch.float64, device='cuda'))
                h_parts.append(-torch.tensor(data.h_l[:, mask], dtype=torch.float64, device='cuda'))
        if data.x_u is not None:
            mask = _mask_finite(data.x_u)
            if mask.any():
                eye_rows = torch.eye(n, dtype=torch.float64, device='cuda')[
                    torch.from_numpy(mask).to('cuda')
                ]
                G_parts.append(eye_rows.unsqueeze(0).expand(B, -1, -1))
                h_parts.append(torch.tensor(data.x_u[:, mask], dtype=torch.float64, device='cuda'))
        if data.x_l is not None:
            mask = _mask_finite(data.x_l)
            if mask.any():
                eye_rows = torch.eye(n, dtype=torch.float64, device='cuda')[
                    torch.from_numpy(mask).to('cuda')
                ]
                G_parts.append(-eye_rows.unsqueeze(0).expand(B, -1, -1))
                h_parts.append(-torch.tensor(data.x_l[:, mask], dtype=torch.float64, device='cuda'))

        if G_parts:
            self._G = torch.cat(G_parts, dim=1)
            self._h = torch.cat(h_parts, dim=1)
        else:
            self._G = torch.empty(B, 0, n, dtype=torch.float64, device='cuda')
            self._h = torch.empty(B, 0, dtype=torch.float64, device='cuda')

    @nvtx.annotate("qpth::setup", color=SOLVER_COLORS["qpth"])
    def setup(self) -> None:
        self._qp_fn = QPFunction(verbose=0, maxIter=self.max_iter, eps=self.tol_abs, check_Q_spd=False)
        self._last_x = None
        self._solve_error: str | None = None

    @nvtx.annotate("qpth::solve", color=SOLVER_COLORS["qpth"])
    def solve(self) -> None:
        # qpth's QPFunction silently returns None on internal failure (most
        # commonly CUDA OOM at large batches, or singular KKT systems for
        # specific problems). Catching here means the whole benchmark sweep
        # keeps running and the failure is recorded as "all unsolved" in
        # ``_collect_result`` instead of crashing the harness.
        try:
            with torch.no_grad():
                out = self._qp_fn(self._Q, self._p, self._G, self._h, self._A, self._b)
            torch.cuda.synchronize()
            if out is None:
                self._solve_error = "QPFunction returned None"
            self._last_x = out
        except Exception as e:
            self._solve_error = f"{type(e).__name__}: {e}"
            self._last_x = None

    def _collect_result(self) -> BatchedQPResult:
        B, n = self._data.B, self._data.n
        if self._last_x is None:
            return BatchedQPResult(
                x=np.full((B, n), np.nan, dtype=np.float64),
                setup_time_ms=0, solve_time_ms=0, solve_times_all=[],
                n_solved=0, total=B,
                solver_name=self.name,
                index_unsolved=list(range(B)),
            )
        # qpth doesn't expose per-problem convergence or iter count.
        return BatchedQPResult(
            x=self._last_x.cpu().numpy(),
            setup_time_ms=0, solve_time_ms=0, solve_times_all=[],
            n_solved=-1, total=B,
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


def _build_batched_cone_A_values(data: BatchedQPData, A_master_sp) -> np.ndarray:
    """Per-batch values for the conic constraint matrix used by both moreau wrappers.

    Both wrappers form the conic A as the row stack
    ``[A_eq[i]; G[i]; -G[i]; I; -I]`` per batch (with the obvious blocks
    omitted when the corresponding data field is None). The structural
    sparsity pattern is identical across batches — only the numerical values
    in ``A_eq`` and ``G`` vary — so we reuse the master ``A_master_sp`` from
    problem 0 and align per-batch values to its (rows, cols) by direct dense
    indexing. This avoids relying on ``scipy.sparse.csr_matrix(A_cone_i).data``
    coming back with the same ordering for every batch.

    Returns an array of shape ``(B, A_master_sp.nnz)``.
    """
    B, n, p = data.B, data.n, data.p
    rows = np.repeat(
        np.arange(A_master_sp.shape[0], dtype=np.int64),
        np.diff(A_master_sp.indptr),
    )
    cols = A_master_sp.indices
    out = np.zeros((B, A_master_sp.nnz), dtype=np.float64)
    eye_n = np.eye(n)
    for i in range(B):
        parts = []
        if p > 0:
            parts.append(data.A[i])
        if data.G is not None and data.h_u is not None:
            parts.append(data.G[i])
        if data.G is not None and data.h_l is not None:
            parts.append(-data.G[i])
        if data.x_u is not None:
            parts.append(eye_n)
        if data.x_l is not None:
            parts.append(-eye_n)
        A_cone_i = np.vstack(parts) if parts else np.zeros((0, n))
        out[i] = A_cone_i[rows, cols]
    return out


class MoreauTorchBatchedSolver(BatchedQPSolver):
    """moreau batched conic solver via the PyTorch interface (GPU).

    Converts QP to conic form:
        min 0.5 x^T P x + q^T x
        s.t.  A_cone x + s = b_cone,  s in K

    where K = {0}^p x R+^(num_ineq) for equality + inequality constraints.

    NOTE: The Moreau solver uses Sparse linear algebra by default!
    """

    @property
    def name(self) -> str:
        return "moreau-torch"

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

        # Batched values on CUDA — moreau's solver runs on cuda (device="cuda"
        # in setup()), so leaving these on CPU forces a H2D copy of every
        # input tensor on every solve() call, polluting the timed window.
        P_nnz = self._P_sp.nnz
        P_vals_np = np.zeros((B, P_nnz), dtype=np.float64)
        for i in range(B):
            P_vals_np[i] = sparse.csr_matrix(data.P[i]).data
        self._P_vals = torch.tensor(P_vals_np, dtype=torch.float64, device='cuda')

        A_vals_np = _build_batched_cone_A_values(data, self._A_sp)
        self._A_vals = torch.tensor(A_vals_np, dtype=torch.float64, device='cuda')

        self._q = torch.tensor(data.c, dtype=torch.float64, device='cuda')

        # b_cone: (B, total_constraints)
        if b_parts_np:
            self._b = torch.tensor(
                np.concatenate(b_parts_np, axis=1),
                dtype=torch.float64, device='cuda',
            )
        else:
            self._b = torch.zeros(B, 0, dtype=torch.float64, device='cuda')

    @nvtx.annotate("moreau-torch::setup", color=SOLVER_COLORS["moreau-torch"])
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

    @nvtx.annotate("moreau-torch::solve", color=SOLVER_COLORS["moreau-torch"])
    def solve(self) -> None:
        with torch.no_grad():
            self._last_sol = self._solver.solve(self._P_vals, self._A_vals, self._q, self._b)
        torch.cuda.synchronize()

    def _collect_result(self) -> BatchedQPResult:
        valid_statuses = {
            moreau.SolverStatus.Solved,
            moreau.SolverStatus.PrimalInfeasible,
            moreau.SolverStatus.DualInfeasible,
            moreau.SolverStatus.AlmostSolved,
            moreau.SolverStatus.AlmostPrimalInfeasible,
            moreau.SolverStatus.AlmostDualInfeasible,
        }
        status = self._solver.info.status  # already host-side
        idx_unsolved = [i for i, s in enumerate(status) if s not in valid_statuses]
        n_solved = len(status) - len(idx_unsolved)
        return BatchedQPResult(
            x=self._last_sol.x.cpu().numpy(),
            setup_time_ms=0, solve_time_ms=0, solve_times_all=[],
            n_solved=n_solved, total=self._data.B,
            solver_name=self.name,
            index_unsolved=idx_unsolved,
            n_iter_max=int(np.max(self._solver.info.iterations)),
        )


class MoreauTorchHackedBatchedSolver(MoreauTorchBatchedSolver):
    """Same problem encoding as ``MoreauTorchBatchedSolver``, but bypasses
    moreau's public ``Solver.solve()`` wrapper and calls the underlying
    autograd ``_SolveFunction.apply`` directly.

    Why: ``Solver.solve()`` does ``[_normalize_status(int(st)) for st in
    cached['status']]`` (moreau/torch/__init__.py:752) on every call to
    populate ``self._info.status``. With batch size B that's B element-wise
    D2H syncs inside the timed window. Calling ``_SolveFunction.apply``
    directly skips that loop; we read the device-resident status array
    once in ``_collect_result()`` via a single bulk D2H.

    Reaches into moreau's private API — may break across moreau versions.
    """

    @property
    def name(self) -> str:
        return "moreau-torch-hacked"

    @nvtx.annotate("moreau-torch-hacked::setup", color=SOLVER_COLORS["moreau-torch-hacked"])
    def setup(self) -> None:
        super().setup()
        # Run moreau's internal P/A setup once so ``_solver._P_values`` and
        # ``_solver._A_values`` are populated for the bypass path. The
        # public ``Solver.solve()`` calls this on every solve, but with
        # unchanged P/A values it's a cached no-op after the first run.
        self._solver.setup(self._P_vals, self._A_vals)

    @nvtx.annotate("moreau-torch-hacked::solve", color=SOLVER_COLORS["moreau-torch-hacked"])
    def solve(self) -> None:
        from moreau.torch import _SolveFunction
        with torch.no_grad():
            x, _z, _s = _SolveFunction.apply(
                self._solver, self._q, self._b,
                self._solver._P_values, self._solver._A_values,
            )
        self._last_x = x
        torch.cuda.synchronize()

    def _collect_result(self) -> BatchedQPResult:
        valid_statuses = {
            moreau.SolverStatus.Solved,
            moreau.SolverStatus.PrimalInfeasible,
            moreau.SolverStatus.DualInfeasible,
            moreau.SolverStatus.AlmostSolved,
            moreau.SolverStatus.AlmostPrimalInfeasible,
            moreau.SolverStatus.AlmostDualInfeasible,
        }
        from moreau.torch import _normalize_status
        cached = self._solver._last_result
        status_dev = cached['status']
        status_np = status_dev.cpu().numpy() if hasattr(status_dev, 'cpu') \
                    else np.asarray(status_dev)
        statuses = [_normalize_status(int(s)) for s in status_np]
        idx_unsolved = [i for i, s in enumerate(statuses) if s not in valid_statuses]
        n_solved = len(statuses) - len(idx_unsolved)

        iters = cached['iterations']
        iters_np = iters.cpu().numpy() if hasattr(iters, 'cpu') else np.asarray(iters)

        return BatchedQPResult(
            x=self._last_x.cpu().numpy(),
            setup_time_ms=0, solve_time_ms=0, solve_times_all=[],
            n_solved=n_solved, total=self._data.B,
            solver_name=self.name,
            index_unsolved=idx_unsolved,
            n_iter_max=int(iters_np.max()),
        )


# ======================================================================
# moreau via JAX
# ======================================================================

try:
    from moreau.jax import Solver as MoreauJaxSolver
    _MOREAU_JAX_AVAILABLE = True
except ImportError:
    _MOREAU_JAX_AVAILABLE = False


class MoreauJaxBatchedSolver(BatchedQPSolver):
    """moreau batched conic solver via the JAX interface (GPU).

    Same conic encoding as ``MoreauTorchBatchedSolver`` (the conic
    constraint matrix and value layout are identical — only the tensor
    framework differs). Unlike the Torch wrapper, moreau's JAX wrapper
    does NOT iterate over the per-batch status tensor inside its
    ``solve()``; status stays as a JAX array on device until the user
    pulls it explicitly. So no ``Hacked`` bypass variant is needed.

    NOTE: The Moreau solver uses Sparse linear algebra by default!
    """

    @property
    def name(self) -> str:
        return "moreau-jax"

    def _prepare_data(self, data: BatchedQPData) -> None:
        if not _MOREAU_JAX_AVAILABLE:
            raise ImportError("moreau.jax is required for MoreauJaxBatchedSolver")
        from scipy import sparse

        self._data = data
        B, n, p, m = data.B, data.n, data.p, data.m

        # Build the conic constraint matrix A_cone and bounds b_cone, identical
        # row blocks to the Torch path:
        #   [A_eq; G (upper); -G (lower); I (x_upper); -I (x_lower)]
        A_parts = []
        b_parts_np = []
        num_zero = 0
        num_nonneg = 0

        if p > 0:
            A_parts.append(data.A[0])
            b_parts_np.append(data.b)
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

        A_cone_np = np.vstack(A_parts) if A_parts else np.zeros((0, n))

        P_np = data.P[0]
        self._P_sp = sparse.csr_matrix(P_np)
        self._A_sp = sparse.csr_matrix(A_cone_np)
        self._num_zero = num_zero
        self._num_nonneg = num_nonneg

        # Per-batch P/A values aligned with the master CSR sparsity pattern.
        # moreau's JAX impl accepts (B, nnz) on the leading dim (see ``q.ndim``
        # dispatch in moreau_cuda/jax_wrapper.py). Calling _solve_raw directly
        # — without jax.vmap — bypasses the custom_vmap rule and lets the
        # underlying batched CUDA kernels do the work.
        P_nnz = self._P_sp.nnz
        P_vals_np = np.zeros((B, P_nnz), dtype=np.float64)
        for i in range(B):
            P_vals_np[i] = sparse.csr_matrix(data.P[i]).data
        A_vals_np = _build_batched_cone_A_values(data, self._A_sp)

        self._P_vals = jnp.asarray(P_vals_np, dtype=jnp.float64)
        self._A_vals = jnp.asarray(A_vals_np, dtype=jnp.float64)
        self._q = jnp.asarray(data.c, dtype=jnp.float64)
        if b_parts_np:
            self._b = jnp.asarray(np.concatenate(b_parts_np, axis=1), dtype=jnp.float64)
        else:
            self._b = jnp.zeros((B, 0), dtype=jnp.float64)

    @nvtx.annotate("moreau-jax::setup", color=SOLVER_COLORS["moreau-jax"])
    def setup(self) -> None:
        B, n = self._data.B, self._data.n

        cones = moreau.Cones(num_zero_cones=self._num_zero, num_nonneg_cones=self._num_nonneg)
        ipm_settings = moreau.IPMSettings(
            direct_solve_method="cudss",
            tol_feas=self.tol_abs,
            cudss_ir_steps=0,
        )
        settings = moreau.Settings(
            batch_size=B,
            max_iter=self.max_iter,
            enable_grad=False,
            ipm_settings=ipm_settings,
            device="cuda",
        )

        self._solver = MoreauJaxSolver(
            n=n, m=self._A_sp.shape[0],
            P_row_offsets=np.asarray(self._P_sp.indptr, dtype=np.int64),
            P_col_indices=np.asarray(self._P_sp.indices, dtype=np.int64),
            A_row_offsets=np.asarray(self._A_sp.indptr, dtype=np.int64),
            A_col_indices=np.asarray(self._A_sp.indices, dtype=np.int64),
            cones=cones,
            settings=settings,
        )
        self._batch_solve = jax.jit(
            jax.vmap(self._solver._solve_raw, in_axes=(0, 0, 0, 0))
        )

    @nvtx.annotate("moreau-jax::solve", color=SOLVER_COLORS["moreau-jax"])
    def solve(self) -> None:
        sol, info = self._batch_solve(
            self._P_vals, self._A_vals, self._q, self._b
        )
        self._last_sol = sol
        self._last_info = info
        # JAX is async — sync on the primal so the timed range covers
        # actual solver work rather than just the dispatch.
        sol.x.block_until_ready()

    def _collect_result(self) -> BatchedQPResult:
        valid_statuses = {
            moreau.SolverStatus.Solved,
            moreau.SolverStatus.PrimalInfeasible,
            moreau.SolverStatus.DualInfeasible,
            moreau.SolverStatus.AlmostSolved,
            moreau.SolverStatus.AlmostPrimalInfeasible,
            moreau.SolverStatus.AlmostDualInfeasible,
        }
        from moreau._types import normalize_status as _normalize_status
        info = self._last_info
        # Bulk D2H of the per-batch status / iterations arrays (one copy each).
        status_np = np.asarray(info.status)
        iters_np  = np.asarray(info.iterations)
        statuses = [_normalize_status(int(s)) for s in status_np]
        idx_unsolved = [i for i, s in enumerate(statuses) if s not in valid_statuses]
        n_solved = len(statuses) - len(idx_unsolved)

        return BatchedQPResult(
            x=np.asarray(self._last_sol.x),
            setup_time_ms=0, solve_time_ms=0, solve_times_all=[],
            n_solved=n_solved, total=self._data.B,
            solver_name=self.name,
            index_unsolved=idx_unsolved,
            n_iter_max=int(iters_np.max()),
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
    def solve(self) -> None:
        xs = self._batch_solve(
            self._Ps, self._cs, self._A_full, self._l_full, self._u_full)
        xs.block_until_ready()
        self._last_xs = xs

    def _collect_result(self) -> BatchedQPResult:
        return BatchedQPResult(
            x=np.asarray(self._last_xs),
            setup_time_ms=0, solve_time_ms=0, solve_times_all=[],
            n_solved=-1,  # NOTE: BoxOSQP doesn't report per-problem convergence easily
            total=self._data.B,
            solver_name=self.name,
            index_unsolved=[],
        )
