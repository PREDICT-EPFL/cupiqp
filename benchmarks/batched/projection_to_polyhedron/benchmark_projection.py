"""
Run the batched Euclidean polytope-projection benchmark and save JSON.

Each batch element gets its own polytope, so this benchmarks B projections
each onto a *different* polytope:

    min  0.5 ||x_i - q_i||^2
    s.t. A_i x_i <= b_i,         i = 1 .. B

equivalently, as a strictly-convex batched QP with P = I, c = -q:

    min  0.5 x_i^T x_i  -  q_i^T x_i      s.t.  A_i x_i <= b_i.

Polytope construction (cheap in any dimension; no ConvexHull):

    A_i  ~  N(0, I)              shape (l, n)        random row directions
    b_i[j]  =  ||A_i[j]||_2  +  slack_j               slack_j ~ U(0.05, 0.5)

For any x with ||x|| <= 1 and any row j we have
    A_i[j]^T x  <=  ||A_i[j]||  <=  ||A_i[j]|| + slack_j  =  b_i[j],
so every polytope strictly contains the unit L2 ball. With l >= 2n
random Gaussian rows the polytope is bounded with overwhelming probability.

Query points (one per polytope, half guaranteed-inside / half typically-outside):
    Inside : random unit direction * radius in [0, 0.95]
             (||q|| < 1, hence inside any polytope containing the unit ball).
    Outside: random unit direction * radius in [3, 6]
             (typically lands past at least one supporting facet).
The actual ``inside_mask`` is computed per-batch from A_i @ q_i <= b_i so
the label agrees with what the QP solver sees.

Solvers run in-process (no subprocess wrapping) so
``nsys profile python3 benchmark_projection.py ...`` directly captures the
NVTX ranges defined in batched_solver_interface.py. Outer loop is batch
size, inner is solver, with smaller batches being prefixes of larger ones
so cross-B comparisons share the same problem family.

This script only runs the benchmark and writes JSON. Run ``plot_results.py``
on the resulting JSON to produce the figures.
"""
import argparse
import gc
import json
import sys
import traceback
from pathlib import Path
_repo_root = str(Path(__file__).resolve().parent.parent.parent.parent)
_batched_dir = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _repo_root)
sys.path.insert(0, _batched_dir)

import numpy as np
import nvtx

from batched_solver_interface import (
    CupiqpDenseBatchedSolver,
    CupiqpSparseBatchedSolver,
    QpaxBatchedSolver,
    QpthBatchedSolver,
    MoreauTorchBatchedSolver,
    MoreauJaxBatchedSolver,
    BatchedQPData,
    SOLVER_COLORS,
)


def _nvtx_color(solver_name: str) -> str:
    """Look up the solver's NVTX color from the shared interface table.

    ``SOLVER_COLORS`` keys cuPIQP variants under the base name "cupiqp",
    so we fall back to the part before the first '-' for that mapping.
    """
    return SOLVER_COLORS.get(solver_name,
                             SOLVER_COLORS.get(solver_name.split("-")[0], "gray"))


_SOLVER_CLASSES = {
    "cupiqp-dense":   CupiqpDenseBatchedSolver,
    "cupiqp-sparse":  CupiqpSparseBatchedSolver,
    "qpax":           QpaxBatchedSolver,
    "qpth":           QpthBatchedSolver,
    "moreau-torch":   MoreauTorchBatchedSolver,
    "moreau-jax":     MoreauJaxBatchedSolver,
}

SOLVER_NAMES = [
    "cupiqp-dense",
    "cupiqp-sparse",
    "qpax",
    "moreau-torch",
    "qpth",
    "moreau-jax",
]

_COL_WIDTH = 10 + 2 + 10 + 2 + 5 + 2 + 11
_SEP = " | "

# ANSI color helpers for the run-header block. Auto-disabled if stdout is
# not a TTY (e.g. piped to a file) so the .log files stay clean.
_USE_COLOR = sys.stdout.isatty()
_BLUE = "\033[94m" if _USE_COLOR else ""
_RESET = "\033[0m" if _USE_COLOR else ""


def _blue(s: str) -> str:
    return f"{_BLUE}{s}{_RESET}"


# ----------------------------------------------------------------------
# Polytope + query generation (cheap; done once per benchmark invocation).
# ----------------------------------------------------------------------

def generate_polytopes(B: int, n: int, n_inequalities: int, seed: int,
                       slack_lo: float = 0.05, slack_hi: float = 0.5,
                       condition_number: float = 1.0,
                       redundancy_frac: float = 0.0):
    """B random polytopes in R^n, each strictly containing an inscribed ellipsoid.

    Two ill-conditioning knobs (both default to off, reproducing the original
    "isotropic, unit ball inside" behavior):

    * ``condition_number`` (kappa >= 1.0) controls per-batch input-space
      anisotropy. Per batch we draw ``log10(d_i) ~ U(-log10(K)/2, +log10(K)/2)``
      for ``i = 1..n`` and use ``D = diag(d_1..d_n)``. The polytope is built
      so that the ellipsoid ``E = {x : sum_i x_i^2 / d_i^2 <= 1}`` (semi-axes
      ``d_i``, condition number ``K``) lies strictly inside. ``K = 1`` reduces
      to the unit-ball-inside guarantee.

    * ``redundancy_frac`` (in ``[0, 1)``) controls active-set degeneracy. The
      last ``n_dup = floor(F * n_inequalities)`` rows are replaced by
      near-parallel duplicates of randomly chosen "parent" rows in the first
      ``n_normal = n_inequalities - n_dup`` slots, with relative perturbation
      ``1e-6``. ``b`` is then derived from the (perturbed) ``A`` so the
      ellipsoid-inside guarantee still holds verbatim. Capped so at least
      ``n + 1`` normal rows remain (otherwise the polytope can be unbounded).

    Returns ``(A, b, D)`` with ``A`` shape ``(B, n_inequalities, n)``,
    ``b`` shape ``(B, n_inequalities)``, ``D`` shape ``(B, n)``.
    """
    if condition_number < 1.0:
        raise ValueError(f"condition_number={condition_number} must be >= 1.0")
    if not (0.0 <= redundancy_frac < 1.0):
        raise ValueError(f"redundancy_frac={redundancy_frac} must be in [0, 1)")

    rng = np.random.default_rng(seed)

    # 1) Random base rows, A ~ N(0, I).
    A = rng.standard_normal((B, n_inequalities, n))

    # 2) Optional near-redundant rows (replaces the LAST n_dup slots so the
    #    "normal" rows live in a contiguous prefix). Cap n_dup so at least
    #    n + 1 normal rows remain — needed for a bounded polytope w.h.p.
    n_dup_target = int(redundancy_frac * n_inequalities)
    n_dup_cap = max(0, n_inequalities - (n + 1))
    n_dup = min(n_dup_target, n_dup_cap)
    if n_dup > 0:
        n_normal = n_inequalities - n_dup
        # Per-batch independent parent selection so different polytopes have
        # different degenerate vertex structure.
        parent_idx = rng.integers(0, n_normal, size=(B, n_dup))
        # ``np.take_along_axis`` indexes along the row axis (axis=1).
        parents = np.take_along_axis(
            A, parent_idx[..., None], axis=1
        )  # (B, n_dup, n)
        eps = 1e-6
        noise = eps * rng.standard_normal((B, n_dup, n))
        A[:, n_normal:, :] = parents + noise

    # 3) Optional anisotropic input-space scaling D, log-uniform with cond(D)=K.
    if condition_number > 1.0:
        log_half = 0.5 * np.log10(condition_number)
        D = 10.0 ** rng.uniform(-log_half, +log_half, size=(B, n))
    else:
        D = np.ones((B, n), dtype=np.float64)

    # 4) Scaled row norms = support function of A over the inscribed ellipsoid:
    #    h_E(A_row) = ||A_row * D||_2 (since E = D * unit_ball).
    A_scaled = A * D[:, None, :]                                 # (B, l, n)
    scaled_row_norms = np.linalg.norm(A_scaled, axis=-1)          # (B, l)

    # 5) b = scaled_row_norm + slack > 0 → ellipsoid strictly inside polytope.
    slack = rng.uniform(slack_lo, slack_hi, (B, n_inequalities))
    b = scaled_row_norms + slack
    return A, b, D


def generate_queries(B: int, n: int, A: np.ndarray, b: np.ndarray, D: np.ndarray,
                     seed: int,
                     in_radius_lo: float = 0.0, in_radius_hi: float = 0.95,
                     out_radius_lo: float = 3.0, out_radius_hi: float = 6.0):
    """One query point per polytope; half guaranteed-inside, half typically-outside.

    Inside half: ``q = D[b] * (z * r)`` with ``r in [0, 0.95]`` and ``z`` on
    the unit sphere. Then ``sum_i q_i^2 / D[b,i]^2 = r^2 < 1``, so ``q`` is in
    the inscribed ellipsoid, hence in the polytope.

    Outside half: same construction with ``r in [3, 6]``. With anisotropic
    ``D``, ``q`` lies outside the ellipsoid but may still be inside the
    polytope along directions where the polytope extends beyond the ellipsoid
    (by ``slack`` per facet); membership is therefore checked post-hoc.

    Returns ``(qs, inside_mask)``. The mask is the *actual* membership
    (A_i @ q_i <= b_i for all rows, vectorized over batches).
    """
    rng = np.random.default_rng(seed + 1)
    n_in = B // 2
    n_out = B - n_in

    z = rng.standard_normal((B, n))
    z /= np.linalg.norm(z, axis=1, keepdims=True)

    r = np.empty(B, dtype=np.float64)
    r[:n_in] = rng.uniform(in_radius_lo, in_radius_hi, n_in)
    r[n_in:] = rng.uniform(out_radius_lo, out_radius_hi, n_out)

    qs = D * (z * r[:, None])                                     # (B, n)
    perm = rng.permutation(B)
    qs = qs[perm]

    # Per-batch membership: A[i] @ qs[i] <= b[i].
    products = np.einsum('bln,bn->bl', A, qs)                     # (B, l)
    inside_mask = (products <= b + 1e-9).all(axis=1)
    return qs, inside_mask


def generate_problem(B_max: int, n: int, n_inequalities: int, seed: int,
                     condition_number: float = 1.0,
                     redundancy_frac: float = 0.0):
    """One-shot generation of the full polytope batch + queries."""
    A, b, D = generate_polytopes(
        B_max, n, n_inequalities, seed,
        condition_number=condition_number,
        redundancy_frac=redundancy_frac,
    )
    qs, inside_mask = generate_queries(B_max, n, A, b, D, seed)
    return A, b, qs, inside_mask


# ----------------------------------------------------------------------
# QP-batch + correctness check.
# ----------------------------------------------------------------------

def _make_data(B: int, A_all, b_all, qs_all, inside_all):
    """Slice the pre-generated arrays to batch size B and build BatchedQPData."""
    if B > qs_all.shape[0]:
        raise ValueError(f"B={B} exceeds B_max={qs_all.shape[0]}; "
                         f"increase the largest --batch-sizes value upfront.")
    A_b = A_all[:B]                                     # (B, l, n)
    b_b = b_all[:B]                                     # (B, l)
    qs = qs_all[:B]                                     # (B, n)
    q_inside = inside_all[:B]                           # (B,)
    n = A_b.shape[2]

    Ps = np.tile(np.eye(n)[None], (B, 1, 1))            # P = I
    cs = -qs.copy()                                     # c = -q

    data = BatchedQPData(P=Ps, c=cs, G=A_b, h_u=b_b)
    return data, A_b, b_b, qs, q_inside


def _verify_projection(A, b, qs, x, q_inside, tol: float):
    """Per-batch correctness check (each polytope is its own).

    1. Each x[i] must be inside its polytope (A[i] x[i] <= b[i] + tol).
    2. If q[i] is already inside polytope_i, then x[i] must equal q[i].

    ``tol`` should be roughly an order of magnitude looser than the solver's
    absolute tolerance, since IPM convergence is in the KKT residual, not in
    the constraint residual directly.
    """
    B = qs.shape[0]
    products = np.einsum('bln,bn->bl', A, x)            # (B, l)
    viol = np.maximum(products - b, 0.0)                # (B, l)
    max_viol_per_batch = viol.max(axis=1)               # (B,)
    n_x_in_poly = int((max_viol_per_batch < tol).sum())

    diffs = np.linalg.norm(x - qs, axis=1)              # (B,)
    if q_inside.any():
        max_proj_err_inside_q = float(diffs[q_inside].max())
        n_correct_inside_q = int((diffs[q_inside] < tol).sum())
    else:
        max_proj_err_inside_q = 0.0
        n_correct_inside_q = 0

    return {
        "n_x_in_polytope": n_x_in_poly,
        "B": int(B),
        "n_q_inside_truth": int(q_inside.sum()),
        "n_correct_inside_q": n_correct_inside_q,
        "max_constraint_violation": float(max_viol_per_batch.max()),
        "max_proj_err_for_inside_q": max_proj_err_inside_q,
        "mean_proj_distance": float(diffs.mean()),
    }


# ----------------------------------------------------------------------
# Single in-process solver run (NVTX-visible).
# ----------------------------------------------------------------------

def _run_one(solver_name: str, B: int,
             A_all, b_all, qs_all, inside_all,
             tol: float, max_iter: int, n_repeats: int):
    """Run one solver on one batch size for the projection problem."""
    cls = _SOLVER_CLASSES[solver_name]
    color = _nvtx_color(solver_name)

    try:
        data, A, b, qs, q_inside = _make_data(B, A_all, b_all, qs_all, inside_all)
        solver = cls(tol_abs=tol, max_iter=max_iter)
        with nvtx.annotate(f"{solver_name}::B={B}::full", color=color):
            r = solver.benchmark(data, n_repeats)

        # Verify against a constraint-violation / projection-error tolerance one
        # decade looser than the IPM's absolute KKT tolerance.
        v = _verify_projection(A, b, qs, r.x, q_inside, tol=10.0 * tol)

        rec = {
            "ok": True,
            "setup_time_ms": float(r.setup_time_ms),
            "solve_time_ms": float(r.solve_time_ms),
            "solve_times_all": [float(t) for t in r.solve_times_all],
            "n_solved": int(r.n_solved),
            "total": int(r.total),
            "n_iter_max": int(r.n_iter_max),
            "n_inequalities": int(b.shape[1]),
            **v,
        }
        nw = max(len(s) for s in _SOLVER_CLASSES)         # solver-name width
        bw = max(len(str(B)), 3)                          # batch-count width
        # qpth doesn't expose a per-problem iter count (n_iter_max == -1); show "-".
        iter_str = f"{r.n_iter_max:>3d}" if r.n_iter_max >= 0 else f"{'-':>3s}"
        print(f"  [{solver_name:<{nw}s}] B={B}: "
              f"setup={r.setup_time_ms:8.1f}ms "
              f"solve={r.solve_time_ms:7.2f}ms "
              f"iter={iter_str} "
              f"n_solved={r.n_solved:>{bw}d}/{r.total:<{bw}d} "
              f"x_in_poly={v['n_x_in_polytope']:>{bw}d}/{B:<{bw}d} "
              f"q_in/out={v['n_q_inside_truth']:>{bw}d}/{B - v['n_q_inside_truth']:<{bw}d} "
              f"max_viol={v['max_constraint_violation']:.2e}",
              flush=True)
        return rec
    except Exception as e:
        nw = max(len(s) for s in _SOLVER_CLASSES)
        print(f"  [{solver_name:<{nw}s}] B={B} skipped: {type(e).__name__}: {e}",
              flush=True)
        traceback.print_exc()
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


# ----------------------------------------------------------------------
# Orchestrator (in-process; outer = problem size, inner = solver).
# ----------------------------------------------------------------------

def _print_table(batch_sizes, per_solver_results):
    solver_names = list(per_solver_results.keys())
    col = f"{'setup':>10s}  {'solve':>10s}  {'iter':>5s}  {'x in poly':>11s}"
    name_hdr = _SEP.join(f"{name:^{_COL_WIDTH}s}" for name in solver_names)
    sub_hdr = f"{'B':>6s}" + _SEP + _SEP.join([col] * len(solver_names))
    print()
    print(f"{'':>6s}" + _SEP + name_hdr)
    print(sub_hdr)
    print("-" * len(sub_hdr))

    for B in batch_sizes:
        cells = []
        for name in solver_names:
            rec = per_solver_results[name].get(str(B))
            if rec is None or not rec.get("ok", False):
                cells.append(f"{'N/A':>10s}  {'N/A':>10s}  {'N/A':>5s}  {'N/A':>11s}")
            else:
                ncorr = rec.get("n_x_in_polytope", -1)
                ntot = rec.get("B", B)
                if rec["n_iter_max"] >= 0:
                    n_iter = f"{rec['n_iter_max']:5d}"
                else:
                    n_iter = f"{'-':>5s}"
                cells.append(
                    f"{rec['setup_time_ms']:10.2f}  {rec['solve_time_ms']:10.2f}  "
                    f"{n_iter}  {ncorr:5d}/{ntot:<5d}")
        print(f"{B:6d}" + _SEP + _SEP.join(cells), flush=True)


def run_benchmark(n=5, n_inequalities=None, batch_sizes=None,
                  tol=1e-8, max_iter=300, n_repeats=10, seed=10,
                  solvers=None,
                  condition_number: float = 1.0,
                  redundancy_frac: float = 0.0):
    if batch_sizes is None:
        batch_sizes = [16, 32, 64, 128, 256, 512]
    if solvers is None:
        solvers = list(SOLVER_NAMES)
    if n_inequalities is None:
        n_inequalities = 5 * n

    B_max = max(batch_sizes)

    # Effective near-redundant count after the (n + 1) lower bound on normal rows.
    n_dup_effective = min(int(redundancy_frac * n_inequalities),
                          max(0, n_inequalities - (n + 1)))

    # Generate the polytope batch + queries once. Smaller B in the sweep
    # uses a prefix slice, so all batch sizes share the same problem family.
    inscribed = ("unit L2 ball" if condition_number == 1.0
                 else f"ellipsoid with cond=={condition_number:g}")
    print(_blue(f"Polytope projection (per-batch polytope): n={n}, n_inequalities={n_inequalities}, "
                f"B_max={B_max}, seed={seed}"))
    print(_blue(f"  min 0.5 ||x_i - q_i||^2  s.t.  A_i x_i <= b_i  (one polytope per batch)"))
    print(_blue(f"  Each polytope contains the {inscribed} by construction."))
    if condition_number != 1.0 or redundancy_frac > 0.0:
        print(_blue(f"  ill-cond: condition_number={condition_number:g}, "
                    f"redundancy_frac={redundancy_frac:g} "
                    f"(=> {n_dup_effective}/{n_inequalities} near-parallel rows)"))
    print(_blue(f"  tol={tol}, max_iter={max_iter}, n_repeats={n_repeats}"))

    A_all, b_all, qs_all, inside_all = generate_problem(
        B_max, n, n_inequalities, seed,
        condition_number=condition_number,
        redundancy_frac=redundancy_frac,
    )
    print(_blue(f"  generated {B_max} polytopes/queries: "
                f"{int(inside_all.sum())} inside / {int((~inside_all).sum())} outside\n"))

    per_solver_raw = {name: {} for name in solvers}
    for B in batch_sizes:
        print(f"\n{'#'*60}")
        print(f"#  Problem size B = {B}")
        print(f"{'#'*60}")
        with nvtx.annotate(f"sweep::B={B}", color="gray"):
            for name in solvers:
                per_solver_raw[name][str(B)] = _run_one(
                    name, B, A_all, b_all, qs_all, inside_all,
                    tol, max_iter, n_repeats,
                )
                gc.collect()

    _print_table(batch_sizes, per_solver_raw)

    results = {}
    for name in solvers:
        agg = dict(solve_median=[], solve_stderr=[],
                   throughput=[], throughput_stderr=[],
                   n_solved=[], total=[], x_in_poly_frac=[],
                   n_iter_max=[])
        for B in batch_sizes:
            rec = per_solver_raw[name].get(str(B))
            if rec is None or not rec.get("ok", False):
                for k in ("solve_median", "solve_stderr",
                          "throughput", "throughput_stderr"):
                    agg[k].append(float("nan"))
                agg["n_solved"].append(0)
                agg["total"].append(B)
                agg["x_in_poly_frac"].append(0.0)
                agg["n_iter_max"].append(-1)   # treated as "no info"
                continue
            times = np.array(rec["solve_times_all"])
            throughputs = B / (times / 1000.0)
            agg["solve_median"].append(float(np.median(times)))
            agg["solve_stderr"].append(float(np.std(times) / np.sqrt(len(times))))
            agg["throughput"].append(float(np.median(throughputs)))
            agg["throughput_stderr"].append(float(np.std(throughputs) / np.sqrt(len(throughputs))))
            agg["n_solved"].append(rec["n_solved"])
            agg["total"].append(rec["total"])
            n_in = rec.get("n_x_in_polytope", 0)
            ntot = rec.get("B", B)
            agg["x_in_poly_frac"].append(n_in / ntot if ntot > 0 else 0.0)
            agg["n_iter_max"].append(int(rec.get("n_iter_max", -1)))
        results[name] = agg

    params = dict(n=n, n_inequalities=int(n_inequalities), B_max=int(B_max),
                  seed=int(seed), tol=tol, max_iter=max_iter,
                  condition_number=float(condition_number),
                  redundancy_frac=float(redundancy_frac),
                  n_dup_effective=int(n_dup_effective))
    return results, batch_sizes, params


# ----------------------------------------------------------------------
# Persistence.
# ----------------------------------------------------------------------

def save_json(results, batch_sizes, params, path: Path):
    payload = {
        "params": params,
        "batch_sizes": list(batch_sizes),
        "results": results,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    print(f"Results saved to {path}")


# ----------------------------------------------------------------------
# CLI.
# ----------------------------------------------------------------------

if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    default_json = here / "results" / "benchmark_projection.json"

    p = argparse.ArgumentParser(
        description="Run the polytope-projection benchmark and save JSON. "
                    "Use plot_results.py on the saved JSON to draw figures.",
    )
    p.add_argument("--save", type=Path, default=default_json,
                   help=f"Save results to this JSON file (default: {default_json}).")

    p.add_argument("--n", type=int, default=5, help="Polytope dimension.")
    p.add_argument("--n-inequalities", dest="n_inequalities", type=int, default=None,
                   help="Number of inequality constraints (rows of A) per polytope. "
                        "Defaults to 5 * n. Note: not all of these are necessarily "
                        "*facets* of the polytope; some may be redundant. >= 2*n "
                        "keeps each polytope bounded with high probability.")
    p.add_argument("--seed", type=int, default=10)

    p.add_argument("--tol", type=float, default=1e-8)
    p.add_argument("--batch-sizes", dest="batch_sizes", type=int, nargs="+", default=None)
    p.add_argument("--max-iter", dest="max_iter", type=int, default=100)
    p.add_argument("--n-repeats", dest="n_repeats", type=int, default=10)
    p.add_argument("--solvers", nargs="+", default=None,
                   help=f"Subset of solvers to run (default: all of {SOLVER_NAMES}).")

    # Ill-conditioning knobs (default off -> reproduces the original
    # isotropic-unit-ball-inside polytope and clean rows).
    p.add_argument("--condition-number", dest="condition_number",
                   type=float, default=1e8,
                   help="Per-batch random log-uniform diagonal scaling D with "
                        "cond(D) == K, so each polytope contains an inscribed "
                        "ellipsoid of semi-axes ratio K. Default 1.0 "
                        "(isotropic, unit ball inside).")
    p.add_argument("--redundancy-frac", dest="redundancy_frac",
                   type=float, default=0.5,
                   help="Fraction of inequality rows replaced by near-parallel "
                        "duplicates (eps=1e-6) of other rows. Causes IPM "
                        "active-set degeneracy. Default 0.0. Capped so that "
                        "at least n+1 'normal' rows remain.")
    args = p.parse_args()

    results, batch_sizes, params = run_benchmark(
        n=args.n, n_inequalities=args.n_inequalities,
        batch_sizes=args.batch_sizes, tol=args.tol,
        max_iter=args.max_iter, n_repeats=args.n_repeats, seed=args.seed,
        solvers=args.solvers,
        condition_number=args.condition_number,
        redundancy_frac=args.redundancy_frac,
    )
    save_json(results, batch_sizes, params, args.save)
    print(f"To plot: python plot_results.py {args.save}")
