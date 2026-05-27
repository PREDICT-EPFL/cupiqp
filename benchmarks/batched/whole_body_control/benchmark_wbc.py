"""Unified Wbc benchmark across robot types.

Choose a robot with ``--robot``:

    --robot anymal_c    ANYmal C quadruped     (na=12, nc=4)
    --robot h1          Unitree H1 humanoid    (na=19, nc=2)
    --robot iiwa14      KUKA iiwa14 arm        (na=7,  no contacts)

Each robot is one ``RobotWbc`` subclass in ``robot_wbc.py``. To benchmark
a new robot, add a subclass + registry entry there; no changes here.

Run::

    python benchmark_wbc.py --robot anymal_c --batch-sizes 64 256 1024 4096
    python benchmark_wbc.py --robot iiwa14   --batch-sizes 64 256 1024

By default, every ``(solver, batch size)`` case runs in a fresh subprocess.
This keeps GPU allocator caches from earlier cases out of later measurements.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

_repo_root   = str(Path(__file__).resolve().parent.parent.parent.parent)
_batched_dir = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _repo_root)
sys.path.insert(0, _batched_dir)

import numpy as np

SOLVER_NAMES = ["cupiqp-dense", "cupiqp-sparse", "qpax", "qpth"]
_SOLVER_CLASSES = {}
SOLVER_COLORS = {}
nvtx = None
make_robot = None


def _load_worker_dependencies():
    """Import GPU-facing modules only inside a process that executes a case."""
    global _SOLVER_CLASSES, SOLVER_COLORS, nvtx, make_robot
    if _SOLVER_CLASSES:
        return

    import nvtx as _nvtx
    from batched_solver_interface import (
        CupiqpDenseBatchedSolver,
        CupiqpSparseBatchedSolver,
        QpaxBatchedSolver,
        QpthBatchedSolver,
        SOLVER_COLORS as _solver_colors,
    )
    from robot_wbc import make_robot as _make_robot

    _SOLVER_CLASSES = {
        "cupiqp-dense":  CupiqpDenseBatchedSolver,
        "cupiqp-sparse": CupiqpSparseBatchedSolver,
        "qpax":          QpaxBatchedSolver,
        "qpth":          QpthBatchedSolver,
    }
    SOLVER_COLORS = _solver_colors
    nvtx = _nvtx
    make_robot = _make_robot


# ----------------------------------------------------------------------
# Per-batch equality + inequality residuals.
# ----------------------------------------------------------------------
def _verify_solution(data, x: np.ndarray, tol: float) -> dict:
    """Check (Ax = b) and per-row/per-variable inequalities including the
    double-sided h_l/h_u and the box constraints x_l/x_u."""
    finite_solution = np.isfinite(x).all(axis=1)
    Ax = np.einsum("bij,bj->bi", data.A, x)
    eq_res = np.abs(Ax - data.b).max(axis=1)

    B = x.shape[0]
    in_viol = np.zeros(B)
    if data.G is not None and data.G.shape[1] > 0:
        Gx = np.einsum("bij,bj->bi", data.G, x)
        if data.h_u is not None:
            in_viol = np.maximum(in_viol, np.maximum(Gx - data.h_u, 0.0).max(axis=1))
        if data.h_l is not None:
            in_viol = np.maximum(in_viol, np.maximum(data.h_l - Gx, 0.0).max(axis=1))

    # Box-constraint violations (ignore +/-inf bounds with nan_to_num).
    if data.x_u is not None:
        in_viol = np.maximum(in_viol,
                             np.maximum(x - np.nan_to_num(data.x_u, nan=np.inf,
                                                          posinf=np.inf,
                                                          neginf=-np.inf), 0.0).max(axis=1))
    if data.x_l is not None:
        in_viol = np.maximum(in_viol,
                             np.maximum(np.nan_to_num(data.x_l, nan=-np.inf,
                                                      posinf=np.inf,
                                                      neginf=-np.inf) - x, 0.0).max(axis=1))

    feasible = (finite_solution & np.isfinite(eq_res) & np.isfinite(in_viol)
                & (eq_res < tol) & (in_viol < tol))
    n_feasible = int(feasible.sum())
    return {
        "n_finite_solution": int(finite_solution.sum()),
        "n_feasible": n_feasible,
        "B": int(B),
        "max_equality_residual": float(eq_res.max()),
        "max_inequality_violation": float(in_viol.max()),
        "mean_equality_residual": float(eq_res.mean()),
    }


def _slice_batch(data, B: int):
    """Return a view of the first ``B`` problems."""
    kw = dict(P=data.P[:B], c=data.c[:B])
    for attr in ("A", "b", "G", "h_l", "h_u", "x_l", "x_u"):
        v = getattr(data, attr)
        if v is not None:
            kw[attr] = v[:B]
    return type(data)(**kw)


# ----------------------------------------------------------------------
# Single solver run on one (solver, B) combination.
# ----------------------------------------------------------------------
def _run_one(solver_name, data, tol: float,
             max_iter: int, n_repeats: int):
    cls = _SOLVER_CLASSES[solver_name]
    color = SOLVER_COLORS.get(solver_name.split("-")[0], "gray")
    B = data.B

    try:
        solver = cls(tol_abs=tol, max_iter=max_iter)
        with nvtx.annotate(f"{solver_name}::B={B}::full", color=color):
            r = solver.benchmark(data, n_repeats)

        v = _verify_solution(data, r.x, tol=10.0 * tol)
        status_reported = r.n_solved >= 0
        all_status_solved = not status_reported or r.n_solved == r.total
        all_finite = v['n_finite_solution'] == B
        all_feasible = v['n_feasible'] == B
        ok = all_finite and all_feasible and all_status_solved
        failure_reasons = []
        if not all_finite:
            failure_reasons.append(
                f"finite primal solutions {v['n_finite_solution']}/{B}"
            )
        if not all_feasible:
            failure_reasons.append(f"feasible solutions {v['n_feasible']}/{B}")
        if not all_status_solved:
            failure_reasons.append(f"reported solved {r.n_solved}/{r.total}")
        validation_error = '; '.join(failure_reasons) if failure_reasons else None

        nw = max(len(s) for s in _SOLVER_CLASSES)
        bw = max(len(str(B)), 3)
        iter_str = f"{r.n_iter_max:>3d}" if r.n_iter_max >= 0 else f"{'-':>3s}"
        print(f"  [{solver_name:<{nw}s}] B={B}: "
              f"setup={r.setup_time_ms:8.1f}ms "
              f"solve={r.solve_time_ms:7.2f}ms "
              f"iter={iter_str} "
              f"n_solved={r.n_solved:>{bw}d}/{r.total:<{bw}d} "
              f"n_feasible={v['n_feasible']:>{bw}d}/{B:<{bw}d} "
              f"|eq|={v['max_equality_residual']:.2e}  "
              f"|ineq|={v['max_inequality_violation']:.2e}"
              f"{'' if ok else '  INVALID: ' + validation_error}",
              flush=True)
        return {
            "ok": ok,
            "timed": True,
            "validation_error": validation_error,
            "status_reported": status_reported,
            "setup_time_ms": float(r.setup_time_ms),
            "solve_time_ms": float(r.solve_time_ms),
            "solve_times_all": [float(t) for t in r.solve_times_all],
            "n_solved": int(r.n_solved),
            "total": int(r.total),
            "n_iter_max": int(r.n_iter_max),
            **v,
        }
    except Exception as e:
        nw = max(len(s) for s in _SOLVER_CLASSES)
        print(f"  [{solver_name:<{nw}s}] B={B} skipped: {type(e).__name__}: {e}",
              flush=True)
        traceback.print_exc()
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


# ----------------------------------------------------------------------
# Sweep driver.
# ----------------------------------------------------------------------
def run_benchmark(robot_name, batch_sizes, tol, max_iter, n_repeats, seed,
                  contact_prob, solvers, generation_batch_size=None):
    _load_worker_dependencies()

    # Forward contact_prob only to legged robots (fixed-base arms ignore it).
    robot_kwargs = {}
    if robot_name in ("anymal_c", "h1"):
        robot_kwargs["contact_prob"] = contact_prob
    robot = make_robot(robot_name, **robot_kwargs)
    print(f"robot: {robot}")
    print(f"  n={robot.n_var}, p={robot.n_eq}, m={robot.n_ineq}")
    print(f"  tol={tol}, max_iter={max_iter}, n_repeats={n_repeats}, seed={seed}")

    B_max = generation_batch_size or max(batch_sizes)
    if B_max < max(batch_sizes):
        raise ValueError("generation_batch_size must cover all requested batch sizes.")
    full_data, _ = robot.generate_problems(B_max, seed=seed)

    per_solver = {name: {} for name in solvers}
    for B in batch_sizes:
        print(f"\n{'#' * 60}")
        print(f"#  Problem size B = {B}")
        print(f"{'#' * 60}")
        data = _slice_batch(full_data, B)
        with nvtx.annotate(f"sweep::B={B}", color="gray"):
            for name in solvers:
                per_solver[name][str(B)] = _run_one(
                    name, data, tol, max_iter, n_repeats,
                )
    return per_solver


def run_isolated_benchmark(robot_name, batch_sizes, tol, max_iter, n_repeats,
                           seed, contact_prob, solvers):
    """Run every measured case in a fresh process and merge its result."""
    per_solver = {name: {} for name in solvers}
    script = Path(__file__).resolve()
    generation_batch_size = max(batch_sizes)

    print("execution: fresh subprocess for every (solver, batch size) case")
    print(f"  shared generated batch size={generation_batch_size}, seed={seed}")
    with tempfile.TemporaryDirectory(prefix="wbc_isolated_") as temp_dir:
        temp_dir = Path(temp_dir)
        for name in solvers:
            for B in batch_sizes:
                result_file = temp_dir / f"{robot_name}_{name}_{B}.json"
                print("\n" + "=" * 60)
                print(f"Starting isolated case: solver={name}, B={B}")
                print("=" * 60, flush=True)
                command = [
                    sys.executable, str(script),
                    "--robot", robot_name,
                    "--batch-sizes", str(B),
                    "--tol", str(tol),
                    "--max-iter", str(max_iter),
                    "--n-repeats", str(n_repeats),
                    "--seed", str(seed),
                    "--contact-prob", str(contact_prob),
                    "--solvers", name,
                    "--generation-batch-size", str(generation_batch_size),
                    "--in-process",
                    "--save", str(result_file),
                ]
                subprocess.run(command, check=True)
                case_payload = json.loads(result_file.read_text())
                per_solver[name][str(B)] = case_payload["results"][name][str(B)]
    return per_solver


def save_json(per_solver, batch_sizes, params, path: Path):
    payload = {"params": params,
               "batch_sizes": list(batch_sizes),
               "results": per_solver}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    print(f"\nResults saved to {path}")


if __name__ == "__main__":
    here = Path(__file__).resolve().parent

    p = argparse.ArgumentParser(
        description="Unified batched Wbc benchmark across robot types.",
    )
    p.add_argument("--robot", required=True,
                   help="Which robot's Wbc QP to benchmark.")
    p.add_argument("--batch-sizes", dest="batch_sizes", type=int, nargs="+",
                   default=[64, 128, 256, 512, 1024, 4096])
    p.add_argument("--tol", type=float, default=1e-8)
    p.add_argument("--max-iter", dest="max_iter", type=int, default=200)
    p.add_argument("--n-repeats", dest="n_repeats", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--contact-prob", dest="contact_prob", type=float, default=0.7,
                   help="(Legged only) per-foot probability of being in contact.")
    p.add_argument("--solvers", nargs="+", choices=SOLVER_NAMES, default=None)
    p.add_argument("--save", type=Path, default=None,
                   help=("Path to save JSON results. Defaults to "
                         "results/benchmark_wbc_<robot>.json"))
    p.add_argument("--in-process", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--generation-batch-size", type=int, default=None,
                   help=argparse.SUPPRESS)
    args = p.parse_args()

    out = (args.save if args.save is not None
           else here / "results" / f"benchmark_wbc_{args.robot}.json")
    solvers = args.solvers if args.solvers else SOLVER_NAMES

    if args.in_process:
        per_solver = run_benchmark(
            robot_name=args.robot,
            batch_sizes=args.batch_sizes, tol=args.tol,
            max_iter=args.max_iter, n_repeats=args.n_repeats,
            seed=args.seed, contact_prob=args.contact_prob,
            solvers=solvers, generation_batch_size=args.generation_batch_size,
        )
        isolation = "in-process-worker"
    else:
        per_solver = run_isolated_benchmark(
            robot_name=args.robot,
            batch_sizes=args.batch_sizes, tol=args.tol,
            max_iter=args.max_iter, n_repeats=args.n_repeats,
            seed=args.seed, contact_prob=args.contact_prob,
            solvers=solvers,
        )
        isolation = "process-per-case"
    params = dict(robot=args.robot, tol=args.tol, max_iter=args.max_iter,
                  n_repeats=args.n_repeats, seed=args.seed,
                  contact_prob=args.contact_prob, isolation=isolation)
    save_json(per_solver, args.batch_sizes, params, out)
