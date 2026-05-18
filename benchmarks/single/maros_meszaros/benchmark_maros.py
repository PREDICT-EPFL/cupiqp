"""Benchmark Maros-Meszaros large QPs (n >= 5000) across the unified single-QP
solver interface.

Loads each problem's ``.mat`` from ``tests/data/maros_meszaros/`` and runs
every registered solver. OCP-only solvers (``hpipm``, ``cyqlone``,
``cupiqp-multistage``) are excluded — Maros-Meszaros problems have no
OCP block structure.

Usage:
    python benchmark_maros.py
    python benchmark_maros.py --device cpu
    python benchmark_maros.py --solvers cupiqp-sparse cuclarabel
    python benchmark_maros.py --problems LISWET1 CONT-100
    python plot_results.py                          # plot the saved JSON
"""

import sys
import os
import json
import argparse
import subprocess
import tempfile

import numpy as np
import scipy.io
import scipy.sparse as sp

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(project_root)

from benchmarks.single.single_solver_interface import (
    SingleQPData,
    SingleQPResult,
    available_solvers,
    OsqpSolver,
    PiqpSparseSolver,
    ClarabelSolver,
    QpalmSolver,
    GurobiSolver,
    QocoSolver,
    CupiqpSparseSolver,
    CuClarabelSolver,
    CuoptSolver,
)

# Solver classes runnable on Maros-Meszaros (flat sparse QP). The OCP-only
# solvers (HpipmSolver, CyqloneSolver, CupiqpMultistageSolver) are
# intentionally not listed — they require ``SingleQPData.ocp`` which these
# problems don't carry.
ALL_SOLVERS = [
    OsqpSolver,
    PiqpSparseSolver,
    ClarabelSolver,
    QpalmSolver,
    GurobiSolver,
    QocoSolver,
    CupiqpSparseSolver,
    CuClarabelSolver,
    CuoptSolver,
]


MAROS_DATA_DIR = os.path.join(
    project_root, 'tests', 'data', 'maros_meszaros')


# Problems with n >= 5000, sorted by n (computed from
# ``tests/data/maros_meszaros/maros_meszaros_summary.md``). 32 problems.
DEFAULT_PROBLEMS = [
    # n = 5427
    'QSHIP12L',
    # n = 10000 (7 problems: CVXQP1/2/3_L, HUES-MOD, HUESTIS, POWELL20)
    'CVXQP1_L', 'CVXQP2_L', 'CVXQP3_L', 'HUES-MOD', 'HUESTIS', 'POWELL20',
    # n = 10002 (12 LISWET problems)
    'LISWET1', 'LISWET2', 'LISWET3', 'LISWET4', 'LISWET5', 'LISWET6',
    'LISWET7', 'LISWET8', 'LISWET9', 'LISWET10', 'LISWET11', 'LISWET12',
    # n = 10197
    'CONT-100', 'CONT-101',
    # n = 14999
    'DTOC3',
    # n = 18009
    'UBH1',
    # n = 20200 (4 problems)
    'AUG2D', 'AUG2DC', 'AUG2DCQP', 'AUG2DQP',
    # n = 40397
    'CONT-200', 'CONT-201',
    # n = 90597 / 93261 / 93263
    'CONT-300', 'BOYD1', 'BOYD2',
]


# ---------------------------------------------------------------------------
# Problem loading
# ---------------------------------------------------------------------------

def load_maros_problem(name: str) -> SingleQPData:
    """Load a Maros-Meszaros ``.mat`` into a :class:`SingleQPData`.

    The ``.mat`` files in ``tests/data/maros_meszaros/`` use the PIQP
    convention with keys ``P, c, A, b, G, h_l, h_u, x_l, x_u``. Empty
    constraint blocks (zero rows) are dropped from the returned data.
    """
    mat_path = os.path.join(MAROS_DATA_DIR, f'{name}.mat')
    if not os.path.isfile(mat_path):
        raise FileNotFoundError(f"Maros problem file not found: {mat_path}")
    raw = scipy.io.loadmat(mat_path)

    P = sp.csc_matrix(raw['P'], dtype=np.float64)
    c = np.asarray(raw['c'], dtype=np.float64).ravel()

    kw = dict(P=P, c=c)

    A = sp.csc_matrix(raw['A'], dtype=np.float64)
    if A.shape[0] > 0:
        kw['A'] = A
        kw['b'] = np.asarray(raw['b'], dtype=np.float64).ravel()

    G = sp.csc_matrix(raw['G'], dtype=np.float64)
    if G.shape[0] > 0:
        kw['G'] = G
        kw['h_l'] = np.asarray(raw['h_l'], dtype=np.float64).ravel()
        kw['h_u'] = np.asarray(raw['h_u'], dtype=np.float64).ravel()

    # x_l / x_u are always (n,); some entries may be +/- inf.
    kw['x_l'] = np.asarray(raw['x_l'], dtype=np.float64).ravel()
    kw['x_u'] = np.asarray(raw['x_u'], dtype=np.float64).ravel()

    return SingleQPData(**kw)


# ---------------------------------------------------------------------------
# Solver selection
# ---------------------------------------------------------------------------

def _run_one_cell_inproc(problem: str, solver_name: str, args) -> dict:
    """Generate the named Maros problem, run ``solver_name``, return result."""
    classes = {cls().name: cls for cls in ALL_SOLVERS}
    cls = classes.get(solver_name)
    if cls is None:
        raise KeyError(f"unknown solver: {solver_name!r}")

    data = load_maros_problem(problem)
    common = dict(
        problem=problem,
        n=int(data.n), p=int(data.p), m=int(data.m),
        nnz_P=int(data.P.nnz),
        nnz_A=int(data.A.nnz) if data.A is not None else 0,
        nnz_G=int(data.G.nnz) if data.G is not None else 0,
        solver_name=solver_name, device=cls.device,
    )

    try:
        solver = cls(tol_abs=args.tol_abs, max_iter=args.max_iter)
        res: SingleQPResult = solver.benchmark(data, n_repeats=args.n_runs)
        t_mean = float(np.mean(res.solve_times_all))
        t_std = float(np.std(res.solve_times_all))
        return dict(common,
                    finite=True,
                    mean_ms=t_mean, std_ms=t_std,
                    median_ms=res.solve_time_ms,
                    setup_ms=res.setup_time_ms,
                    all_times=res.solve_times_all,
                    iters=res.n_iter, obj=res.obj,
                    solved=res.solved, status=res.status)
    except Exception as e:
        return dict(common,
                    finite=False,
                    mean_ms=float('nan'), std_ms=float('nan'),
                    median_ms=float('nan'), setup_ms=float('nan'),
                    all_times=[], iters=-1, obj=float('nan'),
                    solved=False,
                    status=f'FAILED: {type(e).__name__}: {e}')


def _run_solver_subprocess(solver_name: str, problems: list, args) -> list:
    """Spawn ONE ``--worker`` subprocess that runs ``solver_name`` on every
    problem in ``problems`` (in the listed order).

    Sharing a single Python process across problems amortises one-time
    per-solver costs — particularly cuClarabel's ~18-27 s Julia JIT and
    cuPIQP's ~25 s warp tile-kernel compile — over the whole problem
    sweep instead of paying them once per cell. Subprocess isolation is
    still applied BETWEEN solvers so cuPIQP / cuClarabel / cuOpt don't
    corrupt each other's CUDA / RMM state.

    The worker writes its results JSON incrementally after each problem,
    so if the subprocess crashes partway through (illegal memory access,
    OOM, etc.) we still recover the results for problems that finished
    and report FAILED for the rest.
    """
    with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False) as f:
        # Seed an empty list so an early crash yields valid JSON.
        f.write('[]')
        out_path = f.name

    cmd = [
        sys.executable, os.path.abspath(__file__),
        '--worker',
        '--worker_out', out_path,
        '--worker_solver', solver_name,
        '--worker_problems', *problems,
        '--n_runs', str(args.n_runs),
        '--max_iter', str(args.max_iter),
        '--tol_abs', str(args.tol_abs),
    ]

    device = {c().name: c.device for c in ALL_SOLVERS}.get(solver_name, '?')

    def _failed_row(problem: str, reason: str) -> dict:
        return dict(problem=problem, solver_name=solver_name, device=device,
                    finite=False,
                    mean_ms=float('nan'), std_ms=float('nan'),
                    median_ms=float('nan'), setup_ms=float('nan'),
                    all_times=[], iters=-1, obj=float('nan'),
                    solved=False, status=reason)

    proc = None
    try:
        # ``subprocess_timeout`` is interpreted as per-problem here, so a
        # full sweep gets timeout * len(problems). cuClarabel's first cell
        # is the slowest (Julia JIT + first solve); after that, each
        # additional problem is much faster.
        total_timeout = args.subprocess_timeout * max(len(problems), 1)
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=total_timeout)
        with open(out_path) as f:
            results = json.load(f)

        if proc.returncode != 0:
            # Partial run: keep whatever the worker managed to flush, mark
            # the remaining problems as FAILED so they're visible in the
            # output rather than silently dropped.
            done = {r['problem'] for r in results}
            tail_err = (proc.stderr or '')[-300:]
            for p in problems:
                if p not in done:
                    results.append(_failed_row(
                        p, f"FAILED: subprocess rc={proc.returncode} "
                           f"stderr={tail_err!r}"))
        return results
    except subprocess.TimeoutExpired:
        try:
            with open(out_path) as f:
                results = json.load(f)
        except (OSError, json.JSONDecodeError):
            results = []
        done = {r['problem'] for r in results}
        for p in problems:
            if p not in done:
                results.append(_failed_row(
                    p, f"FAILED: subprocess timeout ({total_timeout:.0f}s)"))
        return results
    finally:
        try:
            os.unlink(out_path)
        except OSError:
            pass


def _solver_classes_for(device: str, names_filter):
    """Pick solver classes by device, filtered to the local ``ALL_SOLVERS``."""
    if device == 'all':
        classes = available_solvers()
    elif device in ('cpu', 'gpu'):
        classes = available_solvers(device)
    else:
        raise ValueError(f"--device must be cpu|gpu|all, got {device!r}")

    classes = [cls for cls in classes if cls in ALL_SOLVERS]

    if names_filter:
        wanted = set(names_filter)
        classes = [cls for cls in classes if cls().name in wanted]
        unmatched = wanted - {cls().name for cls in classes}
        if unmatched:
            print(f"WARNING: --solvers {sorted(unmatched)} not available "
                  f"or unknown. Known names: "
                  f"{sorted({c().name for c in ALL_SOLVERS})}")
    return classes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Benchmark Maros-Meszaros large QPs (n > 1e4).')
    parser.add_argument('--device', choices=['cpu', 'gpu', 'all'],
                        default='all',
                        help='Restrict to CPU solvers, GPU solvers, or both.')
    parser.add_argument('--solvers', nargs='+', default=None,
                        help='Optional whitelist of solver names. '
                             'Filters within --device selection.')
    parser.add_argument('--problems', nargs='+', default=None,
                        help='Optional whitelist of problem names. '
                             'Default sweeps all 32 problems with n >= 5000.')
    parser.add_argument('--n_runs', type=int, default=5,
                        help='Timed solves per (problem, solver) (excl. warmup).')
    parser.add_argument('--max_iter', type=int, default=250)
    parser.add_argument('--tol_abs', type=float, default=1e-6)
    parser.add_argument('--out_json', type=str, default=None,
                        help='JSON output path '
                             '(default: ./results/benchmark_maros.json).')
    parser.add_argument('--out_log', type=str, default=None,
                        help='Mirror stdout to this file '
                             '(default: ./results/benchmark_maros.log; '
                             'pass empty string to disable).')
    parser.add_argument('--merge', action='store_true',
                        help='Merge results into --out_json instead of '
                             'overwriting. Existing (problem, solver_name) '
                             'entries are replaced; the rest are kept.')

    parser.add_argument('--inproc', action='store_true',
                        help='Run all cells in the parent process. Default '
                             'is one subprocess per solver (each solver '
                             'processes all problems in sequence).')
    parser.add_argument('--subprocess_timeout', type=float, default=900.0,
                        help='Per-problem subprocess timeout in seconds. '
                             'The total subprocess budget for a solver is '
                             'this times the number of problems.')

    parser.add_argument('--worker', action='store_true',
                        help=argparse.SUPPRESS)
    parser.add_argument('--worker_out', type=str, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument('--worker_solver', type=str, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument('--worker_problems', nargs='+', default=None,
                        help=argparse.SUPPRESS)

    args = parser.parse_args()

    if args.worker:
        # Run ``worker_solver`` on every problem in ``worker_problems``,
        # writing the results JSON incrementally after each problem so a
        # crash mid-sweep doesn't lose what was already finished.
        accumulated: list = []
        for p in args.worker_problems:
            r = _run_one_cell_inproc(p, args.worker_solver, args)
            accumulated.append(r)
            with open(args.worker_out, 'w') as f:
                json.dump(accumulated, f)
        return

    results_dir = os.path.join(current_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    if args.out_log is not None:
        log_path = args.out_log
    elif args.merge and args.solvers:
        suffix = '_'.join(s.replace('/', '_') for s in args.solvers)
        log_path = os.path.join(
            results_dir, f'benchmark_maros.{suffix}.log')
    else:
        log_path = os.path.join(results_dir, 'benchmark_maros.log')
    log_fh = open(log_path, 'w', buffering=1) if log_path else None
    if log_fh is not None:
        class _Tee:
            def __init__(self, *streams): self._streams = streams
            def write(self, s):
                for st in self._streams:
                    st.write(s)
            def flush(self):
                for st in self._streams:
                    st.flush()
        sys.stdout = _Tee(sys.__stdout__, log_fh)

    solver_classes = _solver_classes_for(args.device, args.solvers)
    if not solver_classes:
        print("No solvers available for the given selection. Exiting.")
        if log_fh is not None:
            sys.stdout = sys.__stdout__
            log_fh.close()
        return

    problems = args.problems if args.problems else DEFAULT_PROBLEMS

    print(f"Solvers ({args.device}): "
          f"{[cls().name for cls in solver_classes]}")
    print(f"Problems ({len(problems)}): {problems}")
    print()

    results = []
    print(f"Isolation: "
          f"{'in-process' if args.inproc else 'subprocess per solver'}\n")

    def _print_row(name: str, r: dict) -> None:
        if r.get('finite'):
            print(f"  [{r['problem']:>10}] {name:>15}  [{r['device']}]  "
                  f"{r['mean_ms']:9.2f} ± {r['std_ms']:.2f} ms  "
                  f"setup={r['setup_ms']:8.2f} ms  "
                  f"iter={r['iters']:>4}  "
                  f"obj={r['obj']:.6e}  "
                  f"{'OK' if r['solved'] else 'NOT_SOLVED'} "
                  f"({r['status']})")
        else:
            print(f"  [{r['problem']:>10}] {name:>15}  "
                  f"[{r.get('device', '?')}]  {r['status']}")

    if args.inproc:
        # Parent-process path: iterate problems first, then solvers, like
        # the chain_mass_ocp benchmark. No JIT-amortisation benefit since
        # everything lives in one Python process anyway.
        for problem in problems:
            print(f"{'=' * 70}\nProblem: {problem}\n{'=' * 70}")
            for cls in solver_classes:
                name = cls().name
                r = _run_one_cell_inproc(problem, name, args)
                _print_row(name, r)
                results.append(r)
    else:
        # Per-solver subprocess: outer = solver. The subprocess loads the
        # solver once (paying any JIT / kernel-compile cost just once) and
        # iterates all problems in sequence, writing the results JSON
        # incrementally so a crash mid-sweep only loses the tail.
        for cls in solver_classes:
            name = cls().name
            print(f"{'=' * 70}\nSolver: {name}\n{'=' * 70}")
            solver_results = _run_solver_subprocess(name, problems, args)
            # Worker may have run them out of order if a problem failed
            # early — sort by the requested problem order for the log.
            order = {p: i for i, p in enumerate(problems)}
            solver_results.sort(
                key=lambda r: order.get(r.get('problem'), len(order)))
            for r in solver_results:
                _print_row(name, r)
                results.append(r)

    # ---- Summary table ----
    print(f"\n{'=' * 100}")
    print(f"Summary (mean ± std over {args.n_runs} runs)")
    print('=' * 100)
    print(f"{'problem':>14} {'n':>7} {'solver':>16} {'dev':>4}  "
          f"{'mean ± std (ms)':>22} {'iter':>5}  {'obj':>14}")
    print('-' * 100)
    for r in results:
        if r['finite']:
            t_str = f"{r['mean_ms']:9.2f} ± {r['std_ms']:.2f}"
            obj_str = f"{r['obj']:.4e}"
        else:
            t_str = 'FAILED'
            obj_str = 'N/A'
        print(f"{r['problem']:>14} {r.get('n', 0):>7d} "
              f"{r['solver_name']:>16} {r.get('device', '?'):>4}  "
              f"{t_str:>22} {r['iters']:>5d}  {obj_str:>14}")

    # ---- Save JSON ----
    json_path = args.out_json or os.path.join(results_dir, 'benchmark_maros.json')
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    if args.merge and os.path.exists(json_path):
        with open(json_path) as f:
            existing = json.load(f)
        new_keys = {(r['problem'], r['solver_name']) for r in results}
        kept = [r for r in existing
                if (r['problem'], r['solver_name']) not in new_keys]
        merged = kept + results
        merged.sort(key=lambda r: (r['problem'], r['solver_name']))
        n_replaced = len(existing) - len(kept)
        n_added = len(results) - n_replaced
        with open(json_path, 'w') as f:
            json.dump(merged, f, indent=2)
        print(f"\nMerged into {json_path}: replaced {n_replaced} existing "
              f"entries, added {n_added} new ones (total: {len(merged)}).")
    else:
        if args.merge:
            print(f"\n[--merge requested but {json_path} does not exist; "
                  f"writing a fresh file]")
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {json_path}")
    print(f"To plot:  python {os.path.join(current_dir, 'plot_results.py')} "
          f"--in {json_path}")

    if log_fh is not None:
        print(f"Log saved to {log_path}")
        sys.stdout = sys.__stdout__
        log_fh.close()


if __name__ == '__main__':
    main()
