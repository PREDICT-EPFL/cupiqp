"""Benchmark Portfolio-optimization QP across the unified single-QP solver
interface.

Mean-variance portfolio optimization with a low-rank-plus-diagonal risk
model. Decision variable ``z = [x; y]`` (x = portfolio weights, y =
factor exposures), with ``Sigma = F F^T + D``. Introducing the auxiliary
``y = F^T x`` turns the original problem into the standard QP

    min   x^T D x + y^T y - 1/(2 gamma) mu^T x
    s.t.  F^T x - y = 0           (k equalities)
          1^T x = 1                (1 equality)
          0 <= x <= 1              (n box bounds)

Settings match **ClarabelBenchmarks'**
``src/problem_sets/qp/portfolio_optimization.jl``
(https://github.com/oxfordcontrol/ClarabelBenchmarks): ``F`` density
0.3, ``k = ceil(0.1 n)``, per-n seed ``271324 + n``, the un-halved
objective above (so the standard-form Hessian is
``P = 2 * blockdiag(D, I_k)``).

The script sweeps over (n, k = round(k_n_ratio * n)) and runs every
solver registered in
``benchmarks/single/single_solver_interface.py``. Use ``--device`` to
pick {cpu, gpu, all}. Solvers whose dependencies aren't importable are
skipped silently via ``available_solvers``.

Usage:
    python benchmark_portfolio.py
    python benchmark_portfolio.py --device cpu
    python benchmark_portfolio.py --device gpu --n 1000 2000 5000
    python benchmark_portfolio.py --solvers cupiqp-sparse cuclarabel
"""

import sys
import os
import json
import argparse
import subprocess
import tempfile

import numpy as np
import scipy.sparse as sp

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(project_root)

from benchmarks.single.single_solver_interface import (
    SingleQPData,
    SingleQPResult,
    available_solvers,
    ALL_SOLVERS,
)

# NOTE on subprocess isolation
# ----------------------------
# Running cupiqp + cuopt (cuDF / RMM underneath) + cuclarabel + ... in one
# process eventually corrupts CUDA state and trips an illegal-memory
# access in whichever solver is next to allocate (typically symptomatic
# only at sufficient problem size). To dodge that, each (n, k, solver)
# cell is run in its own short-lived Python subprocess that exits before
# the next one starts. Pass ``--inproc`` to opt out of the wrapper.


# ---------------------------------------------------------------------------
# Problem generation
# ---------------------------------------------------------------------------

def generate_portfolio_problem(n, k, seed=271324, density=0.3,
                               gamma=1.0) -> SingleQPData:
    """Build the portfolio-optimization QP for random factor-model data.

    Generation matches ClarabelBenchmarks'
    ``portfolio_optimization.jl``
    (https://github.com/oxfordcontrol/ClarabelBenchmarks) so that the
    problem instance is comparable across the two test harnesses:

      * ``F``: ``sprandn(n, k, density=0.3)`` factor loadings (standard
        normal nonzeros, Bernoulli sparsity);
      * ``D``: ``Diagonal(rand(n) * sqrt(k))`` asset-specific risk;
      * ``mu``: ``randn(n)`` expected returns;
      * ``gamma``: risk-aversion scalar (default 1);
      * **objective**: ``x^T D x + y^T y - 1/(2 gamma) mu^T x`` — note
        the missing ``1/2`` factor on the quadratic terms compared to
        the OSQP-docs example, so the standard-form Hessian is
        ``P = 2 * blockdiag(D, I_k)``;
      * **per-size seed**: ``rng = MersenneTwister(271324 + n)``. We
        mirror this by seeding numpy's RNG with ``seed + n`` so that the
        problem instance scales reproducibly with n.

    Parameters
    ----------
    n : int
        Number of assets.
    k : int
        Number of factors (rank of the factor-model decomposition).
    seed : int
        RNG seed *base*; the actual seed is ``seed + n`` so that each
        problem size gets a fresh instance, matching
        ClarabelBenchmarks' ``MersenneTwister(271324 + n)`` convention.
    density : float
        Density of the factor-loading matrix F. ClarabelBenchmarks
        default 0.3.
    gamma : float
        Risk-aversion parameter (>0).

    Returns
    -------
    SingleQPData with sparse P (block-diag) and sparse A. Variable layout
    is z = [x (n); y (k)], total N = n + k.
    """
    # Per-n seed (mirrors ClarabelBenchmarks' ``MersenneTwister(271324 + n)``).
    rng = np.random.default_rng(seed + n)

    # F: n x k Bernoulli-sparse matrix with standard-normal nonzeros,
    # matching ``sprandn(rng, n, k, density)`` in ClarabelBenchmarks.
    F = sp.random(n, k, density=density, format='csc',
                  random_state=rng,
                  data_rvs=rng.standard_normal).astype(np.float64)

    # D: diag(rand(n) * sqrt(k))   — asset-specific risk.
    d_diag = rng.random(n) * np.sqrt(k)
    D = sp.diags(d_diag, format='csc')

    mu = rng.standard_normal(n)

    N = n + k

    # Standard-form Hessian P. ClarabelBenchmarks writes the objective
    # without the conventional 1/2 factor (``x'Dx + y'y``), so in
    # ``min 0.5 z^T P z + c^T z`` form we need ``P = 2 * blockdiag(D, I_k)``.
    P = 2.0 * sp.block_diag([D, sp.eye(k, format='csc')], format='csc')

    # c = [-mu / (2 gamma); zeros(k)].
    c = np.hstack([-mu / (2.0 * gamma), np.zeros(k)])

    # Equality A: [[F^T, -I_k];
    #              [1^T,  0  ]]   shape (k+1, n+k).
    A_eq = sp.bmat([[F.T,                          -sp.eye(k, format='csc')],
                    [sp.csc_matrix(np.ones((1, n))), None]],
                   format='csc')
    b_eq = np.hstack([np.zeros(k), 1.0])

    # Box bounds: x in [0, 1], y unbounded.
    x_l = np.hstack([np.zeros(n),       np.full(k, -np.inf)])
    x_u = np.hstack([np.ones(n),        np.full(k, +np.inf)])

    return SingleQPData(
        P=P, c=c,
        A=A_eq, b=b_eq,
        x_l=x_l, x_u=x_u,
    )


# ---------------------------------------------------------------------------
# Per-cell runner (in-process and subprocess flavours)
# ---------------------------------------------------------------------------

def _run_one_cell_inproc(n, k, solver_name, args) -> dict:
    """Generate the (n, k) problem, run ``solver_name``, return a result dict.

    Called from both the in-process path (``--inproc``) and the subprocess
    worker path (``--worker``). Only one solver is invoked per call, so
    no GPU state can bleed between solvers.
    """
    classes = {cls().name: cls for cls in ALL_SOLVERS}
    cls = classes.get(solver_name)
    if cls is None:
        raise KeyError(f"unknown solver: {solver_name!r}")

    data = generate_portfolio_problem(
        n=n, k=k, seed=args.seed,
        density=args.density,
        gamma=args.gamma,
    )

    common = dict(
        n=n, k=k, k_n_ratio=args.current_ratio, N=n + k,
        nnz_P=int(data.P.nnz), nnz_A=int(data.A.nnz),
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


def _run_solver_subprocess(solver_name: str, cells, args) -> list:
    """Spawn ONE ``--worker`` subprocess that runs ``solver_name`` on every
    ``(n, k, ratio)`` cell in ``cells``.

    Sharing a single Python process across cells amortises one-time
    per-solver costs — particularly cuClarabel's ~18-27 s Julia JIT —
    over the whole sweep instead of paying them once per cell. Subprocess
    isolation is still applied BETWEEN solvers so cuPIQP / cuClarabel /
    cuOpt / qoco-gpu don't corrupt each other's CUDA state.

    The worker writes its results JSON incrementally, so a crash partway
    through the cell list still recovers what was already finished.
    """
    with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False) as f:
        f.write('[]')  # valid JSON for an early-crash read
        out_path = f.name

    ns      = [str(c[0]) for c in cells]
    ks      = [str(c[1]) for c in cells]
    ratios  = [str(c[2]) for c in cells]

    cmd = [
        sys.executable, os.path.abspath(__file__),
        '--worker',
        '--worker_out', out_path,
        '--worker_solver', solver_name,
        '--worker_n_vals', *ns,
        '--worker_k_vals', *ks,
        '--worker_ratios', *ratios,
        '--n_runs', str(args.n_runs),
        '--max_iter', str(args.max_iter),
        '--tol_abs', str(args.tol_abs),
        '--seed', str(args.seed),
        '--density', str(args.density),
        '--gamma', str(args.gamma),
    ]
    device = {c().name: c.device for c in ALL_SOLVERS}.get(solver_name, '?')

    def _failed_row(n, k, ratio, reason):
        return dict(n=n, k=k, k_n_ratio=ratio, N=n + k,
                    solver_name=solver_name, device=device,
                    finite=False,
                    mean_ms=float('nan'), std_ms=float('nan'),
                    median_ms=float('nan'), setup_ms=float('nan'),
                    all_times=[], iters=-1, obj=float('nan'),
                    solved=False, status=reason)

    try:
        total_timeout = args.subprocess_timeout * max(len(cells), 1)
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=total_timeout)
        with open(out_path) as f:
            results = json.load(f)
        if proc.returncode != 0:
            done = {(r['n'], r['k']) for r in results}
            tail_err = (proc.stderr or '')[-300:]
            for (n, k, ratio) in cells:
                if (n, k) not in done:
                    results.append(_failed_row(
                        n, k, ratio,
                        f"FAILED: subprocess rc={proc.returncode} "
                        f"stderr={tail_err!r}"))
        return results
    except subprocess.TimeoutExpired:
        try:
            with open(out_path) as f:
                results = json.load(f)
        except (OSError, json.JSONDecodeError):
            results = []
        done = {(r['n'], r['k']) for r in results}
        for (n, k, ratio) in cells:
            if (n, k) not in done:
                results.append(_failed_row(
                    n, k, ratio,
                    f"FAILED: subprocess timeout ({total_timeout:.0f}s)"))
        return results
    finally:
        try:
            os.unlink(out_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Solver selection
# ---------------------------------------------------------------------------

def _solver_classes_for(device: str, names_filter):
    """Pick solver classes by device tag, optionally filtered by name list."""
    if device == 'all':
        classes = available_solvers()
    elif device in ('cpu', 'gpu'):
        classes = available_solvers(device)
    else:
        raise ValueError(f"--device must be cpu|gpu|all, got {device!r}")

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
        description='Benchmark portfolio-optimization QP across CPU/GPU solvers.')
    parser.add_argument('--device', choices=['cpu', 'gpu', 'all'],
                        default='all',
                        help='Restrict to CPU solvers, GPU solvers, or both.')
    parser.add_argument('--solvers', nargs='+', default=None,
                        help='Optional whitelist of solver names '
                             '(e.g. cupiqp-sparse osqp). '
                             'Filters within --device selection.')
    parser.add_argument('--n', type=int, nargs='+',
                        default=[100, 1000, 5000, 10000, 15000,
                                 20000, 25000, 30000],
                        help='Sweep over number of assets n. Default '
                             'matches ClarabelBenchmarks.')
    parser.add_argument('--k_n_ratio', type=float, nargs='+',
                        default=[0.1],
                        help='Factor/asset ratios k/n to test '
                             '(default 0.1 -> k = ceil(0.1 n))')
    parser.add_argument('--density', type=float, default=0.3,
                        help='Density of the factor-loading matrix F. '
                             'ClarabelBenchmarks default is 0.3.')
    parser.add_argument('--gamma', type=float, default=1.0,
                        help='Risk-aversion parameter (default 1).')
    parser.add_argument('--seed', type=int, default=271324,
                        help='RNG seed base; the actual numpy seed is '
                             '``seed + n``, matching ClarabelBenchmarks '
                             '``MersenneTwister(271324 + n)``.')
    parser.add_argument('--n_runs', type=int, default=5,
                        help='Timed solves per configuration (excl. warmup).')
    parser.add_argument('--max_iter', type=int, default=200)
    parser.add_argument('--tol_abs', type=float, default=1e-6)
    parser.add_argument('--out_json', type=str, default=None,
                        help='JSON output path (default: ./benchmark_portfolio.json).')
    parser.add_argument('--out_log', type=str, default=None,
                        help='Mirror stdout to this file '
                             '(default: ./benchmark_portfolio.log; pass empty '
                             'string to disable).')
    parser.add_argument('--merge', action='store_true',
                        help='Merge results into --out_json instead of '
                             'overwriting. Existing (n, k, solver_name) '
                             'entries are replaced; the rest are kept.')

    # Isolation knobs.
    parser.add_argument('--inproc', action='store_true',
                        help='Run all cells in the parent process. '
                             'Default is one subprocess per solver (each '
                             'solver processes all cells in sequence).')
    parser.add_argument('--subprocess_timeout', type=float, default=900.0,
                        help='Per-cell subprocess timeout in seconds. '
                             'The total subprocess budget for a solver is '
                             'this times the number of cells.')
    parser.add_argument('--cell_timeout', type=float, default=150.0,
                        help='Per-cell wall-clock budget in seconds inside '
                             'the worker. A cell that exceeds this is '
                             'recorded as FAILED (TIMEOUT) and the sweep '
                             'continues with the next cell. Daemon-thread '
                             'based, so the slow cell may continue running '
                             'in the background until it finishes or the '
                             'worker exits.')

    # Worker mode — internal, used by the subprocess wrapper. The parent
    # never sets these by hand.
    parser.add_argument('--worker', action='store_true',
                        help=argparse.SUPPRESS)
    parser.add_argument('--worker_out', type=str, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument('--worker_solver', type=str, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument('--worker_n_vals', type=int, nargs='+', default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument('--worker_k_vals', type=int, nargs='+', default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument('--worker_ratios', type=float, nargs='+',
                        default=None, help=argparse.SUPPRESS)

    args = parser.parse_args()

    # Worker mode: run ``worker_solver`` on every (n, k, ratio) cell in
    # the parallel ``worker_n_vals / worker_k_vals / worker_ratios`` lists.
    # Results are written incrementally so a crash mid-sweep still
    # recovers what was already finished.
    #
    # Per-cell timeout is implemented with SIGALRM (NOT a separate
    # thread). A daemon-thread wrapper turned out to break cuPIQP: the
    # CUDA context gets thread-bound on the first cell and subsequent
    # cells in a new thread see corrupted state (immediate
    # ``PIQP_NUMERICAL_ISSUES`` at iter=0). SIGALRM keeps everything in
    # the main thread; the trade-off is that it cannot interrupt code
    # currently inside a C extension (cuDSS factor, juliacall, ...),
    # but in practice those operations finish reasonably quickly and
    # the alarm fires as soon as control returns to Python.
    if args.worker:
        import signal
        cells = list(zip(args.worker_n_vals, args.worker_k_vals, args.worker_ratios))
        accumulated: list = []

        class _CellTimeout(Exception):
            pass

        def _alarm_handler(signum, frame):
            raise _CellTimeout

        signal.signal(signal.SIGALRM, _alarm_handler)
        for n, k, ratio in cells:
            args.current_ratio = ratio
            signal.alarm(int(args.cell_timeout))
            try:
                row = _run_one_cell_inproc(n, k, args.worker_solver, args)
            except _CellTimeout:
                device = ({c().name: c.device for c in ALL_SOLVERS}
                          .get(args.worker_solver, '?'))
                row = dict(n=n, k=k, k_n_ratio=ratio, N=n + k,
                           solver_name=args.worker_solver, device=device,
                           finite=False,
                           mean_ms=float('nan'), std_ms=float('nan'),
                           median_ms=float('nan'), setup_ms=float('nan'),
                           all_times=[], iters=-1, obj=float('nan'),
                           solved=False,
                           status=f'FAILED: cell timeout '
                                  f'({args.cell_timeout:.0f}s)')
            finally:
                signal.alarm(0)
            accumulated.append(row)
            with open(args.worker_out, 'w') as f:
                json.dump(accumulated, f)
        return

    # Tee stdout into a log file so per-solver lines + summary table land
    # on disk alongside the JSON/PNG. Pass --out_log "" to disable.
    log_path = (
        args.out_log if args.out_log is not None
        else os.path.join(current_dir, 'benchmark_portfolio.log')
    )
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

    print(f"Solvers ({args.device}): "
          f"{[cls().name for cls in solver_classes]}")
    print()

    # Build the full (n, k, ratio) cell list upfront so per-solver
    # subprocesses can iterate it.
    import math
    cells = []
    for ratio in args.k_n_ratio:
        for n in args.n:
            k = max(1, math.ceil(n * ratio))
            cells.append((n, k, ratio))

    results = []
    print(f"Isolation: "
          f"{'in-process' if args.inproc else 'subprocess per solver'} "
          f"(cell timeout: {args.cell_timeout:.0f}s)\n")

    def _print_row(name: str, r: dict) -> None:
        head = f"  [n={r['n']:>6} k={r['k']:>5}] {name:>15}  [{r['device']}]"
        if r.get('finite'):
            print(f"{head}  {r['mean_ms']:8.2f} ± {r['std_ms']:.2f} ms  "
                  f"setup={r['setup_ms']:7.2f} ms  "
                  f"iter={r['iters']:>3}  obj={r['obj']:.6e}  "
                  f"{'OK' if r['solved'] else 'NOT_SOLVED'} "
                  f"({r['status']})")
        else:
            print(f"{head}  {r['status']}")

    if args.inproc:
        # Parent-process path: iterate cells then solvers, like before.
        for (n, k, ratio) in cells:
            args.current_ratio = ratio
            print(f"{'=' * 70}\nPortfolio:  n={n}  k={k}  "
                  f"(N = n + k = {n + k})\n{'=' * 70}")
            for cls in solver_classes:
                name = cls().name
                r = _run_one_cell_inproc(n, k, name, args)
                _print_row(name, r)
                results.append(r)
    else:
        # Per-solver subprocess: outer = solver. JIT-heavy backends
        # (cuClarabel's Julia startup, cuPIQP's warp tile-kernel compile)
        # pay their one-time cost once per solver instead of once per
        # cell. Per-cell timeout still applies inside the worker.
        for cls in solver_classes:
            name = cls().name
            print(f"{'=' * 70}\nSolver: {name}\n{'=' * 70}")
            solver_results = _run_solver_subprocess(name, cells, args)
            order = {(n, k): i for i, (n, k, _) in enumerate(cells)}
            solver_results.sort(
                key=lambda r: order.get((r.get('n'), r.get('k')),
                                        len(order)))
            for r in solver_results:
                _print_row(name, r)
                results.append(r)

    # ---- Summary table ----
    print(f"\n{'=' * 100}")
    print(f"Summary (mean ± std over {args.n_runs} runs)")
    print('=' * 100)
    print(f"{'n':>6} {'k':>5} {'N':>7} {'solver':>16} {'dev':>4}  "
          f"{'mean ± std (ms)':>20} {'iter':>5}  {'obj':>13}")
    print('-' * 100)
    for r in results:
        if r['finite']:
            t_str = f"{r['mean_ms']:8.2f} ± {r['std_ms']:.2f}"
            obj_str = f"{r['obj']:.4e}"
        else:
            t_str = 'FAILED'
            obj_str = 'N/A'
        print(f"{r['n']:>6d} {r['k']:>5d} {r['N']:>7d} "
              f"{r['solver_name']:>16} {r['device']:>4}  "
              f"{t_str:>20} {r['iters']:>5d}  {obj_str:>13}")

    # ---- Save JSON ----
    json_path = args.out_json or os.path.join(current_dir, 'benchmark_portfolio.json')
    os.makedirs(os.path.dirname(json_path) or '.', exist_ok=True)

    if args.merge and os.path.exists(json_path):
        with open(json_path) as f:
            existing = json.load(f)
        new_keys = {(r['n'], r['k'], r['solver_name']) for r in results}
        kept = [r for r in existing
                if (r['n'], r['k'], r['solver_name']) not in new_keys]
        merged = kept + results
        merged.sort(key=lambda r: (r['n'], r['k'], r['solver_name']))
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
