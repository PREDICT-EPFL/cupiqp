"""Benchmark Huber-fitting QP across the unified single-QP solver interface.

Robust least-squares with Huber loss, following the OSQP example:
https://osqp.org/docs/examples/huber.html

Original problem (n features, m observations):

    min   sum_i phi_hub( a_i^T x - b_i )

    phi_hub(u) = u^2,       if |u| <= 1
                 2|u| - 1,  if |u| > 1

Equivalent QP with auxiliary variables u (residual), r, s (split positive
and negative tail) — decision z = [x; u; r; s], total dimension N = n + 3m:

    min   u^T u + 2 1^T (r + s)
    s.t.  A x - u - r + s = b           (m equalities)
          r >= 0,  s >= 0                (bounds on the last 2m vars)

The script sweeps over (n, m) and runs every solver registered in
``benchmarks/single/single_solver_interface.py``. Use ``--device`` to pick
{cpu, gpu, all}. Solvers whose dependencies aren't importable are skipped
silently via ``available_solvers``.

Usage:
    python benchmark_huber.py
    python benchmark_huber.py --device cpu
    python benchmark_huber.py --device gpu --m 5000 10000 50000
    python benchmark_huber.py --device all --m_n_ratio 5 20
    python benchmark_huber.py --solvers cupiqp-sparse cuclarabel
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
# only at sufficient problem size). To dodge that, each (m, n, solver)
# cell is run in its own short-lived Python subprocess that exits before
# the next one starts. Pass ``--inproc`` to opt out of the wrapper.


# ---------------------------------------------------------------------------
# Problem generation
# ---------------------------------------------------------------------------

def generate_huber_problem(m, n, seed=0, density=0.125,
                           outlier_frac=0.05) -> SingleQPData:
    """Build the Huber fitting QP for random regression data.

    Generation matches ClarabelBenchmarks' ``huber_fitting.jl``
    (https://github.com/oxfordcontrol/ClarabelBenchmarks) so that the
    *problem instance* is comparable across the two test harnesses:

      * sparse-normal regression matrix ``A`` of density 0.125;
      * ground-truth coefficients ``x_true = randn(n) / sqrt(n)``;
      * inlier noise (95% of rows) ``0.5 * randn``;
      * outlier noise (5% of rows)  ``10  * rand`` (uniform, positive).

    Parameters
    ----------
    m : int
        Number of residuals (observations).
    n : int
        Number of features.
    seed : int
        RNG seed.
    density : float
        Density of the regression matrix A. ClarabelBenchmarks uses 0.125.
    outlier_frac : float
        Fraction of observations corrupted by heavy-tailed noise.
        ClarabelBenchmarks uses 0.05.

    Returns
    -------
    SingleQPData with sparse P, A and dense c, b, x_l, x_u. Variable layout
    is z = [x (n); u (m); r (m); s (m)], total N = n + 3m.
    """
    rng = np.random.default_rng(seed)

    # Regression matrix A: sprandn(m, n, 0.125) — standard-normal nnz
    # values on a Bernoulli sparsity pattern.
    A_data = sp.random(m, n, density=density, format='csr',
                       random_state=rng,
                       data_rvs=rng.standard_normal).astype(np.float64)

    # x_true = randn(n) / sqrt(n)  (dense gaussian, scaled by 1/sqrt(n)).
    x_true = rng.standard_normal(n) / np.sqrt(n)

    # Noise: 95% rows N(0, 0.25), 5% rows U[0, 10].
    inlier_mask = rng.random(m) >= outlier_frac      # 1 - outlier_frac inliers
    noise = np.where(
        inlier_mask,
        0.5 * rng.standard_normal(m),
        10.0 * rng.random(m),
    )
    b = A_data @ x_true + noise

    N = n + 3 * m

    # P = blkdiag(0_n, 2 I_m, 0_m, 0_m) — factor 2 because objective is
    # 0.5 z^T P z and we want u^T u term.
    P_diag = np.zeros(N)
    P_diag[n:n + m] = 2.0
    P = sp.diags(P_diag, format='csr')

    # c = [0_n; 0_m; 2*1_m; 2*1_m]
    c = np.zeros(N)
    c[n + m:] = 2.0

    # A_eq = [A | -I_m | -I_m | +I_m]   (m x N)
    eye_m = sp.eye(m, format='csr')
    A_eq = sp.hstack([A_data, -eye_m, -eye_m, eye_m], format='csr')

    # Bounds: r, s >= 0; x and u unbounded
    x_l = np.full(N, -np.inf)
    x_u = np.full(N, +np.inf)
    x_l[n + m:] = 0.0

    return SingleQPData(
        P=P, c=c,
        A=A_eq, b=b,
        x_l=x_l, x_u=x_u,
    )


# ---------------------------------------------------------------------------
# Per-cell runner (in-process and subprocess flavours)
# ---------------------------------------------------------------------------

def _run_one_cell_inproc(m, n, solver_name, args) -> dict:
    """Generate the (m, n) problem, run ``solver_name``, return a result dict.

    Called from both the in-process path (``--inproc``) and the subprocess
    worker path (``--worker``). Only one solver is invoked per call, so
    no GPU state can bleed between solvers.
    """
    classes = {cls().name: cls for cls in ALL_SOLVERS}
    cls = classes.get(solver_name)
    if cls is None:
        raise KeyError(f"unknown solver: {solver_name!r}")

    data = generate_huber_problem(
        m=m, n=n, seed=args.seed,
        density=args.density,
        outlier_frac=args.outlier_frac,
    )

    common = dict(
        m=m, n=n, m_n_ratio=args.current_ratio, N=data.n + 3 * m,
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
    ``(m, n, ratio)`` cell in ``cells``. Mirror of the maros / portfolio
    pattern: amortises one-time per-solver costs (cuClarabel's ~18-27 s
    Julia JIT, cuPIQP's warp tile-kernel compile) across the whole sweep.
    """
    with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False) as f:
        f.write('[]')
        out_path = f.name

    ms     = [str(c[0]) for c in cells]
    ns     = [str(c[1]) for c in cells]
    ratios = [str(c[2]) for c in cells]

    cmd = [
        sys.executable, os.path.abspath(__file__),
        '--worker',
        '--worker_out', out_path,
        '--worker_solver', solver_name,
        '--worker_m_vals', *ms,
        '--worker_n_vals', *ns,
        '--worker_ratios', *ratios,
        '--n_runs', str(args.n_runs),
        '--max_iter', str(args.max_iter),
        '--tol_abs', str(args.tol_abs),
        '--seed', str(args.seed),
        '--density', str(args.density),
        '--outlier_frac', str(args.outlier_frac),
        '--cell_timeout', str(args.cell_timeout),
    ]
    device = {c().name: c.device for c in ALL_SOLVERS}.get(solver_name, '?')

    def _failed_row(m, n, ratio, reason):
        return dict(m=m, n=n, m_n_ratio=ratio, N=n + 3 * m,
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
            done = {(r['m'], r['n']) for r in results}
            tail_err = (proc.stderr or '')[-300:]
            for (m, n, ratio) in cells:
                if (m, n) not in done:
                    results.append(_failed_row(
                        m, n, ratio,
                        f"FAILED: subprocess rc={proc.returncode} "
                        f"stderr={tail_err!r}"))
        return results
    except subprocess.TimeoutExpired:
        try:
            with open(out_path) as f:
                results = json.load(f)
        except (OSError, json.JSONDecodeError):
            results = []
        done = {(r['m'], r['n']) for r in results}
        for (m, n, ratio) in cells:
            if (m, n) not in done:
                results.append(_failed_row(
                    m, n, ratio,
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
# Plotting lives in the companion ``plot_results.py`` so the benchmark and
# the plot can be re-run independently from the same JSON.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Benchmark Huber fitting QP across CPU/GPU solvers.')
    parser.add_argument('--device', choices=['cpu', 'gpu', 'all'],
                        default='all',
                        help='Restrict to CPU solvers, GPU solvers, or both.')
    parser.add_argument('--solvers', nargs='+', default=None,
                        help='Optional whitelist of solver names '
                             '(e.g. cupiqp-sparse osqp). '
                             'Filters within --device selection.')
    parser.add_argument('--n', type=int, nargs='+',
                        # default=[*range(500, 25_00+1, 500)],
                        # default=[*range(5000, 25_000+1, 5000)],
                        default=[*range(500, 25_00+1, 500), *range(5000, 25_000+1, 5000)],
                        help='Sweep over number of features n.')
    parser.add_argument('--m_n_ratio', type=float, nargs='+',
                        default=[1.5],
                        help='Observation/feature ratios m/n to test '
                             '(default 1.5 -> m = round(1.5 n), matching '
                             'ClarabelBenchmarks).')
    parser.add_argument('--density', type=float, default=0.125,
                        help='Density of the regression matrix A. '
                             'ClarabelBenchmarks default is 0.125.')
    parser.add_argument('--outlier_frac', type=float, default=0.05,
                        help='Fraction of observations corrupted by '
                             'heavy-tailed noise. ClarabelBenchmarks uses 0.05.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_runs', type=int, default=5,
                        help='Timed solves per configuration (excl. warmup).')
    parser.add_argument('--max_iter', type=int, default=200)
    parser.add_argument('--tol_abs', type=float, default=1e-6)
    parser.add_argument('--out_json', type=str, default=None,
                        help='JSON output path (default: ./benchmark_huber.json).')
    parser.add_argument('--out_log', type=str, default=None,
                        help='Mirror stdout to this file '
                             '(default: ./benchmark_huber.log; pass empty '
                             'string to disable).')
    parser.add_argument('--merge', action='store_true',
                        help='Merge results into --out_json instead of '
                             'overwriting. Existing (m, n, solver_name) '
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
    parser.add_argument('--worker_m_vals', type=int, nargs='+', default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument('--worker_n_vals', type=int, nargs='+', default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument('--worker_ratios', type=float, nargs='+',
                        default=None, help=argparse.SUPPRESS)

    args = parser.parse_args()

    # Worker mode: run ``worker_solver`` on every (m, n, ratio) cell from
    # the parallel ``worker_m_vals / worker_n_vals / worker_ratios`` lists.
    # Per-cell timeout uses SIGALRM (NOT a thread); the daemon-thread
    # variant turned out to break cuPIQP, see comments in
    # ``benchmark_portfolio.py``. SIGALRM keeps everything in the main
    # thread at the cost of not preempting C-extension code.
    if args.worker:
        import signal
        cells = list(zip(args.worker_m_vals, args.worker_n_vals, args.worker_ratios))
        accumulated: list = []

        class _CellTimeout(Exception):
            pass

        def _alarm_handler(signum, frame):
            raise _CellTimeout

        signal.signal(signal.SIGALRM, _alarm_handler)
        for m, n, ratio in cells:
            args.current_ratio = ratio
            signal.alarm(int(args.cell_timeout))
            try:
                row = _run_one_cell_inproc(m, n, args.worker_solver, args)
            except _CellTimeout:
                device = ({c().name: c.device for c in ALL_SOLVERS}
                          .get(args.worker_solver, '?'))
                row = dict(m=m, n=n, m_n_ratio=ratio, N=n + 3 * m,
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
        else os.path.join(current_dir, 'benchmark_huber.log')
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

    # Build the full (m, n, ratio) cell list upfront so per-solver
    # subprocesses can iterate it.
    cells = []
    for ratio in args.m_n_ratio:
        for n in args.n:
            m = max(1, int(round(n * ratio)))
            cells.append((m, n, ratio))

    results = []
    print(f"Isolation: "
          f"{'in-process' if args.inproc else 'subprocess per solver'} "
          f"(cell timeout: {args.cell_timeout:.0f}s)\n")

    def _print_row(name: str, r: dict) -> None:
        head = f"  [m={r['m']:>6} n={r['n']:>5}] {name:>15}  [{r['device']}]"
        if r.get('finite'):
            print(f"{head}  {r['mean_ms']:8.2f} ± {r['std_ms']:.2f} ms  "
                  f"setup={r['setup_ms']:7.2f} ms  "
                  f"iter={r['iters']:>3}  obj={r['obj']:.6e}  "
                  f"{'OK' if r['solved'] else 'NOT_SOLVED'} "
                  f"({r['status']})")
        else:
            print(f"{head}  {r['status']}")

    if args.inproc:
        for (m, n, ratio) in cells:
            args.current_ratio = ratio
            print(f"{'=' * 70}\nHuber fit:  m={m}  n={n}  "
                  f"(N = n + 3m = {n + 3 * m})\n{'=' * 70}")
            for cls in solver_classes:
                name = cls().name
                r = _run_one_cell_inproc(m, n, name, args)
                _print_row(name, r)
                results.append(r)
    else:
        # Per-solver subprocess (JIT amortised across cells).
        for cls in solver_classes:
            name = cls().name
            print(f"{'=' * 70}\nSolver: {name}\n{'=' * 70}")
            solver_results = _run_solver_subprocess(name, cells, args)
            order = {(m, n): i for i, (m, n, _) in enumerate(cells)}
            solver_results.sort(
                key=lambda r: order.get((r.get('m'), r.get('n')),
                                        len(order)))
            for r in solver_results:
                _print_row(name, r)
                results.append(r)

    # ---- Summary table ----
    print(f"\n{'=' * 100}")
    print(f"Summary (mean ± std over {args.n_runs} runs)")
    print('=' * 100)
    print(f"{'m':>7} {'n':>6} {'solver':>16} {'dev':>4}  "
          f"{'mean ± std (ms)':>20} {'iter':>5}  {'obj':>13}")
    print('-' * 100)
    for r in results:
        if r['finite']:
            t_str = f"{r['mean_ms']:8.2f} ± {r['std_ms']:.2f}"
            obj_str = f"{r['obj']:.4e}"
        else:
            t_str = 'FAILED'
            obj_str = 'N/A'
        print(f"{r['m']:>7d} {r['n']:>6d} {r['solver_name']:>16} "
              f"{r['device']:>4}  {t_str:>20} {r['iters']:>5d}  {obj_str:>13}")

    # ---- Save JSON ----
    json_path = args.out_json or os.path.join(current_dir, 'benchmark_huber.json')
    os.makedirs(os.path.dirname(json_path) or '.', exist_ok=True)

    if args.merge and os.path.exists(json_path):
        with open(json_path) as f:
            existing = json.load(f)
        new_keys = {(r['m'], r['n'], r['solver_name']) for r in results}
        kept = [r for r in existing
                if (r['m'], r['n'], r['solver_name']) not in new_keys]
        merged = kept + results
        merged.sort(key=lambda r: (r['m'], r['n'], r['solver_name']))
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
