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


# ---------------------------------------------------------------------------
# Problem generation
# ---------------------------------------------------------------------------

def generate_huber_problem(m, n, seed=0, density=0.1,
                           outlier_frac=0.1) -> SingleQPData:
    """Build the Huber fitting QP for random regression data.

    Parameters
    ----------
    m : int
        Number of residuals (observations).
    n : int
        Number of features.
    seed : int
        RNG seed.
    density : float
        Density of the regression matrix A.
    outlier_frac : float
        Fraction of observations corrupted by heavy-tailed noise.

    Returns
    -------
    SingleQPData with sparse P, A and dense c, b, x_l, x_u. Variable layout
    is z = [x (n); u (m); r (m); s (m)], total N = n + 3m.
    """
    rng = np.random.default_rng(seed)

    A_data = sp.random(m, n, density=density, format='csr',
                       random_state=rng).astype(np.float64)
    x_true = (rng.random(n) > 0.5).astype(np.float64) * rng.standard_normal(n)

    noise = rng.standard_normal(m)
    out_mask = rng.random(m) < outlier_frac
    if out_mask.any():
        noise[out_mask] = 10.0 * rng.standard_normal(int(out_mask.sum()))
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
# Plotting
# ---------------------------------------------------------------------------

# Stable color palette per solver name; falls back to grey if unknown.
_SOLVER_COLOR = {
    'osqp':           '#9467bd',
    'piqp-sparse':    '#ff7f0e',
    'cupiqp-sparse':  '#1f77b4',
    'cupiqp-dense':   '#aec7e8',
    'qoco-gpu':       '#2ca02c',
    'cuclarabel':     '#d62728',
    'cuopt':          '#8c564b',
    'moreau-torch':   '#e377c2',
}


def plot_results(results, output_path):
    """Solve time vs. m, one line per (m_n_ratio, solver_name)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    series = {}
    for r in results:
        if not r.get('finite', False):
            continue
        key = (r['m_n_ratio'], r['solver_name'])
        series.setdefault(key, {'m': [], 'mean': [], 'std': []})
        series[key]['m'].append(r['m'])
        series[key]['mean'].append(r['mean_ms'])
        series[key]['std'].append(r['std_ms'])

    ratios = sorted({r['m_n_ratio'] for r in results})
    linestyles = ['-', '--', '-.', ':']
    markers = ['o', 's', 'D', '^', 'v', 'P']
    ratio_style = {ratio: (linestyles[i % len(linestyles)],
                           markers[i % len(markers)])
                   for i, ratio in enumerate(ratios)}

    for (ratio, name), data in sorted(series.items()):
        ls, marker = ratio_style[ratio]
        color = _SOLVER_COLOR.get(name, '#7f7f7f')
        order = np.argsort(data['m'])
        ms_vals = np.array(data['m'])[order]
        means = np.array(data['mean'])[order]
        stds = np.array(data['std'])[order]
        ax.plot(ms_vals, means,
                linestyle=ls, marker=marker, color=color,
                linewidth=2, markersize=7,
                label=f'{name}, m/n={ratio:g}')
        ax.fill_between(ms_vals, means - stds, means + stds,
                        color=color, alpha=0.15)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('m (number of observations)', fontsize=13)
    ax.set_ylabel('Solve time (ms)', fontsize=13)
    ax.set_title('Huber Fitting QP Benchmark', fontsize=15)
    ax.legend(fontsize=9, loc='best', ncol=2)
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"\nPlot saved to {output_path}")
    plt.close(fig)


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
                        default=[int(100*i) for i in [1, 2, 4, 6, 8, 16, 32, 64, 128, 256]],
                        help='Sweep over number of features n.')
    parser.add_argument('--m_n_ratio', type=float, nargs='+',
                        default=[2],
                        help='Observation/feature ratios m/n to test '
                             '(default 2 -> m = n*2).')
    parser.add_argument('--density', type=float, default=0.1)
    parser.add_argument('--outlier_frac', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_runs', type=int, default=5,
                        help='Timed solves per configuration (excl. warmup).')
    parser.add_argument('--max_iter', type=int, default=200)
    parser.add_argument('--tol_abs', type=float, default=1e-6)
    parser.add_argument('--out_json', type=str, default=None,
                        help='JSON output path (default: ./benchmark_huber.json).')
    parser.add_argument('--out_plot', type=str, default=None,
                        help='Plot output path (default: ./benchmark_huber.png).')
    parser.add_argument('--out_log', type=str, default=None,
                        help='Mirror stdout to this file '
                             '(default: ./benchmark_huber.log; pass empty '
                             'string to disable).')
    args = parser.parse_args()

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

    results = []

    for ratio in args.m_n_ratio:
        for n in args.n:
            m = max(1, int(round(n * ratio)))

            print(f"{'=' * 70}")
            print(f"Huber fit:  m={m}  n={n}  (N = n + 3m = {n + 3 * m})")
            print('=' * 70)

            data = generate_huber_problem(
                m=m, n=n, seed=args.seed,
                density=args.density,
                outlier_frac=args.outlier_frac,
            )
            print(f"  nnz(P)={int(data.P.nnz)}  nnz(A)={int(data.A.nnz)}")

            for cls in solver_classes:
                solver = cls(tol_abs=args.tol_abs, max_iter=args.max_iter)
                name = solver.name
                try:
                    res: SingleQPResult = solver.benchmark(
                        data, n_repeats=args.n_runs)
                    t_mean = float(np.mean(res.solve_times_all))
                    t_std = float(np.std(res.solve_times_all))
                    t_median = res.solve_time_ms
                    print(f"  {name:>15}  [{res.device}]  "
                          f"{t_mean:8.2f} ± {t_std:.2f} ms  "
                          f"setup={res.setup_time_ms:7.2f} ms  "
                          f"iter={res.n_iter:>3}  "
                          f"obj={res.obj:.6e}  "
                          f"{'OK' if res.solved else 'NOT_SOLVED'} "
                          f"({res.status})")
                    results.append(dict(
                        m=m, n=n, m_n_ratio=ratio, N=data.n + 3 * m,
                        solver_name=name,
                        device=res.device,
                        finite=True,
                        mean_ms=t_mean, std_ms=t_std, median_ms=t_median,
                        setup_ms=res.setup_time_ms,
                        all_times=res.solve_times_all,
                        iters=res.n_iter, obj=res.obj,
                        solved=res.solved, status=res.status,
                    ))
                except Exception as e:
                    print(f"  {name:>15}  [{cls.device}]  FAILED: "
                          f"{type(e).__name__}: {e}")
                    results.append(dict(
                        m=m, n=n, m_n_ratio=ratio, N=data.n + 3 * m,
                        solver_name=name, device=cls.device,
                        finite=False,
                        mean_ms=float('nan'), std_ms=float('nan'),
                        median_ms=float('nan'), setup_ms=float('nan'),
                        all_times=[], iters=-1, obj=float('nan'),
                        solved=False, status=f'FAILED: {type(e).__name__}',
                    ))

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
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {json_path}")

    # ---- Plot ----
    plot_path = args.out_plot or os.path.join(current_dir, 'benchmark_huber.png')
    try:
        plot_results(results, plot_path)
    except ImportError:
        print("matplotlib not installed -- skipping plot.")

    if log_fh is not None:
        print(f"Log saved to {log_path}")
        sys.stdout = sys.__stdout__
        log_fh.close()


if __name__ == '__main__':
    main()
