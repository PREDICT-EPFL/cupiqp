"""Benchmark cuPIQP (multistage & sparse), cuOpt, and cuClarabel on Chain Mass OCP.

Sweeps over num_masses and horizon, benchmarks solver backends,
prints a summary table, saves results to JSON, and plots solve time vs. horizon.

cuOpt and cuClarabel run as separate processes to avoid CUDA context conflicts.

Usage:
    python benchmark_chain_mass.py
    python benchmark_chain_mass.py --num_masses 10 20 --horizon 200 400 600
    python benchmark_chain_mass.py --n_runs 10 --verbose
    python benchmark_chain_mass.py --no_cuopt --no_cuclarabel
"""

import sys
import os
import json
import argparse
import subprocess
import tempfile
import shutil

import numpy as np
import scipy.sparse as sp
import cupy as cp
from cupyx.scipy.sparse import csr_matrix as gpu_csr_matrix

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from examples.chain_mass_ocp.chain_mass_ocp_problem import ChainMassOCPProblem
from cupiqp.solver import SolverBase

CUOPT_SCRIPT = os.path.join(current_dir, 'bench_cuopt.py')
CUCLARABEL_SCRIPT = os.path.join(current_dir, 'bench_cuclarabel.jl')


# ---------------------------------------------------------------------------
# cuPIQP benchmark helper
# ---------------------------------------------------------------------------

def bench_cupiqp(ocp, kkt_solver, n_runs=5, verbose=False):
    """Run cuPIQP benchmark and return timing results.

    Returns dict with keys: times, status, iters, obj
    """
    solver = SolverBase()
    solver.settings.kkt_solver = kkt_solver
    solver.settings.verbose = verbose
    solver.settings.max_iter = 250

    with cp.cuda.Device(0):
        if kkt_solver == 'multistage_block_cholesky':
            solver.settings.multistage_block_size = ocp.ms_block_size
            solver.setup(
                P=ocp.ms_P, c=ocp.ms_c,
                A=ocp.ms_A, b=ocp.ms_b,
                G=ocp.ms_G, h_u=ocp.ms_h_u,
                h_l=ocp.ms_h_l, x_u=ocp.ms_x_u,
                x_l=ocp.ms_x_l,
            )
        else:  # sparse_ldlt
            solver.setup(
                P=gpu_csr_matrix(ocp.P), c=cp.array(ocp.c),
                A=gpu_csr_matrix(ocp.Aeq), b=cp.array(ocp.beq),
                G=gpu_csr_matrix(ocp.Aineq),
                h_u=cp.array(ocp.bineq_ub), h_l=cp.array(ocp.bineq_lb),
                x_u=cp.array(ocp.xub), x_l=cp.array(ocp.xlb),
            )

        # Warmup
        solver.solve()

        times = []
        for _ in range(n_runs):
            es, ee = cp.cuda.Event(), cp.cuda.Event()
            es.record()
            solver.solve()
            ee.record()
            ee.synchronize()
            times.append(cp.cuda.get_elapsed_time(es, ee))

    return dict(
        times=times,
        status=str(solver._result.info.status),
        iters=int(solver._result.info.iter),
        obj=float(solver._result.info.primal_obj[0]),
    )


# ---------------------------------------------------------------------------
# cuOpt benchmark (subprocess)
# ---------------------------------------------------------------------------

def bench_cuopt(ocp, verbose=False, n_runs=5):
    """Benchmark cuOpt by calling bench_cuopt.py as a subprocess."""
    P_csr = sp.csr_matrix(ocp.P)
    Aeq_csr = sp.csr_matrix(ocp.Aeq)
    Aineq_csr = sp.csr_matrix(ocp.Aineq)

    problem_file = tempfile.mktemp(suffix='.npz')
    result_file = problem_file.replace('.npz', '_result.npz')

    np.savez(problem_file,
             P_indptr=P_csr.indptr, P_indices=P_csr.indices,
             P_data=P_csr.data, P_shape=np.array(P_csr.shape),
             c=ocp.c, xlb=ocp.xlb, xub=ocp.xub,
             Aeq_indptr=Aeq_csr.indptr, Aeq_indices=Aeq_csr.indices,
             Aeq_data=Aeq_csr.data, Aeq_shape=np.array(Aeq_csr.shape),
             beq=ocp.beq, n_eq=Aeq_csr.shape[0],
             Aineq_indptr=Aineq_csr.indptr, Aineq_indices=Aineq_csr.indices,
             Aineq_data=Aineq_csr.data, Aineq_shape=np.array(Aineq_csr.shape),
             bineq_lb=ocp.bineq_lb, bineq_ub=ocp.bineq_ub,
             n_ineq=Aineq_csr.shape[0],
             n_runs=n_runs, verbose=int(verbose))
    try:
        cmd = [sys.executable, CUOPT_SCRIPT, problem_file, result_file]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if verbose and proc.stdout:
            print(proc.stdout)

        if proc.returncode != 0:
            raise RuntimeError(
                f"bench_cuopt.py exited with code {proc.returncode}:\n"
                f"{proc.stderr[-2000:]}")

        result = np.load(result_file, allow_pickle=True)
        return dict(
            times=result['times'].tolist(),
            obj=float(result['obj'][0]),
            iters=int(result['iters'][0]) if 'iters' in result else -1,
            status=str(result['status'][0]),
        )
    finally:
        for f in [problem_file, result_file]:
            if os.path.exists(f):
                os.unlink(f)


# ---------------------------------------------------------------------------
# cuClarabel benchmark (Julia subprocess)
# ---------------------------------------------------------------------------

def piqp_to_clarabel(P, c, Aeq, beq, Aineq, bineq_lb, bineq_ub, xlb, xub):
    """Convert PIQP QP formulation to Clarabel conic form.

    Returns (P_triu_csc, q, A_csc, b, n_eq, n_ineq).
    """
    n = P.shape[0]
    A_rows, b_parts = [], []
    n_eq, n_ineq = 0, 0

    if Aeq.shape[0] > 0:
        A_rows.append(Aeq)
        b_parts.append(beq)
        n_eq = Aeq.shape[0]

    if Aineq.shape[0] > 0:
        ub_mask = np.isfinite(bineq_ub)
        if ub_mask.any():
            A_rows.append(Aineq[ub_mask])
            b_parts.append(bineq_ub[ub_mask])
            n_ineq += int(ub_mask.sum())
        lb_mask = np.isfinite(bineq_lb)
        if lb_mask.any():
            A_rows.append(-Aineq[lb_mask])
            b_parts.append(-bineq_lb[lb_mask])
            n_ineq += int(lb_mask.sum())

    xub_mask = np.isfinite(xub)
    if xub_mask.any():
        A_rows.append(sp.eye(n, format='csc')[xub_mask])
        b_parts.append(xub[xub_mask])
        n_ineq += int(xub_mask.sum())

    xlb_mask = np.isfinite(xlb)
    if xlb_mask.any():
        A_rows.append(-sp.eye(n, format='csc')[xlb_mask])
        b_parts.append(-xlb[xlb_mask])
        n_ineq += int(xlb_mask.sum())

    A_cl = sp.vstack(A_rows, format='csc')
    b_cl = np.concatenate(b_parts)
    P_cl = sp.triu(P, format='csc')

    return P_cl, c.copy(), A_cl, b_cl, n_eq, n_ineq


def find_julia():
    """Find Julia executable."""
    julia = shutil.which('julia')
    if julia:
        return julia
    for candidate in [
        os.path.expanduser('~/.julia/juliaup/bin/julia'),
        os.path.expanduser('~/.juliaup/bin/julia'),
    ]:
        if os.path.isfile(candidate):
            return candidate
    return None


def find_julia_project():
    """Find Julia project with Clarabel + CUDA packages."""
    conda_prefix = os.environ.get('CONDA_PREFIX', '')
    if conda_prefix:
        candidate = os.path.join(conda_prefix, 'julia_env')
        if os.path.isdir(candidate):
            return candidate
    # Try Python prefix (works even without conda activate)
    candidate = os.path.join(sys.prefix, 'julia_env')
    if os.path.isdir(candidate):
        return candidate
    try:
        import site
        for sp_dir in site.getsitepackages():
            candidate = os.path.join(os.path.dirname(sp_dir), 'julia_env')
            if os.path.isdir(candidate):
                return candidate
    except Exception:
        pass
    return None


def bench_cuclarabel(ocp, verbose=False, n_runs=5, julia_exe=None,
                     julia_project=None):
    """Benchmark cuClarabel by calling bench_cuclarabel.jl as a subprocess."""
    P_cl, q_cl, A_cl, b_cl, n_eq, n_ineq = piqp_to_clarabel(
        ocp.P, ocp.c, ocp.Aeq, ocp.beq,
        ocp.Aineq, ocp.bineq_lb, ocp.bineq_ub,
        ocp.xlb, ocp.xub,
    )
    P_csc = P_cl.tocsc()
    A_csc = A_cl.tocsc()

    # Save problem as .npz (1-indexed for Julia SparseMatrixCSC)
    problem_file = tempfile.mktemp(suffix='.npz')
    result_file = problem_file.replace('.npz', '_result.npz')

    np.savez(problem_file,
             P_colptr=(P_csc.indptr + 1).astype(np.int32),
             P_rowval=(P_csc.indices + 1).astype(np.int32),
             P_nzval=np.asarray(P_csc.data, dtype=np.float64),
             P_m=P_csc.shape[0], P_n=P_csc.shape[1],
             A_colptr=(A_csc.indptr + 1).astype(np.int32),
             A_rowval=(A_csc.indices + 1).astype(np.int32),
             A_nzval=np.asarray(A_csc.data, dtype=np.float64),
             A_m=A_csc.shape[0], A_n=A_csc.shape[1],
             q=np.asarray(q_cl, dtype=np.float64),
             b=np.asarray(b_cl, dtype=np.float64),
             n_eq=n_eq, n_ineq=n_ineq,
             n_runs=n_runs, verbose=int(verbose))
    try:
        cmd = [julia_exe, f'--project={julia_project}',
               CUCLARABEL_SCRIPT, problem_file, result_file]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if verbose and proc.stdout:
            print(proc.stdout)

        if proc.returncode != 0:
            raise RuntimeError(
                f"Julia exited with code {proc.returncode}:\n"
                f"{proc.stderr[-2000:]}")

        result = np.load(result_file, allow_pickle=True)
        status_file = result_file.replace('_result.npz', '_status.txt')
        status = 'UNKNOWN'
        if os.path.exists(status_file):
            with open(status_file) as sf:
                status = sf.read().strip()
        return dict(
            times=result['times'].tolist(),
            obj=float(result['obj'][0]),
            iters=int(result['iters'][0]),
            status=status,
        )
    finally:
        for f in [problem_file, result_file,
                  result_file.replace('_result.npz', '_status.txt')]:
            if os.path.exists(f):
                os.unlink(f)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(results, output_path):
    """Plot solve time vs. horizon with one line per (M, solver).

    Colors differentiate solvers, linestyles differentiate num_masses.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    # Organize data by (M, backend)
    series = {}
    for r in results:
        key = (r['M'], r['backend'])
        series.setdefault(key, {'N': [], 'mean': [], 'std': []})
        series[key]['N'].append(r['N'])
        series[key]['mean'].append(r['mean_ms'])
        series[key]['std'].append(r['std_ms'])

    # Color per solver (backend), linestyle/marker per M
    solver_color = {
        'multistage_block_cholesky': '#1f77b4',  # blue
        'sparse_ldlt':               '#ff7f0e',  # orange
        'cuopt':                     '#2ca02c',  # green
        'cuclarabel':                '#d62728',  # red
    }
    solver_label = {
        'multistage_block_cholesky': 'cuPIQP (multistage)',
        'sparse_ldlt':               'cuPIQP (sparse)',
        'cuopt':                     'cuOpt',
        'cuclarabel':                'cuClarabel',
    }

    masses = sorted(set(r['M'] for r in results))
    linestyles = ['-', '--', '-.', ':']
    markers = ['o', 's', 'D', '^']
    mass_style = {m: (linestyles[i % len(linestyles)],
                      markers[i % len(markers)])
                  for i, m in enumerate(masses)}

    for (M, backend), data in sorted(series.items()):
        ls, marker = mass_style[M]
        color = solver_color.get(backend, '#7f7f7f')
        label = solver_label.get(backend, backend)
        order = np.argsort(data['N'])
        ns = np.array(data['N'])[order]
        means = np.array(data['mean'])[order]
        stds = np.array(data['std'])[order]
        ax.plot(ns, means,
                linestyle=ls, marker=marker, color=color,
                linewidth=2, markersize=8,
                label=f'{label}, M={M}')
        ax.fill_between(ns, means - stds, means + stds,
                         color=color, alpha=0.15)

    ax.set_xlabel('Horizon (N)', fontsize=13)
    ax.set_ylabel('Solve Time (ms)', fontsize=13)
    ax.set_title('Chain Mass OCP Benchmark', fontsize=15)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"\nPlot saved to {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Benchmark cuPIQP and cuOpt on Chain Mass OCP')
    parser.add_argument('--num_masses', type=int, nargs='+', default=[20],
                        help='List of num_masses to sweep')
    parser.add_argument('--horizon', type=int, nargs='+',
                        default=[*range(100, 800 + 1, 100)],
                        help='List of horizons to sweep')
    parser.add_argument('--n_runs', type=int, default=5,
                        help='Number of timed runs per configuration')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable solver verbose output')
    parser.add_argument('--no_cuopt', action='store_true',
                        help='Skip cuOpt benchmark')
    parser.add_argument('--no_cuclarabel', action='store_true',
                        help='Skip cuClarabel benchmark')
    args = parser.parse_args()

    cupiqp_backends = ['multistage_block_cholesky', 'sparse_ldlt']
    results = []

    # Detect Julia / cuClarabel availability
    julia_exe = find_julia() if not args.no_cuclarabel else None
    julia_project = find_julia_project() if not args.no_cuclarabel else None
    cuclarabel_ok = (julia_exe is not None and julia_project is not None
                     and os.path.isfile(CUCLARABEL_SCRIPT))
    if not args.no_cuclarabel and not cuclarabel_ok:
        print("WARNING: cuClarabel not available "
              "(Julia or julia_env or bench_cuclarabel.jl not found).")
        print("  Will skip cuClarabel benchmarks.\n")

    for M in args.num_masses:
        for N in args.horizon:
            nx, nu = 2 * M, M - 1
            dim = N * (nx + nu) + nx

            print(f"\n{'=' * 70}")
            print(f"M={M}, N={N}  (n={dim}, n_eq={N * nx}, "
                  f"n_ineq={(N - 1) * nu})")
            print('=' * 70)

            ocp = ChainMassOCPProblem(
                M=M, N=N, randomize_x0=False,
                use_u_diff_cost=True, use_u_diff_constr=True,
            )

            # ---- cuPIQP (both backends) ----
            for backend in cupiqp_backends:
                label = 'ms' if backend == 'multistage_block_cholesky' else 'sp'
                try:
                    r = bench_cupiqp(ocp, backend, args.n_runs, args.verbose)
                    t_mean = float(np.mean(r['times']))
                    t_std = float(np.std(r['times']))
                    t_median = float(np.median(r['times']))
                    print(f"  cuPIQP({label}): {t_mean:8.2f} ± {t_std:.2f} ms  "
                          f"(iters={r['iters']}, obj={r['obj']:.6e}, "
                          f"{r['status']})")
                    results.append(dict(
                        M=M, N=N, dim=dim, backend=backend,
                        mean_ms=t_mean, std_ms=t_std, median_ms=t_median,
                        all_times=r['times'],
                        iters=r['iters'], obj=r['obj'], status=r['status'],
                    ))
                except Exception as e:
                    print(f"  cuPIQP({label}): FAILED -- {e}")
                    results.append(dict(
                        M=M, N=N, dim=dim, backend=backend,
                        mean_ms=float('nan'), std_ms=float('nan'),
                        median_ms=float('nan'),
                        all_times=[], iters=-1, obj=float('nan'),
                        status='FAILED',
                    ))

            # ---- cuOpt ----
            if not args.no_cuopt:
                try:
                    r_cuopt = bench_cuopt(ocp, args.verbose, args.n_runs)
                    t_mean = float(np.mean(r_cuopt['times']))
                    t_std = float(np.std(r_cuopt['times']))
                    t_median = float(np.median(r_cuopt['times']))
                    print(f"  cuOpt:      {t_mean:8.2f} ± {t_std:.2f} ms  "
                          f"(iters={r_cuopt['iters']}, obj={r_cuopt['obj']:.6e}, "
                          f"{r_cuopt['status']})")
                    results.append(dict(
                        M=M, N=N, dim=dim, backend='cuopt',
                        mean_ms=t_mean, std_ms=t_std, median_ms=t_median,
                        all_times=r_cuopt['times'],
                        iters=r_cuopt['iters'], obj=r_cuopt['obj'],
                        status=r_cuopt['status'],
                    ))
                except Exception as e:
                    print(f"  cuOpt:      FAILED -- {e}")
                    results.append(dict(
                        M=M, N=N, dim=dim, backend='cuopt',
                        mean_ms=float('nan'), std_ms=float('nan'),
                        median_ms=float('nan'),
                        all_times=[], iters=-1, obj=float('nan'),
                        status='FAILED',
                    ))

            # ---- cuClarabel ----
            if cuclarabel_ok:
                try:
                    r_clar = bench_cuclarabel(
                        ocp, args.verbose, args.n_runs, julia_exe, julia_project)
                    t_mean = float(np.mean(r_clar['times']))
                    t_std = float(np.std(r_clar['times']))
                    t_median = float(np.median(r_clar['times']))
                    print(f"  cuClarabel: {t_mean:8.2f} ± {t_std:.2f} ms  "
                          f"(iters={r_clar['iters']}, obj={r_clar['obj']:.6e}, "
                          f"{r_clar['status']})")
                    results.append(dict(
                        M=M, N=N, dim=dim, backend='cuclarabel',
                        mean_ms=t_mean, std_ms=t_std, median_ms=t_median,
                        all_times=r_clar['times'],
                        iters=r_clar['iters'], obj=r_clar['obj'],
                        status=r_clar['status'],
                    ))
                except Exception as e:
                    print(f"  cuClarabel: FAILED -- {e}")
                    results.append(dict(
                        M=M, N=N, dim=dim, backend='cuclarabel',
                        mean_ms=float('nan'), std_ms=float('nan'),
                        median_ms=float('nan'),
                        all_times=[], iters=-1, obj=float('nan'),
                        status='FAILED',
                    ))

    # ---- Summary table ----
    all_backends = sorted(set(r['backend'] for r in results))
    backend_short = {
        'multistage_block_cholesky': 'ms',
        'sparse_ldlt': 'sp',
        'cuopt': 'cuopt',
        'cuclarabel': 'clar',
    }

    print(f"\n{'=' * 130}")
    print(f"Summary  (mean ± std of {args.n_runs} runs)")
    print(f"{'=' * 130}")

    # Build header dynamically
    hdr_parts = [f"{'M':>4} {'N':>5} {'dim':>8}"]
    for b in all_backends:
        short = backend_short.get(b, b)
        hdr_parts.append(f"{short + ' (mean±std)':>18}")
    hdr_parts.append(f"{'ratio':>8}")
    for b in all_backends:
        short = backend_short.get(b, b)
        hdr_parts.append(f"{'it_' + short:>7}")
    for b in all_backends:
        short = backend_short.get(b, b)
        hdr_parts.append(f"{'obj_' + short:>12}")
    header = ' | '.join(hdr_parts)
    print(header)
    print('-' * len(header))

    # Group by (M, N) to print side-by-side
    from itertools import groupby
    for (M, N), group in groupby(results, key=lambda r: (r['M'], r['N'])):
        rows = {r['backend']: r for r in group}

        def fmt_t(r):
            mean = r.get('mean_ms', float('nan'))
            std = r.get('std_ms', float('nan'))
            if np.isnan(mean):
                return '               N/A'
            return f'{mean:7.2f}±{std:.2f}ms'

        def fmt_it(r):
            v = r.get('iters', -1)
            return f'{v:7d}' if v >= 0 else '    N/A'

        def fmt_obj(r):
            v = r.get('obj', float('nan'))
            return f'{v:.4e}' if not np.isnan(v) else '         N/A'

        # Ratio: sp/ms or cuopt/ms
        r_ms = rows.get('multistage_block_cholesky', {})
        t_ms_mean = r_ms.get('mean_ms', float('nan'))
        # Use the slowest alternative for ratio
        other_means = []
        for b in all_backends:
            if b != 'multistage_block_cholesky':
                t = rows.get(b, {}).get('mean_ms', float('nan'))
                if not np.isnan(t):
                    other_means.append(t)
        if other_means and not np.isnan(t_ms_mean) and t_ms_mean > 0:
            best_other = min(other_means)
            ratio_s = f'{best_other / t_ms_mean:.1f}x'
        else:
            ratio_s = 'N/A'

        dim = next((rows[b].get('dim', 0) for b in all_backends if b in rows), 0)
        parts = [f"{M:4d} {N:5d} {dim:8d}"]
        for b in all_backends:
            parts.append(f"{fmt_t(rows.get(b, {})):>18}")
        parts.append(f"{ratio_s:>8}")
        for b in all_backends:
            parts.append(f"{fmt_it(rows.get(b, {})):>7}")
        for b in all_backends:
            parts.append(f"{fmt_obj(rows.get(b, {})):>12}")
        print(' | '.join(parts))

    # ---- Save JSON ----
    json_path = os.path.join(current_dir, 'benchmark_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {json_path}")

    # ---- Plot ----
    plot_path = os.path.join(current_dir, 'benchmark_chain_mass.png')
    try:
        plot_results(results, plot_path)
    except ImportError:
        print("\nmatplotlib not installed -- skipping plot. "
              "Install with: pip install matplotlib")


if __name__ == '__main__':
    main()
