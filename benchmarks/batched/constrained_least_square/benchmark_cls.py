"""
Benchmark + plot for batched constrained least squares.

    min  ||Fx - g||^2
    s.t. Ax = b,  x >= 0

Runs cuPIQP (dense + sparse), qpax, qpth, moreau across batch sizes,
prints a per-B table with setup/solve times and failure counts,
then plots solve time and throughput with standard error bars.
"""
import argparse
import json
import sys
from pathlib import Path
_repo_root = str(Path(__file__).resolve().parent.parent.parent.parent)
_batched_dir = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _repo_root)
sys.path.insert(0, _batched_dir)

import numpy as np
import matplotlib.pyplot as plt
from batched_solver_interface import (
    BatchedQPData, CupiqpDenseBatchedSolver, CupiqpSparseBatchedSolver,
    QpaxBatchedSolver, QpthBatchedSolver, MoreauBatchedSolver,
)


def make_constr_least_square_data(B, n, rows_F, rows_A, seed=42):
    rng = np.random.default_rng(seed)
    F = rng.standard_normal((rows_F, n))
    A_eq = rng.standard_normal((rows_A, n))
    Fs = np.tile(F[None], (B, 1, 1))
    gs = rng.standard_normal((B, rows_F))
    As = np.tile(A_eq[None], (B, 1, 1))
    bs = (As @ np.ones((B, n, 1))).squeeze(-1) + 0.1 * rng.standard_normal((B, rows_A))
    Ps = 2 * np.einsum('bji,bjk->bik', Fs, Fs)
    Ps = (Ps + Ps.transpose(0, 2, 1)) / 2
    cs = -2 * np.einsum('bji,bj->bi', Fs, gs)
    xls = rng.standard_normal((B, n))
    return BatchedQPData(P=Ps, c=cs, A=As, b=bs, x_l=xls)


SOLVERS = [
    ("cupiqp-dense", CupiqpDenseBatchedSolver),
    ("cupiqp-sparse", CupiqpSparseBatchedSolver),
    ("qpax", QpaxBatchedSolver),
    ("qpth", QpthBatchedSolver),
    ("moreau", MoreauBatchedSolver),
]


_COL_WIDTH = 10 + 2 + 10 + 2 + 5 + 2 + 11  # setup + solve + iter + fail
_SEP = " | "


def print_header(solver_names):
    col = f"{'setup':>10s}  {'solve':>10s}  {'iter':>5s}  {'fail':>11s}"
    name_hdr = _SEP.join(f"{name:^{_COL_WIDTH}s}" for name in solver_names)
    sub_hdr = f"{'B':>6s}" + _SEP + _SEP.join([col] * len(solver_names))
    print(f"{'':>6s}" + _SEP + name_hdr)
    print(sub_hdr)
    print("-" * len(sub_hdr))


def fmt_result(r, B):
    if r is None:
        return f"{'N/A':>10s}  {'N/A':>10s}  {'N/A':>5s}  {'N/A':>11s}"
    fail = B - r.n_solved if r.n_solved >= 0 else -1
    n_iter = f"{r.n_iter_max:5d}" if r.n_iter_max >= 0 else f"{'-':>5s}"
    return f"{r.setup_time_ms:10.2f}  {r.solve_time_ms:10.2f}  {n_iter}  {fail:5d}/{B:<5d}"


def run_benchmark(n=20, row_F=40, row_A=5, batch_sizes=None, tol=1e-8,
                  max_iter=300, n_repeats=10):
    if batch_sizes is None:
        # batch_sizes = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, int(8192*2), int(8192*4), int(8192*8)]
        # batch_sizes = [2048, 4096, 8192, int(8192*2), int(8192*4), int(8192*8)]
        batch_sizes = [8, 16, 32]

    solver_names = [name for name, _ in SOLVERS]
    print(f"Constrained Least Squares: n={n}, row(F)={row_F}, row(A)={row_A}")
    print(f"  min ||Fx - g||^2  s.t.  Ax = b,  x >= x_l")
    print(f"  tol={tol}, max_iter={max_iter}, n_repeats={n_repeats}")
    print()
    print_header(solver_names)

    results = {name: dict(solve_median=[], solve_stderr=[], throughput=[],
                          throughput_stderr=[], n_solved=[], total=[])
               for name in solver_names}

    for B in batch_sizes:
        data = make_constr_least_square_data(B, n, row_F, row_A, seed=10)
        per_solver = {}
        for name, cls in SOLVERS:
            try:
                solver = cls(tol_abs=tol, max_iter=max_iter)
                r = solver.benchmark(data, n_repeats)
                per_solver[name] = r
                times = np.array(r.solve_times_all)
                throughputs = B / (times / 1000)
                results[name]["solve_median"].append(float(np.median(times)))
                results[name]["solve_stderr"].append(float(np.std(times) / np.sqrt(len(times))))
                results[name]["throughput"].append(float(np.median(throughputs)))
                results[name]["throughput_stderr"].append(float(np.std(throughputs) / np.sqrt(len(throughputs))))
                results[name]["n_solved"].append(r.n_solved)
                results[name]["total"].append(r.total)
            except Exception as e:
                print(f"\n  [{name}] skipped: {e}", flush=True)
                per_solver[name] = None
                for k in ("solve_median", "solve_stderr", "throughput", "throughput_stderr"):
                    results[name][k].append(float("nan"))
                results[name]["n_solved"].append(0)
                results[name]["total"].append(B)

        print(f"{B:6d}" + _SEP + _SEP.join(fmt_result(per_solver[name], B) for name in solver_names),
              flush=True)

    params = dict(n=n, rows_F=row_F, p=row_A, tol=tol, max_iter=max_iter)
    return results, batch_sizes, params


def plot_results(results, batch_sizes, params):
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    colors = {"cupiqp-dense": "#1f77b4", "cupiqp-sparse": "#9467bd",
              "qpax": "#ff7f0e", "qpth": "#2ca02c", "moreau": "#d62728"}
    markers = {"cupiqp-dense": "o", "cupiqp-sparse": "P",
               "qpax": "s", "qpth": "^", "moreau": "D"}

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    fig.suptitle(
        r"Batched Constrained Least Squares: "
        rf"$n\!=\!{params['n']}$, $\mathrm{{rows}}(F)\!=\!{params['rows_F']}$, "
        rf"$\mathrm{{rows}}(A)\!=\!{params['p']}$, $\epsilon\!=\!10^{{{int(np.log10(params['tol']))}}}$",
        fontsize=13,
    )

    B = np.array(batch_sizes)

    ax = axes[0]
    for name in results:
        median = np.array(results[name]["solve_median"])
        stderr = np.array(results[name]["solve_stderr"])
        if np.all(np.isnan(median)):
            continue
        ax.errorbar(B, median, yerr=stderr, color=colors[name], marker=markers[name],
                    linewidth=1.5, markersize=5, capsize=2, label=name)
    ax.set_xlabel(r"Batch size $B$")
    ax.set_ylabel(r"Solve time (ms)")
    ax.set_title(r"Solve Time")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(framealpha=0.9)
    ax.grid(True, which='major', alpha=0.5)
    ax.grid(True, which='minor', alpha=0.2)
    ax.yaxis.set_major_locator(plt.LogLocator(base=10, numticks=15))
    ax.yaxis.set_minor_locator(plt.LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=15))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}" if v >= 1 else f"{v:.1f}"))

    ax = axes[1]
    for name in results:
        tp = np.array(results[name]["throughput"])
        tp_se = np.array(results[name]["throughput_stderr"])
        if np.all(np.isnan(tp)):
            continue
        ax.errorbar(B, tp, yerr=tp_se, color=colors[name], marker=markers[name],
                    linewidth=1.5, markersize=5, capsize=2, label=name)
    ax.set_xlabel(r"Batch size $B$")
    ax.set_ylabel(r"Throughput (QP/s)")
    ax.set_title(r"Solve Throughput")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(framealpha=0.9)
    ax.grid(True, which="major", alpha=0.5)
    ax.grid(True, which="minor", alpha=0.2)

    plt.tight_layout()
    out_pdf = Path(__file__).resolve().parent / "benchmark_cls.pdf"
    out_png = Path(__file__).resolve().parent / "benchmark_cls.png"
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to {out_pdf}")
    print(f"Figure saved to {out_png}")
    plt.show()


def save_json(results, batch_sizes, params, path: Path):
    payload = {
        "params": params,
        "batch_sizes": list(batch_sizes),
        "results": results,
    }
    path.write_text(json.dumps(payload, indent=2))
    print(f"Results saved to {path}")


def load_json(path: Path):
    payload = json.loads(path.read_text())
    return payload["results"], payload["batch_sizes"], payload["params"]


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    default_json = here / "benchmark_cls.json"

    p = argparse.ArgumentParser()
    p.add_argument("--load", type=Path, default=None,
                   help="Load results from a JSON file and plot (skip benchmark).")
    p.add_argument("--save", type=Path, default=default_json,
                   help=f"Save results to this JSON file (default: {default_json}).")
    p.add_argument("--no-plot", action="store_true", help="Skip plotting.")
    p.add_argument("--tol", type=float, default=1e-8)
    p.add_argument("--n", type=int, default=20, help="Number of decision variables.")
    p.add_argument("--row-F", dest="row_F", type=int, default=40,
                   help="Number of rows of F (residual length in ||Fx - g||^2).")
    p.add_argument("--row-A", dest="row_A", type=int, default=5,
                   help="Number of equality constraints (rows of A).")
    p.add_argument("--batch-sizes", dest="batch_sizes", type=int, nargs="+", default=None,
                   help="Space-separated list of batch sizes (e.g. --batch-sizes 8 16 32).")
    p.add_argument("--max-iter", dest="max_iter", type=int, default=300)
    p.add_argument("--n-repeats", dest="n_repeats", type=int, default=10)
    args = p.parse_args()

    if args.load is not None:
        results, batch_sizes, params = load_json(args.load)
    else:
        results, batch_sizes, params = run_benchmark(
            n=args.n, row_F=args.row_F, row_A=args.row_A,
            batch_sizes=args.batch_sizes, tol=args.tol,
            max_iter=args.max_iter, n_repeats=args.n_repeats,
        )
        save_json(results, batch_sizes, params, args.save)

    if not args.no_plot:
        plot_results(results, batch_sizes, params)
