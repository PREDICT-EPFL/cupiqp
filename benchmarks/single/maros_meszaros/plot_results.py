"""Plot Maros-Meszaros benchmark results from a JSON produced by
``benchmark_maros.py``.

Default plot: per-problem grouped bar chart of mean solve-time (log y-axis),
with one bar per solver per problem. FAILED entries are shown as a small
hatched marker at the bottom of the panel.

Usage:
    python plot_results.py
    python plot_results.py --in results/benchmark_maros.json \\
                           --out results/benchmark_maros.pdf
"""

import argparse
import json
import os
import shutil

import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(current_dir, 'results')

_ALPHA = 0.8

# Per-solver visual style: (color, marker, linewidth). Hollow for CPU,
# filled for GPU (rendered downstream).
_SOLVER_STYLE = {
    # cuPIQP family
    'cupiqp-sparse':     ('#d62728', 'D', 1.5),
    'cupiqp-dense':      ('#1f77b4', 'd', 1.5),
    'cupiqp-multistage': ('#9467bd', '*', 1.8),
    # CPU references
    'osqp':              ('#7f7f7f', 'o', 1.1),
    'piqp-sparse':       ('#2ca02c', 's', 1.1),
    'clarabel':          ('#8c564b', 'p', 1.1),
    'hpipm':             ('#e377c2', 'h', 1.1),
    'qpalm':             ('#17a2b8', '8', 1.1),
    'cyqlone':           ('#5b3e96', '*', 1.1),
    # Other GPU
    'cuclarabel':        ('#ff7f0e', 'v', 1.1),
    'cuopt':             ('#17becf', '^', 1.1),
    'qoco-gpu':          ('#bcbd22', '<', 1.1),
    # ML extras
    'moreau-torch':      ('#aec7e8', 'P', 1.1),
    'moreau-jax':        ('#c5b0d5', 'X', 1.1),
    'qpax':              ('#c49c94', '>', 1.1),
    'qpth':              ('#f7b6d2', 'H', 1.1),
}
_DEFAULT_STYLE = ('#999999', 'x', 1.0)

_LEGEND_ORDER = [
    'cupiqp-sparse', 'cupiqp-dense',
    'piqp-sparse', 'clarabel', 'qpalm', 'osqp',
    'cuclarabel', 'cuopt',
]


def _style_for(name: str):
    return _SOLVER_STYLE.get(name, _DEFAULT_STYLE)


def _legend_rank(name: str) -> int:
    try:
        return _LEGEND_ORDER.index(name)
    except ValueError:
        return len(_LEGEND_ORDER)


def latexify(font_size: int = 15):
    import matplotlib as mpl
    has_latex = shutil.which('pdflatex') is not None
    mpl.rcParams.update({
        'text.usetex':       has_latex,
        'font.family':       'serif',
        'font.serif':        ['Computer Modern Roman'],
        'font.size':          font_size,
        'axes.labelsize':     font_size,
        'axes.titlesize':     font_size + 1,
        'xtick.labelsize':    font_size - 2,
        'ytick.labelsize':    font_size - 1,
        'legend.fontsize':    font_size - 2,
        'legend.frameon':     True,
        'legend.framealpha':  0.9,
        'lines.linewidth':    1.2,
        'axes.linewidth':     0.7,
        'xtick.major.width':  0.7,
        'ytick.major.width':  0.7,
        'pdf.fonttype':       42,
        'ps.fonttype':        42,
    })
    if has_latex:
        mpl.rcParams['text.latex.preamble'] = (
            r'\usepackage{amsmath}\usepackage{amssymb}')


def _is_successfully_solved(r: dict) -> bool:
    """True iff the row represents a successful, fully-converged solve.

    'Almost solved' / 'solved_inaccurate' / 'max_iter' are all treated as
    UNSUCCESSFUL — the solver returned a number but the answer is not at
    the requested tolerance, so the wall-clock isn't a fair comparison
    against the converged solvers on the same problem.
    """
    if r is None or not r.get('finite', False):
        return False
    if not r.get('solved', False):
        return False
    status = str(r.get('status', '')).lower()
    bad_markers = ('almost', 'inaccurate', 'max_iter', 'maximum iteration',
                   'not_solved')
    return not any(m in status for m in bad_markers)


def _grouped_bar(results, *, value_fn, ylabel, title, output_path,
                 log_y=True, value_at_failed=None):
    """Shared grouped-bar plotter.

    ``value_fn(row)`` returns the y-value for a row (or ``np.nan`` to mark
    a missing value). ``value_at_failed`` is the y at which to draw the
    'x' marker for FAILED cells (default = panel bottom in log mode, 0 in
    linear mode).

    Bars for runs that completed but did NOT successfully solve the
    problem (max_iter, almost_solved, primal_infeasible-but-actually-
    feasible, ...) are rendered with reduced alpha and a dense
    cross-hatch overlay so they are visually distinct from clean wins.
    """
    import matplotlib.pyplot as plt

    latexify()

    by_key = {}
    for r in results:
        by_key[(r['problem'], r['solver_name'])] = r

    problems = sorted({r['problem'] for r in results},
                      key=lambda p: next(
                          (r['n'] for r in results if r['problem'] == p), 0))
    solvers = sorted({r['solver_name'] for r in results}, key=_legend_rank)

    n_problems = len(problems)
    n_solvers = len(solvers)
    if n_problems == 0 or n_solvers == 0:
        print("No data to plot."); return

    # Wider figure + narrower bar group (0.6 of a unit instead of 0.8)
    # leaves more breathing room between adjacent problems. Within each
    # group, ``intra_gap`` controls the visible space between adjacent
    # solver bars — each bar is shrunk to ``slot_w * (1 - intra_gap)``
    # while staying centred in its own slot, so bars no longer touch
    # but the group's overall footprint is unchanged.
    fig, ax = plt.subplots(figsize=(max(10, n_problems * 0.7), 5))
    group_w = 0.6
    slot_w = group_w / n_solvers
    intra_gap = 0.18
    bar_w = slot_w * (1.0 - intra_gap)
    x_base = np.arange(n_problems)

    unsolved_seen = False
    for j, solver in enumerate(solvers):
        style = _style_for(solver)
        color = style[0]
        any_row = next((by_key[(p, solver)] for p in problems
                        if (p, solver) in by_key), None)
        device = any_row.get('device', 'cpu') if any_row else 'cpu'
        base_hatch = '///' if device == 'cpu' else None
        facecolor = 'white' if device == 'cpu' else color

        heights = [value_fn(by_key.get((p, solver))) for p in problems]
        xs = x_base + (j - (n_solvers - 1) / 2) * slot_w
        bars = ax.bar(xs, heights, width=bar_w,
                      color=facecolor, edgecolor=color,
                      linewidth=1.0, hatch=base_hatch, alpha=_ALPHA,
                      label=solver)

        # Per-cell: lower alpha and overlay a denser hatch on bars whose
        # underlying run didn't fully converge. Keep the edge colour at
        # full saturation so the bar's solver identity stays readable.
        for i, p in enumerate(problems):
            r = by_key.get((p, solver))
            if r is None or not r.get('finite', False):
                continue
            if not _is_successfully_solved(r):
                # bars[i].set_alpha(0.35)
                # Overlay a hatch-only rectangle so the unsolved cell is
                # visually distinct even when the underlying bar is short.
                # Matplotlib draws hatch lines in ``edgecolor``, so set
                # both ``edgecolor='white'`` and ``linewidth=0`` to get
                # white hatch without an extra outline — the bar's own
                # solver-coloured border (from the first ``ax.bar`` call)
                # still shows through underneath.
                h = heights[i]
                if h is not None and np.isfinite(h):
                    ax.bar(xs[i], h, width=bar_w,
                           facecolor='none', edgecolor='white',
                           linewidth=0, hatch='///', alpha=_ALPHA, zorder=3)
                    unsolved_seen = True

        # Mark FAILED entries with a small 'x'.
        for i, p in enumerate(problems):
            r = by_key.get((p, solver))
            if r is None or r.get('finite', False):
                continue
            y_mark = (value_at_failed if value_at_failed is not None
                      else (1.0 if log_y else 0.0))
            ax.plot(xs[i], y_mark, marker='x', markersize=4,
                    color=color, markeredgewidth=1.0)

    ax.set_xticks(x_base)
    ax.set_xticklabels(problems, rotation=45, ha='right')
    if log_y:
        ax.set_yscale('log')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis='y', which='both', alpha=0.3, linewidth=0.5)

    # Build legend: solver entries first, then an explanatory swatch for
    # the unsolved-bar overlay if any unsolved cells are present. The
    # swatch shows white hatch on a faded grey fill — matches what the
    # actual unsolved bars look like (faded solver colour + white hatch
    # overlay).
    legend_handles, legend_labels = ax.get_legend_handles_labels()
    if unsolved_seen:
        from matplotlib.patches import Patch
        legend_handles.append(Patch(facecolor='lightgray', edgecolor='white',
                                    linewidth=0, hatch='///', alpha=_ALPHA,
                                    label='not solved'))
        legend_labels.append('not solved')
    ax.legend(legend_handles, legend_labels, loc='best',
              ncol=min(len(legend_handles), 4))

    fig.tight_layout()
    fig.savefig(output_path)
    print(f"Plot saved to {output_path}")
    plt.close(fig)


def plot_results(results, output_path):
    """Grouped bar chart: x = problem, y = solve_time (log), bar = solver."""
    def _solve_time(r):
        if r is None or not r.get('finite', False):
            return np.nan
        return r['mean_ms']
    _grouped_bar(
        results, value_fn=_solve_time,
        ylabel=r'Solve time (ms)',
        title=r'Maros-Meszaros large QPs ($n > 5000$) — solve time',
        output_path=output_path, log_y=True,
    )


def plot_iters(results, output_path):
    """Grouped bar chart: x = problem, y = iter count (log), bar = solver.

    Iteration count is the most direct measure of algorithmic work that is
    invariant to GPU speed / driver / kernel-launch overhead. An IPM solver
    converging in 10 iters vs an ADMM solver hitting 250 max_iter is a
    cleaner story than wall-clock alone.
    """
    def _iters(r):
        if r is None or not r.get('finite', False):
            return np.nan
        it = r.get('iters', -1)
        # Treat -1 ("solver doesn't report iter") as missing.
        return np.nan if it is None or int(it) < 0 else int(it)
    _grouped_bar(
        results, value_fn=_iters,
        ylabel=r'Iterations',
        title=r'Maros-Meszaros large QPs ($n > 5000$) — iterations',
        output_path=output_path, log_y=False,
    )


def main():
    parser = argparse.ArgumentParser(
        description='Plot Maros-Meszaros benchmark results.')
    parser.add_argument(
        '--in', dest='input_json', type=str,
        default=os.path.join(results_dir, 'benchmark_maros.json'),
        help='Path to the JSON produced by benchmark_maros.py.')
    parser.add_argument(
        '--out', dest='output_pdf', type=str,
        default=os.path.join(results_dir, 'benchmark_maros.pdf'),
        help='Output PDF path for the solve-time plot. The iterations plot '
             'is saved alongside it with ``_iters`` appended before the '
             'extension.')
    args = parser.parse_args()

    with open(args.input_json) as f:
        results = json.load(f)
    os.makedirs(os.path.dirname(args.output_pdf), exist_ok=True)

    plot_results(results, args.output_pdf)

    # Companion iterations plot: ``foo.pdf`` -> ``foo_iters.pdf``.
    root, ext = os.path.splitext(args.output_pdf)
    iter_path = f'{root}_iters{ext}'
    plot_iters(results, iter_path)


if __name__ == '__main__':
    main()
