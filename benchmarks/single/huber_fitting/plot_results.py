"""Plot Huber-fitting benchmark results from a JSON produced by
``benchmark_huber.py``.

The JSON is a flat list of cell dicts with keys:
  m, n, m_n_ratio, N, solver_name, device, finite, mean_ms, std_ms, ...

Usage:
    python plot_results.py
    python plot_results.py --in results/benchmark_huber.json \\
                           --out results/benchmark_huber.pdf
"""

import argparse
import json
import os
import shutil

import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(current_dir, 'results')


_SOLVER_STYLE = {
    'cupiqp-sparse':     ('#d62728', 'D', 1.5),
    'cupiqp-dense':      ('#1f77b4', 'd', 1.5),
    'cupiqp-multistage': ('#9467bd', '*', 1.5),
    'osqp':              ('#7f7f7f', 'o', 1.1),
    'piqp-sparse':       ('#2ca02c', 's', 1.1),
    'clarabel':          ('#8c564b', 'p', 1.1),
    'hpipm':             ('#e377c2', 'h', 1.1),
    'qpalm':             ('#17a2b8', '8', 1.1),
    'cyqlone':           ('#5b3e96', '*', 1.1),
    'gurobi':            ('#800000', 'P', 1.1),
    'cuclarabel':        ('#ff7f0e', 'v', 1.1),
    'cuopt':             ('#17becf', '^', 1.1),
    'qoco-gpu':          ('#bcbd22', '<', 1.1),
    'moreau-torch':      ('#aec7e8', 'P', 1.1),
    'moreau-jax':        ('#c5b0d5', 'X', 1.1),
    'qpax':              ('#c49c94', '>', 1.1),
    'qpth':              ('#f7b6d2', 'H', 1.1),
}
_DEFAULT_STYLE = ('#999999', 'x', 1.0)

_LEGEND_ORDER = [
    'cupiqp-sparse', 'cupiqp-dense',
    'piqp-sparse', 'clarabel', 'qpalm', 'gurobi', 'osqp',
    'cuclarabel', 'cuopt', 'qoco-gpu',
    'moreau-torch', 'moreau-jax', 'qpax', 'qpth',
]


def _style_for(name: str):
    return _SOLVER_STYLE.get(name, _DEFAULT_STYLE)


def _legend_rank(name: str) -> int:
    try:
        return _LEGEND_ORDER.index(name)
    except ValueError:
        return len(_LEGEND_ORDER)


def latexify(font_size: int = 11):
    import matplotlib as mpl
    has_latex = shutil.which('pdflatex') is not None
    mpl.rcParams.update({
        'text.usetex':       has_latex,
        'font.family':       'serif',
        'font.serif':        ['Computer Modern Roman'],
        'font.size':          font_size,
        'axes.labelsize':     font_size,
        'axes.titlesize':     font_size + 1,
        'xtick.labelsize':    font_size - 1,
        'ytick.labelsize':    font_size - 1,
        'legend.fontsize':    font_size - 2,
        'legend.frameon':     True,
        'legend.framealpha':  0.9,
        'lines.linewidth':    1.2,
        'lines.markersize':   5,
        'axes.linewidth':     0.7,
        'xtick.major.width':  0.7,
        'ytick.major.width':  0.7,
        'pdf.fonttype':       42,
        'ps.fonttype':        42,
    })
    if has_latex:
        mpl.rcParams['text.latex.preamble'] = (
            r'\usepackage{amsmath}\usepackage{amssymb}')


def plot_results(results, output_path):
    """Solve time vs. m (observations), one line per (m/n ratio, solver)."""
    import matplotlib.pyplot as plt

    latexify()

    fig, ax = plt.subplots(figsize=(7, 4.5))

    series = {}
    for r in results:
        if not r.get('finite', False):
            continue
        key = (r.get('m_n_ratio', 1.5), r['solver_name'])
        series.setdefault(
            key,
            {'m': [], 'mean': [], 'std': [], 'device': r.get('device', '?')},
        )
        series[key]['m'].append(r['m'])
        series[key]['mean'].append(r['mean_ms'])
        series[key]['std'].append(r['std_ms'])

    ratios = sorted({r.get('m_n_ratio', 1.5) for r in results
                     if r.get('finite', False)})
    cpu_styles = ['--', (0, (3, 1, 1, 1)), (0, (5, 2))]
    gpu_styles = ['-',  (0, (4, 1, 1, 1, 1, 1)), (0, (6, 2))]

    def _linestyle_for(device, ratio):
        i = ratios.index(ratio) if ratio in ratios else 0
        styles = cpu_styles if device == 'cpu' else gpu_styles
        return styles[i % len(styles)]

    def _draw_key(item):
        (ratio, name), _ = item
        return (_legend_rank(name), ratio) if name in _LEGEND_ORDER else (-1, ratio)

    plotted_handles = {}
    for (ratio, name), data in sorted(series.items(), key=_draw_key, reverse=True):
        ls = _linestyle_for(data['device'], ratio)
        color, marker, base_lw = _style_for(name)
        order  = np.argsort(data['m'])
        ms_vals = np.array(data['m'])[order]
        means  = np.array(data['mean'])[order]
        stds   = np.array(data['std'])[order]

        label = name if len(ratios) == 1 else f'{name}, $m/n={ratio:g}$'

        is_cpu = (data['device'] == 'cpu')
        mfc = 'white' if is_cpu else color
        mew = 1.2 if is_cpu else 0.8

        (line,) = ax.plot(
            ms_vals, means,
            linestyle=ls, marker=marker, color=color,
            markersize=6, markeredgewidth=mew,
            markerfacecolor=mfc, markeredgecolor=color,
            linewidth=base_lw,
            label=label,
        )
        if np.any(stds > 0):
            ax.fill_between(ms_vals, means - stds, means + stds,
                            color=color, alpha=0.12, linewidth=0)
        plotted_handles[(ratio, name)] = line

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$m$ (number of observations)')
    ax.set_ylabel(r'Solve time (ms)')
    ax.set_title(r'Huber Fitting QP Benchmark')

    sorted_keys = sorted(
        plotted_handles.keys(),
        key=lambda k: (_legend_rank(k[1]), k[0]),
    )
    ax.legend([plotted_handles[k] for k in sorted_keys],
              [plotted_handles[k].get_label() for k in sorted_keys],
              loc='best', ncol=1)
    ax.grid(True, which='both', alpha=0.3, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(output_path)
    print(f"Plot saved to {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Plot Huber-fitting benchmark.')
    parser.add_argument(
        '--in', dest='input_json', type=str,
        default=os.path.join(results_dir, 'benchmark_huber.json'),
        help='Path to the JSON produced by benchmark_huber.py.')
    parser.add_argument(
        '--out', dest='output_pdf', type=str,
        default=os.path.join(results_dir, 'benchmark_huber.pdf'),
        help='Output PDF path (matplotlib auto-detects extension).')
    parser.add_argument(
        '--m_n_ratio', type=float, nargs='+', default=None,
        help='Optional whitelist of m/n ratios to plot (e.g. 1.5). '
             'Default keeps all ratios present in the JSON.')
    args = parser.parse_args()

    with open(args.input_json) as f:
        results = json.load(f)

    if args.m_n_ratio is not None:
        wanted = set(args.m_n_ratio)
        results = [r for r in results
                   if any(abs(r.get('m_n_ratio', 0) - w) < 1e-9
                          for w in wanted)]
        print(f"Filtered to m/n in {sorted(wanted)}: {len(results)} rows.")

    os.makedirs(os.path.dirname(args.output_pdf), exist_ok=True)
    plot_results(results, args.output_pdf)


if __name__ == '__main__':
    main()
