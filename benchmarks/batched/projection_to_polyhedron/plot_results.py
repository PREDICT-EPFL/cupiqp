"""
Variant of ``plot_results.py`` that renders IPM iteration counts as
line+marker series instead of grouped bars. The solve-time top subplot
is unchanged.

Usage:
    python plot_results.py results/benchmark_projection.json
    python plot_results.py results/cfg01_n5_l25_B16-2048.json --no-show
"""
import argparse
import json
from pathlib import Path

import numpy as np


SOLVER_NAMES = [
    "cupiqp-dense",
    "cupiqp-sparse",
    "qpax",
    "moreau-torch",
    "qpth",
]

_PLOT_COLORS = {
    "cupiqp-dense":      "#1f77b4",
    "cupiqp-sparse":     "#d62728",
    "cupiqp-multistage": "#9467bd",   # not used in this benchmark
    "qpax":              "gray",
    "qpth":              "lightgray",
    "moreau-torch":      "darkgray",
    "moreau-jax":        "dimgray",
}

_ALPHA = 1.0
_PLOT_MARKERS = {
    "cupiqp-dense":  "o",
    "cupiqp-sparse": "P",
    "qpax":          "s",
    "qpth":          "^",
    "moreau-torch":  "D",
    "moreau-jax":    "*",
}


def plot_results(results, batch_sizes, params, out_dir: Path, prefix: str,
                 show: bool = True):
    """Two figures next to ``out_dir``:

    1. ``{prefix}_solve_time.pdf`` -- bars (solve time) on top,
       line+marker series (max iter count) on the bottom.
    2. ``{prefix}_throughput.pdf`` -- log-log throughput vs batch size
       (same as ``plot_results.py``).
    """
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 16,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "legend.fontsize": 14,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
    })

    n_ineq = params.get("n_inequalities", params.get("n_facets"))
    kappa = float(params.get("condition_number", 1.0))
    F = float(params.get("redundancy_frac", 0.0))
    extra = ""
    if kappa != 1.0:
        extra += rf", $\kappa\!=\!10^{{{int(np.log10(kappa))}}}$"
    if F > 0.0:
        extra += rf", $F\!=\!{F:g}$"
    title = (
        rf"Polyhedron Projection: $n\!=\!{params['n']}$, "
        rf"$\ell\!=\!{n_ineq}$ inequalities, "
        rf"$\epsilon\!=\!10^{{{int(np.log10(params['tol']))}}}$"
        + extra
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Figure 1 ------------------------------------------------------
    _ordered_keys = list(SOLVER_NAMES) + [k for k in results if k not in SOLVER_NAMES]
    active_solvers = [
        name for name in _ordered_keys
        if name in results
        and not np.all(np.isnan(np.array(results[name]["solve_median"])))
    ]
    n_solvers = max(len(active_solvers), 1)
    n_groups = len(batch_sizes)
    bar_width = 0.8 / n_solvers              # leave 0.2 inter-group gap
    group_centers = np.arange(n_groups, dtype=np.float64)

    # The IPM iteration cap (where the solver gives up). Read from the
    # JSON params block written by benchmark_projection.py; fall back to
    # 100 for legacy JSONs that didn't carry it.
    iter_cap = float(params.get("max_iter", 100))

    NORMAL_ITER_CAP = 20                       # top of the lower band
    BREAK_THRESHOLD = 30                       # break when any value > this

    all_iters = []
    for name in active_solvers:
        v = np.array(results[name].get("n_iter_max", []), dtype=np.float64)
        v = v[v >= 0]
        if v.size:
            all_iters.append(v)
    if all_iters:
        _flat = np.concatenate(all_iters)
        iter_min_data = float(_flat.min())
        iter_max_data = float(_flat.max())
    else:
        iter_min_data = 0.0
        iter_max_data = 0.0

    # Lower band starts just below the smallest observed iter value.
    iter_lo_y = max(0.0, iter_min_data - 2.0)

    # Break the iter axis when either:
    # - some solver hit a high iter count (original trigger -- keeps small
    #   markers like 9-12 legible alongside outliers like 200), or
    # - the cap sits far above all observed markers, so it has to be
    #   shown in a separate band rather than stretching the single axis.
    iter_break = (
        iter_max_data > BREAK_THRESHOLD
        or iter_cap > 2.0 * max(iter_max_data, float(NORMAL_ITER_CAP))
    )

    if iter_break:
        # Lower band tightly fits the "normal" cluster (anything <= NORMAL_ITER_CAP).
        iter_hi_y = float(NORMAL_ITER_CAP) * 1.1

        # Upper band fits outlier markers (> NORMAL_ITER_CAP) and the cap line.
        if iter_max_data > NORMAL_ITER_CAP:
            outliers = np.concatenate(all_iters)
            outliers = outliers[outliers > NORMAL_ITER_CAP]
            outlier_min = float(outliers.min())
            outlier_max = float(outliers.max())
            band_lo_data = min(outlier_min, iter_cap)
            band_hi_data = max(outlier_max, iter_cap)
            pad = max(50.0, 0.5 * (band_hi_data - band_lo_data))
            hi_bottom = 50.0 * np.floor((band_lo_data - pad) / 50.0)
            hi_top = 50.0 * (int(band_hi_data // 50) + 1)
            if hi_bottom <= NORMAL_ITER_CAP:
                hi_bottom = float(NORMAL_ITER_CAP) + 20.0
        else:
            # No outliers, just the cap line -- narrow band centered on it.
            band_pad = max(5.0, 0.03 * iter_cap)
            hi_bottom = iter_cap - band_pad
            hi_top = iter_cap + band_pad
    else:
        iter_hi_y = max(
            iter_max_data + max(2.0, 0.1 * max(iter_max_data - iter_min_data, 1.0)),
            iter_cap + max(2.0, 0.05 * iter_cap),
        )
        hi_top = hi_bottom = None

    fig_w = max(6.0, min(12.0, 1.2 * n_groups + 2.0)) + 2.2

    if iter_break:
        height_ratios = [1.4, 0.40, 0.60]
    else:
        height_ratios = [1.4, 1.0]
    fig_h = 6.5 + (0.4 if iter_break else 0.0)

    fig1 = plt.figure(figsize=(fig_w, fig_h))
    if iter_break:
        outer = fig1.add_gridspec(
            2, 1, height_ratios=[height_ratios[0], height_ratios[1] + height_ratios[2]],
            hspace=0.15,
        )
        ax_time = fig1.add_subplot(outer[0])
        inner = outer[1].subgridspec(
            2, 1,
            height_ratios=[height_ratios[1], height_ratios[2]],
            hspace=0.20,
        )
        ax_iter_hi = fig1.add_subplot(inner[0], sharex=ax_time)
        ax_iter_lo = fig1.add_subplot(inner[1], sharex=ax_iter_hi)
        iter_axes = (ax_iter_hi, ax_iter_lo)
        ax_iter_xlabel = ax_iter_lo
    else:
        outer = fig1.add_gridspec(
            2, 1, height_ratios=height_ratios, hspace=0.15,
        )
        ax_time = fig1.add_subplot(outer[0])
        ax_iter_lo = fig1.add_subplot(outer[1], sharex=ax_time)
        iter_axes = (ax_iter_lo,)
        ax_iter_xlabel = ax_iter_lo
    ax_time.set_xticks(group_centers)
    ax_time.set_xticklabels([str(B) for B in batch_sizes])
    ax_time.tick_params(labelbottom=True)

    _bar_kw = dict(edgecolor="white", linewidth=0.6, alpha=_ALPHA)
    _err_kw = dict(capsize=2, lw=0.8, ecolor="#444444")

    # ---- top subplot: solve time bars (unchanged) ---------------------
    for i, name in enumerate(active_solvers):
        median = np.array(results[name]["solve_median"], dtype=np.float64)
        stderr = np.array(results[name]["solve_stderr"], dtype=np.float64)
        offset = (i - (n_solvers - 1) / 2.0) * bar_width
        positions = group_centers + offset
        ax_time.bar(
            positions, median, bar_width,
            yerr=np.where(np.isfinite(stderr), np.abs(stderr), 0.0),
            color=_PLOT_COLORS.get(name, "k"),
            label=name,
            error_kw=_err_kw,
            **_bar_kw,
        )
    ax_time.set_ylabel(r"Solve time (ms)")
    ax_time.set_yscale("log")
    ax_time.legend(
        loc="upper left",
        frameon=False,
    )
    ax_time.grid(True, axis="y", which="major", alpha=0.5)
    ax_time.grid(True, axis="y", which="minor", alpha=0.2)
    ax_time.set_axisbelow(True)

    # ---- iter subplot(s): max-iter as markers only --------------------
    # One marker per (solver, B), aligned horizontally with the corresponding
    # solve-time bar so markers within a group don't stack on top of each
    # other. Solvers that don't expose a per-problem iter count (e.g. qpth,
    # n_iter_max == -1) are simply omitted -- they already appear in the
    # top legend.
    for i, name in enumerate(active_solvers):
        iters = np.array(
            results[name].get("n_iter_max", [-1] * n_groups),
            dtype=np.float64,
        )
        iters_plot = np.where(iters >= 0, iters, np.nan)
        if np.all(np.isnan(iters_plot)):
            continue                          # solver never exposes iter info
        offset = (i - (n_solvers - 1) / 2.0) * bar_width
        positions = group_centers + offset
        for ax in iter_axes:
            ax.scatter(
                positions, iters_plot,
                color=_PLOT_COLORS.get(name, "k"),
                marker=_PLOT_MARKERS.get(name, "o"),
                edgecolor="white",
                linewidth=0.6,
                s=100,
                alpha=_ALPHA,
                label=name,
            )

    # Bottom subplot owns the x ticks/label.
    ax_iter_xlabel.set_xticks(group_centers)
    ax_iter_xlabel.set_xticklabels([str(B) for B in batch_sizes])
    ax_iter_xlabel.set_xlabel(r"Batch size $B$")

    ax_iter_lo.set_ylim(iter_lo_y, iter_hi_y)
    ax_iter_lo.set_ylabel(r"Max iterations")
    ax_iter_lo.grid(True, axis="y", which="both", alpha=0.5)
    ax_iter_lo.set_axisbelow(True)

    # Dashed cap line at iter_cap -- drawn on whichever axis it falls in.
    _cap_line_kw = dict(linestyle="--", color="black", linewidth=0.9, alpha=0.9)
    if not iter_break:
        ax_iter_lo.axhline(iter_cap, **_cap_line_kw)

    if iter_break:
        ax_iter_hi.set_ylim(hi_bottom, hi_top)
        ax_iter_hi.axhline(iter_cap, **_cap_line_kw)
        ax_iter_hi.grid(True, axis="y", which="major", alpha=0.5)
        ax_iter_hi.set_axisbelow(True)
        ax_iter_hi.spines["bottom"].set_visible(False)
        ax_iter_lo.spines["top"].set_visible(False)
        ax_iter_hi.tick_params(labelbottom=False, bottom=False)
        d = 0.012
        kwargs = dict(transform=ax_iter_hi.transAxes, color="#444",
                      lw=1.0, clip_on=False)
        ax_iter_hi.plot((-d, +d), (-d, +d), **kwargs)
        kwargs.update(transform=ax_iter_lo.transAxes)
        ax_iter_lo.plot((-d, +d), (1 - d, 1 + d), **kwargs)

    for _ax in (ax_time, *iter_axes):
        _ax.spines["right"].set_visible(False)
        _ax.spines["top"].set_visible(False)
    if not iter_break:
        ax_iter_lo.spines["top"].set_visible(False)

    plt.tight_layout()
    out_pdf_t = out_dir / f"{prefix}_solve_time.pdf"
    fig1.savefig(out_pdf_t, bbox_inches="tight")
    print(f"Figure saved to {out_pdf_t}")

    # ---- Figure 2: throughput line plot (unchanged style) -------------
    fig2, ax2 = plt.subplots(figsize=(6.0, 4.0))
    fig2.suptitle(title, fontsize=13)

    B = np.array(batch_sizes)
    for name in results:
        tp = np.array(results[name]["throughput"])
        tp_se = np.array(results[name]["throughput_stderr"])
        if np.all(np.isnan(tp)):
            continue
        ax2.errorbar(
            B, tp, yerr=tp_se,
            color=_PLOT_COLORS.get(name, "k"),
            marker=_PLOT_MARKERS.get(name, "o"),
            linewidth=1.5, markersize=5, capsize=2, label=name,
        )
    ax2.set_xlabel(r"Batch size $B$")
    ax2.set_ylabel(r"Throughput (projections/s)")
    ax2.set_title(r"Solve Throughput")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.legend(framealpha=0.9)
    ax2.grid(True, which="major", alpha=0.5)
    ax2.grid(True, which="minor", alpha=0.2)

    plt.tight_layout()
    out_pdf_tp = out_dir / f"{prefix}_throughput.pdf"
    fig2.savefig(out_pdf_tp, bbox_inches="tight")
    print(f"Figure saved to {out_pdf_tp}")

    if show:
        plt.show()


def load_json(path: Path):
    payload = json.loads(path.read_text())
    return payload["results"], payload["batch_sizes"], payload["params"]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("json_path", type=Path,
                   help="Path to a JSON file written by benchmark_projection.py.")
    p.add_argument("--no-show", action="store_true",
                   help="Skip plt.show() (useful for headless / scripted runs).")
    args = p.parse_args()

    results, batch_sizes, params = load_json(args.json_path)
    plot_results(
        results, batch_sizes, params,
        out_dir=args.json_path.parent,
        prefix=args.json_path.stem,
        show=not args.no_show,
    )
