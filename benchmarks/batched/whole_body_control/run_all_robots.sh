#!/usr/bin/env bash
# Run the unified WBC benchmark for every supported robot, then plot.
#
# ``benchmark_wbc.py`` starts a fresh Python process for every
# (solver, batch-size) case, so GPU allocator caches from one measurement do
# not reduce the available memory for another. Each robot run writes
# results/benchmark_wbc_<robot>.json. After all robots are done, a two-column
# summary plot is generated automatically:
#   - wbc_summary.{pdf,png}  via plot_wbc_lineplot.py
#     (rows = robots, columns = solve time | max iterations)
#
# Everything printed by this script (and by the python tools it invokes)
# is mirrored to a timestamped log file under ``results/`` so failed runs
# can be diagnosed after the fact. Default name:
#     results/run_<UTC-timestamp>.log
# Override via ``LOG_FILE=...`` (use ``/dev/null`` to disable logging).
#
# Knobs are picked up from the environment so a quick smoke run is just:
#     BATCH_SIZES="64 256" N_REPEATS=3 ./run_all_robots.sh
#
# Skip the plotting step entirely with:
#     SKIP_PLOTS=1 ./run_all_robots.sh
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

ROBOTS=("anymal_c" "h1" "iiwa14")
BATCH_SIZES="${BATCH_SIZES:-64 128 256 512 1024 2048 4096 8192 16384}"
N_REPEATS="${N_REPEATS:-10}"
TOL="${TOL:-1e-8}"
MAX_ITER="${MAX_ITER:-200}"
SEED="${SEED:-0}"
CONTACT_PROB="${CONTACT_PROB:-0.7}"
SKIP_PLOTS="${SKIP_PLOTS:-0}"

mkdir -p results

# Mirror stdout+stderr through ``tee`` into a log file. Must come after
# ``mkdir -p results`` so the file is creatable; must use process
# substitution rather than a normal pipe so ``set -e`` / ``pipefail``
# still see the real exit code of the python commands.
LOG_FILE="${LOG_FILE:-results/run_$(date -u +%Y%m%dT%H%M%SZ).log}"
if [[ "$LOG_FILE" != "/dev/null" ]]; then
    echo "Logging to $LOG_FILE"
    exec > >(tee "$LOG_FILE") 2>&1
fi
echo "----------------------------------------------------------------"
echo "  WBC benchmark sweep — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  host:        $(hostname)"
echo "  cwd:         $HERE"
echo "  python:      $(command -v python)  ($(python --version 2>&1))"
echo "  git commit:  $(git -C "$HERE" rev-parse --short HEAD 2>/dev/null || echo 'n/a')"
echo "  params:      batch_sizes='$BATCH_SIZES'  n_repeats=$N_REPEATS"
echo "               tol=$TOL  max_iter=$MAX_ITER  seed=$SEED  contact_prob=$CONTACT_PROB"
echo "  isolation:   fresh Python process per (solver, batch size) case"
echo "----------------------------------------------------------------"

for robot in "${ROBOTS[@]}"; do
    echo
    echo "================================================================"
    echo "  Benchmarking $robot"
    echo "================================================================"
    python benchmark_wbc.py \
        --robot "$robot" \
        --batch-sizes $BATCH_SIZES \
        --n-repeats "$N_REPEATS" \
        --tol "$TOL" \
        --max-iter "$MAX_ITER" \
        --seed "$SEED" \
        --contact-prob "$CONTACT_PROB" \
        --save "results/benchmark_wbc_${robot}.json"
done

if [[ "$SKIP_PLOTS" == "1" ]]; then
    echo
    echo "All robots done. SKIP_PLOTS=1 — plot manually with:"
    echo "    python plot_wbc_lineplot.py"
    exit 0
fi

echo
echo "================================================================"
echo "  Plotting summary  (rows: robots, cols: solve time | iterations)"
echo "================================================================"
python plot_wbc_lineplot.py --no-show

echo
echo "Done."
