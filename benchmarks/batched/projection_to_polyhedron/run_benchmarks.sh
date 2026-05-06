#!/usr/bin/env bash
# Sweep the polytope-projection benchmark over several (n, n_inequalities) shapes.
# Each batch element gets its own random polytope (random A, b = ||A_row||
# + slack so the unit ball is strictly inside) and one query point. Both
# polytopes and queries are generated in-process at the start of every run.
#
# Per config: benchmark_projection.py writes results/<tag>.json, then
# plot_results.py reads it and writes results/<tag>_solve_time.pdf and
# results/<tag>_throughput.pdf alongside.
#
# Usage:
#     ./run_benchmarks.sh                 # all configs, with plots
#     ./run_benchmarks.sh --no-plot       # skip plotting (faster, headless)
#     ./run_benchmarks.sh --tol 1e-8 ...  # extra flags forwarded to
#                                            benchmark_projection.py per run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-python}"

# ----- edit this list to define the sweep -----
# Format: "n n_inequalities | b1 b2 b3 ..."
# Default n_inequalities is 5*n (>= 2*n needed for each polytope to be bounded
# with high probability).
CONFIGS=(
    "5   25 | 16 32 64 128 256 512 1024 2048"
    "10  50 | 16 32 64 128 256 512 1024 2048"
    "20 100 | 16 32 64 128 256 512 1024 2048"
    "40 200 | 16 32 64 128 256 512 1024 2048"
)
# ---------------------------------------------

SEED="${SEED:-10}"

# Detect --no-plot among the forwarded args so we can skip the plot step
# without passing the (now-unknown) flag to benchmark_projection.py.
PLOT=1
forwarded=()
for arg in "$@"; do
    if [[ "$arg" == "--no-plot" ]]; then
        PLOT=0
    else
        forwarded+=("$arg")
    fi
done

mkdir -p results

idx=0
for cfg in "${CONFIGS[@]}"; do
    idx=$((idx + 1))

    shape_part="${cfg%%|*}"
    batch_part="${cfg##*|}"

    read -r n n_inequalities <<< "$shape_part"
    read -ra batch_arr <<< "$batch_part"
    batch_min="${batch_arr[0]}"
    batch_max="${batch_arr[-1]}"

    tag=$(printf "cfg%02d_n%s_l%s_B%s-%s" \
        "$idx" "$n" "$n_inequalities" "$batch_min" "$batch_max")
    out_json="results/${tag}.json"

    echo "==============================================================="
    echo "  [${idx}] n=${n}, n_inequalities=${n_inequalities}"
    echo "  Batch sizes: ${batch_part}"
    echo "  Output:      ${out_json}"
    echo "==============================================================="

    "$PYTHON" benchmark_projection.py \
        --n "$n" \
        --n-inequalities "$n_inequalities" \
        --seed "$SEED" \
        --batch-sizes $batch_part \
        --save "$out_json" \
        "${forwarded[@]}"

    if [[ "$PLOT" -eq 1 ]]; then
        "$PYTHON" plot_results.py "$out_json" --no-show
    fi
done

echo
echo "Done. Outputs in $SCRIPT_DIR/results/"
