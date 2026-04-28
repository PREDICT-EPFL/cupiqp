#!/usr/bin/env bash
# Sweep the constrained-least-squares benchmark over (n, rows_F, rows_A, batch sizes).
#
# Each entry in CONFIGS is "n row_F row_A | batch_sizes...". The "|" separates
# the QP shape from the space-separated batch-size list so the inner shell loop
# can split cleanly. Output JSONs go to ./out/<tag>.json so multiple runs don't
# overwrite each other.
#
# Prerequisites:
#     A Python environment that has cuPIQP and its dependencies installed,
#     active in this shell (e.g. `conda activate <yourenv>` or `source venv/...`).
#     Override the python binary with PYTHON=... if needed.
#
# Usage:
#     ./run_benchmarks.sh                   # all configs below, with plots
#     ./run_benchmarks.sh --no-plot         # skip plotting (faster, headless)
#     ./run_benchmarks.sh --tol 1e-6 ...    # any extra flags are forwarded
#                                              to benchmark_cls.py for every run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-python}"

# ----- edit this list to define the sweep -----
# Format: "n row_F row_A | b1 b2 b3 ..."
CONFIGS=(
    "20  40   5 | 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768"
    "50  100  10 | 8 16 32 64 128 256 512 1024 2048 4096 8192 16384"
    "100 200  20 | 8 16 32 64 128 256 512 1024 2048 4096 8192"
)
# ---------------------------------------------

mkdir -p results

idx=0
for cfg in "${CONFIGS[@]}"; do
    idx=$((idx + 1))

    # Split on "|" into shape and batch list.
    shape_part="${cfg%%|*}"
    batch_part="${cfg##*|}"

    # Parse "n row_F row_A" from shape_part. read trims whitespace.
    read -r n row_F row_A <<< "$shape_part"

    # Range hint for the batch-size list: "<min>-<max>".
    read -ra batch_arr <<< "$batch_part"
    batch_min="${batch_arr[0]}"
    batch_max="${batch_arr[-1]}"

    # Tag includes the config index so two configs with the same (n, row_F, row_A)
    # but different batch lists never collide on the same output filename.
    tag=$(printf "cfg%02d_n%s_F%s_A%s_B%s-%s" \
        "$idx" "$n" "$row_F" "$row_A" "$batch_min" "$batch_max")
    out_json="results/${tag}.json"
    out_pdf="results/${tag}.pdf"
    out_png="results/${tag}.png"

    echo "==============================================================="
    echo "  [${idx}] n=${n}, row_F=${row_F}, row_A=${row_A}"
    echo "  Batch sizes: ${batch_part}"
    echo "  Output: ${out_json}"
    echo "==============================================================="

    # Remove any stale benchmark_cls.{pdf,png} so the rename below only picks
    # up plots produced by THIS run.
    rm -f benchmark_cls.pdf benchmark_cls.png

    "$PYTHON" benchmark_cls.py \
        --n "$n" \
        --row-F "$row_F" \
        --row-A "$row_A" \
        --batch-sizes $batch_part \
        --save "$out_json" \
        "$@"

    # benchmark_cls.py writes its plots next to the script as benchmark_cls.{pdf,png};
    # rename them per-config so they don't get clobbered.
    if [[ -f benchmark_cls.pdf ]]; then mv benchmark_cls.pdf "$out_pdf"; fi
    if [[ -f benchmark_cls.png ]]; then mv benchmark_cls.png "$out_png"; fi
done

echo
echo "Done. Outputs in $SCRIPT_DIR/results/"
