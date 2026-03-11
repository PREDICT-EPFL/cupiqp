#!/bin/bash
# Profile CuClarabel solver on a Maros-Meszaros QP with NVIDIA Nsight Systems.
#
# Usage:
#   ./profile_maros_cuclarabel.sh <problem_name>
#
# Example:
#   ./profile_maros_cuclarabel.sh CVXQP1_S
#
# Output:
#   benchmarks/nsys_reports/cuclarabel_<problem_name>.nsys-rep

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <problem_name>"
    echo "Example: $0 CVXQP1_S"
    exit 1
fi

PROBLEM_NAME="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPORT_DIR="$SCRIPT_DIR/nsys_reports"
mkdir -p "$REPORT_DIR"

REPORT_FILE="$REPORT_DIR/${PROBLEM_NAME}_cuclarabel"

echo "Profiling CuClarabel on $PROBLEM_NAME ..."
nsys profile \
    --output "$REPORT_FILE" \
    --force-overwrite true \
    --trace cuda,nvtx,osrt \
    --capture-range=cudaProfilerApi \
    --cuda-graph-trace=node \
    julia "$SCRIPT_DIR/bench_maros_cuclarabel.jl" "$PROBLEM_NAME" 0

echo ""
echo "Report saved to: ${REPORT_FILE}.nsys-rep"
