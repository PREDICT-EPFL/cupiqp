#!/bin/bash
# Profile cuOpt solver on a Maros-Meszaros QP with NVIDIA Nsight Systems.
#
# Usage:
#   ./profile_maros_cuopt.sh <problem_name>
#
# Example:
#   ./profile_maros_cuopt.sh CVXQP1_S
#
# Output:
#   benchmarks/nsys_reports/<problem_name>_cuopt.nsys-rep

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

REPORT_FILE="$REPORT_DIR/${PROBLEM_NAME}_cuopt"

echo "Profiling cuOpt on $PROBLEM_NAME ..."
nsys profile \
    --output "$REPORT_FILE" \
    --force-overwrite true \
    --trace cuda,nvtx,osrt \
    --capture-range=cudaProfilerApi \
    --cuda-graph-trace=node \
    python "$SCRIPT_DIR/bench_maros_cuopt.py" "$PROBLEM_NAME" 0

echo ""
echo "Report saved to: ${REPORT_FILE}.nsys-rep"
