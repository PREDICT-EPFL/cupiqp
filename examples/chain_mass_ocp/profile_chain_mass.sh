#!/bin/bash

# Usage: ./profile_chain_mass.sh [num_masses] [horizon] [kkt_solver]
# Example: ./profile_chain_mass.sh 20 500 multistage_block_cholesky

# Default values if not provided
NUM_MASSES=${1:-20}
HORIZON=${2:-500}
SOLVER=${3:-multistage_block_cholesky}

# Construct the output filename
# Format: M{NUM_MASSES}_N{HORIZON}_{SOLVER}
OUTPUT_NAME="M${NUM_MASSES}_N${HORIZON}_${SOLVER}"

echo "Starting nsys profile..."
echo "  Masses: $NUM_MASSES"
echo "  Horizon: $HORIZON"
echo "  Solver: $SOLVER"
echo "  Output: ${OUTPUT_NAME}.nsys-rep"

# Run nsys profile
# We use --force-overwrite true to avoid errors if rerunning
nsys profile --force-overwrite true -o "$OUTPUT_NAME" \
    python example_chain_mass.py \
    --num_masses "$NUM_MASSES" \
    --horizon "$HORIZON" \
    --kkt_solver "$SOLVER"

echo "Profiling complete."
