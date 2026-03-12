#!/usr/bin/env bash
# run_benchmark.sh — Lock CPU/GPU clocks, run benchmark, unlock clocks.
#
# Usage:
#   sudo bash run_benchmark.sh [benchmark args...]
#
# Examples:
#   sudo bash run_benchmark.sh --num_masses 20 --horizon 100 200 300 400 500 600 700 800
#   sudo bash run_benchmark.sh --num_masses 10 20 --horizon 200 400 600 --no_cuopt
#   sudo bash run_benchmark.sh   # uses defaults from benchmark_chain_mass.py

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="/home/fenglong/miniforge3/envs/socu/bin/python"
BENCHMARK_SCRIPT="$SCRIPT_DIR/benchmark_chain_mass.py"

# --- Configuration ---
GPU_GRAPHICS_CLOCK=1500   # MHz — stable mid-range, avoids thermal throttle
GPU_MEMORY_CLOCK=14001    # MHz — max memory clock
CPU_BASE_FREQ=3700000     # kHz — Intel Core Ultra 9 285K base frequency

# =====================================================================
# Lock clocks
# =====================================================================
lock_clocks() {
    echo "========================================"
    echo "  Locking CPU & GPU clocks"
    echo "========================================"

    # --- GPU ---
    echo "[GPU] Enabling persistence mode..."
    nvidia-smi -pm 1

    echo "[GPU] Locking graphics clock to ${GPU_GRAPHICS_CLOCK} MHz..."
    nvidia-smi -lgc ${GPU_GRAPHICS_CLOCK},${GPU_GRAPHICS_CLOCK}

    echo "[GPU] Locking memory clock to ${GPU_MEMORY_CLOCK} MHz..."
    nvidia-smi -lmc ${GPU_MEMORY_CLOCK}

    # --- CPU ---
    echo "[CPU] Setting all cores to performance governor at ${CPU_BASE_FREQ} kHz..."
    for gov in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
        echo performance > "$gov"
    done
    for freq_file in /sys/devices/system/cpu/cpu*/cpufreq/scaling_max_freq; do
        echo ${CPU_BASE_FREQ} > "$freq_file"
    done
    for freq_file in /sys/devices/system/cpu/cpu*/cpufreq/scaling_min_freq; do
        echo ${CPU_BASE_FREQ} > "$freq_file"
    done

    echo ""
    print_status
    echo ""
}

# =====================================================================
# Unlock clocks (restore defaults)
# =====================================================================
unlock_clocks() {
    echo ""
    echo "========================================"
    echo "  Unlocking CPU & GPU clocks"
    echo "========================================"

    # --- GPU ---
    echo "[GPU] Resetting graphics clocks..."
    nvidia-smi -rgc || true

    echo "[GPU] Resetting memory clocks..."
    nvidia-smi -rmc || true

    echo "[GPU] Disabling persistence mode..."
    nvidia-smi -pm 0 || true

    # --- CPU: restore powersave, reset freq range ---
    echo "[CPU] Restoring powersave governor..."
    for freq_file in /sys/devices/system/cpu/cpu*/cpufreq/scaling_min_freq; do
        echo 800000 > "$freq_file" 2>/dev/null || true
    done
    for freq_file in /sys/devices/system/cpu/cpu*/cpufreq/scaling_max_freq; do
        echo 6500000 > "$freq_file" 2>/dev/null || true
    done
    for gov in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
        echo powersave > "$gov" 2>/dev/null || true
    done

    echo ""
    print_status
}

# =====================================================================
# Print current status
# =====================================================================
print_status() {
    echo "--- Current Status ---"

    # CPU
    local gov
    gov=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo "unknown")
    local cpu_min cpu_max
    cpu_min=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq 2>/dev/null || echo "?")
    cpu_max=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>/dev/null || echo "?")
    echo "CPU: governor=${gov}, freq range=${cpu_min}-${cpu_max} kHz"

    # GPU
    nvidia-smi --query-gpu=name,persistence_mode,clocks.current.graphics,clocks.current.memory,power.draw \
        --format=csv,noheader 2>/dev/null || echo "GPU: nvidia-smi query failed"
}

# =====================================================================
# Main
# =====================================================================

# Check root
if [[ $EUID -ne 0 ]]; then
    echo "ERROR: This script must be run as root (sudo)."
    echo "Usage: sudo bash $0 [benchmark args...]"
    exit 1
fi

# Trap to always unlock on exit (Ctrl-C, error, etc.)
trap unlock_clocks EXIT

lock_clocks

echo "========================================"
echo "  Running benchmark"
echo "========================================"
echo "Command: $PYTHON $BENCHMARK_SCRIPT $*"
echo ""

# Run benchmark as the original user (not root) to avoid permission issues
# with conda env / user files. Use SUDO_USER if available.
if [[ -n "${SUDO_USER:-}" ]]; then
    sudo -u "$SUDO_USER" "$PYTHON" "$BENCHMARK_SCRIPT" "$@"
else
    "$PYTHON" "$BENCHMARK_SCRIPT" "$@"
fi

echo ""
echo "========================================"
echo "  Benchmark complete!"
echo "========================================"
