#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# --- Argument Check ---
if [ -z "$1" ]; then
    echo "Error: You must provide the Python script to run as the first argument."
    echo "Usage: $0 <your_script.py> [args_for_your_script...]"
    exit 1
fi

# The first argument is the Python script. Store it and then remove it from the list of arguments.
PYTHON_SCRIPT=$1
shift

echo "--- Starting task on GPU 0 and NUMA Node 0 ---"
echo "  > Python Script: $PYTHON_SCRIPT"
echo "  > Arguments: $@"

# --- Configuration ---
GPU_ID=0
NUMA_NODE=0
CORES_PER_JOB=24 # Adjust based on your setup (Total Cores / Num Nodes)
PYTHON_BIN="${PYTHON_BIN:-python}"

# --- Environment Setup ---
export OMP_NUM_THREADS=$CORES_PER_JOB
export MKL_NUM_THREADS=$CORES_PER_JOB
export TRITON_CACHE_DIR=~/.cache/triton_gpu${GPU_ID}
mkdir -p $TRITON_CACHE_DIR

# --- Execute the Python Script ---
# $PYTHON_SCRIPT now holds the script name, and "$@" holds the rest of the arguments.
if command -v numactl >/dev/null 2>&1; then
    numactl --cpunodebind=$NUMA_NODE --membind=$NUMA_NODE \
    env CUDA_VISIBLE_DEVICES=$GPU_ID \
    "$PYTHON_BIN" $PYTHON_SCRIPT "$@"
else
    echo "  > numactl not found; running without NUMA binding."
    env CUDA_VISIBLE_DEVICES=$GPU_ID \
    "$PYTHON_BIN" $PYTHON_SCRIPT "$@"
fi

echo "--- Task on GPU 0 finished ---"
