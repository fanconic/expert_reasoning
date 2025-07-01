#!/bin/bash
NUMA_NODE=1
CORES_PER_JOB=24
export OMP_NUM_THREADS=$CORES_PER_JOB
export MKL_NUM_THREADS=$CORES_PER_JOB

exec numactl --cpunodebind=$NUMA_NODE --membind=$NUMA_NODE \
python "$@"