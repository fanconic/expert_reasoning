#!/usr/bin/env bash
set -euo pipefail

# Fixed qwen2.5-7B AIRL segment run set:
#   1. base policy + original reward
#   2. SFT policy + original reward
#   3. base policy + two-head AIRL segment reward_mode=mean_g
#   4. SFT policy + two-head AIRL segment reward_mode=mean_g
#
# By default this launches two experiments at a time and exposes two GPUs to
# each process: "0,1" and "2,3". The current train_irl.py entrypoint is not
# DDP, so this is process-level GPU assignment, not distributed training.

PYTHON_BIN="${PYTHON_BIN:-/home/fanconic/miniconda3/envs/quant_env/bin/python}"
CONFIG_ROOT="${CONFIG_ROOT:-configs/math/qwen7b}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/pdata/caf83/icml_math/outputs}"
LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}/logs/qwen7b_four_runs}"
PROJECT="${PROJECT:-expert_reasoning_math_icml}"
SFT_POLICY_DIR="${SFT_POLICY_DIR:-/mnt/pdata/caf83/icml_math/outputs/qwen7b_sft/best_model}"

# Semicolon separates groups; commas inside a group are passed to CUDA_VISIBLE_DEVICES.
# Examples:
#   GPU_GROUPS_CSV="0,1;2,3"  -> two concurrent jobs, each sees two GPUs
#   GPU_GROUPS_CSV="0;1;2;3"  -> up to four single-GPU jobs
GPU_GROUPS_CSV="${GPU_GROUPS_CSV:-0,1;2,3}"
IFS=';' read -r -a GPU_GROUPS <<< "${GPU_GROUPS_CSV}"
MAX_PARALLEL="${MAX_PARALLEL:-${#GPU_GROUPS[@]}}"

RUN_SPECS=(
  "original_base|qwen7b_interval_original_base_kl0p01_ebs256"
  "original_sft|qwen7b_interval_original_sft_kl0p01_ebs256"
  "mean_g_base|qwen7b_interval_mean_g_base_kl0p01_ebs256"
  "mean_g_sft|qwen7b_interval_mean_g_sft_kl0p01_ebs256"
)

if (( MAX_PARALLEL < 1 )); then
  echo "MAX_PARALLEL must be at least 1." >&2
  exit 1
fi

if (( MAX_PARALLEL > ${#GPU_GROUPS[@]} )); then
  echo "MAX_PARALLEL=${MAX_PARALLEL} exceeds available GPU groups (${#GPU_GROUPS[@]})." >&2
  exit 1
fi

if [[ "${SFT_POLICY_DIR}" == /* && ! -d "${SFT_POLICY_DIR}" ]]; then
  echo "SFT_POLICY_DIR does not exist: ${SFT_POLICY_DIR}" >&2
  echo "Set SFT_POLICY_DIR=/path/to/qwen7b_sft/best_model if your SFT checkpoint lives elsewhere." >&2
  exit 1
fi

mkdir -p "${LOG_ROOT}"

wait_for_wave() {
  local failed=0
  local idx
  for idx in "${!PIDS[@]}"; do
    local pid="${PIDS[$idx]}"
    local name="${NAMES[$idx]}"
    if wait "${pid}"; then
      echo "[done] ${name}"
    else
      echo "[failed] ${name} (see ${LOG_ROOT}/${name}.log)" >&2
      failed=1
    fi
  done
  PIDS=()
  NAMES=()
  if (( failed != 0 )); then
    exit 1
  fi
}

launch_run() {
  local config_name="$1"
  local run_name="$2"
  local gpu_group="$3"
  local log_file="${LOG_ROOT}/${run_name}.log"

  echo "[launch] ${run_name} on CUDA_VISIBLE_DEVICES=${gpu_group}"
  (
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export CUDA_VISIBLE_DEVICES="${gpu_group}"
    export HYDRA_FULL_ERROR=1
    export USE_TORCH=1
    export USE_TF=0
    export USE_FLAX=0
    export UNSLOTH_COMPILE_OVERWRITE=0
    export SFT_POLICY_DIR

    "${PYTHON_BIN}" train_irl.py \
      --config-path="${CONFIG_ROOT}" \
      --config-name="airl_segment/${config_name}" \
      wandb.project="${PROJECT}" \
      wandb.run_name="${run_name}" \
      training.output_dir="${OUTPUT_ROOT}/${run_name}" \
      training.beta=0.01 \
      training.per_device_train_batch_size=16 \
      training.gradient_accumulation_steps=16 \
      model.dense_partial_fixed_n=15 \
      model.segment_tokens=15
  ) > "${log_file}" 2>&1 &

  PIDS+=("$!")
  NAMES+=("${run_name}")
}

PIDS=()
NAMES=()

for spec in "${RUN_SPECS[@]}"; do
  IFS='|' read -r config_name run_name <<< "${spec}"

  if (( ${#PIDS[@]} >= MAX_PARALLEL )); then
    wait_for_wave
  fi

  slot="${#PIDS[@]}"
  launch_run "${config_name}" "${run_name}" "${GPU_GROUPS[$slot]}"
done

if (( ${#PIDS[@]} > 0 )); then
  wait_for_wave
fi

echo "All qwen7b four-run experiments completed."
