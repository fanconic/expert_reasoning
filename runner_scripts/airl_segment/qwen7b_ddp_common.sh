#!/usr/bin/env bash
set -euo pipefail

# Shared torchrun launcher for the fixed qwen2.5-7B AIRL segment runs.
# Wrappers set CONFIG_NAME/RUN_NAME plus default GPU groups and ports.

: "${CONFIG_NAME:?CONFIG_NAME must be set by the wrapper script.}"
: "${RUN_NAME:?RUN_NAME must be set by the wrapper script.}"

PYTHON_BIN="${PYTHON_BIN:-/home/fanconic/miniconda3/envs/quant_env/bin/python}"
CONFIG_ROOT="${CONFIG_ROOT:-configs/math/qwen7b}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/pdata/caf83/icml_math/outputs}"
LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}/logs/qwen7b_ddp}"
PROJECT="${PROJECT:-expert_reasoning_math_icml}"
SFT_POLICY_DIR="${SFT_POLICY_DIR:-/mnt/pdata/caf83/icml_math/outputs/qwen7b_sft/best_model}"

GPUS="${GPUS:-${DEFAULT_GPUS:-0,1}}"
MASTER_PORT="${MASTER_PORT:-${DEFAULT_MASTER_PORT:-29600}}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-16}"
EFFECTIVE_BATCH_SIZE="${EFFECTIVE_BATCH_SIZE:-256}"

IFS=',' read -r -a GPU_IDS <<< "${GPUS}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${#GPU_IDS[@]}}"

if (( NPROC_PER_NODE < 1 )); then
  echo "NPROC_PER_NODE must be at least 1." >&2
  exit 1
fi

denom=$((PER_DEVICE_BATCH_SIZE * NPROC_PER_NODE))
if (( EFFECTIVE_BATCH_SIZE % denom != 0 )); then
  echo "EFFECTIVE_BATCH_SIZE=${EFFECTIVE_BATCH_SIZE} is not divisible by PER_DEVICE_BATCH_SIZE * NPROC_PER_NODE = ${denom}." >&2
  exit 1
fi
GRAD_ACCUM=$((EFFECTIVE_BATCH_SIZE / denom))

if [[ "${CONFIG_NAME}" == *sft* && "${SFT_POLICY_DIR}" == /* && ! -d "${SFT_POLICY_DIR}" ]]; then
  echo "SFT_POLICY_DIR does not exist: ${SFT_POLICY_DIR}" >&2
  echo "Set SFT_POLICY_DIR=/path/to/qwen7b_sft/best_model before launching SFT-policy runs." >&2
  exit 1
fi

mkdir -p "${LOG_ROOT}" "${OUTPUT_ROOT}/${RUN_NAME}"
LOG_FILE="${LOG_ROOT}/${RUN_NAME}.log"

cat <<EOF
Launching ${RUN_NAME}
  config: ${CONFIG_NAME}
  GPUs: ${GPUS}
  nproc_per_node: ${NPROC_PER_NODE}
  master_port: ${MASTER_PORT}
  per-device batch: ${PER_DEVICE_BATCH_SIZE}
  grad accumulation: ${GRAD_ACCUM}
  effective batch: ${EFFECTIVE_BATCH_SIZE}
  log: ${LOG_FILE}
EOF

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${GPUS}"
export HYDRA_FULL_ERROR=1
export PYTHONPATH="${PYTHONPATH:-.}"
export USE_TORCH=1
export USE_TF=0
export USE_FLAX=0
export UNSLOTH_COMPILE_OVERWRITE=0
export SFT_POLICY_DIR

exec "${PYTHON_BIN}" -m torch.distributed.run \
  --nnodes=1 \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  train_irl.py \
  --config-path="${CONFIG_ROOT}" \
  --config-name="airl_segment/${CONFIG_NAME}" \
  wandb.project="${PROJECT}" \
  wandb.run_name="${RUN_NAME}" \
  training.output_dir="${OUTPUT_ROOT}/${RUN_NAME}" \
  training.beta=0.01 \
  training.per_device_train_batch_size="${PER_DEVICE_BATCH_SIZE}" \
  training.gradient_accumulation_steps="${GRAD_ACCUM}" \
  model.dense_partial_fixed_n=15 \
  model.segment_tokens=15 \
  2>&1 | tee "${LOG_FILE}"
