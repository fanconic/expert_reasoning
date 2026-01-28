#!/usr/bin/env bash
# GPU1 script
#
# This script runs the following dataset×model bundles (6 runs each):
#  - medreason_rebuttals × qwen7b
#  - gsm8k_rebuttals × qwen3b
#  - medreason_rebuttals × llama3b
#
# Each bundle runs:
#  - base:     eval, grpo_eval, sft_eval
#  - variants: eval+sparse, eval+full, eval+ovr
#
# Crash summary printed at end; exits non-zero if any run failed.

set -u  # do NOT use -e; we want to continue after failures

GPU_NUM="2"
RUNNER="runner_scripts/${GPU_NUM}_run_gpu_node.sh"
EVALUATE_PY="evaluate_pregenerated.py"

declare -A VARIANT_OVERRIDES
VARIANT_OVERRIDES["partial_fixed"]="model.dense_rewards=partial_fixed"
VARIANT_OVERRIDES["ovr"]=""
VARIANT_OVERRIDES["base"]=""

run_name() {
  local model="$1" variant="$2"
  case "$variant" in
    base)   echo "${model}_partial" ;;
    partial_fixed) echo "${model}_partial_fixed" ;;
    ovr)    echo "${model}_ovr" ;;
    *)      echo "${model}_${variant}" ;;
  esac
}

FAILED_RUNS=()

run_cmd() {
  local label="$1"; shift
  echo ""
  echo "▶ $label"
  echo "  $*"
  "$@"
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    FAILED_RUNS+=("$label (exit=$rc) :: $*")
    echo "  ✗ FAILED (exit=$rc)"
  else
    echo "  ✓ OK"
  fi
  return 0
}

run_bundle() {
  local dataset="$1" model="$2"
  local config_path="configs/${dataset}/${model}"

  for variant in base partial_fixed ovr; do
    local override="${VARIANT_OVERRIDES[$variant]}"
    local wname="$(run_name "$model" "$variant")"
    run_cmd \
      "dataset=${dataset} model=${model} config=eval variant=${variant}" \
      bash "$RUNNER" "$EVALUATE_PY" \
        --config-path="$config_path" \
        --config-name="eval" \
        "wandb.run_name=${wname}" \
        ${override}
  done
}

# ======== Bundles on GPU1 ========
run_bundle medreason_rebuttals qwen7b
run_bundle gsm8k_rebuttals     qwen3b
# =================================

bash runner_scripts/${GPU_NUM}_run_gpu_node.sh evaluate_pregenerated.py --config-path=configs/medreason_rebuttals/switch_reward --config-name=eval_qwen7b
bash runner_scripts/${GPU_NUM}_run_gpu_node.sh evaluate_pregenerated.py --config-path=configs/gsm8k_rebuttals/human_error --config-name=eval

echo ""
echo "======================"
echo " Crash report summary "
echo "======================"

if [[ ${#FAILED_RUNS[@]} -eq 0 ]]; then
  echo "All runs succeeded."
  exit 0
else
  echo "Failures: ${#FAILED_RUNS[@]}"
  echo ""
  i=1
  for item in "${FAILED_RUNS[@]}"; do
    echo "  $i) $item"
    ((i++))
  done
  exit 1
fi
