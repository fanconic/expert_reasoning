#!/usr/bin/env bash
# GPU0 script
#
# This script runs the following dataset×model bundles (6 runs each):
#  - gsm8k_rebuttals × qwen7b
#  - medreason_rebuttals × qwen3b
#  - gsm8k_rebuttals × llama3b
#
# Each bundle runs:
#  - base:     eval, grpo_eval, sft_eval
#  - variants: eval+sparse, eval+full, eval+ovr
#
# Crash summary printed at end; exits non-zero if any run failed.

set -u  # do NOT use -e; we want to continue after failures

GPU_NUM="1"
RUNNER="runner_scripts/${GPU_NUM}_run_gpu_node.sh"
EVALUATE_PY="evaluate.py"

declare -A VARIANT_OVERRIDES
VARIANT_OVERRIDES["sparse"]="model.dense_rewards=false"
VARIANT_OVERRIDES["full"]="model.dense_rewards=full"
VARIANT_OVERRIDES["ovr"]=""

run_name() {
  local model="$1" variant="$2"
  case "$variant" in
    base)   echo "${model}_8ga_8gens_clipped" ;;
    sparse) echo "${model}_8ga_8gens_clipped_sparse" ;;
    full)   echo "${model}_8ga_8gens_clipped_full" ;;
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

  # Base configs
  for config in eval grpo_eval sft_eval; do
    run_cmd \
      "dataset=${dataset} model=${model} config=${config} variant=base" \
      bash "$RUNNER" "$EVALUATE_PY" \
        --config-path="$config_path" \
        --config-name="$config"
  done

  # Eval-only variants
  for variant in sparse full ovr; do
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

# ======== Bundles on GPU0 ========
run_bundle gsm8k_rebuttals     qwen7b
run_bundle medreason_rebuttals qwen3b
# =================================

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
