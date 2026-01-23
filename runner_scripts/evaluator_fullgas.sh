#!/usr/bin/env bash
# Clean evaluation runner with crash summary.
# - Continues after failures
# - Reports all failures at end
# - Exits non-zero if any failures occurred

set -u  # error on unset vars (but do NOT use -e; we want to continue after failures)

GPU_NUM="1"
RUNNER="runner_scripts/${GPU_NUM}_run_gpu_node.sh"
EVALUATE_PY="evaluate.py"

# Datasets and models
DATASETS=("gsm8k_rebuttals" "medreason_rebuttals")
MODELS=("qwen3b" "llama3b" "qwen7b" "llama8b")

# Base configs to run (no overrides)
BASE_CONFIGS=("eval" "grpo_eval" "sft_eval")

# Variants for the "eval" config only
# (matches your original script where sparse/full are only run with --config-name=eval)
declare -A VARIANT_OVERRIDES
VARIANT_OVERRIDES["base"]=""
VARIANT_OVERRIDES["sparse"]="model.dense_rewards=false"
VARIANT_OVERRIDES["full"]="model.dense_rewards=full"
VARIANT_OVERRIDES["ovr"]=""

# You can tweak this naming convention if you like.
run_name() {
  local model="$1" variant="$2"
  case "$variant" in
    base)   echo "${model}_8ga_8gens_clipped" ;;
    sparse) echo "${model}_8ga_8gens_clipped_sparse" ;;
    full)   echo "${model}_8ga_8gens_clipped_full" ;;
    ovr)   echo "${model}_ovr" ;;
    *)      echo "${model}_${variant}" ;;
  esac
}

# Collect failures here
FAILED_RUNS=()

run_cmd() {
  local label="$1"
  shift
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
  return 0  # never propagate failure; we want to continue
}

for DATASET in "${DATASETS[@]}"; do
  for MODEL in "${MODELS[@]}"; do
    CONFIG_PATH="configs/${DATASET}/${MODEL}"

    # 1) Base runs: eval, grpo_eval, sft_eval
    for CONFIG_NAME in "${BASE_CONFIGS[@]}"; do
      run_cmd \
        "dataset=${DATASET} model=${MODEL} config=${CONFIG_NAME} variant=base" \
        bash "$RUNNER" "$EVALUATE_PY" \
          --config-path="$CONFIG_PATH" \
          --config-name="$CONFIG_NAME"
    done

    # 2) Extra variants for eval only: sparse and full (plus base is already done above)
    for VARIANT in sparse full; do
      OVERRIDE="${VARIANT_OVERRIDES[$VARIANT]}"
      WNAME="$(run_name "$MODEL" "$VARIANT")"

      # Build override args safely (as separate tokens)
      # Includes wandb.run_name=... plus the variant override (dense_rewards setting)
      run_cmd \
        "dataset=${DATASET} model=${MODEL} config=eval variant=${VARIANT}" \
        bash "$RUNNER" "$EVALUATE_PY" \
          --config-path="$CONFIG_PATH" \
          --config-name="eval" \
          "wandb.run_name=${WNAME}" \
          ${OVERRIDE}
    done
  done
done

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
