#!/usr/bin/env bash
set -u 

# --- Change these for each GPU ---
export GPU_NUM="1"       # 0, 1, 2, 3
DENSITY="sparse"         # sparse, full, partial, partial_fixed
# ---------------------------------

RUNNER="runner_scripts/${GPU_NUM}_run_gpu_node.sh"
COMMON_REWARD_FLAGS="model.reward_lb=-5.0 model.reward_ub=5.0"

FAILED_RUNS=()

# Helper to map full dataset name to directory keyword
get_type() {
    case "$1" in
        "gsm8k_rebuttals")     echo "math" ;;
        "medreason_rebuttals") echo "medicine" ;;
        "mmlu_rebuttals")      echo "mmlu" ;;
    esac
}

run_cmd() {
    local label="$1"; shift
    echo -e "\n▶ $label\n  $*"
    "$@"
    local rc=$?
    [[ $rc -ne 0 ]] && FAILED_RUNS+=("$label (exit=$rc)") && echo "  ✗ FAILED" || echo "  ✓ OK"
}

run_transfer_eval() {
    local DATASET_P="$1" # Policy Dataset (Generations)
    local DATASET_R="$2" # Reward Dataset (Training)
    local REWARD_ARCH="$3"
    local SUFFIX="$4"

    local TYPE_P=$(get_type "$DATASET_P")
    local TYPE_R=$(get_type "$DATASET_R")

    # Handle Sparse naming for Hydra
    local DENSE_VAL="$SUFFIX"
    [[ "$SUFFIX" == "sparse" ]] && DENSE_VAL="false"

    # 1. Policy: Fixed Qwen 7B SFT from the Policy Dataset
    local POLICY_PATH="/mnt/pdata/caf83/icml_${TYPE_P}/outputs/qwen7b_sft/best_model"
    

    # WandB Name: e.g., transfer_qwen3b_full_P-math_R-mmlu
    local WNAME="${REWARD_ARCH}_${SUFFIX}_new"
    local OUTNAME="transfer_${REWARD_ARCH}_${SUFFIX}_P_${TYPE_P}_R_${TYPE_R}"
    
     local OVERRIDE="wandb.run_name=${WNAME} \
                   model.dense_rewards=${DENSE_VAL} \
                   ${COMMON_REWARD_FLAGS} \
                   model.policy_name=${POLICY_PATH}"
    
    # We use the config-path of the Policy Dataset to ensure generation loading matches
    run_cmd "${WNAME}" bash "$RUNNER" evaluate_pregenerated_transfer.py \
        --config-path="configs/${DATASET_P}/${REWARD_ARCH}" --config-name="eval" --out-name=${OUTNAME} $OVERRIDE
}

# --- Execution Matrix ---
DATASETS=("gsm8k_rebuttals" "medreason_rebuttals" "mmlu_rebuttals")
ARCHS=("qwen3b" "llama3b" "llama8b")

for DS_POLICY in "${DATASETS[@]}"; do
    for DS_REWARD in "${DATASETS[@]}"; do
        for ARCH in "${ARCHS[@]}"; do
            run_transfer_eval "$DS_POLICY" "$DS_REWARD" "$ARCH" "$DENSITY"
        done
    done
done

# --- Final Report ---
echo -e "\n======================\nGPU ${GPU_NUM} (${DENSITY}) SUMMARY\n======================"
if [[ ${#FAILED_RUNS[@]} -eq 0 ]]; then
    echo "All 27 transfer evaluations succeeded!"
else
    printf "Failures (%d):\n  %s\n" "${#FAILED_RUNS[@]}" "${FAILED_RUNS[@]}"
    exit 1
fi