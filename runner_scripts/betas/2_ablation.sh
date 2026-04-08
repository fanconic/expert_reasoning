#!/usr/bin/env bash
set -u 

# --- Configuration ---
export GPU_NUM="2" 
RUNNER="runner_scripts/${GPU_NUM}_run_gpu_node.sh"

# Fixed constants for this specific ablation
MODEL="llama3b"
WARMUP_DIR="/mnt/pdata/caf83/icml_math/warmed_up_rewards/llama3b/partial/"

BASE_TRAIN_PARAMS="model.reward_updates_per_policy_step=3 training.beta=0.1 training.max_steps=400 training.buffer_size=50 model.warmup_reward_dir=${WARMUP_DIR}"

FAILED_RUNS=()

run_cmd() {
    local label="$1"; shift
    echo -e "\n▶ $label\n  $*"
    "$@"
    local rc=$?
    [[ $rc -ne 0 ]] && FAILED_RUNS+=("$label (exit=$rc)") && echo "  ✗ FAILED" || echo "  ✓ OK"
}

run_bounds_ablation() {
    local LB="$1"
    local UB="$2"
    
    # Create a clean name for WandB: e.g., qwen3b_partial_range_-5_5
    local WNAME="${MODEL}_partial_range_${LB}_${UB}"
    
    # Overrides for this specific run
    local BOUND_FLAGS="model.reward_lb=${LB} model.reward_ub=${UB}"
    local OVERRIDE="wandb.run_name=${WNAME} model.dense_rewards=partial ${BOUND_FLAGS}"

    echo "------------------------------------------------"
    echo "Starting Ablation: ${WNAME} [LB: ${LB}, UB: ${UB}]"
    echo "------------------------------------------------"
    
    # 1. TRAIN
    run_cmd "${WNAME}_TRAIN" bash "$RUNNER" train_irl.py \
        --config-path="configs/gsm8k_rebuttals/${MODEL}" \
        --config-name="good_run" \
        $OVERRIDE $BASE_TRAIN_PARAMS

    # 2. EVAL
    run_cmd "${WNAME}_EVAL" bash "$RUNNER" evaluate.py \
        --config-path="configs/gsm8k_rebuttals/${MODEL}" \
        --config-name="eval" \
        $OVERRIDE
}

# --- Execution ---
# You can add or remove pairs here as needed.
# Format: run_bounds_ablation <Lower_Bound> <Upper_Bound>

run_bounds_ablation "-10.0" "10.0"

# --- Final Report ---
if [[ ${#FAILED_RUNS[@]} -ne 0 ]]; then
    echo -e "\nFAILURES detected:"
    printf "  %s\n" "${FAILED_RUNS[@]}"
    exit 1
fi
echo "All bound ablations completed successfully."


bash runner_scripts/corruption/2_ablation.sh

bash runner_scripts/sft_reranking/evaluator_2_fullgas.sh