#!/usr/bin/env bash
# Balanced GPU Script
set -u 

# --- Configuration ---
export GPU_NUM="1"  # Change this for each script (0, 1, 2, 3)
RUNNER="runner_scripts/${GPU_NUM}_run_gpu_node.sh"
TRAIN_PARAMS="model.reward_updates_per_policy_step=3 training.beta=0.1 training.buffer_size=50 training.max_steps=400"
COMMON_REWARD_FLAGS="model.reward_lb=-5.0 model.reward_ub=5.0"

FAILED_RUNS=()

# --- Helper Functions ---
run_cmd() {
    local label="$1"; shift
    echo -e "\n▶ $label\n  $*"
    "$@"
    local rc=$?
    [[ $rc -ne 0 ]] && FAILED_RUNS+=("$label (exit=$rc)") && echo "  ✗ FAILED" || echo "  ✓ OK"
}

run_task() {
    local DATASET="$1"
    local MODEL="$2"
    local SUFFIX="$3"

    # Map Dataset to Warmup Type
    local TYPE="math"
    [[ "$DATASET" == "medreason_rebuttals" ]] && TYPE="medicine"
    [[ "$DATASET" == "mmlu_rebuttals" ]] && TYPE="mmlu"

    # Handle Sparse naming
    local DENSE_VAL="$SUFFIX"
    [[ "$SUFFIX" == "sparse" ]] && DENSE_VAL="false"

    local WNAME="${MODEL}_${SUFFIX}_new"
    local OVERRIDE="wandb.run_name=${WNAME} model.dense_rewards=${DENSE_VAL} ${COMMON_REWARD_FLAGS}"
    
    # Append Warmup if not partial
    # if [[ "$SUFFIX" != "partial" ]]; then
    #     local WARMUP_DIR="/mnt/pdata/caf83/icml_${TYPE}/warmed_up_rewards/${MODEL}/${SUFFIX}/"
    #     OVERRIDE="${OVERRIDE} model.warmup_reward_dir=${WARMUP_DIR}"
    # fi

    # 1. TRAIN
    # run_cmd "${WNAME}_TRAIN" bash "$RUNNER" irl_train.py \
    #     --config-path="configs/${DATASET}/${MODEL}" --config-name="good_run" $OVERRIDE $TRAIN_PARAMS

    # 2. EVAL
    run_cmd "${WNAME}_EVAL" bash "$RUNNER" evaluate.py \
        --config-path="configs/${DATASET}/${MODEL}" --config-name="eval" $OVERRIDE
}

# =========================================================
# WORKLOAD SECTION (Distribute these across your 4 scripts)
# =========================================================

for SFX in "full"; do
    # run_task "mmlu_rebuttals" "llama8b" "$SFX"
    # run_task "gsm8k_rebuttals" "qwen3b" "$SFX"
    # run_task "medreason_rebuttals" "qwen3b" "$SFX"
    run_task "medreason_rebuttals" "qwen3b" "$SFX"
done

# =========================================================

# --- Crash Report ---
if [[ ${#FAILED_RUNS[@]} -ne 0 ]]; then
    echo -e "\nFAILURES: ${#FAILED_RUNS[@]}"
    printf "  %s\n" "${FAILED_RUNS[@]}"
    exit 1
fi
echo "All runs on GPU ${GPU_NUM} succeeded."