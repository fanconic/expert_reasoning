# #!/usr/bin/env bash
set -u
export GPU_NUM="1"       # Set to 0, 1, or 2
export MODEL="qwen4b"
DATASET="gsm8k_rebuttals" # Set to gsm8k_rebuttals, medreason_rebuttals, or mmlu_rebuttals

# --- Shared Parameters ---
STEP_LIMIT="training.max_steps=400"
COMMON_REWARD_FLAGS="model.reward_lb=-5.0 model.reward_ub=5.0"
# IRL-specific params
IRL_PARAMS="model.reward_updates_per_policy_step=3 training.beta=0.1 training.buffer_size=50"

FAILED_RUNS=()
run_cmd() {
    local label="$1"; shift
    echo -e "\n▶ Starting: $label"
    "$@"
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        FAILED_RUNS+=("$label (exit=$rc)")
        echo "  ✗ FAILED"
    else
        echo "  ✓ OK"
    fi
}

# # 1. SFT (Now with max_steps)
# run_cmd "${DATASET}_SFT" bash runner_scripts/${GPU_NUM}_run_gpu_node.sh train_sft.py \
#     --config-path=configs/${DATASET}/${MODEL} --config-name=sft_train $STEP_LIMIT
run_cmd "${DATASET}_SFT_EVAL" bash runner_scripts/${GPU_NUM}_run_gpu_node.sh evaluate.py \
    --config-path=configs/${DATASET}/${MODEL} --config-name=sft_eval