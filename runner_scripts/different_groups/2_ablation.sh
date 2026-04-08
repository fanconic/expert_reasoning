set -u 

export GPU_NUM="2" 
RUNNER="runner_scripts/${GPU_NUM}_run_gpu_node.sh"

# Note: Removed fixed batch/grad from TRAIN_PARAMS to set them in the function
WARMUP_DIR="/mnt/pdata/caf83/icml_math/warmed_up_rewards/llama3b/partial/"
BASE_TRAIN_PARAMS="model.reward_updates_per_policy_step=3 training.beta=0.1 training.max_steps=400 training.buffer_size=50 model.warmup_reward_dir=${WARMUP_DIR}"
COMMON_REWARD_FLAGS="model.reward_lb=-5.0 model.reward_ub=5.0"

run_ablation() {
    local MODEL="$1"
    local G="$2"  # num_generations
    
    local WNAME="${MODEL}_partial_G${G}"
    
    # We are specifically doing "partial" for this ablation
    local OVERRIDE="wandb.run_name=${WNAME} model.dense_rewards=partial ${COMMON_REWARD_FLAGS}"
    
    # Ablation-specific overrides
    local ABLATION_FLAGS="sampling.num_generations=${G}"

    echo "▶ Starting Ablation: ${WNAME}"
    
    # 1. TRAIN
    bash "$RUNNER" train_irl.py \
        --config-path="configs/gsm8k_rebuttals/${MODEL}" \
        --config-name="good_run" \
        $OVERRIDE $BASE_TRAIN_PARAMS $ABLATION_FLAGS

    # 2. EVAL (Note: evaluate.py usually doesn't need grad_accum or batch_size unless memory is tight)
    bash "$RUNNER" evaluate.py \
        --config-path="configs/gsm8k_rebuttals/${MODEL}" \
        --config-name="eval" \
        $OVERRIDE
}

# --- Execution ---


# Llama 3B Ablations
run_ablation "llama3b" 4
bash runner_scripts/betas/2_ablation.sh

bash runner_scripts/corruption/2_ablation.sh
