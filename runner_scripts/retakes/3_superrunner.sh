#!/usr/bin/env bash
# Super runner (GPU 3) - balanced workload chunk
set -u

GPU_NUM="3"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

RUNNER="runner_scripts/${GPU_NUM}_run_gpu_node.sh"

# Shared overrides
# - max_steps=280
# - training-time eval every 70 steps
# - gradient accumulation 16
SFT_TRAIN_PARAMS=(
    training.max_steps=400
    eval.eval_steps=100
    training.gradient_accumulation_steps=8
    model.max_prompt_length=300
    model.max_completion_length=824
)
IRL_TRAIN_PARAMS=(
    model.reward_updates_per_policy_step=3
    training.beta=0.1
    training.buffer_size=50
    training.max_steps=400
    eval.eval_steps=100
    model.policy_learning_rate=5e-6
    model.reward_learning_rate=1e-5
    training.gradient_accumulation_steps=8
    model.max_prompt_length=300
    model.max_completion_length=824
)
COMMON_REWARD_FLAGS=(
    model.clip_reward_model=true
    model.reward_lb=-5.0
    model.reward_ub=5.0
)
# Evaluate-only override
EVAL_FLAGS=(
    sampling.temperature=0.5
    model.max_prompt_length=300
    model.max_completion_length=824
)
SFT_LORA_FLAGS=(
    model.lora_rank=256
)
IRL_LORA_FLAGS=(
    model.lora_rank=256
    model.policy_lora_rank=256
    model.reward_lora_rank=256
)
IRL_VARIANTS=(partial_fixed full sparse)

FAILED_RUNS=()

run_cmd() {
    local label="$1"
    shift
    echo -e "\n▶ ${label}\n  $*"
    "$@"
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        FAILED_RUNS+=("${label} (exit=${rc})")
        echo "  ✗ FAILED"
    else
        echo "  ✓ OK"
    fi
}

run_sft_stack() {
    local dataset="$1"  # math | medicine | mmlu
    local model="$2"    # llama3b | llama8b | qwen7b | qwen3b | qwen4b
    local run_name="${model}_sft"
    local wb_project="neurips_airl_${dataset}"
    local output_dir="/mnt/pdata/caf83/neurips2026/${dataset}/outputs/${run_name}"
    local sft_trace_file="${output_dir}/best_model/eval_results_${dataset}_${model}_sft_t0p5.jsonl"

    run_cmd "${run_name}_TRAIN" \
        bash "${RUNNER}" train_sft.py \
            --config-path="configs/${dataset}/${model}" \
            --config-name="sft_train" \
            wandb.run_name="${run_name}" \
            wandb.project="${wb_project}" \
            training.output_dir="${output_dir}" \
            "${SFT_LORA_FLAGS[@]}" \
            "${SFT_TRAIN_PARAMS[@]}"

    run_cmd "${run_name}_EVAL" \
        bash "${RUNNER}" evaluate.py \
            --config-path="configs/${dataset}/${model}" \
            --config-name="sft_eval" \
            wandb.run_name="${run_name}" \
            wandb.project="${wb_project}" \
            model.name="${output_dir}/best_model" \
            "${SFT_LORA_FLAGS[@]}" \
            "${EVAL_FLAGS[@]}" \
            ++eval.output_file="${sft_trace_file}"
}

run_irl_scoring_on_sft_traces() {
    local dataset="$1"
    local model="$2"
    local suffix="$3"
    local dense_val="$4"
    local irl_output_dir="$5"

    local run_name="${model}_${suffix}"
    local wb_project="neurips_airl_${dataset}"
    local sft_run_name="${model}_sft"
    local sft_policy_dir="/mnt/pdata/caf83/neurips2026/${dataset}/outputs/${sft_run_name}/best_model"
    local sft_trace_file="${sft_policy_dir}/eval_results_${dataset}_${model}_sft_t0p5.jsonl"
    local score_output_file="${irl_output_dir}/best_model/eval_results_${dataset}_${model}_${suffix}_on_sft_t0p5_policy_reward.jsonl"

    if [[ ! -f "${sft_trace_file}" ]]; then
        local missing_label="${run_name}_EVAL_SFT_TRACES"
        echo "  ✗ SKIP: Missing SFT trace file ${sft_trace_file}"
        FAILED_RUNS+=("${missing_label} (missing_sft_trace)")
        return
    fi

    run_cmd "${run_name}_EVAL_SFT_TRACES" \
        bash "${RUNNER}" evaluate.py \
            --config-path="configs/${dataset}/${model}" \
            --config-name="irl_eval" \
            wandb.run_name="${run_name}" \
            wandb.project="${wb_project}" \
            model.name="${irl_output_dir}/best_model" \
            model.policy_name="${sft_policy_dir}" \
            model.dense_rewards="${dense_val}" \
            "${IRL_LORA_FLAGS[@]}" \
            "${COMMON_REWARD_FLAGS[@]}" \
            "${EVAL_FLAGS[@]}" \
            ++eval.mode=pregenerated_policy_and_reward \
            ++eval.pregenerated_jsonl_path="${sft_trace_file}" \
            ++eval.compute_policy_log_probs=true \
            ++eval.compute_reward_model_scores=true \
            ++eval.output_file="${score_output_file}"
}

run_irl_variant() {
    local dataset="$1"
    local model="$2"
    local suffix="$3"   # partial | full | sparse

    local dense_val="${suffix}"
    [[ "${suffix}" == "sparse" ]] && dense_val="false"

    local run_name="${model}_${suffix}"
    local wb_project="neurips_airl_${dataset}"
    local output_dir="/mnt/pdata/caf83/neurips2026/${dataset}/outputs/${run_name}"

    # medicine should warmup from scratch again (no warmup checkpoint override),
    # and use fewer reward warmup steps.
    local warmup_override=()
    warmup_override+=(training.reward_warmup_steps=250)

    run_cmd "${run_name}_TRAIN" \
        bash "${RUNNER}" train_irl.py \
            --config-path="configs/${dataset}/${model}" \
            --config-name="irl_train" \
            wandb.run_name="${run_name}" \
            wandb.project="${wb_project}" \
            training.output_dir="${output_dir}" \
            model.dense_rewards="${dense_val}" \
            "${IRL_LORA_FLAGS[@]}" \
            "${COMMON_REWARD_FLAGS[@]}" \
            "${IRL_TRAIN_PARAMS[@]}" \
            "${warmup_override[@]}"

    run_cmd "${run_name}_EVAL" \
        bash "${RUNNER}" evaluate.py \
            --config-path="configs/${dataset}/${model}" \
            --config-name="irl_eval" \
            wandb.run_name="${run_name}" \
            wandb.project="${wb_project}" \
            model.name="${output_dir}/best_model" \
            model.dense_rewards="${dense_val}" \
            "${IRL_LORA_FLAGS[@]}" \
            "${COMMON_REWARD_FLAGS[@]}" \
            "${EVAL_FLAGS[@]}"

    #run_irl_scoring_on_sft_traces "${dataset}" "${model}" "${suffix}" "${dense_val}" "${output_dir}"
}

# run_combo() {
#     local dataset="$1"
#     local model="$2"
#     local suffix="$3"
#     run_sft_stack "${dataset}" "${model}"
#     run_irl_variant "${dataset}" "${model}" "${suffix}"
# }

# Balanced set for GPU 0.
# Keep qwen3b/llama3b at the end as requested.
# run_irl_variant "medicine" "qwen4b" "sparse"
# run_irl_variant "medicine" "qwen7b" "sparse"
# run_irl_variant "medicine" "llama8b" "sparse"
run_irl_variant "medicine" "qwen7b" "full"

echo -e "\n======================\nGPU ${GPU_NUM} SUMMARY\n======================"
if [[ ${#FAILED_RUNS[@]} -ne 0 ]]; then
    echo -e "FAILURES: ${#FAILED_RUNS[@]}"
    printf "  %s\n" "${FAILED_RUNS[@]}"
    exit 1
fi
echo "All runs on GPU ${GPU_NUM} succeeded."
