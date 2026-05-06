#!/usr/bin/env bash
# Super runner (GPU 1) - balanced workload chunk
set -u

GPU_NUM="2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

RUNNER="runner_scripts/${GPU_NUM}_run_gpu_node.sh"

GRPO_TRAIN_PARAMS=(
    training.max_steps=400
    eval.eval_steps=100
    training.gradient_accumulation_steps=8
    training.learning_rate=5e-6
    model.max_prompt_length=300
    model.max_completion_length=824
)
# Evaluate-only override
EVAL_FLAGS=(
    sampling.temperature=0.5
    model.max_prompt_length=300
    model.max_completion_length=824
)
GRPO_LORA_FLAGS=(
    model.lora_rank=256
)

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

run_grpo_stack() {
    local dataset="$1"  # math | medicine | mmlu
    local model="$2"    # llama3b | llama8b | qwen7b | qwen3b | qwen4b
    local run_name="${model}_grpo"
    local wb_project="neurips_airl_${dataset}"
    local output_dir="/mnt/pdata/caf83/neurips2026/${dataset}/outputs/${run_name}"
    local grpo_trace_file="${output_dir}/best_model/eval_results_${dataset}_${model}_grpo_t0p5.jsonl"

    run_cmd "${run_name}_TRAIN" \
        bash "${RUNNER}" train_grpo.py \
            --config-path="configs/${dataset}/${model}" \
            --config-name="grpo_train" \
            wandb.run_name="${run_name}" \
            wandb.project="${wb_project}" \
            training.output_dir="${output_dir}" \
            "${GRPO_LORA_FLAGS[@]}" \
            "${GRPO_TRAIN_PARAMS[@]}"

    run_cmd "${run_name}_EVAL" \
        bash "${RUNNER}" evaluate.py \
            --config-path="configs/${dataset}/${model}" \
            --config-name="grpo_eval" \
            wandb.run_name="${run_name}" \
            wandb.project="${wb_project}" \
            model.name="${output_dir}/best_model" \
            "${GRPO_LORA_FLAGS[@]}" \
            "${EVAL_FLAGS[@]}" \
            ++eval.output_file="${grpo_trace_file}"
}




# Balanced set for GPU 0.
# Keep qwen3b/llama3b at the end as requested.
run_grpo_stack "medicine" "qwen4b"

echo -e "\n======================\nGPU ${GPU_NUM} SUMMARY\n======================"
if [[ ${#FAILED_RUNS[@]} -ne 0 ]]; then
    echo -e "FAILURES: ${#FAILED_RUNS[@]}"
    printf "  %s\n" "${FAILED_RUNS[@]}"
    exit 1
fi
echo "All runs on GPU ${GPU_NUM} succeeded."

bash runner_scripts/transferability_temp05/2_runner.sh