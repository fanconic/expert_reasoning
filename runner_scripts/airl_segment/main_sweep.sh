#!/usr/bin/env bash
set -euo pipefail

CONFIG_ROOT="configs/math/qwen3b"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/pdata/caf83/airl_segment/outputs}"
PROJECT="${PROJECT:-expert_reasoning_airl_segment}"

POLICY_LRS=(5e-7 1e-6)
REWARD_LRS=(1e-6 5e-6)
BETAS=(0.001 0.003 0.01)
REWARD_MODES=(mean_g mean_f mean_g_plus_shape)
POLICY_REWARD_DENSITIES=(sequence interval)
LAMBDA_SHAPES=(0.1 0.3 1.0)

run_train() {
  local config_name="$1"
  local run_name="$2"
  shift 2

  python train_irl.py \
    --config-path="${CONFIG_ROOT}" \
    --config-name="airl_segment/${config_name}" \
    wandb.project="${PROJECT}" \
    wandb.run_name="${run_name}" \
    training.output_dir="${OUTPUT_ROOT}/${run_name}" \
    "$@"
}

for policy_lr in "${POLICY_LRS[@]}"; do
  for reward_lr in "${REWARD_LRS[@]}"; do
    for beta in "${BETAS[@]}"; do
      baseline_name="qwen3b_stabilized_interval_plr${policy_lr}_rlr${reward_lr}_b${beta}"
      run_train "stabilized_interval" "${baseline_name}" \
        model.policy_learning_rate="${policy_lr}" \
        model.reward_learning_rate="${reward_lr}" \
        training.beta="${beta}"

      for reward_mode in "${REWARD_MODES[@]}"; do
        for density in "${POLICY_REWARD_DENSITIES[@]}"; do
          config_name="${reward_mode}"
          [[ "${density}" == "interval" ]] && config_name="interval_${reward_mode}"

          if [[ "${reward_mode}" == "mean_g_plus_shape" ]]; then
            for lambda_shape in "${LAMBDA_SHAPES[@]}"; do
              run_name="qwen3b_airl_segment_${density}_${reward_mode}_ls${lambda_shape}_plr${policy_lr}_rlr${reward_lr}_b${beta}"
              run_train "${config_name}" "${run_name}" \
                model.policy_learning_rate="${policy_lr}" \
                model.reward_learning_rate="${reward_lr}" \
                training.beta="${beta}" \
                model.lambda_shape="${lambda_shape}"
            done
          else
            run_name="qwen3b_airl_segment_${density}_${reward_mode}_plr${policy_lr}_rlr${reward_lr}_b${beta}"
            run_train "${config_name}" "${run_name}" \
              model.policy_learning_rate="${policy_lr}" \
              model.reward_learning_rate="${reward_lr}" \
              training.beta="${beta}"
          fi
        done
      done

      for lambda_local in 0.05 0.1; do
        run_name="qwen3b_airl_segment_sequence_drop_f_ll${lambda_local}_plr${policy_lr}_rlr${reward_lr}_b${beta}"
        run_train "drop_f_local" "${run_name}" \
          model.policy_learning_rate="${policy_lr}" \
          model.reward_learning_rate="${reward_lr}" \
          training.beta="${beta}" \
          model.lambda_local="${lambda_local}"

        run_name="qwen3b_airl_segment_interval_drop_f_ll${lambda_local}_plr${policy_lr}_rlr${reward_lr}_b${beta}"
        run_train "interval_drop_f_local" "${run_name}" \
          model.policy_learning_rate="${policy_lr}" \
          model.reward_learning_rate="${reward_lr}" \
          training.beta="${beta}" \
          model.lambda_local="${lambda_local}"
      done
    done
  done
done
