#!/usr/bin/env bash
set -euo pipefail

CONFIG_ROOT="configs/math/qwen3b"
OUTPUT_ROOT="${OUTPUT_ROOT:-./outputs/airl_segment_smoke}"

run_smoke() {
  local config_name="$1"
  local run_name="$2"

  python train_irl.py \
    --config-path="${CONFIG_ROOT}" \
    --config-name="airl_segment/${config_name}" \
    training.output_dir="${OUTPUT_ROOT}/${run_name}" \
    training.report_to=none \
    training.max_steps=2 \
    eval.do_eval=false \
    wandb.run_name="${run_name}"
}

run_smoke "stabilized_interval" "smoke_stabilized_interval"
run_smoke "mean_g" "smoke_airl_segment_mean_g"
run_smoke "mean_f" "smoke_airl_segment_mean_f"
run_smoke "mean_g_plus_shape" "smoke_airl_segment_mean_g_plus_shape"
run_smoke "drop_f_local" "smoke_airl_segment_drop_f_local"
run_smoke "interval_mean_g" "smoke_airl_segment_interval_mean_g"
run_smoke "interval_mean_f" "smoke_airl_segment_interval_mean_f"
run_smoke "interval_mean_g_plus_shape" "smoke_airl_segment_interval_mean_g_plus_shape"
run_smoke "interval_drop_f_local" "smoke_airl_segment_interval_drop_f_local"
