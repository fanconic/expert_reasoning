# Configs Directory

This folder contains Hydra configs for training and evaluation.

## Structure
- `config_*.yaml`: base/default entry configs used by `train.py`, `train_sft.py`, `train_irl.py`, and `evaluate.py`.
- `<dataset>_rebuttals/<model>/*.yaml`: main experiment configs used for ICML/paper runs.
- `aime/<model>/*.yaml`: AIME-specific training/eval configs.
- `archive/`: legacy configs kept for reproducibility, not recommended as first choice.

## Recommended Starting Points
Use `irl_train.yaml` inside each dataset/model folder for AIRL training baselines.
Examples:
- `configs/gsm8k_rebuttals/qwen3b/irl_train.yaml`
- `configs/medreason_rebuttals/qwen3b/irl_train.yaml`
- `configs/mmlu_rebuttals/qwen3b/irl_train.yaml`

Use `irl_eval.yaml` for AIRL evaluation.

For SFT/GRPO use:
- `sft_train.yaml` + `sft_eval.yaml`
- `grpo_train.yaml` + `grpo_eval.yaml`

## AIRL Segment Critic Experiments
The AIRL segment critic templates live under:
- `configs/math/qwen3b/airl_segment/`
- `configs/math/qwen7b/airl_segment/`

Important flags:
- `model.critic_type=standard|airl_segment`: keep the existing critic or enable the two-head segment critic.
- `model.reward_mode=mean_g|mean_f|mean_g_plus_shape`: choose how segment scores become the policy reward.
- `model.critic_density=sequence|interval|token`: document where the critic produces scores. AIRL segment configs use `interval`.
- `model.policy_reward_density=sequence|interval|token`: choose whether GRPO sees one rollout advantage or interval-local advantages broadcast only to tokens in that interval.
- `model.segment_tokens=15`: fallback fixed segment length when explicit interval boundaries are not used.
- `model.airl_gamma=1.0`: potential-shaping discount in `f = g + gamma * h_next - h_prev`.
- `model.h_head_lr_mult=0.5`: learning-rate multiplier for the potential head.
- `model.h_l2_penalty` and `model.shape_l2_penalty`: small guardrails against the potential absorbing the reward.
- `model.use_segment_local_advantage=false`: optional segment-local token advantage shaping.

Density comparison configs:
- `mean_g.yaml`, `mean_f.yaml`, `mean_g_plus_shape.yaml`: interval critic scores averaged into sequence-level GRPO rewards.
- `interval_mean_g.yaml`, `interval_mean_f.yaml`, `interval_mean_g_plus_shape.yaml`: interval critic scores converted into interval-level GRPO advantages.
- `drop_f_local.yaml`: sequence-level reward with small segment-local drop shaping.
- `interval_drop_f_local.yaml`: interval-level reward with small segment-local drop shaping.

These configs are legacy development templates rather than part of the current
paper/appendix runner set. Their old runner scripts are archived locally under
`runner_scripts/_backup_not_committed/`.

Qwen2.5-7B fixed run set:
- `original_base.yaml`: base policy, original reward.
- `original_sft.yaml`: SFT policy from `SFT_POLICY_DIR`, original reward.
- `mean_g_base.yaml`: base policy, two-head AIRL segment critic, `reward_mode=mean_g`.
- `mean_g_sft.yaml`: SFT policy from `SFT_POLICY_DIR`, two-head AIRL segment critic, `reward_mode=mean_g`.

For DDP throughput runs, the scripts compute gradient accumulation from:
`effective_batch = per_device_batch * gradient_accumulation_steps * nproc_per_node`.
Their defaults are `16 * 8 * 2 = 256`.

See `configs/index.yaml` for a compact canonical mapping used for quick reruns.

## Notes
- Many configs still contain cluster-specific absolute output paths (e.g. `/mnt/pdata/...`).
- For local runs, override paths at launch time, e.g. with Hydra CLI overrides:
  - `training.output_dir=./outputs/<run_name>`
  - `model.warmup_reward_dir=...` (if required)
