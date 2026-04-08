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

See `configs/index.yaml` for a compact canonical mapping used for quick reruns.

## Notes
- Many configs still contain cluster-specific absolute output paths (e.g. `/mnt/pdata/...`).
- For local runs, override paths at launch time, e.g. with Hydra CLI overrides:
  - `training.output_dir=./outputs/<run_name>`
  - `model.warmup_reward_dir=...` (if required)
