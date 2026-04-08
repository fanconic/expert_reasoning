# Learning Reasoning Reward Models from Expert Demonstration via Inverse Reinforcement Learning

<div align="left">
<img src="./assets/figure_1.png" width="800" alt="Method overview diagram">
</div>

## Abstract
Current methods for training reasoning capabilities in large language models (LLMs) primarily rely
on supervised fine-tuning (SFT) to imitate expert traces or on reinforcement learning (RL) guided by
outcome rewards. However, SFT focuses on imitation rather than exploration and generalisation, while
outcome-based RL depends heavily on strictly defined reward functions. To address these limitations, we
introduce an adversarial inverse reinforcement learning (IRL) framework that directly extracts reasoning
reward models at various granularities (sparse, step-wise, and token-level) from expert demonstrations.
We demonstrate that this learned reward offers dual utility. First, it provides a robust training signal,
enabling policies to significantly outperform standard SFT baselines across multiple datasets and models.
Second, it functions as a highly effective inference-time reranker, boosting performance by up to 25
percentage points. Furthermore, our reasoning rewards can serve as a process supervisor that pinpoints
the location of logical errors. This data-driven framework bridges the gap between pure imitation and
traditional RL, advancing process-level reasoning in LLMs.

## What This Repo Covers
- AIRL-style reasoning reward learning (sparse / partial-step / interval / dense variants)
- Policy training baselines: AIRL, SFT, GRPO
- Evaluation and reranking analyses for GSM8K, MedReason, MMLU(-Pro), and AIME variants
- Plot/table generation for pass@k, reranking, calibration, and token-level diagnostics

## Repository Layout
- `train_irl.py`, `train_sft.py`, `train.py`: training entrypoints
- `evaluate.py`: unified evaluation entrypoint (see `docs/EVALUATION_GUIDE.md`)
- `configs/`: training/eval configs (see `configs/README.md`)
- `src/`: implementation modules (models, training, rewards, plotting, data)
- `src/plot_generators/configs/`: YAML specs for plotting runs
- `figures/`: generated outputs and historical artifacts (see `figures/README.md`)
- `docs/REPO_CLEANUP_PLAN.md`: cleanup roadmap and design decisions

## Setup
```bash
conda env create -f environment.yaml
conda activate unsloth_env
```

## Data and Paths
Many configs reference cluster paths under `/mnt/pdata/...`. For local runs, override paths at launch time (especially `training.output_dir` and any dataset/model path overrides).

## Training
Examples (Hydra-based):

```bash
# AIRL (example: GSM8K Qwen3B)
python train_irl.py --config-path=configs/gsm8k_rebuttals/qwen3b --config-name=irl_train

# SFT
python train_sft.py --config-path=configs/gsm8k_rebuttals/qwen3b --config-name=sft_train

# GRPO
python train_grpo.py --config-path=configs/gsm8k_rebuttals/qwen3b --config-name=grpo_train
```

## Evaluation
```bash
# AIRL evaluation
python evaluate.py --config-path=configs/gsm8k_rebuttals/qwen3b --config-name=irl_eval

# SFT evaluation
python evaluate.py --config-path=configs/gsm8k_rebuttals/qwen3b --config-name=sft_eval

# GRPO evaluation
python evaluate.py --config-path=configs/gsm8k_rebuttals/qwen3b --config-name=grpo_eval

# AIME-style output filename (legacy evaluate_aime behavior)
python evaluate.py --config-path=configs/aime/qwen3b --config-name=irl_eval eval.mode=aime

# Pregenerated completions + policy log-probs (legacy evaluate_pregenerated behavior)
python evaluate.py --config-path=configs/gsm8k_rebuttals/qwen3b --config-name=irl_eval eval.mode=pregenerated_policy

# Pregenerated completions + policy + reward model (legacy evaluate_pregenerated_sft behavior)
python evaluate.py --config-path=configs/gsm8k_rebuttals/qwen3b --config-name=irl_eval eval.mode=pregenerated_policy_and_reward
```

See `docs/EVALUATION_GUIDE.md` for full mode details, jsonl input resolution rules, and output naming.

## Plot and Table Generation
Plot scripts now use external YAML specs instead of hardcoded experiment lists.

```bash
# Main plotting runs
python src/plot_generators/plot_main.py \
  --config src/plot_generators/configs/main.yaml \
  --workers 8

# Transferability plotting runs
python src/plot_generators/plot_transfer.py \
  --config src/plot_generators/configs/transfer.yaml \
  --workers 8
```

Useful flags:
- `--ckpt <name>`: override checkpoint folder (default: `best_model`)
- `--output-root <path>`: override output root directory
- `--no-token-figs`: skip expensive token-level visualizations
- `--debug`: run sequentially (easier debugging)
