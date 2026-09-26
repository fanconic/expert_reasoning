# Learning Reusable Reasoning Critics from Expert-Referenced Supervision

## Abstract
Process-level reward models can score intermediate reasoning, but they typically require step-level annotations, preference data, or external verifiers. We study whether paired expert demonstrations can instead train a reusable reasoning critic without any of these. A naive expert-versus-policy classifier risks learning source or formatting cues instead of reasoning quality. We introduce the Prefix-level Expert-Referenced Critic (PERC), which avoids this by using each expert trace three ways: as a positive example, as the reference answer for weakly labelling policy rollouts by agreement, and as the basis for targeted corruptions used as hard negatives, so that policy traces appear on both sides of the classifier and provenance alone cannot separate positives from negatives. Across GSM8K, MMLU-Pro, and MedReason, this critic captures a reusable reasoning-quality signal. As an **inference-time ranker**, it improves Best-of-16 pass@1 in all main reranking settings, transfers positively across all 27 source--target pairs with gains up to 12.7 points, complements consistency-based voting, and continues to improve pass@1 even for policies already trained with it. As a **process-level evaluator**, prefix-value changes localise human-labelled MedReason error units (two doctors, three final-year medical students) better than likelihood-based baselines and identify controlled GSM8K perturbations with up to $78.9%$ Hit@1. It is also a viable **training signal** for GRPO and remains competitive with imitation baselines without task-specific reward tuning. Overall, PERC shows that weak expert-referenced supervision yields reusable critics for inference-time selection and error diagnosis, while remaining feasible as a policy-training reward.


## Repository Overview

<div align="left">
<img src="./assets/figure_1.png" width="800" alt="Method overview diagram">
</div>

## Layout

- `train_irl.py`, `train_sft.py`, `train_grpo.py`: training entrypoints.
- `evaluate.py`: unified evaluation entrypoint; see `docs/EVALUATION_GUIDE.md`.
- `configs/`: Hydra configs for math, medicine, MMLU, ScienceQA-style runs.
- `src/`: model, reward, training, evaluation, data, and table-generation code.
- `runner_scripts/`: cleaned cluster scripts for paper-scale runs, ablations,
  reranking, and localisation.
- `figures/`: generated tables and plots.
- `localisation/`: process-level localisation experiments.
- `localisation/clinician_medreason/`: cleaned anonymised clinician-labelled
  MedReason localisation study.
- `localisation/synthetic_perturbations/`: controlled GSM8K perturbation
  localisation study from the appendix.

Many paper-scale configs reference cluster paths under `/mnt/pdata/...`.
Override dataset, checkpoint, and output paths when running elsewhere.

## Setup

```bash
conda env create -f environment.yaml
conda activate unsloth_env
```

If you use the current local environment instead of the full conda spec,
`requirements-uv-current.txt` records the packages installed for the latest
analysis runs.

## Training

Single-run examples:

```bash
# Reward model / IRL training
python train_irl.py --config-path=configs/math/qwen7b --config-name=irl_train \
  wandb.run_name=qwen7b_partial_fixed \
  model.dense_rewards=partial_fixed \
  training.output_dir=./outputs/qwen7b_partial_fixed

# Supervised fine-tuning
python train_sft.py --config-path=configs/math/qwen7b --config-name=sft_train \
  wandb.run_name=qwen7b_sft \
  training.output_dir=./outputs/qwen7b_sft

# GRPO baseline
python train_grpo.py --config-path=configs/math/qwen7b --config-name=grpo_train \
  wandb.run_name=qwen7b_grpo \
  training.output_dir=./outputs/qwen7b_grpo
```

Paper-scale sweeps are orchestrated by scripts under `runner_scripts/`; see
`runner_scripts/README.md` for the cleaned map. Legacy scratch launchers are
archived locally under `runner_scripts/_backup_not_committed/`.

## Evaluation And Reranking

```bash
# Reward-model evaluation
python evaluate.py --config-path=configs/math/qwen7b --config-name=irl_eval

# SFT evaluation
python evaluate.py --config-path=configs/math/qwen7b --config-name=sft_eval

# GRPO evaluation
python evaluate.py --config-path=configs/math/qwen7b --config-name=grpo_eval

# Pregenerated completions with policy and reward scores
python evaluate.py --config-path=configs/math/qwen7b --config-name=irl_eval \
  eval.mode=pregenerated_policy_and_reward
```

Main paper table and figure generation:

```bash
python src/plot_generators/plot_main.py \
  --config src/plot_generators/configs/main.yaml \
  --workers 8

python src/plot_generators/plot_transfer.py \
  --config src/plot_generators/configs/transfer.yaml \
  --workers 8
```

## Clinician Localisation

The cleaned MedReason clinician study has one public runner and anonymised JSONL
labels:

```bash
python localisation/clinician_medreason/run_experiment.py
```

Inputs:

- `localisation/clinician_medreason/annotations/expert_*.jsonl`
- `localisation/clinician_medreason/metadata/clinician_cases.jsonl`
- external token-level score JSONLs under the paths documented in
  `localisation/clinician_medreason/README.md`

Outputs:

- `localisation/clinician_medreason/results/clinician_localisation_metrics.md`
- `localisation/clinician_medreason/results/clinician_localisation_metrics.json`

The metric is Hit@1 and Hit@+/-1 over clinician-visible reasoning units, with
no-clear and final-answer-only labels mapped to the final visible reasoning
unit.

## Synthetic Perturbation Localisation

The controlled GSM8K localisation experiment from Section 5.2 and Appendix D.5
is collected under `localisation/synthetic_perturbations/`:

```bash
bash localisation/synthetic_perturbations/run_experiment.sh
```

This regenerates the synthetic Hit@1/Hit@7 tables from the compact run
summaries and writes them to `localisation/synthetic_perturbations/results/`.

## Notes

Large raw JSONL traces, logs, smoke outputs, and score dumps are intentionally
ignored. The repository should keep scripts, configs, compact summaries, and
anonymised labels; regenerate large intermediates from the runner scripts when
needed.
