# Final Baseline And Localisation Runners

This folder keeps the rebuttal/final-run scripts that are still represented in
the paper or appendix. Older scratch rescoring jobs and generated logs are
archived under `runner_scripts/_backup_not_committed/`.

## Dense And Interval Restarts

GSM8K Qwen2.5-7B:

```bash
GPU_NUM=0 bash runner_scripts/rebuttal/qwen7b_gsm8k_dense_restart.sh
GPU_NUM=0 bash runner_scripts/rebuttal/qwen7b_gsm8k_interval_fixed_restart.sh
```

MMLU-Pro Llama-3.1-8B:

```bash
GPU_NUM=1 bash runner_scripts/rebuttal/llama8b_mmlu_pro_interval_fixed_restart.sh
GPU_NUM=2 bash runner_scripts/rebuttal/llama8b_mmlu_pro_full_restart.sh
```

MedReason Llama-3.1-8B:

```bash
GPU_NUM=1 bash runner_scripts/rebuttal/llama8b_medreason_interval_fixed_restart.sh
GPU_NUM=2 bash runner_scripts/rebuttal/llama8b_medreason_full_restart.sh
```

The MMLU and MedReason Llama scripts run a fresh 250-step reward-model warmup
before AIRL training and write under the corresponding `/mnt/pdata/.../outputs`
paper directories.

## Warm-Start Variants

The top-level `qwen7b_mmlu_pro_*` and `llama8b_mmlu_pro_*warm*` scripts are
kept for the MMLU warm-start and sparse-reward comparisons reported in the
appendix.

## GAD

Run the Qwen2.5-7B GAD baselines:

```bash
bash runner_scripts/rebuttal/gad/1_gad.sh
bash runner_scripts/rebuttal/gad/2_gad.sh
bash runner_scripts/rebuttal/gad/3_gad.sh
```

The wrappers launch one dataset per GPU:

```text
GPU 1: math
GPU 2: mmlu
GPU 3: medicine
```

## OPSD

Run the non-RL OPSD-token baselines:

```bash
bash runner_scripts/rebuttal/opsd/1_opsd.sh
bash runner_scripts/rebuttal/opsd/2_opsd.sh
bash runner_scripts/rebuttal/opsd/3_opsd.sh
```

These use `opsd.mode=direct`, sample from the current student, and train with
weighted token NLL rather than GRPO.

## Sparse Fixed-Critic RLHF

Run AIRL policy training against a sparse reward model that is warmed up first
and then frozen:

```bash
bash runner_scripts/rebuttal/RLHF/qwen7b_math_sparse_fixed_critic.sh
bash runner_scripts/rebuttal/RLHF/qwen7b_mmlu_sparse_fixed_critic.sh
bash runner_scripts/rebuttal/RLHF/qwen7b_medicine_sparse_fixed_critic.sh
bash runner_scripts/rebuttal/RLHF/qwen4b_math_sparse_fixed_critic.sh
bash runner_scripts/rebuttal/RLHF/llama8b_math_sparse_fixed_critic.sh
```

All scripts accept `GPU_NUM=...`. Override `WARMUP_REWARD_DIR=/path/to/checkpoint`
to use a different sparse critic, or set `WARMUP_REWARD_DIR=none` to force a
fresh sparse warmup.

## Synthetic Localisation

SFT policy-token baselines for the synthetic perturbation experiment:

```bash
GPU_NUM=1 bash runner_scripts/rebuttal/localisation_policy_baselines/run_qwen7b_sft_policy_token_baselines.sh
```

Canonical summaries live under:

```text
localisation/synthetic_perturbations/runs/qwen7b_sft/<model>/<granularity>
```

Original mechanical-perturbation GSM8K scoring:

```bash
GPU_NUM=1 bash runner_scripts/rebuttal/original_synthetic_mistakes/score_original_synthetic_localisation.sh
```

This writes compact scores under:

```text
outputs/gsm8k_process_sensitivity_pregen/rebuttal_scores
```
