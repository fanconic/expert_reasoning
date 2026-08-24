# Rebuttal runners

This folder contains focused restart scripts for rebuttal experiments.

## GSM8K Qwen2.5-7B Dense

Run the dense/full AIRL restart on a specific GPU:

```bash
GPU_NUM=0 bash runner_scripts/rebuttal/qwen7b_gsm8k_dense_restart.sh
```

The script uses the `runner_scripts/retakes` training/evaluation overrides and writes to:

```text
/mnt/pdata/caf83/neurips2026/math/outputs/qwen7b_full_rebuttal_restart
```

W&B logs go to project:

```text
neurips_airl_rebuttal_math
```

## GSM8K Qwen2.5-7B Fixed-Interval

Run the fixed-interval AIRL restart on a specific GPU:

```bash
GPU_NUM=0 bash runner_scripts/rebuttal/qwen7b_gsm8k_interval_fixed_restart.sh
```

This sets:

```text
model.dense_rewards=partial_fixed
model.dense_partial_fixed_n=15
```

and writes to:

```text
/mnt/pdata/caf83/neurips2026/math/outputs/qwen7b_partial_fixed_rebuttal_restart
```

W&B logs go to project:

```text
neurips_airl_rebuttal_math
```

## MMLU-Pro Llama-3.1-8B Restarts

Run the fixed-interval restart on a specific GPU:

```bash
GPU_NUM=1 bash runner_scripts/rebuttal/llama8b_mmlu_pro_interval_fixed_restart.sh
```

Run the full dense restart on a specific GPU:

```bash
GPU_NUM=2 bash runner_scripts/rebuttal/llama8b_mmlu_pro_full_restart.sh
```

Both scripts run a fresh 250-step reward-model warmup before AIRL training,
write under:

```text
/mnt/pdata/caf83/neurips2026/mmlu/outputs
```

and log to W&B project:

```text
neurips_airl_rebuttal_mmlu
```

## MedReason Llama-3.1-8B Restarts

Run the fixed-interval restart on a specific GPU:

```bash
GPU_NUM=1 bash runner_scripts/rebuttal/llama8b_medreason_interval_fixed_restart.sh
```

Run the full dense restart on a specific GPU:

```bash
GPU_NUM=2 bash runner_scripts/rebuttal/llama8b_medreason_full_restart.sh
```

Both scripts use `configs/medicine/llama8b`, run a fresh 250-step reward-model
warmup before AIRL training, write under:

```text
/mnt/pdata/caf83/neurips2026/medicine/outputs
```

and log to W&B project:

```text
neurips_airl_rebuttal_medicine
```

## Qwen2.5-7B GAD

Run the GAD rebuttal baselines on GPUs 1, 2, and 3:

```bash
bash runner_scripts/rebuttal/gad/1_gad.sh
bash runner_scripts/rebuttal/gad/2_gad.sh
bash runner_scripts/rebuttal/gad/3_gad.sh
```

The wrappers launch one dataset per GPU with the same default seed:

```text
GPU 1: math     -> qwen7b_gad_math
GPU 2: mmlu     -> qwen7b_gad_mmlu
GPU 3: medicine -> qwen7b_gad_medicine
```

W&B logs go to project:

```text
neurips_airl_rebuttal_gad_<dataset>
```

## Qwen2.5-7B OPSD

Run the non-RL OPSD-token rebuttal baselines on GPUs 1, 2, and 3:

```bash
bash runner_scripts/rebuttal/opsd/1_opsd.sh
bash runner_scripts/rebuttal/opsd/2_opsd.sh
bash runner_scripts/rebuttal/opsd/3_opsd.sh
```

The wrappers launch one dataset per GPU with the same default seed:

```text
GPU 1: math     -> qwen7b_opsd_math
GPU 2: mmlu     -> qwen7b_opsd_mmlu
GPU 3: medicine -> qwen7b_opsd_medicine
```

W&B logs go to project:

```text
neurips_airl_rebuttal_opsd_<dataset>
```

The scripts use `opsd.mode=direct`, which samples from the current student and
trains with weighted token NLL rather than GRPO.

## Sparse Fixed-Critic RLHF

Run AIRL policy training against a sparse reward model that is warmed up first
and then frozen. The scripts set:

```text
model.dense_rewards=false
model.reward_updates_per_policy_step=0
training.freeze_reward_after_warmup=true
```

Launch the five rebuttal runs with:

```bash
bash runner_scripts/rebuttal/RLHF/qwen7b_math_sparse_fixed_critic.sh
bash runner_scripts/rebuttal/RLHF/qwen7b_mmlu_sparse_fixed_critic.sh
bash runner_scripts/rebuttal/RLHF/qwen7b_medicine_sparse_fixed_critic.sh
bash runner_scripts/rebuttal/RLHF/qwen4b_math_sparse_fixed_critic.sh
bash runner_scripts/rebuttal/RLHF/llama8b_math_sparse_fixed_critic.sh
```

All scripts accept `GPU_NUM=...` and log to:

```text
neurips_airl_rebuttal_rlhf_<dataset>
```

By default, the math and medicine scripts load existing sparse BCE warmup
reward checkpoints. The MMLU Qwen2.5-7B run intentionally does a fresh sparse
warmup before freezing the critic. Override any script with
`WARMUP_REWARD_DIR=/path/to/reward_model_warmup` to use a different critic, or
set `WARMUP_REWARD_DIR=none` to force a fresh sparse warmup.

## SFT Policy Token Localisation

Run the SFT policy-token localisation baselines on a specific GPU:

```bash
GPU_NUM=1 bash runner_scripts/rebuttal/localisation_policy_baselines/run_qwen7b_sft_policy_token_baselines.sh
```

By default this covers `qwen7b`, `qwen4b`, and `llama8b` SFT policies plus the
matching base instruct models against the qwen7b-SFT pregenerated localisation
folders. Canonical run folders live under:

```text
localisation/runs/qwen7b_sft/<model>/<granularity>
```

The old long top-level folder names are symlinks kept for compatibility.
Existing summaries are skipped unless `FORCE=1` is set.

To run only the base model token-probability/log-probability scores:

```bash
GPU_NUM=1 POLICY_KEYS="qwen7b_base qwen4b_base llama8b_base" bash runner_scripts/rebuttal/localisation_policy_baselines/run_qwen7b_sft_policy_token_baselines.sh
```

On machines without an active Conda environment, the script bootstraps itself
with `uv run` when the repo `.venv` is available.

For qwen7b/full reward-model localisation, the localiser scripts default to:

```text
/mnt/pdata/caf83/neurips2026/math/outputs/qwen7b_full_rebuttal_restart/checkpoint-100
```

Override `QWEN7B_FULL_REWARD_CHECKPOINT` if a later restart checkpoint should be
used.

## Original Synthetic Mistake Localisation

Use this for the original mechanical-perturbation GSM8K synthetic set:

```bash
GPU_NUM=1 bash runner_scripts/rebuttal/original_synthetic_mistakes/score_original_synthetic_localisation.sh
```

It scores:

```text
outputs/gsm8k_process_sensitivity_pregen/pair_details.jsonl
```

and writes under:

```text
outputs/gsm8k_process_sensitivity_pregen/rebuttal_scores
```

By default it runs the three base policy models (`qwen7b_base`, `qwen4b_base`,
`llama8b_base`) plus qwen7b/full dense reward scoring with the rebuttal restart
checkpoint above.

## ChatGPT-Step Synthetic Mistake Localisation

Use the ChatGPT-step perturbation scorers for synthetic mistakes:

```bash
bash runner_scripts/rebuttal/chatgpt_step_perturbations/0_score_chatgpt_step_localisation.sh
bash runner_scripts/rebuttal/chatgpt_step_perturbations/1_score_chatgpt_step_localisation.sh
```

These scripts score:

```text
localisation/chatgpt_step_perturbations/gsm8k_qwen7b_sft_step_perturbations_full.jsonl
```

not the natural-trace `pair_details.jsonl` files. They include base policy
models and qwen7b/full dense reward scoring with the rebuttal restart
checkpoint above.
