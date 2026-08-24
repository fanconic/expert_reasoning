# Expert-Step Perturbations

This subfolder mirrors `chatgpt_step_perturbations/`, but starts from expert
GSM8K traces rather than Qwen2.5-7B SFT generations. It is used to test whether
the localisation signal remains meaningful when the clean trace is the expert
demonstration itself.

Important files:

- `gsm8k_expert_step_perturbations_full.jsonl`: full generated expert-step pair
  set kept locally but ignored by git.
- `scores/`: reward-model and policy-token localisation summaries.
- `localisation_expert_step_fair_grid_metrics.*`: final table artifacts.

Generation scripts:

- `src/eval/build_chatgpt_expert_step_perturbation_dataset.py`
- `runner_scripts/rebuttal/expert_step_perturbations/build_full_gsm8k_expert_step_perturbations.sh`

Scoring and tables:

- `src/eval/localisation_reward_on_pairs.py`
- `src/eval/localisation_policy_token_baselines.py`
- `src/plot_generators/table_chatgpt_step_fair_grid_metrics.py`
