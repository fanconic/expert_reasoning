# ChatGPT-Step Perturbations

This subfolder holds the rebuttal localisation set where an Azure OpenAI model
rewrote one reasoning step in a Qwen2.5-7B SFT GSM8K trace so that the step is
fluent but mathematically wrong.

Important files:

- `gsm8k_qwen7b_sft_step_perturbations_full.jsonl`: full generated pair set
  kept locally but ignored by git.
- `scores/`: reward-model and policy-token localisation summaries.
- `localisation_chatgpt_step_*.tex`: table fragments generated from the
  summaries.
- `localisation_chatgpt_step_*metrics.json`: machine-readable table inputs.

Generation scripts:

- `src/eval/build_chatgpt_step_perturbation_dataset.py`
- `runner_scripts/rebuttal/chatgpt_step_perturbations/build_full_gsm8k_step_perturbations.sh`

Scoring and tables:

- `src/eval/localisation_reward_on_pairs.py`
- `src/eval/localisation_policy_token_baselines.py`
- `src/plot_generators/table_chatgpt_step_localisation.py`
- `src/plot_generators/table_chatgpt_step_fair_grid_metrics.py`
- `src/plot_generators/table_chatgpt_step_extra_metrics.py`
