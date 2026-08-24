# Natural Wrong SFT Traces

This subfolder contains localisation artifacts for naturally wrong
Qwen2.5-7B SFT GSM8K generations. Instead of injecting a synthetic edit, an LLM
labels the first mathematically invalid reasoning step in an already-wrong SFT
completion. Reward and policy-token scorers then try to localise that step.

Important files:

- `gsm8k_qwen7b_sft_wrong_step_labels_full.jsonl`: full labelled set kept
  locally but ignored by git.
- `scores/`: reward-model and policy-token localisation summaries.
- `localisation_natural_wrong_sft_*`: final table fragments and
  machine-readable metrics.

Generation scripts:

- `src/eval/label_natural_wrong_sft_steps.py`
- `runner_scripts/rebuttal/natural_wrong_sft/label_qwen7b_sft_wrong_steps.sh`

Scoring and tables:

- `runner_scripts/rebuttal/natural_error_scoring/score_expert_and_wrong_sft_one_gpu.sh`
- `src/eval/localisation_reward_on_pairs.py`
- `src/eval/localisation_policy_token_baselines.py`
- `src/plot_generators/table_natural_error_fair_grid_metrics.py`
- `src/plot_generators/table_natural_wrong_sft_map_only.py`
