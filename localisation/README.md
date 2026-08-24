# Localisation Rebuttal Artifacts

This folder contains the GSM8K localisation artifacts used for the rebuttal
analysis. The committed files are the small, human-readable outputs: run
configs, summary JSON files, generated LaTeX tables, and this documentation.
Large raw traces (`pair_details.jsonl`, policy-token JSONL scores, smoke files,
logs, and figures) are kept on disk for inspection and regeneration, but are
ignored by git.

## Layout

- `*_localisation_from_qwen7b_sft/`: reward-model localisation on the original
  pregenerated Qwen2.5-7B SFT synthetic perturbation pairs.
- `*_localisation_expert/`: reward-model localisation on perturbations built
  from expert traces.
- `chatgpt_step_perturbations/`: LLM-edited fluent-but-wrong step perturbations
  derived from Qwen2.5-7B SFT traces, plus reward and policy localisation
  summaries.
- `expert_step_perturbations/`: LLM-edited perturbations derived from expert
  traces, with matching localisation summaries.
- `natural_wrong_sft/`: naturally wrong Qwen2.5-7B SFT generations whose first
  erroneous step was labelled by an LLM, then scored with the same localisation
  pipeline.
- `recalc32_*`: diagnostic dense-reward recalculations and visualization
  metadata for the reviewer-facing examples.

## Main Artifact Types

- `run_config.json`: exact model, checkpoint, perturbation, and scoring options
  for a run.
- `summary.json`: aggregate localisation metrics such as Hit@1 within a token
  window, MAP/MRR-style scores, and bootstrap intervals.
- `policy_token_baselines_summary.json`: generator-side log-probability,
  probability, entropy, and random-location baselines.
- `*.tex`: final LaTeX table fragments used in the rebuttal.
- `*.json`: machine-readable companions for tables or diagnostic summaries.

## Regeneration

The runner scripts live under `runner_scripts/rebuttal/`:

- `chatgpt_step_perturbations/` builds and scores LLM-edited SFT-step pairs.
- `expert_step_perturbations/` builds LLM-edited expert-step pairs.
- `natural_wrong_sft/` labels naturally wrong SFT traces.
- `natural_error_scoring/` scores expert and naturally wrong sets.
- `localisation_policy_baselines/` runs generator-token baselines.
- `original_synthetic_mistakes/` rescoring for the original synthetic set.

The table builders are in `src/plot_generators/` and write their outputs back
into this folder by default.
