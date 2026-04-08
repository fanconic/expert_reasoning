# Evaluation Guide

`evaluate.py` is now the single evaluation entrypoint.  
Legacy scripts (`evaluate_aime.py`, `evaluate_pregenerated.py`, `evaluate_pregenerated_sft.py`) are thin compatibility shims that call `evaluate.py` with `eval.mode=...`.

## Quick Start

```bash
# Standard generation + reward-model scoring
python evaluate.py --config-path=configs/gsm8k_rebuttals/qwen3b --config-name=irl_eval

# AIME-style output naming
python evaluate.py --config-path=configs/aime/qwen3b --config-name=irl_eval eval.mode=aime

# Score pregenerated generations with policy log-probs
python evaluate.py --config-path=configs/gsm8k_rebuttals/qwen3b --config-name=irl_eval eval.mode=pregenerated_policy

# Score pregenerated generations with both policy and reward model
python evaluate.py --config-path=configs/gsm8k_rebuttals/qwen3b --config-name=irl_eval eval.mode=pregenerated_policy_and_reward
```

## Modes

- `generate` (default):
  - Generates with policy model and evaluates with reward model (if enabled).
  - Default output: `eval_results_<dataset>.jsonl`
- `aime`:
  - Same as `generate`, but writes AIME-style filename.
  - Default output: `eval_results_aime.jsonl`
- `pregenerated_policy`:
  - Loads generations from a jsonl and computes policy log-probs.
  - Reward-model scores default to disabled.
  - Default output: `eval_results_logprobs.jsonl`
- `pregenerated_policy_and_reward`:
  - Loads generations from a jsonl and computes both policy log-probs and reward-model scores.
  - Default output: `eval_results_logprobs_sft.jsonl`

## Pregenerated Jsonl Resolution

For pregenerated modes, input jsonl is resolved in this order:

1. `eval.pregenerated_jsonl_path` if set.
2. `eval.pregenerated_source_dir` + candidate filenames.
3. Mode defaults:
   - `pregenerated_policy`: `<model.name>/eval_results_new.jsonl`
   - `pregenerated_policy_and_reward`: `<model.policy_name>/eval_results_new.jsonl`, then `debug.jsonl`, then `eval_results.jsonl`

You can override candidates with:

- `eval.pregenerated_candidates=[eval_results_new.jsonl,debug.jsonl,eval_results.jsonl]`

## Output Naming

Default output files are mode-specific:

- `generate`: `<model.name>/eval_results_<dataset>.jsonl`
- `aime`: `<model.name>/eval_results_aime.jsonl`
- `pregenerated_policy`: `<model.name>/eval_results_logprobs.jsonl`
- `pregenerated_policy_and_reward`: `<model.name>/eval_results_logprobs_sft.jsonl`

Override output path with:

- `eval.output_file=/path/to/custom.jsonl`

## Useful Eval Config Keys

`configs/config_eval.yaml` now includes:

- `eval.mode`
- `eval.pregenerated_jsonl_path`
- `eval.pregenerated_source_dir`
- `eval.pregenerated_candidates`
- `eval.output_file`
- `eval.compute_policy_log_probs`
- `eval.compute_reward_model_scores`

Use `null` for `compute_*` fields to keep mode-based defaults.
