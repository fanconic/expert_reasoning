# MedReason Clinician Localisation

This directory is the cleaned, anonymised MedReason first-error localisation
study used in the paper. Medical experts labelled where an incorrect
Qwen2.5-7B SFT reasoning trace first becomes medically incorrect,
unsupported, contradictory, or misleading.

## Contents

- `run_experiment.py`: single entrypoint for the clinician localisation metrics.
- `annotations/expert_*.jsonl`: anonymised returned labels, one file per expert.
- `annotations/manifest.json`: label schema and annotation counts.
- `metadata/clinician_cases.jsonl`: clinician-visible cases plus hidden source
  metadata needed to align labels with token-level scores.
- `results/clinician_localisation_metrics.{md,json}`: current paper metrics.
- `appendix_assets/clinician_labelling_interface_example.svg`: example
  labelling interface for the appendix.

The old personal-name annotation files, offline HTML packages, and per-doctor
intermediate score dumps have intentionally been removed from this cleaned
paper directory.

## Run

From the repository root:

```bash
python localisation/clinician_medreason/run_experiment.py
```

This writes:

```text
localisation/clinician_medreason/results/clinician_localisation_metrics.md
localisation/clinician_medreason/results/clinician_localisation_metrics.json
```

Use `--keep-details` to also write one per-case prediction row per signal.

## External Score Files

The annotations and metadata are stored here, but the token-level model scores
are large 16k-row JSONL files and remain outside the repo. The default paths are
the paper paths:

```text
/mnt/pdata/caf83/neurips2026/transfer_rerank_corrected_qwen7b_sft/
/mnt/pdata/caf83/neurips2026/medreason_full_dense_reward_scores_qwen7b_sft/
```

The Qwen2.5-7B SFT log-probability and entropy baseline is also treated as an
external score file; pass it with `--policy-jsonl` when the local trace-scoring
dump is not present. Override paths when running elsewhere:

```bash
python localisation/clinician_medreason/run_experiment.py \
  --interval-root /path/to/transfer_rerank_corrected_qwen7b_sft \
  --dense-root /path/to/medreason_full_dense_reward_scores_qwen7b_sft \
  --policy-jsonl /path/to/qwen7b_policy_logprob_entropy.jsonl
```

## Metric

For each signal, the prediction is the reasoning unit containing the largest
token-level transition:

- reward and log-probability: largest drop;
- entropy: largest increase.

Labels marked as `no_clear_localisable_error` or `final_answer_only_error` are
mapped to the final visible reasoning unit, matching the paper analysis. The
reported metrics are Hit@1 and Hit@+/-1 against the expert-selected reasoning
unit, with stratified bootstrap confidence intervals for pooled results.
