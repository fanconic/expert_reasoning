# Reward-Model Localisation Runs

Canonical run folders are grouped as:

```text
runs/<source>/<model>/<granularity>/
```

Sources:

- `qwen7b_sft`: corruptions applied to pregenerated Qwen2.5-7B SFT traces.
- `expert`: corruptions applied to expert traces.

Models:

- `qwen7b`
- `qwen4b`
- `llama8b`

Granularities:

- `full`: dense token-level reward model.
- `partial_fixed`: fixed-interval reward model.

Each leaf run folder contains the same files as the old flat layout:
`run_config.json`, `summary.json`, and, locally, ignored `pair_details.jsonl`
traces. Top-level symlinks preserve the old flat folder names.
