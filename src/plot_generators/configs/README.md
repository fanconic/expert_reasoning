# Plot Specs

YAML files in this folder define plotting jobs.

## Files
- `main.yaml`: primary paper/rebuttal plotting jobs.
- `transfer.yaml`: transferability plotting jobs.

## Schema (high level)
- `domains`: mapping from logical domain name to outputs root path.
- `experiments`: list of `{airl, sft, grpo, label, domains?}` mappings.
- `preferred_jsonl_files`: fallback order for eval files.
- `aime_jsonl_by_domain` (optional): domain-specific filename preferences.
- `airl_eval_file_from_label`: if true, AIRL file is resolved as `<label>.jsonl`.

Use with:
```bash
python src/plot_generators/plot_main.py --config src/plot_generators/configs/main.yaml
python src/plot_generators/plot_transfer.py --config src/plot_generators/configs/transfer.yaml
```
