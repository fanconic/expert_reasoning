# Figures Directory

This folder stores generated tables/plots and historical experiment outputs.

## Active Output Layout
Current plot generators write outputs to:
- `figures/answer_only/<domain>/<experiment_label>/...`
- `figures/full_cot/<domain>/<experiment_label>/...`

For `figures/answer_only/{math,mmlu,medicine}`, runs are now organized as:
- `standard/`: canonical runs (`*_sparse`, `*_partial`, `*_partial_fixed`, `*_full`; legacy `*_new` names are also supported)
- `ablations/`: all non-standard non-transfer runs
- `transferability/`: runs prefixed with `transfer_`

Each experiment directory typically contains:
- pass@k tables (`*.txt`)
- reranking tables (`*reranking*.txt`)
- summary plots (`*.pdf`)
- optional token-level diagnostics (`dense_rewards_*/*.pdf`)
