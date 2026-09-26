# Controlled GSM8K Localisation

This folder contains the paper-facing controlled synthetic localisation study
from Section 5.2 and Appendix D.5.

The experiment starts from GSM8K reasoning traces and injects one targeted
arithmetic perturbation into an otherwise clean trace. Since the injected token
is known, localisation is evaluated at token level: Hit@1 requires the largest
predicted transition score to fall within 1 token of the injected error, and
Hit@7 allows a 7-token window.

## Rebuild Tables

```bash
bash localisation/synthetic_perturbations/run_experiment.sh
```

This reads the scored synthetic runs under `runs/` plus the rescored rebuttal
outputs under `outputs/gsm8k_process_sensitivity_pregen/rebuttal_scores`, then
writes tables to `results/`.

To rescore the synthetic set on a GPU machine before rebuilding the tables:

```bash
GPU_NUM=0 bash runner_scripts/rebuttal/original_synthetic_mistakes/score_original_synthetic_localisation.sh
GPU_NUM=0 bash runner_scripts/rebuttal/localisation_policy_baselines/run_qwen7b_sft_policy_token_baselines.sh
```

## Files

- `run_experiment.sh`: one-command table regeneration for the synthetic
  localisation results.
- `runs/`: compact run configs and summaries for the controlled GSM8K
  perturbation runs. Raw `pair_details.jsonl` files are ignored.
- `results/table4_controlled_gsm8k_localisation.md`: manuscript Table 4
  controlled-GSM8K slice.
- `results/controlled_gsm8k_*`: Hit@1/Hit@7 source LaTeX tables used for
  appendix checks.
