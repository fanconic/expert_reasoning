# Runner Scripts

This folder contains the cluster launchers used for the paper experiments and
appendix ablations. Scratch launchers, obsolete rescoring jobs, and generated
logs have been moved to the ignored local archive:

```text
runner_scripts/_backup_not_committed/
```

## Core Launchers

- `0_run_gpu_node.sh` to `3_run_gpu_node.sh`: shared single-GPU wrappers.
- `super_runners/`: main AIRL training sweeps.
- `retakes/`: final reruns and GRPO/SFT comparison launchers.
- `qwen4/`: Qwen3-4B training launchers.

## Evaluation And Reranking

- `eval_all_temp05/`: temperature-0.5 evaluation sweeps.
- `sft_reranking_temp05/`: fixed-candidate SFT reranking and policy log-prob
  scoring.
- `transferability_temp05/`: transfer reranking runs used for cross-domain
  reward-model comparisons.

## Appendix Ablations

- `betas/`: KL/beta ablation launchers.
- `different_groups/`: group-size ablation launchers.
- `corruption/`: expert-corruption ablation launchers.

## Final Baselines And Localisation

- `rebuttal/`: final restart, baseline, and localisation runners for the
  paper/appendix. See `runner_scripts/rebuttal/README.md` for details.
