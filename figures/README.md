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



```bash
python src/plot_generators/plot_main.py \
  --config src/plot_generators/configs/main.yaml \
  --domains math,medicine,mmlu \
  --models qwen3b,qwen4b,llama3b,qwen7b,llama8b \
  --variants sparse,partial,partial_fixed,full \
  --airl-file-template 'eval_results_{domain}_{model}_{variant}_t0p5.jsonl' \
  --sft-file-template 'eval_results_{domain}_{model}_sft_t0p5.jsonl' \
  --grpo-file-template 'eval_results_{domain}_{model}_grpo_t0p5.jsonl' \
  --num-generations 16 \
  --reranking-generations 2,3,5,8,16 \
  --output-root ./figures/sft_reranking_temp05 \
```


```bash
python src/plot_generators/plot_reranking_baselines.py \
  --models qwen4b \
  --datasets math \
  --variants partial_fixed \
  --num-generations 2,3,5,8,16 \
  --output-file figures/sft_reranking_temp05/results_reranking_qwen4b_math_partial.txt \
  --pdf-dir figures/sft_reranking_temp05/plots_reranking_compact
```

```bash
DENSITY=partial_fixed  # change to partial if that is what you ran
for R in math medicine mmlu; do
  python src/plot_generators/plot_main.py \
    --config src/plot_generators/configs/main.yaml \
    --domains math,medicine,mmlu \
    --models qwen7b,qwen4b \
    --variants "${DENSITY}" \
    --guided-method "${R}" \
    --airl-run-template 'llama8b_sft' \
    --airl-run-template 'transfer_{model}_{variant}_P_{domain}_R_{method}_t0p5' \
    --sft-run-template 'llama8b_sft' \
    --grpo-run-template 'llama8b_grpo' \
    --sft-file-template 'eval_results_{domain}_llama8b_sft_t0p5.jsonl' \
    --airl-file-template 'transfer_{model}_{variant}_P_{domain}_R_{method}_t0p5.jsonl' \
    --grpo-file-template 'eval_results_{domain}_llama8b_grpo_t0p5.jsonl' \
    --num-generations 16 \
    --reranking-generations 16 \
    --output-root "figures/transferability_ablation_temp05_fixed_llama8b/R_${R}"
done
```


```bash
DENSITY=partial_fixed  # change to partial if that is what you ran
for R in math medicine mmlu; do
  python src/plot_generators/plot_main.py \
    --config src/plot_generators/configs/main.yaml \
    --domains math,medicine,mmlu \
    --models qwen7b,llama8b \
    --variants "${DENSITY}" \
    --guided-method "${R}" \
    --airl-run-template qwen4b_sft' \
    --airl-file-template 'transfer_{model}_{variant}_P_{domain}_R_{method}_t0p5.jsonl' \
    --sft-run-template qwen4b_sft' \
    --grpo-run-template 'qwen4b_grpo' \
    --sft-file-template 'eval_results_{domain}_qwen4b_sft_t0p5.jsonl' \
    --grpo-file-template 'eval_results_{domain}_qwen4b_grpo_t0p5.jsonl' \
    --num-generations 16 \
    --reranking-generations 16 \
    --output-root "figures/transferability_ablation_temp05_fixed_qwen4b/R_${R}"
done
```
