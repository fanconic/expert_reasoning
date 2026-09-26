# Localisation Experiments

This folder contains only the process-level localisation experiments represented
in the paper and appendix.

## Clinician MedReason

`clinician_medreason/` contains the MedReason clinician-labelled localisation
experiment:

```bash
python localisation/clinician_medreason/run_experiment.py
```

That command reads the anonymised expert labels in
`clinician_medreason/annotations/`, aligns them to the labelled cases in
`clinician_medreason/metadata/clinician_cases.jsonl`, loads token-level reward
and policy-score JSONLs, and writes:

```text
localisation/clinician_medreason/results/clinician_localisation_metrics.md
localisation/clinician_medreason/results/clinician_localisation_metrics.json
```

The default metric is Hit@1 and Hit@+/-1 over clinician-visible reasoning
units. Labels marked no-clear or final-answer-only are mapped to the final
visible reasoning unit.

## Controlled GSM8K

`synthetic_perturbations/` contains the controlled GSM8K synthetic
localisation experiment from Section 5.2 and Appendix D.5:

```bash
bash localisation/synthetic_perturbations/run_experiment.sh
```

This regenerates the synthetic perturbation tables under
`synthetic_perturbations/results/` from the compact run summaries in
`synthetic_perturbations/runs/`.

## Local Backup

Exploratory natural-error runs, old HTML annotation packages, smoke outputs,
large token-score dumps, and historical diagnostic folders were moved to
`_backup_not_committed/`. That folder is ignored by git and is only a local
archive. Commit only reproducible code, compact paper summaries, and anonymised
annotation files.
