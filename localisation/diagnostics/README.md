# Localisation Diagnostics

Diagnostic artifacts are separated from the main localisation result folders.

- `recalc32/<model>/<variant>/`: dense-reward rescoring summaries for a
  particular model and smoothing/recalculation variant.
- `recalc32/triptychs/<variant>/`: metadata for the trace-reward triptych
  figures used to inspect specific examples.

Large image/PDF outputs and raw per-pair traces remain on disk but are ignored
by git. The committed files are small `run_config.json`, `summary.json`, and
`*_meta.json` records.
