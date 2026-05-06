import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np


ROOT_DIR = Path("outputs") / "localisation"
DEFAULT_LATEX_STEM = "localisation_hit1at"
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_ALPHA = 0.05
BOOTSTRAP_SEED = 42

# Prefer the same display ordering used in prior tables.
MODEL_ORDER = [
    ("qwen7b", r"\textsc{Qwen2.5-7B}"),
    ("llama8b", r"\textsc{Llama3.1-8B}"),
    ("qwen4b", r"\textsc{Qwen3-4B}"),
]

SOURCE_ORDER = ["expert", "pregenerated"]
SOURCE_LABEL = {
    "expert": r"\textit{Expert}",
    "pregenerated": r"\textit{Pregenerated}",
}

REWARD_NAME_TO_MODEL = {
    "Qwen/Qwen2.5-7B-Instruct": "qwen7b",
    "meta-llama/Llama-3.1-8B-Instruct": "llama8b",
    "Qwen/Qwen3-4B-Instruct-2507": "qwen4b",
}

SOURCE_SUFFIX_TO_KEY = {
    "expert": "expert",
    "from_qwen7b_sft": "pregenerated",
}

RUN_RE = re.compile(
    r"^(?P<model>qwen4b|qwen7b|llama8b)_(?P<mode>full|partial_fixed)_localisation(?:_(?P<suffix>.*))?$"
)


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            rows.append(json.loads(raw))
    return rows


def _to_int(x: Any) -> int | None:
    if x is None:
        return None
    try:
        return int(x)
    except Exception:
        return None


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _sample_se(values: list[float]) -> float | None:
    """
    Returns bootstrap CI half-width for the sample mean.
    Kept function name for compatibility with existing call sites.
    """
    n = len(values)
    if n == 0:
        return None
    if n == 1:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(int(BOOTSTRAP_SEED))
    n_boot = max(100, int(BOOTSTRAP_SAMPLES))
    alpha = min(max(float(BOOTSTRAP_ALPHA), 1e-6), 0.5)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = arr[idx].mean(axis=1)
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return float((hi - lo) / 2.0)


def _fmt_pct_latex(mean: float | None, se: float | None) -> str:
    if mean is None:
        return "-"
    mean_pct = 100.0 * float(mean)
    if se is None:
        return f"{mean_pct:.2f}"
    se_pct = 100.0 * float(se)
    return f"{mean_pct:.2f} $\\pm$ {se_pct:.2f}"


def _fmt_pct_console(mean: float | None, se: float | None) -> str:
    if mean is None:
        return "-"
    if se is None:
        return f"{mean:.4f}"
    return f"{mean:.4f} +/- {se:.4f}"


def _fmt_plain_latex(mean: float | None, se: float | None) -> str:
    if mean is None:
        return "-"
    if se is None:
        return f"{float(mean):.2f}"
    return f"{float(mean):.2f} $\\pm$ {float(se):.2f}"


def _fmt_plain_console(mean: float | None, se: float | None) -> str:
    if mean is None:
        return "-"
    if se is None:
        return f"{float(mean):.4f}"
    return f"{float(mean):.4f} +/- {float(se):.4f}"


def _infer_model(run_name: str, run_cfg: dict[str, Any]) -> str | None:
    m = RUN_RE.match(run_name)
    if m is not None:
        return m.group("model")

    reward_name = run_cfg.get("reward_name")
    if reward_name in REWARD_NAME_TO_MODEL:
        return REWARD_NAME_TO_MODEL[reward_name]

    args = run_cfg.get("args", {}) if isinstance(run_cfg, dict) else {}
    config_path = str(args.get("config", ""))
    config_path = config_path.replace("\\", "/")
    for key, _label in MODEL_ORDER:
        if f"/{key}/" in config_path:
            return key

    ckpt = str(args.get("checkpoint_dir", "")).replace("\\", "/")
    for key, _label in MODEL_ORDER:
        if f"/{key}_" in ckpt:
            return key

    return None


def _normalize_mode(mode: Any) -> str | None:
    if isinstance(mode, str):
        if mode == "full":
            return "full"
        if mode in {"partial_fixed", "partial"}:
            return "partial_fixed"
        if mode == "sparse":
            return "sparse"
    if isinstance(mode, bool):
        return "full" if mode else "sparse"
    return None


def _infer_mode(run_name: str, run_cfg: dict[str, Any]) -> str | None:
    m = RUN_RE.match(run_name)
    if m is not None:
        return m.group("mode")

    args = run_cfg.get("args", {}) if isinstance(run_cfg, dict) else {}
    mode = _normalize_mode(args.get("dense_reward_mode"))
    if mode is not None:
        return mode
    return _normalize_mode(run_cfg.get("dense_rewards"))


def _infer_source(run_name: str, run_cfg: dict[str, Any]) -> str | None:
    m = RUN_RE.match(run_name)
    if m is not None:
        suffix = m.group("suffix")
        if suffix in SOURCE_SUFFIX_TO_KEY:
            return SOURCE_SUFFIX_TO_KEY[suffix]

    args = run_cfg.get("args", {}) if isinstance(run_cfg, dict) else {}
    trace_source = args.get("trace_source", run_cfg.get("trace_source", None))
    if trace_source in {"expert", "pregenerated"}:
        return trace_source

    # Fallback heuristic for non-canonical folder names.
    lower = run_name.lower()
    if "pregenerated" in lower or "from_qwen7b_sft" in lower:
        return "pregenerated"
    if "expert" in lower:
        return "expert"
    return None


def _discover_runs(root_dir: Path, mode_filter: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for run_dir in sorted(root_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        pair_path = run_dir / "pair_details.jsonl"
        if not pair_path.exists():
            continue

        cfg_path = run_dir / "run_config.json"
        run_cfg = _load_json(cfg_path) if cfg_path.exists() else {}

        run_name = run_dir.name
        model = _infer_model(run_name, run_cfg)
        mode = _infer_mode(run_name, run_cfg)
        source = _infer_source(run_name, run_cfg)

        if model is None or mode != mode_filter or source not in {"expert", "pregenerated"}:
            continue

        candidates.append(
            {
                "run_name": run_name,
                "run_dir": run_dir,
                "pair_path": pair_path,
                "run_cfg": run_cfg,
                "model": model,
                "mode": mode,
                "source": source,
            }
        )

    def _priority(run: dict[str, Any]) -> int:
        model = run["model"]
        source = run["source"]
        name = run["run_name"]
        if source == "expert" and name == f"{model}_{mode_filter}_localisation_expert":
            return 3
        if source == "pregenerated" and name == f"{model}_{mode_filter}_localisation_from_qwen7b_sft":
            return 3
        if name.startswith(f"{model}_{mode_filter}_localisation"):
            return 2
        return 1

    # Keep best run for each (model, source, mode):
    # prefer canonical naming, then latest modified.
    by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for c in candidates:
        key = (c["model"], c["source"], c["mode"])
        prev = by_key.get(key)
        if prev is None:
            by_key[key] = c
            continue
        p_prev = _priority(prev)
        p_curr = _priority(c)
        if p_curr > p_prev:
            by_key[key] = c
            continue
        if p_curr < p_prev:
            continue
        # Same priority: pick latest.
        if c["pair_path"].stat().st_mtime >= prev["pair_path"].stat().st_mtime:
            by_key[key] = c

    out = list(by_key.values())
    model_rank = {k: i for i, (k, _l) in enumerate(MODEL_ORDER)}
    source_rank = {k: i for i, k in enumerate(SOURCE_ORDER)}
    out.sort(
        key=lambda r: (
            model_rank.get(r["model"], 999),
            source_rank.get(r["source"], 999),
            r["run_name"],
        )
    )
    return out


def _single_trace_hit1_at_w_raw(
    rows: list[dict[str, Any]],
    window: int,
    only_single_perturbation: bool,
    mode: str,
    partial_fixed_stride: int | None,
    partial_fixed_default_stride: int,
) -> tuple[float | None, float | None, float | None, float | None, int]:
    hits: list[float] = []
    hit_norms: list[float] = []
    w = int(max(0, window))

    # For partial_fixed runs, fallback when metadata stride is missing/invalid.
    stride = int(partial_fixed_stride) if partial_fixed_stride is not None else 0
    if mode == "partial_fixed" and stride <= 0:
        stride = int(max(1, partial_fixed_default_stride))

    for row in rows:
        if only_single_perturbation:
            sev = _to_int(row.get("severity"))
            if sev != 1:
                continue

        pert_seq = row.get("pert_score_seq", [])
        changed_positions = row.get("changed_token_positions", [])
        if not isinstance(pert_seq, list) or not isinstance(changed_positions, list):
            continue
        if len(pert_seq) < 2:
            continue

        changed = sorted(
            {
                int(c)
                for c in changed_positions
                if _to_int(c) is not None and 0 <= int(c) < len(pert_seq)
            }
        )
        if not changed:
            continue

        z = np.asarray([float(v) for v in pert_seq], dtype=np.float64)
        drops = np.maximum(0.0, z[:-1] - z[1:])
        pred = int(np.argmax(drops)) + 1

        t_len = len(pert_seq)
        if mode == "partial_fixed":
            n_buckets = int(math.ceil(t_len / float(stride)))
            pred_bucket = int(pred // stride)
            changed_buckets = sorted({int(c // stride) for c in changed})
            w_bucket = int(math.ceil(w / float(stride)))

            err = min(abs(pred_bucket - b) for b in changed_buckets)
            hit = 1.0 if err <= w_bucket else 0.0
            hits.append(hit)

            local_mask = np.zeros(n_buckets, dtype=bool)
            for b in changed_buckets:
                lo = max(0, int(b) - w_bucket)
                hi = min(n_buckets - 1, int(b) + w_bucket)
                local_mask[lo : hi + 1] = True
            p_rand = float(local_mask.mean()) if n_buckets > 0 else float("nan")
        else:
            err = min(abs(pred - c) for c in changed)
            hit = 1.0 if err <= w else 0.0
            hits.append(hit)

            # Chance baseline for single-point prediction under uniform random index.
            # p_rand = fraction of token indices that would count as a hit.
            local_mask = np.zeros(t_len, dtype=bool)
            for c in changed:
                lo = max(0, int(c) - w)
                hi = min(t_len - 1, int(c) + w)
                local_mask[lo : hi + 1] = True
            p_rand = float(local_mask.mean()) if t_len > 0 else float("nan")

        denom = 1.0 - p_rand
        if math.isfinite(p_rand) and denom > 1e-12:
            hit_norms.append(float((hit - p_rand) / denom))

    return _mean(hits), _sample_se(hits), _mean(hit_norms), _sample_se(hit_norms), len(hits)


def _label_for_model(model_key: str) -> str:
    for key, label in MODEL_ORDER:
        if key == model_key:
            return label
    return model_key


MODE_TABLE_ORDER = ["full", "partial_fixed"]
MODE_LABEL = {
    "full": "Full",
    "partial_fixed": "Partial-Fixed",
}


def _render_latex_table(
    by_model_mode: dict[tuple[str, str], dict[str, Any]],
    window: int,
    only_single_perturbation: bool,
    source: str,
) -> str:
    conf_level = (1.0 - float(BOOTSTRAP_ALPHA)) * 100.0
    latex: list[str] = []
    latex.append(r"% Requires \usepackage{booktabs}")
    latex.append(r"\begin{table}[t]")
    latex.append(r"\centering")
    latex.append(r"\small")
    latex.append(r"\renewcommand{\arraystretch}{1.08}")
    latex.append(r"\begin{tabular}{l r r}")
    latex.append(r"\toprule")
    latex.append(
        rf"\textbf{{Model}} & "
        rf"\textbf{{{MODE_LABEL['full']} Hit@1@{window} (\%)}} & "
        rf"\textbf{{{MODE_LABEL['partial_fixed']} Hit@1@{window} (\%)}} \\"
    )
    latex.append(r"\midrule")

    for model_key, _model_label in MODEL_ORDER:
        full_row = by_model_mode.get((model_key, "full"), None)
        pfix_row = by_model_mode.get((model_key, "partial_fixed"), None)

        full_cell = _fmt_pct_latex(full_row["hit_mean"], full_row["hit_se"]) if full_row is not None else "-"
        pfix_cell = _fmt_pct_latex(pfix_row["hit_mean"], pfix_row["hit_se"]) if pfix_row is not None else "-"

        latex.append(f"{_label_for_model(model_key)} & {full_cell} & {pfix_cell} \\\\")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    source_label = SOURCE_LABEL.get(source, source)
    latex.append(
        rf"\caption{{\textbf{{Single-trace localisation on GSM8K ({source_label}).}} "
        rf"Metric: raw-reward Hit@1@{window} (largest downward step in corrupted trace), "
        rf"reported as percentage mean $\pm$ bootstrapped {conf_level:.0f}\% CI half-width. "
        rf"Bucket-aware evaluation for partial_fixed (15-token plateaus). "
        rf"{'Filtered to single-perturbation rows (severity=1).' if only_single_perturbation else ''}}}"
    )
    latex.append(rf"\label{{tab:localisation_hit1at{window}_{source}_full_vs_partial_raw}}")
    latex.append(r"\end{table}")
    return "\n".join(latex)


def _render_console_table(by_model_mode: dict[tuple[str, str], dict[str, Any]], window: int) -> str:
    conf_level = (1.0 - float(BOOTSTRAP_ALPHA)) * 100.0
    headers = [
        "Model",
        f"Full Hit@1@{window} (+/-CI{conf_level:.0f}%/2)",
        f"Partial-Fixed Hit@1@{window} (+/-CI{conf_level:.0f}%/2)",
    ]
    rows: list[list[str]] = []

    for model_key, _model_label in MODEL_ORDER:
        full_row = by_model_mode.get((model_key, "full"), None)
        pfix_row = by_model_mode.get((model_key, "partial_fixed"), None)

        full_cell = _fmt_pct_console(full_row["hit_mean"], full_row["hit_se"]) if full_row is not None else "-"
        pfix_cell = _fmt_pct_console(pfix_row["hit_mean"], pfix_row["hit_se"]) if pfix_row is not None else "-"

        rows.append([model_key, full_cell, pfix_cell])

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(row: list[str]) -> str:
        return " | ".join(row[i].ljust(widths[i]) for i in range(len(row)))

    sep = "-+-".join("-" * w for w in widths)
    out = [fmt_row(headers), sep]
    for row in rows:
        out.append(fmt_row(row))
    return "\n".join(out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a compact localisation table across full and partial_fixed runs "
            "for a single source (expert or pregenerated), using raw Hit@1@window."
        )
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=ROOT_DIR,
        help="Directory containing localisation run folders.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help=(
            "Output .tex path. Defaults to "
            "<root-dir>/localisation_hit1at<window>_<source>_full_vs_partial_raw.tex."
        ),
    )
    parser.add_argument(
        "--window",
        type=int,
        default=15,
        help="Hit window size for Hit@1@window (token distance).",
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["expert", "pregenerated"],
        default="pregenerated",
        help="Trace source to include in the table.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=2000,
        help="Number of bootstrap resamples for CI half-width.",
    )
    parser.add_argument(
        "--bootstrap-alpha",
        type=float,
        default=0.05,
        help="Bootstrap CI alpha (0.05 => 95% CI).",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=42,
        help="Bootstrap RNG seed.",
    )
    parser.add_argument(
        "--only-single-perturbation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, include only rows with severity == 1.",
    )
    parser.add_argument(
        "--partial-fixed-default-stride",
        type=int,
        default=15,
        help=(
            "Fallback stride for partial_fixed bucket evaluation when not available in run config."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global BOOTSTRAP_SAMPLES, BOOTSTRAP_ALPHA, BOOTSTRAP_SEED
    BOOTSTRAP_SAMPLES = int(args.bootstrap_samples)
    BOOTSTRAP_ALPHA = float(args.bootstrap_alpha)
    BOOTSTRAP_SEED = int(args.bootstrap_seed)

    root_dir = args.root_dir
    if not root_dir.exists():
        raise FileNotFoundError(f"Root directory not found: {root_dir}")

    by_model_mode: dict[tuple[str, str], dict[str, Any]] = {}
    for mode in MODE_TABLE_ORDER:
        runs = _discover_runs(root_dir, mode_filter=mode)
        if not runs:
            continue
        for run in runs:
            if run["source"] != args.source:
                continue
            pair_rows = _load_jsonl(run["pair_path"])
            run_cfg = run.get("run_cfg", {})
            args_cfg = run_cfg.get("args", {}) if isinstance(run_cfg, dict) else {}
            stride = _to_int(run_cfg.get("partial_fixed_stride")) if isinstance(run_cfg, dict) else None
            if stride is None and isinstance(args_cfg, dict):
                stride = _to_int(args_cfg.get("partial_fixed_stride"))
            if stride is None and isinstance(args_cfg, dict):
                stride = _to_int(args_cfg.get("dense_partial_fixed_n"))

            hit_mean, hit_se, _hit_norm_mean, _hit_norm_se, n_used = _single_trace_hit1_at_w_raw(
                pair_rows,
                window=args.window,
                only_single_perturbation=bool(args.only_single_perturbation),
                mode=mode,
                partial_fixed_stride=stride,
                partial_fixed_default_stride=int(args.partial_fixed_default_stride),
            )
            by_model_mode[(run["model"], mode)] = {
                "run_name": run["run_name"],
                "hit_mean": hit_mean,
                "hit_se": hit_se,
                "n_used": n_used,
            }

    if not by_model_mode:
        raise ValueError(
            f"No matching runs found under {root_dir} for source={args.source} "
            f"and modes={MODE_TABLE_ORDER}"
        )

    latex = _render_latex_table(
        by_model_mode,
        window=args.window,
        only_single_perturbation=bool(args.only_single_perturbation),
        source=str(args.source),
    )
    output_file = (
        args.output_file
        if args.output_file is not None
        else root_dir / f"{DEFAULT_LATEX_STEM}{int(args.window)}_{args.source}_full_vs_partial_raw.tex"
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        f.write(latex)

    print(f"Wrote LaTeX table to {output_file}")
    print(
        "Bootstrap settings: "
        f"samples={BOOTSTRAP_SAMPLES}, alpha={BOOTSTRAP_ALPHA}, seed={BOOTSTRAP_SEED}"
    )
    print(f"Source: {args.source}")
    print(f"Filtering: only_single_perturbation={bool(args.only_single_perturbation)}")
    print("\nSelected runs:")
    for model_key, _label in MODEL_ORDER:
        for mode in MODE_TABLE_ORDER:
            key = (model_key, mode)
            row = by_model_mode.get(key)
            if row is None:
                print(f"  - {model_key}/{mode}: MISSING")
            else:
                print(
                    f"  - {model_key}/{mode}: run={row['run_name']} | "
                    f"n={row['n_used']} | Hit@1@{args.window}={row['hit_mean']:.4f}"
                )

    print("\n=== Compact Localisation Table ===")
    print(_render_console_table(by_model_mode, window=args.window))


if __name__ == "__main__":
    main()
