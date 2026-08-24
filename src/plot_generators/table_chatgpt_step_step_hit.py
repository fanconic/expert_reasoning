"""Compute step-level localisation hit for ChatGPT-edited GSM8K traces."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from transformers import AutoTokenizer


DEFAULT_ROOT = Path("localisation/chatgpt_step_perturbations/scores")
DEFAULT_PAIRS = Path(
    "localisation/chatgpt_step_perturbations/gsm8k_qwen7b_sft_step_perturbations_full.jsonl"
)
DEFAULT_JSON = Path("localisation/chatgpt_step_perturbations/localisation_chatgpt_step_step_hit.json")
DEFAULT_TEX = Path("localisation/chatgpt_step_perturbations/localisation_chatgpt_step_step_hit.tex")
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_ALPHA = 0.05
BOOTSTRAP_SEED = 42

MODEL_ORDER = ["qwen7b", "llama8b", "qwen4b"]
MODEL_LABELS = {
    "qwen7b": r"\textsc{Qwen2.5-7B}",
    "llama8b": r"\textsc{Llama3.1-8B}",
    "qwen4b": r"\textsc{Qwen3-4B}",
}
POLICY_VARIANTS = [
    ("base", "Base"),
    ("sft", "SFT"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--pairs-jsonl", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-tex", type=Path, default=DEFAULT_TEX)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--bootstrap-alpha", type=float, default=BOOTSTRAP_ALPHA)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r") as f:
        for line in f:
            raw = line.strip()
            if raw:
                rows.append(json.loads(raw))
    return rows


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    with path.open("w") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")


def _to_int(x: Any) -> int | None:
    try:
        return int(x)
    except Exception:
        return None


def _finite_values(values: Sequence[float]) -> np.ndarray:
    out = []
    for value in values:
        try:
            parsed = float(value)
        except Exception:
            continue
        if math.isfinite(parsed):
            out.append(parsed)
    return np.asarray(out, dtype=np.float64)


def _mean_ci(values: Sequence[float], samples: int, alpha: float, seed: int) -> dict[str, float | int | None]:
    vals = _finite_values(values)
    n = int(vals.shape[0])
    if n == 0:
        return {"mean": None, "ci_halfwidth": None, "n": 0}
    mean = float(vals.mean())
    if n == 1:
        return {"mean": mean, "ci_halfwidth": 0.0, "n": 1}
    rng = np.random.default_rng(int(seed))
    n_boot = max(100, int(samples))
    alpha = min(max(float(alpha), 1e-6), 0.5)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = vals[idx].mean(axis=1)
    lo = float(np.quantile(boot_means, alpha / 2.0))
    hi = float(np.quantile(boot_means, 1.0 - alpha / 2.0))
    return {"mean": mean, "ci_halfwidth": float((hi - lo) / 2.0), "n": n}


def _fmt_pct(metric: dict[str, float | int | None], bold: bool = False) -> str:
    mean = metric.get("mean")
    ci = metric.get("ci_halfwidth")
    if mean is None:
        return "-"
    if ci is None:
        cell = f"{100.0 * float(mean):.2f}"
    else:
        cell = f"{100.0 * float(mean):.2f} $\\pm$ {100.0 * float(ci):.2f}"
    if bold:
        cell = re.sub(r"(-?\d+(?:\.\d+)?)", r"\\textbf{\1}", cell, count=1)
    return cell


def _resolve_tokenizer_name(model_name_or_path: str) -> str:
    path = Path(model_name_or_path)
    if path.exists():
        for candidate in [path, path / "reward_model"]:
            if (candidate / "tokenizer_config.json").exists():
                return str(candidate)
        adapter_config = path / "adapter_config.json"
        if adapter_config.exists():
            cfg = _load_json(adapter_config)
            base = cfg.get("base_model_name_or_path")
            if base:
                return str(base)
    return str(model_name_or_path)


def _load_tokenizer_cached(cache: dict[str, Any], model_name_or_path: str):
    resolved = _resolve_tokenizer_name(model_name_or_path)
    if resolved not in cache:
        cache[resolved] = AutoTokenizer.from_pretrained(
            resolved,
            local_files_only=True,
            trust_remote_code=True,
        )
    return cache[resolved], resolved


def _token_positions_from_char_span(
    tokenizer,
    text: str,
    char_span: Sequence[int] | None,
    max_length: int,
) -> list[int]:
    if char_span is None or len(char_span) != 2:
        return []
    start = _to_int(char_span[0])
    end = _to_int(char_span[1])
    if start is None or end is None or end <= start:
        return []
    enc = tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
    )
    out = []
    for idx, offset in enumerate(enc.get("offset_mapping", [])):
        if not isinstance(offset, (list, tuple)) or len(offset) != 2:
            continue
        tok_start = _to_int(offset[0])
        tok_end = _to_int(offset[1])
        if tok_start is None or tok_end is None or tok_end <= tok_start:
            continue
        if tok_end > start and tok_start < end:
            out.append(int(idx))
    return out


def _prediction_index(row: dict[str, Any], seq_key: str, pred_key: str, detector: str) -> int | None:
    pred = _to_int(row.get(pred_key))
    if pred is not None:
        return pred
    seq = row.get(seq_key)
    if not isinstance(seq, list) or len(seq) < 2:
        return None
    arr = np.asarray(seq, dtype=np.float64)
    if not np.isfinite(arr).all():
        finite = np.isfinite(arr)
        if finite.sum() < 2:
            return None
        arr = arr[finite]
    if detector == "largest_drop":
        scores = np.maximum(0.0, arr[:-1] - arr[1:])
    elif detector == "largest_spike":
        scores = np.maximum(0.0, arr[1:] - arr[:-1])
    else:
        raise ValueError(f"Unknown detector: {detector}")
    return int(np.argmax(scores)) + 1


def _step_hit_and_chance(
    pred_idx: int | None,
    step_positions: Sequence[int],
    seq_len: int,
    localization_mode: str,
    stride: int,
) -> tuple[float, float, int]:
    positions = sorted({int(p) for p in step_positions if 0 <= int(p) < int(seq_len)})
    if seq_len <= 0 or not positions:
        return float("nan"), float("nan"), 0

    if localization_mode == "bucket":
        stride = max(1, int(stride))
        n_units = int(math.ceil(seq_len / float(stride)))
        targets = sorted({p // stride for p in positions})
        pred_unit = None if pred_idx is None else int(pred_idx) // stride
    else:
        n_units = int(seq_len)
        targets = positions
        pred_unit = None if pred_idx is None else int(pred_idx)

    if n_units <= 0 or not targets:
        return float("nan"), float("nan"), 0
    target_set = {int(t) for t in targets if 0 <= int(t) < n_units}
    chance = float(len(target_set) / float(n_units))
    hit = float(pred_unit in target_set) if pred_unit is not None else float("nan")
    return hit, chance, len(target_set)


def _method_specs(root_dir: Path) -> list[dict[str, Any]]:
    specs = []
    for model in MODEL_ORDER:
        for density, signal_label in [("full", "Reward dense"), ("partial_fixed", "Reward interval")]:
            summary_path = root_dir / f"{model}_{density}_reward_localisation" / "summary.json"
            if not summary_path.exists():
                continue
            summary = _load_json(summary_path)
            tokenizer_source = summary.get("checkpoint_dir") or summary.get("reward_name")
            specs.append(
                {
                    "key": f"{model}_{density}_reward",
                    "model_key": model,
                    "signal": signal_label,
                    "path": root_dir / f"{model}_{density}_reward_localisation" / "pair_details.jsonl",
                    "summary_path": summary_path,
                    "tokenizer_source": tokenizer_source,
                    "seq_key": "pert_score_seq",
                    "pred_key": "reward_pred_idx",
                    "detector": "largest_drop",
                    "localization_mode": None,
                    "partial_fixed_stride": None,
                }
            )

    for model in MODEL_ORDER:
        for variant, variant_label in POLICY_VARIANTS:
            summary_path = root_dir / f"{model}_{variant}_policy_token_baselines" / "policy_token_baselines_summary.json"
            detail_path = root_dir / f"{model}_{variant}_policy_token_baselines" / "policy_token_baselines.jsonl"
            if not summary_path.exists() or not detail_path.exists():
                continue
            summary = _load_json(summary_path)
            for seq_key, pred_key, detector, signal_suffix in [
                ("pert_policy_log_probs", "logprob_pred_idx", "largest_drop", "token log-probability"),
                ("pert_policy_entropies", "entropy_pred_idx", "largest_spike", "token entropy"),
            ]:
                specs.append(
                    {
                        "key": f"{model}_{variant}_{signal_suffix.replace(' ', '_')}",
                        "model_key": model,
                        "signal": f"{variant_label} {signal_suffix}",
                        "path": detail_path,
                        "summary_path": summary_path,
                        "tokenizer_source": summary.get("policy_model"),
                        "seq_key": seq_key,
                        "pred_key": pred_key,
                        "detector": detector,
                        "localization_mode": "token",
                        "partial_fixed_stride": None,
                    }
                )
    return specs


def build_results(args: argparse.Namespace) -> dict[str, Any]:
    pair_rows = {
        int(row["prompt_idx"]): row
        for row in _load_jsonl(args.pairs_jsonl)
        if row.get("error") is None and row.get("prompt_idx") is not None
    }
    tokenizer_cache: dict[str, Any] = {}
    methods = []

    for spec in _method_specs(args.root_dir):
        rows = _load_jsonl(spec["path"])
        summary = _load_json(spec["summary_path"])
        tokenizer, tokenizer_source = _load_tokenizer_cached(
            tokenizer_cache,
            str(spec["tokenizer_source"]),
        )
        max_length = int(summary.get("max_length") or 1124)
        hits = []
        chances = []
        step_unit_counts = []
        n_missing = 0
        for row in rows:
            pair_row = pair_rows.get(int(row.get("prompt_idx", -1)))
            if pair_row is None:
                n_missing += 1
                continue
            pert_text = pair_row.get("pert_text") or pair_row.get("wrong_text")
            if not isinstance(pert_text, str):
                n_missing += 1
                continue
            char_span = row.get("target_char_span") or pair_row.get("target_char_span")
            step_positions = _token_positions_from_char_span(
                tokenizer,
                text=pert_text,
                char_span=char_span,
                max_length=max_length,
            )
            seq = row.get(spec["seq_key"])
            seq_len = len(seq) if isinstance(seq, list) else 0
            pred_idx = _prediction_index(
                row,
                seq_key=spec["seq_key"],
                pred_key=spec["pred_key"],
                detector=spec["detector"],
            )
            localization_mode = spec.get("localization_mode") or row.get("localization_mode") or "token"
            stride = spec.get("partial_fixed_stride") or row.get("partial_fixed_stride") or 15
            hit, chance, n_step_units = _step_hit_and_chance(
                pred_idx=pred_idx,
                step_positions=step_positions,
                seq_len=seq_len,
                localization_mode=str(localization_mode),
                stride=int(stride),
            )
            hits.append(hit)
            chances.append(chance)
            step_unit_counts.append(float(n_step_units))

        step_hit = _mean_ci(
            hits,
            samples=int(args.bootstrap_samples),
            alpha=float(args.bootstrap_alpha),
            seed=int(args.bootstrap_seed),
        )
        step_chance = _mean_ci(
            chances,
            samples=int(args.bootstrap_samples),
            alpha=float(args.bootstrap_alpha),
            seed=int(args.bootstrap_seed),
        )
        normalized_vals = []
        for hit, chance in zip(hits, chances):
            if math.isfinite(hit) and math.isfinite(chance) and (1.0 - chance) > 1e-12:
                normalized_vals.append((hit - chance) / (1.0 - chance))
        step_norm = _mean_ci(
            normalized_vals,
            samples=int(args.bootstrap_samples),
            alpha=float(args.bootstrap_alpha),
            seed=int(args.bootstrap_seed),
        )
        step_units = _mean_ci(
            step_unit_counts,
            samples=int(args.bootstrap_samples),
            alpha=float(args.bootstrap_alpha),
            seed=int(args.bootstrap_seed),
        )
        methods.append(
            {
                "key": spec["key"],
                "model_key": spec["model_key"],
                "model": MODEL_LABELS[spec["model_key"]],
                "signal": spec["signal"],
                "detail_path": str(spec["path"]),
                "tokenizer_source": tokenizer_source,
                "detector": spec["detector"],
                "metrics": {
                    "step_hit": step_hit,
                    "step_chance": step_chance,
                    "step_normalized_hit": step_norm,
                    "step_units": step_units,
                    "n_missing": n_missing,
                },
            }
        )

    return {
        "root_dir": str(args.root_dir),
        "pairs_jsonl": str(args.pairs_jsonl),
        "bootstrap": {
            "samples": int(args.bootstrap_samples),
            "alpha": float(args.bootstrap_alpha),
            "seed": int(args.bootstrap_seed),
        },
        "methods": methods,
    }


def build_latex(results: dict[str, Any]) -> str:
    latex = []
    latex.append(r"% Requires \usepackage{multirow}")
    latex.append(r"\begin{table}[t]")
    latex.append(r"\centering")
    latex.append(r"\small")
    latex.append(r"\setlength{\tabcolsep}{4pt}")
    latex.append(
        r"\caption{\textbf{Step-level localisation on ChatGPT-edited GSM8K traces.} "
        r"A prediction is counted as correct if the single detected position falls inside "
        r"the edited reasoning step span. Chance is the fraction of token/bucket positions "
        r"covered by that step. Values are percentages with 95\% bootstrap CI half-widths.}"
    )
    latex.append(r"\label{tab:localisation_chatgpt_step_step_hit}")
    latex.append(r"\begin{tabular}{llccc}")
    latex.append(r"\toprule")
    latex.append(
        r"\textbf{Model} & \textbf{Signal} & \textbf{Step Hit (\%)} & "
        r"\textbf{Chance (\%)} & \textbf{Norm. Hit (\%)} \\"
    )
    latex.append(r"\midrule")

    first = True
    for model_key in MODEL_ORDER:
        methods = [m for m in results["methods"] if m["model_key"] == model_key]
        if not methods:
            continue
        if not first:
            latex.append(r"\midrule")
        first = False
        best = max(
            float(m["metrics"]["step_hit"]["mean"])
            for m in methods
            if m["metrics"]["step_hit"].get("mean") is not None
        )
        for idx, method in enumerate(methods):
            model_cell = rf"\multirow{{{len(methods)}}}{{*}}{{{MODEL_LABELS[model_key]}}}" if idx == 0 else ""
            hit = method["metrics"]["step_hit"]
            mean = hit.get("mean")
            line = [
                model_cell,
                method["signal"],
                _fmt_pct(hit, bold=mean is not None and abs(float(mean) - best) <= 1e-12),
                _fmt_pct(method["metrics"]["step_chance"]),
                _fmt_pct(method["metrics"]["step_normalized_hit"]),
            ]
            latex.append(" & ".join(line) + r" \\")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")
    latex.append("")
    return "\n".join(latex)


def main() -> None:
    args = parse_args()
    results = build_results(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_tex.parent.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_json, results)
    args.output_tex.write_text(build_latex(results))
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_tex}")
    for method in results["methods"]:
        print(
            f"{method['model']} | {method['signal']} | "
            f"step_hit={_fmt_pct(method['metrics']['step_hit'])} | "
            f"chance={_fmt_pct(method['metrics']['step_chance'])} | "
            f"norm={_fmt_pct(method['metrics']['step_normalized_hit'])}"
        )


if __name__ == "__main__":
    main()
