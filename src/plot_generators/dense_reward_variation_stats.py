"""Summarize token-level variation in full dense reward traces.

The reviewer-facing question is whether dense rewards are genuinely token-local
or whether they are flat until the last token. This script computes per-
generation variation statistics from raw eval JSONLs.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from transformers import AutoTokenizer
except Exception:  # pragma: no cover - script fallback for lean envs.
    AutoTokenizer = None  # type: ignore[assignment]


EPS = 1e-6
OUT_DIR = Path("figures/answer_only")
OUT_JSON = OUT_DIR / "dense_reward_variation_stats.json"
OUT_MD = OUT_DIR / "dense_reward_variation_stats.md"

MODEL_NAMES = {
    "qwen7b": "Qwen2.5-7B",
    "qwen4b": "Qwen3-4B",
    "llama8b": "Llama3.1-8B",
}

TOKENIZERS = {
    "qwen7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen4b": "Qwen/Qwen3-4B-Instruct-2507",
    "llama8b": "meta-llama/Llama-3.1-8B-Instruct",
}

TRACE_PATHS = {
    ("gsm8k", "qwen7b"): Path(
        "/mnt/pdata/caf83/neurips2026/math/outputs/"
        "qwen7b_full_rebuttal_restart/best_model/"
        "eval_results_math_qwen7b_full_t0p5.jsonl"
    ),
    ("gsm8k", "qwen4b"): Path(
        "/mnt/pdata/caf83/icml_math/outputs/qwen4b_full/best_model/"
        "eval_results_math_qwen4b_full_t0p5.jsonl"
    ),
    ("gsm8k", "llama8b"): Path(
        "/mnt/pdata/caf83/icml_math/outputs/llama8b_full/best_model/"
        "eval_results_math_llama8b_full_t0p5.jsonl"
    ),
    ("mmlu", "qwen7b"): Path(
        "/mnt/pdata/caf83/neurips2026/mmlu/outputs/"
        "qwen7b_full_rebuttal_warm_reward_lr1e6/best_model/"
        "eval_results_mmlu_qwen7b_full_warm_reward_lr1e6_t0p5.jsonl"
    ),
    ("mmlu", "qwen4b"): Path(
        "/mnt/pdata/caf83/neurips2026/mmlu/outputs/qwen4b_full/best_model/"
        "eval_results_mmlu_qwen4b_full_t0p5.jsonl"
    ),
    ("mmlu", "llama8b"): Path(
        "/mnt/pdata/caf83/neurips2026/mmlu/outputs/"
        "llama8b_full_rebuttal_restart/best_model/"
        "eval_results_mmlu_llama8b_full_t0p5.jsonl"
    ),
    ("medreason", "qwen7b"): Path(
        "/mnt/pdata/caf83/neurips2026/medicine/outputs/qwen7b_full/"
        "best_model/eval_results_medical_kd.jsonl"
    ),
    ("medreason", "qwen4b"): Path(
        "/mnt/pdata/caf83/neurips2026/medicine/outputs/qwen4b_full/"
        "best_model/eval_results_medical_kd.jsonl"
    ),
    ("medreason", "llama8b"): Path(
        "/mnt/pdata/caf83/neurips2026/medicine/outputs/llama8b_full/"
        "best_model/eval_results_medical_kd.jsonl"
    ),
}


@dataclass
class RowStats:
    dataset: str
    model: str
    status: str
    source_path: str
    n_generations: int = 0
    n_valid: int = 0
    n_missing_scores: int = 0
    n_short_scores: int = 0
    marker_counts: dict[str, int] | None = None
    tokenizer_status: str = ""
    mean_reward_tokens: float | None = None
    median_reward_tokens: float | None = None
    mean_reasoning_reward_tokens: float | None = None
    median_reasoning_reward_tokens: float | None = None
    all_nonflat_pct: float | None = None
    reasoning_nonflat_pct: float | None = None
    interior_nonflat_pct: float | None = None
    flat_until_last_pct: float | None = None
    mean_all_range: float | None = None
    median_all_range: float | None = None
    mean_reasoning_range: float | None = None
    median_reasoning_range: float | None = None
    mean_all_std: float | None = None
    mean_reasoning_std: float | None = None
    mean_abs_adjacent_delta: float | None = None
    mean_abs_reasoning_adjacent_delta: float | None = None
    mean_adjacent_change_pct: float | None = None
    mean_reasoning_adjacent_change_pct: float | None = None
    median_prefix_range_share: float | None = None
    median_final_jump_share: float | None = None
    mean_unique_reward_values: float | None = None
    mean_tokenizer_reward_len_ratio: float | None = None


def _generation_text(row: dict[str, Any]) -> str:
    generation = row.get("generation")
    if isinstance(generation, dict):
        return str(generation.get("content", ""))
    return str(generation or "")


def _reward_scores(row: dict[str, Any]) -> list[float] | None:
    scores = row.get("reward_model_score")
    if not isinstance(scores, list):
        return None
    out: list[float] = []
    for value in scores:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            out.append(number)
    return out


def _reasoning_char_span(text: str) -> tuple[int, int, str]:
    open_tag = text.find("<think>")
    close_tag = text.find("</think>")
    if open_tag >= 0 and close_tag > open_tag:
        return open_tag + len("<think>"), close_tag, "think"

    answer_tag = text.find("<answer>")
    if answer_tag > 0:
        return 0, answer_tag, "pre_answer"

    return 0, len(text), "full_completion"


def _load_tokenizer(model: str):
    if AutoTokenizer is None:
        return None, "transformers_unavailable"
    try:
        return (
            AutoTokenizer.from_pretrained(
                TOKENIZERS[model],
                local_files_only=True,
                trust_remote_code=True,
            ),
            "ok",
        )
    except Exception as exc:  # pragma: no cover - environment dependent.
        return None, f"fallback_char_alignment: {type(exc).__name__}: {exc}"


def _reward_slice_for_char_span(
    text: str,
    start_char: int,
    end_char: int,
    reward_len: int,
    tokenizer: Any,
) -> tuple[int, int, int | None]:
    if reward_len <= 0:
        return 0, 0, None

    if tokenizer is not None:
        try:
            enc = tokenizer(
                text,
                add_special_tokens=False,
                return_offsets_mapping=True,
                truncation=False,
            )
            offsets = enc.get("offset_mapping") or []
            token_len = len(offsets)
            token_indices = [
                idx
                for idx, (start, end) in enumerate(offsets)
                if end > start_char and start < end_char and end > start
            ]
            if token_indices and token_len > 0:
                tok_start = min(token_indices)
                tok_end = max(token_indices) + 1
                reward_start = int(math.floor(tok_start * reward_len / token_len))
                reward_end = int(math.ceil(tok_end * reward_len / token_len))
                reward_start = max(0, min(reward_start, reward_len))
                reward_end = max(reward_start, min(reward_end, reward_len))
                return reward_start, reward_end, token_len
        except Exception:
            pass

    text_len = max(len(text), 1)
    reward_start = int(math.floor(start_char * reward_len / text_len))
    reward_end = int(math.ceil(end_char * reward_len / text_len))
    reward_start = max(0, min(reward_start, reward_len))
    reward_end = max(reward_start, min(reward_end, reward_len))
    return reward_start, reward_end, None


def _range(arr: np.ndarray) -> float:
    if arr.size == 0:
        return float("nan")
    return float(np.nanmax(arr) - np.nanmin(arr))


def _std(arr: np.ndarray) -> float:
    if arr.size == 0:
        return float("nan")
    return float(np.nanstd(arr))


def _mean_abs_diff(arr: np.ndarray) -> float:
    if arr.size < 2:
        return float("nan")
    return float(np.mean(np.abs(np.diff(arr))))


def _adjacent_change_pct(arr: np.ndarray) -> float:
    if arr.size < 2:
        return float("nan")
    return float(np.mean(np.abs(np.diff(arr)) > EPS) * 100.0)


def _percent(flags: list[bool]) -> float | None:
    if not flags:
        return None
    return float(np.mean(flags) * 100.0)


def _mean(values: list[float]) -> float | None:
    clean = [value for value in values if math.isfinite(value)]
    if not clean:
        return None
    return float(np.mean(clean))


def _median(values: list[float]) -> float | None:
    clean = [value for value in values if math.isfinite(value)]
    if not clean:
        return None
    return float(np.median(clean))


def _summarize_trace(dataset: str, model: str, path: Path) -> RowStats:
    if not path.exists():
        return RowStats(
            dataset=dataset,
            model=model,
            status="missing_raw_eval_trace",
            source_path=str(path),
        )

    tokenizer, tokenizer_status = _load_tokenizer(model)
    marker_counts: Counter[str] = Counter()
    n_generations = 0
    n_valid = 0
    n_missing_scores = 0
    n_short_scores = 0

    reward_lens: list[float] = []
    reasoning_lens: list[float] = []
    token_len_ratios: list[float] = []
    all_ranges: list[float] = []
    reasoning_ranges: list[float] = []
    all_stds: list[float] = []
    reasoning_stds: list[float] = []
    abs_deltas: list[float] = []
    reasoning_abs_deltas: list[float] = []
    adjacent_change_pcts: list[float] = []
    reasoning_adjacent_change_pcts: list[float] = []
    prefix_range_shares: list[float] = []
    final_jump_shares: list[float] = []
    unique_values: list[float] = []

    all_nonflat: list[bool] = []
    reasoning_nonflat: list[bool] = []
    interior_nonflat: list[bool] = []
    flat_until_last: list[bool] = []

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            n_generations += 1
            row = json.loads(line)
            scores = _reward_scores(row)
            if not scores:
                n_missing_scores += 1
                continue
            if len(scores) < 2:
                n_short_scores += 1
                continue

            text = _generation_text(row)
            start_char, end_char, marker = _reasoning_char_span(text)
            marker_counts[marker] += 1
            reason_start, reason_end, token_len = _reward_slice_for_char_span(
                text, start_char, end_char, len(scores), tokenizer
            )

            arr = np.asarray(scores, dtype=float)
            reason = arr[reason_start:reason_end]
            if reason.size < 2:
                reason = arr

            diff_abs = np.abs(np.diff(arr))
            all_range = _range(arr)
            reasoning_range = _range(reason)
            prefix_range = _range(arr[:-1]) if arr.size > 2 else 0.0
            total_abs_delta = float(np.sum(diff_abs))
            final_jump = float(diff_abs[-1]) if diff_abs.size else 0.0

            n_valid += 1
            reward_lens.append(float(arr.size))
            reasoning_lens.append(float(reason.size))
            if token_len:
                token_len_ratios.append(float(arr.size / token_len))
            all_ranges.append(all_range)
            reasoning_ranges.append(reasoning_range)
            all_stds.append(_std(arr))
            reasoning_stds.append(_std(reason))
            abs_deltas.append(_mean_abs_diff(arr))
            reasoning_abs_deltas.append(_mean_abs_diff(reason))
            adjacent_change_pcts.append(_adjacent_change_pct(arr))
            reasoning_adjacent_change_pcts.append(_adjacent_change_pct(reason))
            prefix_range_shares.append(prefix_range / all_range if all_range > EPS else 0.0)
            final_jump_shares.append(final_jump / total_abs_delta if total_abs_delta > EPS else 0.0)
            unique_values.append(float(len(np.unique(np.round(arr, 6)))))

            all_nonflat.append(all_range > EPS)
            reasoning_nonflat.append(reasoning_range > EPS)
            interior_nonflat.append(prefix_range > EPS)
            flat_until_last.append(prefix_range <= EPS and all_range > EPS)

    return RowStats(
        dataset=dataset,
        model=model,
        status="ok" if n_valid else "no_valid_reward_scores",
        source_path=str(path),
        n_generations=n_generations,
        n_valid=n_valid,
        n_missing_scores=n_missing_scores,
        n_short_scores=n_short_scores,
        marker_counts=dict(marker_counts),
        tokenizer_status=tokenizer_status,
        mean_reward_tokens=_mean(reward_lens),
        median_reward_tokens=_median(reward_lens),
        mean_reasoning_reward_tokens=_mean(reasoning_lens),
        median_reasoning_reward_tokens=_median(reasoning_lens),
        all_nonflat_pct=_percent(all_nonflat),
        reasoning_nonflat_pct=_percent(reasoning_nonflat),
        interior_nonflat_pct=_percent(interior_nonflat),
        flat_until_last_pct=_percent(flat_until_last),
        mean_all_range=_mean(all_ranges),
        median_all_range=_median(all_ranges),
        mean_reasoning_range=_mean(reasoning_ranges),
        median_reasoning_range=_median(reasoning_ranges),
        mean_all_std=_mean(all_stds),
        mean_reasoning_std=_mean(reasoning_stds),
        mean_abs_adjacent_delta=_mean(abs_deltas),
        mean_abs_reasoning_adjacent_delta=_mean(reasoning_abs_deltas),
        mean_adjacent_change_pct=_mean(adjacent_change_pcts),
        mean_reasoning_adjacent_change_pct=_mean(reasoning_adjacent_change_pcts),
        median_prefix_range_share=_median(prefix_range_shares),
        median_final_jump_share=_median(final_jump_shares),
        mean_unique_reward_values=_mean(unique_values),
        mean_tokenizer_reward_len_ratio=_mean(token_len_ratios),
    )


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.1f}"


def _fmt_num(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _build_md(rows: list[RowStats]) -> str:
    lines = [
        "# Dense Reward Variation Stats",
        "",
        "Reasoning span is `<think>...</think>` when present; otherwise it is the generated text before the first `<answer>` tag. Values are computed over individual generations.",
        "",
        "| Dataset | Model | n | Reward toks med. | Reasoning toks med. | Overall non-flat % | Reasoning non-flat % | Interior non-flat % | Flat-until-last % | Mean reasoning range | Mean abs Δ reasoning | Median final-jump share |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        model_label = MODEL_NAMES.get(row.model, row.model)
        if row.status != "ok":
            lines.append(
                f"| {row.dataset} | {model_label} | {row.status} | - | - | - | - | - | - | - | - | - |"
            )
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    row.dataset,
                    model_label,
                    str(row.n_valid),
                    _fmt_num(row.median_reward_tokens, 0),
                    _fmt_num(row.median_reasoning_reward_tokens, 0),
                    _fmt_pct(row.all_nonflat_pct),
                    _fmt_pct(row.reasoning_nonflat_pct),
                    _fmt_pct(row.interior_nonflat_pct),
                    _fmt_pct(row.flat_until_last_pct),
                    _fmt_num(row.mean_reasoning_range),
                    _fmt_num(row.mean_abs_reasoning_adjacent_delta),
                    _fmt_num(row.median_final_jump_share),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "Additional definitions:",
            "",
            "- `Interior non-flat` means rewards vary before the final token.",
            "- `Flat-until-last` means all pre-final rewards are constant but the full trace is not.",
            "- `Median final-jump share` is the final adjacent jump divided by total absolute adjacent movement; values near 1 would indicate end-only variation.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    rows = [
        _summarize_trace(dataset, model, path)
        for (dataset, model), path in TRACE_PATHS.items()
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps([asdict(row) for row in rows], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    OUT_MD.write_text(_build_md(rows), encoding="utf-8")
    print(_build_md(rows))
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
