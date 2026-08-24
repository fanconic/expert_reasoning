"""Reward-model localisation scoring for saved clean/perturbed trace pairs.

This script is the fixed-pair counterpart of ``gsm8k_process_sensitivity.py``:
it does not create perturbations. Instead, it reads a JSONL file containing
``clean_text`` and ``pert_text``/``wrong_text`` pairs, scores both traces with a
learned reward model, and evaluates single-point localisation from the largest
downward reward step in the perturbed trace.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Sequence

os.environ.setdefault("UNSLOTH_COMPILE_OVERWRITE", "0")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

from src.eval.gsm8k_process_sensitivity import (
    aggregate_score,
    changed_token_positions,
    load_reward_model_and_tokenizer,
    score_with_reward_model,
)
from src.utils.utils import set_seed


DEFAULT_PAIRS = (
    PROJECT_ROOT
    / "localisation/chatgpt_step_perturbations/"
    / "gsm8k_qwen7b_sft_step_perturbations_full.jsonl"
)
DEFAULT_WINDOWS = [1, 7]
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_ALPHA = 0.05
BOOTSTRAP_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-jsonl", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=str,
        default="configs/math/qwen7b/irl_eval.yaml",
        help="Model config used for reward model defaults.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Checkpoint dir containing reward_model/adapter_config.json.",
    )
    parser.add_argument(
        "--reward-name",
        type=str,
        default=None,
        help="Base reward model path/name. Defaults to model.reward_name from config.",
    )
    parser.add_argument("--reward-lora-rank", type=int, default=None)
    parser.add_argument("--reward-gpu-memory-utilization", type=float, default=None)
    parser.add_argument(
        "--dense-reward-mode",
        type=str,
        required=True,
        choices=["full", "partial_fixed", "partial", "sparse"],
        help="'full' is dense per-token reward; 'partial_fixed' is the interval reward.",
    )
    parser.add_argument("--dense-partial-fixed-n", type=int, default=15)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-micro-batch", type=int, default=8)
    parser.add_argument("--aggregate", type=str, default="mean", choices=["mean", "last", "discounted_mean"])
    parser.add_argument("--discount-gamma", type=float, default=0.95)
    parser.add_argument("--windows", type=int, nargs="+", default=DEFAULT_WINDOWS)
    parser.add_argument(
        "--target-position-source",
        type=str,
        default="target_char_span",
        choices=["target_char_span", "target_char_start", "step_first_diff", "diff"],
        help=(
            "Use saved edited-step char spans, the span-start token, the first "
            "perturbed-step token that differs from original_step, or token diffs."
        ),
    )
    parser.add_argument("--include-text", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--bootstrap-alpha", type=float, default=BOOTSTRAP_ALPHA)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def _to_int(x: Any) -> int | None:
    try:
        return int(x)
    except Exception:
        return None


def _row_pert_text(row: dict[str, Any]) -> str | None:
    for key in ("pert_text", "wrong_text"):
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _row_target_char_span(row: dict[str, Any], pert_text: str) -> list[int] | None:
    span = row.get("target_char_span")
    if isinstance(span, (list, tuple)) and len(span) == 2:
        start = _to_int(span[0])
        end = _to_int(span[1])
        if start is not None and end is not None and end > start:
            return [int(start), int(end)]

    step = row.get("perturbed_step")
    if isinstance(step, str) and step:
        start = pert_text.find(step)
        if start >= 0:
            return [int(start), int(start + len(step))]
    return None


def _token_spans(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(0), int(m.start()), int(m.end())) for m in re.finditer(r"\w+|[^\w\s]", text or "")]


def _row_step_first_diff_char_span(row: dict[str, Any], pert_text: str) -> list[int] | None:
    original_step = row.get("original_step")
    perturbed_step = row.get("perturbed_step")
    full_step_span = _row_target_char_span(row, pert_text)
    if (
        not isinstance(original_step, str)
        or not isinstance(perturbed_step, str)
        or full_step_span is None
    ):
        return None

    old_tokens = _token_spans(original_step)
    new_tokens = _token_spans(perturbed_step)
    if not new_tokens:
        return None

    sm = SequenceMatcher(
        a=[tok for tok, _start, _end in old_tokens],
        b=[tok for tok, _start, _end in new_tokens],
        autojunk=False,
    )
    for tag, _i1, _i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        if j1 < j2:
            _tok, start, end = new_tokens[j1]
        else:
            fallback_idx = min(j1, len(new_tokens) - 1)
            _tok, start, end = new_tokens[fallback_idx]
        return [int(full_step_span[0] + start), int(full_step_span[0] + end)]
    return [int(full_step_span[0]), int(full_step_span[0] + max(1, new_tokens[0][2] - new_tokens[0][1]))]


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

    try:
        enc = tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
        )
    except Exception:
        return []

    positions: list[int] = []
    for idx, offset in enumerate(enc.get("offset_mapping", [])):
        if not isinstance(offset, (list, tuple)) or len(offset) != 2:
            continue
        tok_start = _to_int(offset[0])
        tok_end = _to_int(offset[1])
        if tok_start is None or tok_end is None or tok_end <= tok_start:
            continue
        if tok_end > start and tok_start < end:
            positions.append(int(idx))
    return positions


def _first_token_position_from_char_span(
    tokenizer,
    text: str,
    char_span: Sequence[int] | None,
    max_length: int,
) -> list[int]:
    positions = _token_positions_from_char_span(
        tokenizer=tokenizer,
        text=text,
        char_span=char_span,
        max_length=max_length,
    )
    return positions[:1]


def _select_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        clean = row.get("clean_text")
        pert = _row_pert_text(row)
        if row.get("error") is not None:
            continue
        if not isinstance(clean, str) or not isinstance(pert, str):
            continue
        row = dict(row)
        row["pert_text"] = pert
        selected.append(row)

    selected = selected[int(args.start_index) :]
    if int(args.max_examples) > 0:
        selected = selected[: int(args.max_examples)]
    return selected


def _prediction_index(seq: Sequence[float]) -> int | None:
    if len(seq) < 2:
        return None
    arr = np.asarray(seq, dtype=np.float64)
    finite = np.isfinite(arr)
    if finite.sum() < 2:
        return None
    if not finite.all():
        arr = arr[finite]
    drops = np.maximum(0.0, arr[:-1] - arr[1:])
    return int(np.argmax(drops)) + 1


def _hit_and_chance(
    pred_idx: int | None,
    target_positions: Sequence[int],
    seq_len: int,
    window: int,
    localization_mode: str,
    partial_fixed_stride: int,
) -> tuple[int | None, float]:
    if seq_len <= 0:
        return None, float("nan")

    if localization_mode == "bucket":
        stride = max(1, int(partial_fixed_stride))
        n_units = int(math.ceil(seq_len / float(stride)))
        targets = sorted({int(t) // stride for t in target_positions if 0 <= int(t) < seq_len})
        pred_unit = None if pred_idx is None else int(pred_idx) // stride
        w = int(math.ceil(max(0, int(window)) / float(stride)))
    else:
        n_units = int(seq_len)
        targets = sorted({int(t) for t in target_positions if 0 <= int(t) < seq_len})
        pred_unit = None if pred_idx is None else int(pred_idx)
        w = int(max(0, window))

    if n_units <= 0 or not targets:
        return None, float("nan")

    local_mask = np.zeros(n_units, dtype=bool)
    for target in targets:
        lo = max(0, target - w)
        hi = min(n_units - 1, target + w)
        local_mask[lo : hi + 1] = True
    chance = float(local_mask.mean())

    if pred_unit is None:
        return None, chance
    hit = int(any(abs(pred_unit - target) <= w for target in targets))
    return hit, chance


def _mean(values: Sequence[float]) -> float | None:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _bootstrap_ci_halfwidth(
    values: Sequence[float],
    samples: int,
    alpha: float,
    seed: int,
) -> float | None:
    vals = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=np.float64)
    n = int(vals.shape[0])
    if n == 0:
        return None
    if n == 1:
        return 0.0
    rng = np.random.default_rng(int(seed))
    n_boot = max(100, int(samples))
    alpha = min(max(float(alpha), 1e-6), 0.5)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = vals[idx].mean(axis=1)
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return float((hi - lo) / 2.0)


def _summarize_metric(
    rows: Sequence[dict[str, Any]],
    key: str,
    windows: Sequence[int],
    bootstrap_samples: int,
    bootstrap_alpha: float,
    bootstrap_seed: int,
) -> dict[str, dict[str, float | int | None]]:
    summary: dict[str, dict[str, float | int | None]] = {}
    for window in windows:
        vals = []
        for row in rows:
            value = row.get(key, {}).get(str(window), None)
            if value is not None:
                vals.append(float(value))
        summary[str(window)] = {
            "mean": _mean(vals),
            "ci_halfwidth": _bootstrap_ci_halfwidth(
                vals,
                samples=bootstrap_samples,
                alpha=bootstrap_alpha,
                seed=bootstrap_seed,
            ),
            "n": len(vals),
        }
    return summary


def _dense_mode_for_config(mode: str) -> bool | str:
    if mode == "sparse":
        return False
    if mode == "full":
        return True
    return mode


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.load(args.config)
    cfg.airl = True
    cfg.model.name = str(args.checkpoint_dir)
    if args.reward_name is not None:
        cfg.model.reward_name = args.reward_name
    if args.reward_lora_rank is not None:
        cfg.model.reward_lora_rank = int(args.reward_lora_rank)
    if args.reward_gpu_memory_utilization is not None:
        cfg.model.reward_gpu_memory_utilization = float(args.reward_gpu_memory_utilization)
    if args.load_in_4bit is not None:
        cfg.model.load_in_4bit = bool(args.load_in_4bit)
    cfg.model.dense_rewards = _dense_mode_for_config(args.dense_reward_mode)
    cfg.model.dense_partial_fixed_n = int(args.dense_partial_fixed_n)
    if not cfg.model.get("reward_lora_rank"):
        cfg.model.reward_lora_rank = 32
    if not cfg.model.get("reward_gpu_memory_utilization"):
        cfg.model.reward_gpu_memory_utilization = 0.2

    rows_all = _load_jsonl(args.pairs_jsonl)
    rows = _select_rows(rows_all, args)
    if not rows:
        raise ValueError("No successful clean/perturbed pairs selected for scoring.")

    reward_model, reward_tokenizer = load_reward_model_and_tokenizer(cfg)
    max_length = int(cfg.model.max_prompt_length + cfg.model.max_completion_length)
    dense_reward = cfg.model.dense_rewards
    partial_fixed_stride = int(getattr(cfg.model, "dense_partial_fixed_n", 15))
    localization_mode = (
        "bucket"
        if isinstance(dense_reward, str) and dense_reward == "partial_fixed"
        else "token"
    )

    prompts_for_scoring = [row["prompt"] for row in rows]
    completions_for_scoring = [
        [{"content": row["clean_text"]}, {"content": row["pert_text"]}]
        for row in rows
    ]

    print(f"Pairs: {args.pairs_jsonl}")
    print(f"Selected rows: {len(rows)} / {len(rows_all)}")
    print(f"Checkpoint: {args.checkpoint_dir}")
    print(f"Dense reward mode: {args.dense_reward_mode}")
    print(f"Max length: {max_length}")

    scores = score_with_reward_model(
        reward_model=reward_model,
        reward_tokenizer=reward_tokenizer,
        prompts_msgs=prompts_for_scoring,
        decoded_per_prompt=completions_for_scoring,
        dense_reward=dense_reward,
        max_length=max_length,
        micro_batch=int(args.max_micro_batch),
        clip_reward_model=bool(getattr(cfg.model, "clip_reward_model", False)),
        reward_lb=float(getattr(cfg.model, "reward_lb", -5.0)),
        reward_ub=float(getattr(cfg.model, "reward_ub", 5.0)),
        dense_partial_fixed_n=partial_fixed_stride,
    )

    detail_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(tqdm(rows, desc="Building reward pair details")):
        clean_seq = np.asarray(scores[idx][0], dtype=np.float64)
        pert_seq = np.asarray(scores[idx][1], dtype=np.float64)
        clean_seq = clean_seq[np.isfinite(clean_seq)]
        pert_seq = pert_seq[np.isfinite(pert_seq)]
        seq_len = int(pert_seq.shape[0])

        diff_changed = changed_token_positions(
            reward_tokenizer=reward_tokenizer,
            prompt_messages=row["prompt"],
            clean_text=row["clean_text"],
            pert_text=row["pert_text"],
            max_length=max_length,
        )
        target_span = _row_target_char_span(row, row["pert_text"])
        target_position_source = "diff"
        changed = diff_changed
        target_error_span = None
        if args.target_position_source in {"target_char_span", "target_char_start", "step_first_diff"}:
            if args.target_position_source == "step_first_diff":
                target_error_span = _row_step_first_diff_char_span(row, row["pert_text"])
                span_changed = _token_positions_from_char_span(
                    tokenizer=reward_tokenizer,
                    text=row["pert_text"],
                    char_span=target_error_span,
                    max_length=max_length,
                )
            elif args.target_position_source == "target_char_start":
                span_changed = _first_token_position_from_char_span(
                    tokenizer=reward_tokenizer,
                    text=row["pert_text"],
                    char_span=target_span,
                    max_length=max_length,
                )
            else:
                span_changed = _token_positions_from_char_span(
                    tokenizer=reward_tokenizer,
                    text=row["pert_text"],
                    char_span=target_span,
                    max_length=max_length,
                )
            if span_changed:
                changed = span_changed
                target_position_source = args.target_position_source
            else:
                target_position_source = "diff_fallback"

        pred_idx = _prediction_index(pert_seq.tolist())
        reward_hit: dict[str, int | None] = {}
        random_hit: dict[str, float] = {}
        for window in args.windows:
            hit, chance = _hit_and_chance(
                pred_idx=pred_idx,
                target_positions=changed,
                seq_len=seq_len,
                window=int(window),
                localization_mode=localization_mode,
                partial_fixed_stride=partial_fixed_stride,
            )
            reward_hit[str(window)] = hit
            random_hit[str(window)] = chance

        clean_agg = aggregate_score(clean_seq, args.aggregate, float(args.discount_gamma))
        pert_agg = aggregate_score(pert_seq, args.aggregate, float(args.discount_gamma))
        margin = float(clean_agg - pert_agg)

        out_row: dict[str, Any] = {
            "prompt_idx": row.get("prompt_idx"),
            "source_row_idx": idx,
            "severity": int(row.get("severity", 1) or 1),
            "variant_idx": int(row.get("variant_idx", 0) or 0),
            "clean_generation_idx": row.get("clean_generation_idx"),
            "answer": row.get("answer"),
            "source_pairs_jsonl": str(args.pairs_jsonl),
            "source_pair_details": row.get("source_pair_details"),
            "reward_checkpoint_dir": str(args.checkpoint_dir),
            "dense_reward_mode": args.dense_reward_mode,
            "localization_mode": localization_mode,
            "partial_fixed_stride": partial_fixed_stride if localization_mode == "bucket" else None,
            "target_position_source": target_position_source,
            "target_char_span": target_span,
            "target_error_char_span": target_error_span,
            "target_perturbed_step": row.get("perturbed_step"),
            "changed_token_positions": changed,
            "diff_changed_token_positions": diff_changed,
            "clean_score_seq": clean_seq.tolist(),
            "pert_score_seq": pert_seq.tolist(),
            "clean_score_agg": clean_agg,
            "pert_score_agg": pert_agg,
            "margin": margin,
            "win": int(margin > 0.0) if math.isfinite(margin) else None,
            "reward_detector": "largest_drop",
            "reward_pred_idx": pred_idx,
            "reward_hit1_at_window": reward_hit,
            "random_hit1_at_window": random_hit,
        }
        if args.include_text:
            out_row["prompt"] = row.get("prompt")
            out_row["question"] = row.get("question")
            out_row["clean_text"] = row.get("clean_text")
            out_row["pert_text"] = row.get("pert_text")
            out_row["original_step"] = row.get("original_step")
            out_row["perturbed_step"] = row.get("perturbed_step")
            out_row["corruption_summary"] = row.get("corruption_summary")
        detail_rows.append(out_row)

    summary = {
        "source_pairs_jsonl": str(args.pairs_jsonl),
        "checkpoint_dir": str(args.checkpoint_dir),
        "reward_name": str(cfg.model.reward_name),
        "dense_reward_mode": args.dense_reward_mode,
        "dense_rewards_config_value": dense_reward,
        "localization_mode": localization_mode,
        "partial_fixed_stride": partial_fixed_stride if localization_mode == "bucket" else None,
        "max_length": max_length,
        "n_rows_total": len(rows_all),
        "n_rows_scored": len(rows),
        "target_position_source": args.target_position_source,
        "windows": [int(w) for w in args.windows],
        "bootstrap": {
            "samples": int(args.bootstrap_samples),
            "alpha": float(args.bootstrap_alpha),
            "seed": int(args.bootstrap_seed),
        },
        "metrics": {
            "reward_largest_drop": _summarize_metric(
                detail_rows,
                key="reward_hit1_at_window",
                windows=args.windows,
                bootstrap_samples=int(args.bootstrap_samples),
                bootstrap_alpha=float(args.bootstrap_alpha),
                bootstrap_seed=int(args.bootstrap_seed),
            ),
            "random_location": _summarize_metric(
                detail_rows,
                key="random_hit1_at_window",
                windows=args.windows,
                bootstrap_samples=int(args.bootstrap_samples),
                bootstrap_alpha=float(args.bootstrap_alpha),
                bootstrap_seed=int(args.bootstrap_seed),
            ),
        },
    }

    run_config = {
        "args": _jsonable(vars(args)),
        "model_name": str(cfg.model.name),
        "reward_name": str(cfg.model.reward_name),
        "dense_rewards": dense_reward,
        "localization_mode": localization_mode,
        "partial_fixed_stride": partial_fixed_stride if localization_mode == "bucket" else None,
        "max_length": max_length,
    }

    detail_path = args.output_dir / "pair_details.jsonl"
    summary_path = args.output_dir / "summary.json"
    run_config_path = args.output_dir / "run_config.json"
    _write_jsonl(detail_path, detail_rows)
    _write_json(summary_path, summary)
    _write_json(run_config_path, run_config)

    print(f"Wrote details: {detail_path}")
    print(f"Wrote summary: {summary_path}")
    print("Summary:")
    for metric_name, by_window in summary["metrics"].items():
        for window, vals in by_window.items():
            mean = vals["mean"]
            ci = vals["ci_halfwidth"]
            if mean is None:
                print(f"  {metric_name} Hit@1@{window}: n/a")
            else:
                print(f"  {metric_name} Hit@1@{window}: {mean:.4f} +/- {ci:.4f}")


if __name__ == "__main__":
    main()
