"""Build a small MedReason clinician localisation annotation set.

The script selects naturally wrong MedReason generations from an eval JSONL,
preferring high reward drops while keeping traces short enough for manual
clinical review. It writes two kinds of artifacts:

* internal metadata with hidden reward/drop fields for later evaluation;
* blinded public case files for annotators.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any, Iterable


DEFAULT_EVAL_JSONL = Path(
    "/mnt/pdata/caf83/neurips2026/medicine/outputs/"
    "transfer_llama8b_partial_fixed_P_medicine_R_medicine/"
    "best_model/eval_results_new.jsonl"
)
DEFAULT_OUTPUT_DIR = Path("localisation/clinician_medreason")

THINK_RE = re.compile(r"<think>(.*?)</think>", flags=re.IGNORECASE | re.DOTALL)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", flags=re.IGNORECASE | re.DOTALL)
SPACE_RE = re.compile(r"\s+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-jsonl", type=Path, default=DEFAULT_EVAL_JSONL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Total cases to select. Defaults to num_annotators * cases_per_annotator.",
    )
    parser.add_argument("--num-annotators", type=int, default=4)
    parser.add_argument("--cases-per-annotator", type=int, default=25)
    parser.add_argument(
        "--assignment-prefix",
        type=str,
        default="doctor",
        help="Prefix for public assignment files, e.g. doctor_1_cases.json.",
    )
    parser.add_argument(
        "--max-reasoning-words",
        type=int,
        default=350,
        help="Keep traces at or below this many words before ranking by reward drop.",
    )
    parser.add_argument(
        "--correct-reward-value",
        type=float,
        default=2.0,
        help="Rows below this correctness_reward_func value are treated as wrong.",
    )
    parser.add_argument(
        "--max-step-words",
        type=int,
        default=45,
        help="Split long trace lines into sentence units when they exceed this length.",
    )
    parser.add_argument(
        "--require-global-max-drop-in-think",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Require the largest reward drop over the whole completion to fall inside "
            "the <think>...</think> span, rather than at the final answer/end."
        ),
    )
    return parser.parse_args()


def _load_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if raw:
                yield line_no, json.loads(raw)


def _write_json(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _prompt_text(prompt: Any) -> str:
    if isinstance(prompt, list):
        for msg in reversed(prompt):
            if isinstance(msg, dict) and msg.get("role") == "user":
                return str(msg.get("content", ""))
        if prompt and isinstance(prompt[-1], dict):
            return str(prompt[-1].get("content", ""))
    return str(prompt or "")


def _generation_text(row: dict[str, Any]) -> str:
    generation = row.get("generation")
    if isinstance(generation, dict):
        return str(generation.get("content", ""))
    return str(generation or "")


def _extract_think(text: str) -> tuple[str, int, int]:
    match = THINK_RE.search(text or "")
    if not match:
        return text or "", 0, len(text or "")
    return match.group(1), int(match.start(1)), int(match.end(1))


def _extract_answer(text: str) -> str:
    match = ANSWER_RE.search(text or "")
    return SPACE_RE.sub(" ", match.group(1)).strip() if match else ""


def _normalize_question(text: str) -> str:
    return SPACE_RE.sub(" ", text or "").strip()


def _question_hash(text: str) -> str:
    return hashlib.sha1(_normalize_question(text).encode("utf-8")).hexdigest()[:10]


def _correctness(row: dict[str, Any]) -> float | None:
    try:
        return float(row.get("correctness_reward_func"))
    except Exception:
        return None


def _reward_drop(
    scores: Any,
    *,
    start_idx: int = 0,
    end_idx: int | None = None,
) -> tuple[float, int | None]:
    if not isinstance(scores, list) or len(scores) < 2:
        return float("-inf"), None
    arr: list[float] = []
    for value in scores:
        try:
            x = float(value)
        except Exception:
            x = float("nan")
        arr.append(x)
    best_drop = float("-inf")
    best_idx: int | None = None
    lo = max(1, int(start_idx) + 1)
    hi = len(arr) if end_idx is None else min(len(arr), int(end_idx))
    if lo >= hi:
        lo, hi = 1, len(arr)
    for idx in range(lo, hi):
        if not math.isfinite(arr[idx - 1]) or not math.isfinite(arr[idx]):
            continue
        drop = arr[idx - 1] - arr[idx]
        if drop > best_drop:
            best_drop = float(drop)
            best_idx = idx
    return best_drop, best_idx


def _approx_reward_span_for_char_span(
    *,
    text: str,
    scores: Any,
    char_start: int,
    char_end: int,
) -> tuple[int, int]:
    if not isinstance(scores, list) or not scores:
        return 0, 0
    text_len = max(1, len(text or ""))
    score_len = len(scores)
    start = int(math.floor(max(0, char_start) / text_len * score_len))
    end = int(math.ceil(min(text_len, max(char_start, char_end)) / text_len * score_len))
    start = max(0, min(score_len - 1, start))
    end = max(start + 2, min(score_len, end))
    return start, end


def _drop_location_metadata(row: dict[str, Any]) -> dict[str, Any]:
    generation = _generation_text(row)
    reasoning, think_start, think_end = _extract_think(generation)
    scores = row.get("reward_model_score")
    span_start, span_end = _approx_reward_span_for_char_span(
        text=generation,
        scores=scores,
        char_start=think_start,
        char_end=think_end,
    )
    full_drop, full_pred_idx = _reward_drop(scores)
    if full_pred_idx is None or not math.isfinite(full_drop):
        location = "no_valid_drop"
    elif full_pred_idx < span_start:
        location = "before_think"
    elif full_pred_idx < span_end:
        location = "inside_think"
    else:
        location = "after_think"
    return {
        "reasoning_text": reasoning,
        "reasoning_reward_token_start_approx": int(span_start),
        "reasoning_reward_token_end_approx": int(span_end),
        "max_completion_token_reward_drop": float(full_drop),
        "model_pred_completion_reward_token_idx": (
            int(full_pred_idx) if full_pred_idx is not None else None
        ),
        "global_max_drop_location": location,
    }


def _is_boilerplate_step(text: str) -> bool:
    clean = text.strip()
    lower = clean.lower().strip(" :")
    if not clean:
        return True
    if re.fullmatch(r"-{2,}", clean):
        return True
    if lower in {
        "finding reasoning paths",
        "finding reasoning path",
        "reasoning process",
        "conclusion",
        "cross-validation",
    }:
        return True
    if clean.endswith(":") and len(clean.split()) <= 8:
        return True
    return False


def _strip_list_marker(text: str, abs_start: int) -> tuple[str, int]:
    match = re.match(r"^((?:\d+[.)]|[-*])\s+)(.*)$", text)
    if not match:
        return text, abs_start
    return match.group(2).strip(), abs_start + len(match.group(1))


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    spans = []
    for match in re.finditer(r"[^.!?\n]+(?:[.!?]+|$)", text):
        chunk = match.group(0)
        if chunk.strip():
            leading = len(chunk) - len(chunk.lstrip())
            trailing = len(chunk.rstrip())
            spans.append((int(match.start() + leading), int(match.start() + trailing)))
    return spans or [(0, len(text))]


def _reasoning_steps(
    generation: str,
    *,
    max_step_words: int,
) -> tuple[str, list[dict[str, Any]]]:
    reasoning, think_start, _think_end = _extract_think(generation)
    steps: list[dict[str, Any]] = []
    for line_match in re.finditer(r"[^\r\n]+", reasoning):
        raw_line = line_match.group(0)
        stripped = raw_line.strip()
        if not stripped:
            continue

        leading = len(raw_line) - len(raw_line.lstrip())
        abs_start = think_start + line_match.start() + leading
        text, text_abs_start = _strip_list_marker(stripped, abs_start)
        if _is_boilerplate_step(text):
            continue

        units = [(0, len(text))]
        if len(text.split()) > int(max_step_words):
            units = _sentence_spans(text)

        for rel_start, rel_end in units:
            unit = text[rel_start:rel_end].strip()
            if _is_boilerplate_step(unit):
                continue
            unit_leading = len(text[rel_start:rel_end]) - len(text[rel_start:rel_end].lstrip())
            start = text_abs_start + rel_start + unit_leading
            end = text_abs_start + rel_end
            steps.append(
                {
                    "step_id": len(steps) + 1,
                    "text": SPACE_RE.sub(" ", unit).strip(),
                    "char_span": [int(start), int(end)],
                }
            )
    return reasoning, steps


def _candidate_from_row(
    line_no: int,
    row: dict[str, Any],
    *,
    correct_reward_value: float,
    max_step_words: int,
    require_global_max_drop_in_think: bool,
) -> dict[str, Any] | None:
    correctness = _correctness(row)
    if correctness is None or correctness >= float(correct_reward_value):
        return None

    generation = _generation_text(row)
    drop_meta = _drop_location_metadata(row)
    if require_global_max_drop_in_think and drop_meta["global_max_drop_location"] != "inside_think":
        return None

    reasoning = str(drop_meta["reasoning_text"])
    _reasoning_for_display, steps = _reasoning_steps(
        generation,
        max_step_words=max_step_words,
    )
    if not steps:
        return None

    scores = row.get("reward_model_score")
    max_drop, pred_idx = _reward_drop(
        scores,
        start_idx=int(drop_meta["reasoning_reward_token_start_approx"]),
        end_idx=int(drop_meta["reasoning_reward_token_end_approx"]),
    )
    if pred_idx is None or not math.isfinite(max_drop) or max_drop <= 0.0:
        return None

    question = _prompt_text(row.get("prompt"))
    q_hash = _question_hash(question)
    return {
        "source_line_no": int(line_no),
        "question_hash": q_hash,
        "prompt": row.get("prompt"),
        "question": question,
        "generation_idx": row.get("generation_idx"),
        "generation": generation,
        "reasoning_text": reasoning,
        "model_final_answer": _extract_answer(generation),
        "steps": steps,
        "word_count": len(reasoning.split()),
        "step_count": len(steps),
        "correctness_reward_func": correctness,
        "reward_model_score": scores,
        "reward_score_len": len(scores or []),
        "max_reasoning_token_reward_drop": float(max_drop),
        "model_pred_reasoning_reward_token_idx": int(pred_idx),
        **drop_meta,
        "source_eval_jsonl": str(DEFAULT_EVAL_JSONL),
    }


def _public_case(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": row["case_id"],
        "rank": row["rank"],
        "question": row["question"],
        "model_final_answer": row["model_final_answer"],
        "steps": [
            {"step_id": step["step_id"], "text": step["text"]}
            for step in row["steps"]
        ],
    }


def _write_review_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case_id",
                "assignment",
                "rank",
                "source_line_no",
                "generation_idx",
                "max_reasoning_token_reward_drop",
                "word_count",
                "step_count",
                "question",
                "model_final_answer",
                "steps",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "case_id": row["case_id"],
                    "assignment": row["assignment"],
                    "rank": row["rank"],
                    "source_line_no": row["source_line_no"],
                    "generation_idx": row["generation_idx"],
                    "max_reasoning_token_reward_drop": (
                        f"{row['max_reasoning_token_reward_drop']:.6f}"
                    ),
                    "word_count": row["word_count"],
                    "step_count": row["step_count"],
                    "question": row["question"],
                    "model_final_answer": row["model_final_answer"],
                    "steps": "\n".join(
                        f"{step['step_id']}. {step['text']}" for step in row["steps"]
                    ),
                }
            )


def main() -> None:
    args = parse_args()
    num_annotators = int(args.num_annotators)
    cases_per_annotator = int(args.cases_per_annotator)
    if num_annotators < 1:
        raise ValueError("--num-annotators must be positive.")
    if cases_per_annotator < 1:
        raise ValueError("--cases-per-annotator must be positive.")
    max_cases = (
        int(args.max_cases)
        if args.max_cases is not None
        else num_annotators * cases_per_annotator
    )
    if max_cases < 1:
        raise ValueError("--max-cases must be positive.")
    assignment_names = [
        f"{str(args.assignment_prefix).strip() or 'doctor'}_{idx}"
        for idx in range(1, num_annotators + 1)
    ]

    candidates: list[dict[str, Any]] = []
    wrong_rows = 0
    global_drop_location_counts: Counter[str] = Counter()
    short_global_drop_location_counts: Counter[str] = Counter()
    for line_no, row in _load_jsonl(args.eval_jsonl):
        correctness = _correctness(row)
        if correctness is not None and correctness < float(args.correct_reward_value):
            wrong_rows += 1
            drop_meta = _drop_location_metadata(row)
            global_drop_location_counts[str(drop_meta["global_max_drop_location"])] += 1
            generation = _generation_text(row)
            reasoning = str(drop_meta["reasoning_text"])
            _reasoning_for_display, steps = _reasoning_steps(
                generation,
                max_step_words=int(args.max_step_words),
            )
            if steps and len(reasoning.split()) <= int(args.max_reasoning_words):
                short_global_drop_location_counts[
                    str(drop_meta["global_max_drop_location"])
                ] += 1
        candidate = _candidate_from_row(
            line_no,
            row,
            correct_reward_value=float(args.correct_reward_value),
            max_step_words=int(args.max_step_words),
            require_global_max_drop_in_think=bool(args.require_global_max_drop_in_think),
        )
        if candidate is not None:
            candidate["source_eval_jsonl"] = str(args.eval_jsonl)
            candidates.append(candidate)

    short_candidates = [
        row for row in candidates
        if int(row["word_count"]) <= int(args.max_reasoning_words)
    ]
    if len(short_candidates) < max_cases:
        raise ValueError(
            f"Only {len(short_candidates)} candidates have word_count <= "
            f"{args.max_reasoning_words}; lower --max-cases or increase the cap."
        )

    best_by_question: dict[str, dict[str, Any]] = {}
    for row in short_candidates:
        old = best_by_question.get(row["question_hash"])
        if old is None:
            best_by_question[row["question_hash"]] = row
            continue
        key = (float(row["max_reasoning_token_reward_drop"]), -int(row["word_count"]))
        old_key = (
            float(old["max_reasoning_token_reward_drop"]),
            -int(old["word_count"]),
        )
        if key > old_key:
            best_by_question[row["question_hash"]] = row

    selected = sorted(
        best_by_question.values(),
        key=lambda row: (
            -float(row["max_reasoning_token_reward_drop"]),
            int(row["word_count"]),
        ),
    )[:max_cases]
    if len(selected) < max_cases:
        raise ValueError(
            f"Only {len(selected)} deduplicated candidates remain; cannot build "
            f"{max_cases} cases."
        )

    for idx, row in enumerate(selected, start=1):
        row["rank"] = idx
        row["case_id"] = f"MRLOC-{idx:03d}"
        row["assignment"] = assignment_names[(idx - 1) % len(assignment_names)]

    public_cases = [_public_case(row) for row in selected]
    assigned_cases = {
        name: [_public_case(row) for row in selected if row["assignment"] == name]
        for name in assignment_names
    }

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_dir / "selection_internal.jsonl", selected)
    _write_json(out_dir / "cases_public.json", public_cases)
    for name, rows in assigned_cases.items():
        _write_json(out_dir / f"{name}_cases.json", rows)
    _write_json(
        out_dir / "assignment_manifest.json",
        {
            "assignment_names": assignment_names,
            "cases_per_annotator": cases_per_annotator,
            "selected_cases": len(selected),
        },
    )
    _write_review_csv(out_dir / "selection_review.csv", selected)

    word_counts = [int(row["word_count"]) for row in selected]
    step_counts = [int(row["step_count"]) for row in selected]
    drops = [float(row["max_reasoning_token_reward_drop"]) for row in selected]
    summary = {
        "eval_jsonl": str(args.eval_jsonl),
        "selection_rule": (
            "wrong rows only; reasoning word_count <= max_reasoning_words; "
            "deduplicate by question keeping highest max_reasoning_token_reward_drop; "
            "select top max_cases by max_reasoning_token_reward_drop with word_count tie-break"
        ),
        "correct_reward_value": float(args.correct_reward_value),
        "max_reasoning_words": int(args.max_reasoning_words),
        "max_step_words": int(args.max_step_words),
        "require_global_max_drop_in_think": bool(args.require_global_max_drop_in_think),
        "num_annotators": num_annotators,
        "cases_per_annotator": cases_per_annotator,
        "assignment_names": assignment_names,
        "wrong_rows_seen": int(wrong_rows),
        "global_largest_drop_location_counts_wrong_rows": dict(
            sorted(global_drop_location_counts.items())
        ),
        "global_largest_drop_location_counts_short_wrong_rows": dict(
            sorted(short_global_drop_location_counts.items())
        ),
        "eligible_wrong_rows": int(len(candidates)),
        "short_wrong_rows": int(len(short_candidates)),
        "deduplicated_short_questions": int(len(best_by_question)),
        "selected_cases": int(len(selected)),
        "assignments": {name: len(rows) for name, rows in assigned_cases.items()},
        "max_reasoning_token_reward_drop": {
            "min": min(drops),
            "median": median(drops),
            "max": max(drops),
        },
        "word_count": {
            "min": min(word_counts),
            "median": median(word_counts),
            "max": max(word_counts),
        },
        "step_count": {
            "min": min(step_counts),
            "median": median(step_counts),
            "max": max(step_counts),
        },
    }
    _write_json(out_dir / "selection_summary.json", summary)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
