"""Build an extra MedReason clinician batch with non-final reward drops.

This is a follow-up to the main clinician localisation set. It selects wrong
MedReason generations where the largest reward drop inside the reasoning trace
maps to a displayed reasoning unit before the final numbered unit. The outputs
are isolated from the original 100-case set so earlier annotations remain
reproducible.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import importlib.util
import json
import math
import re
import shutil
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL_JSONL = Path(
    "/mnt/pdata/caf83/neurips2026/medicine/outputs/"
    "transfer_llama8b_partial_fixed_P_medicine_R_medicine/"
    "best_model/eval_results_new.jsonl"
)
DEFAULT_EXISTING_INTERNAL = ROOT / "localisation/clinician_medreason/selection_internal.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "localisation/clinician_medreason/nonfinal_drop_batch"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base_builder = load_module(
    ROOT / "src/eval/build_medreason_clinician_localisation_set.py",
    "medreason_clinician_base_builder",
)
offline_builder = load_module(
    ROOT / "localisation/clinician_medreason/build_offline_package.py",
    "medreason_clinician_offline_builder",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-jsonl", type=Path, default=DEFAULT_EVAL_JSONL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--existing-internal",
        type=Path,
        action="append",
        default=[DEFAULT_EXISTING_INTERNAL],
        help=(
            "Selection-internal JSONL to exclude. Can be passed multiple times; "
            "defaults to the original clinician selection."
        ),
    )
    parser.add_argument("--assignment-name", type=str, default="doctor_5")
    parser.add_argument("--case-prefix", type=str, default="MRLOC-NF")
    parser.add_argument("--cases", type=int, default=25)
    parser.add_argument("--max-reasoning-words", type=int, default=300)
    parser.add_argument("--max-step-words", type=int, default=45)
    parser.add_argument("--correct-reward-value", type=float, default=2.0)
    parser.add_argument(
        "--require-conclusion-final-consistency",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Reject cases where the final reasoning conclusion appears not to "
            "support the final <answer> option."
        ),
    )
    parser.add_argument(
        "--allow-final-drop",
        action="store_true",
        help="Disable the strict filter requiring the predicted drop step to be before the last step.",
    )
    return parser.parse_args()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_existing_ids(paths: list[Path]) -> tuple[set[str], set[int]]:
    question_hashes: set[str] = set()
    source_lines: set[int] = set()
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                question_hashes.add(str(row.get("question_hash")))
                try:
                    source_lines.add(int(row.get("source_line_no")))
                except Exception:
                    pass
    return question_hashes, source_lines


def infer_correct_answers(
    eval_jsonl: Path,
    correct_reward_value: float,
) -> dict[str, str]:
    answers_by_question: dict[str, Counter[str]] = {}
    for _line_no, row in base_builder._load_jsonl(eval_jsonl):
        correctness = base_builder._correctness(row)
        if correctness is None or correctness < float(correct_reward_value):
            continue
        question = base_builder._prompt_text(row.get("prompt"))
        question_hash = base_builder._question_hash(question)
        answer = base_builder._extract_answer(base_builder._generation_text(row))
        if not answer:
            continue
        answers_by_question.setdefault(question_hash, Counter())[answer] += 1

    return {
        question_hash: counts.most_common(1)[0][0]
        for question_hash, counts in answers_by_question.items()
        if counts
    }


def predicted_drop_step(candidate: dict[str, Any]) -> int | None:
    generation = str(candidate.get("generation") or "")
    scores = candidate.get("reward_model_score") or []
    token_idx = candidate.get("model_pred_reasoning_reward_token_idx")
    if token_idx is None or not isinstance(scores, list):
        return None

    spans: list[tuple[int, int, int]] = []
    for step in candidate.get("steps") or []:
        span = step.get("char_span") or [0, 0]
        start, end = base_builder._approx_reward_span_for_char_span(
            text=generation,
            scores=scores,
            char_start=int(span[0]),
            char_end=int(span[1]),
        )
        spans.append((int(step["step_id"]), start, end))
    if not spans:
        return None

    best_step = None
    best_distance = None
    for step_id, start, end in spans:
        if start <= int(token_idx) < end:
            return step_id
        distance = min(abs(int(token_idx) - start), abs(int(token_idx) - max(start, end - 1)))
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_step = step_id
    return best_step


def normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def content_tokens(text: str) -> list[str]:
    stop = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "by",
        "for",
        "in",
        "is",
        "of",
        "on",
        "or",
        "the",
        "to",
        "with",
    }
    return [tok for tok in normalize_text(text).split() if tok not in stop]


def parse_answer_choices(question: str) -> dict[str, str]:
    choices: dict[str, str] = {}
    pattern = re.compile(
        r"(?m)^\s*([A-D])\.\s*(.*?)(?=^\s*[A-D]\.\s*|\Z)",
        flags=re.S,
    )
    for match in pattern.finditer(question or ""):
        choices[match.group(1)] = " ".join(match.group(2).split())
    return choices


def parse_final_answer(final_answer: str) -> tuple[str | None, str]:
    match = re.match(r"\s*([A-D])\.\s*(.*)", final_answer or "", flags=re.S)
    if not match:
        return None, " ".join(str(final_answer or "").split())
    return match.group(1), " ".join(match.group(2).split())


def conclusion_text(reasoning_text: str) -> str:
    text = reasoning_text or ""
    matches = list(re.finditer(r"\bconclusion\s*:", text, flags=re.I))
    if matches:
        return text[matches[-1].end() :]
    return text[-700:]


def fuzzy_token_coverage(needle: str, haystack: str) -> float:
    needle_tokens = content_tokens(needle)
    hay_tokens = content_tokens(haystack)
    if not needle_tokens:
        return 0.0
    hits = 0
    for token in needle_tokens:
        if token in hay_tokens:
            hits += 1
            continue
        if any(difflib.SequenceMatcher(None, token, other).ratio() >= 0.82 for other in hay_tokens):
            hits += 1
    return hits / len(needle_tokens)


def final_answer_supported_by_conclusion(candidate: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    final_letter, final_text = parse_final_answer(str(candidate.get("model_final_answer") or ""))
    choices = parse_answer_choices(str(candidate.get("question") or ""))
    conclusion = conclusion_text(str(candidate.get("reasoning_text") or ""))
    conclusion_norm = normalize_text(conclusion)

    if final_letter is None or final_letter not in choices:
        return False, {
            "reason": "missing_or_unparseable_final_answer",
            "final_letter": final_letter,
        }

    explicit_letter_patterns = [
        r"\b(?:answer|option|choice)\s*(?:is|:)?\s*([A-D])\b",
        r"\b(?:therefore|thus|hence),?\s*(?:the\s*)?(?:answer|option|choice)\s*(?:is|:)?\s*([A-D])\b",
    ]
    explicit_letters: list[str] = []
    for pattern in explicit_letter_patterns:
        explicit_letters.extend(re.findall(pattern, conclusion, flags=re.I))
    explicit_letters = [letter.upper() for letter in explicit_letters]
    if explicit_letters and explicit_letters[-1] != final_letter:
        return False, {
            "reason": "conclusion_mentions_different_option_letter",
            "final_letter": final_letter,
            "conclusion_letter": explicit_letters[-1],
        }

    final_choice_text = choices.get(final_letter, final_text)
    final_score = max(
        fuzzy_token_coverage(final_text, conclusion),
        fuzzy_token_coverage(final_choice_text, conclusion),
    )
    choice_scores = {
        letter: fuzzy_token_coverage(choice_text, conclusion)
        for letter, choice_text in choices.items()
    }
    best_letter = max(choice_scores, key=lambda letter: choice_scores[letter]) if choice_scores else None
    best_score = choice_scores.get(best_letter, 0.0) if best_letter else 0.0

    final_text_norm = normalize_text(final_text)
    exact_final_text_seen = bool(final_text_norm and final_text_norm in conclusion_norm)
    enough_support = final_score >= 0.67 or exact_final_text_seen
    contradicted_by_other_choice = (
        best_letter is not None
        and best_letter != final_letter
        and best_score >= 0.67
        and best_score > final_score + 0.15
    )

    return enough_support and not contradicted_by_other_choice, {
        "reason": "supported" if enough_support and not contradicted_by_other_choice else "final_answer_not_supported_by_conclusion",
        "final_letter": final_letter,
        "final_text": final_text,
        "final_support_score": final_score,
        "best_supported_choice": best_letter,
        "best_supported_choice_score": best_score,
        "exact_final_text_seen": exact_final_text_seen,
    }


def public_case(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": row["case_id"],
        "rank": row["rank"],
        "question": row["question"],
        "model_final_answer": row["model_final_answer"],
        "correct_answer": row.get("correct_answer", ""),
        "steps": [
            {"step_id": step["step_id"], "text": step["text"]}
            for step in row["steps"]
        ],
    }


def write_review_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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
                "model_pred_reasoning_step_id",
                "step_count",
                "word_count",
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
                    "max_reasoning_token_reward_drop": f"{row['max_reasoning_token_reward_drop']:.6f}",
                    "model_pred_reasoning_step_id": row["model_pred_reasoning_step_id"],
                    "step_count": row["step_count"],
                    "word_count": row["word_count"],
                    "question": row["question"],
                    "model_final_answer": row["model_final_answer"],
                    "steps": "\n".join(
                        f"{step['step_id']}. {step['text']}" for step in row["steps"]
                    ),
                }
            )


def build_html_package(out_dir: Path, assignment_name: str, rows: list[dict[str, Any]]) -> None:
    package_dir = out_dir / "offline_package_with_headings"
    if package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir(parents=True)

    cases = [offline_builder.public_case_with_headings(row) for row in rows]
    html = offline_builder.HTML_TEMPLATE.format(
        doctor_label=offline_builder.display_label(assignment_name),
        case_set=assignment_name,
        cases_json=json.dumps(cases, ensure_ascii=True),
    )
    (package_dir / f"{assignment_name}_annotation.html").write_text(html, encoding="utf-8")
    offline_builder.write_answer_sheet(
        package_dir / f"{assignment_name}_blank_answer_sheet.csv",
        cases,
    )
    (package_dir / "README_FOR_DOCTORS.txt").write_text(
        offline_builder.MEDIC_README,
        encoding="utf-8",
    )
    (package_dir / "START_HERE.txt").write_text(
        (
            "START HERE\n\n"
            f"Send {assignment_name}_annotation.html and README_FOR_DOCTORS.txt to the doctor.\n"
            "They can open the HTML file locally in a browser. No website or login is required.\n"
            "At the end, they should click Download Answers and send the downloaded JSONL file back.\n"
        ),
        encoding="utf-8",
    )
    zip_base = out_dir / f"medreason_clinician_{assignment_name}_nonfinal_drop_with_headings"
    shutil.make_archive(str(zip_base), "zip", package_dir)


def main() -> None:
    args = parse_args()
    if args.cases < 1:
        raise ValueError("--cases must be positive")

    used_question_hashes, used_source_lines = load_existing_ids(args.existing_internal)
    correct_answer_by_question = infer_correct_answers(
        args.eval_jsonl,
        correct_reward_value=float(args.correct_reward_value),
    )

    candidates: list[dict[str, Any]] = []
    wrong_rows = 0
    nonfinal_short_rows = 0
    final_short_rows = 0
    conclusion_final_inconsistent_rows = 0
    missing_correct_answer_rows = 0
    for line_no, row in base_builder._load_jsonl(args.eval_jsonl):
        correctness = base_builder._correctness(row)
        if correctness is not None and correctness < float(args.correct_reward_value):
            wrong_rows += 1

        candidate = base_builder._candidate_from_row(
            line_no,
            row,
            correct_reward_value=float(args.correct_reward_value),
            max_step_words=int(args.max_step_words),
            require_global_max_drop_in_think=True,
        )
        if candidate is None:
            continue
        if candidate["question_hash"] in used_question_hashes:
            continue
        if int(candidate["source_line_no"]) in used_source_lines:
            continue
        if int(candidate["word_count"]) > int(args.max_reasoning_words):
            continue
        correct_answer = correct_answer_by_question.get(candidate["question_hash"])
        if not correct_answer:
            missing_correct_answer_rows += 1
            continue
        candidate["correct_answer"] = correct_answer

        pred_step = predicted_drop_step(candidate)
        if pred_step is None:
            continue
        candidate["model_pred_reasoning_step_id"] = int(pred_step)
        is_final_step = int(pred_step) >= int(candidate["step_count"])
        if is_final_step:
            final_short_rows += 1
            if not args.allow_final_drop:
                continue
        else:
            nonfinal_short_rows += 1

        if args.require_conclusion_final_consistency:
            supported, support_meta = final_answer_supported_by_conclusion(candidate)
            candidate["conclusion_final_consistency"] = support_meta
            if not supported:
                conclusion_final_inconsistent_rows += 1
                continue
        else:
            supported, support_meta = final_answer_supported_by_conclusion(candidate)
            candidate["conclusion_final_consistency"] = support_meta
        candidates.append(candidate)

    best_by_question: dict[str, dict[str, Any]] = {}
    for row in candidates:
        old = best_by_question.get(row["question_hash"])
        key = (float(row["max_reasoning_token_reward_drop"]), -int(row["word_count"]))
        if old is None:
            best_by_question[row["question_hash"]] = row
            continue
        old_key = (float(old["max_reasoning_token_reward_drop"]), -int(old["word_count"]))
        if key > old_key:
            best_by_question[row["question_hash"]] = row

    selected = sorted(
        best_by_question.values(),
        key=lambda row: (
            -float(row["max_reasoning_token_reward_drop"]),
            int(row["word_count"]),
        ),
    )[: int(args.cases)]
    if len(selected) < int(args.cases):
        raise ValueError(
            f"Only {len(selected)} deduplicated candidates available; cannot build {args.cases} cases."
        )

    for idx, row in enumerate(selected, start=1):
        row["rank"] = idx
        row["case_id"] = f"{args.case_prefix}-{idx:03d}"
        row["assignment"] = args.assignment_name
        row["source_eval_jsonl"] = str(args.eval_jsonl)
        row["selection_filter"] = "largest_reasoning_drop_step_before_final_reasoning_step"

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "selection_internal.jsonl", selected)
    public_cases = [public_case(row) for row in selected]
    write_json(out_dir / "cases_public.json", public_cases)
    write_json(out_dir / f"{args.assignment_name}_cases.json", public_cases)
    write_json(
        out_dir / "assignment_manifest.json",
        {
            "assignment_names": [args.assignment_name],
            "cases_per_annotator": int(args.cases),
            "selected_cases": len(selected),
            "selection_filter": "largest drop before final reasoning step",
        },
    )
    write_review_csv(out_dir / "selection_review.csv", selected)
    build_html_package(out_dir, args.assignment_name, selected)

    word_counts = [int(row["word_count"]) for row in selected]
    step_counts = [int(row["step_count"]) for row in selected]
    drops = [float(row["max_reasoning_token_reward_drop"]) for row in selected]
    pred_steps = [int(row["model_pred_reasoning_step_id"]) for row in selected]
    summary = {
        "eval_jsonl": str(args.eval_jsonl),
        "existing_internal_excluded": [str(path) for path in args.existing_internal],
        "assignment_name": args.assignment_name,
        "case_prefix": args.case_prefix,
        "selected_cases": len(selected),
        "selection_rule": (
            "wrong rows only; global largest drop inside <think>; reasoning word_count "
            "<= max_reasoning_words; exclude original selection; predicted largest "
            "reasoning-drop step must be before the final displayed reasoning step; "
            "deduplicate by question; rank by max_reasoning_token_reward_drop."
        ),
        "correct_reward_value": float(args.correct_reward_value),
        "max_reasoning_words": int(args.max_reasoning_words),
        "max_step_words": int(args.max_step_words),
        "wrong_rows_seen": int(wrong_rows),
        "eligible_nonfinal_short_rows": int(nonfinal_short_rows),
        "excluded_final_step_short_rows": int(final_short_rows),
        "excluded_conclusion_final_inconsistent_rows": int(conclusion_final_inconsistent_rows),
        "excluded_missing_correct_answer_rows": int(missing_correct_answer_rows),
        "correct_answers_inferred_for_questions": int(len(correct_answer_by_question)),
        "require_conclusion_final_consistency": bool(args.require_conclusion_final_consistency),
        "deduplicated_questions": int(len(best_by_question)),
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
        "model_pred_reasoning_step_id": {
            "min": min(pred_steps),
            "median": median(pred_steps),
            "max": max(pred_steps),
        },
        "zip_path": str(
            out_dir / f"medreason_clinician_{args.assignment_name}_nonfinal_drop_with_headings.zip"
        ),
    }
    write_json(out_dir / "selection_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
