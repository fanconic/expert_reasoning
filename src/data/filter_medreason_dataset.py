#!/usr/bin/env python3
"""
Filter the MedReason dataset by removing all token-length violating examples.

This script does NOT trim/cut text. It removes any example that violates:
  - prompt length > max_prompt_tokens (for any configured tokenizer)
  - target length > max_target_tokens (for base or corrupted targets, any tokenizer)

After filtering, it optionally rebalances splits exactly as requested:
  1) Move examples from eval -> test until test reaches target_test_size
  2) Move remaining eval examples -> train
  3) Leave eval empty

Outputs:
  - HuggingFace dataset saved with save_to_disk(output_path)
  - Optional split exports: train/eval/test JSONL + CSV
  - JSON report with detailed stats
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import shutil
import unicodedata
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk
from transformers import AutoTokenizer


# Keep this prompt text aligned with training-time MedReason formatting.
SYSTEM_PROMPT_MEDREASON = """
You are an expert medical AI. You must analyze clinical scenarios and adhere strictly to this output format:

1. Enclose your step-by-step reasoning within `<think>` and `</think>` tags. Keep your reasoning strictly <500 words.
2. Immediately after, output your final conclusion within `<answer>` and `</answer>` tags.
   The content must be exactly in the format: `<LETTER>. <ANSWER_TEXT>`, where:
   - `<LETTER>` is one of A, B, C, D
   - `<ANSWER_TEXT>` is the exact option text corresponding to that letter

Example:
<think>
[Your step-by-step clinical reasoning]
</think>
<answer>
C. Hyperthyroidism
</answer>
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter MedReason rows by token-length constraints (drop-only), then rebalance splits "
            "with eval -> test up to target size -> remaining eval to train."
        )
    )

    parser.add_argument(
        "--input-path",
        type=str,
        default="/mnt/pdata/caf83/data/expert_reasoning/medreason_corrupted_full",
        help="Path to source DatasetDict on disk.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Path to save filtered DatasetDict. Default: <input-path>_token_filtered_no_violations",
    )

    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=300,
        help="Maximum allowed prompt token length (inclusive).",
    )
    parser.add_argument(
        "--max-target-tokens",
        type=int,
        default=824,
        help="Maximum allowed target token length (inclusive).",
    )
    parser.add_argument(
        "--target-test-size",
        type=int,
        default=1000,
        help="Desired test size after rebalance using eval examples.",
    )

    parser.add_argument(
        "--qwen-tokenizer",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="HF repo id/path for Qwen tokenizer.",
    )
    parser.add_argument(
        "--llama-tokenizer",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="HF repo id/path for Llama tokenizer.",
    )

    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to load tokenizers from local cache only.",
    )
    parser.add_argument(
        "--export-jsonl-csv",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to export each split as both JSONL and CSV files.",
    )
    parser.add_argument(
        "--overwrite-output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to delete output path first if it already exists.",
    )
    parser.add_argument(
        "--num-filter-examples",
        type=int,
        default=5,
        help="Number of removed-example previews to print during filtering.",
    )

    return parser.parse_args()


def build_prompt_messages(question: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT_MEDREASON},
        {"role": "user", "content": question},
    ]


def build_prompt_text(tokenizer, question: str, add_generation_prompt: bool = True) -> str:
    return tokenizer.apply_chat_template(
        build_prompt_messages(question),
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def _preview_text(value: Any, max_chars: int = 160) -> str:
    text = str(value).replace("\n", " ").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _ensure_answer_letter(answer: Any) -> Optional[str]:
    """Return normalized answer letter if already in {A,B,C,D}, else None."""
    if answer is None:
        return None
    s = str(answer).strip().upper()
    if re.fullmatch(r"[A-D]", s):
        return s
    return None


def format_choice_answer(letter: str, choice_text: str) -> str:
    """Canonical final-answer format used in saved rows and targets."""
    return f"{letter.strip().upper()}. {str(choice_text).strip()}"


def normalize_choice_answer_with_text(answer: Any) -> Optional[str]:
    """
    Normalize final answer to canonical '<LETTER>. <ANSWER_TEXT>' form.

    Returns None if answer cannot be parsed as letter+text.
    """
    if answer is None:
        return None
    s = str(answer).strip()
    m = re.fullmatch(r"\s*([A-Da-d])\s*[\.\)\:\-]\s*(.+\S)\s*$", s)
    if not m:
        return None
    return format_choice_answer(m.group(1), m.group(2))


def _strip_choice_prefix(text: str) -> str:
    """Strip leading choice label prefixes like 'A. ', 'B) ', 'C:'."""
    return re.sub(r"^\s*[A-D]\s*[\.\)\:]\s*", "", text.strip(), flags=re.IGNORECASE)


def _normalize_for_match(text: Any) -> str:
    """
    Canonical text normalization for matching answer strings to choice text.

    Keeps this intentionally conservative to avoid accidental remaps.
    """
    s = str(text or "")
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\\/", "/")
    s = re.sub(r"(?is)</?answer>", " ", s)
    s = s.replace("’", "'").replace("‘", "'")
    s = s.replace("“", '"').replace("”", '"')
    s = s.replace("–", "-").replace("—", "-")
    s = _strip_choice_prefix(s)
    s = s.lower()
    # Remove punctuation noise while keeping alnum/space.
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def clean_base_reasoning_text(reasoning: Any) -> str:
    """
    Remove markdown artifacts from base (non-corrupted) reasoning only.

    Requested cleanup:
      - remove '###'
      - remove '**'
    """
    text = str(reasoning or "")
    text = text.replace("###", "")
    text = text.replace("**", "")
    return text


def parse_question_choices(question: Any) -> Dict[str, str]:
    """
    Parse A/B/C/D answer choices from question text.

    Expected line formats include:
      - 'A. ...'
      - 'B) ...'
      - 'C: ...'
    """
    q = str(question or "")
    matches = re.findall(r"(?m)^\s*([A-D])[\.\)\:]\s*(.+?)\s*$", q)
    choices: Dict[str, str] = {}
    for label, text in matches:
        clean_text = text.strip()
        if clean_text:
            choices[label.upper()] = clean_text
    return choices


def map_answer_text_to_choice_letter(
    answer_text: Any,
    question: Any,
    choices: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """
    Map answer text to its corresponding A/B/C/D choice letter.

    This is the required path for this dataset:
      answer string -> choice text match -> output letter.
    """
    if answer_text is None:
        return None

    raw = str(answer_text).strip()
    if not raw:
        return None

    if choices is None:
        choices = parse_question_choices(question)
    if not all(lbl in choices for lbl in ("A", "B", "C", "D")):
        return None

    # Candidate variants for matching.
    candidates: List[str] = []
    seen = set()

    def _add_candidate(v: str) -> None:
        key = v.strip()
        if key and key not in seen:
            candidates.append(key)
            seen.add(key)

    _add_candidate(raw)
    raw_no_tags = re.sub(r"(?is)</?answer>", " ", raw).strip()
    _add_candidate(raw_no_tags)
    _add_candidate(_strip_choice_prefix(raw_no_tags))
    _add_candidate(
        re.sub(
            r"(?i)^\s*(?:the\s+)?(?:final\s+)?(?:answer|option|choice)\s*(?:is|:)?\s*",
            "",
            raw_no_tags,
        ).strip()
    )

    # Build exact/normalized lookup tables from question choices.
    # If multiple labels map to same text, keep all and resolve deterministically.
    exact_to_labels: Dict[str, List[str]] = {}
    norm_to_labels: Dict[str, List[str]] = {}

    def _insert_label(mapping: Dict[str, List[str]], key: str, label: str) -> None:
        if key not in mapping:
            mapping[key] = [label]
        elif label not in mapping[key]:
            mapping[key].append(label)

    for label, choice_text in choices.items():
        _insert_label(exact_to_labels, choice_text.strip(), label)
        _insert_label(norm_to_labels, _normalize_for_match(choice_text), label)

    # 1) Exact text match to a choice string.
    for candidate in candidates:
        labels = exact_to_labels.get(candidate.strip())
        if labels:
            # Deterministic tie-break for duplicate choice text.
            return labels[0]

    # 2) Normalized text match to a choice string.
    for candidate in candidates:
        labels = norm_to_labels.get(_normalize_for_match(candidate))
        if labels:
            return labels[0]

    # 3) If candidate is already a bare letter, accept it.
    for candidate in candidates:
        letter = _ensure_answer_letter(candidate)
        if letter is not None:
            return letter

    # 4) Unique containment fallback for slightly noisy text.
    for candidate in candidates:
        c_norm = _normalize_for_match(candidate)
        if len(c_norm) < 3:
            continue
        matched_labels = []
        for lbl, choice_text in choices.items():
            o_norm = _normalize_for_match(choice_text)
            if c_norm in o_norm or o_norm in c_norm:
                matched_labels.append(lbl)
        if len(matched_labels) == 1:
            return matched_labels[0]

    # 5) Fuzzy fallback for minor typos (if clearly best match).
    for candidate in candidates:
        c_norm = _normalize_for_match(candidate)
        if len(c_norm) < 3:
            continue
        scored = []
        for lbl, choice_text in choices.items():
            o_norm = _normalize_for_match(choice_text)
            ratio = difflib.SequenceMatcher(None, c_norm, o_norm).ratio()
            scored.append((ratio, lbl))
        scored.sort(reverse=True)
        if len(scored) >= 2:
            best_ratio, best_lbl = scored[0]
            second_ratio = scored[1][0]
            if best_ratio >= 0.90 and (best_ratio - second_ratio) >= 0.05:
                return best_lbl
        elif scored and scored[0][0] >= 0.90:
            return scored[0][1]

    return None


def map_answer_text_to_choice_with_text(
    answer_text: Any,
    question: Any,
    choices: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """
    Map answer text to canonical '<LETTER>. <ANSWER_TEXT>' form using question choices.
    """
    if choices is None:
        choices = parse_question_choices(question)
    letter = map_answer_text_to_choice_letter(
        answer_text=answer_text,
        question=question,
        choices=choices,
    )
    if letter is None:
        return None
    choice_text = choices.get(letter)
    if not choice_text:
        return None
    return format_choice_answer(letter, choice_text)


def build_target_text(reasoning: str, answer: str) -> str:
    answer_with_text = normalize_choice_answer_with_text(answer)
    if answer_with_text is None:
        raise ValueError(
            "Answer must be in '<LETTER>. <ANSWER_TEXT>' format at target build time, "
            f"got: {answer!r}"
        )
    return f"<think>\n{reasoning}\n</think>\n<answer>\n{answer_with_text}\n</answer>"


def prompt_len_by_tokenizer(question: str, tokenizers: Dict[str, object]) -> Dict[str, int]:
    lengths = {}
    for name, tok in tokenizers.items():
        prompt_text = build_prompt_text(tok, question)
        lengths[name] = len(tok(prompt_text, add_special_tokens=True).input_ids)
    return lengths


def target_len_by_tokenizer(reasoning: str, answer: str, tokenizers: Dict[str, object]) -> Dict[str, int]:
    lengths = {}
    target_text = build_target_text(reasoning, answer)
    for name, tok in tokenizers.items():
        lengths[name] = len(tok(target_text, add_special_tokens=True).input_ids)
    return lengths


def collect_violation_details(
    ex: Dict,
    tokenizers: Dict[str, object],
    max_prompt_tokens: int,
    max_target_tokens: int,
) -> Tuple[List[Dict], Dict]:
    """
    Return violations and a normalized copy of the example.

    A single example may have multiple detail records (e.g., both tokenizers violate).
    The returned example has normalized `answer`/`corrupted_answers` when possible.
    """
    details: List[Dict] = []
    row = dict(ex)
    row["reasoning"] = clean_base_reasoning_text(row.get("reasoning"))

    # Parse choices once; all answer mapping is done against these choices.
    choices = parse_question_choices(row.get("question"))
    if not all(lbl in choices for lbl in ("A", "B", "C", "D")):
        details.append(
            {
                "kind": "question_choice_parse",
                "tokenizer": "n/a",
                "length": -1,
                "limit": -1,
                "question_preview": _preview_text(row.get("question")),
            }
        )
        return details, row

    prompt_lens = prompt_len_by_tokenizer(row["question"], tokenizers)
    for tokenizer_name, length in prompt_lens.items():
        if length > max_prompt_tokens:
            details.append(
                {
                    "kind": "base_prompt",
                    "tokenizer": tokenizer_name,
                    "length": int(length),
                    "limit": int(max_prompt_tokens),
                }
            )

    # Base answer must map from answer text -> '<LETTER>. <ANSWER_TEXT>'.
    base_answer_with_text = map_answer_text_to_choice_with_text(
        answer_text=row.get("answer"),
        question=row.get("question"),
        choices=choices,
    )
    if base_answer_with_text is None:
        details.append(
            {
                "kind": "base_answer_mapping",
                "tokenizer": "n/a",
                "length": -1,
                "limit": -1,
                "raw_answer_preview": _preview_text(row.get("answer")),
            }
        )
    else:
        row["answer"] = base_answer_with_text
        base_target_lens = target_len_by_tokenizer(row["reasoning"], row["answer"], tokenizers)
        for tokenizer_name, length in base_target_lens.items():
            if length > max_target_tokens:
                details.append(
                    {
                        "kind": "base_target",
                        "tokenizer": tokenizer_name,
                        "length": int(length),
                        "limit": int(max_target_tokens),
                    }
                )

    # Corrupted answers are also mapped text -> '<LETTER>. <ANSWER_TEXT>'.
    corrupted_reasonings = list(row.get("corrupted_reasonings", []) or [])
    corrupted_answers = list(row.get("corrupted_answers", []) or [])
    mapped_corrupt_answers_with_text: List[str] = []
    corrupt_answer_valid: List[bool] = []

    for corruption_index, ca in enumerate(corrupted_answers):
        answer_with_text = map_answer_text_to_choice_with_text(
            answer_text=ca,
            question=row.get("question"),
            choices=choices,
        )
        if answer_with_text is None:
            details.append(
                {
                    "kind": "corrupt_answer_mapping",
                    "corruption_index": int(corruption_index),
                    "tokenizer": "n/a",
                    "length": -1,
                    "limit": -1,
                    "raw_answer_preview": _preview_text(ca),
                }
            )
            mapped_corrupt_answers_with_text.append(str(ca).strip())
            corrupt_answer_valid.append(False)
        else:
            mapped_corrupt_answers_with_text.append(answer_with_text)
            corrupt_answer_valid.append(True)

    row["corrupted_answers"] = mapped_corrupt_answers_with_text

    # Token length checks for corrupted targets only run when mapping succeeded.
    for corruption_index, (cr, ca_with_text) in enumerate(
        zip(corrupted_reasonings, mapped_corrupt_answers_with_text)
    ):
        if corruption_index < len(corrupt_answer_valid) and not corrupt_answer_valid[corruption_index]:
            continue

        corrupt_lens = target_len_by_tokenizer(cr, ca_with_text, tokenizers)
        for tokenizer_name, length in corrupt_lens.items():
            if length > max_target_tokens:
                details.append(
                    {
                        "kind": "corrupt_target",
                        "corruption_index": int(corruption_index),
                        "tokenizer": tokenizer_name,
                        "length": int(length),
                        "limit": int(max_target_tokens),
                    }
                )

    return details, row


def filter_non_violating_examples(
    dsd: DatasetDict,
    tokenizers: Dict[str, object],
    max_prompt_tokens: int,
    max_target_tokens: int,
    num_filter_examples: int = 5,
) -> Tuple[DatasetDict, Dict, List[Dict], List[Dict]]:
    """
    Remove rows with any violation from every split.

    Returns:
      - filtered DatasetDict
      - stats dict
      - list of violation-detail rows for optional reporting/debugging
      - list of preview examples removed during filtering
    """
    filtered_splits: Dict[str, Dataset] = {}
    removed_rows: List[Dict] = []
    removed_example_previews: List[Dict] = []

    stats = {
        "rows_in": 0,
        "rows_kept": 0,
        "rows_removed": 0,
        "rows_in_by_split": {},
        "rows_kept_by_split": {},
        "rows_removed_by_split": {},
        # Count rows removed by coarse reason kind (base_prompt/base_target/corrupt_target).
        "rows_removed_by_kind": {},
    }

    for split_name, split_ds in dsd.items():
        kept_rows: List[Dict] = []
        removed_here = 0
        stats["rows_in_by_split"][split_name] = len(split_ds)

        for idx, ex in enumerate(split_ds):
            stats["rows_in"] += 1
            details, normalized_row = collect_violation_details(
                ex=ex,
                tokenizers=tokenizers,
                max_prompt_tokens=max_prompt_tokens,
                max_target_tokens=max_target_tokens,
            )

            if details:
                removed_here += 1
                stats["rows_removed"] += 1

                # Each row contributes at most once per kind to the per-kind row count.
                row_kinds_seen = set()
                for detail in details:
                    kind = detail["kind"]
                    if kind not in row_kinds_seen:
                        stats["rows_removed_by_kind"][kind] = stats["rows_removed_by_kind"].get(kind, 0) + 1
                        row_kinds_seen.add(kind)

                    removed_rows.append(
                        {
                            "split": split_name,
                            "index": int(idx),
                            **detail,
                        }
                    )

                if len(removed_example_previews) < max(0, num_filter_examples):
                    removed_example_previews.append(
                        {
                            "split": split_name,
                            "index": int(idx),
                            "question_preview": _preview_text(ex.get("question")),
                            "answer_raw": _preview_text(ex.get("answer")),
                            "answer_normalized": normalized_row.get("answer"),
                            "violation_kinds": sorted({d["kind"] for d in details}),
                            "first_violation": details[0],
                        }
                    )
                continue

            # Keep normalized answer fields in the saved dataset.
            kept_rows.append(normalized_row)
            stats["rows_kept"] += 1

        if kept_rows:
            filtered_split = Dataset.from_list(kept_rows, features=split_ds.features)
        else:
            filtered_split = split_ds.select([])
        filtered_splits[split_name] = filtered_split
        stats["rows_kept_by_split"][split_name] = len(filtered_split)
        stats["rows_removed_by_split"][split_name] = int(removed_here)

    return DatasetDict(filtered_splits), stats, removed_rows, removed_example_previews


def _concat_safely(a: Dataset, b: Dataset) -> Dataset:
    if len(a) == 0:
        return b
    if len(b) == 0:
        return a
    return concatenate_datasets([a, b])


def rebalance_eval_into_test_and_train(
    dsd: DatasetDict,
    target_test_size: int,
) -> Tuple[DatasetDict, Dict]:
    """
    Rebalance splits as requested:
      - Move eval -> test until test reaches target_test_size
      - Move remaining eval -> train
      - Final eval becomes empty
    """
    for required_split in ("train", "eval", "test"):
        if required_split not in dsd:
            raise KeyError(f"Required split '{required_split}' missing. Found: {list(dsd.keys())}")

    train_ds = dsd["train"]
    eval_ds = dsd["eval"]
    test_ds = dsd["test"]

    test_needed = max(0, int(target_test_size) - len(test_ds))
    eval_to_test_count = min(test_needed, len(eval_ds))

    if eval_to_test_count > 0:
        eval_to_test = eval_ds.select(range(eval_to_test_count))
    else:
        eval_to_test = eval_ds.select([])

    if eval_to_test_count < len(eval_ds):
        eval_to_train = eval_ds.select(range(eval_to_test_count, len(eval_ds)))
    else:
        eval_to_train = eval_ds.select([])

    out = DatasetDict(
        {
            "train": _concat_safely(train_ds, eval_to_train),
            "eval": eval_ds.select([]),
            "test": _concat_safely(test_ds, eval_to_test),
        }
    )

    rebalance_stats = {
        "target_test_size": int(target_test_size),
        "train_before": len(train_ds),
        "eval_before": len(eval_ds),
        "test_before": len(test_ds),
        "eval_to_test": int(eval_to_test_count),
        "eval_to_train": int(len(eval_ds) - eval_to_test_count),
        "train_after": len(out["train"]),
        "eval_after": len(out["eval"]),
        "test_after": len(out["test"]),
    }
    return out, rebalance_stats


def save_dataset_and_report(
    dsd: DatasetDict,
    output_path: str,
    export_jsonl_csv: bool,
    report: Dict,
    overwrite_output: bool,
) -> None:
    if os.path.exists(output_path):
        if overwrite_output:
            shutil.rmtree(output_path)
        else:
            raise FileExistsError(
                f"Output path exists: {output_path}. Use --overwrite-output to replace it."
            )

    dsd.save_to_disk(output_path)

    if export_jsonl_csv:
        for split_name, split_ds in dsd.items():
            split_ds.to_json(os.path.join(output_path, f"{split_name}.jsonl"))
            split_ds.to_csv(os.path.join(output_path, f"{split_name}.csv"))

    report_path = os.path.join(output_path, "length_filter_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def main() -> None:
    args = parse_args()

    input_path = args.input_path
    output_path = args.output_path or (input_path + "_token_filtered_no_violations")

    tokenizer_names = {
        "qwen2.5": args.qwen_tokenizer,
        "llama3.2": args.llama_tokenizer,
    }

    print("Loading tokenizers...")
    tokenizers = {
        name: AutoTokenizer.from_pretrained(
            repo_id,
            local_files_only=args.local_files_only,
            use_fast=True,
        )
        for name, repo_id in tokenizer_names.items()
    }

    print(f"Loading dataset from: {input_path}")
    dsd = load_from_disk(input_path)

    print("Filtering violating rows (drop-only, no trimming)...")
    dsd_filtered, filter_stats, removed_rows, removed_example_previews = filter_non_violating_examples(
        dsd=dsd,
        tokenizers=tokenizers,
        max_prompt_tokens=args.max_prompt_tokens,
        max_target_tokens=args.max_target_tokens,
        num_filter_examples=args.num_filter_examples,
    )

    if removed_example_previews:
        print()
        print(f"Sample removed examples ({len(removed_example_previews)} shown):")
        for i, sample in enumerate(removed_example_previews, start=1):
            first = sample["first_violation"]
            print(
                f"  [{i}] {sample['split']}[{sample['index']}] "
                f"kinds={','.join(sample['violation_kinds'])}"
            )
            print(
                f"      answer_raw={sample['answer_raw']!r} "
                f"answer_normalized={sample['answer_normalized']!r}"
            )
            print(
                f"      first_violation={first.get('kind')} "
                f"tokenizer={first.get('tokenizer')} "
                f"length={first.get('length')} limit={first.get('limit')}"
            )
            print(f"      question={sample['question_preview']}")

    print("Rebalancing splits: eval -> test (up to target) -> remaining eval to train...")
    dsd_rebalanced, split_rebalance_stats = rebalance_eval_into_test_and_train(
        dsd=dsd_filtered,
        target_test_size=args.target_test_size,
    )

    # Attach final split sizes for easy consumption.
    filter_stats["split_rebalance"] = split_rebalance_stats
    filter_stats["train_rows_final"] = len(dsd_rebalanced["train"])
    filter_stats["eval_rows_final"] = len(dsd_rebalanced["eval"])
    filter_stats["test_rows_final"] = len(dsd_rebalanced["test"])

    # Summarize detail-level violations by (kind, tokenizer) for debugging.
    violations_by_kind_tokenizer: Dict[str, int] = {}
    for row in removed_rows:
        key = f"{row.get('kind')}::{row.get('tokenizer')}"
        violations_by_kind_tokenizer[key] = violations_by_kind_tokenizer.get(key, 0) + 1

    report = {
        "source_path": input_path,
        "output_path": output_path,
        "max_prompt_tokens": int(args.max_prompt_tokens),
        "max_target_tokens": int(args.max_target_tokens),
        "target_test_size": int(args.target_test_size),
        "tokenizers": tokenizer_names,
        "filter_stats": filter_stats,
        "removed_violation_rows": int(len(removed_rows)),
        "violations_by_kind_tokenizer": violations_by_kind_tokenizer,
        "removed_examples_preview": removed_example_previews,
        "created_utc": datetime.utcnow().isoformat() + "Z",
    }

    print(f"Saving dataset + report to: {output_path}")
    save_dataset_and_report(
        dsd=dsd_rebalanced,
        output_path=output_path,
        export_jsonl_csv=args.export_jsonl_csv,
        report=report,
        overwrite_output=args.overwrite_output,
    )

    print("Done.")
    print("Final split sizes:")
    print(f"  train: {filter_stats['train_rows_final']}")
    print(f"  eval:  {filter_stats['eval_rows_final']}")
    print(f"  test:  {filter_stats['test_rows_final']}")


if __name__ == "__main__":
    main()
