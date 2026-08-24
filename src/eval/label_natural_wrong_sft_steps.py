"""Label first erroneous steps in naturally wrong SFT GSM8K generations.

The output is pair-style JSONL compatible with the localisation scorers:
``clean_text`` is a correct SFT reference generation for the same prompt, and
``pert_text``/``wrong_text`` is a naturally wrong SFT generation. The LLM labels
the first mathematically invalid reasoning step in the wrong trace.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.build_chatgpt_step_perturbation_dataset import (  # noqa: E402
    ANSWER_RE,
    THINK_RE,
    DEFAULT_API_VERSION,
    DEFAULT_DEPLOYMENT,
    DEFAULT_ENDPOINT,
    _extract_answer,
    _extract_json,
    _load_jsonl,
    _normalize_azure_endpoint,
    _prompt_text,
    _to_int,
    call_azure,
    make_azure_client,
)


DEFAULT_GENERATIONS = Path(
    "/mnt/pdata/caf83/icml_math/outputs/qwen7b_sft/best_model/"
    "eval_results_math_qwen7b_sft_t0p5.jsonl"
)
DEFAULT_REFERENCE = PROJECT_ROOT / "localisation/runs/qwen7b_sft/qwen7b/full/pair_details.jsonl"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "localisation/natural_wrong_sft/"
    / "gsm8k_qwen7b_sft_wrong_step_labels_smoke.jsonl"
)

_THREAD_LOCAL = threading.local()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations-jsonl", type=Path, default=DEFAULT_GENERATIONS)
    parser.add_argument("--reference-pair-details", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--wrong-per-prompt", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--deployment",
        type=str,
        default=os.environ.get("AZURE_OPENAI_DEPLOYMENT")
        or os.environ.get("AZURE_OPENAI_MODEL")
        or DEFAULT_DEPLOYMENT,
    )
    parser.add_argument("--azure-endpoint", type=str, default=None)
    parser.add_argument("--api-version", type=str, default=None)
    parser.add_argument("--max-completion-tokens", type=int, default=900)
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default="off",
        help="Use 'off'/'none' for Azure reasoning_effort='none'; use 'omit' to omit.",
    )
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--retry-sleep-seconds", type=float, default=1.0)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument(
        "--correct-reward-value",
        type=float,
        default=2.0,
        help="Value of correctness_reward_func indicating a correct generation.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output file if it exists.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append missing successful (prompt_idx, wrong_generation_idx) labels.",
    )
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def get_thread_azure_client(args: argparse.Namespace):
    client = getattr(_THREAD_LOCAL, "azure_client", None)
    if client is None:
        endpoint = args.azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT") or DEFAULT_ENDPOINT
        api_version = args.api_version or os.environ.get("AZURE_OPENAI_API_VERSION") or DEFAULT_API_VERSION
        endpoint, deployment, api_version = _normalize_azure_endpoint(
            endpoint=endpoint,
            deployment=str(args.deployment),
            api_version=api_version,
        )
        args.azure_endpoint = endpoint
        args.deployment = deployment
        args.api_version = api_version
        client = make_azure_client(args)
        _THREAD_LOCAL.azure_client = client
    return client


def _normalize_question(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _generation_text(row: dict[str, Any]) -> str:
    generation = row.get("generation")
    if isinstance(generation, dict):
        return str(generation.get("content", ""))
    return str(generation or "")


def _correctness(row: dict[str, Any]) -> float | None:
    try:
        return float(row.get("correctness_reward_func"))
    except Exception:
        return None


def _is_correct(row: dict[str, Any], correct_reward_value: float) -> bool:
    value = _correctness(row)
    return value is not None and abs(float(value) - float(correct_reward_value)) <= 1e-9


def _is_wrong(row: dict[str, Any], correct_reward_value: float) -> bool:
    value = _correctness(row)
    return value is not None and value < float(correct_reward_value)


def _load_reference_by_question(path: Path) -> dict[str, dict[str, Any]]:
    refs: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return refs
    for row in _load_jsonl(path):
        prompt = row.get("prompt")
        if not isinstance(prompt, list):
            continue
        question = _prompt_text(prompt)
        key = _normalize_question(question)
        if key and key not in refs:
            refs[key] = row
    return refs


def _group_generations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups_by_question: dict[str, dict[str, Any]] = {}
    ordered: list[dict[str, Any]] = []
    for row in rows:
        prompt = row.get("prompt")
        if not isinstance(prompt, list):
            continue
        question = _prompt_text(prompt)
        key = _normalize_question(question)
        if not key:
            continue
        group = groups_by_question.get(key)
        if group is None:
            group = {"question": question, "prompt": prompt, "rows": []}
            groups_by_question[key] = group
            ordered.append(group)
        group["rows"].append(row)
    return ordered


def _reasoning_steps(trace: str) -> list[dict[str, Any]]:
    match = THINK_RE.search(trace or "")
    if not match:
        return []
    content = match.group(1)
    base = match.start(1)
    steps: list[dict[str, Any]] = []
    for line_match in re.finditer(r"[^\r\n]+", content):
        raw = line_match.group(0)
        stripped = raw.strip()
        if not stripped:
            continue
        leading = len(raw) - len(raw.lstrip())
        trailing = len(raw.rstrip())
        start = base + line_match.start() + leading
        end = base + line_match.start() + trailing
        steps.append({"index": len(steps), "text": stripped, "char_span": [int(start), int(end)]})
    if steps:
        return steps

    # Fallback for one-line traces: split on sentence-like boundaries.
    for sent_match in re.finditer(r"[^.!?\n]+(?:[.!?]+|$)", content):
        stripped = sent_match.group(0).strip()
        if not stripped:
            continue
        leading = len(sent_match.group(0)) - len(sent_match.group(0).lstrip())
        trailing = len(sent_match.group(0).rstrip())
        start = base + sent_match.start() + leading
        end = base + sent_match.start() + trailing
        steps.append({"index": len(steps), "text": stripped, "char_span": [int(start), int(end)]})
    return steps


def _format_steps(steps: list[dict[str, Any]]) -> str:
    return "\n".join(f"[{step['index']}] {step['text']}" for step in steps)


def _build_messages(
    question: str,
    answer: str | None,
    correct_trace: str,
    wrong_trace: str,
    wrong_steps: list[dict[str, Any]],
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You label GSM8K reasoning errors for localisation evaluation. "
                "Return valid JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                "A model produced a wrong GSM8K solution. Identify the first reasoning step "
                "where the wrong solution becomes mathematically invalid or unjustified. "
                "Choose from the numbered steps exactly as listed. Do not choose a later step "
                "if an earlier step already contains the error. If the reasoning is valid but "
                "only the final answer tag is wrong, set label to \"answer_only\" and use "
                "first_wrong_step_index = null.\n\n"
                "Return JSON with exactly these keys:\n"
                "- label: one of \"invalid_step\", \"answer_only\", \"ambiguous\"\n"
                "- first_wrong_step_index: integer step index, or null\n"
                "- first_wrong_step: exact text of that listed step, or empty string\n"
                "- corrected_step: a corrected version of the step, or empty string\n"
                "- error_summary: one short phrase\n"
                "- confidence: number between 0 and 1\n\n"
                f"Question:\n{question}\n\n"
                f"Gold final answer:\n{answer or ''}\n\n"
                f"Correct reference solution:\n{correct_trace}\n\n"
                f"Wrong model solution:\n{wrong_trace}\n\n"
                f"Numbered wrong-solution reasoning steps:\n{_format_steps(wrong_steps)}\n"
            ),
        },
    ]


def _validate_label(obj: dict[str, Any], wrong_steps: list[dict[str, Any]]) -> tuple[str, int | None, str, list[int] | None]:
    label = str(obj.get("label", "")).strip()
    if label not in {"invalid_step", "answer_only", "ambiguous"}:
        raise ValueError(f"Invalid label: {label!r}")
    raw_idx = obj.get("first_wrong_step_index")
    if label != "invalid_step":
        return label, None, "", None
    idx = _to_int(raw_idx)
    if idx is None or idx < 0 or idx >= len(wrong_steps):
        raise ValueError(f"Invalid first_wrong_step_index: {raw_idx!r}")
    step = wrong_steps[idx]
    returned_step = str(obj.get("first_wrong_step", "")).strip()
    if returned_step and returned_step != step["text"]:
        raise ValueError(
            "first_wrong_step does not exactly match the numbered step at "
            f"index {idx}: {returned_step!r}"
        )
    return label, idx, str(step["text"]), list(step["char_span"])


def _load_done_keys(path: Path) -> set[tuple[int, int]]:
    done: set[tuple[int, int]] = set()
    if not path.exists():
        return done
    with path.open("r") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            prompt_idx = _to_int(row.get("prompt_idx"))
            gen_idx = _to_int(row.get("wrong_generation_idx"))
            if prompt_idx is not None and gen_idx is not None and row.get("error") is None:
                done.add((int(prompt_idx), int(gen_idx)))
    return done


def build_items(args: argparse.Namespace) -> list[dict[str, Any]]:
    refs = _load_reference_by_question(args.reference_pair_details)
    ref_prompt_ids = [_to_int(row.get("prompt_idx")) for row in refs.values()]
    fallback_base = max([int(x) for x in ref_prompt_ids if x is not None] or [-1]) + 1
    groups = _group_generations(_load_jsonl(args.generations_jsonl))
    items: list[dict[str, Any]] = []
    fallback_prompt_idx = fallback_base
    for group in groups:
        question = str(group["question"])
        key = _normalize_question(question)
        ref = refs.get(key, {})
        prompt_idx = _to_int(ref.get("prompt_idx"))
        if prompt_idx is None:
            prompt_idx = fallback_prompt_idx
        fallback_prompt_idx += 1

        correct_rows = [
            row for row in group["rows"]
            if _is_correct(row, float(args.correct_reward_value)) and THINK_RE.search(_generation_text(row))
        ]
        wrong_rows = [
            row for row in group["rows"]
            if _is_wrong(row, float(args.correct_reward_value)) and THINK_RE.search(_generation_text(row))
        ]
        if not correct_rows or not wrong_rows:
            continue
        correct_rows = sorted(correct_rows, key=lambda r: _to_int(r.get("generation_idx")) or 0)
        wrong_rows = sorted(wrong_rows, key=lambda r: _to_int(r.get("generation_idx")) or 0)
        if int(args.wrong_per_prompt) > 0:
            wrong_rows = wrong_rows[: int(args.wrong_per_prompt)]
        clean_row = correct_rows[0]
        clean_text = _generation_text(clean_row)
        answer = ref.get("answer") or _extract_answer(clean_text)
        for wrong_row in wrong_rows:
            wrong_text = _generation_text(wrong_row)
            wrong_steps = _reasoning_steps(wrong_text)
            if not wrong_steps:
                continue
            items.append(
                {
                    "prompt_idx": int(prompt_idx),
                    "severity": 1,
                    "variant_idx": wrong_row.get("generation_idx"),
                    "prompt": group["prompt"],
                    "question": question,
                    "answer": answer,
                    "clean_text": clean_text,
                    "correct_text": clean_text,
                    "clean_generation_idx": clean_row.get("generation_idx"),
                    "wrong_text": wrong_text,
                    "pert_text": wrong_text,
                    "wrong_generation_idx": wrong_row.get("generation_idx"),
                    "wrong_correctness_reward_func": wrong_row.get("correctness_reward_func"),
                    "clean_correctness_reward_func": clean_row.get("correctness_reward_func"),
                    "source_generations_jsonl": str(args.generations_jsonl),
                    "reference_pair_details": str(args.reference_pair_details),
                    "wrong_steps": wrong_steps,
                }
            )
    return items


def process_item(item: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    messages = _build_messages(
        question=str(item["question"]),
        answer=item.get("answer"),
        correct_trace=str(item["clean_text"]),
        wrong_trace=str(item["wrong_text"]),
        wrong_steps=list(item["wrong_steps"]),
    )
    out = {
        **{k: v for k, v in item.items() if k != "wrong_steps"},
        "created_at": datetime.now(timezone.utc).isoformat(),
        "azure_deployment": args.deployment,
        "label_task": "natural_wrong_sft_first_error_step",
    }
    if args.dry_run:
        out["messages"] = messages
        out["dry_run"] = True
        out["error"] = None
        return out

    client = get_thread_azure_client(args)
    errors: list[str] = []
    raw = None
    parsed: dict[str, Any] | None = None
    for attempt in range(1, int(args.max_retries) + 2):
        try:
            raw = call_azure(
                azure_client=client,
                messages=messages
                if attempt == 1
                else messages
                + [
                    {
                        "role": "user",
                        "content": (
                            "Your previous attempt failed validation: "
                            f"{errors[-1]}\nTry again. Choose exactly one listed step index "
                            "for invalid_step labels and return valid JSON only."
                        ),
                    }
                ],
                deployment=str(args.deployment),
                max_completion_tokens=int(args.max_completion_tokens),
                reasoning_effort=str(args.reasoning_effort),
            )
            parsed = _extract_json(raw)
            label, idx, step_text, span = _validate_label(parsed, list(item["wrong_steps"]))
            out.update(
                {
                    "label": label,
                    "first_wrong_step_index": idx,
                    "first_wrong_step": step_text,
                    "original_step": "",
                    "perturbed_step": step_text,
                    "target_char_span": span,
                    "target_error_char_span": span,
                    "corrected_step": str(parsed.get("corrected_step", "")).strip(),
                    "error_summary": str(parsed.get("error_summary", "")).strip(),
                    "confidence": parsed.get("confidence"),
                    "raw_model_output": raw,
                    "parsed_model_output": parsed,
                    "attempts": attempt,
                    "error": None if label == "invalid_step" else label,
                }
            )
            return out
        except Exception as exc:
            errors.append(str(exc))
            if attempt <= int(args.max_retries) and float(args.retry_sleep_seconds) > 0:
                time.sleep(float(args.retry_sleep_seconds))

    out.update(
        {
            "label": None,
            "first_wrong_step_index": None,
            "first_wrong_step": "",
            "target_char_span": None,
            "target_error_char_span": None,
            "raw_model_output": raw,
            "parsed_model_output": parsed,
            "attempts": int(args.max_retries) + 1,
            "validation_errors": errors,
            "error": errors[-1] if errors else "Unknown error",
        }
    )
    return out


def _error_item(item: dict[str, Any], args: argparse.Namespace, exc: BaseException) -> dict[str, Any]:
    return {
        **{k: v for k, v in item.items() if k != "wrong_steps"},
        "created_at": datetime.now(timezone.utc).isoformat(),
        "azure_deployment": args.deployment,
        "label_task": "natural_wrong_sft_first_error_step",
        "target_char_span": None,
        "target_error_char_span": None,
        "attempts": 0,
        "validation_errors": [str(exc)],
        "error": str(exc),
    }


def write_result(f, out: dict[str, Any], written: int, total: int) -> int:
    f.write(json.dumps(out, ensure_ascii=False) + "\n")
    f.flush()
    written += 1
    status = "ok" if out.get("error") is None else f"error:{out.get('error')}"
    print(
        f"[{written}/{total}] prompt_idx={out.get('prompt_idx')} "
        f"wrong_generation_idx={out.get('wrong_generation_idx')} status={status} "
        f"attempts={out.get('attempts', 0)}",
        flush=True,
    )
    return written


def main() -> None:
    args = parse_args()
    if int(args.num_workers) < 1:
        raise ValueError("--num-workers must be >= 1.")
    if args.output_file.exists() and args.force:
        args.output_file.unlink()
    elif args.output_file.exists() and not args.resume:
        raise FileExistsError(
            f"Output already exists: {args.output_file}. Use --force or --resume."
        )

    rng = random.Random(int(args.seed))
    items = build_items(args)
    items = items[int(args.start_index) :]
    if int(args.max_examples) > 0:
        items = items[: int(args.max_examples)]
    if args.shuffle:
        rng.shuffle(items)
    if args.resume:
        done = _load_done_keys(args.output_file)
        items = [
            item for item in items
            if (int(item["prompt_idx"]), int(item["wrong_generation_idx"])) not in done
        ]

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.output_file.exists() and args.resume and not args.force else "w"
    written = 0
    with args.output_file.open(mode) as f:
        if int(args.num_workers) == 1:
            for item in items:
                written = write_result(f, process_item(item, args), written, len(items))
        else:
            with ThreadPoolExecutor(max_workers=int(args.num_workers)) as executor:
                future_to_item = {executor.submit(process_item, item, args): item for item in items}
                for future in as_completed(future_to_item):
                    item = future_to_item[future]
                    try:
                        out = future.result()
                    except Exception as exc:
                        out = _error_item(item, args, exc)
                    written = write_result(f, out, written, len(items))

    print(f"Wrote {written} rows to {args.output_file}")


if __name__ == "__main__":
    main()
