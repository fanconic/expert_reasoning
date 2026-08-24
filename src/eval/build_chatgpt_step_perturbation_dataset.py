"""Build a small ChatGPT-edited GSM8K perturbation dataset for localisation tests.

The source traces are the same pregenerated Qwen7B-SFT traces used by the
Table-5 localisation runs. Unlike the mechanical operator/number perturbations,
this script asks the Azure OpenAI model to rewrite one reasoning step into a
fluent but mathematically wrong step, while preserving the surrounding trace and
answer tags. The output includes explicit step/span metadata so later evaluation
can target a semantic corruption region rather than an exact edited token.
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
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from openai import AzureOpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


AZURE_CONNECTION_PATH = PROJECT_ROOT / "src/azure/azure_connection.py"
FALLBACK_DEPLOYMENT = "gpt-5-claudio"
FALLBACK_ENDPOINT = "https://vdslabazuremloai-eus2.openai.azure.com/"
FALLBACK_API_VERSION = "2025-01-01-preview"


def _read_azure_connection_defaults() -> tuple[str, str, str]:
    deployment = FALLBACK_DEPLOYMENT
    endpoint = FALLBACK_ENDPOINT
    api_version = FALLBACK_API_VERSION
    try:
        text = AZURE_CONNECTION_PATH.read_text()
        m_deploy = re.search(r"DEPLOYMENT\s*=\s*['\"]([^'\"]+)['\"]", text)
        m_endpoint = re.search(r"endpoint\s*=\s*['\"]([^'\"]+)['\"]", text)
        if m_deploy:
            deployment = m_deploy.group(1)
        if m_endpoint:
            endpoint = m_endpoint.group(1)
        endpoint, parsed_deployment, parsed_api_version = _normalize_azure_endpoint(
            endpoint=endpoint,
            deployment=deployment,
            api_version=api_version,
        )
        deployment = parsed_deployment
        api_version = parsed_api_version
    except Exception:
        pass
    return deployment, endpoint, api_version


def _normalize_azure_endpoint(
    endpoint: str,
    deployment: str,
    api_version: str,
) -> tuple[str, str, str]:
    parsed = urlparse(endpoint)
    if not parsed.scheme or not parsed.netloc:
        return endpoint, deployment, api_version

    parts = [p for p in parsed.path.split("/") if p]
    if "deployments" in parts:
        idx = parts.index("deployments")
        if idx + 1 < len(parts):
            deployment = parts[idx + 1]
        query = parse_qs(parsed.query)
        if query.get("api-version"):
            api_version = query["api-version"][0]
        endpoint = f"{parsed.scheme}://{parsed.netloc}/"
    return endpoint, deployment, api_version


DEFAULT_DEPLOYMENT, DEFAULT_ENDPOINT, DEFAULT_API_VERSION = _read_azure_connection_defaults()
_THREAD_LOCAL = threading.local()


DEFAULT_SOURCE = (
    PROJECT_ROOT
    / "localisation/runs/qwen7b_sft/qwen7b/full/pair_details.jsonl"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "localisation/chatgpt_step_perturbations/"
    / "gsm8k_qwen7b_sft_step_perturbations_smoke.jsonl"
)

THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", flags=re.DOTALL | re.IGNORECASE)
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", flags=re.DOTALL | re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-pair-details", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--max-examples",
        type=int,
        default=0,
        help="Maximum rows to generate after start-index. Use 0 or negative for all available rows.",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--deployment",
        type=str,
        default=os.environ.get("AZURE_OPENAI_DEPLOYMENT")
        or os.environ.get("AZURE_OPENAI_MODEL")
        or DEFAULT_DEPLOYMENT,
    )
    parser.add_argument(
        "--azure-endpoint",
        type=str,
        default=None,
        help="Azure OpenAI endpoint. Defaults to AZURE_OPENAI_ENDPOINT or src.azure.azure_connection.endpoint.",
    )
    parser.add_argument(
        "--api-version",
        type=str,
        default=None,
        help="Azure OpenAI API version. Defaults to AZURE_OPENAI_API_VERSION or 2025-01-01-preview.",
    )
    parser.add_argument("--max-completion-tokens", type=int, default=1200)
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default="off",
        help=(
            "Reasoning effort passed to Azure OpenAI. Use 'off'/'none' to request "
            "Azure reasoning_effort='none'; use 'omit' to omit the parameter entirely."
        ),
    )
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of concurrent Azure request workers. Use 1 for sequential generation.",
    )
    parser.add_argument("--retry-sleep-seconds", type=float, default=1.0)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument(
        "--reject-token-like-edits",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reject perturbations whose returned step differs only by a tiny numeric/operator edit.",
    )
    parser.add_argument(
        "--min-step-token-edits",
        type=int,
        default=4,
        help="Minimum token-level edit count between original_step and perturbed_step.",
    )
    parser.add_argument(
        "--only-table-valid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the same severity-1/Table-5-valid row filter as the localisation tables.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output file if it already exists.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If output exists, append missing successful prompt_idx rows instead of starting over.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle selected rows after filtering/slicing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write prompts that would be sent, without calling Azure OpenAI.",
    )
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r") as f:
        for line in f:
            raw = line.strip()
            if raw:
                rows.append(json.loads(raw))
    return rows


def _to_int(x: Any) -> int | None:
    try:
        return int(x)
    except Exception:
        return None


def _table_valid_row(row: dict[str, Any]) -> bool:
    if _to_int(row.get("severity")) != 1:
        return False
    pert_seq = row.get("pert_score_seq", [])
    changed_positions = row.get("changed_token_positions", [])
    if not isinstance(pert_seq, list) or len(pert_seq) < 2:
        return False
    if not isinstance(changed_positions, list):
        return False
    changed = {
        int(c)
        for c in changed_positions
        if _to_int(c) is not None and 0 <= int(c) < len(pert_seq)
    }
    return bool(changed)


def select_source_rows(rows: list[dict[str, Any]], only_table_valid: bool) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen_prompt_idx: set[int] = set()
    for row in rows:
        prompt_idx = _to_int(row.get("prompt_idx"))
        if prompt_idx is None or prompt_idx in seen_prompt_idx:
            continue
        if only_table_valid and not _table_valid_row(row):
            continue
        selected.append(row)
        seen_prompt_idx.add(prompt_idx)
    return selected


def _prompt_text(prompt: list[dict[str, Any]]) -> str:
    for msg in reversed(prompt):
        if msg.get("role") == "user":
            return str(msg.get("content", ""))
    return str(prompt[-1].get("content", "")) if prompt else ""


def _extract_answer(text: str) -> str | None:
    match = ANSWER_RE.search(text or "")
    return match.group(1).strip() if match else None


def _extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            return json.loads(stripped[start : end + 1])
        raise


def _find_span(text: str, needle: str) -> list[int] | None:
    if not needle:
        return None
    start = text.find(needle)
    if start < 0:
        compact_text = re.sub(r"\s+", " ", text)
        compact_needle = re.sub(r"\s+", " ", needle)
        compact_start = compact_text.find(compact_needle)
        if compact_start < 0:
            return None
        return None
    return [int(start), int(start + len(needle))]


def build_messages(question: str, clean_text: str, answer: str | None) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You create controlled GSM8K reasoning perturbations for evaluation. "
                "Return valid JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                "Rewrite exactly one reasoning step inside the <think> block so that it is "
                "fluent, plausible, and mathematically wrong. Do not make a typo-like edit "
                "or a single obvious operator/number swap. Invalid: changing only one number, "
                "operator, symbol, or final arithmetic value. Prefer changing one full sentence "
                "or reasoning step by altering the stated reasoning assumption, quantity "
                "definition, or relation between quantities. Preserve all other reasoning text as much as possible. "
                "Keep the final <answer> unchanged, even if the edited reasoning becomes "
                "inconsistent with it. Keep the <think>...</think><answer>...</answer> format.\n\n"
                "Return JSON with exactly these keys:\n"
                "- perturbed_trace: full edited trace including tags\n"
                "- original_step: exact original step/sentence you replaced\n"
                "- perturbed_step: exact replacement step/sentence in perturbed_trace\n"
                "- corruption_summary: one short phrase describing the wrong reasoning\n\n"
                f"Question:\n{question}\n\n"
                f"Gold final answer: {answer or ''}\n\n"
                f"Clean trace:\n{clean_text}\n"
            ),
        },
    ]


def make_azure_client(args: argparse.Namespace):
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    endpoint = args.azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT") or DEFAULT_ENDPOINT
    api_version = args.api_version or os.environ.get("AZURE_OPENAI_API_VERSION") or DEFAULT_API_VERSION
    endpoint, _deployment, api_version = _normalize_azure_endpoint(
        endpoint=endpoint,
        deployment=str(args.deployment),
        api_version=api_version,
    )

    if api_key:
        return AzureOpenAI(
            azure_endpoint=endpoint,
            api_version=api_version,
            api_key=api_key,
        )
    raise ValueError("AZURE_OPENAI_API_KEY is not set.")


def get_thread_azure_client(args: argparse.Namespace):
    client = getattr(_THREAD_LOCAL, "azure_client", None)
    if client is None:
        client = make_azure_client(args)
        _THREAD_LOCAL.azure_client = client
    return client


def call_azure(
    azure_client,
    messages: list[dict[str, str]],
    deployment: str,
    max_completion_tokens: int,
    reasoning_effort: str | None,
) -> str:
    kwargs = {
        "model": deployment,
        "messages": messages,
        "n": 1,
        "max_completion_tokens": max_completion_tokens,
    }
    if reasoning_effort:
        effort = str(reasoning_effort).lower()
        if effort in {"off", "none", "false", "0"}:
            kwargs["reasoning_effort"] = "none"
        elif effort != "omit":
            kwargs["reasoning_effort"] = str(reasoning_effort)
    response = azure_client.chat.completions.create(**kwargs)
    content = response.choices[0].message.content
    if not content:
        raise ValueError("Azure response did not contain message content.")
    return content.strip()


def _step_tokens(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", text or "", flags=re.UNICODE)


def _changed_tokens(a: str, b: str) -> tuple[list[str], list[str], int]:
    a_toks = _step_tokens(a)
    b_toks = _step_tokens(b)
    sm = SequenceMatcher(a=a_toks, b=b_toks, autojunk=False)
    old_changed: list[str] = []
    new_changed: list[str] = []
    edit_count = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        old_changed.extend(a_toks[i1:i2])
        new_changed.extend(b_toks[j1:j2])
        edit_count += max(i2 - i1, j2 - j1)
    return old_changed, new_changed, edit_count


def _is_numeric_or_operator_token(tok: str) -> bool:
    return bool(re.fullmatch(r"[\d,.$%/+-]+|[=×x*÷–—-]", tok.strip()))


def _validate_step_edit(
    original_step: str,
    perturbed_step: str,
    reject_token_like_edits: bool,
    min_step_token_edits: int,
) -> dict[str, Any]:
    old_changed, new_changed, edit_count = _changed_tokens(original_step, perturbed_step)
    changed_all = old_changed + new_changed
    numeric_or_operator_only = bool(changed_all) and all(
        _is_numeric_or_operator_token(tok) for tok in changed_all
    )
    stats = {
        "step_edit_count": int(edit_count),
        "numeric_or_operator_only": bool(numeric_or_operator_only),
        "old_changed_tokens": old_changed,
        "new_changed_tokens": new_changed,
    }
    if reject_token_like_edits:
        if edit_count < int(min_step_token_edits):
            raise ValueError(
                f"Step edit too small: {edit_count} token edits < {int(min_step_token_edits)}."
            )
        if numeric_or_operator_only:
            raise ValueError("Step edit only changed numeric/operator tokens.")
    return stats


def validate_perturbation(
    clean_text: str,
    obj: dict[str, Any],
    reject_token_like_edits: bool,
    min_step_token_edits: int,
) -> tuple[str, list[int], dict[str, Any]]:
    perturbed = str(obj.get("perturbed_trace", "")).strip()
    if not perturbed:
        raise ValueError("Missing perturbed_trace.")
    if not THINK_RE.search(perturbed) or not ANSWER_RE.search(perturbed):
        raise ValueError("Perturbed trace is missing <think> or <answer> tags.")
    if perturbed == clean_text:
        raise ValueError("Perturbed trace is identical to the clean trace.")

    clean_answer = _extract_answer(clean_text)
    pert_answer = _extract_answer(perturbed)
    if clean_answer is not None and pert_answer != clean_answer:
        raise ValueError(f"Answer changed from {clean_answer!r} to {pert_answer!r}.")

    original_step = str(obj.get("original_step", "")).strip()
    perturbed_step = str(obj.get("perturbed_step", "")).strip()
    if not original_step:
        raise ValueError("Missing original_step.")
    if not perturbed_step:
        raise ValueError("Missing perturbed_step.")
    if original_step not in clean_text:
        raise ValueError("original_step is not an exact substring of clean_text.")
    span = _find_span(perturbed, perturbed_step)
    if span is None:
        raise ValueError("perturbed_step is not an exact substring of perturbed_trace.")
    edit_stats = _validate_step_edit(
        original_step=original_step,
        perturbed_step=perturbed_step,
        reject_token_like_edits=reject_token_like_edits,
        min_step_token_edits=min_step_token_edits,
    )
    return perturbed, span, edit_stats


def _load_done_prompt_ids(path: Path) -> set[int]:
    done: set[int] = set()
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
            if prompt_idx is not None and row.get("error") is None and row.get("wrong_text"):
                done.add(prompt_idx)
    return done


def process_row(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    question = _prompt_text(row["prompt"])
    clean_text = row["clean_text"]
    answer = row.get("answer") or _extract_answer(clean_text)
    messages = build_messages(question=question, clean_text=clean_text, answer=answer)

    out: dict[str, Any] = {
        "prompt_idx": row.get("prompt_idx"),
        "clean_generation_idx": row.get("clean_generation_idx"),
        "prompt": row.get("prompt"),
        "question": question,
        "answer": answer,
        "clean_text": clean_text,
        "correct_text": clean_text,
        "source_pair_details": str(args.source_pair_details),
        "source_perturbation_fns": row.get("perturb_fns"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "azure_deployment": args.deployment,
    }

    if args.dry_run:
        out["messages"] = messages
        out["dry_run"] = True
        return out

    azure_client = get_thread_azure_client(args)
    errors: list[str] = []
    raw = None
    parsed: dict[str, Any] | None = None
    success = False
    attempt_messages = list(messages)
    for attempt in range(1, int(args.max_retries) + 2):
        try:
            raw = call_azure(
                azure_client=azure_client,
                messages=attempt_messages,
                deployment=str(args.deployment),
                max_completion_tokens=int(args.max_completion_tokens),
                reasoning_effort=str(args.reasoning_effort),
            )
            parsed = _extract_json(raw)
            perturbed, span, edit_stats = validate_perturbation(
                clean_text,
                parsed,
                reject_token_like_edits=bool(args.reject_token_like_edits),
                min_step_token_edits=int(args.min_step_token_edits),
            )
            out.update(
                {
                    "pert_text": perturbed,
                    "wrong_text": perturbed,
                    "original_step": str(parsed.get("original_step", "")).strip(),
                    "perturbed_step": str(parsed.get("perturbed_step", "")).strip(),
                    "corruption_summary": str(parsed.get("corruption_summary", "")).strip(),
                    "target_char_span": span,
                    "edit_stats": edit_stats,
                    "raw_model_output": raw,
                    "attempts": attempt,
                    "error": None,
                }
            )
            success = True
            break
        except Exception as exc:
            err = str(exc)
            errors.append(err)
            if attempt <= int(args.max_retries):
                attempt_messages = list(messages) + [
                    {
                        "role": "user",
                        "content": (
                            "Your previous attempt failed validation: "
                            f"{err}\nTry again. Rewrite one full reasoning step; do not make only "
                            "a single numeric/operator edit; keep the final answer unchanged; "
                            "return valid JSON only."
                        ),
                    }
                ]
                if float(args.retry_sleep_seconds) > 0:
                    time.sleep(float(args.retry_sleep_seconds))

    if not success:
        out.update(
            {
                "pert_text": None,
                "wrong_text": None,
                "target_char_span": None,
                "raw_model_output": raw,
                "parsed_model_output": parsed,
                "attempts": int(args.max_retries) + 1,
                "validation_errors": errors,
                "error": errors[-1] if errors else "Unknown error",
            }
        )

    if float(args.sleep_seconds) > 0:
        time.sleep(float(args.sleep_seconds))
    return out


def _error_row(row: dict[str, Any], args: argparse.Namespace, exc: BaseException) -> dict[str, Any]:
    clean_text = row.get("clean_text")
    if isinstance(clean_text, str):
        answer = row.get("answer") or _extract_answer(clean_text)
    else:
        clean_text = ""
        answer = row.get("answer")
    prompt = row.get("prompt") if isinstance(row.get("prompt"), list) else []
    question = _prompt_text(prompt)
    return {
        "prompt_idx": row.get("prompt_idx"),
        "clean_generation_idx": row.get("clean_generation_idx"),
        "prompt": row.get("prompt"),
        "question": question,
        "answer": answer,
        "clean_text": clean_text,
        "correct_text": clean_text,
        "source_pair_details": str(args.source_pair_details),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "azure_deployment": args.deployment,
        "pert_text": None,
        "wrong_text": None,
        "target_char_span": None,
        "attempts": 0,
        "validation_errors": [str(exc)],
        "error": str(exc),
    }


def write_result(f, out: dict[str, Any], written: int, total: int) -> int:
    f.write(json.dumps(out, ensure_ascii=False) + "\n")
    f.flush()
    written += 1
    status = "ok" if out.get("error") is None else "error"
    print(
        f"[{written}/{total}] prompt_idx={out.get('prompt_idx')} "
        f"status={status} attempts={out.get('attempts', 0)}",
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
            f"Output already exists: {args.output_file}. Use --force to overwrite or --resume to append."
        )

    rng = random.Random(int(args.seed))
    rows = select_source_rows(
        _load_jsonl(args.source_pair_details),
        only_table_valid=bool(args.only_table_valid),
    )
    rows = rows[int(args.start_index) :]
    if int(args.max_examples) > 0:
        rows = rows[: int(args.max_examples)]
    if args.shuffle:
        rng.shuffle(rows)

    done_prompt_ids = _load_done_prompt_ids(args.output_file) if args.resume else set()
    rows = [
        row for row in rows
        if _to_int(row.get("prompt_idx")) not in done_prompt_ids
    ]

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    mode = "a" if args.output_file.exists() and args.resume and not args.force else "w"
    with args.output_file.open(mode) as f:
        if int(args.num_workers) == 1:
            for row in rows:
                written = write_result(f, process_row(row, args), written, len(rows))
        else:
            with ThreadPoolExecutor(max_workers=int(args.num_workers)) as executor:
                future_to_row = {executor.submit(process_row, row, args): row for row in rows}
                for future in as_completed(future_to_row):
                    row = future_to_row[future]
                    try:
                        out = future.result()
                    except Exception as exc:
                        out = _error_row(row, args, exc)
                    written = write_result(f, out, written, len(rows))

    print(f"Wrote {written} rows to {args.output_file}")


if __name__ == "__main__":
    main()
