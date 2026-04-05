import argparse
import os
import re
from collections import Counter
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer


SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


SOURCE_ALIASES: Dict[str, List[str]] = {
    "aime_amc": ["aime_amc", "amc_aime"],
    "amc_aime": ["amc_aime", "aime_amc"],
}

TOKENIZER_ALIASES: Dict[str, str] = {
    "qwen2.b-3b-instruct": "Qwen/Qwen2.5-3B-Instruct",
}


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def strip_outer_math_delimiters(text: str) -> str:
    if not text:
        return text
    text = text.strip()
    text = re.sub(r"^\$+|\$+$", "", text).strip()
    text = re.sub(r"^\\\((.*)\\\)$", r"\1", text, flags=re.DOTALL).strip()
    text = re.sub(r"^\\\[(.*)\\\]$", r"\1", text, flags=re.DOTALL).strip()
    return text


def unwrap_latex_wrappers(text: str) -> str:
    if not text:
        return text
    wrappers = ("text", "textbf", "mathrm", "operatorname", "mbox")
    out = text.strip()
    changed = True
    while changed:
        changed = False
        for wrapper in wrappers:
            pattern = rf"^\\{wrapper}\s*{{(.*)}}$"
            m = re.match(pattern, out, flags=re.DOTALL)
            if m:
                out = m.group(1).strip()
                changed = True
    return out


def clean_extracted_answer(text: str) -> str:
    if not text:
        return ""
    out = text.strip()
    out = out.replace("\u00a0", " ")
    out = strip_outer_math_delimiters(out)
    out = unwrap_latex_wrappers(out)
    out = re.sub(r"^#+\s*", "", out).strip()
    out = re.sub(
        r"(?i)^(final answer|answer|thus|therefore|so|hence)\s*(is|:)?\s*",
        "",
        out,
    ).strip()
    out = out.strip(" .")
    return out


def extract_boxed_contents(text: str) -> List[str]:
    """
    Extract every '\\boxed{...}' payload, handling nested braces.
    """
    if not text:
        return []

    results: List[str] = []
    marker = r"\boxed"
    i = 0
    n = len(text)
    while i < n:
        start = text.find(marker, i)
        if start == -1:
            break
        j = start + len(marker)
        while j < n and text[j].isspace():
            j += 1
        if j >= n or text[j] != "{":
            i = j
            continue

        depth = 0
        content_start = j + 1
        k = j
        while k < n:
            ch = text[k]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    results.append(text[content_start:k].strip())
                    break
            k += 1

        i = k + 1

    return results


def extract_answer_from_solution(solution: str) -> Tuple[str, str]:
    """
    Heuristic extraction because NuminaMath-CoT's amc_aime rows are plain solutions.
    Returns: (answer, extraction_method)
    """
    if not solution:
        return "", "empty_solution"

    boxed = extract_boxed_contents(solution)
    if boxed:
        return clean_extracted_answer(boxed[-1]), "boxed"

    hash_match = re.search(r"(?im)^####\s*(.+?)\s*$", solution)
    if hash_match:
        return clean_extracted_answer(hash_match.group(1)), "hash_delimiter"

    phrase_patterns = [
        r"(?is)final answer\s*(?:is|:)\s*(.+?)(?:$|\n)",
        r"(?is)therefore,?\s+the\s+answer\s*(?:is|:)\s*(.+?)(?:$|\n)",
        r"(?is)thus,?\s+the\s+answer\s*(?:is|:)\s*(.+?)(?:$|\n)",
        r"(?is)the\s+correct\s+answer\s*(?:is|:)\s*(.+?)(?:$|\n)",
    ]
    for pattern in phrase_patterns:
        matches = re.findall(pattern, solution)
        if matches:
            candidate = clean_extracted_answer(matches[-1])
            if candidate:
                return candidate, "final_answer_phrase"

    tail = solution[-300:]
    option_match = re.search(r"(?i)(?:choice|answer)\D{0,25}\(?([A-E])\)?", tail)
    if option_match:
        return option_match.group(1).upper(), "choice_letter_tail"

    lines = [ln.strip() for ln in solution.splitlines() if ln.strip()]
    if lines:
        return clean_extracted_answer(lines[-1]), "last_line_fallback"

    return "", "unparsed"


def parse_whole_number(answer: str) -> Optional[int]:
    """
    Parse a non-negative whole number from answer text.
    Accepts representations like "199", "199.0", "(C) 199.00", "answer is 199.0".
    Returns None if not parseable as a whole number.
    """
    if not answer:
        return None

    text = normalize_whitespace(clean_extracted_answer(answer))
    text = text.replace(",", "")
    text = text.replace(r"\,", "")
    text = text.replace(r"\ ", " ")

    def parse_integral_numeric_string(s: str) -> Optional[int]:
        if not re.fullmatch(r"[+-]?\d+(?:\.\d+)?", s):
            return None
        try:
            value = Decimal(s)
        except InvalidOperation:
            return None
        if value != value.to_integral_value():
            return None
        as_int = int(value)
        return as_int if as_int >= 0 else None

    direct = parse_integral_numeric_string(text)
    if direct is not None:
        return direct

    with_choice = re.fullmatch(
        r"\(?[A-Ea-e]\)?\s*[-:\)]?\s*([+-]?\d+(?:\.\d+)?)", text
    )
    if with_choice:
        parsed = parse_integral_numeric_string(with_choice.group(1))
        if parsed is not None:
            return parsed

    tokens = re.findall(r"(?<![\d.])[+-]?\d+(?:\.\d+)?(?![\d.])", text)
    if len(tokens) == 1:
        parsed = parse_integral_numeric_string(tokens[0])
        if parsed is not None:
            return parsed

    return None


def extract_integer_0_999(answer: str) -> Optional[int]:
    """
    Return an integer answer in [0, 999] when clearly parseable, else None.
    Accepts float-looking forms like "199.0" only when they are whole numbers.
    """
    whole = parse_whole_number(answer)
    if whole is None:
        return None
    return whole if whole <= 999 else None


def count_tokens(tokenizer: Any, text: str) -> int:
    if not text:
        return 0
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def token_count_chat(
    tokenizer: Any, messages: List[Dict[str, str]], add_generation_prompt: bool = False
) -> int:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
            )
            return len(ids)
        except TypeError:
            ids = tokenizer.apply_chat_template(messages, tokenize=True)
            return len(ids)
        except Exception:
            pass

    text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
    if add_generation_prompt:
        text += "\nassistant:"
    return count_tokens(tokenizer, text)


def resolve_source(ds: Dataset, requested_source: str) -> str:
    source_counts = Counter(ds["source"])

    if source_counts.get(requested_source, 0) > 0:
        return requested_source

    for candidate in SOURCE_ALIASES.get(requested_source, []):
        if source_counts.get(candidate, 0) > 0:
            print(
                f"[info] Requested source '{requested_source}' was not found. "
                f"Using '{candidate}' instead."
            )
            return candidate

    available = ", ".join(sorted(source_counts.keys()))
    raise ValueError(
        f"Source '{requested_source}' not found in dataset. Available sources: {available}"
    )


def load_and_filter_source(
    dataset_name: str, split: str, source: str, cache_dir: Optional[str]
) -> Tuple[Dataset, str]:
    kwargs = {}
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        kwargs["cache_dir"] = cache_dir

    ds = load_dataset(dataset_name, split=split, **kwargs)
    resolved_source = resolve_source(ds, source)

    source_col = ds["source"]
    selected_indices = [i for i, s in enumerate(source_col) if s == resolved_source]
    filtered = ds.select(selected_indices)
    print(f"[info] Loaded {len(filtered)} rows for source='{resolved_source}'.")
    return filtered, resolved_source


def format_sft_rows(ds: Dataset) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx, ex in enumerate(tqdm(ds, desc="Formatting SFT rows")):
        question = (ex.get("problem") or "").strip()
        reasoning = (ex.get("solution") or "").strip()
        answer, answer_method = extract_answer_from_solution(reasoning)
        answer_whole_number = parse_whole_number(answer)
        answer_int_0_999 = extract_integer_0_999(answer)
        if answer_whole_number is not None:
            # Canonicalize whole-number answers to integer text, e.g. "199.0" -> "199".
            answer = str(answer_whole_number)

        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        target = (
            "<think>\n"
            f"{reasoning}\n"
            "</think>\n"
            "<answer>\n"
            f"{answer}\n"
            "</answer>"
        )

        rows.append(
            {
                "id": idx,
                "source": ex.get("source", ""),
                "question": question,
                "prompt": prompt,
                "reasoning": reasoning,
                "answer": answer,
                "target": target,
                "answer_parse_method": answer_method,
                "answer_extracted": bool(normalize_whitespace(answer)),
                "answer_whole_number": answer_whole_number,
                "is_whole_number_answer": answer_whole_number is not None,
                "answer_int_0_999": answer_int_0_999,
                "is_int_answer_0_999": answer_int_0_999 is not None,
            }
        )
    return rows


def add_token_stats(ds: Dataset, tokenizer: Any) -> Dataset:
    rows = []
    for ex in tqdm(ds, desc="Tokenizing prompt/response"):
        prompt_messages = ex["prompt"]
        assistant_message = [{"role": "assistant", "content": ex["target"]}]

        prompt_num_tokens = token_count_chat(
            tokenizer, prompt_messages, add_generation_prompt=True
        )
        full_chat_num_tokens = token_count_chat(
            tokenizer, prompt_messages + assistant_message, add_generation_prompt=False
        )
        generated_num_tokens = max(full_chat_num_tokens - prompt_num_tokens, 0)

        response_num_tokens = count_tokens(tokenizer, ex["target"])
        cot_num_tokens = count_tokens(tokenizer, ex["reasoning"])
        answer_num_tokens = count_tokens(tokenizer, ex["answer"])

        row = dict(ex)
        row.update(
            {
                "prompt_num_tokens": int(prompt_num_tokens),
                "response_num_tokens": int(response_num_tokens),
                "cot_num_tokens": int(cot_num_tokens),
                "answer_num_tokens": int(answer_num_tokens),
                "generated_num_tokens": int(generated_num_tokens),
                "full_chat_num_tokens": int(full_chat_num_tokens),
            }
        )
        rows.append(row)
    return Dataset.from_list(rows)


def filter_by_token_budget(
    ds: Dataset, max_prompt_tokens: int, max_generated_tokens: int
) -> Dataset:
    before = len(ds)
    filtered = ds.filter(
        lambda x: (
            x["prompt_num_tokens"] < max_prompt_tokens
            and x["generated_num_tokens"] < max_generated_tokens
        )
    )
    after = len(filtered)
    dropped = before - after
    drop_pct = (100.0 * dropped / before) if before else 0.0
    print(
        "[info] Applied token filter: "
        f"prompt_num_tokens < {max_prompt_tokens} and "
        f"generated_num_tokens < {max_generated_tokens}. "
        f"Kept {after}/{before} rows (dropped {dropped}, {drop_pct:.2f}%)."
    )
    return filtered


def filter_to_int_0_999_answers(ds: Dataset) -> Dataset:
    before = len(ds)
    filtered = ds.filter(lambda x: x["is_int_answer_0_999"])
    after = len(filtered)
    dropped = before - after
    drop_pct = (100.0 * dropped / before) if before else 0.0
    print(
        "[info] Kept only integer answers in [0, 999]. "
        f"Kept {after}/{before} rows (dropped {dropped}, {drop_pct:.2f}%)."
    )
    return filtered


def load_tokenizer_with_aliases(tokenizer_name: str) -> Any:
    try:
        return AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    except Exception as first_error:
        alias = TOKENIZER_ALIASES.get(tokenizer_name.lower())
        if alias and alias != tokenizer_name:
            print(
                f"[info] Tokenizer '{tokenizer_name}' not found. "
                f"Falling back to '{alias}'."
            )
            return AutoTokenizer.from_pretrained(alias, trust_remote_code=True)
        raise first_error


def make_splits(df: Dataset, test_size: int, seed: int) -> DatasetDict:
    if len(df) <= 1:
        print(
            "[info] Dataset too small for train/test split. "
            "Saving everything in the train split."
        )
        return DatasetDict({"train": df})

    effective_test_size = min(test_size, len(df) - 1)
    if effective_test_size != test_size:
        print(
            f"[info] Requested test_size={test_size} exceeds available data. "
            f"Using test_size={effective_test_size}."
        )

    split = df.train_test_split(test_size=effective_test_size, seed=seed)
    return DatasetDict({"train": split["train"], "test": split["test"]})


def word_count(text: str) -> int:
    return len(str(text).split()) if text else 0


def print_comprehensive_stats(dsd: DatasetDict) -> None:
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)

    combined_df = pd.concat(
        [dsd[split].to_pandas() for split in dsd.keys()], ignore_index=True
    )

    for split in dsd.keys():
        print(f"{split.upper()}: {len(dsd[split])} examples")

    print("\nAverage token lengths:")
    print(f"  prompt_num_tokens:    {combined_df['prompt_num_tokens'].mean():.1f}")
    print(f"  generated_num_tokens: {combined_df['generated_num_tokens'].mean():.1f}")
    print(f"  response_num_tokens:  {combined_df['response_num_tokens'].mean():.1f}")
    print(f"  cot_num_tokens:       {combined_df['cot_num_tokens'].mean():.1f}")
    print(f"  answer_num_tokens:    {combined_df['answer_num_tokens'].mean():.1f}")

    print("\nAverage word lengths:")
    print(f"  reasoning words: {combined_df['reasoning'].apply(word_count).mean():.1f}")
    print(f"  answer words:    {combined_df['answer'].apply(word_count).mean():.1f}")

    extraction_rate = 100.0 * combined_df["answer_extracted"].mean()
    print(f"\nAnswer extraction success: {extraction_rate:.2f}%")

    parse_counts = combined_df["answer_parse_method"].value_counts()
    print("Answer parse methods:")
    for method, count in parse_counts.items():
        print(f"  {method}: {count}")

    int_count_total = int(combined_df["is_int_answer_0_999"].sum())
    total_count = len(combined_df)
    int_pct_total = (100.0 * int_count_total / total_count) if total_count else 0.0
    print(
        "\nInteger answers in [0, 999]: "
        f"{int_count_total}/{total_count} ({int_pct_total:.2f}%)"
    )
    print("Integer-answer count by split:")
    for split in dsd.keys():
        split_df = dsd[split].to_pandas()
        split_total = len(split_df)
        split_int = int(split_df["is_int_answer_0_999"].sum())
        split_pct = (100.0 * split_int / split_total) if split_total else 0.0
        print(f"  {split}: {split_int}/{split_total} ({split_pct:.2f}%)")

    print("=" * 60 + "\n")


def export_readable_files(dsd: DatasetDict, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for split, dataset in dsd.items():
        dataset.to_csv(os.path.join(output_dir, f"{split}.csv"), index=False)
        dataset.to_json(
            os.path.join(output_dir, f"{split}.jsonl"), orient="records", lines=True
        )


def build_dataset(args: argparse.Namespace) -> Dataset:
    raw_ds, resolved_source = load_and_filter_source(
        dataset_name=args.dataset_name,
        split=args.split,
        source=args.source,
        cache_dir=args.cache_dir,
    )

    if args.max_samples is not None:
        raw_ds = raw_ds.select(range(min(len(raw_ds), args.max_samples)))
        print(f"[info] Using first {len(raw_ds)} rows after max_samples.")

    print(f"[info] Formatting examples for SFT from source '{resolved_source}'.")
    rows = format_sft_rows(raw_ds)
    formatted_ds = Dataset.from_list(rows)
    formatted_ds = filter_to_int_0_999_answers(formatted_ds)

    print(f"[info] Loading tokenizer: {args.tokenizer}")
    tokenizer = load_tokenizer_with_aliases(args.tokenizer)

    tokenized_ds = add_token_stats(formatted_ds, tokenizer=tokenizer)
    filtered_ds = filter_by_token_budget(
        tokenized_ds,
        max_prompt_tokens=args.max_prompt_tokens,
        max_generated_tokens=args.max_generated_tokens,
    )
    return filtered_ds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="AI-MO/NuminaMath-CoT")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--source", type=str, default="aime_amc")
    parser.add_argument("--tokenizer", type=str, default="qwen2.b-3b-instruct")
    parser.add_argument("--outdir", type=str, default="data/aime_amc_sft")
    parser.add_argument("--cache_dir", type=str, default="/tmp/hf/datasets")
    parser.add_argument("--test_size", type=int, default=200)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_prompt_tokens", type=int, default=300)
    parser.add_argument("--max_generated_tokens", type=int, default=824)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = build_dataset(args)
    dsd = make_splits(df, test_size=args.test_size, seed=args.seed)

    dsd.save_to_disk(args.outdir)
    export_readable_files(dsd, args.outdir)
    print_comprehensive_stats(dsd)
    print(f"[info] Saved dataset artifacts to: {args.outdir}")


if __name__ == "__main__":
    main()
