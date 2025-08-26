#!/usr/bin/env python3
"""
Create the 'countdown' dataset splits from
'anmolagarwal999/train_countdown_sft_deepseek_qwen_distilled_32b_dataset'.

Outputs a Hugging Face DatasetDict saved with save_to_disk().
"""

import argparse
import re
from typing import Optional

import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.model_selection import train_test_split


def extract_answer(text: str) -> Optional[str]:
    """Extract content between <answer>...</answer> tags."""
    if not isinstance(text, str):
        return None
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return match.group(1).strip() if match else None


def extract_think(text: str) -> Optional[str]:
    """Extract content before the first </think> tag (inclusive end tag not kept)."""
    if not isinstance(text, str):
        return None
    match = re.search(r"(.*?)</think>", text, re.DOTALL)
    return match.group(1).strip() if match else None


def build_dataframe(max_len) -> pd.DataFrame:
    # Load HF dataset
    ds = load_dataset(
        "anmolagarwal999/train_countdown_sft_deepseek_qwen_distilled_32b_dataset"
    )["train"]

    # Convert to pandas
    df = ds.to_pandas()

    # Be safe with missing counts
    for col in ["num_gt_tokens", "num_prompt_tokens"]:
        if col not in df.columns:
            df[col] = 0
    df[["num_gt_tokens", "num_prompt_tokens"]] = df[
        ["num_gt_tokens", "num_prompt_tokens"]
    ].fillna(0)

    # Filter short examples
    df = df[(df["num_gt_tokens"] + df["num_prompt_tokens"]) < max_len]

    # Expect 'conversations' like a list of dicts with "value" fields:
    #   x[1]["value"] contains the question with "Show your work"
    #   x[2]["value"] contains <think>... </think><answer>...</answer>
    def get_question(conv):
        try:
            return conv[1]["value"].split("Show your work")[0].strip()
        except Exception:
            return None

    def get_target(conv):
        try:
            return extract_think(conv[2]["value"])
        except Exception:
            return None

    def get_answer(conv):
        try:
            return extract_answer(conv[2]["value"])
        except Exception:
            return None

    df["question"] = df["conversations"].apply(get_question)
    df["target"] = df["conversations"].apply(get_target)
    df["answer"] = df["conversations"].apply(get_answer)

    keep_cols = [
        "question",
        "target",
        "answer",
        "num_prompt_tokens",
        "num_gt_tokens",
        "reward_model",
    ]
    # Only keep columns that exist
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].dropna(subset=["question", "answer"]).reset_index(drop=True)
    return df


def make_splits(
    df: pd.DataFrame, test_size: int, val_size: int, seed: int
) -> DatasetDict:
    if len(df) < (test_size + val_size + 1):
        raise ValueError(
            f"Not enough rows ({len(df)}) for requested splits: "
            f"test={test_size}, val={val_size}."
        )

    df_train, df_test = train_test_split(df, test_size=test_size, random_state=seed)
    df_train, df_val = train_test_split(df_train, test_size=val_size, random_state=seed)

    return DatasetDict(
        {
            "train": Dataset.from_pandas(df_train, preserve_index=False),
            "eval": Dataset.from_pandas(df_val, preserve_index=False),
            "test": Dataset.from_pandas(df_test, preserve_index=False),
        }
    )


def main():
    parser = argparse.ArgumentParser(description="Create countdown dataset splits.")
    parser.add_argument(
        "--outdir", type=str, default="../data/countdown", help="Output directory."
    )
    parser.add_argument("--test_size", type=int, default=400, help="Test set size.")
    parser.add_argument("--val_size", type=int, default=400, help="Validation set size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--max_len", type=int, default=512, help="Max length of the prompt+response.")
    args = parser.parse_args()

    print("Loading and preparing dataframe...")
    df = build_dataframe(args.max_len)
    print(f"Total rows after filtering: {len(df)}")

    print("Creating splits...")
    dsd = make_splits(df, test_size=args.test_size, val_size=args.val_size, seed=args.seed)

    print(f"Saving to {args.outdir} ...")
    dsd.save_to_disk(args.outdir)

    print("Done.")
    print(
        {
            "train": len(dsd["train"]),
            "eval": len(dsd["eval"]),
            "test": len(dsd["test"]),
            "outdir": args.outdir,
        }
    )


if __name__ == "__main__":
    main()