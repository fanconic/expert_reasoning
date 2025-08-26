#!/usr/bin/env python3
"""
Create the 'medical_o1' dataset by merging:
- FreedomIntelligence/medical-o1-verifiable-problem  (GRPO-ish set)
- FreedomIntelligence/medical-o1-reasoning-SFT (config: 'en')  (SFT set)

We:
- Rename columns to a common schema
- Inner-merge on question
- Filter by total word count (< 512 across question + reasoning + response)
- Split into train/eval/test by fixed sizes
- Save a DatasetDict via save_to_disk()
"""

import argparse
from typing import Any

import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.model_selection import train_test_split


def word_count(x: Any) -> int:
    s = "" if x is None else str(x)
    return len(s.split())


def build_dataframe(max_len) -> pd.DataFrame:
    # Load both datasets
    data_grpo = load_dataset("FreedomIntelligence/medical-o1-verifiable-problem")[
        "train"
    ]
    data_sft = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en")[
        "train"
    ]

    df_grpo = data_grpo.to_pandas()
    df_sft = data_sft.to_pandas()

    # Normalise column names
    df_grpo = df_grpo.rename(
        columns={
            "Open-ended Verifiable Question": "question",
            "Ground-True Answer": "answer",
        }
    )
    df_sft = df_sft.rename(
        columns={
            "Question": "question",
            "Complex_CoT": "reasoning",
            "Response": "response",
        }
    )

    # Inner join on question to ensure aligned pairs
    df_merged = df_sft.merge(df_grpo, on="question", suffixes=("_sft", "_grpo"), how="inner")

    # Length-based filtering
    df_merged["len_question"] = df_merged["question"].apply(word_count)
    df_merged["len_reasoning"] = df_merged["reasoning"].apply(word_count)
    df_merged["len_response"] = df_merged["response"].apply(word_count)
    df_merged["sum_words"] = (
        df_merged["len_question"] + df_merged["len_reasoning"] + df_merged["len_response"]
    )

    df_merged = df_merged[df_merged["sum_words"] < max_len].reset_index(drop=True)

    # Keep compact, useful columns
    keep_cols = ["question", "reasoning", "response", "answer"]
    keep_cols = [c for c in keep_cols if c in df_merged.columns]
    return df_merged[keep_cols]


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
    parser = argparse.ArgumentParser(description="Create medical_o1 dataset splits.")
    parser.add_argument(
        "--outdir", type=str, default="data/medical_o1", help="Output directory."
    )
    parser.add_argument("--test_size", type=int, default=2000, help="Test set size.")
    parser.add_argument("--val_size", type=int, default=2000, help="Validation set size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--max_len", type=int, default=512, help="Max length of the prompt+response.")
    args = parser.parse_args()

    print("Loading and merging data...")
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