import argparse
from typing import Any

from datasets import load_dataset, Dataset, DatasetDict
from sklearn.model_selection import train_test_split
import textwrap
import pandas as pd

def word_count(x: Any) -> int:
    s = "" if x is None else str(x)
    return len(s.split())


# Replace pandas-based build_dataframe with a HF Dataset-based pipeline
def build_dataframe(max_len) -> Dataset:
    """
    Build a HuggingFace Dataset (not a pandas DataFrame).
    - keep only examples from dataset_name in ['medqa','medmcqa']
    - concatenates question + options
    - extracts response as text before ". Explanation:" if present, otherwise uses the original answer
    - sets answer = response (matching original script)
    - computes len_question, len_reasoning, len_response, sum_words
    - filters by sum_words < max_len
    - keeps only ["question","reasoning","response","answer"]
    """
    ds = load_dataset("UCSC-VLAA/MedReason")["train"]

    # keep only the two desired source datasets
    ds = ds.filter(lambda x: x.get("dataset_name", "") in ["medqa", "medmcqa"])

    def munge(example):
        # create concatenated question and extract response/answer
        q = example.get("question", "") or ""
        opts = example.get("options", "") or ""
        full_q = q + "\n" + opts if opts else q

        raw_answer = example.get("answer", "") or ""
        # Use the part before ". Explanation:" if that marker exists, otherwise keep original answer
        if ". Explanation:" in raw_answer:
            resp = raw_answer.split(". Explanation:")[0]
        else:
            resp = raw_answer

        reasoning = example.get("reasoning", "") or ""

        len_q = len(str(full_q).split())
        len_r = len(str(reasoning).split())
        len_resp = len(str(resp).split())
        sum_words = len_q + len_r + len_resp

        return {
            "question": full_q,
            "reasoning": reasoning,
            "response": resp,
            "answer": resp,
            "len_question": len_q,
            "len_reasoning": len_r,
            "len_response": len_resp,
            "sum_words": sum_words,
        }

    ds = ds.map(munge, remove_columns=[])  # keep all until we remove extras below

    # filter by total length
    ds = ds.filter(lambda x: x["sum_words"] < max_len)

    # keep useful columns including length fields so we can compute stats later
    keep_cols = ["question", "reasoning", "response", "answer", "len_question", "len_reasoning", "len_response", "sum_words"]
    cols_to_remove = [c for c in ds.column_names if c not in keep_cols]
    if cols_to_remove:
        ds = ds.remove_columns(cols_to_remove)

    return ds


def make_splits(
    df: Dataset, test_size: int, val_size: int, seed: int
) -> DatasetDict:
    """
    Create train/eval/test splits from a HuggingFace Dataset using integer sizes.
    """
    total = len(df)
    if total < (test_size + val_size + 1):
        raise ValueError(
            f"Not enough rows ({total}) for requested splits: "
            f"test={test_size}, val={val_size}."
        )

    # split off test
    first_split = df.train_test_split(test_size=test_size, seed=seed)
    ds_test = first_split["test"]
    ds_train_remaining = first_split["train"]

    # split remaining into train/val (val_size is absolute)
    second_split = ds_train_remaining.train_test_split(test_size=val_size, seed=seed)
    ds_val = second_split["test"]
    ds_train = second_split["train"]

    return DatasetDict(
        {
            "train": ds_train,
            "eval": ds_val,
            "test": ds_test,
        }
    )


def display_examples(dsd, split: str = "train", n: int = 3):
    """
    Print up to `n` examples from the given split of the DatasetDict `dsd`.
    Shows full (untruncated) question, reasoning, response, and answer.
    """
    # Ensure pandas won't truncate long strings if we accidentally show a DataFrame
    pd.set_option("display.max_colwidth", None)

    if split not in dsd:
        print(f"Split '{split}' not found in dataset. Available: {list(dsd.keys())}")
        return

    df = dsd[split].to_pandas().reset_index(drop=True)
    if df.empty:
        print(f"No examples in split '{split}'.")
        return

    n = min(n, len(df))
    sep = "=" * 80
    for i in range(n):
        row = df.iloc[i]
        print(sep)
        print(f"Example {i+1}/{n} (split='{split}')")
        print("-" * 80)
        # Use get with fallback to empty string for missing columns
        q = row.get("question", "") if isinstance(row, dict) else row.get("question", "")
        r = row.get("reasoning", "") if isinstance(row, dict) else row.get("reasoning", "")
        resp = row.get("response", "") if isinstance(row, dict) else row.get("response", "")
        ans = row.get("answer", "") if isinstance(row, dict) else row.get("answer", "")

        # Print with labels; preserve original newlines
        print("\nQuestion:\n")
        print(q)
        print("\nReasoning:\n")
        print(r)
        print("\nResponse:\n")
        print(resp)
        print("\nAnswer:\n")
        print(ans)
        print(sep)
        print()


# Add helper to print split sizes
def print_split_sizes(dsd):
    """
    Print sizes for train/eval/test splits in a tidy, aligned format.
    """
    splits = ["train", "eval", "test"]
    print("\nDataset split sizes:")
    max_name_len = max(len(s) for s in splits)
    for s in splits:
        size = len(dsd[s]) if s in dsd else 0
        print(f"  {s.rjust(max_name_len)} : {size}")
    total = sum(len(dsd[s]) for s in splits if s in dsd)
    print(f"  {'total'.rjust(max_name_len)} : {total}\n")


def print_longest_stats(ds: Dataset):
    """
    Print numeric stats only: longest question, longest reasoning, longest response,
    and the max sum_words (word counts).
    """
    if len(ds) == 0:
        print("No examples to compute longest stats.")
        return

    # Ensure length columns exist
    if "len_question" not in ds.column_names or "len_reasoning" not in ds.column_names or "len_response" not in ds.column_names or "sum_words" not in ds.column_names:
        ds = ds.map(
            lambda x: {
                "len_question": len(str(x.get("question", "")).split()),
                "len_reasoning": len(str(x.get("reasoning", "")).split()),
                "len_response": len(str(x.get("response", "")).split()),
                "sum_words": len(str(x.get("question", "")).split())
                + len(str(x.get("reasoning", "")).split())
                + len(str(x.get("response", "")).split()),
            }
        )

    q_lens = ds["len_question"]
    r_lens = ds["len_reasoning"]
    resp_lens = ds["len_response"]
    sums = ds["sum_words"]

    # compute maxima (safely)
    max_q = max(q_lens) if q_lens else 0
    max_r = max(r_lens) if r_lens else 0
    max_resp = max(resp_lens) if resp_lens else 0
    max_sum = max(sums) if sums else 0

    print("\nLongest-item statistics (word counts):")
    print(f"  Longest question : {max_q}")
    print(f"  Longest reasoning: {max_r}")
    print(f"  Longest response : {max_resp}")
    print(f"  Largest sum_words: {max_sum}\n")


def main():
    parser = argparse.ArgumentParser(description="Create medical_o1 dataset splits.")
    parser.add_argument(
        "--outdir", type=str, default="data/medreason", help="Output directory."
    )
    parser.add_argument("--test_size", type=int, default=2000, help="Test set size.")
    parser.add_argument("--val_size", type=int, default=2000, help="Validation set size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--max_len", type=int, default=512, help="Max length of the prompt+response.")
    parser.add_argument("--show_examples", action="store_true", help="Print up to 3 example records after saving.")
    parser.add_argument("--examples_split", type=str, default="train", help="Which split to show examples from (train/eval/test).")
    parser.add_argument("--examples_n", type=int, default=3, help="Max number of examples to show.")
    args = parser.parse_args()

    print("Loading and merging data...")
    df = build_dataframe(args.max_len)
    print(f"Total rows after filtering: {len(df)}")
    # print longest-question/response/sum stats from the filtered dataset
    print_longest_stats(df)

    print("Creating splits...")
    dsd = make_splits(df, test_size=args.test_size, val_size=args.val_size, seed=args.seed)

    print(f"Saving to {args.outdir} ...")
    dsd.save_to_disk(args.outdir)

    print("Done.")
    # Print sizes in a clear, formatted way and the output directory
    print_split_sizes(dsd)
    print(f"Saved dataset to: {args.outdir}")

    # Optionally display examples in a beautiful, untruncated format
    if args.show_examples:
        print("\nDisplaying examples from the dataset:\n")
        display_examples(dsd, split=args.examples_split, n=args.examples_n)


if __name__ == "__main__":
    main()