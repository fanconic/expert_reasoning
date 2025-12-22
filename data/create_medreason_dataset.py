import argparse
import re
import concurrent.futures
from typing import Any, List, Dict
import os
import textwrap
import pandas as pd
import json

from datasets import load_dataset, Dataset, DatasetDict
from openai import AzureOpenAI 
from tqdm import tqdm

from openai import AzureOpenAI


def corrupt_entry_all_distractors(row: Dict[str, Any]) -> Dict[str, Any]:
    question = row.get("question", "")
    correct_reasoning = row.get("reasoning", "")
    correct_answer = row.get("answer", "")
    
    prompt = f"""
You are a medical expert creating adversarial training examples for a multiple-choice question.
Below is a medical question, the correct reasoning, and the correct answer.

Your task:
1. Identify the 3 incorrect options (distractors) in the question.
2. For EACH of the 3 incorrect options, write a plausible reasoning trace that leads to that specific wrong answer.
3. The reasoning should sound logical and medical (hallucinating facts if necessary) but must conclude with the wrong answer.
4. The reasoning style should match the format of the correct reasoning with **Finding reasoning paths, reasoning, conclusion**

Format your output exactly as 3 sequential blocks:

<block>
<think>
[Reasoning for Distractor 1]
</think>
<answer>
[Distractor 1 Text]
</answer>
</block>

<block>
<think>
[Reasoning for Distractor 2]
</think>
<answer>
[Distractor 2 Text]
</answer>
</block>

<block>
<think>
[Reasoning for Distractor 3]
</think>
<answer>
[Distractor 3 Text]
</answer>
</block>

---
Question: 
{question}

Correct Reasoning (do not use):
{correct_reasoning}

Correct Answer (do not use):
{correct_answer}
"""

    messages = [
        {"role": "system", "content": "You are a medical expert generating adversarial wrong-reasoning examples."},
        {"role": "user", "content": prompt},
    ]

    try:
        response = client.chat.completions.create(
            model=DEPLOYMENT,
            messages=messages,
            n=1,
            reasoning_effort="minimal",
            temperature=1.0, 
        )

        chat_output = response.choices[0].message.content.strip()

        # Parse all blocks using regex
        pattern = re.compile(r"<think>\s*(.*?)\s*</think>.*?<answer>\s*(.*?)\s*</answer>", re.DOTALL)
        matches = pattern.findall(chat_output)

        corrupted_reasonings = []
        corrupted_answers = []

        for match in matches:
            corrupted_reasonings.append(match[0].strip())
            corrupted_answers.append(match[1].strip())

        row['corrupted_reasonings'] = corrupted_reasonings
        row['corrupted_answers'] = corrupted_answers
        row['is_corrupted'] = len(corrupted_reasonings) >= 3 

    except Exception as e:
        print(f"API Error: {e}")
        row['corrupted_reasonings'] = []
        row['corrupted_answers'] = []
        row['is_corrupted'] = False

    return row

def process_batch_with_threadpool(dataset_list: List[Dict], max_workers: int = 10) -> List[Dict]:
    results = []
    pbar = tqdm(total=len(dataset_list), desc="Corrupting Reasonings")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_row = {executor.submit(corrupt_entry_all_distractors, row): row for row in dataset_list}
        
        for future in concurrent.futures.as_completed(future_to_row):
            try:
                data = future.result()
                results.append(data)
            except Exception as exc:
                print(f"Row generation generated an exception: {exc}")
                orig = future_to_row[future]
                orig['is_corrupted'] = False
                orig['corrupted_reasonings'] = []
                orig['corrupted_answers'] = []
                results.append(orig)
            
            pbar.update(1)
                
    pbar.close()
    return results

def build_dataframe(max_len, max_samples=None, workers=10) -> Dataset:
    ds = load_dataset("UCSC-VLAA/MedReason")["train"]

    # Filter source datasets
    ds = ds.filter(lambda x: x.get("dataset_name", "") in ["medqa", "medmcqa"])
    
    if max_samples:
        ds = ds.select(range(min(len(ds), max_samples)))

    def pre_munge(example):
        q = example.get("question", "") or ""
        opts = example.get("options", "") or ""
        full_q = q + "\n" + opts if opts else q

        raw_answer = example.get("answer", "") or ""
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
            "sum_words": sum_words
        }

    ds = ds.map(pre_munge, remove_columns=ds.column_names)
    ds = ds.filter(lambda x: x["sum_words"] < max_len)

    print(f"Dataset size before corruption: {len(ds)}")

    data_list = [item for item in ds]
    
    print(f"Starting API calls with {workers} workers...")
    processed_list = process_batch_with_threadpool(data_list, max_workers=workers)
    
    final_ds = Dataset.from_list(processed_list)
    final_ds = final_ds.filter(lambda x: x['is_corrupted'] is True)
    
    return final_ds

def make_splits(df: Dataset, test_size: int, val_size: int, seed: int) -> DatasetDict:
    total = len(df)
    if total < (test_size + val_size + 1):
        raise ValueError(f"Not enough rows ({total}) for requested splits.")

    first_split = df.train_test_split(test_size=test_size, seed=seed)
    ds_test = first_split["test"]
    ds_train_remaining = first_split["train"]

    second_split = ds_train_remaining.train_test_split(test_size=val_size, seed=seed)
    ds_val = second_split["test"]
    ds_train = second_split["train"]

    return DatasetDict({"train": ds_train, "eval": ds_val, "test": ds_test})

def word_count(text):
    if not text: return 0
    return len(str(text).split())

def print_comprehensive_stats(dsd: DatasetDict):
    """
    Calculates and prints min/max/avg word counts for all relevant fields.
    """
    print("\n" + "="*50)
    print("DATASET STATISTICS (Word Counts)")
    print("="*50)

    combined_df = pd.concat([dsd[split].to_pandas() for split in dsd.keys()], ignore_index=True)
    total_rows = len(combined_df)

    print(f"Total Examples: {total_rows}")
    for split in dsd.keys():
        print(f"  - {split}: {len(dsd[split])}")
    print("-" * 50)

    q_lens = combined_df['question'].apply(word_count)
    print(f"QUESTION Lengths:")
    print(f"  Min: {q_lens.min():<5} Max: {q_lens.max():<5} Avg: {q_lens.mean():.1f}")

    r_lens = combined_df['reasoning'].apply(word_count)
    print(f"CORRECT REASONING Lengths:")
    print(f"  Min: {r_lens.min():<5} Max: {r_lens.max():<5} Avg: {r_lens.mean():.1f}")
    
    # Flatten the lists of corrupted reasonings to get accurate stats
    all_corrupted_reasonings = combined_df['corrupted_reasonings'].explode()
    cr_lens = all_corrupted_reasonings.apply(word_count)
    
    print(f"CORRUPTED REASONING Lengths (n={len(cr_lens)}):")
    print(f"  Min: {cr_lens.min():<5} Max: {cr_lens.max():<5} Avg: {cr_lens.mean():.1f}")
    print("="*50 + "\n")

def export_readable_files(dsd: DatasetDict, output_dir: str):
    """
    Saves the dataset splits as CSV and JSONL for easier manual inspection.
    """
    print(f"Exporting human-readable files to {output_dir}...")
    
    for split, dataset in dsd.items():
        # Define paths
        csv_path = os.path.join(output_dir, f"{split}.csv")
        jsonl_path = os.path.join(output_dir, f"{split}.jsonl")
        
        # Save CSV (Pandas handles list columns by stringifying them, e.g. "['a', 'b']")
        dataset.to_csv(csv_path, index=False)
        
        # Save JSONL (Cleaner for nested structures)
        dataset.to_json(jsonl_path, orient="records", lines=True)
        
        print(f"  Saved {split}.csv and {split}.jsonl")
    print("\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="data/medreason_corrupted_full")
    parser.add_argument("--test_size", type=int, default=1500)
    parser.add_argument("--val_size", type=int, default=1500)
    parser.add_argument("--max_samples", type=int, default=None, help="Limit rows for testing")
    parser.add_argument("--workers", type=int, default=10, help="Thread pool workers")
    args = parser.parse_args()

    # 1. Build & Corrupt
    df = build_dataframe(max_len=1024, max_samples=args.max_samples, workers=args.workers)
    
    # 2. Split
    dsd = make_splits(df, args.test_size, args.val_size, seed=42)
    
    # 3. Save HF Dataset
    dsd.save_to_disk(args.outdir)
    print(f"Saved HF dataset to {args.outdir}")
    
    # 4. Save CSV/JSONL for inspection
    export_readable_files(dsd, args.outdir)
    
    # 5. Print Stats
    print_comprehensive_stats(dsd)

if __name__ == "__main__":
    main()