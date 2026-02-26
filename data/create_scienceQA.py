import argparse
import re
import concurrent.futures
from typing import Any, List, Dict
import os
import pandas as pd
from tqdm import tqdm

from datasets import load_dataset, Dataset, DatasetDict
from openai import AzureOpenAI


def corrupt_entry_all_distractors(row: Dict[str, Any]) -> Dict[str, Any]:
    """Generates reasoning for each incorrect option in ScienceQA."""
    question = row.get("question", "")
    correct_reasoning = row.get("reasoning", "")
    correct_answer = row.get("answer", "")
    
    # Updated prompt to be more general for Science/General knowledge
    prompt = f"""
You are an expert educator creating adversarial training examples for a science question.
Below is a question, the correct reasoning, and the correct answer.

Your task:
1. Identify all incorrect options (distractors) provided in the question.
2. For EACH incorrect option, write a plausible reasoning trace that leads to that specific wrong answer, similar to the correct reasoning one.
3. The reasoning should sound logical (hallucinating facts if necessary) but must conclude with the wrong answer.
4. The reasoning style should match the format of the correct reasoning

Format your output exactly as sequential blocks:

<block>
<think>
[Reasoning for Distractor]
</think>
<answer>
[Distractor Label and Text, e.g., B. Incorrect Choice]
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
        {"role": "system", "content": "You are a science expert generating adversarial wrong-reasoning examples."},
        {"role": "user", "content": prompt},
    ]

    try:
        response = client.chat.completions.create(
            model=DEPLOYMENT,
            messages=messages,
            n=1,
            temperature=1.0, 
        )

        chat_output = response.choices[0].message.content.strip()
        pattern = re.compile(r"<think>\s*(.*?)\s*</think>.*?<answer>\s*(.*?)\s*</answer>", re.DOTALL)
        matches = pattern.findall(chat_output)

        corrupted_reasonings = []
        corrupted_answers = []

        for match in matches:
            corrupted_reasonings.append(match[0].strip())
            corrupted_answers.append(match[1].strip())

        row['corrupted_reasonings'] = corrupted_reasonings
        row['corrupted_answers'] = corrupted_answers
        # ScienceQA can have 2 to 5 options; we check if we got at least one corruption
        row['is_corrupted'] = len(corrupted_reasonings) > 0 

    except Exception as e:
        print(f"API Error: {e}")
        row['corrupted_reasonings'] = []
        row['corrupted_answers'] = []
        row['is_corrupted'] = False

    return row


def build_science_qa_split(max_len, max_samples=None, workers=10, split="train", corrupt=True) -> Dataset:
    """
    Builds and corrupts a specific predefined split (train, validation, or test).
    """
    # 1. Load the specific predefined split 
    try:
        ds = load_dataset("derek-thomas/ScienceQA")[split]
    except KeyError:
        # Handle cases where source naming might differ (e.g., 'val' vs 'validation')
        mapping = {"eval": "validation", "validation": "validation", "test": "test", "train": "train"}
        ds = load_dataset("derek-thomas/ScienceQA")[mapping[split]]
        
    ds = ds.filter(lambda x: x["image"] is None)
    
    if max_samples:
        ds = ds.select(range(min(len(ds), max_samples)))

    def pre_munge(example):
        choices = example.get("choices", [])
        
        # Format Question with labels A. B. C. [cite: 1098]
        formatted_question = example.get("question", "") + "\n\nAnswer Choices:\n" + \
                             "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices))

        ans_idx = example.get("answer")
        if ans_idx is not None and 0 <= ans_idx < len(choices):
            formatted_answer = f"{chr(65+ans_idx)}. {choices[ans_idx]}"
        else:
            formatted_answer = None

        # Expert reasoning trace: lecture + solution [cite: 67, 381]
        lecture = example.get("lecture", "") or ""
        solution = example.get("solution", "") or ""
        reasoning = (lecture + "\n\n" + solution).strip()
        
        # Word count for length filtering [cite: 250, 303]
        sum_words = len(formatted_question.split()) + len(reasoning.split())

        return {
            "question": formatted_question,
            "reasoning": reasoning,
            "answer": formatted_answer,
            "response": formatted_answer,
            "sum_words": sum_words
        }

    ds = ds.map(pre_munge, remove_columns=ds.column_names)
    ds = ds.filter(lambda x: x["sum_words"] < max_len and x["answer"] is not None)

    if not corrupt:
        print(f"ScienceQA {split} (Clean) size: {len(ds)}")
        # For non-corrupted splits, initialize empty adversarial fields for schema consistency
        def add_empty_fields(example):
            return {**example, "corrupted_reasonings": [], "corrupted_answers": [], "is_corrupted": False}
        return ds.map(add_empty_fields)

    print(f"ScienceQA {split} (Corrupting) size: {len(ds)}")
    data_list = [item for item in ds]
    processed_list = process_batch_with_threadpool(data_list, max_workers=workers)
    
    final_ds = Dataset.from_list(processed_list)
    return final_ds # Do not filter by is_corrupted if you want to keep failed API rows, or keep it to ensure quality


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
    parser.add_argument("--outdir", type=str, default="data/scienceqa")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit rows per split for testing")
    parser.add_argument("--workers", type=int, default=10, help="Thread pool workers")
    args = parser.parse_args()

    # 1. Process each predefined split independently 
    # We map 'eval' to 'validation' to match ScienceQA naming conventions
    splits_to_process = ["train", "validation", "test"]
    ds_dict = {}

    for split in splits_to_process:
        # Only trigger adversarial corruption for the training split
        should_corrupt = (split == "train")
        
        processed_split = build_science_qa_split(
            max_len=1024, 
            max_samples=args.max_samples, 
            workers=args.workers, 
            split=split,
            corrupt=should_corrupt
        )
        ds_dict[split] = processed_split

    # 2. Package as a DatasetDict
    dsd = DatasetDict(ds_dict)
    
    # 3. Save HF Dataset
    dsd.save_to_disk(args.outdir)
    print(f"Saved predefined HF dataset splits to {args.outdir}")
    
    # 4. Save CSV/JSONL for inspection
    export_readable_files(dsd, args.outdir)
    
    # 5. Print Stats
    print_comprehensive_stats(dsd)

if __name__ == "__main__":
    main()