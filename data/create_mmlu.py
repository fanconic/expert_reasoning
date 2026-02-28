import argparse
import re
import concurrent.futures
from typing import Any, List, Dict
import os
import textwrap
import pandas as pd
import json

from datasets import load_dataset, Dataset, DatasetDict
from tqdm import tqdm

from openai import AzureOpenAI


def generate_gold_reasoning(question: str, correct_answer: str) -> str:
    """Generates a high-quality Chain of Thought for the correct answer."""
    prompt = f"""
You are an expert scientist and educator.
Below is a science question and the verified correct answer.

Your task:
Write a logical, step-by-step reasoning trace (Chain of Thought) explaining why the answer is correct.
- Provide clear scientific justification.
- Match the style of a high-level textbook explanation.
- Conclude by explicitly stating the correct answer.

---
Question: 
{question}

Correct Answer: 
{correct_answer}

Reasoning Trace:
"""
    try:
        response = client.chat.completions.create(
            model=DEPLOYMENT,
            messages=[
                {"role": "system", "content": "You are a scientific expert providing rigorous reasoning."},
                {"role": "user", "content": prompt}
            ],
            temperature=1.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating gold reasoning: {e}")
        return ""

def corrupt_entry_all_distractors(row: Dict[str, Any]) -> Dict[str, Any]:
    # Ensure we have a Gold Reasoning first
    if not row.get("reasoning"):
        row["reasoning"] = generate_gold_reasoning(row["question"], row["answer"])
    
    if not row["reasoning"]:
        row['is_corrupted'] = False
        return row

    question = row["question"]
    correct_reasoning = row["reasoning"]
    correct_answer = row["answer"]
    
    prompt = f"""
You are an expert educator creating adversarial training examples.
Below is a question, the correct reasoning, and the correct answer.

Your task:
1. Identify the incorrect options (distractors) provided in the question.
2. For EACH incorrect option, write a plausible reasoning trace that leads to that specific wrong answer.
3. The reasoning should sound logical but must conclude with the wrong answer.
4. Match the formatting of the correct reasoning provided.

Format your output as sequential blocks:

<block>
<think>
[Reasoning for Distractor]
</think>
<answer>
[Distractor Label and Text]
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
    try:
        response = client.chat.completions.create(
            model=DEPLOYMENT,
            messages=[
                {"role": "system", "content": "You are a medical expert generating adversarial wrong-reasoning examples."},
                {"role": "user", "content": prompt},
            ],
            temperature=1.0, 
        )

        chat_output = response.choices[0].message.content.strip()
        pattern = re.compile(r"<think>\s*(.*?)\s*</think>.*?<answer>\s*(.*?)\s*</answer>", re.DOTALL)
        matches = pattern.findall(chat_output)

        corrupted_reasonings = [m[0].strip() for m in matches]
        corrupted_answers = [m[1].strip() for m in matches]

        row['corrupted_reasonings'] = corrupted_reasonings
        row['corrupted_answers'] = corrupted_answers
        row['is_corrupted'] = len(corrupted_reasonings) >= 1 
    except Exception as e:
        print(f"API Error in Corruption: {e}")
        row['is_corrupted'] = False

    return row

def process_batch_with_threadpool(dataset_list: List[Dict], max_workers: int = 10) -> List[Dict]:
    results = []
    pbar = tqdm(total=len(dataset_list), desc="Processing Pipeline")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_row = {executor.submit(corrupt_entry_all_distractors, row): row for row in dataset_list}
        for future in concurrent.futures.as_completed(future_to_row):
            try:
                results.append(future.result())
            except Exception as exc:
                orig = future_to_row[future]
                orig['is_corrupted'] = False
                results.append(orig)
            pbar.update(1)
    pbar.close()
    return results

def label_to_index(label: str):
    if not label: return None
    m = re.match(r"\s*([A-Za-z])", str(label))
    return ord(m.group(1).upper()) - 65 if m else None

def build_dataframe(max_len, max_samples=None, workers=10) -> Dataset:
    ds = load_dataset("TIGER-Lab/MMLU-Pro")["test"]
    if max_samples:
        ds = ds.select(range(min(len(ds), max_samples)))

    def pre_munge(example):
        choices = example.get("options", [])
        formatted_question = example.get("question", "") + "\n\nAnswer Choices:\n" + \
                             "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices))
        
        ans_idx = label_to_index(example.get("answer"))
        formatted_answer = f"{chr(65+ans_idx)}. {choices[ans_idx]}" if ans_idx is not None else None
        
        return {
            "question": formatted_question,
            "reasoning": "", # Initially empty, filled by the pipeline
            "answer": formatted_answer,
            "category": example.get("category", "Unknown"),
        }

    ds = ds.map(pre_munge, remove_columns=ds.column_names)
    data_list = [item for item in ds]
    
    print(f"Starting pipeline with {workers} workers...")
    processed_list = process_batch_with_threadpool(data_list, max_workers=workers)
    
    final_ds = Dataset.from_list(processed_list)
    return final_ds.filter(lambda x: x['is_corrupted'] is True)

def make_splits(df: Dataset, test_size: int, val_size: int, seed: int) -> DatasetDict:
    # Ensure splits don't exceed data size
    test_size = min(test_size, len(df) // 4)
    val_size = min(val_size, len(df) // 4)
    
    first_split = df.train_test_split(test_size=test_size, seed=seed)
    second_split = first_split["train"].train_test_split(test_size=val_size, seed=seed)
    return DatasetDict({"train": second_split["train"], "eval": second_split["test"], "test": first_split["test"]})

def word_count(text):
    return len(str(text).split()) if text else 0

def print_comprehensive_stats(dsd: DatasetDict):
    print("\n" + "="*50 + "\nDATASET STATISTICS\n" + "="*50)
    combined_df = pd.concat([dsd[split].to_pandas() for split in dsd.keys()], ignore_index=True)
    for split in dsd.keys():
        print(f"{split.upper()}: {len(dsd[split])} examples")
    
    r_lens = combined_df['reasoning'].apply(word_count)
    print(f"\nGOLD REASONING: Avg {r_lens.mean():.1f} words")
    
    all_cr = combined_df['corrupted_reasonings'].explode()
    cr_lens = all_cr.apply(word_count)
    print(f"CORRUPTED REASONING: Avg {cr_lens.mean():.1f} words")
    print("="*50 + "\n")

def export_readable_files(dsd: DatasetDict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    for split, dataset in dsd.items():
        dataset.to_csv(os.path.join(output_dir, f"{split}.csv"), index=False)
        dataset.to_json(os.path.join(output_dir, f"{split}.jsonl"), orient="records", lines=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="data/mmlu_pro")
    parser.add_argument("--test_size", type=int, default=1500)
    parser.add_argument("--val_size", type=int, default=1500)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    df = build_dataframe(max_len=1024, max_samples=args.max_samples, workers=args.workers)
    
    if len(df) < (args.test_size + args.val_size + 1):
        print("Dataset too small for requested splits. Saving all to train.")
        dsd = DatasetDict({"train": df})
    else:
        dsd = make_splits(df, args.test_size, args.val_size, seed=42)
    
    dsd.save_to_disk(args.outdir)
    export_readable_files(dsd, args.outdir)
    print_comprehensive_stats(dsd)

if __name__ == "__main__":
    main()