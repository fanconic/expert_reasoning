from datasets import load_dataset, Dataset
import re

SYSTEM_PROMPT = """
Respond in the following format:
<think>
...
</think>
<answer>
...
</answer>
"""


def extract_hash_answer(text: str) -> str:
    """
    Extract the answer from text that uses '####' as a delimiter.

    Args:
        text (str): The text containing an answer after '####' delimiter.

    Returns:
        str: The extracted answer text after '####', stripped of whitespace.
             If '####' is not present, returns the original text stripped.
    """
    if "####" not in text:
        return text.strip()
    return text.split("####")[1].strip()


def extract_boxed_integer(input_string):
    """
    Extracts the integer from inside a \(\boxed{}\) notation using regex.

    Args:
        input_string (str): The input string containing a boxed integer.

    Returns:
        str: The extracted integer as a string.
    """
    match = re.search(r"\\boxed{(\d+)}", input_string)
    return match.group(1) if match else "No boxed value found"


def extract_think_content(input_string):
    """
    Extracts the content between <think> and </think> tags using regex.

    Args:
        input_string (str): The input string containing <think> tags.

    Returns:
        str: The extracted content as a string.
    """
    match = re.search(r"<think>(.*?)</think>", input_string, re.DOTALL)
    return match.group(1).strip() if match else "No think content found"


def get_gsm8k_questions(split="train", ratio: float = 1.0):
    """
    Load and preprocess the GSM8K dataset.

    Args:
        split (str): Dataset split to load ('train', 'test', etc.).

    Returns:
        Dataset: Processed dataset with prompts formatted for model input
                and extracted answers.
    """
    data = load_dataset("openai/gsm8k", "main")[split]
    data = data.select(range(int(len(data) * ratio)))
    data = data.map(
        lambda x: {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": x["question"]},
            ],
            "answer": extract_hash_answer(x["answer"]),
        }
    )
    return data


def get_curated_thoughts(split="train", ratio: float = 1.0):
    """
    Loads and processes the onnookk/format_vs_content_reasoning_clean_gsm8k dataset for knowledge distillation.
    Assumes the dataset contains keys: 'question', 'reasoning', 'answer'.
    Adjust the processing if the field names differ.
    """
    data = load_dataset("onnookk/format_vs_content_reasoning_clean_gsm8k", split=split)
    data = data.select(range(int(len(data) * ratio)))
    data = data.map(
        lambda x: {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": x["question"]},
                {
                    "role": "assistant",
                    "content": "<think>\n"
                    + extract_think_content(x["answer"])
                    + "\n</think>\n"
                    + "<answer>\n"
                    + extract_boxed_integer(x["answer"])
                    + "\n</answer>",
                },
            ]
        }
    )
    return data


def get_dataset(name: str, split: str = "train", ratio: float = 1.0):
    """
    Load a dataset by name and split.

    Args:
        name (str): Name of the dataset to load.
        split (str): Dataset split to load (default: 'train').

    Returns:
        Dataset: The requested dataset processed for model training.

    Raises:
        NotImplementedError: If the requested dataset is not yet implemented.
        ValueError: If the dataset name is not supported.
    """
    if name.lower() == "gsm8k":
        return get_gsm8k_questions(split, ratio)
    elif name.lower() == "countdown":
        raise NotImplementedError("Countdown dataset not implemented")
    elif name.lower() == "curatedthoughts":
        return get_curated_thoughts(split, ratio)
    else:
        raise ValueError(f"Dataset {name} not supported")
