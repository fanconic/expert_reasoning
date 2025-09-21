from datasets import load_dataset, Dataset, load_from_disk
import re

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

# UTILS

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


def extract_boxed_integer(input_string: str) -> str:
    match = re.search(r"\\boxed{(\d+)}", input_string)
    return match.group(1) if match else input_string.strip()


def extract_think_content(input_string: str) -> str:
    match = re.search(r"<think>(.*?)</think>", input_string, re.DOTALL)
    return match.group(1).strip() if match else input_string.strip()

# GSM8K Dataset

def get_gsm8k_grpo(split="train", ratio: float = 1.0, no_system=False):
    """
    Load and preprocess the GSM8K dataset.

    Args:
        split (str): Dataset split to load ('train', 'test', etc.).

    Returns:
        Dataset: Processed dataset with prompts formatted for model input
                and extracted answers.
    """
    data = load_dataset("openai/gsm8k", "main")[split]
    # optionally subsample
    if ratio < 1.0:
        data = data.select(range(int(len(data) * ratio)))
        
    if no_system:
        data = data.map(
            lambda x: {
                "prompt": [
                    {"role": "user", "content": SYSTEM_PROMPT + "\n\n" + x["question"]},
                ],
                "answer": extract_hash_answer(x["answer"]),
            }
        )
    else:
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


def get_gsm8k_distillation(split: str = "train", ratio: float = 1.0,  no_system=False) -> Dataset:
    """
    Load GSM8K questions plus CuratedThoughts reasoning for KD:
      - Uses the 'onnookk/format_vs_content_reasoning_clean_gsm8k' dataset,
        which supplies both the question and a full chain-of-thought+boxed answer.
    Returns a Dataset with fields:
      - prompt: list[dict(role,content)]  (system + user)
      - target: str containing <think>…</think><answer>…</answer>
    """
    # this curated set has both the question and the full COT+boxed answer
    ds = load_dataset("onnookk/format_vs_content_reasoning_clean_gsm8k", split=split)
    # optionally subsample
    if ratio < 1.0:
        ds = ds.select(range(int(len(ds) * ratio)))

    def munge(example):
        reasoning = extract_think_content(example["answer"])
        answer = extract_boxed_integer(example["answer"])
        # build prompt + target
        if no_system:
            prompt = [
                {"role": "user", "content": SYSTEM_PROMPT + "\n\n" + example["question"]},
            ]
        else:
            prompt = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["question"]},
            ]
        target = (
            "<think>\n"
            f"{reasoning}\n"
            "</think>\n"
            "<answer>\n"
            f"{answer}\n"
            "</answer>"
        )
        return {"prompt": prompt, "target": target, "answer": answer}

    return ds.map(munge, remove_columns=ds.column_names)

# Countdown dataset

def get_countdown_grpo(split="train", ratio: float = 1.0):
    """
    Load and preprocess the countdown dataset.

    Args:
        split (str): Dataset split to load ('train', 'test', etc.).

    Returns:
        Dataset: Processed dataset with prompts formatted for model input
                and extracted answers.
    """
    data = load_from_disk("data/countdown")[split]
    # optionally subsample
    if ratio < 1.0:
        data = data.select(range(int(len(data) * ratio)))
    data = data.map(
        lambda x: {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": x["question"]},
            ],
            "answer": x["reward_model"],
        }
    )
    return data


def get_countdown_distillation(split: str = "train", ratio: float = 1.0) -> Dataset:
    """
    Load countdown questions plus CuratedThoughts reasoning for KD:
    Returns a Dataset with fields:
      - prompt: list[dict(role,content)]  (system + user)
      - target: str containing <think>…</think><answer>…</answer>
    """
    # this curated set has both the question and the full COT+boxed answer
    ds = load_from_disk("data/countdown")[split]
    # optionally subsample
    if ratio < 1.0:
        ds = ds.select(range(int(len(ds) * ratio)))

    def munge(example):
        reasoning = example["target"]
        answer = example["answer"]
        reward_model_answer = example["reward_model"]
        # build prompt + target
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["question"]},
        ]
        target = (
            "<think>\n"
            f"{reasoning}\n"
            "</think>\n"
            "<answer>\n"
            f"{answer}\n"
            "</answer>"
        )
        return {"prompt": prompt, "target": target, "answer": reward_model_answer}

    return ds.map(munge, remove_columns=ds.column_names)


# Medical Dataset
def get_medical_grpo(split="train", ratio: float = 1.0):
    """
    Load and preprocess the medical o1 dataset.

    Args:
        split (str): Dataset split to load ('train', 'test', etc.).

    Returns:
        Dataset: Processed dataset with prompts formatted for model input
                and extracted answers.
    """
    data = load_from_disk("/data/medical_o1")[split]
    # optionally subsample
    if ratio < 1.0:
        data = data.select(range(int(len(data) * ratio)))
    data = data.map(
        lambda x: {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": x["question"]},
            ],
            "answer": x["answer"],
        }
    )
    return data


def get_medical_distillation(split: str = "train", ratio: float = 1.0) -> Dataset:
    """
    Load o1 medical questions for KD:
    Returns a Dataset with fields:
      - prompt: list[dict(role,content)]  (system + user)
      - target: str containing <think>…</think><answer>…</answer>
    """
    # this curated set has both the question and the full COT+boxed answer
    ds = load_from_disk("/data/medical_o1")[split]
    # optionally subsample
    if ratio < 1.0:
        ds = ds.select(range(int(len(ds) * ratio)))

    def munge(example):
        reasoning = example["reasoning"]
        answer = example["answer"]
        # build prompt + target
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["question"]},
        ]
        target = (
            "<think>\n"
            f"{reasoning}\n"
            "</think>\n"
            "<answer>\n"
            f"{answer}\n"
            "</answer>"
        )
        return {"prompt": prompt, "target": target, "answer": answer}

    return ds.map(munge, remove_columns=ds.column_names)



def get_dataset(name: str, split: str = "train", ratio: float = 1.0, no_system=False):
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
        return get_gsm8k_grpo(split, ratio, no_system=no_system)
    elif name.lower() == "gsm8k_kd":
        return get_gsm8k_distillation(split, ratio, no_system=no_system)
    elif name.lower() == "countdown":
        return get_countdown_grpo(split, ratio, no_system=no_system)
    elif name.lower() == "countdown_kd":
        return get_countdown_distillation(split, ratio, no_system=no_system)
    elif name.lower() == "medical":
        return get_medical_grpo(split, ratio, no_system=no_system)
    elif name.lower() == "medical_kd":
        return get_medical_distillation(split, ratio, no_system=no_system)
    else:
        raise ValueError(f"Dataset {name} not supported")
