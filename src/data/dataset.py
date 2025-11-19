from datasets import load_dataset, Dataset, load_from_disk
import random
import re

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

SYSTEM_PROMPT_MEDREASON = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
    "The answer only includes the final answer, without any explanation, and is one of the options provided in the question, without the letter label, i.e."
    "Question ... Answer Options: \nA. answer1 \nB. answer2 \nC. answer3 \nD. \n<think> reasoning process here </think><answer> answer2 </answer>" 
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


def make_perturbed_completions(
    neg_perturb_fns,
    num_neg_perturbations_per_expert,
    expert_completions,
):
    """
    expert_completions: list of chat-format completions
        [[{"role": "assistant", "content": "..."}], ...]
    Returns:
        perturbed: list in the same chat format
        src_idx:  list[int], index of the source completion each perturbed one came from
    """
    if not neg_perturb_fns or num_neg_perturbations_per_expert == 0:
        return [], []

    fns = neg_perturb_fns if isinstance(neg_perturb_fns, list) else [neg_perturb_fns]
    perturbed, src_idx = [], []

    for i, comp in enumerate(expert_completions):
        base = comp[0]["content"]
        for _ in range(max(1, int(num_neg_perturbations_per_expert))):
            fn = random.choice(fns)
            try:
                corrupted = fn(base)
                # Fallback if fn returns something non-string or empty
                if not isinstance(corrupted, str) or corrupted == "":
                    corrupted = base
            except Exception:
                corrupted = base

            perturbed.append([{"role": "assistant", "content": corrupted}])
            src_idx.append(i)

    return perturbed, src_idx


def _extract_answer_from_target(target: str) -> str | None:
    """
    Given a target string of the form:
        <think> ... </think>
        <answer>
        ...
        </answer>
    extract the inner answer text.
    """
    m = re.search(
        r"<answer>\s*(.*?)\s*</answer>", target, flags=re.DOTALL | re.IGNORECASE
    )
    if not m:
        return None
    return m.group(1).strip()


def get_gsm8k_distillation(
    split: str = "train",
    ratio: float = 1.0,
    no_system: bool = False,
    expert_error_rate: float = 0.0,
    neg_perturb_fns=None,
    num_neg_perturbations_per_expert=0,
) -> Dataset:
    """
    Load GSM8K questions plus CuratedThoughts reasoning for KD:
      - Uses the 'onnookk/format_vs_content_reasoning_clean_gsm8k' dataset,
        which supplies both the question and a full chain-of-thought+boxed answer.

    Returns a Dataset with fields:
      - prompt: list[dict(role,content)]  (system + user)
      - target: str containing <think>…</think><answer>…</answer>
      - answer: final answer string (possibly corrupted if expert_error_rate > 0)
      - is_expert_error: bool indicating whether this example was perturbed
    """
    ds = load_dataset("onnookk/format_vs_content_reasoning_clean_gsm8k", split=split)

    # optionally subsample
    if ratio < 1.0:
        ds = ds.select(range(int(len(ds) * ratio)))

    if expert_error_rate > 0.0:
        print(f"Injecting expert errors at rate {expert_error_rate}")

    def munge(example):
        # Original (correct) expert reasoning and answer from the curated dataset
        reasoning = extract_think_content(example["answer"])
        true_answer = extract_boxed_integer(example["answer"])

        # build prompt
        if no_system:
            prompt = [
                {
                    "role": "user",
                    "content": SYSTEM_PROMPT + "\n\n" + example["question"],
                },
            ]
        else:
            prompt = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["question"]},
            ]

        # build canonical target string
        target = (
            "<think>\n"
            f"{reasoning}\n"
            "</think>\n"
            "<answer>\n"
            f"{true_answer}\n"
            "</answer>"
        )

        answer = true_answer
        is_expert_error = False

        # Maybe corrupt expert target + answer
        if (
            expert_error_rate > 0.0
            and neg_perturb_fns is not None
            and random.random() < expert_error_rate
        ):
            is_expert_error = True

            # Use make_perturbed_completions on a single completion
            expert_completion = [[{"role": "assistant", "content": target}]]
            perturbed, _ = make_perturbed_completions(
                neg_perturb_fns=neg_perturb_fns,
                num_neg_perturbations_per_expert=num_neg_perturbations_per_expert,
                expert_completions=expert_completion,
            )

            # unpack
            corrupted_target = perturbed[0][0]["content"]
            target = corrupted_target

            # try to re-extract the answer from the corrupted target
            corrupted_answer = _extract_answer_from_target(corrupted_target)
            if corrupted_answer is not None:
                answer = corrupted_answer
            else:
                # if we can't parse, fall back to original; mark as not-an-error to avoid surprises
                target = expert_completion[0][0]["content"]
                answer = true_answer
                is_expert_error = False

        return {
            "prompt": prompt,
            "target": target,
            "answer": answer,
            "is_expert_error": is_expert_error,
        }

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
def get_medical_grpo(
    split="train",
    ratio: float = 1.0,
    no_system: bool = False,
    expert_error_rate: float = 0.0,
    neg_perturb_fns=None,
    num_neg_perturbations_per_expert=0,
):
    """
    Load and preprocess the medical o1 dataset.

    Args:
        split (str): Dataset split to load ('train', 'test', etc.).

    Returns:
        Dataset: Processed dataset with prompts formatted for model input
                and extracted answers.
    """
    data = load_from_disk("./data/medreason")[split]
    # optionally subsample
    if ratio < 1.0:
        data = data.select(range(int(len(data) * ratio)))
    data = data.map(
        lambda x: {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT_MEDREASON},
                {"role": "user", "content": x["question"]},
            ],
            "answer": x["answer"],
        }
    )
    return data


def get_medical_distillation(
    split: str = "train",
    ratio: float = 1.0,
    no_system: bool = False,
    expert_error_rate: float = 0.0,
    neg_perturb_fns=None,
    num_neg_perturbations_per_expert=0,
) -> Dataset:
    """
    Load o1 medical questions for KD:
    Returns a Dataset with fields:
      - prompt: list[dict(role,content)]  (system + user)
      - target: str containing <think>…</think><answer>…</answer>
    """
    # this curated set has both the question and the full COT+boxed answer
    ds = load_from_disk("./data/medreason")[split]
    # optionally subsample
    if ratio < 1.0:
        ds = ds.select(range(int(len(ds) * ratio)))

    def munge(example):
        reasoning = example["reasoning"]
        answer = example["answer"]
        # build prompt + target
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT_MEDREASON},
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


def get_dataset(
    name: str,
    split: str = "train",
    ratio: float = 1.0,
    no_system=False,
    expert_error_rate: float = 0.0,
    neg_perturb_fns=None,
    num_neg_perturbations_per_expert: int = 0,
):
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
        return get_gsm8k_distillation(
            split,
            ratio,
            no_system=no_system,
            expert_error_rate=expert_error_rate,
            neg_perturb_fns=neg_perturb_fns,
            num_neg_perturbations_per_expert=num_neg_perturbations_per_expert,
        )
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
