import re
from typing import List, Optional

### GSM8K REWARD FUNCTIONS


# compile once, with DOTALL so '.' matches newlines
STRICT_FMT = re.compile(
    r"^<think>\s*.*?\s*</think>\s*<answer>\s*.*?\s*</answer>\s*$", flags=re.DOTALL
)
SOFT_FMT = re.compile(r"<think>.*?</think>.*?<answer>.*?</answer>", flags=re.DOTALL)


def strict_format_reward_func(completions, **kwargs):
    """
    Calculate reward based on strict adherence to the expected XML format.

    The expected format is:
    <think>
    ...
    </think>
    <answer>
    ...
    </answer>

    Args:
        completions: List of model completions, each containing response content.
        **kwargs: Additional keyword arguments.

    Returns:
        list: A list of rewards (0.5 for correctly formatted responses, 0.0 otherwise).
    """
    responses = [c[0]["content"].strip() for c in completions]
    return [0.5 if STRICT_FMT.match(r) else 0.0 for r in responses]


def soft_format_reward_func(completions, **kwargs):
    """
    Calculate reward based on a more lenient check of XML format.

    Checks if the response contains <think> and <answer> tags in any format.

    Args:
        completions: List of model completions, each containing response content.
        **kwargs: Additional keyword arguments.

    Returns:
        list: A list of rewards (0.5 for responses with both tags, 0.0 otherwise).
    """
    responses = [c[0]["content"] for c in completions]
    return [0.5 if SOFT_FMT.search(r) else 0.0 for r in responses]


def extract_xml_answer(text: str) -> str:
    """
    Extract the answer from XML-formatted text.

    Args:
        text (str): The text containing XML tags with an answer.

    Returns:
        str: The extracted answer text between <answer> tags, stripped of whitespace.
    """
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def correctness_reward_func(prompts, completions, answer, **kwargs):
    """
    Calculate reward based on whether the extracted answer matches the ground truth.

    Args:
        prompts: The input prompts (not used in this function).
        completions: List of model completions, each containing response content.
        answer: The ground truth answer to compare against.
        **kwargs: Additional keyword arguments.

    Returns:
        list: A list of rewards (2.0 for correct answers, 0.0 for incorrect ones).
    """
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]


def int_reward_func(completions, **kwargs):
    """
    Calculate reward based on whether the extracted answer is a digit.

    Args:
        completions: List of model completions, each containing response content.
        **kwargs: Additional keyword arguments.

    Returns:
        list: A list of rewards (0.5 for digit answers, 0.0 otherwise).
    """
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]


def count_xml(text) -> float:
    """
    Calculate a score based on the presence and formatting of XML tags.

    Awards partial points for each correctly formatted tag and penalizes
    extra content after the closing </answer> tag.

    Args:
        text (str): The text to analyze for XML formatting.

    Returns:
        float: A score between 0.0 and 0.5 based on XML formatting quality.
    """
    count = 0.0
    if text.count("<think>\n") == 1:
        count += 0.125
    if text.count("\n</think>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count


def xmlcount_reward_func(completions, **kwargs):
    """
    Calculate rewards based on XML tag formatting quality.

    Uses the count_xml helper function to score each completion.

    Args:
        completions: List of model completions, each containing response content.
        **kwargs: Additional keyword arguments.

    Returns:
        list: A list of rewards based on XML formatting quality.
    """
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]


def get_reward_functions(dataset_name: str) -> List:
    """
    Return a list of all reward functions to be used during training.

    Returns:
        list: A list of reward functions in the order they should be applied.
    """
    if dataset_name == "gsm8k" or dataset_name == "gsm8k_kd":
        return [
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            int_reward_func,
            correctness_reward_func,
        ]
    elif dataset_name == "countdown" or dataset_name == "countdown_kd":
        return [
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            answer_reward_function,
        ]
    elif dataset_name == "medical" or dataset_name == "medical_kd":
        return [
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func
        ]
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")


def eval_correctness(completions, answer):
    """
    Calculate reward based on whether the extracted answer matches the ground truth for the EVALUATION of pass@n

    Args:
        completions: List of model completions, each containing response content.
        answer: The ground truth answer to compare against.

    Returns:
        list: A list of rewards (1.0 for correct answers, 0.0 for incorrect ones).
    """
    responses = [completion["content"] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [r == answer for r in extracted_responses]


### COUNTDOWN REWARD FUNCTIONS
def answer_reward_function(
    response: str, numbers: List[int] = None, target: int = None
) -> float:
    """
    Checks if the answer uses all numbers exactly once and evaluates to the target
    """
    answer_regex = r"<answer>(.*?)<\/answer>"
    answer_match = re.search(answer_regex, response, re.DOTALL)
    if not answer_match:
        return 0.0

    answer_content = answer_match.group(1)
    if not answer_content:
        return 0.0

    allowed_chars = r"^[0-9+\-*/() ]+$"
    if not re.match(allowed_chars, answer_content):
        return 0.0

    # Check if the answer uses all numbers exactly once
    used_numbers = [int(n) for n in re.findall(r"\d+", answer_content)]
    if sorted(used_numbers) != sorted(numbers):
        return 0.0

    # Check if the answer evaluates to the target
    try:
        result = eval(answer_content, {"__builtins__": None}, {})
        if abs(float(result) - float(target)) < 1e-5:
            return 2.0
    except:
        pass

    return 0.0