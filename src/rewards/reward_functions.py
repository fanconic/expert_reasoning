import re
from typing import List, Optional
from difflib import SequenceMatcher

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
    if text.count("<think>") == 1:
        count += 0.125
    if text.count("</think>") == 1:
        count += 0.125
    if text.count("<answer>") == 1:
        count += 0.125
        count -= len(text.split("</answer>")[-1]) * 0.001
    if text.count("</answer>") == 1:
        count += 0.125
        count -= (len(text.split("</answer>")[-1]) - 1) * 0.001
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
        ], None
    elif dataset_name == "countdown" or dataset_name == "countdown_kd":
        return [
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            answer_reward_function,
        ], None
    elif dataset_name == "medical" or dataset_name == "medical_kd":
        return [
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func
        ], None
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")


def eval_correctness_gsm8k(completions, answer):
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


def eval_correctness_countdown(completions, answer):
    """
    Calculate reward based on whether the extracted answer matches the ground truth for the EVALUATION of pass@n

    Args:
        completions: List of model completions, each containing response content.
        answer: The ground truth answer to compare against.

    Returns:
        list: A list of rewards (1.0 for correct answers, 0.0 for incorrect ones).
    """
    responses = [completion["content"] for completion in completions]
    numbers = [answer["nums"]] * len(responses)
    targets = [answer["target"]] * len(responses)
    rewards = [answer_reward_function_single(r,a,t) for r, a, t in zip(responses, numbers, targets)]
    return [r == 2.0 for r in rewards]

def eval_correctness_medical_o1(completions, answer):
    """
    Calculate reward based on whether the extracted answer matches the ground truth for the EVALUATION of pass@n

    Args:
        completions: List of model completions, each containing response content.
        answer: The ground truth answer to compare against.

    Returns:
        list: A list of rewards (1.0 for correct answers, 0.0 for incorrect ones).
    """
    pass


### COUNTDOWN REWARD FUNCTIONS
def answer_reward_function_single(
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
    answer_content = answer_content.replace("\n","")
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

def answer_reward_function(
    prompts, completions, answer, **kwargs
) -> float:
    responses = [completion[0]["content"] for completion in completions]
    numbers = [a["nums"] for a in answer]
    targets = [a["target"] for a in answer]
    return [answer_reward_function_single(r,a,t) for r, a, t in zip(responses, numbers, targets)]


#### MEDICAL_O1 REWARD FUNCTION
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- Global, lazily-initialised verifier state ---
_VERIFIER_TOKENIZER = "FreedomIntelligence/medical_o1_verifier_3B_Qwen2.5"
_VERIFIER_MODEL = "FreedomIntelligence/medical_o1_verifier_3B_Qwen2.5"
_VERIFIER_TEMPLATE = """<Model Response>
{}
</Model Response>

<Reference Answer>
{}
</Reference Answer>

Your task is to evaluate the model response by comparing it to the reference answer. If the model response is correct and aligns with the reference answer, output "True" . If it is incorrect or fails to select the correct option (if options are provided), output "False" . {}"""

def _load_verifier(model_path: str = "FreedomIntelligence/medical_o1_verifier_3B_Qwen2.5"):
    """
    Lazily load the verifier model/tokenizer once.
    """
    global _VERIFIER_TOKENIZER, _VERIFIER_MODEL
    if _VERIFIER_TOKENIZER is not None and _VERIFIER_MODEL is not None:
        return _VERIFIER_TOKENIZER, _VERIFIER_MODEL

    _VERIFIER_TOKENIZER = AutoTokenizer.from_pretrained(model_path)
    # Try flash_attention_2 if available, fall back gracefully
    try:
        _VERIFIER_MODEL = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
            attn_implementation="flash_attention_2",
            num_labels=2,
        )
    except Exception:
        _VERIFIER_MODEL = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
            num_labels=2,
        )
    _VERIFIER_MODEL.eval()
    return _VERIFIER_TOKENIZER, _VERIFIER_MODEL


def _extract_model_response_text(raw_response: str, prefer_answer_tag: bool = True) -> str:
    """
    Optionally pull just the <answer>...</answer> span from the model response.
    Falls back to the full response if no tag is present.
    """
    if prefer_answer_tag:
        m = re.search(r"<answer>(.*?)</answer>", raw_response, flags=re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return raw_response.strip()


@torch.inference_mode()
def _verifier_batch_scores(
    responses: list[str],
    reference_texts: list[str],
    model_path: str = "FreedomIntelligence/medical_o1_verifier_3B_Qwen2.5",
    threshold: float = 0.5,
    return_probs: bool = False,
) -> list[float]:
    """
    Use the verifier in a single batched pass.
    Returns a list of 1.0 (True) / 0.0 (False) unless return_probs=True (then probabilities for 'True' class).
    """
    tokenizer, model = _load_verifier(model_path)
    eos = tokenizer.eos_token or ""

    # Build batched inputs
    prompts = [
        _VERIFIER_TEMPLATE.format(
            _extract_model_response_text(r),
            ref,
            eos
        )
        for r, ref in zip(responses, reference_texts)
    ]

    # Tokenize with padding/truncation
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=min(getattr(tokenizer, "model_max_length", 1024), 1024),  # be safe
    ).to(model.device)

    logits = model(**enc, return_dict=True).logits
    probs = F.softmax(logits, dim=-1)  # class 0 = "False", class 1 = "True" (by convention here)
    p_true = probs[:, 1].detach().float().cpu()

    if return_probs:
        return p_true.tolist()

    return (p_true > threshold).float().tolist()


def answer_reward_function_llm_verifier(
    prompts,
    completions,
    answer,
    *,
    model_path: str = "FreedomIntelligence/medical_o1_verifier_3B",
    threshold: float = 0.5,
    reward_true: float = 2.0,
    reward_false: float = 0.0,
    scale_by_probability: bool = False,
    prefer_answer_tag: bool = True,
    **kwargs,
) -> list[float]:
    """
    GRPO-compatible reward function using an LLM verifier.

    Args:
        prompts: (unused, present for API parity)
        completions: list of completion message lists; we take completions[i][0]["content"] as the model's text.
        answer: list of dicts or strings; see _coerce_reference_text for accepted formats.
        model_path: HF path to the verifier classifier.
        threshold: decision threshold on P(True).
        reward_true / reward_false: rewards to emit for pass/fail.
        scale_by_probability: if True, reward = reward_true * P(True) + reward_false * (1 - P(True)).
        prefer_answer_tag: if True, try to extract <answer>...</answer> from the model response before sending to verifier.

    Returns:
        List[float] of rewards.
    """
    # Extract raw response strings
    responses_raw = [c[0]["content"] if isinstance(c, list) and len(c) and "content" in c[0] else str(c) for c in completions]
    # Pre-trim to avoid massive prompts
    responses = [_extract_model_response_text(r, prefer_answer_tag=prefer_answer_tag) for r in responses_raw]

    # Coerce reference answers
    reference_texts = [_coerce_reference_text(a) for a in answer]

    # Get verifier judgements/probs
    if scale_by_probability:
        p_true = _verifier_batch_scores(
            responses, reference_texts, model_path=model_path, threshold=threshold, return_probs=True
        )
        return [reward_true * p + reward_false * (1.0 - p) for p in p_true]
    else:
        binary = _verifier_batch_scores(
            responses, reference_texts, model_path=model_path, threshold=threshold, return_probs=False
        )
        return [reward_true if b == 1.0 else reward_false for b in binary]
