import re
import random
import logging
from src.azure.azure_connection import client, DEPLOYMENT

# Suppress noisy INFO-level HTTP logs from Azure/httpx/urllib3 so lines like
# "INFO HTTP 200 OK" are not printed to stdout.
for _lg in ("azure.core.pipeline.policies.http_logging_policy", "azure", "httpx", "urllib3"):
    logging.getLogger(_lg).setLevel(logging.WARNING)
    
_ANSWER_RE = re.compile(r"(<answer>\s*)(.*?)(\s*</answer>)", flags=re.DOTALL)
_THINK_RE  = re.compile(r"<think>\s*(.*?)\s*</think>", flags=re.DOTALL)

def _replace_answer_block(text: str, new_answer: str) -> str:
    m = _ANSWER_RE.search(text)
    if not m:
        return text
    return text[:m.start(2)] + new_answer + text[m.end(2):]

def _extract_answer(text: str) -> str | None:
    m = _ANSWER_RE.search(text)
    return None if not m else m.group(2).strip()

def _extract_think_numbers(text: str) -> list[str]:
    m = _THINK_RE.search(text)
    if not m:
        return []
    think = m.group(1)
    return re.findall(r"\b[+-]?\d+(?:\.\d+)?\b", think)

def flip_operator_in_one_step(text: str) -> str:
    """
    Corrupt exactly ONE arithmetic line inside <think> by flipping an operator
    to a random different operator (*, /, +, -) while leaving numbers and RHS unchanged.
    This makes the step false but preserves formatting and tags.
    """
    m = re.search(r"<think>\s*(.*?)\s*</think>", text, flags=re.DOTALL)
    if not m:
        return text
    think = m.group(1)
    lines = think.splitlines()

    # find candidate lines with an operator
    idxs = [i for i, ln in enumerate(lines) if ("*" in ln) or ("/" in ln) or ("+" in ln) or ("-" in ln)]
    if not idxs:
        return text

    i = random.choice(idxs)
    ln = lines[i]
    
    # Find the first operator in the line
    operators = ['*', '/', '+', '-']
    for op in operators:
        if op in ln:
            # Get remaining operators to choose from
            remaining_ops = [o for o in operators if o != op]
            new_op = random.choice(remaining_ops)
            ln = ln.replace(op, new_op, 1)
            break
    
    lines[i] = ln

    new_think = "\n".join(lines)
    return text[:m.start(1)] + new_think + text[m.end(1):]


def corrupt_answer_nearby_number(text: str) -> str:
    """
    If the answer is numeric, nudge it by the smallest unit that preserves its
    formatting (int: ±1; float: ±10^-decimals). If boolean, flip. Otherwise,
    make a tiny, single-typo corruption. Tags & layout preserved.
    """
    orig = _extract_answer(text)
    if orig is None:
        return text

    s = orig.strip()
    # numeric?
    if re.fullmatch(r"[+-]?\d+", s):
        v = int(s)
        delta = random.choice([-1, 1])
        new = str(v + delta)
        if new == s:
            new = str(v + 2 * delta)
        return _replace_answer_block(text, new)

    if re.fullmatch(r"[+-]?\d+\.\d+", s):
        decimals = len(s.split(".")[1])
        step = 10 ** (-decimals)
        sign = random.choice([-1, 1])
        v = float(s)
        new_v = v + sign * step
        new = f"{new_v:.{decimals}f}"
        if new == s:
            new_v = v + 2 * sign * step
            new = f"{new_v:.{decimals}f}"
        return _replace_answer_block(text, new)

    # booleans
    low = s.lower()
    if low in {"yes", "true"}:
        return _replace_answer_block(text, "no")
    if low in {"no", "false"}:
        return _replace_answer_block(text, "yes")

    # fallback: minimal character corruption (swap two adjacent chars)
    if len(s) >= 2:
        i = random.randrange(0, len(s) - 1)
        corrupted = s[:i] + s[i+1] + s[i] + s[i+2:]
        return _replace_answer_block(text, corrupted)
    # last resort: append a subtle mark
    return _replace_answer_block(text, s + ".")


def corrupt_answer_with_think_number(text: str) -> str:
    """
    Replace the answer with a *different* number that appears in <think>.
    If none exist or all equal the current answer, fall back to nearby-number corruption.
    Keeps tags & spacing intact.
    """
    orig = _extract_answer(text)
    if orig is None:
        return text

    numbers = _extract_think_numbers(text)
    if numbers:
        candidates = [n for n in numbers if n.strip() != orig.strip()]
        if candidates:
            new = random.choice(candidates)
            return _replace_answer_block(text, new)

    # fallback
    return corrupt_answer_nearby_number(text)



def corrupt_numbers(text: str) -> str:
    def repl(m):
        num = m.group(0)
        try:
            if "." in num:
                val = float(num)
                return f"{val + random.choice([-1.0, 1.0])*random.uniform(0.5, 2.0):.3f}"
            else:
                val = int(num)
                return str(val + random.choice([-10, -5, -2, -1, 1, 2, 5, 10]))
        except Exception:
            return num
    return re.sub(r"\b\d+(\.\d+)?\b", repl, text, count=max(1, len(re.findall(r'\d', text)) // 4))


def flip_clinical_direction(promtp:str, text: str) -> str:
    """
    Inverts common medical directional terms and prefixes.
    e.g., "Hypertension" -> "Hypotension", "increase" -> "decrease".
    Attacks the physiological reasoning directly.
    """
    # A robust map of medical antonyms common in MedQA/MedMCQA
    pairs = [
        ("increase", "decrease"), ("increases", "decreases"), ("increased", "decreased"),
        ("elevated", "reduced"), ("elevation", "depression"),
        ("high", "low"), ("higher", "lower"),
        ("positive", "negative"), ("positivity", "negativity"),
        ("hyper", "hypo"), ("Hyper", "Hypo"),
        ("acute", "chronic"), ("Acute", "Chronic"),
        ("agonist", "antagonist"), ("stimulation", "inhibition"),
        ("indicated", "contraindicated"),
        ("proximal", "distal"), ("lateral", "medial"),
        ("sympathetic", "parasympathetic"),
        ("benign", "malignant")
    ]
    
    # Flatten map for bidirectional lookup
    swap_map = {}
    for a, b in pairs:
        swap_map[a] = b
        swap_map[b] = a

    m = re.search(r"<reasoning>\s*(.*?)\s*</reasoning>", text, flags=re.DOTALL)
    if not m: return text
    
    reasoning = m.group(1)
    
    # Identify all swappable tokens
    # Using specific regex to capture "hyper" inside words like "hypertension" is tricky,
    # so we focus on whole words or specific known prefixes if distinct.
    tokens = re.findall(r'\b[A-Za-z]+\b', reasoning)
    
    candidates = []
    for i, token in enumerate(tokens):
        # Check whole word matches
        if token in swap_map:
            candidates.append((token, swap_map[token]))
            continue
        
        # Check prefix matches (specifically for hyper/hypo)
        lower_token = token.lower()
        if lower_token.startswith("hyper") and len(token) > 5:
            new_word = "hypo" + token[5:]
            # Preserve capitalization
            if token[0].isupper(): new_word = new_word.capitalize()
            candidates.append((token, new_word))
        elif lower_token.startswith("hypo") and len(token) > 5:
            new_word = "hyper" + token[4:]
            if token[0].isupper(): new_word = new_word.capitalize()
            candidates.append((token, new_word))

    if not candidates:
        return text

    # Select one valid candidate to flip (perturbing too many destroys coherence)
    target, replacement = random.choice(candidates)
    
    # Replace only the first occurrence of this specific instance to simulate a subtle reasoning error
    new_reasoning = reasoning.replace(target, replacement, 1)
    return text[:m.start(1)] + new_reasoning + text[m.end(1):]


def reverse_causal_chains(promtp:str, text: str) -> str:
    """
    Finds lines with arrow indicators (->, implies, leads to) and swaps 
    the cause and effect.
    e.g., "Diabetes -> Retinopathy" becomes "Retinopathy -> Diabetes"
    """
    m = re.search(r"<reasoning>\s*(.*?)\s*</reasoning>", text, flags=re.DOTALL)
    if not m: return text
    
    lines = m.group(1).splitlines()
    new_lines = []
    
    arrow_patterns = [
        (r"(.*?) -> (.*)", " -> "),
        (r"(.*?) leads to (.*)", " leads to "),
        (r"(.*?) causes (.*)", " causes ")
    ]
    
    for line in lines:
        perturbed = False
        for pattern, separator in arrow_patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match and not perturbed:
                # 50% chance to flip this line if found
                if random.random() < 0.5:
                    cause, effect = match.group(1), match.group(2)
                    # Clean up punctuation slightly
                    effect = effect.strip().rstrip('.')
                    
                    # Construct reversed line
                    new_line = f"{effect}{separator}{cause}."
                    new_lines.append(new_line)
                    perturbed = True
        
        if not perturbed:
            new_lines.append(line)

    new_reasoning = "\n".join(new_lines)
    return text[:m.start(1)] + new_reasoning + text[m.end(1):]


def corrupt_demographics(promtp:str, text: str) -> str:
    """
    Swaps demographic keywords in the reasoning trace.
    e.g., Swaps "child/pediatric" with "adult/elderly".
    This makes the reasoning apply to the wrong patient group.
    """
    m = re.search(r"<reasoning>\s*(.*?)\s*</reasoning>", text, flags=re.DOTALL)
    if not m: return text
    
    reasoning = m.group(1)
    
    demo_map = {
        "child": "adult", "pediatric": "geriatric",
        "adult": "child", "geriatric": "pediatric",
        "elderly": "young", "young": "elderly",
        "woman": "man", "man": "woman",
        "female": "male", "male": "female",
        "boy": "girl", "girl": "boy"
    }
    
    # Find words to swap
    words = re.findall(r'\b\w+\b', reasoning)
    candidates = [w for w in words if w.lower() in demo_map]
    
    if not candidates:
        return text
        
    # Flip one demographic term
    target = random.choice(candidates)
    replacement = demo_map[target.lower()]
    
    # Match case
    if target[0].isupper():
        replacement = replacement.capitalize()
        
    # Replace all instances of this specific demographic term to make the 
    # hallucination consistent throughout the trace
    new_reasoning = re.sub(rf"\b{target}\b", replacement, reasoning)
    return text[:m.start(1)] + new_reasoning + text[m.end(1):]


def corrupt_guidelines_modalities(promtp:str, text: str) -> str:
    """
    Targets authoritative statements (should
    """
    m = re.search(r"<reasoning>\s*(.*?)\s*</reasoning>", text, flags=re.DOTALL)
    if not m: return text
    
    reasoning = m.group(1)
    
    # Regex to find sentences with modal verbs
    # We look for "should/must/advised" followed by "be" or a verb
    pattern = r"\b(should|must|is advised to|recommended to)\b\s+(\w+)"
    
    def replacer(match):
        modal = match.group(1)
        verb = match.group(2)
        
        # 1. Flip to negation
        if "not" not in modal and "avoid" not in verb:
            return f"{modal} NOT {verb}"
        
        # 2. If already negated (rare), remove negation
        if "not" in modal:
            return modal.replace(" not", "") + f" {verb}"
            
        return match.group(0)

    # Apply to a subset of matches to avoid over-corrupting
    # count=1 ensures we only break the logic once, making it harder to spot
    new_reasoning = re.sub(pattern, replacer, reasoning, count=1, flags=re.IGNORECASE)
    return text[:m.start(1)] + new_reasoning + text[m.end(1):]


def corrupt_with_chatgpt_wrong_reasoning(question: str, text: str) -> str:
    """
    Uses Azure OpenAI ChatGPT to generate a purposefully wrong reasoning trace
    (and optionally a corrupted answer) from the existing reasoning trace.
    The original prompt is included in context.
    """
    
    
    # Extract reasoning and answer
    reasoning_match = re.search(r"<think>\s*(.*?)\s*</think>", text, flags=re.DOTALL)
    answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.DOTALL)
    if not reasoning_match:
        return text
    reasoning = reasoning_match.group(1).strip()
    answer = answer_match.group(1).strip() if answer_match else None

    # Construct corruption instruction
    corruption_instruction = (
        "Given the following reasoning trace and answer, rewrite the reasoning trace so that it is purposefully incorrect, "
        "introducing plausible but wrong logic or facts. Optionally, also corrupt the answer to be wrong. "
        "Keep the output in the same format, with <think>...</think> and <answer>...</answer> tags.\n\n"
        f"Question: {question}\n\n"
        f"<think>\n{reasoning}\n</think>\n"
    )
    if answer:
        corruption_instruction += f"<answer>\n{answer}\n</answer>\n"

    # Compose messages with original prompt in context
    messages = [
        {"role": "system", "content": "You are a medical expert at generating subtle yet wrong reasoning traces for testing AI models."},
        {"role": "user", "content": corruption_instruction},
    ]
    
    try:
        response = client.chat.completions.create(
            model=DEPLOYMENT,
            messages=messages,
            n=1,
            reasoning_effort="minimal",
            max_completion_tokens=824
        )

        chat_output = response.choices[0].message.content.strip()

        # Extract new reasoning and answer from ChatGPT output
        new_reasoning_match = re.search(r"<think>\s*(.*?)\s*</think>", chat_output, flags=re.DOTALL)
        new_answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", chat_output, flags=re.DOTALL)
        new_reasoning = new_reasoning_match.group(1).strip() if new_reasoning_match else reasoning
        new_answer = new_answer_match.group(1).strip() if new_answer_match else (answer if answer else None)

        # Replace reasoning and answer in original text (use original spans to avoid index corruption)
        replacements = []
        replacements.append((reasoning_match.start(1), reasoning_match.end(1), new_reasoning))
        if answer_match and new_answer is not None:
            replacements.append((answer_match.start(1), answer_match.end(1), new_answer))

        # Perform replacements in a single pass from the original text
        replacements.sort(key=lambda x: x[0])  # sort by start index (should already be ordered)
        last = 0
        parts = []
        for s, e, rep in replacements:
            parts.append(text[last:s])
            parts.append(rep)
            last = e
        parts.append(text[last:])
        new_text = "".join(parts)
        return new_text

    except Exception as e:
        # On error, return original text
        print(f"ChatGPT API error: {e}")
        return text



PERTURB_FN_MAP = {
    "flip_operator_in_one_step": flip_operator_in_one_step,
    "corrupt_answer_with_think_number": corrupt_answer_with_think_number,
    "corrupt_answer_nearby_number": corrupt_answer_nearby_number,
    "corrupt_numbers": corrupt_numbers,
    "corrupt_guidelines_modalities": corrupt_guidelines_modalities,
    "flip_clinical_direction": flip_clinical_direction,
    "reverse_causal_chains": reverse_causal_chains,
    "corrupt_demographics": corrupt_demographics,
    "corrupt_with_chatgpt_wrong_reasoning": corrupt_with_chatgpt_wrong_reasoning
}