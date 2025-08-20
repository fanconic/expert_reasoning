import re
import random

def flip_operator_in_one_step(text: str) -> str:
    """
    Corrupt exactly ONE arithmetic line inside <think> by flipping an operator
    (x -> +, + -> -, - -> +) while leaving numbers and RHS unchanged.
    This makes the step false but preserves formatting and tags.
    """
    m = re.search(r"<think>\s*(.*?)\s*</think>", text, flags=re.DOTALL)
    if not m:
        return text
    think = m.group(1)
    lines = think.splitlines()

    # find candidate lines with an operator
    idxs = [i for i, ln in enumerate(lines) if (" x " in ln) or (" + " in ln) or (" - " in ln)]
    if not idxs:
        return text

    i = random.choice(idxs)
    ln = lines[i]
    if " x " in ln:
        ln = ln.replace(" x ", " + ", 1)
    elif " + " in ln:
        ln = ln.replace(" + ", " - ", 1)
    elif " - " in ln:
        ln = ln.replace(" - ", " + ", 1)
    lines[i] = ln

    new_think = "\n".join(lines)
    return text[:m.start(1)] + new_think + text[m.end(1):]

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
    (x -> +, + -> -, - -> +) while leaving numbers and RHS unchanged.
    This makes the step false but preserves formatting and tags.
    """
    m = re.search(r"<think>\s*(.*?)\s*</think>", text, flags=re.DOTALL)
    if not m:
        return text
    think = m.group(1)
    lines = think.splitlines()

    # find candidate lines with an operator
    idxs = [i for i, ln in enumerate(lines) if (" x " in ln) or (" + " in ln) or (" - " in ln)]
    if not idxs:
        return text

    i = random.choice(idxs)
    ln = lines[i]
    if " x " in ln:
        ln = ln.replace(" x ", " + ", 1)
    elif " + " in ln:
        ln = ln.replace(" + ", " - ", 1)
    elif " - " in ln:
        ln = ln.replace(" - ", " + ", 1)
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
                return str(val + random.choice([-2, -1, 1, 2]))
        except Exception:
            return num
    return re.sub(r"\b\d+(\.\d+)?\b", repl, text, count=max(1, len(re.findall(r'\d', text)) // 4))


PERTURB_FN_MAP = {
    "flip_operator_in_one_step": flip_operator_in_one_step,
    "corrupt_answer_with_think_number": corrupt_answer_with_think_number,
    "corrupt_answer_nearby_number": corrupt_answer_nearby_number,
    "corrupt_numbers": corrupt_numbers
}