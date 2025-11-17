"""
plot_helpers.py

Helper utilities for reading eval_result.jsonl files, computing metrics,
creating plots, and saving outputs. Extracted from the original notebook-style
script and made reusable.
"""

from __future__ import annotations
import scienceplots
import os
import math
import re
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from matplotlib.colors import TwoSlopeNorm, Normalize, LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch
from matplotlib.textpath import TextPath
import matplotlib as mpl

plt.style.use("bright")
plt.rcParams["font.family"] = "sans-serif"

import matplotlib.colors as mcolors

# Grab system default colours
c0 = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]  # usually blue
c1 = plt.rcParams["axes.prop_cycle"].by_key()["color"][1]  # usually orange/red

# Create a custom diverging cmap: negative = c1, positive = c0
CUSTOM_COLOR_MAP = mcolors.LinearSegmentedColormap.from_list(
    "custom_div", [c1, "white", c0]
)

# -------------------------------
# Parsing / reward helpers
# -------------------------------
STRICT_FMT = re.compile(
    r"^<think>\s*.*?\s*</think>\s*<answer>\s*.*?\s*</answer>\s*$", flags=re.DOTALL
)
SOFT_FMT = re.compile(r"<think>.*?</think>.*?<answer>.*?</answer>", flags=re.DOTALL)


def strict_format_reward_func(completions, **kwargs):
    responses = [c[0]["content"].strip() for c in completions]
    return [0.5 if STRICT_FMT.match(r) else 0.0 for r in responses]


def soft_format_reward_func(completions, **kwargs):
    responses = [c[0]["content"] for c in completions]
    return [0.5 if SOFT_FMT.search(r) else 0.0 for r in responses]


def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def correctness_reward_func(prompts, completions, answer, **kwargs):
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]


def int_reward_func(completions, **kwargs):
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]


def count_xml(text) -> float:
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


# -------------------------------
# Token visualisation utilities
# -------------------------------
_QWEN_SPECIAL_RE = re.compile(r"<\|[^>]*\|>")
_BYTE_BPE_SPACEISH = {"▁", "Ġ"}
_BYTE_BPE_NEWLINEISH = {"Ċ", "ċ", "ĉ", "č"}


def _restore_angle_brackets(s: str) -> str:
    s = re.sub(r"¿\s*(.*?)\s*¿", r"<\1>", s)
    s = re.sub(r"^¿\s*", "<", s)
    s = re.sub(r"\s*¿$", ">", s)
    s = s.replace("‹", "<").replace("›", ">")
    s = s.replace("¿", "")
    return s


def _is_allowed_char(ch: str) -> bool:
    cat = unicodedata.category(ch)
    if ch in ("\n", "\r", "\t"):
        return False
    if cat and cat[0] == "C":
        return False
    return ch.isprintable()


def clean_qwen_token(tok: str) -> str:
    if tok is None:
        return ""
    tok = unicodedata.normalize("NFKC", tok)
    tok = _QWEN_SPECIAL_RE.sub("", tok)
    for marker in _BYTE_BPE_SPACEISH | _BYTE_BPE_NEWLINEISH:
        tok = tok.replace(marker, " ")
    tok = (
        tok.replace("\ufffd", "")
        .replace("\u200b", "")
        .replace("\u200c", "")
        .replace("\u200d", "")
        .replace("\ufeff", "")
    )
    tok = _restore_angle_brackets(tok)
    tok = tok.replace("¿", "")
    tok = "".join(ch for ch in tok if _is_allowed_char(ch))
    tok = re.sub(r"(?<!\\)\$", r"\\$", tok)
    tok = re.sub(r"\s+", " ", tok).strip()
    return tok


def normalise(values, mode="minmax", max_val=None, min_val=None):
    v = np.array(values, dtype=float)
    if v.size == 0:
        return v.tolist()
    if mode == "minmax":
        vmin, vmax = float(v.min()), float(v.max())
        if np.isclose(vmin, vmax):
            return np.zeros_like(v).tolist()
        return ((v - vmin) / (vmax - vmin)).tolist()
    elif mode == "sigmoid":
        return ((1 / (1 + np.exp(-v))) * 2 - 1).tolist()
    elif mode == "diverging":
        # Map [-V, 0, +V] -> [0, 0.5, 1], so negatives are <0.5 (red), positives >0.5 (blue)
        if max_val is None:
            vmax = float(np.max(np.abs(v)))
        else:
            vmax = float(max(np.abs([max_val, min_val])))
        if np.isclose(vmax, 0.0):
            return (0.5 * np.ones_like(v)).tolist()  # all near zero → neutral white
        return (((v / vmax) + 1.0) / 2.0).tolist()
    else:
        raise ValueError("Unknown normalisation mode")


def text_size_px(text, font_size, dpi, font_properties=None):
    fp = font_properties or FontProperties(size=font_size, family="DejaVu Sans")
    tp = TextPath((0, 0), text, prop=fp)
    bb = tp.get_extents()
    w_in = bb.width / 72.0
    h_in = bb.height / 72.0
    return w_in * dpi, h_in * dpi


def _safe_text_size_px(text, font_size, dpi, font_properties):
    try:
        return text_size_px(text, font_size, dpi, font_properties)
    except Exception:
        t2 = text.encode("ascii", "ignore").decode("ascii") or "?"
        try:
            return text_size_px(t2, font_size, dpi, font_properties)
        except Exception:
            approx_w = max(1.0, 0.6 * len(t2) * font_size * dpi / 72.0)
            approx_h = font_size * dpi / 72.0
            return approx_w, approx_h


def _line_height_px(font_size, dpi, font_properties):
    _, h = _safe_text_size_px("Ag", font_size, dpi, font_properties)
    return h


def _wrap_text_to_width(text, font_size, dpi, font_properties, max_text_width_px):
    words = text.split()
    lines, cur = [], ""
    for w in words:
        test = (cur + " " + w).strip()
        w_px, _ = _safe_text_size_px(test, font_size, dpi, font_properties)
        if w_px <= max_text_width_px*0.95 or not cur:
            cur = test
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def make_text_reward_image(
    tokens,
    scores,
    out_path,
    prompt_text: Optional[str] = None,
    title: Optional[str] = None,
    cmap_name: str = "Blues",
    pad_x: int = 8,
    pad_y: int = 6,
    gap_x: int = 8,
    gap_y: int = 10,
    font_size: int = 16,
    max_width_px: int = 1400,
    dpi: int = 200,
    font_properties: Optional[FontProperties] = None,
    show_colorbar: bool = True,   # <--- NEW ARG
    max_val=None,
    min_val=None
):
    fp = font_properties or FontProperties(size=font_size, family="DejaVu Sans")
    assert len(tokens) == len(scores), "tokens and scores must have same length"

    # Store original min/max for colourbar scaling
    norm_scores = np.array(normalise(scores, mode="diverging", max_val=max_val, min_val=min_val))

    cleaned_tokens = [clean_qwen_token(t) for t in tokens]
    display_tokens = [t if t else " " for t in cleaned_tokens]

    widths, heights = [], []
    for tok in display_tokens:
        w, h = _safe_text_size_px(tok, font_size, dpi, fp)
        widths.append(w)
        heights.append(h)
    text_h = max(heights) if heights else font_size

    pill_h = text_h + 2 * pad_y
    row_height_px = pill_h + gap_y

    rows, cur_row, cur_w = [], [], 0
    for tok, sc, w_text in zip(display_tokens, norm_scores, widths):
        min_text_w = 0.6 * font_size * dpi / 72.0
        pill_w = max(w_text, min_text_w) + 2 * pad_x
        w_total = pill_w + gap_x
        if cur_row and cur_w + w_total > max_width_px*0.95:
            rows.append(cur_row)
            cur_row, cur_w = [], 0
        cur_row.append((tok, sc, pill_w))
        cur_w += w_total
    if cur_row:
        rows.append(cur_row)

    left_margin = 10
    right_margin = 10
    top_margin = 12
    between_title_and_question = 20
    between_question_and_ra = 50
    between_ra_and_pills = 120
    line_spacing = 1.15

    title_h = 0
    if title:
        title_h = _line_height_px(font_size + 2, dpi, fp)

    question_lines, q_lines_h = [], 0
    if prompt_text:
        q_text = clean_qwen_token(prompt_text)
        max_text_w = max_width_px - left_margin - right_margin
        question_lines = _wrap_text_to_width(q_text, font_size, dpi, fp, max_text_w)
        base_q_line_h = _line_height_px(font_size, dpi, fp)
        q_line_h = base_q_line_h * line_spacing
        q_lines_h = len(question_lines) * q_line_h

    label_h = _line_height_px(font_size, dpi, fp)
    ra_label_h = _line_height_px(font_size + 1, dpi, fp)

    header_block_h = 0
    if title:
        header_block_h += title_h + between_title_and_question
    if prompt_text:
        header_block_h += label_h + q_lines_h + between_question_and_ra
    header_block_h += ra_label_h + between_ra_and_pills

    width_in = max_width_px / dpi
    height_px = int(top_margin + header_block_h + len(rows) * row_height_px)
    height_in = max(1.0, height_px / dpi)

    fig = plt.figure(figsize=(width_in, height_in), dpi=dpi)
    ax = plt.axes([0, 0, 1, 1])
    ax.set_xlim(0, max_width_px)
    ax.set_ylim(0, height_px)
    ax.axis("off")

    cmap = CUSTOM_COLOR_MAP
    y = height_px - top_margin

    if title:
        ax.text(
            left_margin,
            y,
            title,
            fontsize=font_size + 2,
            fontstyle="italic",
            va="top",
            ha="left",
            fontproperties=fp,
        )
        y -= title_h + between_title_and_question

    if prompt_text:
        ax.text(
            left_margin,
            y,
            "Question:",
            fontsize=font_size,
            fontweight="bold",
            va="top",
            ha="left",
            fontproperties=fp,
        )
        y -= label_h + between_title_and_question
        if question_lines:
            base_q_line_h = _line_height_px(font_size, dpi, fp) * line_spacing
            for line in question_lines:
                ax.text(
                    left_margin,
                    y,
                    line,
                    fontsize=font_size,
                    va="top",
                    ha="left",
                    fontproperties=fp,
                )
                y -= base_q_line_h
        y -= between_question_and_ra

    ax.text(
        left_margin,
        y,
        "Reasoning + Answer:",
        fontsize=font_size,
        fontweight="bold",
        va="top",
        ha="left",
        fontproperties=fp,
    )
    y -= ra_label_h + between_ra_and_pills

    y_top = y + pill_h
    for row in rows:
        x = left_margin
        for tok, sc, pill_w in row:
            y_pill = y_top - pill_h
            if tok == "[PAD]":
                face = (0.92, 0.92, 0.92, 1.0)
                edge = (0.7, 0.7, 0.7, 1.0)
                txt_col = "black"
            else:
                face = cmap(sc)
                lum = 0.299 * face[0] + 0.587 * face[1] + 0.114 * face[2]
                txt_col = "black"
                edge = (0.75, 0.75, 0.75, 1.0)

            ax.add_patch(
                FancyBboxPatch(
                    (x, y_pill),
                    pill_w,
                    pill_h,
                    boxstyle="round,pad=0.0,rounding_size=8",
                    linewidth=1.0,
                    edgecolor=edge,
                    facecolor=face,
                )
            )

            ax.text(
                x + pill_w / 2,
                y_pill + pill_h / 2,
                tok,
                fontsize=font_size,
                va="center",
                ha="center",
                color=txt_col,
                fontproperties=fp,
            )
            x += pill_w + 8
        y_top -= row_height_px
        
    if show_colorbar:
        vmax = max(scores) if max_val is None else max_val
        vmin = min(scores) if min_val is None else min_val

        if vmin < 0 and vmax > 0:
            # Case 1: Diverging
            norm = TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax)
            cmap_used = cmap   # diverging cmap

        elif vmin >= 0:
            # Case 2: All positive → white → positive colour
            base_color = cmap(1.0)  # positive side
            cmap_used = LinearSegmentedColormap.from_list("seq_pos", ["white", base_color])
            norm = Normalize(vmin=vmin, vmax=vmax)

        else:
            # Case 3: All negative → white → negative colour (inverted)
            base_color = cmap(0.0)  # negative side
            cmap_used = LinearSegmentedColormap.from_list("seq_neg", [base_color, "white"])
            norm = Normalize(vmin=vmin, vmax=vmax)

        sm = mpl.cm.ScalarMappable(cmap=cmap_used, norm=norm)
        sm.set_array([])

        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Reward score", rotation=270, labelpad=15)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


# -------------------------------
# Metrics utilities
# -------------------------------


def _pass_at_k(num_correct: int, num_samples: int, k: int) -> float:
    if num_correct == 0 or k > num_samples:
        return 0.0
    return 1.0 - math.comb(num_samples - num_correct, k) / math.comb(num_samples, k)


def compute_pass_at_k(all_correct_flags: List[List[bool]], ks: Iterable[int]):
    totals = {k: 0.0 for k in ks}
    for flags in all_correct_flags:
        n = len(flags)
        m = sum(flags)
        for k in ks:
            totals[k] += _pass_at_k(m, n, k)
    num_problems = len(all_correct_flags)
    return {k: totals[k] / num_problems for k in ks}


def compute_success_at_k_from_scores(all_correct_flags, all_scores, ks):
    num_problems = len(all_correct_flags)
    totals = {k: 0.0 for k in ks}
    for flags, scores in zip(all_correct_flags, all_scores):
        scores = np.asarray(scores, dtype=float)
        flags = np.asarray(flags, dtype=bool)
        N = len(flags)
        order = np.argsort(scores)[::-1]
        for k in ks:
            if k > N:
                continue
            topk = order[:k]
            totals[k] += float(flags[topk].any())
    return {k: totals[k] / num_problems for k in ks}


def bootstrap_ci(
    metric_fn, all_correct_flags, ks, all_scores=None, n_boot=1000, alpha=0.05, seed=42
):
    rng = np.random.default_rng(seed)
    n = len(all_correct_flags)
    bootstrapped = {k: [] for k in ks}
    for _ in range(n_boot):
        idxs = rng.integers(0, n, size=n)
        flags_bs = [all_correct_flags[i] for i in idxs]
        if all_scores is not None:
            scores_bs = [all_scores[i] for i in idxs]
            metrics = metric_fn(flags_bs, scores_bs, ks)
        else:
            metrics = metric_fn(flags_bs, ks)
        for k in ks:
            bootstrapped[k].append(metrics[k])
    ci = {}
    for k in ks:
        lower = np.percentile(bootstrapped[k], 100 * alpha / 2)
        upper = np.percentile(bootstrapped[k], 100 * (1 - alpha / 2))
        ci[k] = (lower, upper)
    return ci


def compute_advantages(rewards, gamma=0.99, baseline=None):
    T = len(rewards)
    advantages = np.zeros(T)
    for t in range(T):
        discounted_sum = 0
        for s in range(t, T):
            discounted_sum += (gamma ** (s - t)) * rewards[s]
        if baseline is not None:
            advantages[t] = discounted_sum - baseline[t]
        else:
            advantages[t] = discounted_sum
    return advantages


def extract_flags(df: pd.DataFrame, num_generations: int = 16, disc: bool = True):
    all_correct_flags = []
    for i in range(0, len(df), num_generations):
        sub_df = df.iloc[i : i + num_generations]
        all_correct_flags.append(
            np.array(sub_df.correctness_reward_func == 2, dtype=int).tolist()
        )
    return all_correct_flags


# -------------------------------
# IO + plotting orchestration
# -------------------------------


def read_and_enhance(jsonl_path: str, gamma: float = 0.9) -> pd.DataFrame:
    df = pd.read_json(jsonl_path, lines=True)
    df["reward_model_score_np"] = df["reward_model_score"].apply(
        lambda x: (np.array(x, dtype=float))[~np.isnan(np.array(x, dtype=float))]
    )
    df["mean_rewards"] = df["reward_model_score_np"].apply(lambda x: np.nanmean(x))
    df["reward_model_score_np_discounted"] = df["reward_model_score_np"].apply(
        lambda r: compute_advantages(r, gamma=gamma)
    )
    df["mean_rewards_discounted"] = df["reward_model_score_np_discounted"].apply(
        lambda x: np.nanmean(x)
    )

    from transformers import AutoTokenizer

    if "qwen" in str(jsonl_path) and "response_token" not in df.columns:
        tokeniser = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
        df = df.copy()
        df["response_token_ids"] = df.apply(
            lambda x: tokeniser(x["generation"]["content"] + tokeniser.eos_token)[
                "input_ids"
            ],
            axis=1,
        )
        df["response_token"] = df.apply(
            lambda x: tokeniser.convert_ids_to_tokens(x["response_token_ids"]), axis=1
        )
    elif "llama" in str(jsonl_path) and "response_token" not in df.columns:
        tokeniser = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
        df = df.copy()
        # need to take away the first one, because llama tokeniser puts a `<|begin_of_text|>` there.
        df["response_token_ids"] = df.apply(
            lambda x: tokeniser(x["generation"]["content"] + tokeniser.eos_token)[
                "input_ids"
            ][1:],
            axis=1,
        )
        df["response_token"] = df.apply(
            lambda x: tokeniser.convert_ids_to_tokens(x["response_token_ids"]), axis=1
        )
    else:
        raise NotImplemented(
            "`llama` or `qwen` not found in output dir, do not know which tokeniser to use."
        )

    df["answer_positions"] = df["response_token"].apply(
        lambda x: (
            (x.index("answer"), -4)
            if "answer" in x and x.index("answer") < len(x) - 4
            else (-10, -4)
        )
    )
    df["selector"] = df.apply(
        lambda x: np.nanmean(
            x.reward_model_score_np[x.answer_positions[0] : x.answer_positions[1]]
        ),
        axis=1,
    )
    df["selector_discounted"] = df.apply(
        lambda x: np.nanmean(
            x.reward_model_score_np_discounted[
                x.answer_positions[0] : x.answer_positions[1]
            ]
        ),
        axis=1,
    )
    # df["selector"] = df["mean_rewards"]
    # df["selector_discounted"] = df["mean_rewards_discounted"]
    return df


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_latex_table_txt(
    results: Dict, cis: Dict, ks: Iterable[int], out_file: str | Path
):
    """
    Write a LaTeX table fragment (4 columns for k in {1,3,5,10}).

    Keys expected in `results`/`cis`:
      - "Outcome Sup."        (GRPO row content)
      - "Exp. Reas. (ours)"   (AIRL row content)
      - "SFT"                 (SFT row content)
    """

    def _fmt_row(vals_label: str) -> str:
        return (
            f"{results[vals_label][1]:.2f} [{cis[vals_label][1][0]:.2f}, {cis[vals_label][1][1]:.2f}] & "
            f"{results[vals_label][3]:.2f} [{cis[vals_label][3][0]:.2f}, {cis[vals_label][3][1]:.2f}] & "
            f"{results[vals_label][5]:.2f} [{cis[vals_label][5][0]:.2f}, {cis[vals_label][5][1]:.2f}] & "
            f"{results[vals_label][10]:.2f} [{cis[vals_label][10][0]:.2f}, {cis[vals_label][10][1]:.2f}] \\\\"
        )

    lines = []
    lines.append("& \\rowcolor{gray!20}\\textcolor{gray!90}{GRPO}")
    lines.append(
        "                & \\textcolor{gray!90}{" + _fmt_row("Outcome Sup.") + "}"
    )
    lines.append("& AIRL (ours)    & " + _fmt_row("Exp. Reas. (ours)"))
    lines.append("& SFT            & " + _fmt_row("SFT"))

    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n".join(lines))


def print_latex_table(results: Dict, cis: Dict, ks: Iterable[int]) -> None:
    """
    Print the exact LaTeX fragment to stdout, so you can copy/paste into your paper.
    """

    def _fmt_row(vals_label: str) -> str:
        return (
            f"{results[vals_label][1]:.2f} [{cis[vals_label][1][0]:.2f}, {cis[vals_label][1][1]:.2f}] & "
            f"{results[vals_label][3]:.2f} [{cis[vals_label][3][0]:.2f}, {cis[vals_label][3][1]:.2f}] & "
            f"{results[vals_label][5]:.2f} [{cis[vals_label][5][0]:.2f}, {cis[vals_label][5][1]:.2f}] & "
            f"{results[vals_label][10]:.2f} [{cis[vals_label][10][0]:.2f}, {cis[vals_label][10][1]:.2f}] \\\\"
        )

    print("& \\rowcolor{gray!20}\\textcolor{gray!90}{GRPO}")
    print("                & \\textcolor{gray!90}{" + _fmt_row("Outcome Sup.") + "}")
    print("& AIRL (ours)    & " + _fmt_row("Exp. Reas. (ours)"))
    print("& SFT            & " + _fmt_row("SFT"))


def compute_pass_results_ci(datasets: Dict[str, List[List[bool]]], ks: Iterable[int]):
    """
    Return (results, cis) dictionaries used for pass@k tables/plots.
    """
    results, cis = {}, {}
    for label, flags in datasets.items():
        res = compute_pass_at_k(flags, ks)
        ci = bootstrap_ci(compute_pass_at_k, flags, ks)
        results[label] = res
        cis[label] = ci
    return results, cis


def plot_pass_at_k(
    datasets: Dict[str, List[List[bool]]],
    ks: Iterable[int],
    out_path: str | Path,
    title: str = "pass@k comparison",
):
    results = {}
    cis = {}
    for label, flags in datasets.items():
        res = compute_pass_at_k(flags, ks)
        ci = bootstrap_ci(compute_pass_at_k, flags, ks)
        results[label] = res
        cis[label] = ci

    prop_cycle = plt.rcParams.get("axes.prop_cycle")
    colors = prop_cycle.by_key()["color"] if prop_cycle else [None] * 3
    styles = {
        "Outcome Sup.": {
            "color": colors[2] if len(colors) > 2 else None,
            "marker": "x",
            "linestyle": "--",
        },
        "Exp. Reas. (ours)": {
            "color": colors[0] if colors else None,
            "marker": "x",
            "linestyle": "--",
        },
        "SFT": {
            "color": colors[1] if len(colors) > 1 else None,
            "marker": "x",
            "linestyle": "--",
        },
    }

    plt.figure(figsize=(6, 3))
    for label in results:
        means = [results[label][k] for k in ks]
        ci = [cis[label][k] for k in ks]
        lower = [m - c[0] for m, c in zip(means, ci)]
        upper = [c[1] - m for m, c in zip(means, ci)]
        style = styles.get(label, {"color": None, "marker": "x", "linestyle": "--"})
        plt.errorbar(
            ks,
            means,
            yerr=[lower, upper],
            label=label,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            capsize=4,
            markersize=6,
        )
    plt.xlabel("k")
    plt.ylabel("pass@k")
    # plt.title(title)
    plt.legend()
    plt.grid()
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_success_at_k_given(
    df: pd.DataFrame,
    ks: Iterable[int],
    num_generations: int,
    out_path: str | Path,
    title: str,
):
    # Extract flags + scores
    all_correct_flags, all_scores = [], []
    for i in range(0, len(df), num_generations):
        sub_df = df.iloc[i : i + num_generations]
        all_correct_flags.append(
            np.array(sub_df.correctness_reward_func == 2, dtype=int).tolist()
        )
        all_scores.append(sub_df["selector_discounted"].tolist())

    all_dummy_scores = [[0.0] * num_generations for _ in range(len(all_correct_flags))]

    results_given = compute_success_at_k_from_scores(all_correct_flags, all_scores, ks)
    cis_given = bootstrap_ci(
        compute_success_at_k_from_scores, all_correct_flags, ks, all_scores=all_scores
    )

    results_uniform = compute_success_at_k_from_scores(
        all_correct_flags, all_dummy_scores, ks
    )
    cis_uniform = bootstrap_ci(
        compute_success_at_k_from_scores,
        all_correct_flags,
        ks,
        all_scores=all_dummy_scores,
    )

    prop_cycle = plt.rcParams.get("axes.prop_cycle")
    colors = prop_cycle.by_key()["color"] if prop_cycle else [None, None]
    styles = {
        "Reward Reranker": {
            "color": colors[0] if colors else None,
            "marker": "x",
            "linestyle": "--",
        },
        "Random Ranking": {
            "color": colors[1] if len(colors) > 1 else None,
            "marker": "x",
            "linestyle": "--",
        },
    }

    plt.figure(figsize=(6, 3))
    for label, (results_model, cis_model) in {
        "Reward Reranker": (results_given, cis_given),
        "Random Ranking": (results_uniform, cis_uniform),
    }.items():
        means = [results_model[k] for k in ks]
        ci = [cis_model[k] for k in ks]
        lower = [m - c[0] for m, c in zip(means, ci)]
        upper = [c[1] - m for m, c in zip(means, ci)]
        style = styles[label]
        plt.errorbar(
            ks,
            means,
            yerr=[lower, upper],
            label=label,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            capsize=4,
            markersize=6,
        )

    plt.xlabel("k")
    plt.ylabel(rf"pass@k$\mid${num_generations}")
    plt.title(title)
    plt.legend()
    plt.grid()
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_reward_distributions(
    df: pd.DataFrame, out_pdf: str | Path, out_pdf_discounted: str | Path
):
    import scipy.stats as stats

    correct = df[df.correctness_reward_func == 2].mean_rewards
    wrong = df[df.correctness_reward_func == 0].mean_rewards

    t_stat, p_value = stats.ttest_ind(correct, wrong, equal_var=False)

    plt.figure(figsize=(6, 3))
    sns.histplot(
        wrong,
        label="Wrong Answer",
        kde=True,
        stat="probability",
        bins=50,
        color="C1",
        alpha=0.5,
        edgecolor=None,
        shrink=0.85,
        linewidth=0,
    )
    sns.histplot(
        correct,
        label="Correct Answer",
        kde=True,
        stat="probability",
        bins=50,
        color="C0",
        alpha=0.5,
        edgecolor=None,
        shrink=0.85,
        linewidth=0,
    )
    plt.legend()
    plt.xlabel("Mean Rewards")
    plt.ylabel("Probability")
    # plt.title("Distribution of Rewards based on Correctness")
    p_text = "$p < 0.001$" if p_value < 0.001 else f"p = {p_value:.3f}"
    text = f"t = {t_stat:.2f}, {p_text}"
    plt.text(
        0.03,
        0.78,
        text,
        transform=plt.gca().transAxes,
        fontsize=10,
        va="top",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
    )
    ensure_dir(Path(out_pdf).parent)
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()

    correct = df[df.correctness_reward_func == 2].mean_rewards_discounted
    wrong = df[df.correctness_reward_func == 0].mean_rewards_discounted
    t_stat, p_value = stats.ttest_ind(correct, wrong, equal_var=False)

    plt.figure(figsize=(6, 3))
    sns.histplot(
        wrong,
        label="Wrong Answer",
        kde=True,
        stat="probability",
        bins=50,
        color="C1",
        alpha=0.5,
        edgecolor=None,
        shrink=0.85,
        linewidth=0,
    )
    sns.histplot(
        correct,
        label="Correct Answer",
        kde=True,
        stat="probability",
        bins=50,
        color="C0",
        alpha=0.5,
        edgecolor=None,
        shrink=0.85,
        linewidth=0,
    )
    plt.legend()
    plt.xlabel("Mean Discounted Rewards")
    plt.ylabel("Probability")
    # plt.title("Distribution of Discounted Rewards based on Correctness")
    p_text = "$p < 0.001$" if p_value < 0.001 else f"p = {p_value:.3f}"
    text = f"t = {t_stat:.2f}, {p_text}"
    plt.text(
        0.03,
        0.78,
        text,
        transform=plt.gca().transAxes,
        fontsize=10,
        va="top",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
    )
    plt.savefig(out_pdf_discounted, bbox_inches="tight")
    plt.close()


def plot_rewards_vs_discounted(df: pd.DataFrame, out_pdf: str | Path):
    # Pick a reasonable example: near-zero mean but correct
    idx = df[(abs(df["mean_rewards"]) < 0.01) & (df["correctness_reward_func"] == 2)][
        "mean_rewards"
    ].idxmax()
    rewards = df.loc[idx, "reward_model_score_np"]
    discounted_rewards = df.loc[idx, "reward_model_score_np_discounted"]

    plt.figure(figsize=(16, 4))
    plt.subplot(1, 2, 1)
    sns.barplot(x=list(range(len(rewards))), y=rewards, color="C0")
    # plt.title("Raw Rewards")
    plt.xlabel("Token Timestep")
    plt.ylabel("Reward")
    plt.xticks(rotation=90)

    plt.subplot(1, 2, 2)
    sns.barplot(
        x=list(range(len(discounted_rewards))), y=discounted_rewards, color="C1"
    )
    # plt.title("Discounted Rewards")
    plt.xlabel("Timestep")
    plt.ylabel("Discounted Reward")
    plt.xticks(rotation=90)

    plt.tight_layout()
    ensure_dir(Path(out_pdf).parent)
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()


def plot_formatting_distributions(
    df: pd.DataFrame, out_pdf: str | Path, out_pdf_discounted: str | Path
):
    import scipy.stats as stats

    plt.figure(figsize=(10, 5))

    sns.histplot(
        df[df.strict_format_reward_func == 0].selector,
        label="Wrong Format",
        kde=True,
        stat="probability",
        bins=50,
        color="C1",
        alpha=0.5,
        edgecolor=None,
        shrink=0.85,
        linewidth=0,
    )
    sns.histplot(
        df[df.strict_format_reward_func == 0.5].selector,
        label="Correct Format",
        kde=True,
        stat="probability",
        bins=50,
        color="C0",
        alpha=0.5,
        edgecolor=None,
        shrink=0.85,
        linewidth=0,
    )
    correct = df[df.strict_format_reward_func == 0.5].selector
    wrong = df[df.strict_format_reward_func == 0.0].selector
    t_stat, p_value = stats.ttest_ind(correct, wrong, equal_var=False)
    plt.legend()
    plt.xlabel("Mean Discounted Rewards")
    plt.ylabel("Probability")
    # plt.title("Distribution of Rewards based on Formatting")
    p_text = "$p < 0.001$" if p_value < 0.001 else f"p = {p_value:.3f}"
    text = f"t = {t_stat:.2f}, {p_text}"
    plt.text(
        0.02,
        0.95,
        text,
        transform=plt.gca().transAxes,
        fontsize=10,
        va="top",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
    )
    ensure_dir(Path(out_pdf).parent)
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 5))
    sns.histplot(
        df[df.strict_format_reward_func == 0].selector_discounted,
        label="Wrong Format",
        kde=True,
        stat="probability",
        bins=50,
        color="C1",
        alpha=0.5,
        edgecolor=None,
        shrink=0.85,
        linewidth=0,
    )
    sns.histplot(
        df[df.strict_format_reward_func == 0.5].selector_discounted,
        label="Correct Format",
        kde=True,
        stat="probability",
        bins=50,
        color="C0",
        alpha=0.5,
        edgecolor=None,
        shrink=0.85,
        linewidth=0,
    )

    correct = df[df.strict_format_reward_func == 0.5].selector_discounted
    wrong = df[df.strict_format_reward_func == 0.0].selector_discounted
    t_stat, p_value = stats.ttest_ind(correct, wrong, equal_var=False)
    plt.legend()
    plt.xlabel("Mean Discounted Rewards")
    plt.ylabel("Probability")
    # plt.title("Distribution of Discounted Rewards based on Formatting")
    p_text = "$p < 0.001$" if p_value < 0.001 else f"p = {p_value:.3f}"
    text = f"t = {t_stat:.2f}, {p_text}"
    plt.text(
        0.02,
        0.95,
        text,
        transform=plt.gca().transAxes,
        fontsize=10,
        va="top",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
    )
    plt.savefig(out_pdf_discounted, bbox_inches="tight")
    plt.close()


def plot_reward_correlations(df: pd.DataFrame, out_pdf: str | Path):
    reward_cols = [
        "selector",
        "selector_discounted",
        "xmlcount_reward_func",
        "strict_format_reward_func",
        "int_reward_func",
        "correctness_reward_func",
    ]
    rename_map = {
        "selector": "Rewards",
        "selector_discounted": "Rewards\n(Discounted)",
        "xmlcount_reward_func": "XML Count",
        "strict_format_reward_func": "Strict Format",
        "int_reward_func": "Integer",
        "correctness_reward_func": "Correctness",
    }
    corr_matrix = df[reward_cols].corr()
    corr_matrix = corr_matrix.rename(index=rename_map, columns=rename_map)
    plt.figure(figsize=(6, 4))
    ax = sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap=CUSTOM_COLOR_MAP,
        vmin=-1,
        vmax=1,
        cbar_kws={"shrink": 0.8, "label": "Correlation"},
        linewidths=0.5,
        square=True,
    )
    # plt.title("Correlation Matrix of GRPO Reward Functions with Reward Model", fontsize=14, pad=20)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    ensure_dir(Path(out_pdf).parent)
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()


# -------------------------------
# Orchestrator to run everything for one experiment trio
# -------------------------------


def run_all_plots(
    df_airl: pd.DataFrame,
    df_sft: pd.DataFrame,
    df_grpo: pd.DataFrame,
    out_dir: str | Path,
    num_generations: int = 16,
    make_token_figs: bool = True,
):
    out_dir = ensure_dir(out_dir)

    ks = [1, 3, 5, 10]
    datasets = {
        "Outcome Sup.": extract_flags(df_grpo, num_generations),
        "Exp. Reas. (ours)": extract_flags(df_airl, num_generations),
        "SFT": extract_flags(df_sft, num_generations),
    }

    # NEW: compute + print + save LaTeX table fragment
    results, cis = compute_pass_results_ci(datasets, ks)
    print_latex_table(results, cis, ks)  # for direct copy/paste in your terminal
    save_latex_table_txt(results, cis, ks, Path(out_dir) / "pass_at_k_table.txt")

    plot_pass_at_k(
        datasets, ks, out_dir / "pass_at_k_all.pdf", title="pass@k comparison"
    )

    # success@k|N for AIRL (expert reasoning)
    plot_success_at_k_given(
        df_airl,
        ks,
        num_generations,
        out_dir / "pass_atkN_expert.pdf",
        title=r"Expert Reasoning: pass@k$\mid$N comparison",
    )

    # distributions by correctness (AIRL)
    plot_reward_distributions(
        df_airl,
        out_dir / "correctness_reward_distribution.pdf",
        out_dir / "correctness_reward_distribution_discounted.pdf",
    )

    # # raw vs discounted
    # plot_rewards_vs_discounted(df_airl, out_dir / "rewards_vs_discounted.pdf")

    # # formatting distributions
    # plot_formatting_distributions(
    #     df_airl,
    #     out_dir / "format_rewards.pdf",
    #     out_dir / "format_rewards_discounted.pdf",
    # )

    # # correlation heatmap
    # plot_reward_correlations(df_airl, out_dir / "reward_correlation_matrix.pdf")

    # # Token-based dense reward visualisations (best-effort; requires tokenizer + fields)
    # if make_token_figs:
    #     colour_map = CUSTOM_COLOR_MAP
    #     discs = [False, True]

    #     for disc in discs:
    #         reward_score_name = (
    #             "reward_model_score_np_discounted" if disc else "reward_model_score_np"
    #         )
    #         postfix = "discounted" if disc else "raw"
    #         mean_name = "mean_rewards_discounted" if disc else "mean_rewards"

    #         if "response_token" in df_airl.columns:
    #             plt.rcParams["text.usetex"] = False
    #             correct_mean = df_airl[df_airl["correctness_reward_func"] == 2][mean_name].mean()
    #             wrong_mean =  df_airl[df_airl["correctness_reward_func"] == 0][mean_name].mean()
    #             overall_mean = df_airl[mean_name].mean()
                
    #             df_airl["reward_model_standard"] = df_airl[reward_score_name].apply(lambda x: x - overall_mean)
                
                
    #             positive_indicies = df_airl[(abs(df_airl[mean_name]- correct_mean) < 0.1) & (df_airl["correctness_reward_func"] == 2) & (df_airl["int_reward_func"] == 0.5)][mean_name].index[:5]
    #             negative_indicies = df_airl[(abs(df_airl[mean_name]- wrong_mean) < 0.1) & (df_airl["correctness_reward_func"] != 2) & (df_airl["int_reward_func"] == 0.5)][mean_name].index[:5]
    #             all_indices = np.concatenate([positive_indicies, negative_indicies ])
    #             df_airl["reward_model_max"] = df_airl["reward_model_standard"].apply(lambda x: max(x))
    #             df_airl["reward_model_min"] = df_airl["reward_model_standard"].apply(lambda x: min(x))
    #             max_value = df_airl.loc[all_indices, "reward_model_max"].max()
    #             min_value = df_airl.loc[all_indices, "reward_model_min"].min()
            
    #             for i, idx in enumerate(positive_indicies):
    #                 tokens = df_airl.loc[idx, "response_token"]
    #                 scores = df_airl.loc[idx, "reward_model_standard"]
    #                 question = df_airl.loc[idx, "prompt"][1]["content"]
    #                 make_text_reward_image(
    #                     tokens,
    #                     scores,
    #                     out_dir / f"dense_rewards_{postfix}/true_{i}.pdf",
    #                     cmap_name=colour_map,
    #                     prompt_text=question,
    #                     font_size=18,
    #                     dpi=300,
    #                     max_width_px=4000,
    #                     max_val=max_value,
    #                     min_val=min_value
    #                 )
            
    #             for i, idx in enumerate(negative_indicies):
    #                 tokens = df_airl.loc[idx, "response_token"]
    #                 scores = df_airl.loc[idx, "reward_model_standard"]
    #                 question = df_airl.loc[idx, "prompt"][1]["content"]
    #                 make_text_reward_image(
    #                     tokens,
    #                     scores,
    #                     out_dir / f"dense_rewards_{postfix}/wrong_{i}.pdf",
    #                     cmap_name=colour_map,
    #                     prompt_text=question,
    #                     font_size=18,
    #                     dpi=300,
    #                     max_width_px=4000,
    #                     max_val=max_value,
    #                     min_val=min_value
    #                 )
                

    # return path for reference
    return Path(out_dir)
