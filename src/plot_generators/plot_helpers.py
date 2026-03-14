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




def strict_format_reward_func(response, **kwargs):
    return 0.5 if STRICT_FMT.match(response) else 0.0


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
NEWLINE_CHAR = "Ċ"
SPACE_CHAR = "Ġ"
_QWEN_SPECIAL_RE = re.compile(r"<\|.*?\|>")

def _escape_fmt(text: str) -> str:
    """
    Escapes special Matplotlib characters. 
    Mainly '$' which triggers math mode if unescaped.
    """
    if not text:
        return ""
    # Replace '$' with '\$' to prevent Matplotlib from expecting an equation
    return text.replace("$", "\\$")

class TextMeasurer:
    """Helper to measure text width exactly using Matplotlib's engine."""
    def __init__(self, font_properties, dpi):
        self.fp = font_properties
        self.dpi = dpi
        self.fig = plt.figure(dpi=dpi)
        self.fig.canvas.draw()
        self.renderer = self.fig.canvas.get_renderer()
        
    def get_text_width(self, text, fontsize):
        # We assume text is already escaped/safe when passed here
        txt = plt.Text(0, 0, text, fontproperties=self.fp, fontsize=fontsize)
        txt.set_figure(self.fig)
        bbox = txt.get_window_extent(renderer=self.renderer)
        return bbox.width

    def wrap_text(self, text, fontsize, max_width_px):
        # Normalize spaces and split
        words = text.replace("\n", " \n ").split(" ")
        lines = []
        current_line = []
        current_width = 0
        space_w = self.get_text_width(" ", fontsize)

        for word in words:
            if word == "\n":
                lines.append(" ".join(current_line))
                current_line = []
                current_width = 0
                continue
            if not word: continue
            
            w = self.get_text_width(word, fontsize)
            
            if current_line and (current_width + space_w + w <= max_width_px):
                current_line.append(word)
                current_width += space_w + w
            elif not current_line:
                current_line.append(word)
                current_width = w
            else:
                lines.append(" ".join(current_line))
                current_line = [word]
                current_width = w
                
        if current_line: lines.append(" ".join(current_line))
        return [l for l in lines if l.strip()]

    def close(self):
        plt.close(self.fig)

def _analyze_token(raw_tok: str, score: float) -> dict:
    if raw_tok is None:
        return {"text": "", "score": score, "is_start_word": False, "newlines_after": 0, "is_special": False}
    
    if raw_tok in ["<|im_end|>", "<|endoftext|>"]:
        return {"text": "", "score": score, "is_start_word": False, "newlines_after": 1, "is_special": True}
    
    is_start_word = raw_tok.startswith(SPACE_CHAR)
    newlines_after = raw_tok.count(NEWLINE_CHAR)
    
    # Clean standard token artifacts
    txt = raw_tok.replace(SPACE_CHAR, "").replace(NEWLINE_CHAR, "")
    txt = _QWEN_SPECIAL_RE.sub("", txt).replace("\u0120", " ").strip()
    
    # --- FIX: Escape special characters in the token text ---
    txt = _escape_fmt(txt)
    
    is_special = txt.startswith("<") and txt.endswith(">")

    return {
        "text": txt, "score": score, "is_start_word": is_start_word,
        "newlines_after": newlines_after, "is_special": is_special
    }
    
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


def make_text_reward_image(
    tokens: List[str],
    scores: List[float],
    out_path: str,
    prompt_text: Optional[str] = None,
    title: Optional[str] = None,
    cmap_name: str = "Blues",
    pad_x: int = 5,
    pad_y: int = 4,
    gap_word: int = 10,
    gap_subword: int = 1,
    gap_y: int = 10,
    font_size: int = 12,
    max_width_px: int = 800, 
    dpi: int = 200,
    font_properties: Optional[FontProperties] = None,
    show_colorbar: bool = True,
    max_val=None,
    min_val=None
):
    fp = font_properties or FontProperties(size=font_size, family="DejaVu Sans")
    measurer = TextMeasurer(fp, dpi)
    
    # --- 1. SETUP COLOR NORMALIZATION ---
    vals = np.array(scores)
    if max_val is None:
        abs_max = np.max(np.abs(vals)) if vals.size > 0 else 1.0
    else:
        abs_max = max(abs(max_val), abs(min_val)) if min_val is not None else abs(max_val)
    if abs_max == 0: abs_max = 1.0 
    
    norm_obj = mcolors.Normalize(vmin=-abs_max, vmax=abs_max)
    
    # --- 2. PREPARE PILLS ---
    try:
        norm_scores = np.array(normalise(scores, mode="diverging", max_val=max_val, min_val=min_val))
    except NameError:
        norm_scores = vals

    processed_tokens = []
    for t, sc in zip(tokens, norm_scores):
        processed_tokens.append(_analyze_token(t, sc))

    for item in processed_tokens:
        if item["text"]:
            item["width"] = measurer.get_text_width(item["text"], font_size)
        else:
            item["width"] = 0

    pill_h = (font_size * 1.4 * dpi / 72) + 2 * pad_y 
    row_height_px = pill_h + gap_y
    min_text_w = measurer.get_text_width("i", font_size) * 1.5 

    rows = []
    cur_row = []
    cur_x = 0
    
    for item in processed_tokens:
        txt = item["text"]
        if txt:
            if not cur_row: gap = 0
            elif item["is_start_word"]: gap = gap_word
            else: gap = gap_subword

            pill_w = max(item["width"], min_text_w) + 2 * pad_x
            w_total = gap + pill_w
            
            if cur_row and (cur_x + w_total > max_width_px * 0.96):
                rows.append(cur_row)
                cur_row = []
                cur_x = 0
                gap = 0
            
            cur_row.append({
                "text": txt, "score": item["score"], "pill_w": pill_w, 
                "gap_before": gap, "is_special": item["is_special"]
            })
            cur_x += gap + pill_w
        
        if item["newlines_after"] > 0:
            if cur_row: rows.append(cur_row)
            cur_row = []
            cur_x = 0
            for _ in range(item["newlines_after"] - 1): rows.append([])

    if cur_row: rows.append(cur_row)

    # --- 3. BUILD LAYOUT BLOCKS ---
    layout_blocks = []
    content_width = max_width_px - 40 

    if title:
        safe_title = _escape_fmt(title)
        t_lines = measurer.wrap_text(safe_title, font_size + 2, content_width)
        layout_blocks.append(("text_lines", t_lines, {"size": font_size + 2, "style": "italic", "weight": "normal"}))
        layout_blocks.append(("spacer", 20, {}))

    if prompt_text:
        layout_blocks.append(("text_lines", ["Question:"], {"size": font_size, "style": "normal", "weight": "bold"}))
        clean_prompt = prompt_text.replace(SPACE_CHAR, " ").replace(NEWLINE_CHAR, "\n")
        safe_prompt = _escape_fmt(clean_prompt)
        p_lines = measurer.wrap_text(safe_prompt, font_size, content_width)
        layout_blocks.append(("text_lines", p_lines, {"size": font_size, "style": "normal", "weight": "normal"}))
        layout_blocks.append(("spacer", 60, {}))

    layout_blocks.append(("text_lines", ["Reasoning + Answer:"], {"size": font_size, "style": "normal", "weight": "bold"}))
    layout_blocks.append(("spacer", 20, {}))
    layout_blocks.append(("pills", rows, {}))

    measurer.close()

    # --- 4. CALCULATE GEOMETRY ---
    total_text_height = 0
    top_margin = 20
    bottom_margin = 20
    
    def get_line_height(fs): return fs * 1.5 * dpi / 72

    for kind, content, params in layout_blocks:
        if kind == "text_lines":
            lh = get_line_height(params["size"])
            total_text_height += len(content) * lh
        elif kind == "spacer":
            total_text_height += content
        elif kind == "pills":
            total_text_height += len(content) * row_height_px

    final_h_px = top_margin + total_text_height + bottom_margin

    # --- 5. FIGURE DIMENSIONS (FIXED MARGINS) ---
    cbar_width_px = 30 if show_colorbar else 0
    cbar_pad_px = 40 if show_colorbar else 0
    
    # FIX: Add specific margin for the labels (numbers) on the right
    # 80px should be plenty for standard fonts to not get cut off
    cbar_labels_margin = 80 if show_colorbar else 20 

    total_fig_width_px = max_width_px + cbar_pad_px + cbar_width_px + cbar_labels_margin

    width_in = total_fig_width_px / dpi
    height_in = max(1.0, final_h_px / dpi)
    
    fig = plt.figure(figsize=(width_in, height_in), dpi=dpi)
    
    # Text Axis covers the WHOLE figure, but we only draw in the max_width_px area
    ax = plt.axes([0, 0, 1, 1])
    ax.set_xlim(0, total_fig_width_px)
    ax.set_ylim(0, final_h_px)
    ax.axis("off")

    try:
        cmap = plt.get_cmap(cmap_name)
    except:
        cmap = plt.cm.Blues

    # --- 6. DRAW TEXT CONTENT ---
    y = final_h_px - top_margin
    left_margin = 20

    for kind, content, params in layout_blocks:
        if kind == "text_lines":
            lh = get_line_height(params["size"])
            for line in content:
                ax.text(
                    left_margin, y, line,
                    fontsize=params["size"],
                    fontstyle=params["style"],
                    fontweight=params["weight"],
                    va="top", ha="left",
                    fontproperties=fp
                )
                y -= lh
        
        elif kind == "spacer":
            y -= content
            
        elif kind == "pills":
            y_row_top = y
            for row in content:
                x = left_margin
                if not row:
                    y_row_top -= row_height_px
                    continue
                
                for pill in row:
                    x += pill["gap_before"]
                    
                    sc = pill["score"]
                    if pill["is_special"]:
                        face = (0.95, 0.95, 0.95, 1.0)
                        edge = (0.8, 0.8, 0.8, 1.0)
                        txt_col = "black"
                    else:
                        face = cmap(sc)
                        lum = 0.299 * face[0] + 0.587 * face[1] + 0.114 * face[2]
                        txt_col = "white" if lum < 0.5 else "black"
                        edge = (0.9, 0.9, 0.9, 0.0)

                    # Draw Box
                    y_pill = y_row_top - pill_h
                    patch = FancyBboxPatch(
                        (x, y_pill), pill["pill_w"], pill_h,
                        boxstyle="round,pad=0.0,rounding_size=6",
                        linewidth=1.0, edgecolor=edge, facecolor=face
                    )
                    ax.add_patch(patch)

                    # Draw Text
                    ax.text(
                        x + pill["pill_w"]/2, y_pill + pill_h/2,
                        pill["text"],
                        fontsize=font_size,
                        va="center", ha="center",
                        color=txt_col, fontproperties=fp
                    )
                    x += pill["pill_w"]
                
                y_row_top -= row_height_px
            y = y_row_top

    # --- 7. DRAW COLORBAR ---
    if show_colorbar:
        # Calculate position in normalized coordinates (0 to 1)
        # Left edge of bar starts after text area + padding
        cb_left = (max_width_px + cbar_pad_px) / total_fig_width_px
        cb_width = cbar_width_px / total_fig_width_px
        
        # Center vertically
        cb_height = 0.8
        cb_bottom = (1.0 - cb_height) / 2.0
        
        cax = fig.add_axes([cb_left, cb_bottom, cb_width, cb_height])
        
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_obj)
        sm.set_array([]) 
        
        cbar = fig.colorbar(sm, cax=cax, orientation='vertical')
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(labelsize=font_size-2, size=0)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
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

def discounted_mean(scores, gamma=0.9):
    """
    Calculates a weighted average where the last element has the highest weight (1.0),
    and previous elements decay by a factor of gamma. Handles NaNs.
    """
    # Ensure input is a numpy array
    scores = np.array(scores)
    
    # Create a mask for valid (non-NaN) values
    mask = ~np.isnan(scores)
    
    # If all values are NaN, return NaN
    if not np.any(mask):
        return np.nan
        
    # Generate weights: [gamma^(n-1), ..., gamma^1, 1]
    n = len(scores)
    weights = gamma ** np.arange(n)[::-1]
    
    # Apply mask to both scores and weights
    valid_scores = scores[mask]
    valid_weights = weights[mask]
    
    # Calculate weighted average
    return np.sum(valid_scores * valid_weights) / np.sum(valid_weights)


def read_and_enhance(jsonl_path: str, gamma: float = 0.9, answer_only: bool = False) -> pd.DataFrame:
    df = pd.read_json(jsonl_path, lines=True)

    # df["reward_model_score_np_discounted"] = df["reward_model_score_np"].apply(
    #     lambda r: compute_advantages(r, gamma=gamma)
    # )
    # df["mean_rewards_discounted"] = df["reward_model_score_np_discounted"].apply(
    #     lambda x: np.nanmean(x)
    # )

    from transformers import AutoTokenizer

    if "qwen" in str(jsonl_path) and "response_token" not in df.columns:
        tokeniser = AutoTokenizer.from_pretrained("unsloth/qwen2.5-7b-instruct-unsloth-bnb-4bit")
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
        df["reward_model_score"] = df["reward_model_score"].apply(lambda x: [x[0]] + x)
    elif "llama" in str(jsonl_path) and "response_token" not in df.columns:
        tokeniser = AutoTokenizer.from_pretrained("unsloth/llama-3.1-8b-instruct-unsloth-bnb-4bit")
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
    df["reward_model_score_np"] = df["reward_model_score"].apply(
        lambda x: (np.array(x, dtype=float))[~np.isnan(np.array(x, dtype=float))]
    )
    df["mean_rewards"] = df["reward_model_score_np"].apply(lambda x: np.nanmean(x))
    df["strict_format_reward_func"] = df.generation.apply(lambda x: strict_format_reward_func(x["content"]))
    df["xmlcount_reward_func"] = df.generation.apply(lambda x: count_xml(x["content"]))
    df["answer_positions"] = df["response_token"].apply(
        lambda x: (
            (x.index("answer"), -4)
            if "answer" in x and x.index("answer") < len(x) - 4
            else (-10, -4)
        )
    )
    if answer_only:
        # df["selector"] = df.apply(
        #     lambda x: np.nanmean(
        #         x.reward_model_score_np[x.answer_positions[0] : x.answer_positions[1]]
        #     ),
        #     axis=1,
        # )
        df["selector"] = df["reward_model_score_np"].apply(lambda x: discounted_mean(x, gamma=0.95))
    else:
        df["selector"] = df["mean_rewards"].copy()
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
    
def save_latex_table_txt_reranking(
    results: Dict, cis: Dict, ks: Iterable[int], out_file: str | Path
):
    """
    Write a LaTeX table fragment (4 columns for k in {1,3,5,10}).

    Keys expected in `results`/`cis`:
      - "Random Rerankning"               
      - "Reasoning Reranking"             
    """

    def _fmt_row(vals_label: str) -> str:
        return (
            f"{results[vals_label][1]:.2f} [{cis[vals_label][1][0]:.2f}, {cis[vals_label][1][1]:.2f}] & "
            f"{results[vals_label][3]:.2f} [{cis[vals_label][3][0]:.2f}, {cis[vals_label][3][1]:.2f}] & "
            f"{results[vals_label][5]:.2f} [{cis[vals_label][5][0]:.2f}, {cis[vals_label][5][1]:.2f}] & "
            f"{results[vals_label][10]:.2f} [{cis[vals_label][10][0]:.2f}, {cis[vals_label][10][1]:.2f}] \\\\"
        )

    lines = []
    lines.append("Random Reranking & " + _fmt_row("random"))
    lines.append("Reasoning Reranking & " + _fmt_row("reward"))
    lines.append("Length Reranking & " + _fmt_row("heuristic"))
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
        all_scores.append(sub_df["selector"].tolist())

    all_dummy_scores = [[0.0] * num_generations for _ in range(len(all_correct_flags))]
    
    
    all_scores_heuristic = []
    df["length_heuristic"] = df["generation"].apply(lambda x: len(x['content']))
    for i in range(0, len(df), num_generations):
        sub_df = df.iloc[i : i + num_generations]
        all_scores_heuristic.append(sub_df["length_heuristic"].tolist())
    

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
    
    results_heuristic = compute_success_at_k_from_scores(
        all_correct_flags, all_scores_heuristic, ks
    )
    cis_heuristic = bootstrap_ci(
        compute_success_at_k_from_scores,
        all_correct_flags,
        ks,
        all_scores=all_scores_heuristic,
    )
    
    results = {
        "reward": results_given,
        "random": results_uniform,
        "heuristic": results_heuristic,
    } 
    cis = {
        "reward": cis_given,
        "random": cis_uniform,
        "heuristic": cis_heuristic,
    }
    save_latex_table_txt_reranking(results, cis, ks, Path(out_path).parent / "pass_at_k_table_reranking.txt")

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
        # "Length Reranker": {
        #     "color": colors[2] if len(colors) > 1 else None,
        #     "marker": "x",
        #     "linestyle": "--",
        # },
    }

    plt.figure(figsize=(6, 3))
    for label, (results_model, cis_model) in {
        "Reward Reranker": (results_given, cis_given),
        "Random Ranking": (results_uniform, cis_uniform),
        #"Length Reranker": (results_heuristic, cis_heuristic),
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

    # correct = df[df.correctness_reward_func == 2].mean_rewards_discounted
    # wrong = df[df.correctness_reward_func == 0].mean_rewards_discounted
    # t_stat, p_value = stats.ttest_ind(correct, wrong, equal_var=False)

    # plt.figure(figsize=(6, 3))
    # sns.histplot(
    #     wrong,
    #     label="Wrong Answer",
    #     kde=True,
    #     stat="probability",
    #     bins=50,
    #     color="C1",
    #     alpha=0.5,
    #     edgecolor=None,
    #     shrink=0.85,
    #     linewidth=0,
    # )
    # sns.histplot(
    #     correct,
    #     label="Correct Answer",
    #     kde=True,
    #     stat="probability",
    #     bins=50,
    #     color="C0",
    #     alpha=0.5,
    #     edgecolor=None,
    #     shrink=0.85,
    #     linewidth=0,
    # )
    # plt.legend()
    # plt.xlabel("Mean Discounted Rewards")
    # plt.ylabel("Probability")
    # # plt.title("Distribution of Discounted Rewards based on Correctness")
    # p_text = "$p < 0.001$" if p_value < 0.001 else f"p = {p_value:.3f}"
    # text = f"t = {t_stat:.2f}, {p_text}"
    # plt.text(
    #     0.03,
    #     0.78,
    #     text,
    #     transform=plt.gca().transAxes,
    #     fontsize=10,
    #     va="top",
    #     bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
    # )
    # plt.savefig(out_pdf_discounted, bbox_inches="tight")
    # plt.close()


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

    # plt.figure(figsize=(10, 5))
    # sns.histplot(
    #     df[df.strict_format_reward_func == 0].selector_discounted,
    #     label="Wrong Format",
    #     kde=True,
    #     stat="probability",
    #     bins=50,
    #     color="C1",
    #     alpha=0.5,
    #     edgecolor=None,
    #     shrink=0.85,
    #     linewidth=0,
    # )
    # sns.histplot(
    #     df[df.strict_format_reward_func == 0.5].selector_discounted,
    #     label="Correct Format",
    #     kde=True,
    #     stat="probability",
    #     bins=50,
    #     color="C0",
    #     alpha=0.5,
    #     edgecolor=None,
    #     shrink=0.85,
    #     linewidth=0,
    # )

    # correct = df[df.strict_format_reward_func == 0.5].selector_discounted
    # wrong = df[df.strict_format_reward_func == 0.0].selector_discounted
    # t_stat, p_value = stats.ttest_ind(correct, wrong, equal_var=False)
    # plt.legend()
    # plt.xlabel("Mean Discounted Rewards")
    # plt.ylabel("Probability")
    # # plt.title("Distribution of Discounted Rewards based on Formatting")
    # p_text = "$p < 0.001$" if p_value < 0.001 else f"p = {p_value:.3f}"
    # text = f"t = {t_stat:.2f}, {p_text}"
    # plt.text(
    #     0.02,
    #     0.95,
    #     text,
    #     transform=plt.gca().transAxes,
    #     fontsize=10,
    #     va="top",
    #     bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
    # )
    # plt.savefig(out_pdf_discounted, bbox_inches="tight")
    # plt.close()


def plot_reward_correlations(df: pd.DataFrame, out_pdf: str | Path):
    reward_cols = [
        "selector",
        #"selector_discounted",
        "xmlcount_reward_func",
        "strict_format_reward_func",
        #"int_reward_func",
        "correctness_reward_func",
    ]
    rename_map = {
        "selector": "Rewards",
        #"selector_discounted": "Rewards\n(Discounted)",
        "xmlcount_reward_func": "XML Count",
        "strict_format_reward_func": "Strict Format",
        #"int_reward_func": "Integer",
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

    #NEW: compute + print + save LaTeX table fragment
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

    # raw vs discounted
    #plot_rewards_vs_discounted(df_airl, out_dir / "rewards_vs_discounted.pdf")

    # formatting distributions
    plot_formatting_distributions(
        df_airl,
        out_dir / "format_rewards.pdf",
        out_dir / "format_rewards_discounted.pdf",
    )

    # correlation heatmap
    plot_reward_correlations(df_airl, out_dir / "reward_correlation_matrix.pdf")

    # Token-based dense reward visualisations (best-effort; requires tokenizer + fields)
    if make_token_figs:
        colour_map = CUSTOM_COLOR_MAP
        discs = [False]

        for disc in discs:
            reward_score_name = (
                "reward_model_score_np_discounted" if disc else "reward_model_score_np"
            )
            postfix = "discounted" if disc else "raw"
            mean_name = "mean_rewards_discounted" if disc else "mean_rewards"
            if "response_token" in df_airl.columns:
                plt.rcParams["text.usetex"] = False
                # 1. Calculate Means
                # Note: Keeping your logic where 'wrong' mean is based on 0, 
                # but sampling pool is based on != 2.
                correct_mean = df_airl[df_airl["correctness_reward_func"] == 2][mean_name].mean()
                wrong_mean   = df_airl[df_airl["correctness_reward_func"] == 0][mean_name].mean()
                overall_mean = df_airl[mean_name].mean()
                # 2. Standardise rewards (Vectorized is faster than .apply)
                df_airl["prompt_idx"] = np.arange(len(df_airl)) // 16
                df_airl["reward_model_standard"] = df_airl[reward_score_name] - overall_mean

                # 1. Find row index of Correct answer with HIGHEST 'selector'
                # idxmax returns the index label where the max value is found
                pos_series = df_airl[df_airl["correctness_reward_func"] == 2].groupby('prompt_idx')['selector'].idxmax()

                # 2. Find row index of Wrong answer with LOWEST 'selector'
                # idxmin returns the index label where the min value is found
                neg_series = df_airl[df_airl["correctness_reward_func"] == 0].groupby('prompt_idx')['selector'].idxmin()

                # 3. Merge the two series on 'prompt_idx'
                # This aligns them and drops groups that don't have both a correct and wrong answer
                aligned_pairs = pd.merge(pos_series, neg_series, on='prompt_idx', suffixes=('_pos', '_neg'))

                # 4. Extract the aligned lists of indices
                positive_indices = aligned_pairs['selector_pos'].tolist()[:5]
                negative_indices = aligned_pairs['selector_neg'].tolist()[:5]


                # # --- Positive Indices ---
                # # Filter first: Correctness == 2 AND Strict Format == 0.5
                # pos_subset = df_airl[
                #     (df_airl["correctness_reward_func"] == 2) #& 
                #     #(df_airl["strict_format_reward_func"] == 0.5)
                # ]
                # # Find the 5 points with the smallest absolute difference from correct_mean
                # positive_indices = (pos_subset[mean_name] - correct_mean).abs().nsmallest(5).index


                # # --- Negative Indices ---
                # # Filter first: Correctness != 2 AND Strict Format == 0.5
                # neg_subset = df_airl[
                #     (df_airl["correctness_reward_func"] != 2) #& 
                #     #(df_airl["strict_format_reward_func"] == 0.5)
                # ]
                # # Find the 5 points with the smallest absolute difference from wrong_mean
                # negative_indices = (neg_subset[mean_name] - wrong_mean).abs().nsmallest(5).index
                all_indices = np.concatenate([positive_indices, negative_indices ])
                df_airl["reward_model_max"] = df_airl["reward_model_standard"].apply(lambda x: max(x))
                df_airl["reward_model_min"] = df_airl["reward_model_standard"].apply(lambda x: min(x))
                max_value = df_airl.loc[all_indices, "reward_model_max"].max()
                min_value = df_airl.loc[all_indices, "reward_model_min"].min()
            
                for i, idx in enumerate(positive_indices):
                    tokens = df_airl.loc[idx, "response_token"]
                    scores = df_airl.loc[idx, "reward_model_standard"]
                    question = df_airl.loc[idx, "prompt"][1]["content"]
                    make_text_reward_image(
                        tokens,
                        scores,
                        out_dir / f"dense_rewards_{postfix}/true_{i}.pdf",
                        cmap_name=colour_map,
                        prompt_text=question,
                        font_size=18,
                        dpi=300,
                        max_width_px=4000,
                        max_val=max_value,
                        min_val=min_value
                    )
            
                for i, idx in enumerate(negative_indices):
                    tokens = df_airl.loc[idx, "response_token"]
                    scores = df_airl.loc[idx, "reward_model_standard"]
                    question = df_airl.loc[idx, "prompt"][1]["content"]
                    make_text_reward_image(
                        tokens,
                        scores,
                        out_dir / f"dense_rewards_{postfix}/wrong_{i}.pdf",
                        cmap_name=colour_map,
                        prompt_text=question,
                        font_size=18,
                        dpi=300,
                        max_width_px=4000,
                        max_val=max_value,
                        min_val=min_value
                    )
                

    # return path for reference
    return Path(out_dir)
