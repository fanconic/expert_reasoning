"""Token-level reward visualisation helpers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch


NEWLINE_CHAR = "Ċ"
SPACE_CHAR = "Ġ"
_QWEN_SPECIAL_RE = re.compile(r"<\|.*?\|>")


def _escape_fmt(text: str) -> str:
    """Escape Matplotlib special characters in plain text labels."""
    if not text:
        return ""
    return text.replace("$", "\\$")


class TextMeasurer:
    """Helper to measure text width exactly using Matplotlib's renderer."""

    def __init__(self, font_properties, dpi):
        self.fp = font_properties
        self.dpi = dpi
        self.fig = plt.figure(dpi=dpi)
        self.fig.canvas.draw()
        self.renderer = self.fig.canvas.get_renderer()

    def get_text_width(self, text, fontsize):
        txt = plt.Text(0, 0, text, fontproperties=self.fp, fontsize=fontsize)
        txt.set_figure(self.fig)
        bbox = txt.get_window_extent(renderer=self.renderer)
        return bbox.width

    def wrap_text(self, text, fontsize, max_width_px):
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
            if not word:
                continue

            width = self.get_text_width(word, fontsize)

            if current_line and (current_width + space_w + width <= max_width_px):
                current_line.append(word)
                current_width += space_w + width
            elif not current_line:
                current_line.append(word)
                current_width = width
            else:
                lines.append(" ".join(current_line))
                current_line = [word]
                current_width = width

        if current_line:
            lines.append(" ".join(current_line))
        return [line for line in lines if line.strip()]

    def close(self):
        plt.close(self.fig)


def _analyze_token(raw_tok: str, score: float) -> dict:
    if raw_tok is None:
        return {
            "text": "",
            "score": score,
            "is_start_word": False,
            "newlines_after": 0,
            "is_special": False,
        }

    if raw_tok in ["<|im_end|>", "<|endoftext|>"]:
        return {
            "text": "",
            "score": score,
            "is_start_word": False,
            "newlines_after": 1,
            "is_special": True,
        }

    is_start_word = raw_tok.startswith(SPACE_CHAR)
    newlines_after = raw_tok.count(NEWLINE_CHAR)

    txt = raw_tok.replace(SPACE_CHAR, "").replace(NEWLINE_CHAR, "")
    txt = _QWEN_SPECIAL_RE.sub("", txt).replace("\u0120", " ").strip()
    txt = _escape_fmt(txt)

    is_special = txt.startswith("<") and txt.endswith(">")

    return {
        "text": txt,
        "score": score,
        "is_start_word": is_start_word,
        "newlines_after": newlines_after,
        "is_special": is_special,
    }


def normalise(values, mode="minmax", max_val=None, min_val=None):
    values_np = np.array(values, dtype=float)
    if values_np.size == 0:
        return values_np.tolist()
    if mode == "minmax":
        value_min, value_max = float(values_np.min()), float(values_np.max())
        if np.isclose(value_min, value_max):
            return np.zeros_like(values_np).tolist()
        return ((values_np - value_min) / (value_max - value_min)).tolist()
    if mode == "sigmoid":
        return ((1 / (1 + np.exp(-values_np))) * 2 - 1).tolist()
    if mode == "diverging":
        if max_val is None:
            value_max = float(np.max(np.abs(values_np)))
        else:
            value_max = float(max(np.abs([max_val, min_val])))
        if np.isclose(value_max, 0.0):
            return (0.5 * np.ones_like(values_np)).tolist()
        return (((values_np / value_max) + 1.0) / 2.0).tolist()
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
    min_val=None,
):
    fp = font_properties or FontProperties(size=font_size, family="DejaVu Sans")
    measurer = TextMeasurer(fp, dpi)

    values = np.array(scores)
    if max_val is None:
        abs_max = np.max(np.abs(values)) if values.size > 0 else 1.0
    else:
        abs_max = (
            max(abs(max_val), abs(min_val)) if min_val is not None else abs(max_val)
        )
    if abs_max == 0:
        abs_max = 1.0

    norm_obj = mcolors.Normalize(vmin=-abs_max, vmax=abs_max)

    norm_scores = np.array(
        normalise(scores, mode="diverging", max_val=max_val, min_val=min_val)
    )

    processed_tokens = []
    for tok, score in zip(tokens, norm_scores):
        processed_tokens.append(_analyze_token(tok, score))

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
            if not cur_row:
                gap = 0
            elif item["is_start_word"]:
                gap = gap_word
            else:
                gap = gap_subword

            pill_w = max(item["width"], min_text_w) + 2 * pad_x
            width_total = gap + pill_w

            if cur_row and (cur_x + width_total > max_width_px * 0.96):
                rows.append(cur_row)
                cur_row = []
                cur_x = 0
                gap = 0

            cur_row.append(
                {
                    "text": txt,
                    "score": item["score"],
                    "pill_w": pill_w,
                    "gap_before": gap,
                    "is_special": item["is_special"],
                }
            )
            cur_x += gap + pill_w

        if item["newlines_after"] > 0:
            if cur_row:
                rows.append(cur_row)
            cur_row = []
            cur_x = 0
            for _ in range(item["newlines_after"] - 1):
                rows.append([])

    if cur_row:
        rows.append(cur_row)

    layout_blocks = []
    content_width = max_width_px - 40

    if title:
        safe_title = _escape_fmt(title)
        title_lines = measurer.wrap_text(safe_title, font_size + 2, content_width)
        layout_blocks.append(
            (
                "text_lines",
                title_lines,
                {"size": font_size + 2, "style": "italic", "weight": "normal"},
            )
        )
        layout_blocks.append(("spacer", 20, {}))

    if prompt_text:
        layout_blocks.append(
            (
                "text_lines",
                ["Question:"],
                {"size": font_size, "style": "normal", "weight": "bold"},
            )
        )
        clean_prompt = prompt_text.replace(SPACE_CHAR, " ").replace(NEWLINE_CHAR, "\n")
        safe_prompt = _escape_fmt(clean_prompt)
        prompt_lines = measurer.wrap_text(safe_prompt, font_size, content_width)
        layout_blocks.append(
            (
                "text_lines",
                prompt_lines,
                {"size": font_size, "style": "normal", "weight": "normal"},
            )
        )
        layout_blocks.append(("spacer", 60, {}))

    layout_blocks.append(
        (
            "text_lines",
            ["Reasoning + Answer:"],
            {"size": font_size, "style": "normal", "weight": "bold"},
        )
    )
    layout_blocks.append(("spacer", 20, {}))
    layout_blocks.append(("pills", rows, {}))

    measurer.close()

    total_text_height = 0
    top_margin = 20
    bottom_margin = 20

    def get_line_height(fs):
        return fs * 1.5 * dpi / 72

    for kind, content, params in layout_blocks:
        if kind == "text_lines":
            line_h = get_line_height(params["size"])
            total_text_height += len(content) * line_h
        elif kind == "spacer":
            total_text_height += content
        elif kind == "pills":
            total_text_height += len(content) * row_height_px

    final_h_px = top_margin + total_text_height + bottom_margin

    cbar_width_px = 30 if show_colorbar else 0
    cbar_pad_px = 40 if show_colorbar else 0
    cbar_labels_margin = 80 if show_colorbar else 20

    total_fig_width_px = max_width_px + cbar_pad_px + cbar_width_px + cbar_labels_margin

    width_in = total_fig_width_px / dpi
    height_in = max(1.0, final_h_px / dpi)

    fig = plt.figure(figsize=(width_in, height_in), dpi=dpi)

    ax = plt.axes([0, 0, 1, 1])
    ax.set_xlim(0, total_fig_width_px)
    ax.set_ylim(0, final_h_px)
    ax.axis("off")

    cmap = plt.get_cmap(cmap_name)

    y = final_h_px - top_margin
    left_margin = 20

    for kind, content, params in layout_blocks:
        if kind == "text_lines":
            line_h = get_line_height(params["size"])
            for line in content:
                ax.text(
                    left_margin,
                    y,
                    line,
                    fontsize=params["size"],
                    fontstyle=params["style"],
                    fontweight=params["weight"],
                    va="top",
                    ha="left",
                    fontproperties=fp,
                )
                y -= line_h

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

                    score = pill["score"]
                    if pill["is_special"]:
                        face = (0.95, 0.95, 0.95, 1.0)
                        edge = (0.8, 0.8, 0.8, 1.0)
                        txt_col = "black"
                    else:
                        face = cmap(score)
                        lum = 0.299 * face[0] + 0.587 * face[1] + 0.114 * face[2]
                        txt_col = "white" if lum < 0.5 else "black"
                        edge = (0.9, 0.9, 0.9, 0.0)

                    y_pill = y_row_top - pill_h
                    patch = FancyBboxPatch(
                        (x, y_pill),
                        pill["pill_w"],
                        pill_h,
                        boxstyle="round,pad=0.0,rounding_size=6",
                        linewidth=1.0,
                        edgecolor=edge,
                        facecolor=face,
                    )
                    ax.add_patch(patch)

                    ax.text(
                        x + pill["pill_w"] / 2,
                        y_pill + pill_h / 2,
                        pill["text"],
                        fontsize=font_size,
                        va="center",
                        ha="center",
                        color=txt_col,
                        fontproperties=fp,
                    )
                    x += pill["pill_w"]

                y_row_top -= row_height_px
            y = y_row_top

    if show_colorbar:
        cb_left = (max_width_px + cbar_pad_px) / total_fig_width_px
        cb_width = cbar_width_px / total_fig_width_px

        cb_height = 0.8
        cb_bottom = (1.0 - cb_height) / 2.0

        cax = fig.add_axes([cb_left, cb_bottom, cb_width, cb_height])

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_obj)
        sm.set_array([])

        cbar = fig.colorbar(sm, cax=cax, orientation="vertical")
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(labelsize=font_size - 2, size=0)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
