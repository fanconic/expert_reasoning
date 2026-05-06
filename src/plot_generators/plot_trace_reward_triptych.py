import argparse
import json
import math
import textwrap
from pathlib import Path
from typing import Any

if "MPLCONFIGDIR" not in __import__("os").environ:
    __import__("os").environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer

try:
    # Reuses project plotting style setup.
    from src.plot_generators import plot_helpers as _plot_helpers  # noqa: F401
except Exception:
    _plot_helpers = None

try:
    from src.plot_generators.token_viz import make_text_reward_image
except Exception:
    make_text_reward_image = None


DEFAULT_ROOT = Path("outputs") / "localisation"
DEFAULT_RUNS = {
    "qwen4b": "qwen4b_full_localisation_expert",
    "qwen7b": "qwen7b_full_localisation_expert",
    "llama8b": "llama8b_full_localisation_expert",
}


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r") as f:
        for line in f:
            raw = line.strip()
            if raw:
                rows.append(json.loads(raw))
    return rows


def _key(row: dict[str, Any]) -> tuple[int, int, int]:
    return (int(row["prompt_idx"]), int(row["severity"]), int(row["variant_idx"]))


def _prompt_text(prompt_msgs: list[dict]) -> str:
    for msg in reversed(prompt_msgs):
        if msg.get("role") == "user":
            return str(msg.get("content", ""))
    return str(prompt_msgs[-1].get("content", "")) if prompt_msgs else ""


def _format_metrics(row: dict[str, Any]) -> str:
    hit = row.get("hit_at", {}) or {}
    hitn = row.get("hit_at_norm", {}) or {}
    margin = row.get("margin", float("nan"))
    gap = row.get("localization_gap", float("nan"))
    out = (
        f"margin={margin:.3f}, gap={gap:.3f}, "
        f"H@1={float(hit.get('1', float('nan'))):.3f}, "
        f"NH@1={float(hitn.get('1', float('nan'))):.3f}"
    )
    ldm_norm = row.get("ldm_norm", None)
    onset = row.get("onset_lag", None)
    if isinstance(ldm_norm, (int, float)) and math.isfinite(float(ldm_norm)):
        out += f", LDMn={float(ldm_norm):.3f}"
    if isinstance(onset, (int, float)) and math.isfinite(float(onset)):
        out += f", lag={float(onset):.2f}"
    return out


def _truncate_wrapped(text: str, width: int, max_chars: int) -> str:
    clipped = text if len(text) <= max_chars else text[: max_chars - 3] + "..."
    return textwrap.fill(clipped, width=width)


def _load_run_bundle(root: Path, run_name: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run_dir = root / run_name
    cfg = _load_json(run_dir / "run_config.json")
    rows = _load_jsonl(run_dir / "pair_details.jsonl")
    return cfg, rows


def _fit_positionwise_normalizer(
    clean_seqs: list[list[float]],
    n_bins: int,
    mode: str,
    eps: float = 1e-6,
) -> dict[str, Any]:
    n_bins = max(2, int(n_bins))
    per_bin: list[list[float]] = [[] for _ in range(n_bins)]
    global_vals: list[float] = []

    for seq in clean_seqs:
        arr = np.asarray(seq, dtype=float)
        if arr.size == 0:
            continue
        n = arr.shape[0]
        for i, v in enumerate(arr):
            u = (i + 0.5) / float(n)
            b = min(n_bins - 1, int(u * n_bins))
            per_bin[b].append(float(v))
            global_vals.append(float(v))

    if not global_vals:
        return {
            "n_bins": n_bins,
            "mu": np.zeros(n_bins, dtype=float),
            "sigma": np.ones(n_bins, dtype=float),
            "eps": float(eps),
        }

    g = np.asarray(global_vals, dtype=float)
    if mode == "robust":
        global_mu = float(np.median(g))
        global_sigma = float(1.4826 * np.median(np.abs(g - global_mu)))
    else:
        global_mu = float(np.mean(g))
        global_sigma = float(np.std(g))
    if (not np.isfinite(global_sigma)) or global_sigma < eps:
        global_sigma = 1.0

    mu = np.zeros(n_bins, dtype=float)
    sigma = np.ones(n_bins, dtype=float)
    for b in range(n_bins):
        vals = np.asarray(per_bin[b], dtype=float)
        if vals.size == 0:
            mu[b] = global_mu
            sigma[b] = global_sigma
            continue
        if mode == "robust":
            m = float(np.median(vals))
            s = float(1.4826 * np.median(np.abs(vals - m)))
        else:
            m = float(np.mean(vals))
            s = float(np.std(vals))
        if (not np.isfinite(s)) or s < eps:
            s = global_sigma
        mu[b] = m
        sigma[b] = s
    return {"n_bins": n_bins, "mu": mu, "sigma": sigma, "eps": float(eps)}


def _apply_positionwise_normalizer(
    seq: list[float],
    normalizer: dict[str, Any],
    smooth_window: int,
) -> np.ndarray:
    arr = np.asarray(seq, dtype=float)
    if arr.size == 0:
        return arr

    n_bins = int(normalizer["n_bins"])
    mu = np.asarray(normalizer["mu"], dtype=float)
    sigma = np.asarray(normalizer["sigma"], dtype=float)
    eps = float(normalizer.get("eps", 1e-6))

    n = arr.shape[0]
    out = np.zeros(n, dtype=float)
    for i in range(n):
        u = (i + 0.5) / float(n)
        b = min(n_bins - 1, int(u * n_bins))
        out[i] = (arr[i] - mu[b]) / max(float(sigma[b]), eps)

    w = int(max(1, smooth_window))
    if w > 1 and out.shape[0] > 1:
        kernel = np.ones(w, dtype=float) / float(w)
        out = np.convolve(out, kernel, mode="same")
    return out


def _find_common_key(
    rows_by_model: dict[str, dict[tuple[int, int, int], dict[str, Any]]],
    prompt_idx: int,
    severity: int,
    variant_idx: int,
) -> tuple[int, int, int]:
    key_sets = [set(d.keys()) for d in rows_by_model.values()]
    common = set.intersection(*key_sets)
    if not common:
        raise ValueError("No common (prompt_idx, severity, variant_idx) keys found across models.")

    target = (prompt_idx, severity, variant_idx)
    if target in common:
        return target

    # Fallback: first by sorted order for deterministic behavior.
    return sorted(common)[0]


def _token_viz(
    output_dir: Path,
    model_key: str,
    kind: str,
    text: str,
    scores: list[float],
    prompt_text: str,
    checkpoint_dir: str,
    title_suffix: str,
) -> Path | None:
    if make_text_reward_image is None:
        return None

    tok_dir = Path(checkpoint_dir) / "reward_model"
    if not tok_dir.exists():
        tok_dir = Path(checkpoint_dir)

    tokenizer = AutoTokenizer.from_pretrained(str(tok_dir), local_files_only=True)
    eos = tokenizer.eos_token or ""
    ids = tokenizer(text + eos, add_special_tokens=False)["input_ids"]
    toks = tokenizer.convert_ids_to_tokens(ids)

    n = min(len(toks), len(scores))
    if n == 0:
        return None

    out_path = output_dir / f"{model_key}_{kind}_token_rewards.png"
    make_text_reward_image(
        tokens=toks[:n],
        scores=scores[:n],
        out_path=str(out_path),
        prompt_text=prompt_text,
        title=f"{model_key} {kind} | {title_suffix}",
        cmap_name="RdBu",
        max_width_px=1400,
        font_size=11,
        show_colorbar=True,
    )
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize clean/corrupted trace and per-token reward trajectories "
            "side-by-side for qwen4b, qwen7b, llama8b from localisation pair files."
        )
    )
    parser.add_argument("--root-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--qwen4b-run", type=str, default=DEFAULT_RUNS["qwen4b"])
    parser.add_argument("--qwen7b-run", type=str, default=DEFAULT_RUNS["qwen7b"])
    parser.add_argument("--llama8b-run", type=str, default=DEFAULT_RUNS["llama8b"])
    parser.add_argument("--prompt-idx", type=int, default=0)
    parser.add_argument("--severity", type=int, default=5)
    parser.add_argument("--variant-idx", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "localisation" / "trace_triptych",
    )
    parser.add_argument("--prefix", type=str, default="trace_reward_triptych")
    parser.add_argument("--text-wrap-width", type=int, default=90)
    parser.add_argument("--text-max-chars", type=int, default=1400)
    parser.add_argument(
        "--trace-transform",
        type=str,
        default="raw",
        choices=["raw", "zscore_smooth"],
        help="How to render traces: raw rewards or local-standardized+smoothed rewards.",
    )
    parser.add_argument(
        "--zscore-bins",
        type=int,
        default=20,
        help="Relative-position bins for local standardization (when --trace-transform zscore_smooth).",
    )
    parser.add_argument(
        "--zscore-mode",
        type=str,
        default="robust",
        choices=["robust", "standard"],
        help="Per-bin stats for local standardization.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=3,
        help="Centered moving-average window for smoothed normalized traces.",
    )
    parser.add_argument(
        "--skip-token-viz",
        action="store_true",
        help="Skip detailed token-text heatmaps per model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_map = {
        "qwen4b": args.qwen4b_run,
        "qwen7b": args.qwen7b_run,
        "llama8b": args.llama8b_run,
    }

    cfgs: dict[str, dict[str, Any]] = {}
    rows_by_model: dict[str, dict[tuple[int, int, int], dict[str, Any]]] = {}
    rows_list_by_model: dict[str, list[dict[str, Any]]] = {}

    for model_key, run_name in run_map.items():
        cfg, rows = _load_run_bundle(args.root_dir, run_name)
        cfgs[model_key] = cfg
        rows_by_model[model_key] = {_key(r): r for r in rows}
        rows_list_by_model[model_key] = rows

    normalizer_by_model: dict[str, dict[str, Any]] = {}
    if args.trace_transform == "zscore_smooth":
        for model_key in run_map:
            clean_refs: list[list[float]] = []
            for row in rows_list_by_model[model_key]:
                seq = row.get("clean_score_seq", []) or []
                if isinstance(seq, list) and seq:
                    clean_refs.append([float(v) for v in seq])
            normalizer_by_model[model_key] = _fit_positionwise_normalizer(
                clean_seqs=clean_refs,
                n_bins=args.zscore_bins,
                mode=args.zscore_mode,
            )

    chosen_key = _find_common_key(
        rows_by_model=rows_by_model,
        prompt_idx=args.prompt_idx,
        severity=args.severity,
        variant_idx=args.variant_idx,
    )

    selected = {m: rows_by_model[m][chosen_key] for m in run_map}
    prompt_idx, severity, variant_idx = chosen_key

    # Use qwen7b text as canonical display text (all should normally match).
    canonical = selected["qwen7b"]
    clean_text = canonical["clean_text"]
    pert_text = canonical["pert_text"]
    prompt_text = _prompt_text(canonical.get("prompt", []))

    # Figure: top text panes + bottom triptych.
    fig, axes = plt.subplots(1, 3, figsize=(22, 8), sharey=True)
    fig.subplots_adjust(top=0.58, left=0.05, right=0.99, bottom=0.08, wspace=0.12)

    transformed_selected: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    all_vals = []
    for model_key, row in selected.items():
        clean_raw = [float(v) for v in (row.get("clean_score_seq", []) or [])]
        pert_raw = [float(v) for v in (row.get("pert_score_seq", []) or [])]
        if args.trace_transform == "zscore_smooth":
            normalizer = normalizer_by_model[model_key]
            clean = _apply_positionwise_normalizer(clean_raw, normalizer, args.smooth_window)
            pert = _apply_positionwise_normalizer(pert_raw, normalizer, args.smooth_window)
        else:
            clean = np.asarray(clean_raw, dtype=float)
            pert = np.asarray(pert_raw, dtype=float)
        transformed_selected[model_key] = (clean, pert)
        all_vals.extend(clean.tolist())
        all_vals.extend(pert.tolist())
    if all_vals:
        y_min = float(np.nanpercentile(all_vals, 1))
        y_max = float(np.nanpercentile(all_vals, 99))
        if y_min >= y_max:
            y_min = float(np.nanmin(all_vals))
            y_max = float(np.nanmax(all_vals))
    else:
        y_min, y_max = -5.0, 5.0
    pad = 0.08 * (y_max - y_min + 1e-8)
    y_min -= pad
    y_max += pad

    for ax, model_key in zip(axes, ["qwen4b", "qwen7b", "llama8b"]):
        row = selected[model_key]
        clean, pert = transformed_selected[model_key]
        changed = [int(p) for p in (row.get("changed_token_positions", []) or [])]

        x_clean = np.arange(len(clean))
        x_pert = np.arange(len(pert))

        ax.plot(x_clean, clean, color="C0", linewidth=1.6, label="Clean")
        ax.plot(x_pert, pert, color="C1", linewidth=1.6, label="Corrupted")
        for p in changed:
            ax.axvline(p, color="crimson", alpha=0.15, linewidth=1.0)

        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("Token index")
        ax.set_title(
            f"{model_key}\n{_format_metrics(row)}",
            fontsize=10,
        )
        ax.grid(alpha=0.22, linestyle="--", linewidth=0.6)

    axes[0].set_ylabel("Normalized Reward (z)" if args.trace_transform == "zscore_smooth" else "Reward")
    axes[0].legend(loc="upper right", fontsize=9, frameon=True)

    clean_block = _truncate_wrapped(clean_text, args.text_wrap_width, args.text_max_chars)
    pert_block = _truncate_wrapped(pert_text, args.text_wrap_width, args.text_max_chars)
    prompt_block = _truncate_wrapped(prompt_text, args.text_wrap_width, min(500, args.text_max_chars))

    fig.text(
        0.02,
        0.98,
        "Prompt:\n" + prompt_block,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f6f8fa", "edgecolor": "#d0d7de"},
    )
    fig.text(
        0.35,
        0.98,
        "Clean trace:\n" + clean_block,
        va="top",
        ha="left",
        fontsize=8.7,
        family="monospace",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f8fff8", "edgecolor": "#9dd89d"},
    )
    fig.text(
        0.67,
        0.98,
        "Corrupted trace:\n" + pert_block,
        va="top",
        ha="left",
        fontsize=8.7,
        family="monospace",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#fff8f8", "edgecolor": "#e2a8a8"},
    )

    fig.suptitle(
        f"Token Reward Comparison [{args.trace_transform}] | "
        f"key=(prompt={prompt_idx}, severity={severity}, variant={variant_idx})",
        y=0.62,
        fontsize=13,
    )

    stem = f"{args.prefix}_p{prompt_idx}_s{severity}_v{variant_idx}"
    out_png = args.output_dir / f"{stem}.png"
    out_pdf = args.output_dir / f"{stem}.pdf"
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Save raw text snapshot for easier manual inspection.
    text_dump = args.output_dir / f"{stem}_trace.txt"
    with open(text_dump, "w") as f:
        f.write(f"prompt_idx={prompt_idx}, severity={severity}, variant_idx={variant_idx}\n\n")
        f.write("=== PROMPT ===\n")
        f.write(prompt_text + "\n\n")
        f.write("=== CLEAN ===\n")
        f.write(clean_text + "\n\n")
        f.write("=== CORRUPTED ===\n")
        f.write(pert_text + "\n")

    token_viz_paths = []
    if not args.skip_token_viz:
        title_suffix = f"p={prompt_idx}, s={severity}, v={variant_idx}"
        for model_key in ["qwen4b", "qwen7b", "llama8b"]:
            row = selected[model_key]
            checkpoint_dir = (
                cfgs[model_key].get("args", {}).get("checkpoint_dir")
                or cfgs[model_key].get("model_name", "")
            )
            if args.trace_transform == "zscore_smooth":
                clean_scores = transformed_selected[model_key][0].tolist()
                pert_scores = transformed_selected[model_key][1].tolist()
            else:
                clean_scores = [float(v) for v in (row.get("clean_score_seq", []) or [])]
                pert_scores = [float(v) for v in (row.get("pert_score_seq", []) or [])]

            p_clean = _token_viz(
                output_dir=args.output_dir,
                model_key=model_key,
                kind="clean",
                text=row.get("clean_text", ""),
                scores=clean_scores,
                prompt_text=prompt_text,
                checkpoint_dir=checkpoint_dir,
                title_suffix=title_suffix,
            )
            p_pert = _token_viz(
                output_dir=args.output_dir,
                model_key=model_key,
                kind="corrupt",
                text=row.get("pert_text", ""),
                scores=pert_scores,
                prompt_text=prompt_text,
                checkpoint_dir=checkpoint_dir,
                title_suffix=title_suffix,
            )
            if p_clean is not None:
                token_viz_paths.append(p_clean)
            if p_pert is not None:
                token_viz_paths.append(p_pert)

    meta_out = args.output_dir / f"{stem}_meta.json"
    with open(meta_out, "w") as f:
        json.dump(
            {
                "chosen_key": {
                    "prompt_idx": prompt_idx,
                    "severity": severity,
                    "variant_idx": variant_idx,
                },
                "run_map": run_map,
                "trace_transform": args.trace_transform,
                "transform_params": {
                    "zscore_bins": args.zscore_bins,
                    "zscore_mode": args.zscore_mode,
                    "smooth_window": args.smooth_window,
                },
                "output_files": {
                    "comparison_png": str(out_png),
                    "comparison_pdf": str(out_pdf),
                    "trace_text": str(text_dump),
                    "token_viz": [str(p) for p in token_viz_paths],
                },
                "metrics": {
                    model_key: {
                        "margin": selected[model_key].get("margin"),
                        "gap": selected[model_key].get("localization_gap"),
                        "hit_at": selected[model_key].get("hit_at"),
                        "hit_at_norm": selected[model_key].get("hit_at_norm"),
                    }
                    for model_key in ["qwen4b", "qwen7b", "llama8b"]
                },
            },
            f,
            indent=2,
        )

    print(f"Chosen key: prompt_idx={prompt_idx}, severity={severity}, variant_idx={variant_idx}")
    print(f"Saved comparison plot: {out_png}")
    print(f"Saved comparison plot: {out_pdf}")
    print(f"Saved trace text:      {text_dump}")
    if token_viz_paths:
        print("Saved token-reward visualizations:")
        for p in token_viz_paths:
            print(f"  - {p}")
    print(f"Saved metadata:        {meta_out}")


if __name__ == "__main__":
    main()
