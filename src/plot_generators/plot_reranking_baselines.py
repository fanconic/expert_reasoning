import argparse
import os
import re
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib
import matplotlib.pyplot as plt

try:
    import scienceplots  # noqa: F401
except Exception:
    scienceplots = None

try:
    plt.style.use("bright")
except OSError:
    # Fallback if scienceplots style is unavailable in the environment.
    pass
plt.rcParams["font.family"] = "sans-serif"

try:
    from src.plot_generators.run_paths import resolve_run_dir
except ModuleNotFoundError:
    from run_paths import resolve_run_dir

# ================= CONFIGURATION DEFAULTS =================

DEFAULT_ROOT_DIR = Path("figures") / "sft_reranking_temp05"

MODEL_ORDER = [
    ("qwen3b", r"\texttt{Qwen2.5-3B}"),
    ("llama3b", r"\texttt{Llama3.2-3B}"),
    ("qwen4b", r"\texttt{Qwen2.5-4B}"),
    ("qwen7b", r"\texttt{Qwen2.5-7B}"),
    ("llama8b", r"\texttt{Llama3.1-8B}"),
]

VARIANT_ORDER = [
    ("sparse", r"\textit{Sparse}"),
    ("partial", r"\textit{Step-wise}"),
    ("partial_fixed", r"\textit{Interval}"),
    ("full", r"\textit{Dense}"),
]

DATASET_ORDER = [
    ("math", "GSM8K"),
    ("medicine", "MedReason"),
    ("mmlu", "MMLU-Pro"),
]

DEFAULT_NUM_GENERATIONS = [2, 3, 5, 8, 16]
VALUE_PATTERN = r"(\d+\.\d+\s*\[\s*\d+\.\d+,\s*\d+\.\d+\s*\])"
METHOD_LABELS = {
    "random": "Random",
    "logp": "Log-Prob",
    "majority": "Majority",
    "reward": "Reward (Ours)",
    "weighted_majority": "Weighted Maj. (Ours)",
}


def _parse_csv(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _parse_int_csv(raw: str) -> list[int]:
    values = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    return values


def _resolve_data_root(root_dir: Path) -> Path:
    """
    Resolve where dataset folders live.

    Supported layouts:
      1) <root>/answer_only/<dataset>/...
      2) <root>/<dataset>/...
    """
    answer_only_root = root_dir / "answer_only"
    if answer_only_root.exists() and answer_only_root.is_dir():
        return answer_only_root
    return root_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build compact reranking baseline LaTeX tables with rows over Best-of-N."
        )
    )
    parser.add_argument("--root-dir", type=str, default=str(DEFAULT_ROOT_DIR))
    parser.add_argument(
        "--datasets",
        type=str,
        default=",".join([k for k, _ in DATASET_ORDER]),
        help="Comma-separated datasets (e.g. math,medicine,mmlu).",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="qwen7b",
        help="Comma-separated model keys (e.g. qwen7b,llama8b).",
    )
    parser.add_argument(
        "--variants",
        type=str,
        default="partial",
        help=(
            "Comma-separated reward-model variants (e.g. sparse,partial,partial_fixed,full). "
            "Supports legacy names ending with '_new_from_sft'."
        ),
    )
    parser.add_argument(
        "--num-generations",
        type=str,
        default="2,3,5,8,16",
        help="Comma-separated Best-of-N rows to include.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help=(
            "Output .txt file path. "
            "Default: <root-dir>/results_reranking_baselines_compact.txt"
        ),
    )
    parser.add_argument(
        "--show-variant-in-caption",
        action="store_true",
        help="Always include variant/method name in the table caption.",
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Skip generating publication-style PDF plots.",
    )
    parser.add_argument(
        "--pdf-dir",
        type=str,
        default=None,
        help=(
            "Directory where PDF plots are written. "
            "Default: <root-dir>/plots_reranking_compact"
        ),
    )
    parser.add_argument(
        "--no-full-table",
        action="store_true",
        help="Skip generating the full legacy-style LaTeX tables (all models/datasets).",
    )
    parser.add_argument(
        "--full-output-file",
        type=str,
        default=None,
        help=(
            "Path for full legacy-style LaTeX tables. "
            "Default: <root-dir>/results_reranking_baselines_all.txt"
        ),
    )
    return parser.parse_args()


def extract_reranking_p1(filepath: Path) -> dict[str, str | None]:
    results: dict[str, str | None] = {
        "Random": None,
        "Reward": None,
        "Logprobs": None,
        "Majority": None,
        "Weighted_Majority": None,
    }
    if not filepath.exists():
        return results

    content = filepath.read_text().replace("\n", " ")
    rows = content.split(r"\\")

    for row in rows:
        matches = re.findall(VALUE_PATTERN, row)
        if not matches:
            continue

        row_lower = row.lower()
        if "random" in row_lower:
            results["Random"] = matches[0]
        elif "reasoning reranking" in row_lower or "reward" in row_lower:
            results["Reward"] = matches[0]
        elif "log probability" in row_lower or "log prob" in row_lower:
            results["Logprobs"] = matches[0]
        elif "weighted majority" in row_lower:
            results["Weighted_Majority"] = matches[0]
        elif "majority voting" in row_lower or "majority" in row_lower:
            results["Majority"] = matches[0]

    return results


def extract_sft_pass1(filepath: Path) -> str | None:
    if not filepath.exists():
        return None

    content = filepath.read_text().replace("\n", " ")
    rows = content.split(r"\\")
    for row in rows:
        if "sft" not in row.lower():
            continue
        matches = re.findall(VALUE_PATTERN, row)
        if not matches:
            continue
        return matches[0]
    return None


def _parse_mean_percent(val_str: str | None) -> float | None:
    if not val_str:
        return None
    try:
        head = (
            val_str.split()[0]
            .replace(r"\textbf{", "")
            .replace(r"\underline{", "")
            .replace("}", "")
        )
        return float(head) * 100.0
    except Exception:
        return None


def _format_pct(v: float | None, decimals: int = 0) -> str:
    if v is None:
        return "-"
    return f"{v:.{decimals}f}\\%"


def _format_delta_pp(v: float | None, *, is_best: bool) -> str:
    if v is None:
        return "-"
    d = int(round(v))
    txt = f"{d:+d} pp"
    return rf"\textbf{{{txt}}}" if is_best else txt


def _format_delta_arrow(v: float | None, *, is_best: bool) -> str:
    if v is None:
        return "-"

    d = round(float(v), 1)
    # Avoid rendering -0.0
    if abs(d) < 0.05:
        d = 0.0

    if d > 0:
        core = rf"\textcolor{{insightteal}}{{($\uparrow$ +{d:.1f})}}"
    elif d < 0:
        core = rf"\textcolor{{purple}}{{($\downarrow$ {d:.1f})}}"
    else:
        core = r"(0.0)"
    return rf"\textbf{{{core}}}" if is_best else core


def _label_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _run_name_candidates(model_key: str, variant_key: str) -> list[str]:
    candidates = []
    primary = f"{model_key}_{variant_key}"
    candidates.append(primary)

    if not variant_key.endswith("_new_from_sft"):
        candidates.append(f"{model_key}_{variant_key}_new_from_sft")

    # Keep order but deduplicate.
    deduped = []
    seen = set()
    for cand in candidates:
        if cand in seen:
            continue
        seen.add(cand)
        deduped.append(cand)
    return deduped


def _resolve_reranking_file(
    dataset_dir: Path,
    model_key: str,
    variant_key: str,
    num_gen: int,
) -> Path:
    rerank_name = f"pass_at_k_table_reranking_{num_gen}.txt"
    for run_name in _run_name_candidates(model_key, variant_key):
        run_dir = resolve_run_dir(dataset_dir, run_name, required_file=rerank_name)
        if run_dir is not None:
            return run_dir / rerank_name

    # Fallback path only for diagnostics; likely missing.
    return dataset_dir / f"{model_key}_{variant_key}" / rerank_name


def _resolve_pass_table_file(
    dataset_dir: Path,
    model_key: str,
    variant_key: str,
) -> Path:
    pass_name = "pass_at_k_table.txt"
    for run_name in _run_name_candidates(model_key, variant_key):
        run_dir = resolve_run_dir(dataset_dir, run_name, required_file=pass_name)
        if run_dir is not None:
            return run_dir / pass_name

    return dataset_dir / f"{model_key}_{variant_key}" / pass_name


def _collect_rows(
    *,
    dataset_dir: Path,
    model_key: str,
    variant_key: str,
    num_generations: list[int],
) -> list[dict[str, float | int | None]]:
    sft_pass1 = _parse_mean_percent(
        extract_sft_pass1(_resolve_pass_table_file(dataset_dir, model_key, variant_key))
    )

    rows: list[dict[str, float | int | None]] = []
    for n in num_generations:
        fpath = _resolve_reranking_file(dataset_dir, model_key, variant_key, n)
        entry = extract_reranking_p1(fpath)

        # Keep Random anchored to SFT pass@1 for all N rows.
        rand = sft_pass1
        logp = _parse_mean_percent(entry["Logprobs"])
        maj = _parse_mean_percent(entry["Majority"])
        rew = _parse_mean_percent(entry["Reward"])
        wmaj = _parse_mean_percent(entry["Weighted_Majority"])

        d_log = None if rand is None or logp is None else (logp - rand)
        d_maj = None if rand is None or maj is None else (maj - rand)
        d_rew = None if rand is None or rew is None else (rew - rand)
        d_wmaj = None if rand is None or wmaj is None else (wmaj - rand)

        deltas = [d for d in (d_log, d_maj, d_rew, d_wmaj) if d is not None]
        best_delta = max(deltas) if deltas else None

        rows.append(
            {
                "n": n,
                "random": rand,
                "logp": logp,
                "majority": maj,
                "reward": rew,
                "weighted_majority": wmaj,
                "d_log": d_log,
                "d_maj": d_maj,
                "d_rew": d_rew,
                "d_wmaj": d_wmaj,
                "best": best_delta,
            }
        )

    has_any_data = False
    for row in rows:
        if any(
            row[key] is not None
            for key in ["random", "logp", "majority", "reward", "weighted_majority"]
        ):
            has_any_data = True
            break

    return rows if has_any_data else []


def _build_compact_table_from_rows(
    *,
    rows: list[dict[str, float | int | None]],
    dataset_key: str,
    dataset_label: str,
    model_key: str,
    model_label: str,
    variant_key: str,
    variant_label: str,
    show_variant_in_caption: bool,
) -> str:
    lines = []
    lines.append(r"\begin{table}[h!]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lccccc}")
    lines.append(r"\toprule")
    lines.append(
        r"$N$ & \textbf{Random} & $\Delta$ Log-Prob & $\Delta$ Majority & $\Delta$ Reward (Ours) & $\Delta$ Weighted Maj.\ (Ours) \\"
    )
    lines.append(r"\midrule")

    for row in rows:
        best = row["best"]
        d_log = _format_delta_pp(
            row["d_log"], is_best=(best is not None and row["d_log"] == best)
        )
        d_maj = _format_delta_pp(
            row["d_maj"], is_best=(best is not None and row["d_maj"] == best)
        )
        d_rew = _format_delta_pp(
            row["d_rew"], is_best=(best is not None and row["d_rew"] == best)
        )
        d_wmaj = _format_delta_pp(
            row["d_wmaj"], is_best=(best is not None and row["d_wmaj"] == best)
        )

        lines.append(
            rf"\textbf{{Best-of-{int(row['n'])}}} & {_format_pct(row['random'])} & {d_log} & {d_maj} & {d_rew} & {d_wmaj} \\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    if show_variant_in_caption:
        caption = (
            rf"Sample Size ($N$) Impact ({model_label}, {dataset_label}, {variant_label})."
        )
    else:
        caption = rf"Sample Size ($N$) Impact ({model_label}, {dataset_label})."
    lines.append(rf"\caption{{{caption}}}")
    lines.append(
        rf"\label{{tab:sample_size_{_label_key(model_key)}_{_label_key(dataset_key)}_{_label_key(variant_key)}}}"
    )
    lines.append(r"\end{table}")
    return "\n".join(lines)


def _plot_compact_pdf(
    *,
    rows: list[dict[str, float | int | None]],
    model_label: str,
    dataset_label: str,
    variant_label: str,
    show_variant_in_caption: bool,
    out_path: Path,
) -> None:
    x_vals = [int(r["n"]) for r in rows]

    prop_cycle = plt.rcParams.get("axes.prop_cycle")
    colors = prop_cycle.by_key()["color"] if prop_cycle else [None] * 5
    styles = {
        "random": {
            "color": colors[2] if len(colors) > 2 else None,
            "marker": "x",
            "linestyle": "--",
        },
        "logp": {
            "color": colors[1] if len(colors) > 1 else None,
            "marker": "x",
            "linestyle": "--",
        },
        "majority": {
            "color": colors[5] if len(colors) > 5 else None,
            "marker": "x",
            "linestyle": "--",
        },
        "reward": {
            "color": colors[0] if colors else None,
            "marker": "x",
            "linestyle": "--",
        },
        "weighted_majority": {
            "color": colors[4] if len(colors) > 4 else None,
            "marker": "x",
            "linestyle": "--",
        },
    }

    plt.figure(figsize=(7, 4))

    series_map = [
        ("random", [r["random"] for r in rows]),
        ("logp", [r["logp"] for r in rows]),
        ("majority", [r["majority"] for r in rows]),
        ("reward", [r["reward"] for r in rows]),
        ("weighted_majority", [r["weighted_majority"] for r in rows]),
    ]

    plotted_any = False
    for key, y in series_map:
        style = styles[key]
        x_plot = []
        y_plot = []
        for x, yy in zip(x_vals, y):
            if yy is None:
                continue
            x_plot.append(x)
            y_plot.append(float(yy))

        if not y_plot:
            continue

        plotted_any = True
        plt.plot(
            x_plot,
            y_plot,
            label=METHOD_LABELS[key],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=6,
            linewidth=2,
        )

    if not plotted_any:
        plt.close()
        return

    plt.xticks(x_vals)
    plt.xlabel("N")
    plt.ylabel("Best-of-N (%)")

    plt.legend()
    plt.grid()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close()


def _build_metric_cache(
    *,
    data_root: Path,
    dataset_keys: list[str],
    model_keys: list[str],
    variant_keys: list[str],
    num_generations: list[int],
) -> dict[tuple[str, str, str, int], dict[str, float | int | None]]:
    cache: dict[tuple[str, str, str, int], dict[str, float | int | None]] = {}
    for dataset_key in dataset_keys:
        dataset_dir = data_root / dataset_key
        for model_key in model_keys:
            for variant_key in variant_keys:
                rows = _collect_rows(
                    dataset_dir=dataset_dir,
                    model_key=model_key,
                    variant_key=variant_key,
                    num_generations=num_generations,
                )
                by_n = {int(r["n"]): r for r in rows}
                for n in num_generations:
                    cache[(dataset_key, model_key, variant_key, n)] = by_n.get(
                        n,
                        {
                            "n": n,
                            "random": None,
                            "logp": None,
                            "majority": None,
                            "reward": None,
                            "weighted_majority": None,
                            "d_log": None,
                            "d_maj": None,
                            "d_rew": None,
                            "d_wmaj": None,
                            "best": None,
                        },
                    )
    return cache


def _build_full_tables_latex(
    *,
    data_root: Path,
    num_generations: list[int],
) -> tuple[str, int]:
    model_keys = [k for k, _ in MODEL_ORDER]
    variant_keys = [k for k, _ in VARIANT_ORDER]
    dataset_keys = [k for k, _ in DATASET_ORDER]
    dataset_labels = dict(DATASET_ORDER)
    model_labels = dict(MODEL_ORDER)
    variant_labels = dict(VARIANT_ORDER)

    cache = _build_metric_cache(
        data_root=data_root,
        dataset_keys=dataset_keys,
        model_keys=model_keys,
        variant_keys=variant_keys,
        num_generations=num_generations,
    )

    out_tables: list[str] = [
        r"% Requires \usepackage{booktabs}, \usepackage{xcolor}, \usepackage{multirow}"
    ]
    table_count = 0

    for n in num_generations:
        lines: list[str] = []
        lines.append(f"% ==========================================")
        lines.append(f"% Full reranking baseline table for N = {n}")
        lines.append(f"% ==========================================")
        lines.append(r"\begin{table}[h!]")
        lines.append(r"\centering")
        lines.append(r"\scriptsize")
        lines.append(r"\resizebox{\textwidth}{!}{%")
        lines.append(r"\begin{tabular}{ll ccccc ccccc ccccc}")
        lines.append(r"\toprule")
        lines.append(
            r"& & \multicolumn{5}{c}{\textbf{\textsc{"
            + dataset_labels.get("math", "GSM8K")
            + r"}}} & \multicolumn{5}{c}{\textbf{\textsc{"
            + dataset_labels.get("medicine", "MedReason")
            + r"}}} & \multicolumn{5}{c}{\textbf{\textsc{"
            + dataset_labels.get("mmlu", "MMLU-Pro")
            + r"}}} \\"
        )
        lines.append(r"\cmidrule(lr){3-7} \cmidrule(lr){8-12} \cmidrule(lr){13-17}")
        metric_header = (
            r"Random & $\Delta$ Logp. & $\Delta$ Maj. & $\Delta$ Rew. & $\Delta$ W.Maj."
        )
        lines.append(
            rf"\textbf{{Backbone}} & \textbf{{Method}} & {metric_header} & {metric_header} & {metric_header} \\"
        )
        lines.append(r"\midrule")

        for m_idx, model_key in enumerate(model_keys):
            model_label = model_labels.get(model_key, model_key)
            first_row = True
            for variant_key in variant_keys:
                variant_label = variant_labels.get(variant_key, variant_key)
                row_cells: list[str] = []
                for dataset_key in dataset_keys:
                    row = cache[(dataset_key, model_key, variant_key, n)]
                    d_log = row["d_log"]
                    d_maj = row["d_maj"]
                    d_rew = row["d_rew"]
                    d_wmaj = row["d_wmaj"]
                    deltas = [
                        d for d in (d_log, d_maj, d_rew, d_wmaj) if d is not None
                    ]
                    best_delta = max(deltas) if deltas else None

                    rand_fmt = _format_pct(row["random"], decimals=1)
                    d_log_fmt = _format_delta_arrow(
                        d_log, is_best=(best_delta is not None and d_log == best_delta)
                    )
                    d_maj_fmt = _format_delta_arrow(
                        d_maj, is_best=(best_delta is not None and d_maj == best_delta)
                    )
                    d_rew_fmt = _format_delta_arrow(
                        d_rew, is_best=(best_delta is not None and d_rew == best_delta)
                    )
                    d_wmaj_fmt = _format_delta_arrow(
                        d_wmaj,
                        is_best=(best_delta is not None and d_wmaj == best_delta),
                    )
                    row_cells.extend(
                        [rand_fmt, d_log_fmt, d_maj_fmt, d_rew_fmt, d_wmaj_fmt]
                    )

                model_cell = (
                    rf"\multirow{{{len(variant_keys)}}}{{*}}{{\textbf{{{model_label}}}}}"
                    if first_row
                    else ""
                )
                lines.append(
                    f"{model_cell} & {variant_label} & " + " & ".join(row_cells) + r" \\"
                )
                first_row = False

            if m_idx < len(model_keys) - 1:
                lines.append(r"\midrule")

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}%")
        lines.append(r"}")
        lines.append(
            rf"\caption{{\textbf{{Best-of-{n} Reranking Performance \& Baselines (\%).}} "
            r"Random is SFT pass@1. Deltas are percentage-point changes for each reranker. "
            r"\textcolor{insightteal}{Blue arrows (\,$\uparrow$\,)} indicate gains and "
            r"\textcolor{purple}{purple arrows (\,$\downarrow$\,)} indicate drops.}}"
        )
        lines.append(rf"\label{{tab:reranking_baselines_full_N{n}}}")
        lines.append(r"\end{table}")
        lines.append("")

        out_tables.extend(lines)
        table_count += 1

    return "\n".join(out_tables) + ("\n" if out_tables else ""), table_count


def main() -> None:
    args = parse_args()

    root_dir = Path(args.root_dir)
    data_root = _resolve_data_root(root_dir)
    datasets = _parse_csv(args.datasets)
    models = _parse_csv(args.models)
    variants = _parse_csv(args.variants)
    show_variant_in_caption = args.show_variant_in_caption or len(variants) > 1
    num_generations = [n for n in _parse_int_csv(args.num_generations) if n > 0]
    if not num_generations:
        raise ValueError("At least one positive value is required for --num-generations")

    model_labels = dict(MODEL_ORDER)
    variant_labels = dict(VARIANT_ORDER)
    dataset_labels = dict(DATASET_ORDER)
    make_pdf = not args.no_pdf
    pdf_dir = (
        Path(args.pdf_dir) if args.pdf_dir else root_dir / "plots_reranking_compact"
    )
    make_full_table = not args.no_full_table

    output_file = (
        Path(args.output_file)
        if args.output_file
        else root_dir / "results_reranking_baselines_compact.txt"
    )
    full_output_file = (
        Path(args.full_output_file)
        if args.full_output_file
        else root_dir / "results_reranking_baselines_all.txt"
    )

    out_tables = []
    produced_tables = 0
    produced_plots = 0

    for dataset_key in datasets:
        dataset_dir = data_root / dataset_key
        dataset_label = dataset_labels.get(dataset_key, dataset_key)
        for model_key in models:
            model_label = model_labels.get(model_key, model_key)
            for variant_key in variants:
                variant_label = variant_labels.get(variant_key, variant_key)
                rows = _collect_rows(
                    dataset_dir=dataset_dir,
                    model_key=model_key,
                    variant_key=variant_key,
                    num_generations=num_generations,
                )
                if not rows:
                    continue

                table = _build_compact_table_from_rows(
                    rows=rows,
                    dataset_key=dataset_key,
                    dataset_label=dataset_label,
                    model_key=model_key,
                    model_label=model_label,
                    variant_key=variant_key,
                    variant_label=variant_label,
                    show_variant_in_caption=show_variant_in_caption,
                )
                produced_tables += 1
                out_tables.append(
                    f"% dataset={dataset_key}, model={model_key}, variant={variant_key}\n{table}"
                )

                if make_pdf:
                    pdf_name = (
                        f"sample_size_{_label_key(model_key)}_"
                        f"{_label_key(dataset_key)}_{_label_key(variant_key)}.pdf"
                    )
                    out_pdf = pdf_dir / pdf_name
                    _plot_compact_pdf(
                        rows=rows,
                        model_label=model_label,
                        dataset_label=dataset_label,
                        variant_label=variant_label,
                        show_variant_in_caption=show_variant_in_caption,
                        out_path=out_pdf,
                    )
                    if out_pdf.exists():
                        produced_plots += 1

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n\n".join(out_tables) + ("\n" if out_tables else ""))

    produced_full_tables = 0
    if make_full_table:
        full_text, produced_full_tables = _build_full_tables_latex(
            data_root=data_root,
            num_generations=num_generations,
        )
        full_output_file.parent.mkdir(parents=True, exist_ok=True)
        full_output_file.write_text(full_text)

    if make_pdf:
        print(
            f"Success! Loaded values from '{data_root}'. Created '{output_file}' with "
            f"{produced_tables} compact table(s) and {produced_plots} PDF plot(s) in '{pdf_dir}'. "
            f"Full tables: {produced_full_tables} in '{full_output_file}'."
        )
    else:
        print(
            f"Success! Loaded values from '{data_root}'. Created '{output_file}' with "
            f"{produced_tables} compact sample-size table(s). "
            f"Full tables: {produced_full_tables} in '{full_output_file}'."
        )


if __name__ == "__main__":
    main()
