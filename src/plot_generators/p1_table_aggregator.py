import os
import re
from pathlib import Path
import pandas as pd

try:
    from src.plot_generators.run_paths import collect_run_names, resolve_run_dir
except ModuleNotFoundError:
    from run_paths import collect_run_names, resolve_run_dir

# ================= CONFIGURATION =================

ROOT_DIR = os.path.join("figures", "answer_only")

# Row Order (Algorithms)
ALGO_ORDER = [
    ("GRPO", "Outcome Reward"),
    ("SFT", "SFT"),
    ("sparse", "Ours (\\textit{Sparse})"),
    #("partial", "Ours (\\textit{Step-wise})"),
    ("partial_fixed", "Ours (\\textit{Interval})"),
    ("full", "Ours (\\textit{Dense})"),
    # ('ovr', 'Ours (\\textit{Step-wise + OVR})')
]
MAIN_TABLE_ORDER = [
    ("SFT", "SFT"),
    ("sparse", r"\textit{Sparse}"),
    ("partial_fixed", r"\textit{Interval}"),
    ("full", r"\textit{Dense}"),
]

# Model Order (Now used for Sub-Headers)
MODEL_ORDER = [
#    ('qwen3b', r'\texttt{Qwen2.5-3B}'),
#    ('llama3b', r'\texttt{Llama3.2-3B}'),
    ('qwen7b', r'\texttt{Qwen2.5-7B}'),
    ('llama8b', r'\texttt{Llama3.1-8B}'),
    ('qwen4b', r'\texttt{Qwen3-4B}')
]

DATASET_COLUMNS = [
    ("math", r"\textbf{\textsc{GSM8K}}"),
#    ("aime_2024", r"\textbf{\textsc{AIME 2024}}"),
#    ("aime_2025", r"\textbf{\textsc{AIME 2025}}"),
    ("mmlu", r"\textbf{\textsc{MMLU-Pro}}"),
    ("medicine", r"\textbf{\textsc{MedReason}}"),
]
DATASETS = [d for d, _ in DATASET_COLUMNS]
VALUE_PATTERN = r"(\d+\.\d+\s*\[\s*\d+\.\d+,\s*\d+\.\d+\s*\])"
VARIANT_ONLY_ALGO_KEYS = {"full", "partial", "partial_fixed", "sparse"}
VARIANT_ONLY_ORDER = ["full", "partial", "partial_fixed", "sparse"]
BASELINE_ALGO_KEYS = ["GRPO", "SFT"]
MODELS_PER_TABLE = 3


def chunk_items(items, size):
    """Splits a list into contiguous chunks of max length `size`."""
    return [items[i : i + size] for i in range(0, len(items), size)]


def build_panel_caption(caption_body, model_order, panel_idx, panel_total):
    """Builds a caption with panel-specific model names."""
    model_names = ", ".join([name for _, name in model_order])
    if panel_total > 1:
        panel_note = rf" This panel reports: {model_names} (Panel {panel_idx}/{panel_total})."
    else:
        panel_note = rf" This panel reports: {model_names}."
    return rf"\caption{{{caption_body}{panel_note}}}"


def build_p1_latex_table_generic(
    formatted_data, algo_order, caption, label, model_order=None
):
    """Builds a transposed Pass@1 LaTeX table for the selected algorithms."""
    active_model_order = model_order if model_order is not None else MODEL_ORDER

    latex = []
    latex.append(r"% Requires \usepackage{booktabs}, \usepackage{xcolor}")
    latex.append(r"\begin{table*}[t]")
    latex.append(r"\centering")
    latex.append(r"\small")
    latex.append(r"\setlength{\tabcolsep}{4pt}")
    latex.append(r"\resizebox{\textwidth}{!}{%")

    num_dataset_cols = len(DATASET_COLUMNS)
    num_model_cols = len(active_model_order)
    total_columns = 1 + num_dataset_cols * num_model_cols
    latex.append(
        r"\begin{tabular}{" + " ".join(["l"] + ["c"] * (total_columns - 1)) + r"}"
    )
    latex.append(r"\toprule")

    dataset_group_header = " & ".join(
        [
            rf"\multicolumn{{{num_model_cols}}}{{c}}{{{dataset_label}}}"
            for _dataset_key, dataset_label in DATASET_COLUMNS
        ]
    )
    latex.append(r"\textbf{Method} & " + dataset_group_header + r" \\")

    cmidrules = []
    for idx, _ in enumerate(DATASET_COLUMNS):
        start = 2 + idx * num_model_cols
        end = start + num_model_cols - 1
        cmidrules.append(rf"\cmidrule(lr){{{start}-{end}}}")
    latex.append("".join(cmidrules))

    model_row = " & ".join(
        [
            rf"\textbf{{{model_name}}}"
            for _dataset_key, _dataset_label in DATASET_COLUMNS
            for _model_key, model_name in active_model_order
        ]
    )
    latex.append(r"& " + model_row + r" \\")
    latex.append(r"\midrule")

    for algo_key, algo_name in algo_order:
        row_cells = [
            formatted_data[model_key][algo_key][dataset]
            for dataset, _dataset_label in DATASET_COLUMNS
            for model_key, _model_name in active_model_order
        ]
        latex.append(f"{algo_name} & {' & '.join(row_cells)} \\\\")

        if algo_key == "GRPO":
            latex.append(r"\hdashline")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}%")
    latex.append(r"}")
    latex.append(caption)
    latex.append(label)
    latex.append(r"\end{table*}")
    return latex


def build_p1_latex_table_main(formatted_data, caption, label, model_order=None):
    """Builds the main Pass@1 LaTeX table in the Method/Granularity style."""
    active_model_order = model_order if model_order is not None else MODEL_ORDER

    latex = []
    latex.append(r"% Requires \usepackage{booktabs}, \usepackage{multirow}, \usepackage{arydshln}")
    latex.append(r"\begin{table*}[h!]")
    latex.append(r"\scriptsize")
    latex.append(r"\renewcommand{\arraystretch}{1.16}")
    latex.append(r"\setlength{\tabcolsep}{3.8pt}")
    latex.append(r"\centering")
    latex.append(r"\resizebox{\textwidth}{!}{%")

    num_dataset_cols = len(DATASET_COLUMNS)
    num_model_cols = len(active_model_order)
    total_columns = 2 + num_dataset_cols * num_model_cols
    latex.append(
        r"\begin{tabular}{"
        + " ".join(["l", "l"] + ["c"] * (total_columns - 2))
        + r"}"
    )
    latex.append(r"\toprule")

    dataset_group_header = " & ".join(
        [
            rf"\multicolumn{{{num_model_cols}}}{{c}}{{{dataset_label}}}"
            for _dataset_key, dataset_label in DATASET_COLUMNS
        ]
    )
    latex.append(
        r"\textbf{Method} & \textbf{Granularity} & " + dataset_group_header + r" \\"
    )

    cmidrules = []
    for idx, _ in enumerate(DATASET_COLUMNS):
        start = 3 + idx * num_model_cols
        end = start + num_model_cols - 1
        cmidrules.append(rf"\cmidrule(lr){{{start}-{end}}}")
    latex.append("".join(cmidrules))

    model_row = " & ".join(
        [
            rf"\textbf{{\tiny{model_name}}}"
            for _dataset_key, _dataset_label in DATASET_COLUMNS
            for _model_key, model_name in active_model_order
        ]
    )
    latex.append(r"& & " + model_row + r" \\")
    latex.append(r"\midrule")

    sft_cells = [
        formatted_data[model_key]["SFT"][dataset]
        for dataset, _dataset_label in DATASET_COLUMNS
        for model_key, _model_name in active_model_order
    ]
    latex.append(r"%\rowcolor{black!6}")
    latex.append(f"SFT &  & {' & '.join(sft_cells)} \\\\")
    latex.append(rf"\cdashline{{1-{total_columns}}}[0.5pt/1.8pt]")

    rairl_rows = [
        ("sparse", r"\textit{Sparse}"),
        ("partial_fixed", r"\textit{Interval}"),
        ("full", r"\textit{Dense}"),
    ]

    for idx, (algo_key, granularity) in enumerate(rairl_rows):
        row_cells = [
            formatted_data[model_key][algo_key][dataset]
            for dataset, _dataset_label in DATASET_COLUMNS
            for model_key, _model_name in active_model_order
        ]
        if idx == 0:
            latex.append(
                rf"\multirow{{{len(rairl_rows)}}}{{*}}{{R-AIRL}} & {granularity} & "
                + " & ".join(row_cells)
                + r" \\"
            )
        else:
            latex.append(
                rf"& {granularity} & " + " & ".join(row_cells) + r" \\"
            )

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}}")
    latex.append(r"\vspace{0.6em}")
    latex.append(caption)
    latex.append(label)
    latex.append(r"\end{table*}")
    return latex


def build_formatted_data(data, algo_order):
    """Formats Pass@1 cells and computes best/second within the selected methods."""
    formatted_data = {
        model_key: {
            algo_key: {dataset: "-" for dataset in DATASETS} for algo_key, _ in algo_order
        }
        for model_key, _ in MODEL_ORDER
    }

    ranking_algo_keys = [algo_key for algo_key, _ in algo_order if algo_key != "GRPO"]

    for model_key, _ in MODEL_ORDER:
        for dataset in DATASETS:
            k_idx = 0  # Pass@1

            means = []
            for algo_key in ranking_algo_keys:
                val_str = data[model_key][algo_key].get(dataset, [None] * 4)[k_idx]
                mean = get_mean(val_str)
                if mean >= 0:
                    means.append(mean)

            unique = sorted(list(set(means)), reverse=True)
            best_val = unique[0] if unique else -1.0
            second_val = unique[1] if len(unique) > 1 else -1.0

            for algo_key, _ in algo_order:
                val_str = data[model_key][algo_key].get(dataset, [None] * 4)[k_idx]
                raw_mean = get_mean(val_str)

                is_best = raw_mean == best_val and raw_mean > -1 and algo_key != "GRPO"
                is_second = (
                    raw_mean == second_val and raw_mean > -1 and algo_key != "GRPO"
                )
                is_collapsed = raw_mean < 0.20 and raw_mean > -1

                formatted_data[model_key][algo_key][dataset] = format_cell(
                    val_str, is_best, is_second, is_collapsed
                )

    return formatted_data


def extract_all_k(filepath, method_label):
    """Reads file and extracts [p1, p3, p5, p10]."""
    if not os.path.exists(filepath):
        return [None] * 4

    with open(filepath, "r") as f:
        lines = f.readlines()
    full_text = " ".join([line.strip() for line in lines])
    rows = full_text.split(r"\\")

    for row in rows:
        if method_label in row:
            if method_label == "GRPO" and "GRPO" not in row:
                continue
            if method_label == "SFT" and "SFT" not in row:
                continue
            if method_label == "AIRL" and "AIRL" not in row:
                continue

            matches = re.findall(VALUE_PATTERN, row)
            if matches:
                if len(matches) < 4:
                    matches += [None] * (4 - len(matches))
                return matches[:4]
    return [None] * 4


def get_mean(val_str):
    """Parses '0.79 [0.77, 0.81]' to float 0.79"""
    if not val_str:
        return -1.0
    try:
        clean = (
            val_str.replace(r"\textbf{", "")
            .replace(r"\underline{", "")
            .replace("}", "")
        )
        return float(clean.split()[0])
    except:
        return -1.0


def format_cell(val_str, is_best=False, is_second=False, is_collapsed=False):
    """
    Converts 0.79 [0.77, 0.81] -> 79.0 {\scriptsize\color{gray}$\pm$ 2.0}
    Handles Mode Collapse (< 20%).
    """
    if not val_str:
        return "-"

    parts = val_str.split(" ", 1)
    if len(parts) != 2:
        return val_str

    raw_mean_str, raw_interval = parts[0], parts[1]

    # --- 1. Convert Mean to Percentage ---
    try:
        clean_mean = (
            raw_mean_str.replace(r"\textbf{", "")
            .replace(r"\underline{", "")
            .replace("}", "")
        )
        mean_val = float(clean_mean) * 100
        mean_disp = f"{mean_val:.1f}"
    except:
        mean_val = 100.0
        mean_disp = "100.0"

    # --- 2. Handle Mode Collapse ---
    if is_collapsed:
        # Gray out everything, add asterisk
        # Parse 95% CI and display as half-width around the mean (in percentage points)
        try:
            clean_ci = raw_interval.strip("[]")
            low, high = clean_ci.split(",")
            low_pp, high_pp = float(low.strip()) * 100, float(high.strip()) * 100
            pm_disp = f"$\\pm$ {(high_pp - low_pp) / 2.0:.1f}"
        except:
            pm_disp = raw_interval

        return f"\\textcolor{{gray}}{{{mean_disp}$^*$ {{\\scriptsize {pm_disp}}}}}"

    # --- 3. Format Interval as 95% CI half-width (Percentage points) ---
    try:
        clean_ci = raw_interval.strip("[]")
        low, high = clean_ci.split(",")
        low_pp, high_pp = float(low.strip()) * 100, float(high.strip()) * 100
        ci_disp = f"$\\pm$ {(high_pp - low_pp) / 2.0:.1f}"
    except:
        ci_disp = raw_interval

    fmt_interval = f"{{\\scriptsize\\color{{gray}}{ci_disp}}}"

    # --- 4. Format Mean (Bold/Underline) ---
    fmt_mean = mean_disp
    if is_best:
        fmt_mean = f"\\textbf{{{mean_disp}}}"
    elif is_second:
        fmt_mean = f"\\underline{{{mean_disp}}}"

    return f"{fmt_mean} {fmt_interval}"


def main():
    data = {
        m: {a: {dataset: [None] * 4 for dataset in DATASETS} for a, _ in ALGO_ORDER}
        for m, _ in MODEL_ORDER
    }

    # --- Step 1: Extraction ---
    for dataset in DATASETS:
        ds_path = Path(ROOT_DIR) / dataset
        if not ds_path.exists():
            continue

        for model_key, _ in MODEL_ORDER:
            # Search logic for Baselines
            sft_found, grpo_found = False, False
            candidates = sorted(
                [
                    name
                    for name in collect_run_names(
                        ds_path, required_file="pass_at_k_table.txt"
                    )
                    if name.startswith(model_key)
                ]
            )
            if model_key in candidates:
                candidates.insert(0, candidates.pop(candidates.index(model_key)))

            for folder in candidates:
                if sft_found and grpo_found:
                    break
                run_dir = resolve_run_dir(
                    ds_path, folder, required_file="pass_at_k_table.txt"
                )
                if run_dir is None:
                    continue
                fpath = run_dir / "pass_at_k_table.txt"

                if not sft_found:
                    vals = extract_all_k(fpath, "SFT")
                    if any(vals):
                        data[model_key]["SFT"][dataset] = vals
                        sft_found = True
                if not grpo_found:
                    vals = extract_all_k(fpath, "GRPO")
                    if any(vals):
                        data[model_key]["GRPO"][dataset] = vals
                        grpo_found = True

            # Standard variants
            for algo_key, _ in ALGO_ORDER:
                if algo_key in ["GRPO", "SFT"]:
                    continue
                variant_folder = f"{model_key}_{algo_key}"
                variant_dir = resolve_run_dir(
                    ds_path, variant_folder, required_file="pass_at_k_table.txt"
                )
                variant_file = (
                    variant_dir / "pass_at_k_table.txt"
                    if variant_dir is not None
                    else ds_path / variant_folder / "pass_at_k_table.txt"
                )
                data[model_key][algo_key][dataset] = extract_all_k(variant_file, "AIRL")

    # --- Step 2: Analyze Rankings & Formatting ---
    formatted_data_main = build_formatted_data(data, MAIN_TABLE_ORDER)

    # --- Step 3: Generate Main LaTeX Table ---
    main_caption = (
        r"\caption{\textbf{Held-out pass@1.} \textbf{Bold} indicates the best performance, and \underline{underlined} the second best, compared between SFT and our methods in the demonstration-only setting. Values are reported as mean $\pm$ half-width of the 95\% confidence interval bootstrapped over the test set.}"
    )
    main_label = r"\label{tab:p1_results_main}"
    main_table = build_p1_latex_table_main(
        formatted_data_main,
        main_caption,
        main_label,
        model_order=MODEL_ORDER,
    )
    main_table.append("")
    model_panels = chunk_items(MODEL_ORDER, MODELS_PER_TABLE)

    algo_name_by_key = {algo_key: algo_name for algo_key, algo_name in ALGO_ORDER}
    variant_comparison_tables = []
    for variant_key in VARIANT_ONLY_ORDER:
        if variant_key not in VARIANT_ONLY_ALGO_KEYS or variant_key not in algo_name_by_key:
            continue

        subset_order = [(key, algo_name_by_key[key]) for key in BASELINE_ALGO_KEYS]
        subset_order.append((variant_key, algo_name_by_key[variant_key]))

        subset_formatted_data = build_formatted_data(data, subset_order)

        variant_readable = algo_name_by_key[variant_key]
        variant_caption_body = (
            rf"\textbf{{Pass@1 Performance (\%) for Outcome Reward + SFT + {variant_readable}.}} This table keeps verifiable reward, SFT, and a single AIRL variant. \textbf{{Bold}} indicates the best result and \underline{{underline}} indicates the second best among SFT and the AIRL variant. * symbolises adversarial mode collapse (results grayed out). Values are reported as mean $\pm$ half-width of the 95\% confidence interval (in percentage points)."
        )
        for panel_idx, model_panel in enumerate(model_panels, start=1):
            panel_caption = build_panel_caption(
                variant_caption_body, model_panel, panel_idx, len(model_panels)
            )
            if panel_idx == 1:
                variant_label = rf"\label{{tab:p1_results_{variant_key}_with_baselines}}"
            else:
                variant_label = (
                    rf"\label{{tab:p1_results_{variant_key}_with_baselines_panel{panel_idx}}}"
                )

            variant_comparison_tables.extend(
                build_p1_latex_table_generic(
                    subset_formatted_data,
                    subset_order,
                    panel_caption,
                    variant_label,
                    model_order=model_panel,
                )
            )
            variant_comparison_tables.append("")

        variant_comparison_tables.append("")

    output_path = os.path.join(ROOT_DIR, "results_p1_temp05.txt")
    with open(output_path, "w") as f:
        f.write("\n".join(main_table))

    output_path_variants = os.path.join(ROOT_DIR, "results_p1_temp05_variants_only.txt")
    with open(output_path_variants, "w") as f:
        f.write("\n".join(variant_comparison_tables).rstrip())

    print(f"Success! Created '{output_path}'")
    print(f"Success! Created '{output_path_variants}'")


if __name__ == "__main__":
    main()
