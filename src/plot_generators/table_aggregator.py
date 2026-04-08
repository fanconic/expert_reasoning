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
    ("GRPO", "Verifiable Reward"),
    ("SFT", "Supervised Fine-Tuning"),
    ("sparse_new", r"Ours (\textit{Sparse})"),
    ("partial_new", r"Ours (\textit{Step-wise})"),
    ("partial_fixed_new", r"Ours (\textit{Interval})"),
    ("full_new", r"Ours (\textit{Dense})"),
    # ('ovr_new', r'Ours (\textit{Step-wise + OVR})')
]

# Model Order (Sub-headers)
MODEL_ORDER = [
    ("qwen3b", r"\texttt{Qwen2.5-3B}"),
    ("llama3b", r"\texttt{Llama3.2-3B}"),
    ("qwen7b", r"\texttt{Qwen2.5-7B}"),
    ("llama8b", r"\texttt{Llama3.1-8B}"),
    ("qwen4b", r"\texttt{Qwen3-4B}"),
]

# Datasets
DATASETS = ["math", "medicine", "mmlu"]
VALUE_PATTERN = r"(\d+\.\d+\s*\[\s*\d+\.\d+,\s*\d+\.\d+\s*\])"


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
    Converts 0.79 [0.77, 0.81] -> 79.0 {\footnotesize\color{gray}[77.0, 81.0]}
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
        # Gray out everything, add asterisk. Parse interval to percentage first.
        try:
            clean_ci = raw_interval.strip("[]")
            low, high = clean_ci.split(",")
            ci_disp = f"[{float(low)*100:.1f}, {float(high)*100:.1f}]"
        except:
            ci_disp = raw_interval

        return (
            f"\\textcolor{{lightgray}}{{{mean_disp}$^*$ {{\\footnotesize {ci_disp}}}}}"
        )

    # --- 3. Format Interval (Percentage) ---
    try:
        clean_ci = raw_interval.strip("[]")
        low, high = clean_ci.split(",")
        ci_disp = f"[{float(low)*100:.1f}, {float(high)*100:.1f}]"
    except:
        ci_disp = raw_interval

    fmt_interval = f"{{\\footnotesize\\color{{gray}}{ci_disp}}}"

    # --- 4. Format Mean (Bold/Underline) ---
    fmt_mean = mean_disp
    if is_best:
        fmt_mean = f"\\textbf{{{mean_disp}}}"
    elif is_second:
        fmt_mean = f"\\underline{{{mean_disp}}}"

    return f"{fmt_mean} {fmt_interval}"


def main():
    data = {m: {a: {} for a, _ in ALGO_ORDER} for m, _ in MODEL_ORDER}

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

    # --- Step 2: Determine Formatting (Percentages & Highlights) ---
    # Init structure: model -> algo -> dataset -> list of 4 formatted strings
    formatted_data = {m: {a: {} for a, _ in ALGO_ORDER} for m, _ in MODEL_ORDER}

    for model_key, _ in MODEL_ORDER:
        for dataset in DATASETS:

            # Temporary storage to rebuild rows later
            # structure: col_results[k_index][algo_key] = "formatted_string"
            col_results = [{} for _ in range(4)]

            # Iterate column by column (Pass@1, 3, 5, 10)
            for k_idx in range(4):

                # Ranking logic (Skip GRPO)
                means = []
                for algo_key, _ in ALGO_ORDER:
                    if algo_key == "GRPO":
                        continue
                    val_str = data[model_key][algo_key][dataset][k_idx]
                    mean = get_mean(val_str)
                    if mean >= 0:
                        means.append(mean)

                unique = sorted(list(set(means)), reverse=True)
                best_val = unique[0] if unique else -1.0
                second_val = unique[1] if len(unique) > 1 else -1.0

                # Format each cell in this column
                for algo_key, _ in ALGO_ORDER:
                    val_str = data[model_key][algo_key][dataset][k_idx]
                    raw_mean = get_mean(val_str)

                    is_best = (
                        raw_mean == best_val and raw_mean > -1 and algo_key != "GRPO"
                    )
                    is_second = (
                        raw_mean == second_val and raw_mean > -1 and algo_key != "GRPO"
                    )
                    is_collapsed = raw_mean < 0.20 and raw_mean > -1

                    fmt_str = format_cell(val_str, is_best, is_second, is_collapsed)
                    col_results[k_idx][algo_key] = fmt_str

            # Re-organize into formatted_data[model][algo][dataset] = [p1, p3, p5, p10]
            for algo_key, _ in ALGO_ORDER:
                row_list = []
                for k_idx in range(4):
                    row_list.append(col_results[k_idx][algo_key])
                formatted_data[model_key][algo_key][dataset] = row_list

    # --- Step 3: Generate LaTeX ---
    latex = []
    latex.append(
        r"% Requires \usepackage{booktabs}, \usepackage{xcolor}, \usepackage{arydshln}"
    )
    latex.append(r"\begin{table*}[t]")
    latex.append(r"\centering")
    latex.append(r"\resizebox{\textwidth}{!}{%")
    # Layout: Method | GSM8K (4 cols) | MedReason (4 cols) | MMLU (4 cols) -> 13 columns total
    latex.append(r"\begin{tabular}{l | cccc | cccc | cccc }")
    latex.append(r"\toprule")

    # Header 1: Datasets
    latex.append(
        r"& \multicolumn{4}{c}{\textbf{\textsc{GSM8K}}} & \multicolumn{4}{c}{\textbf{\textsc{MedReason}}} & \multicolumn{4}{c}{\textbf{\textsc{MMLU-Pro}}} \\"
    )
    latex.append(r"\cmidrule(lr){2-5} \cmidrule(lr){6-9} \cmidrule(lr){10-13}")

    # Header 2: Metrics
    latex.append(
        r"\textbf{Method} & pass@1 & pass@3 & pass@5 & pass@10 & pass@1 & pass@3 & pass@5 & pass@10 & pass@1 & pass@3 & pass@5 & pass@10 \\"
    )
    latex.append(r"\midrule")

    for model_key, model_name in MODEL_ORDER:
        # === SUB-HEADER ROW ===
        latex.append(f"\\multicolumn{{13}}{{l}}{{\\textbf{{{model_name}}}}} \\\\")

        for algo_key, algo_name in ALGO_ORDER:
            # Fetch lists [p1, p3, p5, p10]
            vals_math = formatted_data[model_key][algo_key]["math"]
            vals_med = formatted_data[model_key][algo_key]["medicine"]
            vals_mmlu = formatted_data[model_key][algo_key]["mmlu"]

            # Combine
            all_vals = vals_math + vals_med + vals_mmlu
            val_str = " & ".join(all_vals)

            # Indent Algo Name
            latex.append(f"\\hspace{{1em}}{algo_name} & {val_str} \\\\")

            # Dashed Line after GRPO
            if algo_key == "GRPO":
                latex.append(
                    r"\arrayrulecolor{gray!60}\cdashline{1-9}\arrayrulecolor{black}"
                )

        if model_key != MODEL_ORDER[-1][0]:
            latex.append(r"\midrule")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}%")
    latex.append(r"}")
    latex.append(
        r"\caption{\textbf{Pass@k Performance (k=1,3,5,10).} \textbf{Bold} indicates the best performance compared between SFT and our methods. Verifiable reward is provided as a reference upper bound. * symbolises an adversarial mode collapse (results grayed out). The values inside brackets indicate the 95\% confidence interval.}"
    )
    latex.append(r"\label{tab:full_results}")
    latex.append(r"\end{table*}")

    output_path = os.path.join(ROOT_DIR, "results_table_mmlu.txt")
    with open(output_path, "w") as f:
        f.write("\n".join(latex))

    print(f"Success! Table saved to: {output_path}")


if __name__ == "__main__":
    main()
