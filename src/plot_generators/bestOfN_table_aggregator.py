import os
import re
from pathlib import Path
import pandas as pd

try:
    from src.plot_generators.run_paths import resolve_run_dir
except ModuleNotFoundError:
    from run_paths import resolve_run_dir

# ================= CONFIGURATION =================

ROOT_DIR = os.path.join("figures", "answer_only")

# Rows: The Methods
ALGO_ORDER = [
    ("sparse", "\\textit{Sparse}"),
    # ("partial", "\\textit{Step-wise}"),
    ("partial_fixed", "\\textit{Interval}"),
    ("full", "\\textit{Dense}"),
    # ('ovr', '\\textit{Step-wise + OVR}')
]

# Backbones (Grouped rows)
MODEL_ORDER = [
 #   ("qwen3b", r"\texttt{Qwen2.5-3B}"),
 #   ("llama3b", r"\texttt{Llama3.2-3B}"),
    ("qwen7b", r"\texttt{Qwen2.5-7B}"),
    ("llama8b", r"\texttt{Llama3.1-8B}"),
    ("qwen4b", r"\texttt{Qwen3-4B}"),
]

DATASETS = ["math", "mmlu", "medicine"]
VALUE_PATTERN = r"(\d+\.\d+\s*\[\s*\d+\.\d+,\s*\d+\.\d+\s*\])"


def extract_reranking_p1(filepath):
    if not os.path.exists(filepath):
        return {"Random": None, "Reward": None}

    with open(filepath, "r") as f:
        content = f.read().replace("\n", " ")

    rows = content.split(r"\\")
    results = {"Random": None, "Reward": None}

    for row in rows:
        if "Random Reranking" in row:
            matches = re.findall(VALUE_PATTERN, row)
            if matches:
                results["Random"] = matches[0]
        elif "Reasoning Reranking" in row:
            matches = re.findall(VALUE_PATTERN, row)
            if matches:
                results["Reward"] = matches[0]

    return results


def extract_pass1_from_table(filepath, method_label="AIRL"):
    """Extract pass@1 value string from pass_at_k_table.txt for the selected method."""
    if not os.path.exists(filepath):
        return None

    with open(filepath, "r") as f:
        content = f.read().replace("\n", " ")

    rows = content.split(r"\\")
    for row in rows:
        if method_label not in row:
            continue
        matches = re.findall(VALUE_PATTERN, row)
        if matches:
            return matches[0]
    return None


def get_mean(val_str):
    if not val_str:
        return -100.0
    try:
        clean = (
            val_str.replace(r"\textbf{", "")
            .replace(r"\underline{", "")
            .replace("}", "")
        )
        return float(clean.split()[0]) * 100
    except:
        return -100.0


def format_cell(val_str):
    """
    Formats: 79 [76, 81] as mean ± half-width (percentage points).
    """
    if not val_str:
        return "-"

    parts = val_str.split(" ", 1)
    if len(parts) != 2:
        return val_str

    mean_str, interval = parts[0], parts[1]

    # 1. Convert Mean to Percentage (No Decimals)
    try:
        clean_mean = (
            mean_str.replace(r"\textbf{", "")
            .replace(r"\underline{", "")
            .replace("}", "")
        )
        mean_val = float(clean_mean) * 100
        mean_disp = f"{mean_val:.1f}"
    except:
        mean_val = 100.0
        mean_disp = "100"

    # 2. Convert 95% CI to half-width in percentage points
    try:
        clean_ci = interval.strip("[]")
        low, high = clean_ci.split(",")
        low_pct = float(low.strip()) * 100
        high_pct = float(high.strip()) * 100
        ci_disp = f"$\\pm$ {(high_pct - low_pct) / 2.0:.1f}"
    except:
        ci_disp = interval

    # 3. Mode Collapse Check (< 20%)
    if mean_val < 15.0:
        return f"\\textcolor{{gray}}{{{mean_disp}$^*$ \\tiny {ci_disp}}}"

    # Inline format: Mean ± CI half-width
    return f"{mean_disp} {{\\scriptsize\\color{{gray}}{ci_disp}}}"


def format_delta(base_str, rew_str):
    if not base_str or not rew_str:
        return "-"

    try:
        # Parse baseline (Pass@1 from pass_at_k_table.txt)
        base_clean = base_str.split()[0].replace(r"\textbf{", "").replace("}", "")
        base_val = float(base_clean) * 100

        # Parse Reward
        rew_clean = rew_str.split()[0].replace(r"\textbf{", "").replace("}", "")
        rew_val = float(rew_clean) * 100

        delta = rew_val - base_val

        # Format: +2
        d_text = f"{delta:+.1f}"

        if delta > 0:
            return f"\\textbf{{\\textcolor{{insightteal}}{{($\\uparrow$ {d_text})}}}}"
        elif delta < 0:
            return f"\\textcolor{{purple}}{{($\\downarrow$ {d_text})}}"
        else:
            return f"({d_text})"
    except:
        return "-"


def main():
    # Data Storage
    data = {m: {a: {} for a, _ in ALGO_ORDER} for m, _ in MODEL_ORDER}

    # --- Step 1: Extraction ---
    for dataset in DATASETS:
        ds_path = Path(ROOT_DIR) / dataset
        if not ds_path.exists():
            continue

        for model_key, _ in MODEL_ORDER:
            for algo_key, _ in ALGO_ORDER:
                folder = f"{model_key}_{algo_key}"
                run_dir = resolve_run_dir(
                    ds_path, folder, required_file="pass_at_k_table_reranking_16.txt"
                )
                fpath = (
                    run_dir / "pass_at_k_table_reranking_16.txt"
                    if run_dir is not None
                    else ds_path / folder / "pass_at_k_table_reranking_16.txt"
                )
                p1_path = (
                    run_dir / "pass_at_k_table.txt"
                    if run_dir is not None
                    else ds_path / folder / "pass_at_k_table.txt"
                )

                res = extract_reranking_p1(fpath)
                pass1_airl = extract_pass1_from_table(p1_path, method_label="AIRL")
                if pass1_airl is not None:
                    res["Random"] = pass1_airl
                data[model_key][algo_key][dataset] = res

    # --- Step 2: Generate LaTeX ---
    latex = []
    latex.append(
        r"% Requires \usepackage{booktabs}, \usepackage{xcolor}, \usepackage{multirow}, \usepackage{arydshln}"
    )
    latex.append(r"\begin{table*}[h!]")
    latex.append(r"\scriptsize")
    latex.append(r"\renewcommand{\arraystretch}{1.16}")
    latex.append(r"\setlength{\tabcolsep}{3.8pt}")
    latex.append(r"\centering")
    latex.append(r"\resizebox{\textwidth}{!}{%")

    # Columns: Method | Granularity | GSM8K(3) | MMLU-Pro(3) | MedReason(3)
    latex.append(r"\begin{tabular}{l l c c c c c c c c c}")
    latex.append(r"\toprule")

    latex.append(
        r"\textbf{Method} & \textbf{Granularity} & \multicolumn{3}{c}{\textbf{\textsc{GSM8K}}} & \multicolumn{3}{c}{\textbf{\textsc{MMLU-Pro}}} & \multicolumn{3}{c}{\textbf{\textsc{MedReason}}} \\"
    )
    latex.append(r"\cmidrule(lr){3-5} \cmidrule(lr){6-8} \cmidrule(lr){9-11}")
    latex.append(
        r"& & \textbf{Random} & \textbf{Reward} & $\mathbf{\Delta}$ & \textbf{Random} & \textbf{Reward} & $\mathbf{\Delta}$ & \textbf{Random} & \textbf{Reward} & $\mathbf{\Delta}$ \\"
    )
    latex.append(r"\midrule")

    for model_key, model_name in MODEL_ORDER:
        first_row = True
        num_algos = len(ALGO_ORDER)

        for algo_key, algo_name in ALGO_ORDER:
            row_cells = []

            # GSM8K Data
            gsm_entry = data[model_key][algo_key]["math"]
            gsm_rand = format_cell(gsm_entry["Random"])
            gsm_rew = format_cell(gsm_entry["Reward"])
            gsm_delta = format_delta(gsm_entry["Random"], gsm_entry["Reward"])
            row_cells.extend([gsm_rand, gsm_rew, gsm_delta])

            # MMLU-Pro Data
            mmlu_entry = data[model_key][algo_key]["mmlu"]
            mmlu_rand = format_cell(mmlu_entry["Random"])
            mmlu_rew = format_cell(mmlu_entry["Reward"])
            mmlu_delta = format_delta(mmlu_entry["Random"], mmlu_entry["Reward"])
            row_cells.extend([mmlu_rand, mmlu_rew, mmlu_delta])

            # MedReason Data
            med_entry = data[model_key][algo_key]["medicine"]
            med_rand = format_cell(med_entry["Random"])
            med_rew = format_cell(med_entry["Reward"])
            med_delta = format_delta(med_entry["Random"], med_entry["Reward"])
            row_cells.extend([med_rand, med_rew, med_delta])

            # Method Label (Multirow)
            if first_row:
                model_col = f"\\multirow{{{num_algos}}}{{*}}{{{model_name}}}"
            else:
                model_col = ""

            # Build Row
            latex.append(
                f"{model_col} & {algo_name} & " + " & ".join(row_cells) + r" \\"
            )
            first_row = False

        # Rule between models
        if model_key != MODEL_ORDER[-1][0]:
            latex.append(r"\cdashline{1-11}[0.5pt/1.8pt]")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}}")
    latex.append(r"\vspace{0.6em}")
    latex.append(
        r"\caption{\textbf{Best-of-16 reranking performance (\%).} Reward-guided reranking is compared against random selection from the same sample set (baseline pass@1 in expectation). $\Delta$ reports percentage-point change (reward minus random).}"
    )
    latex.append(r"\label{tab:reranking_main}")
    latex.append(r"\end{table*}")

    output_file = os.path.join(ROOT_DIR, "reranking_results_main.txt")
    with open(output_file, "w") as f:
        f.write("\n".join(latex))

    print(f"Success! Created '{output_file}'")


if __name__ == "__main__":
    main()
