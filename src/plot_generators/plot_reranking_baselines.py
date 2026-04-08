import os
import re
from pathlib import Path

try:
    from src.plot_generators.run_paths import resolve_run_dir
except ModuleNotFoundError:
    from run_paths import resolve_run_dir

# ================= CONFIGURATION =================

ROOT_DIR = os.path.join("figures", "answer_only")

# Rows: The Methods
ALGO_ORDER = [
    ("sparse_new_from_sft", "\\textit{Sparse}"),
    ("partial_new_from_sft", "\\textit{Step-wise}"),
    ("partial_fixed_new_from_sft", "\\textit{Interval}"),
    ("full_new_from_sft", "\\textit{Dense}"),
    # ('ovr', '\\textit{Step-wise + OVR}')
]

# Backbones (Grouped rows)
MODEL_ORDER = [
    ("qwen3b", r"\texttt{Qwen2.5-3B}"),
    ("llama3b", r"\texttt{Llama3.2-3B}"),
    ("qwen7b", r"\texttt{Qwen2.5-7B}"),
    ("llama8b", r"\texttt{Llama3.1-8B}"),
]

DATASETS = ["math", "medicine", "mmlu"]
VALUE_PATTERN = r"(\d+\.\d+\s*\[\s*\d+\.\d+,\s*\d+\.\d+\s*\])"
NUM_GENERATIONS = [2, 3, 5, 8, 16]


def extract_reranking_p1(filepath):
    results = {
        "Random": None,
        "Reward": None,
        "Logprobs": None,
        "Majority": None,
        "Weighted_Majority": None,
    }
    if not os.path.exists(filepath):
        return results

    with open(filepath, "r") as f:
        content = f.read().replace("\n", " ")

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
        # Order matters here: Check "weighted majority" before "majority"
        elif "weighted majority" in row_lower:
            results["Weighted_Majority"] = matches[0]
        elif "majority voting" in row_lower or "majority" in row_lower:
            results["Majority"] = matches[0]

    return results


def format_cell(val_str):
    """
    Formats the baseline Random score: 79 [76, 81] in percentage with no decimals.
    """
    if not val_str:
        return "-"

    parts = val_str.split(" ", 1)
    if len(parts) != 2:
        return val_str

    try:
        clean_mean = (
            parts[0]
            .replace(r"\textbf{", "")
            .replace(r"\underline{", "")
            .replace("}", "")
        )
        mean_val = float(clean_mean) * 100
        mean_disp = f"{mean_val:.0f}"
    except:
        mean_val = 100.0
        mean_disp = "100"

    try:
        clean_ci = parts[1].strip("[]")
        low, high = clean_ci.split(",")
        low_pct = float(low) * 100
        high_pct = float(high) * 100
        ci_disp = f"[{low_pct:.0f}, {high_pct:.0f}]"
    except:
        ci_disp = parts[1]

    # Mode Collapse Check (< 20%)
    if mean_val < 20.0:
        return f"\\textcolor{{gray}}{{{mean_disp}$^*$ \\tiny {ci_disp}}}"

    return f"{mean_disp} \\tiny\\textcolor{{gray}}{{{ci_disp}}}"


def get_numeric_delta(rand_str, target_str):
    """
    Calculates the numeric delta between the random baseline and the target.
    Returns a float or None.
    """
    if not rand_str or not target_str:
        return None

    try:
        r_clean = rand_str.split()[0].replace(r"\textbf{", "").replace("}", "")
        r_val = float(r_clean) * 100

        t_clean = target_str.split()[0].replace(r"\textbf{", "").replace("}", "")
        t_val = float(t_clean) * 100

        return t_val - r_val
    except:
        return None


def format_delta_value(delta, is_best=False):
    """
    Formats the delta. Only bolds if is_best is True.
    """
    if delta is None:
        return "-"

    d_text = f"{delta:+.0f}"

    # Apply standard colors based on positive/negative
    if delta > 0:
        formatted_str = f"\\textcolor{{insightteal}}{{{d_text}}}"
    elif delta < 0:
        formatted_str = f"\\textcolor{{purple}}{{{d_text}}}"
    else:
        formatted_str = f"{d_text}"

    # Bold only the best one
    if is_best:
        return f"\\textbf{{{formatted_str}}}"
    else:
        return formatted_str


def main():
    all_latex_output = []

    for num_gen in NUM_GENERATIONS:
        # Data Storage (re-initialized for each generation count)
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
                        ds_path,
                        folder,
                        required_file=f"pass_at_k_table_reranking_{num_gen}.txt",
                    )
                    fpath = (
                        run_dir / f"pass_at_k_table_reranking_{num_gen}.txt"
                        if run_dir is not None
                        else ds_path
                        / folder
                        / f"pass_at_k_table_reranking_{num_gen}.txt"
                    )

                    res = extract_reranking_p1(fpath)
                    data[model_key][algo_key][dataset] = res

        # --- Step 2: Generate LaTeX for this N ---
        latex = []
        if num_gen == NUM_GENERATIONS[0]:
            latex.append(
                r"% Requires \usepackage{booktabs}, \usepackage{xcolor}, \usepackage{multirow}"
            )

        latex.append(f"% ==========================================")
        latex.append(f"% Table for N = {num_gen}")
        latex.append(f"% ==========================================")
        latex.append(r"\begin{table*}[t]")
        latex.append(r"\centering")
        latex.append(r"\scriptsize")
        latex.append(r"\resizebox{\textwidth}{!}{%")

        # Cols: Backbone | Method | 3 x (Random | dLog | dMaj | dWMaj | dRew) -> 2 + 15 = 17 cols
        latex.append(r"\begin{tabular}{ll ccccc ccccc ccccc}")
        latex.append(r"\toprule")

        # Header 1: Datasets (span 5 columns each)
        latex.append(
            r"& & \multicolumn{5}{c}{\textbf{\textsc{GSM8K}}} & \multicolumn{5}{c}{\textbf{\textsc{MedReason}}} & \multicolumn{5}{c}{\textbf{\textsc{MMLU-Pro}}} \\"
        )
        latex.append(r"\cmidrule(lr){3-7} \cmidrule(lr){8-12} \cmidrule(lr){13-17}")

        # Header 2: Metrics
        header_metrics = r"Random & $\Delta$ Logp. & $\Delta$ Maj. & $\Delta$ Rew.  & $\Delta$ W.Maj."
        latex.append(
            f"\\textbf{{Backbone}} & \\textbf{{Method}} & {header_metrics} & {header_metrics} & {header_metrics} \\\\"
        )
        latex.append(r"\midrule")

        for model_key, model_name in MODEL_ORDER:
            first_row = True
            num_algos = len(ALGO_ORDER)

            for algo_key, algo_name in ALGO_ORDER:
                row_cells = []

                for dataset in DATASETS:
                    entry = data[model_key][algo_key].get(
                        dataset,
                        {
                            "Random": None,
                            "Logprobs": None,
                            "Majority": None,
                            "Weighted_Majority": None,
                            "Reward": None,
                        },
                    )

                    # Format the base Random cell
                    rand_fmt = format_cell(entry["Random"])

                    # Calculate numeric deltas
                    d_log_num = get_numeric_delta(entry["Random"], entry["Logprobs"])
                    d_maj_num = get_numeric_delta(entry["Random"], entry["Majority"])
                    d_rew_num = get_numeric_delta(entry["Random"], entry["Reward"])
                    d_wmaj_num = get_numeric_delta(
                        entry["Random"], entry["Weighted_Majority"]
                    )

                    # Find the highest improvement (max delta) among the four
                    valid_deltas = [
                        d
                        for d in (d_log_num, d_maj_num, d_rew_num, d_wmaj_num)
                        if d is not None
                    ]
                    max_d = max(valid_deltas) if valid_deltas else None

                    # Format cells, explicitly passing True only to the highest value
                    d_log = format_delta_value(
                        d_log_num, is_best=(d_log_num == max_d and max_d is not None)
                    )
                    d_maj = format_delta_value(
                        d_maj_num, is_best=(d_maj_num == max_d and max_d is not None)
                    )
                    d_wmaj = format_delta_value(
                        d_wmaj_num, is_best=(d_wmaj_num == max_d and max_d is not None)
                    )
                    d_rew = format_delta_value(
                        d_rew_num, is_best=(d_rew_num == max_d and max_d is not None)
                    )

                    row_cells.extend([rand_fmt, d_log, d_maj, d_rew, d_wmaj])

                # Backbone Label (Multirow)
                if first_row:
                    model_col = (
                        f"\\multirow{{{num_algos}}}{{*}}{{\\textbf{{{model_name}}}}}"
                    )
                else:
                    model_col = ""

                # Build Row
                latex.append(
                    f"{model_col} & {algo_name} & " + " & ".join(row_cells) + r" \\"
                )
                first_row = False

            # Rule between models
            if model_key != MODEL_ORDER[-1][0]:
                latex.append(r"\midrule")

        latex.append(r"\bottomrule")
        latex.append(r"\end{tabular}%")
        latex.append(r"}")
        latex.append(
            f"\\caption{{\\textbf{{Best-of-{num_gen} Reranking Performance \\& Baselines (\\%).}} The \\textit{{Random}} column reports the absolute percentage with 95\\% CI on the SFT-trained model generations. The $\\Delta$ columns denote the percentage-point change when using (i) logprobs, (ii) majority voting, (iii) reward model reranking, and (iv) reward-weighted majority voting respectively. \\textbf{{Bold}} indicates the highest improvement for that dataset. \\textcolor{{insightteal}}{{Blue}} indicates an improvement over Random, \\textcolor{{purple}}{{purple}} indicates degradation.}}"
        )
        latex.append(f"\\label{{tab:reranking_baselines_N{num_gen}}}")
        latex.append(r"\end{table*}")

        # Add a couple of blank lines between tables for readability
        latex.append("\n\n")

        all_latex_output.extend(latex)

    output_file = os.path.join(ROOT_DIR, "results_reranking_baselines_all.txt")
    with open(output_file, "w") as f:
        f.write("\n".join(all_latex_output))

    print(f"Success! Created '{output_file}' with {len(NUM_GENERATIONS)} tables.")


if __name__ == "__main__":
    main()
