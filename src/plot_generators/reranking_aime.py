import os
import re
from pathlib import Path

try:
    from src.plot_generators.run_paths import resolve_run_dir
except ModuleNotFoundError:
    from run_paths import resolve_run_dir

# ================= CONFIGURATION =================

ROOT_DIR = os.path.join("figures", "answer_only")
RERANKING_FILE = "pass_at_k_table_reranking_16.txt"

# Rows: Methods
ALGO_ORDER = [
    ("sparse", r"\textit{Sparse}"),
    ("partial", r"\textit{Step-wise}"),
    ("partial_fixed", r"\textit{Interval}"),
    ("full", r"\textit{Dense}"),
]

# Backbones
MODEL_ORDER = [
    ("qwen3b", r"\texttt{Qwen2.5-3B}"),
    ("qwen4b", r"\texttt{Qwen3-4B}"),
]

# Datasets (AIME only)
DATASET_COLUMNS = [
    ("aime_2024", r"\textbf{\textsc{AIME 2024}}"),
    ("aime_2025", r"\textbf{\textsc{AIME 2025}}"),
]

VALUE_PATTERN = r"(\d+\.\d+\s*\[\s*\d+\.\d+,\s*\d+\.\d+\s*\])"


def extract_reranking_p1(filepath: str) -> dict:
    """Parse pass@1 for Random and Reasoning reranking from reranking table txt."""
    results = {"Random": None, "Reward": None}
    if not os.path.exists(filepath):
        return results

    with open(filepath, "r") as f:
        content = f.read().replace("\n", " ")

    rows = content.split(r"\\")
    for row in rows:
        row_lower = row.lower()
        if "random reranking" in row_lower:
            matches = re.findall(VALUE_PATTERN, row)
            if matches:
                results["Random"] = matches[0]
        elif "reasoning reranking" in row_lower or "reward reranking" in row_lower:
            matches = re.findall(VALUE_PATTERN, row)
            if matches:
                results["Reward"] = matches[0]

    return results


def get_mean(val_str: str) -> float:
    if not val_str:
        return -100.0
    try:
        clean = (
            val_str.replace(r"\textbf{", "")
            .replace(r"\underline{", "")
            .replace("}", "")
        )
        return float(clean.split()[0]) * 100.0
    except Exception:
        return -100.0


def format_cell(val_str: str) -> str:
    """Formats: 0.0345 [0.0000, 0.1034] -> 3.45 [0.00, 10.34]."""
    if not val_str:
        return "-"

    parts = val_str.split(" ", 1)
    if len(parts) != 2:
        return val_str

    mean_str, interval = parts[0], parts[1]
    try:
        mean_val = float(mean_str) * 100.0
        mean_disp = f"{mean_val:.2f}"
    except Exception:
        mean_disp = mean_str

    try:
        clean_ci = interval.strip("[]")
        low, high = clean_ci.split(",")
        ci_disp = f"[{float(low) * 100.0:.2f}, {float(high) * 100.0:.2f}]"
    except Exception:
        ci_disp = interval

    return f"{mean_disp} \\tiny\\textcolor{{gray}}{{{ci_disp}}}"


def format_delta(rand_str: str, rew_str: str) -> str:
    if not rand_str or not rew_str:
        return "-"

    try:
        rand_val = float(rand_str.split()[0]) * 100.0
        rew_val = float(rew_str.split()[0]) * 100.0
        delta = rew_val - rand_val
        d_text = f"{delta:+.2f}"

        if delta > 0:
            return f"\\textbf{{\\textcolor{{insightteal}}{{($\\uparrow$ {d_text})}}}}"
        if delta < 0:
            return f"\\textcolor{{purple}}{{($\\downarrow$ {d_text})}}"
        return f"({d_text})"
    except Exception:
        return "-"


def main() -> None:
    # Data storage: model -> algo -> dataset -> {Random, Reward, is_best, is_second}
    data = {
        model_key: {
            algo_key: {
                dataset_key: {
                    "Random": None,
                    "Reward": None,
                    "is_best": False,
                    "is_second": False,
                }
                for dataset_key, _ in DATASET_COLUMNS
            }
            for algo_key, _ in ALGO_ORDER
        }
        for model_key, _ in MODEL_ORDER
    }

    # --- Step 1: Extraction ---
    for dataset_key, _ in DATASET_COLUMNS:
        ds_path = Path(ROOT_DIR) / dataset_key
        if not ds_path.exists():
            continue

        for model_key, _ in MODEL_ORDER:
            for algo_key, _ in ALGO_ORDER:
                folder_name = f"{model_key}_{algo_key}"
                run_dir = resolve_run_dir(
                    ds_path, folder_name, required_file=RERANKING_FILE
                )
                fpath = (
                    run_dir / RERANKING_FILE
                    if run_dir is not None
                    else ds_path / folder_name / RERANKING_FILE
                )
                data[model_key][algo_key][dataset_key] = extract_reranking_p1(fpath)

    # --- Step 2: Ranking Logic (best Reward per model/dataset) ---
    for model_key, _ in MODEL_ORDER:
        for dataset_key, _ in DATASET_COLUMNS:
            means = []
            for algo_key, _ in ALGO_ORDER:
                m = get_mean(data[model_key][algo_key][dataset_key]["Reward"])
                if m > -1:
                    means.append(m)

            unique = sorted(list(set(means)), reverse=True)
            best = unique[0] if unique else -99
            second = unique[1] if len(unique) > 1 else -99

            for algo_key, _ in ALGO_ORDER:
                m = get_mean(data[model_key][algo_key][dataset_key]["Reward"])
                data[model_key][algo_key][dataset_key]["is_best"] = m == best and m > -1
                data[model_key][algo_key][dataset_key]["is_second"] = (
                    m == second and m > -1
                )

    # --- Step 3: Generate LaTeX ---
    latex = []
    latex.append(
        r"% Requires \usepackage{booktabs}, \usepackage{xcolor}, \usepackage{multirow}"
    )
    latex.append(r"\begin{table}[t]")
    latex.append(r"\centering")
    latex.append(r"\scriptsize")
    latex.append(r"\resizebox{\columnwidth}{!}{%")
    latex.append(r"\begin{tabular}{ll ccc ccc}")
    latex.append(r"\toprule")
    latex.append(
        r"& & \multicolumn{3}{c}{\textbf{\textsc{AIME 2024}}} & \multicolumn{3}{c}{\textbf{\textsc{AIME 2025}}} \\"
    )
    latex.append(r"\cmidrule(lr){3-5} \cmidrule(lr){6-8}")
    latex.append(
        r"\textbf{Backbone} & \textbf{Method} & Random & Reward & $\Delta$ (pp) & Random & Reward & $\Delta$ (pp)\\"
    )
    latex.append(r"\midrule")

    for model_key, model_name in MODEL_ORDER:
        first_row = True
        num_algos = len(ALGO_ORDER)

        for algo_key, algo_name in ALGO_ORDER:
            row_cells = []
            for dataset_key, _ in DATASET_COLUMNS:
                entry = data[model_key][algo_key][dataset_key]
                rand = format_cell(entry["Random"])
                rew = format_cell(entry["Reward"])
                delta = format_delta(entry["Random"], entry["Reward"])
                row_cells.extend([rand, rew, delta])

            model_col = ""
            if first_row:
                model_col = (
                    f"\\multirow{{{num_algos}}}{{*}}{{\\textbf{{{model_name}}}}}"
                )

            latex.append(
                f"{model_col} & {algo_name} & " + " & ".join(row_cells) + r" \\"
            )
            first_row = False

        if model_key != MODEL_ORDER[-1][0]:
            latex.append(r"\midrule")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}%")
    latex.append(r"}")
    latex.append(
        r"\caption{\textbf{Best-of-16 Reranking Performance on AIME (\%).} "
        r"Comparison of Random selection vs. Reasoning Reward reranking. "
        r"$\Delta$ indicates percentage-point improvement over Random. "
        r"\textcolor{insightteal}{Blue} is positive and \textcolor{purple}{purple} is negative.}"
    )
    latex.append(r"\label{tab:reranking_aime}")
    latex.append(r"\end{table}")

    output_file = os.path.join(ROOT_DIR, "results_reranking_aime.txt")
    with open(output_file, "w") as f:
        f.write("\n".join(latex))

    print(f"Success! Created '{output_file}'")


if __name__ == "__main__":
    main()
