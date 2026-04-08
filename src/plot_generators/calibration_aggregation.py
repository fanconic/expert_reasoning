import os
import re
from pathlib import Path

try:
    from src.plot_generators.run_paths import resolve_run_dir
except ModuleNotFoundError:
    from run_paths import resolve_run_dir

# ================= CONFIGURATION =================
ROOT_DIR = Path("figures/answer_only")

ALGO_ORDER = [
    ("sparse_new", r"\textit{Sparse}"),
    ("partial_new", r"\textit{Step-wise}"),
    ("partial_fixed_new", r"\textit{Interval}"),
    ("full_new", r"\textit{Dense}"),
]

MODEL_ORDER = [
    ("qwen3b", r"\texttt{Qwen2.5-3B}"),
    ("llama3b", r"\texttt{Llama3.2-3B}"),
    ("qwen7b", r"\texttt{Qwen2.5-7B}"),
    ("llama8b", r"\texttt{Llama3.1-8B}"),
    ("qwen4b", r"\texttt{Qwen3-4B}"),
]

DATASETS = [("math", "GSM8K"), ("medicine", "MedReason"), ("mmlu", "MMLU-Pro")]
VALUE_PATTERN = r"(\d*\.?\d+\s*\[\s*\d*\.?\d+,\s*\d*\.?\d+\s*\])"


def extract_calibration_metrics(filepath):
    results = {"AUROC": None, "ECE": None}
    if not filepath.exists():
        return results

    with open(filepath, "r") as f:
        content = f.read()

    for line in content.split("\\\\"):
        if "Exp. Reas. (ours)" in line:
            matches = re.findall(VALUE_PATTERN, line)
            if len(matches) >= 2:
                results["AUROC"] = matches[0]
                results["ECE"] = matches[1]
    return results


def format_calib_cell(val_str):
    """
    Converts 0.XXXX [0.YYYY, 0.ZZZZ] to XX [YY, ZZ].
    Uses integer rounding for 'full' percentages.
    """
    if not val_str:
        return "-"

    try:
        # Extract all numbers from the string
        nums = [float(n) for n in re.findall(r"\d*\.?\d+", val_str)]
        if len(nums) < 3:
            return "-"

        # Multiply by 100 and round to nearest integer
        mean_p = int(round(nums[0] * 100))
        low_p = int(round(nums[1] * 100))
        high_p = int(round(nums[2] * 100))

        return f"{mean_p} \\tiny\\textcolor{{gray}}{{[{low_p}, {high_p}]}}"
    except (ValueError, IndexError):
        return "-"


def main():
    latex = []
    latex.append(r"\begin{table*}[t]")
    latex.append(r"\centering\scriptsize")
    latex.append(r"\resizebox{\textwidth}{!}{%")
    latex.append(r"\begin{tabular}{ll cc cc cc}")
    latex.append(r"\toprule")

    # Header 1
    header1 = "&"
    for _, ds_name in DATASETS:
        header1 += f" & \multicolumn{{2}}{{c}}{{\\textbf{{\\textsc{{{ds_name}}}}}}}"
    latex.append(header1 + r" \\")

    latex.append(r"\cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8}")

    # Header 2
    header2 = r"\textbf{Backbone} & \textbf{Method} "
    for _ in DATASETS:
        header2 += r" & AUROC (\%) $\uparrow$ & ECE (\%) $\downarrow$ "
    latex.append(header2 + r" \\")
    latex.append(r"\midrule")

    for model_key, model_name in MODEL_ORDER:
        for i, (algo_key, algo_name) in enumerate(ALGO_ORDER):
            row_cells = []
            for ds_key, _ in DATASETS:
                ds_path = ROOT_DIR / ds_key
                run_name = f"{model_key}_{algo_key}"
                run_dir = resolve_run_dir(
                    ds_path, run_name, required_file="calibration_metrics_table.txt"
                )
                fpath = (
                    run_dir / "calibration_metrics_table.txt"
                    if run_dir is not None
                    else ds_path / run_name / "calibration_metrics_table.txt"
                )
                metrics = extract_calibration_metrics(fpath)

                auroc = format_calib_cell(metrics["AUROC"])
                ece = format_calib_cell(metrics["ECE"])
                row_cells.extend([auroc, ece])

            m_label = (
                f"\\multirow{{{len(ALGO_ORDER)}}}{{*}}{{\\textbf{{{model_name}}}}}"
                if i == 0
                else ""
            )
            latex.append(f"{m_label} & {algo_name} & " + " & ".join(row_cells) + r" \\")

        if model_key != MODEL_ORDER[-1][0]:
            latex.append(r"\midrule")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}}")
    latex.append(
        r"\caption{\textbf{Critic Calibration Metrics.} All values are reported as percentages (\%). AUROC indicates ranking ability; ECE measures calibration error (lower is better).}"
    )
    latex.append(r"\end{table*}")

    output_path = ROOT_DIR / "results_calibration_table.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write("\n".join(latex))
    print(f"Created {output_path}")


if __name__ == "__main__":
    main()
