import os
import re
from pathlib import Path

# ================= CONFIGURATION =================
ROOT_DIR = os.path.join('figures', 'answer_only')

ALGO_ORDER = [
    ('sparse_new', '\\textit{Sparse}'),
    ('partial_new', '\\textit{Step-wise}'),
    ('partial_fixed_new', '\\textit{Interval}'),
    ('full_new', '\\textit{Dense}'),
]

MODEL_ORDER = [
    ('qwen3b', r'\texttt{Qwen2.5-3B}'),
    ('llama3b', r'\texttt{Llama3.2-3B}'),
    ('qwen7b', r'\texttt{Qwen2.5-7B}'),
    ('llama8b', r'\texttt{Llama3.1-8B}')
]

DATASETS = [('math', 'GSM8K'), ('medicine', 'MedReason'), ('mmlu', 'MMLU-Pro')]
VALUE_PATTERN = r"(\d+\.\d+\s*\[\s*\d+\.\d+,\s*\d+\.\d+\s*\])"

def extract_calibration_metrics(filepath):
    """
    Extracts AUROC and ECE for the 'Exp. Reas. (ours)' row.
    """
    results = {'AUROC': None, 'ECE': None}
    if not os.path.exists(filepath):
        return results

    with open(filepath, 'r') as f:
        content = f.read()
    
    # Look for the row containing our primary method
    for line in content.split('\\\\'):
        if "Exp. Reas. (ours)" in line:
            matches = re.findall(VALUE_PATTERN, line)
            if len(matches) >= 2:
                results['AUROC'] = matches[0]
                results['ECE'] = matches[1]
    return results

def format_calib_cell(val_str, multiply_by_100=True):
    """
    Formats the metric. AUROC and ECE are often cleaner as 0.XX, 
    but we can convert to XX.X if you prefer percentage style.
    """
    if not val_str: return "-"
    
    parts = val_str.split(' ', 1)
    mean_val = float(parts[0])
    ci = parts[1]
    
    if multiply_by_100:
        mean_disp = f"{mean_val * 100:.1f}"
        # Parse CI to multiply by 100
        ci_vals = re.findall(r"\d+\.\d+", ci)
        ci_disp = f"[{float(ci_vals[0])*100:.1f}, {float(ci_vals[1])*100:.1f}]"
    else:
        mean_disp = f"{mean_val:.3f}"
        ci_disp = ci

    return f"{mean_disp} \\tiny\\textcolor{{gray}}{{{ci_disp}}}"

def main():
    latex = []
    latex.append(r"\begin{table*}[t]")
    latex.append(r"\centering\scriptsize")
    latex.append(r"\resizebox{\textwidth}{!}{%")
    # 2 (labels) + 3 datasets * 2 metrics = 8 columns
    latex.append(r"\begin{tabular}{ll cc cc cc}")
    latex.append(r"\toprule")
    
    # Header 1
    header1 = "&"
    for _, ds_name in DATASETS:
        header1 += f" & \multicolumn{{2}}{{c}}{{\\textbf{{\\textsc{{{ds_name}}}}}}}"
    latex.append(header1 + r" \\")
    
    # cmidrules
    latex.append(r"\cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8}")
    
    # Header 2
    header2 = r"\textbf{Backbone} & \textbf{Method} "
    for _ in DATASETS:
        header2 += r" & AUROC $\uparrow$ & ECE $\downarrow$ "
    latex.append(header2 + r" \\")
    latex.append(r"\midrule")

    for model_key, model_name in MODEL_ORDER:
        first_row = True
        for algo_key, algo_name in ALGO_ORDER:
            row_cells = []
            for ds_key, _ in DATASETS:
                fpath = os.path.join(ROOT_DIR, ds_key, f"{model_key}_{algo_key}", 'calibration_metrics_table.txt')
                metrics = extract_calibration_metrics(fpath)
                
                # We use multiply_by_100=True to match your pass@k table style
                auroc = format_calib_cell(metrics['AUROC'], multiply_by_100=True)
                ece = format_calib_cell(metrics['ECE'], multiply_by_100=True)
                row_cells.extend([auroc, ece])
            
            m_label = f"\\multirow{{{len(ALGO_ORDER)}}}{{*}}{{\\textbf{{{model_name}}}}}" if first_row else ""
            latex.append(f"{m_label} & {algo_name} & " + " & ".join(row_cells) + r" \\")
            first_row = False
        
        if model_key != MODEL_ORDER[-1][0]:
            latex.append(r"\midrule")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}}")
    latex.append(r"\caption{\textbf{Critic Calibration Metrics.} AUROC (\%) indicates the ability to rank correct vs. incorrect traces. ECE (\%) measures the Expected Calibration Error (lower is better).}")
    latex.append(r"\end{table*}")

    output_path = os.path.join(ROOT_DIR, "results_calibration_table.txt")
    with open(output_path, "w") as f:
        f.write("\n".join(latex))
    print(f"Created {output_path}")

if __name__ == "__main__":
    main()