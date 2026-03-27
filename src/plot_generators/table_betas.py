import os
import re
from pathlib import Path

# ================= CONFIGURATION =================
ROOT_DIR = os.path.join('figures', 'answer_only')
DATASET = 'math' 

ALGO_ORDER = [
    ('partial_range_11', '(-1, 1)'),
    ('partial_range_33', '(-3, 3)'),
    ('partial_new', '(-5, 5)'),
    ('partial_range_1010', '(-10, 10)'),
    ('partial_range_02', '(0, 2)'),
]

MODEL_ORDER = [
    ('llama3b', r'\texttt{Qwen2.5-3B}'),
]

VALUE_PATTERN = r"(\d+\.\d+\s*\[\s*\d+\.\d+,\s*\d+\.\d+\s*\])"

# ================= HELPERS =================

def extract_p1(filepath):
    if not os.path.exists(filepath): return None
    with open(filepath, 'r') as f:
        content = f.read().replace('\n', ' ')
    rows = content.split(r'\\')
    for row in rows:
        if "AIRL" in row or "Exp. Reas." in row:
            matches = re.findall(VALUE_PATTERN, row)
            return matches[0] if matches else None
    return None

def extract_reranking_metrics(filepath):
    """Returns (RandomMean, RewardMean) as floats for Delta calculation"""
    if not os.path.exists(filepath): return None, None
    with open(filepath, 'r') as f:
        content = f.read().replace('\n', ' ')
    
    rand_match = re.search(r"Random Reranking.*?(\d+\.\d+)", content)
    rew_match = re.search(r"Reasoning Reranking.*?(\d+\.\d+)", content)
    
    r_val = float(rand_match.group(1)) if rand_match else None
    w_val = float(rew_match.group(1)) if rew_match else None
    return r_val, w_val

def extract_calibration(filepath):
    if not os.path.exists(filepath): return None, None
    with open(filepath, 'r') as f:
        content = f.read().replace('\n', ' ')
    rows = content.split(r'\\')
    for row in rows:
        if "Exp. Reas." in row:
            matches = re.findall(VALUE_PATTERN, row)
            if len(matches) >= 2: return matches[0], matches[1]
    return None, None

def format_full_cell(val_str, decimals=1):
    """
    Converts '0.792 [0.771, 0.813]' -> '79.2 \tiny\textcolor{gray}{[77.1, 81.3]}'
    """
    if not val_str: return "-"
    
    try:
        # Extract mean and the content inside brackets
        mean_part = val_str.split(' ')[0]
        interval_part = re.search(r"\[(.*?)\]", val_str).group(1)
        
        # Convert to percentages
        mean_val = float(mean_part) * 100
        low, high = [float(x.strip()) * 100 for x in interval_part.split(',')]
        
        return f"{mean_val:.{decimals}f} \\tiny\\textcolor{{gray}}{{[{low:.{decimals}f}, {high:.{decimals}f}]}}"
    except Exception:
        return val_str

# ================= MAIN =================

def main():
    latex = []
    latex.append(r"\begin{table}[h]")
    latex.append(r"\centering\small")
    latex.append(r"\resizebox{\columnwidth}{!}{%")
    latex.append(r"\begin{tabular}{l cccc}")
    latex.append(r"\toprule")
    latex.append(r"\textbf{$\beta$ (LB, UB)} & \textbf{Pass@1} & \textbf{$\Delta$ Rerank} & \textbf{AUROC $\uparrow$} & \textbf{ECE $\downarrow$} \\")
    latex.append(r"\midrule")

    for model_key, model_name in MODEL_ORDER:
        for algo_key, n_label in ALGO_ORDER:
            folder_path = os.path.join(ROOT_DIR, DATASET, f"{model_key}_{algo_key}")
            
            # 1. Pass@1 (with CI)
            p1_raw = extract_p1(os.path.join(folder_path, 'pass_at_k_table.txt'))
            p1_disp = format_full_cell(p1_raw)

            # 2. Reranking Delta (pp difference)
            rand_val, rew_val = extract_reranking_metrics(os.path.join(folder_path, 'pass_at_k_table_reranking.txt'))
            if rand_val is not None and rew_val is not None:
                delta = (rew_val - rand_val) * 100
                delta_disp = f"\\textbf{{{delta:+.1f}}}" if delta > 0 else f"{delta:+.1f}"
            else:
                delta_disp = "-"

            # 3. Calibration (AUROC and ECE with CIs)
            auroc_raw, ece_raw = extract_calibration(os.path.join(folder_path, 'calibration_metrics_table.txt'))
            auroc_disp = format_full_cell(auroc_raw)
            ece_disp = format_full_cell(ece_raw)

            latex.append(f"{n_label} & {p1_disp} & {delta_disp} & {auroc_disp} & {ece_disp} \\\\")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}%")
    latex.append(r"}")
    latex.append(r"\caption{\textbf{Ablation on Reward Clipping ($\beta$)} for Qwen2.5-3B (Step-wise) on GSM8K. All metrics are in percentages (\%). Intervals indicate 95\% confidence intervals.}")
    latex.append(r"\end{table}")

    output_file = os.path.join(ROOT_DIR, DATASET, "ablation_betas_with_ci.txt")
    with open(output_file, "w") as f:
        f.write("\n".join(latex))
    
    print(f"Success! Ablation table with CIs created at {output_file}")

if __name__ == "__main__":
    main()