import os
import re
import pandas as pd

# ================= CONFIGURATION =================

ROOT_DIR = os.path.join('figures', 'answer_only')

# Rows: The Methods
ALGO_ORDER = [
    ('sparse', '\\textit{Sparse}'),
    ('partial', '\\textit{Step-wise}'),
    ('partial_fixed', '\\textit{Interval}'),
    ('full', '\\textit{Dense}'),
    #('ovr', '\\textit{Step-wise + OVR}')
]

# Backbones (Grouped rows)
MODEL_ORDER = [
    ('qwen3b', r'\texttt{Qwen2.5-3B}'),
    ('llama3b', r'\texttt{Llama3.2-3B}'),
    ('qwen7b', r'\texttt{Qwen2.5-7B}'),
    ('llama8b', r'\texttt{Llama3.1-8B}')
]

DATASETS = ['math', 'medicine']
VALUE_PATTERN = r"(\d+\.\d+\s*\[\s*\d+\.\d+,\s*\d+\.\d+\s*\])"

def extract_reranking_p1(filepath):
    if not os.path.exists(filepath):
        return {'Random': None, 'Reward': None}

    with open(filepath, 'r') as f:
        content = f.read().replace('\n', ' ') 
    
    rows = content.split(r'\\')
    results = {'Random': None, 'Reward': None}

    for row in rows:
        if "Random Reranking" in row:
            matches = re.findall(VALUE_PATTERN, row)
            if matches: results['Random'] = matches[0]
        elif "Reasoning Reranking" in row:
            matches = re.findall(VALUE_PATTERN, row)
            if matches: results['Reward'] = matches[0]
            
    return results

def get_mean(val_str):
    if not val_str: return -100.0
    try:
        clean = val_str.replace(r'\textbf{', '').replace(r'\underline{', '').replace('}', '')
        return float(clean.split()[0]) * 100
    except:
        return -100.0

def format_cell(val_str, is_best=False, is_second=False):
    """
    Formats: 79 [76, 81] in percentage with no decimals.
    """
    if not val_str: return "-"
    
    parts = val_str.split(' ', 1)
    if len(parts) != 2: return val_str
    
    mean_str, interval = parts[0], parts[1]
    
    # 1. Convert Mean to Percentage (No Decimals)
    try:
        clean_mean = mean_str.replace(r'\textbf{', '').replace(r'\underline{', '').replace('}', '')
        mean_val = float(clean_mean) * 100
        mean_disp = f"{mean_val:.0f}"
    except:
        mean_val = 100.0
        mean_disp = "100"

    # 2. Convert Interval to Percentage
    try:
        clean_ci = interval.strip('[]')
        low, high = clean_ci.split(',')
        low_pct = float(low) * 100
        high_pct = float(high) * 100
        ci_disp = f"[{low_pct:.0f}, {high_pct:.0f}]"
    except:
        ci_disp = interval

    # 3. Mode Collapse Check (< 20%)
    if mean_val < 20.0:
        return f"\\textcolor{{gray}}{{{mean_disp}$^*$ \\tiny {ci_disp}}}"

    # 4. Apply Highlights
    fmt_mean = mean_disp
    if is_best:
        fmt_mean = f"\\textbf{{{mean_disp}}}"
    elif is_second:
        fmt_mean = f"\\underline{{{mean_disp}}}"
        
    # Inline format: Mean [CI]
    return f"{fmt_mean} \\tiny\\textcolor{{gray}}{{{ci_disp}}}"

def format_delta(rand_str, rew_str):
    if not rand_str or not rew_str: return "-"
    
    try:
        # Parse Random
        r_clean = rand_str.split()[0].replace(r'\textbf{', '').replace('}', '')
        r_val = float(r_clean) * 100
        
        # Parse Reward
        rew_clean = rew_str.split()[0].replace(r'\textbf{', '').replace('}', '')
        rew_val = float(rew_clean) * 100
        
        delta = rew_val - r_val
        
        # Format: +2
        d_text = f"{delta:+.0f}"
        
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
        ds_path = os.path.join(ROOT_DIR, dataset)
        if not os.path.exists(ds_path): continue

        for model_key, _ in MODEL_ORDER:
            for algo_key, _ in ALGO_ORDER:
                folder = f"{model_key}_{algo_key}"
                fpath = os.path.join(ds_path, folder, 'pass_at_k_table_reranking.txt')
                
                res = extract_reranking_p1(fpath)
                data[model_key][algo_key][dataset] = res

    # --- Step 2: Ranking Logic (Best Reward per Model/Dataset) ---
    for model_key, _ in MODEL_ORDER:
        for dataset in DATASETS:
            means = []
            for algo_key, _ in ALGO_ORDER:
                val = data[model_key][algo_key][dataset]['Reward']
                m = get_mean(val)
                if m > -1: means.append(m)
            
            unique = sorted(list(set(means)), reverse=True)
            best = unique[0] if unique else -99
            second = unique[1] if len(unique) > 1 else -99
            
            for algo_key, _ in ALGO_ORDER:
                val = data[model_key][algo_key][dataset]['Reward']
                m = get_mean(val)
                data[model_key][algo_key][dataset]['is_best'] = (m == best and m > -1)
                data[model_key][algo_key][dataset]['is_second'] = (m == second and m > -1)

    # --- Step 3: Generate LaTeX ---
    latex = []
    latex.append(r"% Requires \usepackage{booktabs}, \usepackage{xcolor}, \usepackage{multirow}")
    latex.append(r"\begin{table*}[t]")
    latex.append(r"\centering")
    latex.append(r"\scriptsize")
    latex.append(r"\resizebox{\textwidth}{!}{%")
    
    # Cols: Backbone | Method | GSM(Rand, Rew, Delta) | Med(Rand, Rew, Delta)
    # Total 8 columns
    latex.append(r"\begin{tabular}{ll ccc ccc}")
    latex.append(r"\toprule")
    
    # Header 1: Datasets
    latex.append(r"& & \multicolumn{3}{c}{\textbf{\textsc{GSM8K}}} & \multicolumn{3}{c}{\textbf{\textsc{MedReason}}} \\")
    latex.append(r"\cmidrule(lr){3-5} \cmidrule(lr){6-8}")
    
    # Header 2: Metrics
    latex.append(r"\textbf{Backbone} & \textbf{Method} & Random & Reward & $\Delta$ (pp) & Random & Reward & $\Delta$ (pp)\\")
    latex.append(r"\midrule")

    for model_key, model_name in MODEL_ORDER:
        first_row = True
        num_algos = len(ALGO_ORDER)
        
        for algo_key, algo_name in ALGO_ORDER:
            row_cells = []
            
            # GSM8K Data
            gsm_entry = data[model_key][algo_key]['math']
            gsm_rand = format_cell(gsm_entry['Random'])
            gsm_rew = format_cell(gsm_entry['Reward'], gsm_entry.get('is_best'), gsm_entry.get('is_second'))
            gsm_delta = format_delta(gsm_entry['Random'], gsm_entry['Reward'])
            row_cells.extend([gsm_rand, gsm_rew, gsm_delta])
            
            # MedReason Data
            med_entry = data[model_key][algo_key]['medicine']
            med_rand = format_cell(med_entry['Random'])
            med_rew = format_cell(med_entry['Reward'], med_entry.get('is_best'), med_entry.get('is_second'))
            med_delta = format_delta(med_entry['Random'], med_entry['Reward'])
            row_cells.extend([med_rand, med_rew, med_delta])
            
            # Backbone Label (Multirow)
            if first_row:
                model_col = f"\\multirow{{{num_algos}}}{{*}}{{\\textbf{{{model_name}}}}}"
            else:
                model_col = ""
            
            # Build Row
            latex.append(f"{model_col} & {algo_name} & " + " & ".join(row_cells) + r" \\")
            first_row = False
        
        # Rule between models
        if model_key != MODEL_ORDER[-1][0]:
            latex.append(r"\midrule")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}%")
    latex.append(r"}")
    latex.append(r"\caption{\textbf{Best-of-N Reranking Performance (\%).} Comparison of Random selection vs. Reward Model selection. Values are percentages. \textbf{Bold} is best, \underline{underline} is second best. $\Delta$ indicates percentage-point improvement. \textcolor{insightteal}{Blue} is positive, \textcolor{purple}{purple} is negative. * symbolises an adversarial mode collapse (results grayed out).}")
    latex.append(r"\label{tab:reranking_long}")
    latex.append(r"\end{table*}")

    output_file = os.path.join(ROOT_DIR, "results_reranking_long.txt")
    with open(output_file, "w") as f:
        f.write("\n".join(latex))
    
    print(f"Success! Created '{output_file}'")

if __name__ == "__main__":
    main()