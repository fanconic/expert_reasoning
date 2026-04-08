import os
import re
import pandas as pd

# ================= CONFIGURATION =================

ROOT_DIR = os.path.join('figures', 'answer_only')

# Row Order (Algorithms)
ALGO_ORDER = [
    ('SFT', 'SFT'),
    ('sparse', 'Ours (\\textit{Sparse})'),
    ('partial', 'Ours (\\textit{Step-wise})'),
    ('partial_fixed', 'Ours (\\textit{Interval})'),
    ('full', 'Ours (\\textit{Dense})'),
    #('ovr', 'Ours (\\textit{Step-wise + OVR})')
]

# Model Order (Now used for Sub-Headers)
MODEL_ORDER = [
    ('qwen3b', r'\texttt{Qwen2.5-3B}'),
    # ('llama3b', r'\texttt{Llama3.2-3B}'),
    # ('qwen7b', r'\texttt{Qwen2.5-7B}'),
    # ('llama8b', r'\texttt{Llama3.1-8B}')
    ('qwen4b', r'\texttt{Qwen3-4B}')
]

DATASET_COLUMNS = [
    #('math', r'\textbf{\textsc{GSM8K}}'),
    ('aime_2024', r'\textbf{\textsc{AIME 2024}}'),
    ('aime_2025', r'\textbf{\textsc{AIME 2025}}'),
    #('mmlu', r'\textbf{\textsc{MMLU-Pro}}'),
    #('medicine', r'\textbf{\textsc{MedReason}}'),
]
DATASETS = [d for d, _ in DATASET_COLUMNS]
VALUE_PATTERN = r"(\d+\.\d+\s*\[\s*\d+\.\d+,\s*\d+\.\d+\s*\])"

def extract_all_k(filepath, method_label):
    """Reads file and extracts [p1, p3, p5, p10]."""
    if not os.path.exists(filepath): return [None] * 4

    with open(filepath, 'r') as f:
        lines = f.readlines()
    full_text = " ".join([line.strip() for line in lines])
    rows = full_text.split(r'\\')

    for row in rows:
        if method_label in row:
            if method_label == 'GRPO' and 'GRPO' not in row: continue
            if method_label == 'SFT' and 'SFT' not in row: continue
            if method_label == 'AIRL' and 'AIRL' not in row: continue

            matches = re.findall(VALUE_PATTERN, row)
            if matches:
                if len(matches) < 4: matches += [None] * (4 - len(matches))
                return matches[:4]
    return [None] * 4

def get_mean(val_str):
    """Parses '0.79 [0.77, 0.81]' to float 0.79"""
    if not val_str: return -1.0
    try:
        clean = val_str.replace(r'\textbf{', '').replace(r'\underline{', '').replace('}', '')
        return float(clean.split()[0])
    except:
        return -1.0

def format_cell(val_str, is_best=False, is_second=False, is_collapsed=False):
    """
    Converts 0.79 [0.77, 0.81] -> 79.0 {\scriptsize\color{gray}[77.0, 81.0]}
    Handles Mode Collapse (< 20%).
    """
    if not val_str: return "-"
    
    parts = val_str.split(' ', 1)
    if len(parts) != 2: return val_str
    
    raw_mean_str, raw_interval = parts[0], parts[1]
    
    # --- 1. Convert Mean to Percentage ---
    try:
        clean_mean = raw_mean_str.replace(r'\textbf{', '').replace(r'\underline{', '').replace('}', '')
        mean_val = float(clean_mean) * 100
        mean_disp = f"{mean_val:.2f}"
    except:
        mean_val = 100.0
        mean_disp = "100.0"

    # --- 2. Handle Mode Collapse ---
    if is_collapsed:
        # Gray out everything, add asterisk
        # Parse interval to percentage first
        try:
            clean_ci = raw_interval.strip('[]')
            low, high = clean_ci.split(',')
            ci_disp = f"[{float(low)*100:.2f}, {float(high)*100:.2f}]"
        except:
            ci_disp = raw_interval
            
        return f"\\textcolor{{gray}}{{{mean_disp}$^*$ {{\\scriptsize {ci_disp}}}}}"

    # --- 3. Format Interval (Percentage) ---
    try:
        clean_ci = raw_interval.strip('[]')
        low, high = clean_ci.split(',')
        ci_disp = f"[{float(low)*100:.2f}, {float(high)*100:.2f}]"
    except:
        ci_disp = raw_interval
        
    fmt_interval = f"{{\\scriptsize\\color{{gray}}{ci_disp}}}"

    # --- 4. Format Mean (Bold/Underline) ---
    fmt_mean = mean_disp
    if is_best:
        fmt_mean = f"\\textbf{{{mean_disp}}}"
    elif is_second:
        fmt_mean = f"\\underline{{{mean_disp}}}"

    return f"{fmt_mean}"   #{fmt_interval}"

def main():
    data = {
        m: {
            a: {dataset: [None] * 4 for dataset in DATASETS}
            for a, _ in ALGO_ORDER
        }
        for m, _ in MODEL_ORDER
    }

    # --- Step 1: Extraction ---
    for dataset in DATASETS:
        ds_path = os.path.join(ROOT_DIR, dataset)
        if not os.path.exists(ds_path): continue

        for model_key, _ in MODEL_ORDER:
            # Search logic for Baselines
            sft_found = False
            candidates = sorted([f for f in os.listdir(ds_path) if f.startswith(model_key)])
            if model_key in candidates: candidates.insert(0, candidates.pop(candidates.index(model_key)))

            for folder in candidates:
                if sft_found:
                    break
                fpath = os.path.join(ds_path, folder, 'pass_at_k_table_reward_weighted.txt')
                
                if not sft_found:
                    vals = extract_all_k(fpath, 'SFT')
                    if any(vals): 
                        data[model_key]['SFT'][dataset] = vals
                        sft_found = True

            # Standard variants
            for algo_key, _ in ALGO_ORDER:
                if algo_key in ['SFT']: continue
                variant_folder = f"{model_key}_{algo_key}"
                variant_file = os.path.join(ds_path, variant_folder, 'pass_at_k_table_reward_weighted.txt')
                data[model_key][algo_key][dataset] = extract_all_k(variant_file, 'AIRL')

    # --- Step 2: Analyze Rankings & Formatting ---
    # We do this before generating latex to handle percentages cleanly
    formatted_data = {
        m: {
            a: {dataset: ["-"] * 4 for dataset in DATASETS}
            for a, _ in ALGO_ORDER
        }
        for m, _ in MODEL_ORDER
    }
    
    for model_key, _ in MODEL_ORDER:
        for dataset in DATASETS:
            col_results = [{} for _ in range(4)]

            # Rank and format each pass@k column independently.
            for k_idx in range(4):
                means = []
                for algo_key, _ in ALGO_ORDER:
                    val_str = data[model_key][algo_key].get(dataset, [None] * 4)[k_idx]
                    mean = get_mean(val_str)
                    if mean >= 0:
                        means.append(mean)

                unique = sorted(list(set(means)), reverse=True)
                best_val = unique[0] if unique else -1.0
                second_val = unique[1] if len(unique) > 1 else -1.0

                for algo_key, _ in ALGO_ORDER:
                    val_str = data[model_key][algo_key].get(dataset, [None] * 4)[k_idx]
                    raw_mean = get_mean(val_str)

                    is_best = (raw_mean == best_val and raw_mean > -1)
                    is_second = (raw_mean == second_val and raw_mean > -1)
                    col_results[k_idx][algo_key] = format_cell(val_str, is_best, is_second)

            for algo_key, _ in ALGO_ORDER:
                formatted_data[model_key][algo_key][dataset] = [
                    col_results[k_idx].get(algo_key, "-") for k_idx in range(4)
                ]

    # --- Step 3: Generate Cleaner LaTeX ---
    latex = []
    latex.append(r"% Requires \usepackage{booktabs}, \usepackage{xcolor}")
    latex.append(r"\begin{table}[h!]") 
    latex.append(r"\centering")
    latex.append(r"\small")
    num_datasets = len(DATASET_COLUMNS)
    total_cols = 1 + num_datasets
    col_spec = "l " + " ".join(["c"] * num_datasets)
    latex.append(f"\\begin{{tabular}}{{{col_spec}}}") 
    latex.append(r"\toprule")
    
    dataset_header = " & ".join([label for _, label in DATASET_COLUMNS])
    latex.append(f"& {dataset_header} \\\\")
    metric_header = " & ".join(["Pass@1"] * num_datasets)
    latex.append(f"\\textbf{{Method}} & {metric_header} \\\\")
    latex.append(r"\midrule")

    for model_key, model_name in MODEL_ORDER:
        # === SUB-HEADER ROW ===
        latex.append(f"\\multicolumn{{{total_cols}}}{{l}}{{\\textbf{{{model_name}}}}} \\\\")
        
        for algo_key, algo_name in ALGO_ORDER:
            dataset_vals = [formatted_data[model_key][algo_key][dataset][0] for dataset, _ in DATASET_COLUMNS]
            
            # Add horizontal spacing for hierarchy
            latex.append(f"\\hspace{{1em}}{algo_name} & {' & '.join(dataset_vals)} \\\\")

        # Midrule between models
        if model_key != MODEL_ORDER[-1][0]:
            latex.append(r"\midrule")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}%")
    latex.append(r"\caption{\textbf{Pass@1 Performance (\%).} \textbf{Bold} indicates the best performance compared between SFT and our methods.}")
    latex.append(r"\label{tab:p1_results_aime}")
    latex.append(r"\end{table}")

    output_path = os.path.join(ROOT_DIR, "results_p1_aime.txt")
    with open(output_path, "w") as f:
        f.write("\n".join(latex))
    
    print(f"Success! Created '{output_path}'")

if __name__ == "__main__":
    main()
