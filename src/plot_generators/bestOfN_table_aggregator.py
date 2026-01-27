import os
import re
import pandas as pd

# ================= CONFIGURATION =================

ROOT_DIR = os.path.join('figures', 'answer_only')

# Rows: The Methods (Transposed from before)
ALGO_ORDER = [
    ('sparse', '\\textit{Sparse}'),
    ('partial', '\\textit{Step-wise}'),
    ('partial_fixed', '\\textit{Interval}'),
    ('full', '\\textit{Dense}'),
    #('ovr', '\\textit{Step-wise + OVR}')
]

# Columns: The Backbones
MODEL_ORDER = [
    ('qwen3b', r'\texttt{Qwen2.5-3B}'),
    ('llama3b', r'\texttt{Llama3.2-3B}'),
    ('qwen7b', r'\texttt{Qwen2.5-7B}'),
    ('llama8b', r'\texttt{Llama3.1-8B}')
]

DATASETS = ['math', 'medicine']
DATASET_LABELS = [r'\textsc{GSM8K}', r'\textsc{MedReason}']

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
        return float(clean.split()[0])*100
    except:
        return -100.0

def format_compact_cell(entry, is_best=False, is_second=False):
    """
    Returns percentage format: \shortstack{ 81.2 (+2.0) \\ \small[80.1, 82.3] }
    """
    val_str = entry['Reward']
    rand_str = entry['Random']
    
    if not val_str: return "-"

    # 1. Parse Reward and convert to Percentage
    parts = val_str.split(' ', 1)
    if len(parts) != 2: return val_str
    rew_mean_str, rew_interval = parts[0], parts[1]
    
    try:
        clean_rew = rew_mean_str.replace(r'\textbf{', '').replace(r'\underline{', '').replace('}', '')
        # MULTIPLY BY 100 HERE
        rew_val = float(clean_rew) * 100
        rew_mean_disp = f"{rew_val:.0f}" # Format to 1 decimal place
    except:
        rew_val = 100.0
        rew_mean_disp = "100.0"

   

    # 2. Parse Interval and convert to Percentage
    # Interval comes in as "[0.77, 0.81]". We need to parse, multiply, and rebuild.
    try:
        # Remove brackets and split
        clean_ci = rew_interval.strip('[]')
        low, high = clean_ci.split(',')
        low_pct = float(low) * 100
        high_pct = float(high) * 100
        rew_interval_disp = f"[{low_pct:.0f}, {high_pct:.0f}]"
    except:
        rew_interval_disp = rew_interval # Fallback if parsing fails
        
     # --- Mode Collapse Check (< 20.0%) ---
    if rew_val < 20.0:
        return f"\\shortstack{{\\textcolor{{gray}}{{{rew_mean_disp}$^*$}} \\\\ \\small\\textcolor{{gray}}{{{rew_interval_disp}}}}}"

    # 3. Calculate Delta (Percentage Points)
    delta_str = ""
    if rand_str:
        try:
            r_parts = rand_str.split(' ', 1)
            clean_rand = r_parts[0].replace(r'\textbf{', '').replace(r'\underline{', '').replace('}', '')
            # MULTIPLY BY 100 HERE
            rand_val = float(clean_rand) * 100
            
            delta_val = rew_val - rand_val
            
            d_text = f"({delta_val:+.0f})" 
            
            if delta_val > 0:
                delta_str = f"\\textbf{{\\textcolor{{teal}}{{{d_text}}}}}"
            elif delta_val < 0:
                delta_str = f"\\textcolor{{purple}}{{{d_text}}}"
            else:
                delta_str = d_text
        except:
            pass

    # 4. Format Mean (Bold/Underline)
    fmt_mean = rew_mean_disp

    # 5. Format Interval (Gray, Small)
    fmt_interval = f"\\small\\color{'{gray}'}{rew_interval_disp}"

    return f"\\shortstack{{{fmt_mean} {delta_str} \\\\ {fmt_interval}}}"

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

    # --- Step 2: Determine Highlight (Best Reward Mean per Column) ---
    # In transposed view: "Column" = (Model, Dataset). 
    # We compare across "Rows" (Algorithms).
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

    # --- Step 3: Generate Table ---
    latex = []
    latex.append(r"% Requires \usepackage{booktabs}, \usepackage{xcolor}, \usepackage{multirow}")
    latex.append(r"\begin{table*}[t]")
    latex.append(r"\centering")
    latex.append(r"\resizebox{\textwidth}{!}{%")
    
    # Structure: Algo | Qwen3B(GSM, Med) | Llama3B(GSM, Med) ...
    # 1 label col + (4 models * 2 datasets) = 9 columns
    latex.append(r"\begin{tabular}{l cc cc cc cc}")
    latex.append(r"\toprule")
    
    # Header 1: Backbones
    # \cmidrule spans the 2 columns for each model
    # Indices: Col 1 is Algo. Qwen3B is 2-3. Llama3B is 4-5. Qwen7B is 6-7. Llama8B is 8-9.
    header_1 = r"& \multicolumn{2}{c}{\textbf{" + MODEL_ORDER[0][1] + r"}}" + \
               r"& \multicolumn{2}{c}{\textbf{" + MODEL_ORDER[1][1] + r"}}" + \
               r"& \multicolumn{2}{c}{\textbf{" + MODEL_ORDER[2][1] + r"}}" + \
               r"& \multicolumn{2}{c}{\textbf{" + MODEL_ORDER[3][1] + r"}} \\"
    latex.append(header_1)
    latex.append(r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9}")
    
    # Header 2: Datasets
    # Repeat GSM8K & MedReason for each model
    ds_row_items = []
    for _ in MODEL_ORDER:
        ds_row_items.append(r"\textbf{\textsc{GSM8K}}")
        ds_row_items.append(r"\textbf{\textsc{MedReason}}")

    latex.append(r"\textbf{Method} & " + " & ".join(ds_row_items) + r" \\")
    latex.append(r"\midrule")

    # Rows: Algorithms
    for algo_key, algo_name in ALGO_ORDER:
        row_cells = []
        
        # Iterate Models then Datasets
        for model_key, _ in MODEL_ORDER:
            for dataset in DATASETS:
                entry = data[model_key][algo_key][dataset]
                
                # Format
                cell_str = format_compact_cell(entry, entry.get('is_best'), entry.get('is_second'))
                row_cells.append(cell_str)
        
        # Add extra vertical spacing between rows because of \shortstack
        latex.append(f"{algo_name} & " + " & ".join(row_cells) + r" \\[0.5em]")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}%")
    latex.append(r"}")
    latex.append(r"\caption{\textbf{Best-of-N Reranking Improvement.} Values represent Best-of-N mean (\%) and 95\% confidence interval. The values in parenthesis denote the percentage point improvement ($\Delta$) over random selection. \textcolor{teal}{Teal} indicates positive improvement, \textcolor{purple}{pink} negative.}")
    latex.append(r"\label{tab:reranking_transposed}")
    latex.append(r"\end{table*}")

    output_file = os.path.join(ROOT_DIR, "results_reranking_transposed.txt")
    with open(output_file, "w") as f:
        f.write("\n".join(latex))
    
    print(f"Success! Created '{output_file}'")

if __name__ == "__main__":
    main()