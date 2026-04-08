import os
import re

# ================= CONFIGURATION =================
ROOT_DIR = os.path.join('figures', 'answer_only')

# The source of generations (as shown in your drawing header)
POLICY_MODEL = 'qwen7b'

# Columns: Reward Model Backbones
RM_MODELS = [
    ('llama3b', 'Llama 3.2-3B'),
    ('qwen3b', 'Qwen 2.5-3B'),
    ('llama8b', 'Llama 3.1-8B')
]

# Rows/Sub-cols: Datasets
DATASETS = [
    ('math', 'GSM8K'),
    ('medicine', 'MedReason'),
    ('mmlu', 'MMLU-Pro')
]

# We assume a specific reward method to show (e.g., 'Step-wise')
METHOD_KEY = 'partial' 

VALUE_PATTERN = r"(\d+\.\d+\s*\[\s*\d+\.\d+,\s*\d+\.\d+\s*\])"
NUM_GENERATIONS = 16 

def extract_values(filepath):
    """Parses Random baseline and Reward reranking score."""
    results = {'Random': None, 'Reward': None}
    if not os.path.exists(filepath):
        return results

    with open(filepath, 'r') as f:
        content = f.read().replace('\n', ' ')
    
    rows = content.split(r'\\')
    for row in rows:
        matches = re.findall(VALUE_PATTERN, row)
        if not matches: continue
        
        row_lower = row.lower()
        if "random" in row_lower:
            results['Random'] = matches[0]
        elif "reasoning reranking" in row_lower or "reward" in row_lower:
            results['Reward'] = matches[0]
    return results

def get_delta(res):
    """Calculates Reward - Random in percentage points."""
    if not res['Random'] or not res['Reward']: return None
    try:
        rand = float(res['Random'].split()[0]) * 100
        rew = float(res['Reward'].split()[0]) * 100
        # Return as rounded integer
        return round(rew - rand)
    except:
        return None

def main():
    latex = []
    latex.append(r"\begin{table*}[h!]")
    latex.append(r"\centering")
    latex.append(r"\scriptsize")
    
    latex.append(r"\begin{tabular}{l rrr rrr rrr}")
    latex.append(r"\toprule")
    
    # Header Row 1: Backbones
    header_row1 = [r"\multicolumn{1}{c}{\texttt{Qwen2.5-7B} }"]
    for _, model_name in RM_MODELS:
        header_row1.append(f"\\multicolumn{{3}}{{c}}{{ \\texttt{{{model_name}}} }}")
    latex.append(" & ".join(header_row1) + r" \\")
    
    # Header Row 2: Reward Datasets
    header_row2 = [r"\textit{(step-wise)}"] 
    for _ in RM_MODELS:
        for _, ds_name in DATASETS:
            header_row2.append(f"\\tiny \\textsc{{{ds_name}}}")
    latex.append(" & ".join(header_row2) + r" \\")
    latex.append(r"\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10}")

    # --- Data Rows ---
    for i, (p_ds_key, p_ds_name) in enumerate(DATASETS):
        row_cells = []
        row_cells.append(f"\\textsc{{{p_ds_name}}}")

        for rm_key, _ in RM_MODELS:
            for r_ds_key, _ in DATASETS:
                folder_name = f"transfer_{rm_key}_{METHOD_KEY}_P_{p_ds_key}_R_{r_ds_key}"
                fpath = os.path.join(ROOT_DIR, p_ds_key, folder_name, f'pass_at_k_table_reranking_{NUM_GENERATIONS}.txt')
                print(f"Extracting from {fpath}...")
                res = extract_values(fpath)
                delta = get_delta(res)
                
                if delta is None:
                    row_cells.append("-")
                else:
                    # Formatting logic
                    color = "insightteal" if delta >= 0 else "purple"
                    arrow = "$\\uparrow$" if delta >= 0 else "$\\downarrow$"
                    # Use .0f for full percentage (no decimals)
                    val_text = f"{abs(delta):.0f}"
                    
                    cell_str = f"\\textcolor{{{color}}}{{{arrow} {val_text}}}"
                    
                    if p_ds_key == r_ds_key:
                        cell_str = f"\\textbf{{{cell_str}}}" 
                    
                    row_cells.append(cell_str)
        
        latex.append(" & ".join(row_cells) + r" \\")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    
    caption = f"\\caption[{{Reward Model Transferability (Best-of-{NUM_GENERATIONS} $\\Delta$ over Random)}}]{{\\textbf{{Reward Model Transferability (Best-of-{NUM_GENERATIONS} $\\Delta$ over Random).}} Generations are sourced from \\texttt{{Qwen2.5-7B}} SFT models. We score them using Reward Models trained on different task distributions. Diagonal entries (bold) represent in-distribution performance. Values are rounded to the nearest percentage point.}}"
    latex.append(caption)
    latex.append(r"\end{table*}")

    output_file = os.path.join(ROOT_DIR, "transfer_matrix.txt")
    with open(output_file, "w") as f:
        f.write("\n".join(latex))
    
    print(f"Created transfer matrix at {output_file}")

if __name__ == "__main__":
    main()