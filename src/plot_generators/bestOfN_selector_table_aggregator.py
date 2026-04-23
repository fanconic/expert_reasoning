import os
from pathlib import Path

try:
    from src.plot_generators.run_paths import resolve_run_dir
except ModuleNotFoundError:
    from run_paths import resolve_run_dir

# ================= CONFIGURATION =================

ROOT_DIR = os.path.join("figures", "answer_only")
SELECTOR_FILE = "pass_at_k_table_selector_variants_16.txt"

# Rows: The Methods
ALGO_ORDER = [
    ("sparse", r"\textit{Sparse}"),
    ("partial", r"\textit{Step-wise}"),
    ("partial_fixed", r"\textit{Interval}"),
    ("full", r"\textit{Dense}"),
]

# Backbones (Grouped rows)
MODEL_ORDER = [
    ("qwen3b", r"\texttt{Qwen2.5-3B}"),
    ("llama3b", r"\texttt{Llama3.2-3B}"),
    ("qwen7b", r"\texttt{Qwen2.5-7B}"),
    ("llama8b", r"\texttt{Llama3.1-8B}"),
    ("qwen4b", r"\texttt{Qwen3-4B}"),
]

DATASET_COLUMNS = [
    ("math", r"\textbf{\textsc{GSM8K}}"),
    ("medicine", r"\textbf{\textsc{MedReason}}"),
    ("mmlu", r"\textbf{\textsc{MMLU-Pro}}"),
]

SELECTOR_STRATEGIES = [
    ("discounted", "Discounted Mean (gamma=0.95) [current]", r"\texttt{Discounted}"),
    ("mean", "Uniform Mean", r"\texttt{Mean}"),
    ("last", "Last Token", r"\texttt{Last}"),
    ("tail3", "Tail Mean (k=3)", r"\texttt{Tail-3}"),
    ("top3", "Top-3 Mean", r"\texttt{Top-3}"),
    ("softmax2", "Softmax-Weighted (beta=2)", r"\texttt{Softmax-2}"),
    ("power2", "Power Mean (p=2)", r"\texttt{Power-2}"),
    ("trimmed10", r"Trimmed Mean (10\%)", r"\texttt{Trimmed-10}"),
    ("answer_boost2", "Answer-Boosted Mean (x2)", r"\texttt{Ans-Boost}"),
]

CURRENT_STRATEGY_KEY = "discounted"
LABEL_TO_KEY = {label.replace(r"\%", "%"): key for key, label, _ in SELECTOR_STRATEGIES}
KEY_TO_SHORT = {key: short for key, _, short in SELECTOR_STRATEGIES}


def extract_selector_variant_p1(filepath):
    """
    Reads pass@1 from pass_at_k_table_selector_variants_16.txt.
    Returns: {strategy_key -> pass@1 float|None}
    """
    results = {key: None for key, _, _ in SELECTOR_STRATEGIES}
    if not os.path.exists(filepath):
        return results

    with open(filepath, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line.endswith(r"\\"):
                continue
            if line.startswith("\\") or line.startswith("%"):
                continue

            parts = [p.strip() for p in line[:-2].split("&")]
            if len(parts) < 2:
                continue

            label = parts[0].replace(r"\%", "%")
            key = LABEL_TO_KEY.get(label)
            if key is None:
                continue

            try:
                pass1 = float(parts[1])
            except ValueError:
                continue

            results[key] = pass1

    return results


def format_pct(value, is_best=False, is_second=False):
    if value is None:
        return "-"

    pct = value * 100.0
    disp = f"{pct:.1f}"

    if pct < 15.0:
        return f"\\textcolor{{gray}}{{{disp}$^*$}}"

    if is_best:
        return f"\\textbf{{{disp}}}"
    if is_second:
        return f"\\underline{{{disp}}}"
    return disp


def format_delta(delta):
    """delta is in absolute units (0-1). Display in percentage points."""
    if delta is None:
        return "-"

    delta_pp = delta * 100.0
    d_text = f"{delta_pp:+.1f}"

    if delta_pp > 0:
        return f"\\textbf{{\\textcolor{{insightteal}}{{($\\uparrow$ {d_text})}}}}"
    if delta_pp < 0:
        return f"\\textcolor{{purple}}{{($\\downarrow$ {d_text})}}"
    return f"({d_text})"


def format_winner(strategy_key):
    if not strategy_key:
        return "-"
    return KEY_TO_SHORT.get(strategy_key, strategy_key)


def main():
    # Raw strategy scores:
    # data[model][algo][dataset][strategy] = pass@1
    data = {
        model_key: {
            algo_key: {
                dataset_key: {key: None for key, _, _ in SELECTOR_STRATEGIES}
                for dataset_key, _ in DATASET_COLUMNS
            }
            for algo_key, _ in ALGO_ORDER
        }
        for model_key, _ in MODEL_ORDER
    }

    # Summary per run:
    # summary[model][algo][dataset] = {current, best, delta, winner, is_best, is_second}
    summary = {
        model_key: {
            algo_key: {
                dataset_key: {
                    "current": None,
                    "best": None,
                    "delta": None,
                    "winner": None,
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
                run_name = f"{model_key}_{algo_key}"
                run_dir = resolve_run_dir(ds_path, run_name, required_file=SELECTOR_FILE)
                fpath = (
                    run_dir / SELECTOR_FILE
                    if run_dir is not None
                    else ds_path / run_name / SELECTOR_FILE
                )

                data[model_key][algo_key][dataset_key] = extract_selector_variant_p1(fpath)

    # --- Step 2: Compute per-run best strategy + delta over current ---
    for model_key, _ in MODEL_ORDER:
        for algo_key, _ in ALGO_ORDER:
            for dataset_key, _ in DATASET_COLUMNS:
                scores = data[model_key][algo_key][dataset_key]
                current = scores.get(CURRENT_STRATEGY_KEY)
                valid_scores = {
                    key: value for key, value in scores.items() if value is not None
                }

                if valid_scores:
                    winner = max(valid_scores, key=valid_scores.get)
                    best = valid_scores[winner]
                else:
                    winner = None
                    best = None

                delta = None
                if best is not None and current is not None:
                    delta = best - current

                summary[model_key][algo_key][dataset_key]["current"] = current
                summary[model_key][algo_key][dataset_key]["best"] = best
                summary[model_key][algo_key][dataset_key]["delta"] = delta
                summary[model_key][algo_key][dataset_key]["winner"] = winner

    # --- Step 3: Highlighting (best/second best by best-score per model+dataset) ---
    for model_key, _ in MODEL_ORDER:
        for dataset_key, _ in DATASET_COLUMNS:
            values = []
            for algo_key, _ in ALGO_ORDER:
                value = summary[model_key][algo_key][dataset_key]["best"]
                if value is not None:
                    values.append(value)

            unique = sorted(list(set(values)), reverse=True)
            best_val = unique[0] if unique else None
            second_val = unique[1] if len(unique) > 1 else None

            for algo_key, _ in ALGO_ORDER:
                value = summary[model_key][algo_key][dataset_key]["best"]
                summary[model_key][algo_key][dataset_key]["is_best"] = (
                    value is not None and best_val is not None and value == best_val
                )
                summary[model_key][algo_key][dataset_key]["is_second"] = (
                    value is not None and second_val is not None and value == second_val
                )

    # --- Step 4: Generate LaTeX ---
    latex = []
    latex.append(
        r"% Requires \usepackage{booktabs}, \usepackage{xcolor}, \usepackage{multirow}"
    )
    latex.append(r"\begin{table*}[t]")
    latex.append(r"\centering")
    latex.append(r"\scriptsize")
    latex.append(r"\resizebox{\textwidth}{!}{%")
    latex.append(r"\begin{tabular}{ll cccc cccc cccc}")
    latex.append(r"\toprule")

    dataset_headers = " & ".join(
        [
            rf"\multicolumn{{4}}{{c}}{{{dataset_label}}}"
            for _, dataset_label in DATASET_COLUMNS
        ]
    )
    latex.append(rf"& & {dataset_headers} \\")
    latex.append(r"\cmidrule(lr){3-6} \cmidrule(lr){7-10} \cmidrule(lr){11-14}")
    latex.append(
        r"\textbf{Backbone} & \textbf{Method} & Current & Best & $\Delta$ (pp) & Winner & Current & Best & $\Delta$ (pp) & Winner & Current & Best & $\Delta$ (pp) & Winner \\"
    )
    latex.append(r"\midrule")

    for model_key, model_name in MODEL_ORDER:
        first_row = True
        num_algos = len(ALGO_ORDER)

        for algo_key, algo_name in ALGO_ORDER:
            row_cells = []
            for dataset_key, _ in DATASET_COLUMNS:
                entry = summary[model_key][algo_key][dataset_key]
                current_cell = format_pct(entry["current"])
                best_cell = format_pct(
                    entry["best"], entry["is_best"], entry["is_second"]
                )
                delta_cell = format_delta(entry["delta"])
                winner_cell = format_winner(entry["winner"])
                row_cells.extend([current_cell, best_cell, delta_cell, winner_cell])

            if first_row:
                model_col = (
                    f"\\multirow{{{num_algos}}}{{*}}{{\\textbf{{{model_name}}}}}"
                )
            else:
                model_col = ""

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
        r"\caption{\textbf{Best-of-16 Reranking with Dense-Reward Aggregation Strategies (\%).} "
        r"Each run reports the current selector (\texttt{Discounted}), the best selector among all tested aggregation strategies, "
        r"the percentage-point gain over current, and the winning strategy ID. "
        r"\textbf{Bold} marks the best and \underline{underline} the second-best \textit{Best} score among methods for each backbone and dataset. "
        r"\textcolor{insightteal}{Blue} indicates positive gain and \textcolor{purple}{purple} negative gain.}"
    )
    latex.append(r"\label{tab:reranking_selector_variants_long}")
    latex.append(r"\end{table*}")

    output_file = os.path.join(ROOT_DIR, "results_reranking_selector_variants_long.txt")
    with open(output_file, "w") as f:
        f.write("\n".join(latex))

    print(f"Success! Created '{output_file}'")


if __name__ == "__main__":
    main()
