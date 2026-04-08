from pathlib import Path
from typing import Dict, List, Optional, Set

# ================= CONFIGURATION =================

ROOT_DIR = Path("figures") / "answer_only" / "mmlu"
RERANK_FILE = "pass_at_k_table_reranking_by_category.txt"
OUTPUT_FILE = (
    Path("figures") / "answer_only" / "results_reranking_long_mmlu_categories.txt"
)

# Rows: The Methods (bestOfN style)
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

METHOD_MAP = {
    "Random Reranking": "Random",
    "Reasoning Reranking": "Reward",
}


def latex_escape(text: str) -> str:
    escaped = str(text)
    escaped = escaped.replace("\\", r"\textbackslash{}")
    escaped = escaped.replace("&", r"\&")
    escaped = escaped.replace("%", r"\%")
    escaped = escaped.replace("$", r"\$")
    escaped = escaped.replace("#", r"\#")
    escaped = escaped.replace("_", r"\_")
    escaped = escaped.replace("{", r"\{")
    escaped = escaped.replace("}", r"\}")
    escaped = escaped.replace("~", r"\textasciitilde{}")
    escaped = escaped.replace("^", r"\textasciicircum{}")
    return escaped


def capitalize_category(name: str) -> str:
    return " ".join(part.capitalize() for part in name.split())


def parse_reranking_by_category(path: Path) -> Dict[str, Dict[str, Optional[float]]]:
    """
    Parse rows from pass_at_k_table_reranking_by_category.txt.
    Returns:
      {category: {"Random": float|None, "Reward": float|None}}
    """
    parsed: Dict[str, Dict[str, Optional[float]]] = {}

    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line.endswith(r"\\"):
                continue
            if line.startswith("\\") or line.startswith("%"):
                continue

            parts = [p.strip() for p in line[:-2].split("&")]
            if len(parts) < 6:
                continue

            category, method_name, pass1_str = parts[0], parts[1], parts[2]
            mapped = METHOD_MAP.get(method_name)
            if mapped is None:
                continue

            try:
                pass1_val = float(pass1_str)
            except ValueError:
                continue

            if category not in parsed:
                parsed[category] = {"Random": None, "Reward": None}
            parsed[category][mapped] = pass1_val

    return parsed


def discover_run_dirs(root_dir: Path) -> List[Path]:
    if not root_dir.exists():
        return []
    runs = []
    for f in sorted(root_dir.rglob(RERANK_FILE)):
        d = f.parent
        if d.name.startswith("transfer_"):
            continue
        runs.append(d)
    # de-duplicate
    deduped = []
    seen = set()
    for d in runs:
        if d in seen:
            continue
        deduped.append(d)
        seen.add(d)
    return deduped


def select_variant_run(
    model_key: str,
    algo_key: str,
    parsed_by_run: Dict[str, Dict[str, Dict[str, Optional[float]]]],
) -> Optional[str]:
    preferred = [
        f"{model_key}_{algo_key}_new",
        f"{model_key}_{algo_key}",
        f"{model_key}_{algo_key}_new_from_sft",
        f"{model_key}_{algo_key}_from_sft",
    ]
    for name in preferred:
        if name in parsed_by_run:
            return name

    prefix = f"{model_key}_{algo_key}"
    fallback = sorted(
        name
        for name in parsed_by_run
        if name.startswith(prefix) and not name.startswith("transfer_")
    )
    return fallback[0] if fallback else None


def get_mean_from_float(v: Optional[float]) -> float:
    if v is None:
        return -100.0
    return v * 100.0


def format_cell(
    val: Optional[float], is_best: bool = False, is_second: bool = False
) -> str:
    """
    Formats percentage with no decimals (bestOfN style).
    """
    if val is None:
        return "-"

    mean_val = val * 100.0
    mean_disp = f"{mean_val:.0f}"

    if mean_val < 15.0:
        return f"\\textcolor{{gray}}{{{mean_disp}$^*$}}"

    # Keep same spirit as bestOfN script (no bold/underline currently applied there)
    _ = is_best
    _ = is_second
    return mean_disp


def format_delta(rand_val: Optional[float], rew_val: Optional[float]) -> str:
    if rand_val is None or rew_val is None:
        return "-"

    delta = (rew_val - rand_val) * 100.0
    d_text = f"{delta:+.0f}"

    if delta > 0:
        return f"\\textbf{{\\textcolor{{insightteal}}{{($\\uparrow$ {d_text})}}}}"
    if delta < 0:
        return f"\\textcolor{{purple}}{{($\\downarrow$ {d_text})}}"
    return f"({d_text})"


def main() -> None:
    run_dirs = discover_run_dirs(ROOT_DIR)
    if not run_dirs:
        print(f"No runs with {RERANK_FILE} found in: {ROOT_DIR}")
        return

    parsed_by_run: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}
    categories_set: Set[str] = set()

    for run_dir in run_dirs:
        parsed = parse_reranking_by_category(run_dir / RERANK_FILE)
        parsed_by_run[run_dir.name] = parsed
        categories_set.update(parsed.keys())

    categories = sorted(categories_set, key=lambda s: s.lower())
    if not categories:
        print(f"No category reranking rows found under: {ROOT_DIR}")
        return

    # data[model][algo][category] = {"Random": val, "Reward": val, "is_best": bool, "is_second": bool}
    data: Dict[str, Dict[str, Dict[str, Dict[str, Optional[float]]]]] = {
        model_key: {
            algo_key: {
                cat: {
                    "Random": None,
                    "Reward": None,
                    "is_best": False,
                    "is_second": False,
                }
                for cat in categories
            }
            for algo_key, _ in ALGO_ORDER
        }
        for model_key, _ in MODEL_ORDER
    }

    # --- Step 1: Extraction ---
    for model_key, _ in MODEL_ORDER:
        for algo_key, _ in ALGO_ORDER:
            run_name = select_variant_run(model_key, algo_key, parsed_by_run)
            if run_name is None:
                continue

            parsed = parsed_by_run[run_name]
            for cat in categories:
                entry = parsed.get(cat, {})
                data[model_key][algo_key][cat]["Random"] = entry.get("Random")
                data[model_key][algo_key][cat]["Reward"] = entry.get("Reward")

    # --- Step 2: Ranking Logic (Best Reward per Model/Category) ---
    for model_key, _ in MODEL_ORDER:
        for cat in categories:
            means = []
            for algo_key, _ in ALGO_ORDER:
                m = get_mean_from_float(data[model_key][algo_key][cat]["Reward"])
                if m > -1:
                    means.append(m)

            unique = sorted(list(set(means)), reverse=True)
            best = unique[0] if unique else -99
            second = unique[1] if len(unique) > 1 else -99

            for algo_key, _ in ALGO_ORDER:
                m = get_mean_from_float(data[model_key][algo_key][cat]["Reward"])
                data[model_key][algo_key][cat]["is_best"] = m == best and m > -1
                data[model_key][algo_key][cat]["is_second"] = m == second and m > -1

    # --- Step 3: Generate LaTeX ---
    latex: List[str] = []
    latex.append(
        r"% Requires \usepackage{booktabs}, \usepackage{xcolor}, \usepackage{multirow}"
    )
    latex.append(r"\begin{table*}[t]")
    latex.append(r"\centering")
    latex.append(r"\scriptsize")
    latex.append(r"\resizebox{\textwidth}{!}{%")

    # Cols: Backbone | Method | (Category x (Random, Reward, Delta))
    col_spec = "ll " + " ".join(["ccc"] * len(categories))
    latex.append(rf"\begin{{tabular}}{{{col_spec}}}")
    latex.append(r"\toprule")

    # Header 1: Category groups
    group_header = " & ".join(
        rf"\multicolumn{{3}}{{c}}{{\textbf{{\textsc{{{latex_escape(capitalize_category(cat))}}}}}}}"
        for cat in categories
    )
    latex.append(f"& & {group_header} \\\\")

    # Header rules for each category group
    cmidrules = []
    start_col = 3
    for idx in range(len(categories)):
        left = start_col + 3 * idx
        right = left + 2
        cmidrules.append(rf"\cmidrule(lr){{{left}-{right}}}")
    latex.append(" ".join(cmidrules))

    # Header 2: Metrics repeated
    metric_header = " & ".join(["Random & Reward & $\\Delta$ (pp)"] * len(categories))
    latex.append(rf"\textbf{{Backbone}} & \textbf{{Method}} & {metric_header}\\")
    latex.append(r"\midrule")

    for model_key, model_name in MODEL_ORDER:
        first_row = True
        num_algos = len(ALGO_ORDER)

        for algo_key, algo_name in ALGO_ORDER:
            row_cells: List[str] = []

            for cat in categories:
                entry = data[model_key][algo_key][cat]
                rand_cell = format_cell(entry["Random"])
                rew_cell = format_cell(
                    entry["Reward"],
                    entry.get("is_best", False),
                    entry.get("is_second", False),
                )
                delta_cell = format_delta(entry["Random"], entry["Reward"])
                row_cells.extend([rand_cell, rew_cell, delta_cell])

            model_col = (
                rf"\multirow{{{num_algos}}}{{*}}{{\textbf{{{model_name}}}}}"
                if first_row
                else ""
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
        r"\caption{\textbf{Best-of-N Reranking Performance by MMLU Category (\%).} "
        r"Each category reports Random selection, Reward reranking, and the percentage-point delta. "
        r"\textcolor{insightteal}{Blue} indicates positive delta and \textcolor{purple}{purple} negative delta. "
        r"* marks values below 15\%.}"
    )
    latex.append(r"\label{tab:reranking_long_mmlu_categories}")
    latex.append(r"\end{table*}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text("\n".join(latex), encoding="utf-8")
    print(f"Success! Created '{OUTPUT_FILE}'")


if __name__ == "__main__":
    main()
