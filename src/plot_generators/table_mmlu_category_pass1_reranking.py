import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


ROOT_DIR = Path("figures") / "answer_only" / "mmlu"
PASS_FILE = "pass_at_k_table_by_category.txt"
DEFAULT_OUTPUT = Path("figures") / "answer_only" / "results_p1_mmlu_categories.txt"

# Match p1_table_aggregator row structure
ALGO_ORDER = [
    ("GRPO", "Verifiable Reward"),
    ("SFT", "SFT"),
    ("sparse", r"Ours (\textit{Sparse})"),
    ("partial", r"Ours (\textit{Step-wise})"),
    ("partial_fixed", r"Ours (\textit{Interval})"),
    ("full", r"Ours (\textit{Dense})"),
]

# Match broad model ordering used in other aggregators
MODEL_ORDER = [
    ("qwen3b", r"\texttt{Qwen2.5-3B}"),
    ("llama3b", r"\texttt{Llama3.2-3B}"),
    ("qwen7b", r"\texttt{Qwen2.5-7B}"),
    ("llama8b", r"\texttt{Llama3.1-8B}"),
    ("qwen4b", r"\texttt{Qwen3-4B}"),
]

METHOD_MAP = {
    "Outcome Sup.": "GRPO",
    "AIRL (ours)": "AIRL",
    "SFT": "SFT",
}

FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?"
PASS_WITH_BOUNDS_RE = re.compile(
    rf"^\s*({FLOAT_PATTERN})\s*\[\s*({FLOAT_PATTERN})\s*,\s*({FLOAT_PATTERN})\s*\]\s*$"
)

# (mean, optional half-width CI), both in probability units (0-1)
Pass1Entry = Tuple[float, Optional[float]]


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


def parse_pass1_entry(pass1_raw: str) -> Optional[Pass1Entry]:
    """Parse pass@1 with optional CI bounds encoded as `mean [low, high]`."""
    pass1_raw = pass1_raw.strip()
    if not pass1_raw:
        return None

    match = PASS_WITH_BOUNDS_RE.match(pass1_raw)
    if match:
        mean_val = float(match.group(1))
        low = float(match.group(2))
        high = float(match.group(3))
        if low <= high:
            return mean_val, max(0.0, (high - low) / 2.0)
        return mean_val, None

    try:
        return float(pass1_raw), None
    except ValueError:
        return None


def parse_optional_ci_from_columns(
    parts: List[str], mean_val: float
) -> Optional[float]:
    """
    If rows contain explicit pass@1 lower/upper bound columns after pass@1,
    parse them as half-width CI. Returns None when unavailable.
    """
    if len(parts) < 8:
        return None

    try:
        low = float(parts[3].strip())
        high = float(parts[4].strip())
    except ValueError:
        return None

    if low <= mean_val <= high:
        return max(0.0, (high - low) / 2.0)
    return None


def parse_category_pass_table(path: Path) -> Dict[str, Dict[str, Pass1Entry]]:
    """
    Parse rows from pass_at_k_table_by_category.txt.
    Returns mapping: method_key -> {category -> (pass@1_mean, optional_half_width_ci)}
      method_key in {"GRPO", "SFT", "AIRL"}
    """
    parsed: Dict[str, Dict[str, Pass1Entry]] = {"GRPO": {}, "SFT": {}, "AIRL": {}}

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

            parsed_entry = parse_pass1_entry(pass1_str)
            if parsed_entry is None:
                continue
            mean_val, half_ci = parsed_entry

            if half_ci is None:
                half_ci = parse_optional_ci_from_columns(parts, mean_val)

            parsed[mapped][category] = (mean_val, half_ci)

    return parsed


def format_cell(
    val: Optional[float],
    half_ci: Optional[float] = None,
    is_best: bool = False,
    is_second: bool = False,
    is_collapsed: bool = False,
) -> str:
    if val is None:
        return "-"

    mean_disp = f"{val * 100:.1f}"
    ci_disp = (
        f" {{\\scriptsize\\color{{gray}}$\\pm$ {half_ci * 100:.1f}}}"
        if half_ci is not None
        else ""
    )

    if is_collapsed:
        return f"\\textcolor{{gray}}{{{mean_disp}$^*$}}{ci_disp}"

    if is_best:
        return f"\\textbf{{{mean_disp}}}{ci_disp}"
    if is_second:
        return f"\\underline{{{mean_disp}}}{ci_disp}"
    return f"{mean_disp}{ci_disp}"


def find_run_dirs(root_dir: Path) -> List[Path]:
    if not root_dir.exists():
        return []
    run_dirs = sorted({p.parent for p in root_dir.rglob(PASS_FILE)})
    return [d for d in run_dirs if d.is_dir()]


def ordered_candidates_for_model(model_key: str, run_names: List[str]) -> List[str]:
    candidates = sorted(name for name in run_names if name.startswith(model_key))
    if model_key in candidates:
        candidates.insert(0, candidates.pop(candidates.index(model_key)))
    return candidates


def first_existing_variant(
    model_key: str,
    algo_key: str,
    parsed_by_run: Dict[str, Dict[str, Dict[str, Pass1Entry]]],
) -> Optional[str]:
    exact_names = [
        f"{model_key}_{algo_key}",
        f"{model_key}_{algo_key}_new",
    ]
    for name in exact_names:
        if name in parsed_by_run:
            return name

    prefix = f"{model_key}_{algo_key}"
    fuzzy = sorted(name for name in parsed_by_run if name.startswith(prefix))
    return fuzzy[0] if fuzzy else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create p1-style MMLU category pass@1 LaTeX table (categories as columns)."
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=ROOT_DIR,
        help=f"Directory containing MMLU run folders (default: {ROOT_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output .txt path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    run_dirs = find_run_dirs(args.root_dir)
    if not run_dirs:
        print(f"No run folders with {PASS_FILE} found under: {args.root_dir}")
        return

    parsed_by_run: Dict[str, Dict[str, Dict[str, Pass1Entry]]] = {}
    all_categories: Set[str] = set()

    for run_dir in run_dirs:
        parsed = parse_category_pass_table(run_dir / PASS_FILE)
        parsed_by_run[run_dir.name] = parsed
        all_categories.update(parsed.get("GRPO", {}).keys())
        all_categories.update(parsed.get("SFT", {}).keys())
        all_categories.update(parsed.get("AIRL", {}).keys())

    categories = sorted(all_categories, key=lambda s: s.lower())
    if not categories:
        print(f"No category pass@1 rows found in: {args.root_dir}")
        return

    # data[model][algo][category] = (pass@1 mean, optional half-width CI) or None
    data: Dict[str, Dict[str, Dict[str, Optional[Pass1Entry]]]] = {
        model_key: {
            algo_key: {cat: None for cat in categories} for algo_key, _ in ALGO_ORDER
        }
        for model_key, _ in MODEL_ORDER
    }

    run_names = [d.name for d in run_dirs]

    for model_key, _ in MODEL_ORDER:
        candidates = ordered_candidates_for_model(model_key, run_names)

        # Baselines from first available run for this model (mirrors p1 approach)
        sft_found = False
        grpo_found = False
        for run_name in candidates:
            parsed = parsed_by_run[run_name]

            if not sft_found and parsed.get("SFT"):
                for cat, v in parsed["SFT"].items():
                    data[model_key]["SFT"][cat] = v
                sft_found = True

            if not grpo_found and parsed.get("GRPO"):
                for cat, v in parsed["GRPO"].items():
                    data[model_key]["GRPO"][cat] = v
                grpo_found = True

            if sft_found and grpo_found:
                break

        # AIRL variants
        for algo_key, _ in ALGO_ORDER:
            if algo_key in {"GRPO", "SFT"}:
                continue

            run_name = first_existing_variant(model_key, algo_key, parsed_by_run)
            if run_name is None:
                continue

            parsed = parsed_by_run[run_name]
            for cat, v in parsed.get("AIRL", {}).items():
                data[model_key][algo_key][cat] = v

    # Highlighting: best/second among SFT + AIRL variants, excluding GRPO
    formatted_data: Dict[str, Dict[str, Dict[str, str]]] = {
        model_key: {
            algo_key: {cat: "-" for cat in categories} for algo_key, _ in ALGO_ORDER
        }
        for model_key, _ in MODEL_ORDER
    }

    for model_key, _ in MODEL_ORDER:
        for cat in categories:
            means: List[float] = []
            for algo_key, _ in ALGO_ORDER:
                if algo_key == "GRPO":
                    continue
                entry = data[model_key][algo_key][cat]
                if entry is not None:
                    means.append(entry[0])

            unique_sorted = sorted(set(means), reverse=True)
            best_val = unique_sorted[0] if unique_sorted else None
            second_val = unique_sorted[1] if len(unique_sorted) > 1 else None

            for algo_key, _ in ALGO_ORDER:
                entry = data[model_key][algo_key][cat]
                if entry is None:
                    formatted_data[model_key][algo_key][cat] = "-"
                    continue
                val, half_ci = entry

                is_best = (
                    algo_key != "GRPO" and best_val is not None and val == best_val
                )
                is_second = (
                    algo_key != "GRPO" and second_val is not None and val == second_val
                )
                is_collapsed = val < 0.03

                formatted_data[model_key][algo_key][cat] = format_cell(
                    val,
                    half_ci=half_ci,
                    is_best=is_best,
                    is_second=is_second,
                    is_collapsed=is_collapsed,
                )

    # Build LaTeX in p1_table_aggregator style, but with categories as columns
    n_cols = 1 + len(categories)
    col_spec = "l " + " ".join(["c"] * len(categories))

    latex: List[str] = []
    latex.append(
        r"% Requires \usepackage{booktabs}, \usepackage{xcolor}, \usepackage{arydshln}"
    )
    latex.append(r"\begin{table*}[t]")
    latex.append(r"\centering")
    latex.append(r"\scriptsize")
    latex.append(r"\resizebox{\textwidth}{!}{%")
    latex.append(rf"\begin{{tabular}}{{{col_spec}}}")
    latex.append(r"\toprule")

    category_header = " & ".join(
        rf"\textbf{{\textsc{{{latex_escape(capitalize_category(cat))}}}}}"
        for cat in categories
    )
    latex.append(f"& {category_header} \\\\")
    latex.append(r"\midrule")

    for model_idx, (model_key, model_name) in enumerate(MODEL_ORDER):
        latex.append(rf"\multicolumn{{{n_cols}}}{{l}}{{\textbf{{{model_name}}}}} \\")

        for algo_key, algo_name in ALGO_ORDER:
            row_vals = [formatted_data[model_key][algo_key][cat] for cat in categories]
            latex.append(
                rf"\hspace{{1em}}{algo_name} & " + " & ".join(row_vals) + r" \\"
            )

            if algo_key == "GRPO":
                latex.append(
                    rf"\arrayrulecolor{{gray!60}}\cdashline{{1-{n_cols}}}\arrayrulecolor{{black}}"
                )

        if model_idx < len(MODEL_ORDER) - 1:
            latex.append(r"\midrule")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}%")
    latex.append(r"}")
    latex.append(
        r"\caption{\textbf{MMLU-Pro Pass@1 by Category (\%).} "
        r"Table layout matches the p1 table structure, with categories as columns. "
        r"\textbf{Bold} marks the best and underlining marks the second-best among SFT and AIRL variants. "
        r"Verifiable reward is shown as reference; * marks values below 20\%. "
        r"When available, uncertainty is shown as mean $\pm$ half-width of the 95\% confidence interval "
        r"(percentage points), rounded to one decimal place.}"
    )
    latex.append(r"\label{tab:mmlu_category_p1}")
    latex.append(r"\end{table*}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(latex), encoding="utf-8")

    print(f"Success! Created '{args.output}'")


if __name__ == "__main__":
    main()
