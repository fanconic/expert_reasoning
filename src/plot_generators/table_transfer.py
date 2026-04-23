import os
import re
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

try:
    import scienceplots  # noqa: F401
except Exception:
    scienceplots = None

try:
    plt.style.use("bright")
except OSError:
    pass
plt.rcParams["font.family"] = "sans-serif"

try:
    from src.plot_generators.run_paths import resolve_run_dir
except ModuleNotFoundError:
    from run_paths import resolve_run_dir

# ================= CONFIGURATION =================
_ROOT_CANDIDATES = [
    Path("figures") / "transferability_ablation_temp05_fixed",
    Path("figures") / "sft_reranking_temp05",
]
ROOT_DIR = next((p for p in _ROOT_CANDIDATES if p.exists()), _ROOT_CANDIDATES[0])

# The source of generations (as shown in your drawing header)
POLICY_MODEL = "llama8b"

# Columns: Reward Model Backbones
RM_MODELS = [
    #("llama3b", "Llama3.2-3B"),
    #("qwen3b", "Qwen2.5-3B"),
    #("llama8b", "Llama3.1-8B"),
    ("qwen7b", "Qwen2.5-7B"),
    ("qwen4b", "Qwen3-4B"),
]

# Rows/Sub-cols: Datasets
DATASETS = [("math", "GSM8K"), ("medicine", "MedReason"), ("mmlu", "MMLU-Pro")]

# We assume a specific reward method to show (e.g., 'Step-wise')
METHOD_KEY = "partial_fixed"

VALUE_PATTERN = r"(\d+\.\d+\s*\[\s*\d+\.\d+,\s*\d+\.\d+\s*\])"
NUM_GENERATIONS = 16


def resolve_data_root(root_dir: Path) -> Path:
    """Resolve dataset root; support both <root>/<ds> and <root>/answer_only/<ds>."""
    answer_only_root = root_dir / "answer_only"
    if answer_only_root.exists() and answer_only_root.is_dir():
        return answer_only_root
    return root_dir


def extract_reward(filepath: Path) -> str | None:
    """Parses Reasoning Reranking pass@1 from reranking table."""
    if not os.path.exists(filepath):
        return None

    with open(filepath, "r") as f:
        content = f.read().replace("\n", " ")

    rows = content.split(r"\\")
    for row in rows:
        matches = re.findall(VALUE_PATTERN, row)
        if not matches:
            continue

        row_lower = row.lower()
        if "reasoning reranking" in row_lower or "reward" in row_lower:
            return matches[0]
    return None


def extract_sft_pass1(filepath: Path) -> str | None:
    """Parses SFT pass@1 from pass_at_k_table.txt."""
    if not os.path.exists(filepath):
        return None

    with open(filepath, "r") as f:
        content = f.read().replace("\n", " ")

    rows = content.split(r"\\")
    for row in rows:
        if "sft" not in row.lower():
            continue
        matches = re.findall(VALUE_PATTERN, row)
        if not matches:
            continue
        return matches[0]
    return None


def _parse_mean_percent(val_str: str | None) -> float | None:
    if val_str is None:
        return None
    try:
        return float(val_str.split()[0]) * 100.0
    except Exception:
        return None


def get_delta(sft_val: str | None, reward_val: str | None) -> float | None:
    """Calculates Reward - SFT pass@1 in percentage points."""
    rand = _parse_mean_percent(sft_val)
    rew = _parse_mean_percent(reward_val)
    if rand is None or rew is None:
        return None
    return rew - rand


def _resolve_transfer_files(
    root_dir: Path,
    p_ds_key: str,
    r_ds_key: str,
    rm_key: str,
    method_key: str,
    num_generations: int,
) -> tuple[Path, Path]:
    """
    Resolve transfer files for both supported layouts.

    Layout A (new, split by reward dataset):
      <root>/R_<reward_ds>/answer_only/<policy_ds>/standard/<rm>_<method>/
          pass_at_k_table_reranking_<N>.txt
          pass_at_k_table.txt

    Layout B (legacy):
      <root>/answer_only/<policy_ds>/transfer_<rm>_<method>_P_<policy_ds>_R_<reward_ds>/
          pass_at_k_table_reranking_<N>.txt
          pass_at_k_table.txt
    """
    rerank_name = f"pass_at_k_table_reranking_{num_generations}.txt"
    run_name = f"{rm_key}_{method_key}"

    # Layout A: reward dataset split roots (R_math, R_medicine, R_mmlu)
    reward_root = root_dir / f"R_{r_ds_key}"
    if reward_root.exists() and reward_root.is_dir():
        reward_data_root = resolve_data_root(reward_root)
        ds_path = reward_data_root / p_ds_key
        run_dir = resolve_run_dir(ds_path, run_name, required_file=rerank_name)
        if run_dir is not None:
            return run_dir / rerank_name, run_dir / "pass_at_k_table.txt"

        # Fallback expected path under standard.
        fallback_run_dir = ds_path / "standard" / run_name
        return fallback_run_dir / rerank_name, fallback_run_dir / "pass_at_k_table.txt"

    # Layout B: transfer_* run folders under policy dataset root.
    data_root = resolve_data_root(root_dir)
    ds_path = data_root / p_ds_key
    folder_name = f"transfer_{rm_key}_{method_key}_P_{p_ds_key}_R_{r_ds_key}"
    run_dir = resolve_run_dir(ds_path, folder_name, required_file=rerank_name)
    if run_dir is not None:
        return run_dir / rerank_name, run_dir / "pass_at_k_table.txt"

    fallback_run_dir = ds_path / folder_name
    return fallback_run_dir / rerank_name, fallback_run_dir / "pass_at_k_table.txt"


def _load_transfer_matrices() -> dict[str, np.ndarray]:
    """
    Build a (policy-dataset x reward-dataset) delta matrix for each reward model key.
    Values are Reward - SFT pass@1 in percentage points, rounded to 1 decimal.
    Missing entries are NaN.
    """
    n = len(DATASETS)
    matrices: dict[str, np.ndarray] = {}

    for rm_key, _ in RM_MODELS:
        mat = np.full((n, n), np.nan, dtype=float)
        for i, (p_ds_key, _) in enumerate(DATASETS):
            for j, (r_ds_key, _) in enumerate(DATASETS):
                rerank_fpath, pass_fpath = _resolve_transfer_files(
                    root_dir=ROOT_DIR,
                    p_ds_key=p_ds_key,
                    r_ds_key=r_ds_key,
                    rm_key=rm_key,
                    method_key=METHOD_KEY,
                    num_generations=NUM_GENERATIONS,
                )
                print(f"Extracting reward from {rerank_fpath}...")
                print(f"Extracting SFT pass@1 from {pass_fpath}...")

                reward_val = extract_reward(rerank_fpath)
                sft_val = extract_sft_pass1(pass_fpath)
                delta = get_delta(sft_val, reward_val)
                if delta is None:
                    continue
                d = round(float(delta), 1)
                if abs(d) < 0.05:
                    d = 0.0
                mat[i, j] = d
        matrices[rm_key] = mat

    return matrices


def _render_latex_table(matrices: dict[str, np.ndarray]) -> str:
    latex = []
    latex.append(r"\begin{table*}[h!]")
    latex.append(r"\centering")
    latex.append(r"\resizebox{\textwidth}{!}{%")
    
    latex.append(r"\begin{tabular}{l rrr rrr rrr}")
    latex.append(r"\toprule")

    # Header Row 1: Backbones
    header_row1 = [r"\multicolumn{1}{c}{\texttt{Qwen2.5-7B} }"]
    for _, model_name in RM_MODELS:
        header_row1.append(f"\\multicolumn{{3}}{{c}}{{ \\texttt{{{model_name}}} }}")
    latex.append(" & ".join(header_row1) + r" \\")

    # Header Row 2: Reward Datasets
    header_row2 = [r"\textit{(interval)}"]
    for _ in RM_MODELS:
        for _, ds_name in DATASETS:
            header_row2.append(f"\\scriptsize \\textsc{{{ds_name}}}")
    latex.append(" & ".join(header_row2) + r" \\")
    latex.append(r"\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10}")

    # --- Data Rows ---
    for i, (p_ds_key, p_ds_name) in enumerate(DATASETS):
        row_cells = []
        row_cells.append(f"\\textsc{{{p_ds_name}}}")

        for rm_key, _ in RM_MODELS:
            for r_ds_key, _ in DATASETS:
                d = matrices.get(rm_key, np.full((len(DATASETS), len(DATASETS)), np.nan))[
                    i, [k for k, _ in DATASETS].index(r_ds_key)
                ]

                if np.isnan(d):
                    row_cells.append("-")
                else:
                    # Formatting logic
                    color = "insightteal" if d >= 0 else "purple"
                    arrow = "$\\uparrow$" if d >= 0 else "$\\downarrow$"
                    val_text = f"{abs(d):.1f}"

                    cell_str = f"\\textcolor{{{color}}}{{{arrow} {val_text}}}"

                    if p_ds_key == r_ds_key:
                        cell_str = f"\\textit{{{cell_str}}}"

                    row_cells.append(cell_str)

        latex.append(" & ".join(row_cells) + r" \\")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}}")

    caption = f"\\caption[{{Reward Model Transferability (Best-of-{NUM_GENERATIONS} $\\Delta$ over SFT Pass@1)}}]{{\\textbf{{Reward Model Transferability (Best-of-{NUM_GENERATIONS} $\\Delta$ over SFT Pass@1).}} Generations are sourced from \\texttt{{Qwen2.5-7B}} SFT models. We score them using Reward Models trained on different task distributions. Diagonal entries (italic) represent in-distribution performance. Values are rounded to one decimal place.}}"
    latex.append(caption)
    latex.append(r"\end{table*}")
    return "\n".join(latex)


def _plot_transfer_graphs(matrices: dict[str, np.ndarray], output_path: Path) -> None:
    """Create a directional transferability network plot (one panel per reward model)."""
    task_labels = [name for _, name in DATASETS]
    task_order = {name: idx for idx, name in enumerate(task_labels)}

    n_models = len(RM_MODELS)
    fig, axes = plt.subplots(1, n_models, figsize=(4.6 * n_models, 4.8))
    if n_models == 1:
        axes = [axes]

    for ax, (rm_key, rm_name) in zip(axes, RM_MODELS):
        mat = matrices.get(rm_key, np.full((len(DATASETS), len(DATASETS)), np.nan))
        G = nx.DiGraph()
        for task in task_labels:
            G.add_node(task)

        for i, source in enumerate(task_labels):
            for j, target in enumerate(task_labels):
                weight = mat[i, j]
                if np.isnan(weight):
                    continue
                G.add_edge(source, target, weight=float(weight))

        pos = nx.circular_layout(G)

        nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            node_size=2600,
            node_color="#f0f0f0",
            edgecolors="black",
            linewidths=1.2,
        )
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=11, font_weight="bold")

        for u, v, d in G.edges(data=True):
            weight = float(d["weight"])
            color = "#4575b4" if weight >= 0 else "#7b3294"
            width = 0.7 + abs(weight) * 0.35
            if u == v:
                rad = 0.35
            else:
                rad = 0.16 if task_order[u] < task_order[v] else -0.16
            nx.draw_networkx_edges(
                G,
                pos,
                ax=ax,
                edgelist=[(u, v)],
                edge_color=[color],
                width=width,
                arrows=True,
                arrowsize=18,
                connectionstyle=f"arc3,rad={rad}",
                min_source_margin=16,
                min_target_margin=16,
            )

        edge_labels = {(u, v): f"{d['weight']:+.1f}" for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(
            G, pos, ax=ax, edge_labels=edge_labels, label_pos=0.34, font_size=9
        )

        ax.set_title(f"Directional Transferability: {rm_name}", fontsize=12)
        ax.axis("off")

    fig.suptitle(
        rf"Transferability ($\Delta$ over SFT Pass@1, Best-of-{NUM_GENERATIONS})",
        fontsize=14,
        y=1.02,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    matrices = _load_transfer_matrices()
    latex = _render_latex_table(matrices)

    output_file = ROOT_DIR / "transfer_matrix_temp05.txt"
    with open(output_file, "w") as f:
        f.write(latex)

    graph_output = ROOT_DIR / "transfer_matrix_temp05_graph_all_models.pdf"
    _plot_transfer_graphs(matrices, graph_output)

    print(f"Created transfer matrix at {output_file}")
    print(f"Created transfer graph plot at {graph_output}")


if __name__ == "__main__":
    main()
