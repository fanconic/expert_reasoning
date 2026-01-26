"""
plot_main.py

Run plotting for multiple experiments across two domains (math + medicine).

Usage:
    python plot_main.py
    python plot_main.py --ckpt best_model
    python plot_main.py --no-token-figs
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Mapping, Union

from plot_helpers import ensure_dir, read_and_enhance, run_all_plots

# -------------------------------
# Config
# -------------------------------

DOMAINS: Dict[str, Path] = {
    "math": Path("/mnt/pdata/caf83/icml_math/outputs"),
    "medicine": Path("/mnt/pdata/caf83/icml_medicine/outputs"),
}

DEFAULT_CKPT = "best_model"

RunName = Union[str, Mapping[str, str]]  # either "foo" or {"math": "foo", "medicine": "bar"}

EXPERIMENTS: List[Dict[str, Any]] = [
    # Base runs (AIRL differs slightly by domain)
    # {
    #     "airl": {"math": "qwen3b_8ga_8gens", "medicine": "qwen3b_correct_corrupt_clipped"},
    #     "sft": "qwen3b_sft",
    #     "grpo": "qwen3b_grpo",
    #     "label": "qwen3b",
    # },
    # {
    #     "airl": {"math": "llama3b_8ga_8gens_reward_clipped", "medicine": "llama3b_correct_corrupt_clipped"},
    #     "sft": "llama3b_sft",
    #     "grpo": "llama3b_grpo",
    #     "label": "llama3b",
    # },
    # {
    #     "airl": {"math": "qwen7b_8ga_8gens", "medicine": "qwen7b_correct_corrupt_clipped"},
    #     "sft": "qwen7b_sft",
    #     "grpo": "qwen7b_grpo",
    #     "label": "qwen7b",
    # },
    # {
    #     "airl": {"math": "llama8b_8ga_8gens_clipped_reward", "medicine": "llama8b_correct_corrupt_clipped"},
    #     "sft": "llama8b_sft",
    #     "grpo": "llama8b_grpo",
    #     "label": "llama8b",
    # },

    # # Sparse
    # {"airl": "qwen3b_8ga_8gens_clipped_sparse", "sft": "qwen3b_sft", "grpo": "qwen3b_grpo", "label": "qwen3b_sparse"},
    # {"airl": "llama3b_8ga_8gens_clipped_sparse", "sft": "llama3b_sft", "grpo": "llama3b_grpo", "label": "llama3b_sparse"},
    # {"airl": "qwen7b_8ga_8gens_clipped_sparse", "sft": "qwen7b_sft", "grpo": "qwen7b_grpo", "label": "qwen7b_sparse"},
    # {"airl": "llama8b_8ga_8gens_clipped_sparse", "sft": "llama8b_sft", "grpo": "llama8b_grpo", "label": "llama8b_sparse"},

    # # Full
    # {"airl": "qwen3b_8ga_8gens_clipped_full", "sft": "qwen3b_sft", "grpo": "qwen3b_grpo", "label": "qwen3b_full"},
    # {"airl": "llama3b_8ga_8gens_clipped_full", "sft": "llama3b_sft", "grpo": "llama3b_grpo", "label": "llama3b_full"},
    # {"airl": "qwen7b_8ga_8gens_clipped_full", "sft": "qwen7b_sft", "grpo": "qwen7b_grpo", "label": "qwen7b_full"},
    # {"airl": "llama8b_8ga_8gens_clipped_full", "sft": "llama8b_sft", "grpo": "llama8b_grpo", "label": "llama8b_full"},

    # # OVR
    # {"airl": "qwen3b_ovr", "sft": "qwen3b_sft", "grpo": "qwen3b_grpo", "label": "qwen3b_ovr"},
    # {"airl": "llama3b_ovr", "sft": "llama3b_sft", "grpo": "llama3b_grpo", "label": "llama3b_ovr"},
    # {"airl": "qwen7b_ovr", "sft": "qwen7b_sft", "grpo": "qwen7b_grpo", "label": "qwen7b_ovr"},
    # {"airl": "llama8b_ovr", "sft": "llama8b_sft", "grpo": "llama8b_grpo", "label": "llama8b_ovr"},

    # # Partial fixed
    # {"airl": "qwen3b_partial_fixed", "sft": "qwen3b_sft", "grpo": "qwen3b_grpo", "label": "qwen3b_partial_fixed"},
    # {"airl": "llama3b_partial_fixed", "sft": "llama3b_sft", "grpo": "llama3b_grpo", "label": "llama3b_partial_fixed"},
    # {"airl": "qwen7b_partial_fixed", "sft": "qwen7b_sft", "grpo": "qwen7b_grpo", "label": "qwen7b_partial_fixed"},
    # {"airl": "llama8b_partial_fixed", "sft": "llama8b_sft", "grpo": "llama8b_grpo", "label": "llama8b_partial_fixed"},
    
     # switch reward models
    {"airl": "llama3b_switch_reward", "sft": "llama3b_sft", "grpo": "llama3b_grpo", "label": "llama3b_switch_reward"},
    {"airl": "llama8b_switch_reward", "sft": "llama8b_sft", "grpo": "llama8b_grpo", "label": "llama8b_switch_reward"},
    {"airl": "qwen3b_switch_reward", "sft": "qwen3b_sft", "grpo": "qwen3b_grpo", "label": "qwen3b_switch_reward"},
    {"airl": "qwen7b_switch_reward", "sft": "qwen7b_sft", "grpo": "qwen7b_grpo", "label": "qwen7b_switch_reward"},
    
     # humar error
    {"airl": "qwen3b_human_error", "sft": "qwen3b_sft", "grpo": "qwen3b_grpo", "label": "qwen3b_human_error"},
    {"airl": "llama3b_human_error", "sft": "llama3b_sft", "grpo": "llama3b_grpo", "label": "llama3b_human_error"},

]


# -------------------------------
# Helpers
# -------------------------------

# Add a global collector for failures
FAILED_PLOTS: List[str] = []

def resolve_run(name_or_map: RunName, domain: str) -> str:
    """Allow run name to be either a single string or per-domain mapping."""
    if isinstance(name_or_map, str):
        return name_or_map
    try:
        return name_or_map[domain]
    except KeyError as e:
        raise KeyError(f"Missing run name for domain='{domain}' in {name_or_map}") from e


def eval_jsonl(base: Path, run_name: str, ckpt: str) -> Path:
    return base / run_name / ckpt / "eval_results.jsonl"


def exp_label(exp: Dict[str, Any]) -> str:
    if exp.get("label"):
        return str(exp["label"])
    # fallback if label omitted
    return f"{exp['airl']}__{exp['sft']}__{exp['grpo']}"


def run_one_experiment(domain: str, base: Path, exp: Dict[str, Any], ckpt: str, make_token_figs: bool, answer_only: bool) -> None:
    airl_run = resolve_run(exp["airl"], domain)
    sft_run = resolve_run(exp["sft"], domain)     # usually str, but this keeps it flexible
    grpo_run = resolve_run(exp["grpo"], domain)

    airl_p = eval_jsonl(base, airl_run, ckpt)
    sft_p = eval_jsonl(base, sft_run, ckpt)
    grpo_p = eval_jsonl(base, grpo_run, ckpt)

    label = exp_label(exp)
    out_dir = Path("./figures") / ("answer_only" if answer_only else "full_cot") / domain / label
    ensure_dir(str(out_dir))

    missing = [p for p in (airl_p, sft_p, grpo_p) if not p.exists()]
    if missing:
        msg = f"{domain}/{label} (answer_only={answer_only}) ckpt={ckpt} - missing: {[str(m) for m in missing]}"
        FAILED_PLOTS.append(msg)
        print(f"[WARNING] Skipping '{answer_only}/{domain}/{label}/' — missing: {[str(m) for m in missing]}")
        return

    print(f"[INFO] Running '{answer_only}/{domain}/{label}/' (ckpt={ckpt})")

    try:
        df_airl = read_and_enhance(str(airl_p), answer_only=answer_only)
        df_sft = read_and_enhance(str(sft_p), answer_only=answer_only)
        df_grpo = read_and_enhance(str(grpo_p), answer_only=answer_only)
    except Exception as e:
        msg = f"{domain}/{label} (answer_only={answer_only}) ckpt={ckpt} - read_error: {e}"
        FAILED_PLOTS.append(msg)
        print(f"[ERROR] Failed to read/enhance for '{domain}/{label}': {e}")
        return

    try:
        run_all_plots(
            df_airl,
            df_sft,
            df_grpo,
            str(out_dir),
            num_generations=16,
            make_token_figs=make_token_figs,
        )
    except Exception as e:
        msg = f"{domain}/{label} (answer_only={answer_only}) ckpt={ckpt} - plotting_error: {e}"
        FAILED_PLOTS.append(msg)
        print(f"[ERROR] Plotting failed for '{domain}/{label}': {e}")
        return

    print(f"[DONE] Saved figures to: {out_dir}")


# -------------------------------
# Main
# -------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=DEFAULT_CKPT, help="Checkpoint folder name (default: best_model)")
    parser.add_argument("--no-token-figs", action="store_true", help="Skip token-level dense reward figures")
    args = parser.parse_args()

    for answer_only in [True, False]:  # can add False here if needed
        for domain, base_path in DOMAINS.items():
            for exp in EXPERIMENTS:
                run_one_experiment(
                    domain=domain,
                    base=base_path,
                    exp=exp,
                    ckpt=args.ckpt,
                    make_token_figs=True,
                    answer_only=answer_only,
                )

    # Print a concise summary of any failures
    if FAILED_PLOTS:
        print("[SUMMARY] Failed to produce plots for the following combinations:")
        for rec in FAILED_PLOTS:
            print(" -", rec)
    else:
        print("[SUMMARY] All plots generated successfully.")

if __name__ == "__main__":
    main()