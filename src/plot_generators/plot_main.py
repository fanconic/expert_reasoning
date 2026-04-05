"""
plot_main.py

Run plotting for multiple experiments across two domains (math + medicine).
Parallelized version using ProcessPoolExecutor.

Usage:
    python plot_main.py
    python plot_main.py --ckpt best_model --workers 16
    python plot_main.py --no-token-figs
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

from pathlib import Path

# Try to import tqdm for progress bar, fallback if not available
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterator, **kwargs):
        return iterator

from plot_helpers import ensure_dir, read_and_enhance, run_all_plots

# -------------------------------
# Config
# -------------------------------

DOMAINS: Dict[str, Path] = {
    "math": Path("/mnt/pdata/caf83/icml_math/outputs"),
    "medicine": Path("/mnt/pdata/caf83/icml_medicine/outputs"),
    "mmlu": Path("/mnt/pdata/caf83/icml_mmlu/outputs"),
}

DEFAULT_CKPT = "best_model"

RunName = Union[str, Mapping[str, str]]

EXPERIMENTS: List[Dict[str, Any]] = [
    # Stepwise
    # {"airl": "qwen3b_partial_new", "sft": "qwen3b_sft", "grpo": "qwen3b_grpo", "label": "qwen3b_partial_new_from_sft"},
    # {"airl": "llama3b_partial_new", "sft": "llama3b_sft", "grpo": "llama3b_grpo", "label": "llama3b_partial_new_from_sft"},
    # {"airl": "qwen7b_partial_new", "sft": "qwen7b_sft", "grpo": "qwen7b_grpo", "label": "qwen7b_partial_new_from_sft"},
    # {"airl": "llama8b_partial_new", "sft": "llama8b_sft", "grpo": "llama8b_grpo", "label": "llama8b_partial_new_from_sft"},


    # # Sparse
    # {"airl": "qwen3b_sparse_new", "sft": "qwen3b_sft", "grpo": "qwen3b_grpo", "label": "qwen3b_sparse_new_from_sft"},
    # {"airl": "llama3b_sparse_new", "sft": "llama3b_sft", "grpo": "llama3b_grpo", "label": "llama3b_sparse_new_from_sft"},
    # {"airl": "qwen7b_sparse_new", "sft": "qwen7b_sft", "grpo": "qwen7b_grpo", "label": "qwen7b_sparse_new_from_sft"},
    # {"airl": "llama8b_sparse_new", "sft": "llama8b_sft", "grpo": "llama8b_grpo", "label": "llama8b_sparse_new_from_sft"},

    # # Full
    # {"airl": "qwen3b_full_new", "sft": "qwen3b_sft", "grpo": "qwen3b_grpo", "label": "qwen3b_full_new_from_sft"},
    # {"airl": "llama3b_full_new", "sft": "llama3b_sft", "grpo": "llama3b_grpo", "label": "llama3b_full_new_from_sft"},
    # {"airl": "qwen7b_full_new", "sft": "qwen7b_sft", "grpo": "qwen7b_grpo", "label": "qwen7b_full_new_from_sft"},
    # {"airl": "llama8b_full_new", "sft": "llama8b_sft", "grpo": "llama8b_grpo", "label": "llama8b_full_new_from_sft"},

    # # Partial fixed
    # {"airl": "qwen3b_partial_fixed_new", "sft": "qwen3b_sft", "grpo": "qwen3b_grpo", "label": "qwen3b_partial_fixed_new_from_sft"},
    # {"airl": "llama3b_partial_fixed_new", "sft": "llama3b_sft", "grpo": "llama3b_grpo", "label": "llama3b_partial_fixed_new_from_sft"},
    # {"airl": "qwen7b_partial_fixed_new", "sft": "qwen7b_sft", "grpo": "qwen7b_grpo", "label": "qwen7b_partial_fixed_new_from_sft"},
    # {"airl": "llama8b_partial_fixed_new", "sft": "llama8b_sft", "grpo": "llama8b_grpo", "label": "llama8b_partial_fixed_new_from_sft"},
    
    # Qwen4b
    # {"airl": "qwen4b_sparse", "sft": "qwen4b_sft", "grpo": "qwen4b_grpo", "label": "qwen4b_sparse", "domains": ["math"]},
    # {"airl": "qwen4b_partial", "sft": "qwen4b_sft", "grpo": "qwen4b_grpo", "label": "qwen4b_partial", "domains": ["math"]},
    # {"airl": "qwen4b_partial_fixed", "sft": "qwen4b_sft", "grpo": "qwen4b_grpo", "label": "qwen4b_partial_fixed", "domains": ["math"]},
    # {"airl": "qwen4b_full", "sft": "qwen4b_sft", "grpo": "qwen4b_grpo", "label": "qwen4b_full", "domains": ["math"]},
    
    {"airl": "qwen4b_sparse", "sft": "qwen4b_sft", "grpo": "qwen4b_grpo", "label": "qwen4b_sparse_new"},
    {"airl": "qwen4b_partial", "sft": "qwen4b_sft", "grpo": "qwen4b_grpo", "label": "qwen4b_partial_new"},
    {"airl": "qwen4b_partial_fixed", "sft": "qwen4b_sft", "grpo": "qwen4b_grpo", "label": "qwen4b_partial_fixed_new"},
    {"airl": "qwen4b_full", "sft": "qwen4b_sft", "grpo": "qwen4b_grpo", "label": "qwen4b_full_new"},
    
     # Betas
    # {"airl": "qwen3b_partial_range_-1.0_1.0", "sft": "qwen3b_sft", "grpo": "qwen3b_grpo", "label": "qwen3b_partial_range_11", "domains": ["math"]},
    # {"airl": "qwen3b_partial_range_-3.0_3.0", "sft": "qwen3b_sft", "grpo": "qwen3b_grpo", "label": "qwen3b_partial_range_33", "domains": ["math"]},
    # {"airl": "qwen3b_partial_range_-10.0_10.0", "sft": "qwen3b_sft", "grpo": "qwen3b_grpo", "label": "qwen3b_partial_range_1010", "domains": ["math"]},
    # {"airl": "qwen3b_partial_range_0.0_2.0", "sft": "qwen3b_sft", "grpo": "qwen3b_grpo", "label": "qwen3b_partial_range_02", "domains": ["math"]},
    
    # {"airl": "llama3b_partial_range_-1.0_1.0", "sft": "llama3b_sft", "grpo": "llama3b_grpo", "label": "llama3b_partial_range_11", "domains": ["math"]},
    # {"airl": "llama3b_partial_range_-3.0_3.0", "sft": "llama3b_sft", "grpo": "llama3b_grpo", "label": "llama3b_partial_range_33", "domains": ["math"]},
    # {"airl": "llama3b_partial_range_-10.0_10.0", "sft": "llama3b_sft", "grpo": "llama3b_grpo", "label": "llama3b_partial_range_1010", "domains": ["math"]},
    # {"airl": "llama3b_partial_range_0.0_2.0", "sft": "llama3b_sft", "grpo": "llama3b_grpo", "label": "llama3b_partial_range_02", "domains": ["math"]},
    # {"airl": "llama3b_partial_range_-1000_1000", "sft": "llama3b_sft", "grpo": "llama3b_grpo", "label": "llama3b_partial_range_unlimited", "domains": ["math"]},
    
    # # Groups
    # {"airl": "qwen3b_partial_G4", "sft": "qwen3b_sft", "grpo": "qwen3b_grpo", "label": "qwen3b_partial_G4", "domains": ["math"]},
    # {"airl": "qwen3b_partial_G16", "sft": "qwen3b_sft", "grpo": "qwen3b_grpo", "label": "qwen3b_partial_G16", "domains": ["math"]},
    # {"airl": "llama3b_partial_G4", "sft": "llama3b_sft", "grpo": "llama3b_grpo", "label": "llama3b_partial_G4", "domains": ["math"]},
    # {"airl": "llama3b_partial_G16", "sft": "llama3b_sft", "grpo": "llama3b_grpo", "label": "llama3b_partial_G16", "domains": ["math"]},
    
    # #  # Corruption
    # {"airl": "qwen3b_perturb_none", "sft": "qwen3b_sft", "grpo": "qwen3b_grpo", "label": "qwen3b_perturb_none", "domains": ["math"]},
    # {"airl": "qwen3b_perturb_numbers_only", "sft": "qwen3b_sft", "grpo": "qwen3b_grpo", "label": "qwen3b_perturb_numbers_only", "domains": ["math"]},
    # {"airl": "qwen3b_perturb_nearby_and_numbers", "sft": "qwen3b_sft", "grpo": "qwen3b_grpo", "label": "qwen3b_perturb_nearby_and_numbers", "domains": ["math"]},
    # {"airl": "qwen3b_perturb_no_switch", "sft": "qwen3b_sft", "grpo": "qwen3b_grpo", "label": "qwen3b_perturb_no_switch", "domains": ["math"]},
    
    # {"airl": "llama3b_perturb_none", "sft": "llama3b_sft", "grpo": "qwen3b_grpo", "label": "llama3b_perturb_none", "domains": ["math"]},
    # {"airl": "llama3b_perturb_numbers_only", "sft": "llama3b_sft", "grpo": "llama3b_grpo", "label": "llama3b_perturb_numbers_only", "domains": ["math"]},
    # {"airl": "llama3b_perturb_nearby_and_numbers", "sft": "llama3b_sft", "grpo": "llama3b_grpo", "label": "llama3b_perturb_nearby_and_numbers", "domains": ["math"]},
    # {"airl": "llama3b_perturb_no_switch", "sft": "llama3b_sft", "grpo": "llama3b_grpo", "label": "llama3b_perturb_no_switch", "domains": ["math"]},



    # OVR
    # {"airl": "qwen3b_ovr", "sft": "qwen3b_sft", "grpo": "qwen3b_grpo", "label": "qwen3b_ovr"},
    # {"airl": "llama3b_ovr", "sft": "llama3b_sft", "grpo": "llama3b_grpo", "label": "llama3b_ovr"},
    # {"airl": "qwen7b_ovr", "sft": "qwen7b_sft", "grpo": "qwen7b_grpo", "label": "qwen7b_ovr"},
    # {"airl": "llama8b_ovr", "sft": "llama8b_sft", "grpo": "llama8b_grpo", "label": "llama8b_ovr"},
    
    #  # switch reward models
    # {"airl": "llama3b_switch_reward", "sft": "llama3b_sft", "grpo": "llama3b_grpo", "label": "llama3b_switch_reward"},
    # {"airl": "llama8b_switch_reward", "sft": "llama8b_sft", "grpo": "llama8b_grpo", "label": "llama8b_switch_reward"},
    # {"airl": "qwen3b_switch_reward", "sft": "qwen3b_sft", "grpo": "qwen3b_grpo", "label": "qwen3b_switch_reward"},
    # {"airl": "qwen7b_switch_reward", "sft": "qwen7b_sft", "grpo": "qwen7b_grpo", "label": "qwen7b_switch_reward"},
    
    #  # humar error
    # {"airl": "qwen3b_human_error", "sft": "qwen3b_sft", "grpo": "qwen3b_grpo", "label": "qwen3b_human_error"},
    # {"airl": "llama3b_human_error", "sft": "llama3b_sft", "grpo": "llama3b_grpo", "label": "llama3b_human_error"},
    
    # # Discounted reward models
    # {"airl": "llama3b_discounted", "sft": "llama3b_sft", "grpo": "llama3b_grpo", "label": "llama3b_discounted", "domains": ["math"]},
    # {"airl": "llama8b_discounted", "sft": "llama8b_sft", "grpo": "llama8b_grpo", "label": "llama8b_discounted", "domains": ["math"]},
    # {"airl": "qwen3b_discounted", "sft": "qwen3b_sft", "grpo": "qwen3b_grpo", "label": "qwen3b_discounted", "domains": ["math"]},
    # {"airl": "qwen7b_discounted", "sft": "qwen7b_sft", "grpo": "qwen7b_grpo", "label": "qwen7b_discounted", "domains": ["math"]},
]


# -------------------------------
# Helpers
# -------------------------------

def resolve_run(name_or_map: RunName, domain: str) -> Optional[str]:
    """
    Resolve run name. Returns None if the domain is missing in the mapping.
    """
    if isinstance(name_or_map, str):
        return name_or_map
    # If it's a dict, get the domain key; return None if missing
    return name_or_map.get(domain)


def eval_jsonl(base: Path, run_name: str, ckpt: str) -> Path:
    # define the preferred "new" path
    
    # new_path = base / run_name / ckpt / "eval_results_new.jsonl"
    # # Check if it exists on disk
    # if new_path.exists():
    #     return new_path
    
    
    
    new_path = base / run_name / ckpt / "eval_results_new.jsonl" #"eval_results_logprobs_sft.jsonl"
    # Check if it exists on disk
    if new_path.exists():
        return new_path
    
    new_path = base / run_name / ckpt / "debug.jsonl"
    # Check if it exists on disk
    if new_path.exists():
        return new_path
        
    # Fallback to the original filename
    return base / run_name / ckpt / "eval_results.jsonl"


def exp_label(exp: Dict[str, Any]) -> str:
    if exp.get("label"):
        return str(exp["label"])
    return f"{exp['airl']}__{exp['sft']}__{exp['grpo']}"


def run_one_experiment(domain: str, base: Path, exp: Dict[str, Any], ckpt: str, make_token_figs: bool, answer_only: bool) -> Optional[str]:
    """
    Runs one experiment. 
    Returns: None (success/skip), or error string (failure).
    """
    try:
        # 1. Explicit Domain Filter Check
        # If the experiment dict has a "domains" list, check if current domain is in it.
        if "domains" in exp and domain not in exp["domains"]:
            return None # Skip silently (not relevant for this domain)

        # 2. Resolve Run Names
        airl_run = resolve_run(exp["airl"], domain)
        sft_run = resolve_run(exp["sft"], domain)
        grpo_run = resolve_run(exp["grpo"], domain)

        # 3. Implicit Missing Key Check
        # If any run name is None (because it was missing from the mapping), skip.
        if airl_run is None or sft_run is None or grpo_run is None:
            # We skip silently here. If you want to see what's skipped, uncomment the print:
            # print(f"[SKIP] Skipping {domain}/{exp_label(exp)} (configuration missing for this domain)")
            return None 

        airl_p = eval_jsonl(base, airl_run, ckpt)
        sft_p = eval_jsonl(base, sft_run, ckpt)
        grpo_p = eval_jsonl(base, grpo_run, ckpt)

        label = exp_label(exp)
        out_dir = Path("./figures") / ("answer_only" if answer_only else "full_cot") / domain / label
        ensure_dir(str(out_dir))

        # 4. File Existence Check
        missing = [p for p in (airl_p, sft_p, grpo_p) if not p.exists()]
        if missing:
            # If the files are genuinely missing on disk, we usually want to know.
            return f"{domain}/{label} (answer_only={answer_only}) - MISSING FILES: {[str(m) for m in missing]}"

        # Read Data
        df_airl = read_and_enhance(str(airl_p), answer_only=answer_only)
        df_sft = read_and_enhance(str(sft_p), answer_only=answer_only)
        df_grpo = read_and_enhance(str(grpo_p), answer_only=answer_only)

        # Plot
        run_all_plots(
            df_airl,
            df_sft,
            df_grpo,
            str(out_dir),
            num_generations=16,
            make_token_figs=make_token_figs,
        )
        
        return None # Success

    except Exception as e:
        label = exp_label(exp)
        return f"{domain}/{label} (answer_only={answer_only}) - CRASH: {str(e)}"


# -------------------------------
# Main
# -------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=DEFAULT_CKPT)
    parser.add_argument("--no-token-figs", action="store_true")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    # 1. Add Debug Argument
    parser.add_argument("--debug", action="store_true", help="Run sequentially for debugging with IPython")
    args = parser.parse_args()

    tasks = []
    
    for answer_only in [True]:
        for domain, base_path in DOMAINS.items():
            for exp in EXPERIMENTS:
                tasks.append({
                    "domain": domain,
                    "base": base_path,
                    "exp": exp,
                    "ckpt": args.ckpt,
                    "make_token_figs": not args.no_token_figs,
                    "answer_only": answer_only
                })

    print(f"[INFO] Starting {len(tasks)} tasks...")
    
    failed_plots = []

    # 2. Conditional Execution
    if args.debug:
        print("[INFO] Running in DEBUG mode (sequential execution).")
        # Run sequentially in the main process
        for t in tqdm(tasks, desc="Processing (Debug)"):
            result = run_one_experiment(
                t["domain"], t["base"], t["exp"], t["ckpt"], t["make_token_figs"], t["answer_only"]
            )
            if result is not None:
                failed_plots.append(result)
    else:
        print(f"[INFO] Running in PARALLEL mode using {args.workers} workers.")
        # Run in parallel
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    run_one_experiment, 
                    t["domain"], t["base"], t["exp"], t["ckpt"], t["make_token_figs"], t["answer_only"]
                ): t for t in tasks
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
                result = future.result()
                if result is not None:
                    failed_plots.append(result)

    print("\n" + "="*30)
    if failed_plots:
        print(f"[SUMMARY] {len(failed_plots)} tasks failed/missing:")
        for msg in failed_plots:
            print(f" - {msg}")
    else:
        print("[SUMMARY] All relevant plots generated successfully.")
    print("="*30)

if __name__ == "__main__":
    main()