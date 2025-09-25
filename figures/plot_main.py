"""
plot_main.py

Main entry point. Define your experiments (triplets of run folders), loop through
and generate figures/tables into an experiment-specific output folder.

Usage:
    python plot_main.py

Edit the `EXPERIMENTS` list below as needed. Each experiment is a dict with keys:
    - airl: run folder name for AIRL (expert reasoning)
    - sft:  run folder name for SFT
    - grpo: run folder name for GRPO / outcome-supervised
    - ckpt: checkpoint number as str or int (default '500')
    - label (optional): custom label used to name the output folder

Figures are saved under:
    {BASE}/figures/{label or airl__sft__grpo}/
where BASE defaults to "/mnt/pdata/caf83/tabular_reasoning/outputs".
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

from plot_helpers import (
    ensure_dir,
    read_and_enhance,
    run_all_plots,
)

# -------------------------------
# Config
# -------------------------------
BASE = Path("/mnt/pdata/caf83/tabular_reasoning/outputs")
DEFAULT_CKPT = "500"

# Define your experiments here
EXPERIMENTS: List[Dict] = [
    {
        "airl": "qwen3b_airl_09_bce2.5",
        "sft": "qwen3b_sft",
        "grpo": "qwen3b_grpo",
        "ckpt": DEFAULT_CKPT,
        "label": "qwen3b",
    },
    {
        "airl": "llama3_airl",
        "sft": "llama3_sft",
        "grpo": "llama3_grpo",
        "ckpt": DEFAULT_CKPT,
        "label": "llama3b",
    },
    {
        "airl": "qwen7b_airl_09_bce_15",
        "sft": "qwen7b_sft",
        "grpo": "qwen7b_grpo",
        "ckpt": DEFAULT_CKPT,
        "label": "qwen7b",
    },
    {
        "airl": "llama8_airl_6",
        "sft": "llama8_sft",
        "grpo": "llama8_grpo_2",
        "ckpt": DEFAULT_CKPT,
        "label": "llama8b",
    },
    # ### WGAN
    {
        "airl": "qwen3_airl_wgan",
        "sft": "qwen3b_sft",
        "grpo": "qwen3b_grpo",
        "ckpt": DEFAULT_CKPT,
        "label": "qwen3b_wgan",
    },
    {
        "airl": "qwen7_airl_wgan",
        "sft": "qwen7b_sft",
        "grpo": "qwen7b_grpo",
        "ckpt": DEFAULT_CKPT,
        "label": "qwen7b_wgan",
    },
    {
        "airl": "llama3_airl_wgan",
        "sft": "llama3_sft",
        "grpo": "llama3_grpo",
        "ckpt": DEFAULT_CKPT,
        "label": "llama3b_wgan",
    },
    {
        "airl": "llama8_airl_wgan",
        "sft": "llama8_sft",
        "grpo": "llama8_grpo_2",
        "ckpt": DEFAULT_CKPT,
        "label": "llama8b_wgan",
    },
    ### FROM SFT
    {
        "airl": "qwen3b_airl_from_sft",
        "sft": "qwen3b_sft",
        "grpo": "qwen3b_grpo",
        "ckpt": DEFAULT_CKPT,
        "label": "qwen3b_from_sft",
    },
    {
        "airl": "llama3_airl_sft",
        "sft": "llama3_sft",
        "grpo": "llama3_grpo",
        "ckpt": DEFAULT_CKPT,
        "label": "llama3b_from_sft",
    },
    {
        "airl": "qwen7b_airl_from_sft",
        "sft": "qwen7b_sft",
        "grpo": "qwen7b_grpo",
        "ckpt": DEFAULT_CKPT,
        "label": "qwen7b_from_sft",
    },
    {
        "airl": "llama8_airl_from_sft",
        "sft": "llama8_sft",
        "grpo": "llama8_grpo_2",
        "ckpt": DEFAULT_CKPT,
        "label": "llama8b_from_sft",
    },
    ## NO PERTURBATION
    {
        "airl": "llama8_airl_noper", # When running I accidendentally mislabelled it
        "sft": "qwen7b_sft",
        "grpo": "qwen7b_grpo",
        "ckpt": DEFAULT_CKPT,
        "label": "qwen7b_noper",
    },
    {
        "airl": "llama3_airl_noper", # When running I accidendentally mislabelled it
        "sft": "llama3_sft",
        "grpo": "llama3_grpo",
        "ckpt": DEFAULT_CKPT,
        "label": "llama3b_noper",
    },
    {
        "airl": "qwen3_airl_noper", # When running I accidendentally mislabelled it
        "sft": "qwen3b_sft",
        "grpo": "qwen3b_grpo",
        "ckpt": DEFAULT_CKPT,
        "label": "qwen3b_noper",
    },
    
]

# -------------------------------
# Helpers
# -------------------------------


def jsonl_path(run_name: str, ckpt: str | int) -> Path:
    return BASE / run_name / f"checkpoint-{ckpt}" / "eval_results.jsonl"


def label_for(exp: Dict) -> str:
    if "label" in exp and exp["label"]:
        return exp["label"]
    return f"{exp['airl']}__{exp['sft']}__{exp['grpo']}"


# -------------------------------
# Main
# -------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base",
        type=str,
        default=str(BASE),
        help="Base outputs directory containing the run folders",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Override checkpoint number for all experiments",
    )
    parser.add_argument(
        "--no-token-figs",
        action="store_true",
        help="Skip token-level dense reward figures",
    )
    args = parser.parse_args()

    base_dir = Path(args.base)

    for exp in EXPERIMENTS:
        ckpt = (
            args.ckpt if args.ckpt is not None else str(exp.get("ckpt", DEFAULT_CKPT))
        )
        airl_run, sft_run, grpo_run = exp["airl"], exp["sft"], exp["grpo"]

        airl_p = base_dir / airl_run / f"checkpoint-{ckpt}" / "eval_results.jsonl"
        sft_p = base_dir / sft_run / f"checkpoint-{ckpt}" / "eval_results.jsonl"
        grpo_p = base_dir / grpo_run / f"checkpoint-{ckpt}" / "eval_results.jsonl"

        label = label_for(exp)
        out_dir = f"./figures/{label}"
        ensure_dir(out_dir)

        missing = [p for p in [airl_p, sft_p, grpo_p] if not p.exists()]
        if missing:
            print(
                f"[WARNING] Skipping '{label}' — missing files: {[str(m) for m in missing]}"
            )
            continue

        print(f"[INFO] Running experiment '{label}' (ckpt={ckpt})")

        # Load and enhance
        try:
            df_airl = read_and_enhance(str(airl_p))
            df_sft = read_and_enhance(str(sft_p))
            df_grpo = read_and_enhance(str(grpo_p))
        except Exception as e:
            print(f"[ERROR] Failed to read/enhance data for '{label}': {e}")
            continue

        # Orchestrate plots
        try:
            run_all_plots(
                df_airl,
                df_sft,
                df_grpo,
                out_dir,
                num_generations=16,
                make_token_figs=not args.no_token_figs,
            )
        except Exception as e:
            print(f"[ERROR] Plotting failed for '{label}': {e}")
            continue

        print(f"[DONE] Saved figures to: {out_dir}")


if __name__ == "__main__":
    main()
