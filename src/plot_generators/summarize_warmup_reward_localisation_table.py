"""Build a pre-AIRL warmup reward localisation table."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.plot_generators import summarize_cross_domain_scored_localisation as score  # noqa: E402
from src.plot_generators import summarize_full_localisation_table as full_table  # noqa: E402


DEFAULT_OUTPUT_DIR = Path("localisation/full_three_domain_localisation")


WARMUP_PATHS: dict[str, dict[str, dict[str, Path]]] = {
    "gsm8k": {
        "qwen7b": {
            "dense": Path("localisation/natural_wrong_sft/scores/qwen7b_warmup_full_reward_localisation/pair_details.jsonl"),
            "interval": Path(
                "localisation/natural_wrong_sft/scores/"
                "qwen7b_warmup_partial_fixed_reward_localisation/pair_details.jsonl"
            ),
        },
        "llama8b": {
            "dense": Path("localisation/natural_wrong_sft/scores/llama8b_warmup_full_reward_localisation/pair_details.jsonl"),
            "interval": Path(
                "localisation/natural_wrong_sft/scores/"
                "llama8b_warmup_partial_fixed_reward_localisation/pair_details.jsonl"
            ),
        },
        "qwen4b": {
            "dense": Path("localisation/natural_wrong_sft/scores/qwen4b_warmup_full_reward_localisation/pair_details.jsonl"),
            "interval": Path(
                "localisation/natural_wrong_sft/scores/"
                "qwen4b_warmup_partial_fixed_reward_localisation/pair_details.jsonl"
            ),
        },
    },
    "medreason": {
        "qwen7b": {
            "dense": Path(
                "localisation/medreason_natural_wrong_sft/scores/"
                "qwen7b_medicine_sft_warmup_full_reward_localisation/pair_details.jsonl"
            ),
            "interval": Path(
                "localisation/medreason_natural_wrong_sft/scores/"
                "qwen7b_medicine_sft_warmup_partial_fixed_reward_localisation/pair_details.jsonl"
            ),
        },
        "llama8b": {
            "dense": Path(
                "localisation/medreason_natural_wrong_sft/scores/"
                "llama8b_medicine_sft_warmup_full_reward_localisation/pair_details.jsonl"
            ),
            "interval": Path(
                "localisation/medreason_natural_wrong_sft/scores/"
                "llama8b_medicine_sft_warmup_partial_fixed_reward_localisation/pair_details.jsonl"
            ),
        },
        "qwen4b": {
            "dense": Path(
                "localisation/medreason_natural_wrong_sft/scores/"
                "qwen4b_medicine_sft_warmup_full_reward_localisation/pair_details.jsonl"
            ),
            "interval": Path(
                "localisation/medreason_natural_wrong_sft/scores/"
                "qwen4b_medicine_sft_warmup_partial_fixed_reward_localisation/pair_details.jsonl"
            ),
        },
    },
    "mmlu_pro": {
        "qwen7b": {
            "dense": Path(
                "localisation/mmlu_pro_natural_wrong_sft/scores/"
                "qwen7b_mmlu_warmup_full_reward_localisation/pair_details.jsonl"
            ),
            "interval": Path(
                "localisation/mmlu_pro_natural_wrong_sft/scores/"
                "qwen7b_mmlu_warmup_partial_fixed_reward_localisation/pair_details.jsonl"
            ),
        },
        "llama8b": {
            "dense": Path(
                "localisation/mmlu_pro_natural_wrong_sft/scores/"
                "llama8b_mmlu_sft_warmup_full_reward_localisation/pair_details.jsonl"
            ),
            "interval": Path(
                "localisation/mmlu_pro_natural_wrong_sft/scores/"
                "llama8b_mmlu_sft_warmup_partial_fixed_reward_localisation/pair_details.jsonl"
            ),
        },
        "qwen4b": {
            "dense": Path(
                "localisation/mmlu_pro_natural_wrong_sft/scores/"
                "qwen4b_mmlu_warmup_full_reward_localisation/pair_details.jsonl"
            ),
            "interval": Path(
                "localisation/mmlu_pro_natural_wrong_sft/scores/"
                "qwen4b_mmlu_warmup_partial_fixed_reward_localisation/pair_details.jsonl"
            ),
        },
    },
}


METHODS = [
    ("dense", "Warmup dense reward", "token"),
    ("interval", "Warmup interval reward", "interval15"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--region", choices=list(score.REGIONS), default="think")
    parser.add_argument("--stride", type=int, default=score.INTERVAL_STRIDE)
    parser.add_argument("--max-length", type=int, default=score.MAX_LENGTH)
    parser.add_argument("--bootstrap-samples", type=int, default=score.BOOTSTRAP_SAMPLES)
    parser.add_argument("--bootstrap-alpha", type=float, default=score.BOOTSTRAP_ALPHA)
    parser.add_argument("--bootstrap-seed", type=int, default=score.BOOTSTRAP_SEED)
    return parser.parse_args()


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    results = []
    missing = []
    for dataset in full_table.DATASETS:
        label_path = Path(dataset["labels"])
        if not label_path.exists():
            missing.append({"dataset": dataset["label"], "model": "*", "method": "labels", "path": str(label_path)})
            continue
        label_keys, text_by_key = full_table._load_labels(label_path)
        dataset_paths = WARMUP_PATHS.get(str(dataset["key"]), {})
        for model_key, model_info in dataset["methods"].items():
            model_paths = dataset_paths.get(model_key, {})
            for method_name, method_label, grid in METHODS:
                path = model_paths.get(method_name)
                if path is None:
                    missing.append(
                        {
                            "dataset": dataset["label"],
                            "model": model_info["model"],
                            "method": method_label,
                            "path": "<not configured>",
                        }
                    )
                    continue
                result = full_table._score_one(
                    dataset=dataset,
                    model_key=model_key,
                    method_key=method_name,
                    method_label=method_label,
                    path=path,
                    seq_key="pert_score_seq",
                    detector="largest_drop",
                    is_policy=False,
                    append_eos=True,
                    grid=grid,
                    label_keys=label_keys,
                    text_by_key=text_by_key,
                    args=args,
                )
                if result is None:
                    missing.append(
                        {
                            "dataset": dataset["label"],
                            "model": model_info["model"],
                            "method": method_label,
                            "path": str(path),
                        }
                    )
                else:
                    results.append(result)

    compact: dict[tuple[str, str], dict[str, Any]] = {}
    for result in results:
        key = (result["dataset"], result["model"])
        row = compact.setdefault(
            key,
            {
                "Dataset": result["dataset"],
                "Error Source": result["error_source"],
                "Model": result["model"],
                "N": str(result["n_scored"]),
            },
        )
        method = result["method_key"]
        metrics = result["metrics"]
        row[f"{method} Hit@7"] = full_table._metric_cell(metrics["hit1_w7"], metrics["chance_hit1_w7"])
        row[f"{method} MAP"] = full_table._metric_cell(metrics["map_tie_aware"], metrics["chance_map"])

    dataset_order = {dataset["label"]: i for i, dataset in enumerate(full_table.DATASETS)}
    model_order = {"Qwen2.5-7B": 0, "Llama3.1-8B": 1, "Qwen3-4B": 2}
    table = []
    for _key, row in sorted(
        compact.items(),
        key=lambda item: (dataset_order.get(item[0][0], 99), model_order.get(item[0][1], 99)),
    ):
        for method in ("dense", "interval"):
            row.setdefault(f"{method} Hit@7", "--")
            row.setdefault(f"{method} MAP", "--")
        table.append(row)

    return {
        "region": args.region,
        "metric_notes": {
            "checkpoint": "Reward-model warmup checkpoint before adversarial/AIRL updates.",
            "hit": "Hit@1 within +/-7 units on each method's native grid.",
            "cell_format": "value (value - exact_random_chance), in percentage points.",
            "map": "Tie-aware MAP; expected AP is used within equal-score blocks.",
        },
        "results": results,
        "missing": missing,
        "table": table,
    }


def _markdown_table(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "_No rows._\n"
    headers = [
        "Dataset",
        "Model",
        "N",
        "dense Hit@7",
        "dense MAP",
        "interval Hit@7",
        "interval MAP",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "--")) for header in headers) + " |")
    return "\n".join(lines) + "\n"


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Warmup Reward Localisation Table",
        "",
        (
            "Region: `<think>` only. Scores use reward-model warmup checkpoints "
            "before adversarial/AIRL updates. Dense reward is on the token grid; "
            "interval reward is on its native 15-token grid. Cells are "
            "`value (delta over exact random chance)` in percentage points. "
            "MAP is tie-aware."
        ),
        "",
        _markdown_table(payload["table"]),
    ]
    if payload["missing"]:
        lines.extend(["", "## Missing Artifacts", ""])
        for item in payload["missing"]:
            lines.append(
                f"- {item['dataset']} / {item['model']} / {item['method']}: `{item['path']}`"
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    output_json = args.output_json or args.output_dir / "warmup_reward_localisation_table.json"
    output_md = args.output_md or args.output_dir / "warmup_reward_localisation_table.md"
    payload = build_payload(args)
    score._write_json(output_json, payload)
    _write_markdown(output_md, payload)
    print(f"Wrote {output_json}")
    print(f"Wrote {output_md}")
    print(_markdown_table(payload["table"]))
    if payload["missing"]:
        print("Missing artifacts:")
        for item in payload["missing"]:
            print(f"  - {item['dataset']} / {item['model']} / {item['method']}: {item['path']}")


if __name__ == "__main__":
    main()
