"""Build one localisation table across GSM8K, MedReason, and MMLU-Pro."""

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


DEFAULT_OUTPUT_DIR = Path("localisation/full_three_domain_localisation")


DATASETS: list[dict[str, Any]] = [
    {
        "key": "gsm8k",
        "label": "GSM8K",
        "error_source": "Qwen2.5-7B-SFT natural errors",
        "labels": Path(
            "localisation/natural_wrong_sft/scores/_inputs/"
            "natural_wrong_sft_valid_target_char_span_actual_wrong_answer.jsonl"
        ),
        "methods": {
            "qwen7b": {
                "model": "Qwen2.5-7B",
                "dense": Path("localisation/natural_wrong_sft/scores/qwen7b_full_reward_localisation/pair_details.jsonl"),
                "interval": Path(
                    "localisation/natural_wrong_sft/scores/"
                    "qwen7b_partial_fixed_rebuttal_restart_reward_localisation/pair_details.jsonl"
                ),
                "policy": Path(
                    "localisation/natural_wrong_sft/scores/"
                    "qwen7b_sft_policy_token_baselines/policy_token_baselines.jsonl"
                ),
            },
            "llama8b": {
                "model": "Llama3.1-8B",
                "dense": Path("localisation/natural_wrong_sft/scores/llama8b_full_reward_localisation/pair_details.jsonl"),
                "interval": Path(
                    "localisation/natural_wrong_sft/scores/llama8b_partial_fixed_reward_localisation/pair_details.jsonl"
                ),
                "policy": Path(
                    "localisation/natural_wrong_sft/scores/"
                    "llama8b_sft_policy_token_baselines/policy_token_baselines.jsonl"
                ),
            },
            "qwen4b": {
                "model": "Qwen3-4B",
                "dense": Path("localisation/natural_wrong_sft/scores/qwen4b_full_reward_localisation/pair_details.jsonl"),
                "interval": Path(
                    "localisation/natural_wrong_sft/scores/qwen4b_partial_fixed_reward_localisation/pair_details.jsonl"
                ),
                "policy": Path(
                    "localisation/natural_wrong_sft/scores/"
                    "qwen4b_sft_policy_token_baselines/policy_token_baselines.jsonl"
                ),
            },
        },
    },
    {
        "key": "medreason",
        "label": "MedReason",
        "error_source": "Qwen2.5-7B-Med-SFT natural errors",
        "labels": Path(
            "localisation/medreason_natural_wrong_sft/"
            "medreason_qwen7b_sft_wrong_step_labels_invalid_step_only.jsonl"
        ),
        "methods": {
            "qwen7b": {
                "model": "Qwen2.5-7B",
                "dense": Path(
                    "localisation/medreason_natural_wrong_sft/scores/"
                    "qwen7b_medicine_sft_full_reward_localisation/pair_details.jsonl"
                ),
                "interval": Path(
                    "localisation/medreason_natural_wrong_sft/scores/"
                    "qwen7b_medicine_sft_partial_fixed_reward_localisation/pair_details.jsonl"
                ),
                "policy": Path(
                    "localisation/medreason_natural_wrong_sft/scores/"
                    "qwen7b_medicine_sft_policy_token_baselines/policy_token_baselines.jsonl"
                ),
            },
            "llama8b": {
                "model": "Llama3.1-8B",
                "dense": Path(
                    "localisation/medreason_natural_wrong_sft/scores/"
                    "llama8b_medicine_sft_full_reward_localisation/pair_details.jsonl"
                ),
                "interval": Path(
                    "localisation/medreason_natural_wrong_sft/scores/"
                    "llama8b_medicine_sft_partial_fixed_reward_localisation/pair_details.jsonl"
                ),
                "policy": Path(
                    "localisation/medreason_natural_wrong_sft/scores/"
                    "llama8b_medicine_sft_policy_token_baselines/policy_token_baselines.jsonl"
                ),
            },
            "qwen4b": {
                "model": "Qwen3-4B",
                "dense": Path(
                    "localisation/medreason_natural_wrong_sft/scores/"
                    "qwen4b_medicine_sft_full_reward_localisation/pair_details.jsonl"
                ),
                "interval": Path(
                    "localisation/medreason_natural_wrong_sft/scores/"
                    "qwen4b_medicine_sft_partial_fixed_reward_localisation/pair_details.jsonl"
                ),
                "policy": Path(
                    "localisation/medreason_natural_wrong_sft/scores/"
                    "qwen4b_medicine_sft_policy_token_baselines/policy_token_baselines.jsonl"
                ),
            },
        },
    },
    {
        "key": "mmlu_pro",
        "label": "MMLU-Pro",
        "error_source": "Llama3.1-8B-MMLU-SFT natural errors",
        "labels": Path(
            "localisation/mmlu_pro_natural_wrong_sft/"
            "mmlu_pro_llama8b_sft_wrong_step_labels_invalid_step_only.jsonl"
        ),
        "methods": {
            "qwen7b": {
                "model": "Qwen2.5-7B",
                "dense": Path(
                    "localisation/mmlu_pro_natural_wrong_sft/scores/"
                    "qwen7b_mmlu_full_reward_localisation/pair_details.jsonl"
                ),
                "interval": Path(
                    "localisation/mmlu_pro_natural_wrong_sft/scores/"
                    "qwen7b_mmlu_partial_fixed_reward_localisation/pair_details.jsonl"
                ),
                "policy": Path(
                    "localisation/mmlu_pro_natural_wrong_sft/scores/"
                    "qwen7b_mmlu_policy_token_baselines/policy_token_baselines.jsonl"
                ),
            },
            "llama8b": {
                "model": "Llama3.1-8B",
                "dense": Path(
                    "localisation/mmlu_pro_natural_wrong_sft/scores/"
                    "llama8b_mmlu_sft_full_reward_localisation/pair_details.jsonl"
                ),
                "interval": Path(
                    "localisation/mmlu_pro_natural_wrong_sft/scores/"
                    "llama8b_mmlu_sft_partial_fixed_reward_localisation/pair_details.jsonl"
                ),
                "policy": Path(
                    "localisation/mmlu_pro_natural_wrong_sft/scores/"
                    "llama8b_mmlu_sft_policy_token_baselines/policy_token_baselines.jsonl"
                ),
            },
            "qwen4b": {
                "model": "Qwen3-4B",
                "dense": Path(
                    "localisation/mmlu_pro_natural_wrong_sft/scores/"
                    "qwen4b_mmlu_full_reward_localisation/pair_details.jsonl"
                ),
                "interval": Path(
                    "localisation/mmlu_pro_natural_wrong_sft/scores/"
                    "qwen4b_mmlu_partial_fixed_reward_localisation/pair_details.jsonl"
                ),
                "policy": Path(
                    "localisation/mmlu_pro_natural_wrong_sft/scores/"
                    "qwen4b_mmlu_policy_token_baselines/policy_token_baselines.jsonl"
                ),
            },
        },
    },
]


METHODS = [
    ("dense", "Dense reward", "pert_score_seq", "largest_drop", False, True, "token"),
    ("interval", "Interval reward", "pert_score_seq", "largest_drop", False, True, "interval15"),
    ("logprob", "Log-prob", "pert_policy_log_probs", "largest_drop", True, False, "token"),
    ("entropy", "Entropy", "pert_policy_entropies", "largest_spike", True, False, "token"),
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


def _load_labels(path: Path) -> tuple[set[tuple[Any, ...]], dict[tuple[Any, ...], str]]:
    rows = score._load_jsonl(path)
    keys = {score._row_key(row) for row in rows}
    text_by_key = {
        score._row_key(row): row.get("pert_text") or row.get("wrong_text")
        for row in rows
        if isinstance(row.get("pert_text") or row.get("wrong_text"), str)
    }
    return keys, text_by_key


def _metric_cell(metric: dict[str, Any], chance: dict[str, Any] | None = None) -> str:
    if metric.get("mean") is None:
        return "--"
    mean = 100.0 * float(metric["mean"])
    ci = metric.get("ci_halfwidth")
    if chance is None or chance.get("mean") is None:
        return f"{mean:.2f}" + (f" +/- {100.0 * float(ci):.2f}" if ci is not None else "")
    delta = 100.0 * (float(metric["mean"]) - float(chance["mean"]))
    return f"{mean:.2f} ({delta:+.2f})"


def _score_one(
    *,
    dataset: dict[str, Any],
    model_key: str,
    method_key: str,
    method_label: str,
    path: Path,
    seq_key: str,
    detector: str,
    is_policy: bool,
    append_eos: bool,
    grid: str,
    label_keys: set[tuple[Any, ...]],
    text_by_key: dict[tuple[Any, ...], str],
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    spec = {
        "key": method_key,
        "method_key": method_key,
        "signal": method_label,
        "path": path,
        "seq_key": seq_key,
        "detector": detector,
        "is_policy": is_policy,
        "append_eos": append_eos,
        "native_grid": grid,
    }
    rows = [row for row in score._load_jsonl(path) if score._row_key(row) in label_keys]
    result = score._score_method(
        rows=rows,
        spec=spec,
        text_by_key=text_by_key,
        dataset={"key": dataset["key"], "label": dataset["label"], "model": dataset["methods"][model_key]["model"]},
        region=args.region,
        grid=grid,
        stride=int(args.stride),
        max_length=int(args.max_length),
        bootstrap_samples=int(args.bootstrap_samples),
        bootstrap_alpha=float(args.bootstrap_alpha),
        bootstrap_seed=int(args.bootstrap_seed),
    )
    result["error_source"] = dataset["error_source"]
    return result


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    results = []
    missing = []
    for dataset in DATASETS:
        label_path = Path(dataset["labels"])
        if not label_path.exists():
            missing.append({"dataset": dataset["label"], "model": "*", "method": "labels", "path": str(label_path)})
            continue
        label_keys, text_by_key = _load_labels(label_path)
        for model_key, model_info in dataset["methods"].items():
            for method_name, method_label, seq_key, detector, is_policy, append_eos, grid in METHODS:
                path_key = "policy" if method_name in {"logprob", "entropy"} else method_name
                path = Path(model_info[path_key])
                result = _score_one(
                    dataset=dataset,
                    model_key=model_key,
                    method_key=method_name,
                    method_label=method_label,
                    path=path,
                    seq_key=seq_key,
                    detector=detector,
                    is_policy=is_policy,
                    append_eos=append_eos,
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
        row[f"{method} Hit@7"] = _metric_cell(metrics["hit1_w7"], metrics["chance_hit1_w7"])
        row[f"{method} MAP"] = _metric_cell(metrics["map_tie_aware"], metrics["chance_map"])

    method_order = ["dense", "interval", "logprob", "entropy"]
    table = []
    dataset_order = {dataset["label"]: i for i, dataset in enumerate(DATASETS)}
    model_order = {"Qwen2.5-7B": 0, "Llama3.1-8B": 1, "Qwen3-4B": 2}
    for _key, row in sorted(
        compact.items(),
        key=lambda item: (dataset_order.get(item[0][0], 99), model_order.get(item[0][1], 99)),
    ):
        for method in method_order:
            row.setdefault(f"{method} Hit@7", "--")
            row.setdefault(f"{method} MAP", "--")
        table.append(row)

    return {
        "region": args.region,
        "metric_notes": {
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
        "logprob Hit@7",
        "logprob MAP",
        "entropy Hit@7",
        "entropy MAP",
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
        "# Full Three-Domain Localisation Table",
        "",
        (
            "Region: `<think>` only. Reward rows use largest reward drop; "
            "log-probability uses largest log-probability drop; entropy uses "
            "largest entropy spike. Dense/log-prob/entropy are on token grid; "
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
    output_json = args.output_json or args.output_dir / "full_localisation_table.json"
    output_md = args.output_md or args.output_dir / "full_localisation_table.md"
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
