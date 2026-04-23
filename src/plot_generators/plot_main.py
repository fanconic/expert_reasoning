"""Run plotting jobs on the new evaluation generations (temperature 0.5 sweep)."""

from __future__ import annotations

import argparse
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

try:
    from src.plot_generators.plot_helpers import (
        count_xml,
        read_and_enhance,
        run_all_plots,
        strict_format_reward_func,
    )
except ModuleNotFoundError:
    from plot_helpers import count_xml, read_and_enhance, run_all_plots, strict_format_reward_func


DEFAULT_SPEC_PATH = Path(__file__).resolve().parent / "configs" / "main.yaml"
DEFAULT_MODELS = ("qwen3b", "qwen4b", "llama3b", "qwen7b", "llama8b")
DEFAULT_DOMAINS = ("math", "medicine", "mmlu")
DEFAULT_VARIANTS = ("sparse", "partial", "partial_fixed", "full")
DEFAULT_AIRL_RUN_TEMPLATE = "{model}_{variant}"
DEFAULT_SFT_RUN_TEMPLATE = "{model}_sft"
DEFAULT_GRPO_RUN_TEMPLATE = "{model}_grpo"
DEFAULT_AIRL_FILE_TEMPLATE = "eval_results_{domain}_{model}_{variant}_t0p5.jsonl"
DEFAULT_SFT_FILE_TEMPLATE = "eval_results_{domain}_{model}_sft_t0p5.jsonl"
DEFAULT_GRPO_FILE_TEMPLATE = "eval_results_{domain}_{model}_grpo_t0p5.jsonl"


@dataclass(frozen=True)
class PlotTask:
    domain: str
    model: str
    variant: str
    answer_only: bool
    airl_jsonl: Path
    sft_jsonl: Path
    grpo_jsonl: Path
    output_dir: Path

    @property
    def label(self) -> str:
        return f"{self.domain}/{self.model}/{self.variant}"


def _parse_csv_arg(raw: str | None, default: tuple[str, ...]) -> list[str]:
    if raw is None:
        return list(default)
    values = [x.strip() for x in raw.split(",")]
    return [x for x in values if x]


def _parse_int_csv_arg(raw: str | None, default: list[int]) -> list[int]:
    if raw is None:
        return list(default)
    values: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_SPEC_PATH),
        help="Path to plotting YAML spec (used for domain roots/output root).",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Checkpoint folder name override (default from spec, usually best_model).",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Override output root (default from spec).",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated models (default: qwen3b,qwen4b,llama3b,qwen7b,llama8b).",
    )
    parser.add_argument(
        "--domains",
        type=str,
        default=None,
        help="Comma-separated domains (default: math,medicine,mmlu).",
    )
    parser.add_argument(
        "--variants",
        type=str,
        default=None,
        help="Comma-separated AIRL variants (default: sparse,partial,partial_fixed,full).",
    )
    parser.add_argument(
        "--guided-method",
        type=str,
        default="chunk",
        help=(
            "Method placeholder value used in templates (e.g. chunk or topk). "
            "Only used if templates include {method}."
        ),
    )
    parser.add_argument(
        "--airl-run-template",
        type=str,
        default=DEFAULT_AIRL_RUN_TEMPLATE,
        help=(
            "AIRL run-name template relative to each domain root. "
            "Available fields: {domain},{model},{variant},{method}."
        ),
    )
    parser.add_argument(
        "--sft-run-template",
        type=str,
        default=DEFAULT_SFT_RUN_TEMPLATE,
        help=(
            "SFT run-name template relative to each domain root. "
            "Available fields: {domain},{model},{variant},{method}."
        ),
    )
    parser.add_argument(
        "--grpo-run-template",
        type=str,
        default=DEFAULT_GRPO_RUN_TEMPLATE,
        help=(
            "GRPO run-name template relative to each domain root. "
            "Available fields: {domain},{model},{variant},{method}."
        ),
    )
    parser.add_argument(
        "--airl-file-template",
        type=str,
        default=DEFAULT_AIRL_FILE_TEMPLATE,
        help=(
            "AIRL jsonl filename template. "
            "Available fields: {domain},{model},{variant},{method}."
        ),
    )
    parser.add_argument(
        "--sft-file-template",
        type=str,
        default=DEFAULT_SFT_FILE_TEMPLATE,
        help=(
            "SFT jsonl filename template. "
            "Available fields: {domain},{model},{variant},{method}."
        ),
    )
    parser.add_argument(
        "--grpo-file-template",
        type=str,
        default=DEFAULT_GRPO_FILE_TEMPLATE,
        help=(
            "GRPO jsonl filename template. "
            "Available fields: {domain},{model},{variant},{method}."
        ),
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=16,
        help="Number of generations per prompt in each jsonl.",
    )
    parser.add_argument(
        "--reranking-generations",
        type=str,
        default=None,
        help=(
            "Comma-separated generation counts for reranking outputs "
            "(e.g. 2,3,5,8,16). Defaults to --num-generations."
        ),
    )
    parser.add_argument(
        "--selector-mode",
        type=str,
        default=os.environ.get("PLOT_SELECTOR_MODE", "auto"),
        help=(
            "Dense-reward selector mode passed to read_and_enhance "
            "(e.g. auto, discounted, mean, last, tail3, top3, softmax2, power2, trimmed10, answer_boost)."
        ),
    )
    parser.add_argument("--no-token-figs", action="store_true")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument(
        "--debug", action="store_true", help="Run sequentially for easier debugging"
    )
    return parser.parse_args()


def load_plot_spec(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid plotting spec (expected mapping): {path}")
    return data


def _resolve_jsonl(
    *,
    domain_root: Path,
    run_name: str,
    ckpt: str,
    file_template: str,
    domain: str,
    model: str,
    variant: str,
    method: str,
) -> Path:
    format_values = {
        "domain": domain,
        "model": model,
        "variant": variant,
        "method": method,
    }
    try:
        filename = file_template.format(**format_values)
    except KeyError as exc:
        raise ValueError(
            f"Invalid filename template '{file_template}'. "
            f"Unknown key: {exc}. Allowed keys: {sorted(format_values.keys())}"
        ) from exc
    return domain_root / run_name / ckpt / filename


def build_tasks(
    spec: dict[str, Any],
    *,
    ckpt: str | None,
    output_root: str | None,
    models: list[str],
    domains: list[str],
    variants: list[str],
    guided_method: str,
    airl_run_template: str,
    sft_run_template: str,
    grpo_run_template: str,
    airl_file_template: str,
    sft_file_template: str,
    grpo_file_template: str,
) -> list[PlotTask]:
    spec_domains = spec.get("domains", {})
    if not isinstance(spec_domains, dict):
        raise ValueError("Spec key 'domains' must be a mapping.")

    resolved_ckpt = ckpt or str(spec.get("default_ckpt", "best_model"))
    resolved_output_root = Path(output_root or spec.get("output_root", "./figures"))

    answer_only_modes = spec.get("answer_only_modes", [True])
    if not isinstance(answer_only_modes, list) or not answer_only_modes:
        answer_only_modes = [True]

    tasks: list[PlotTask] = []
    missing_domains = [d for d in domains if d not in spec_domains]
    if missing_domains:
        raise ValueError(
            "Missing domain roots in spec for: " + ", ".join(sorted(missing_domains))
        )

    for domain in domains:
        domain_root = Path(spec_domains[domain])
        for model in models:
            fmt_values = {
                "domain": domain,
                "model": model,
                "variant": "",
                "method": guided_method,
            }
            try:
                sft_fmt = {**fmt_values, "variant": "sft"}
                grpo_fmt = {**fmt_values, "variant": "grpo"}
                sft_run_name = sft_run_template.format(**sft_fmt)
                grpo_run_name = grpo_run_template.format(**grpo_fmt)
            except KeyError as exc:
                raise ValueError(
                    "Invalid run-name template. "
                    f"Unknown key: {exc}. Allowed keys: {sorted(fmt_values.keys())}"
                ) from exc

            sft_jsonl = _resolve_jsonl(
                domain_root=domain_root,
                run_name=sft_run_name,
                ckpt=resolved_ckpt,
                file_template=sft_file_template,
                domain=domain,
                model=model,
                variant="sft",
                method=guided_method,
            )
            grpo_jsonl = _resolve_jsonl(
                domain_root=domain_root,
                run_name=grpo_run_name,
                ckpt=resolved_ckpt,
                file_template=grpo_file_template,
                domain=domain,
                model=model,
                variant="grpo",
                method=guided_method,
            )

            for variant in variants:
                fmt_values["variant"] = variant
                try:
                    airl_run_name = airl_run_template.format(**fmt_values)
                except KeyError as exc:
                    raise ValueError(
                        "Invalid AIRL run-name template. "
                        f"Unknown key: {exc}. Allowed keys: {sorted(fmt_values.keys())}"
                    ) from exc

                airl_jsonl = _resolve_jsonl(
                    domain_root=domain_root,
                    run_name=airl_run_name,
                    ckpt=resolved_ckpt,
                    file_template=airl_file_template,
                    domain=domain,
                    model=model,
                    variant=variant,
                    method=guided_method,
                )
                run_label = f"{model}_{variant}"

                for answer_only in answer_only_modes:
                    mode_dir = "answer_only" if answer_only else "all_tokens"
                    out_dir = (
                        resolved_output_root
                        / mode_dir
                        / domain
                        / "standard"
                        / run_label
                    )

                    tasks.append(
                        PlotTask(
                            domain=domain,
                            model=model,
                            variant=variant,
                            answer_only=bool(answer_only),
                            airl_jsonl=airl_jsonl,
                            sft_jsonl=sft_jsonl,
                            grpo_jsonl=grpo_jsonl,
                            output_dir=out_dir,
                        )
                    )

    return tasks


def _compute_selector_from_lists(series: pd.Series) -> pd.Series:
    def _mean_or_nan(values: Any) -> float:
        if not isinstance(values, (list, np.ndarray)):
            return float("nan")
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        return float(np.nanmean(arr)) if arr.size else float("nan")

    return series.apply(_mean_or_nan)


def _ensure_selector(df: pd.DataFrame) -> pd.DataFrame:
    if "selector" in df.columns:
        return df

    df = df.copy()
    if "mean_rewards" in df.columns:
        df["selector"] = pd.to_numeric(df["mean_rewards"], errors="coerce")
        return df

    if "reward_model_score" in df.columns:
        selector = _compute_selector_from_lists(df["reward_model_score"])
        df["selector"] = pd.to_numeric(selector, errors="coerce")
        return df

    if "mean_log_probs" in df.columns:
        df["selector"] = pd.to_numeric(df["mean_log_probs"], errors="coerce")
        return df

    if "policy_log_probs" in df.columns:
        selector = _compute_selector_from_lists(df["policy_log_probs"])
        df["selector"] = pd.to_numeric(selector, errors="coerce")
        return df

    if "correctness_reward_func" in df.columns:
        correctness = pd.to_numeric(df["correctness_reward_func"], errors="coerce")
        df["selector"] = correctness.fillna(0.0) / 2.0
        return df

    df["selector"] = 0.0
    return df


def _safe_read_and_enhance(
    jsonl_path: Path, answer_only: bool, selector_mode: str
) -> pd.DataFrame:
    try:
        return read_and_enhance(
            str(jsonl_path), answer_only=answer_only, selector_mode=selector_mode
        )
    except Exception as exc:
        print(
            f"[WARNING] read_and_enhance failed for {jsonl_path}. "
            f"Falling back to lightweight parser. ({type(exc).__name__}: {exc})"
        )

        df = pd.read_json(jsonl_path, lines=True)

        if "generation" in df.columns:
            content = df["generation"].apply(
                lambda x: x.get("content", "") if isinstance(x, dict) else str(x)
            )
            df["strict_format_reward_func"] = content.apply(strict_format_reward_func)
            df["xmlcount_reward_func"] = content.apply(count_xml)

        if "reward_model_score" in df.columns:
            selector = _compute_selector_from_lists(df["reward_model_score"])
            df["mean_rewards"] = selector

        return df


def _run_task(
    task: PlotTask,
    *,
    make_token_figs: bool,
    num_generations: int,
    reranking_generations: list[int],
    selector_mode: str,
) -> str | None:
    required = [task.airl_jsonl, task.sft_jsonl, task.grpo_jsonl]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        return f"[MISSING] {task.label} -> " + ", ".join(missing)

    try:
        df_airl = _safe_read_and_enhance(
            task.airl_jsonl, answer_only=task.answer_only, selector_mode=selector_mode
        )
        df_sft = _safe_read_and_enhance(
            task.sft_jsonl, answer_only=task.answer_only, selector_mode=selector_mode
        )
        df_grpo = _safe_read_and_enhance(
            task.grpo_jsonl, answer_only=task.answer_only, selector_mode=selector_mode
        )

        df_airl = _ensure_selector(df_airl)
        df_sft = _ensure_selector(df_sft)
        df_grpo = _ensure_selector(df_grpo)

        run_all_plots(
            df_airl=df_airl,
            df_sft=df_sft,
            df_grpo=df_grpo,
            out_dir=task.output_dir,
            num_generations=num_generations,
            reranking_generations=reranking_generations,
            make_token_figs=make_token_figs,
        )
        return None
    except Exception as exc:
        tb = traceback.format_exc(limit=8)
        return f"[FAILED] {task.label} ({type(exc).__name__}: {exc})\n{tb}"


def execute_tasks(
    tasks: list[PlotTask],
    *,
    workers: int,
    debug: bool,
    make_token_figs: bool,
    num_generations: int,
    reranking_generations: list[int],
    selector_mode: str,
) -> list[str]:
    failed: list[str] = []
    if debug or workers <= 1:
        for idx, task in enumerate(tasks, start=1):
            print(f"[{idx}/{len(tasks)}] {task.label}")
            msg = _run_task(
                task,
                make_token_figs=make_token_figs,
                num_generations=num_generations,
                reranking_generations=reranking_generations,
                selector_mode=selector_mode,
            )
            if msg is not None:
                failed.append(msg)
        return failed

    max_workers = max(1, workers)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _run_task,
                task,
                make_token_figs=make_token_figs,
                num_generations=num_generations,
                reranking_generations=reranking_generations,
                selector_mode=selector_mode,
            ): task
            for task in tasks
        }
        completed = 0
        for future in as_completed(futures):
            completed += 1
            task = futures[future]
            print(f"[{completed}/{len(tasks)}] {task.label}")
            msg = future.result()
            if msg is not None:
                failed.append(msg)

    return failed


def main() -> None:
    args = parse_args()
    spec = load_plot_spec(args.config)

    models = _parse_csv_arg(args.models, DEFAULT_MODELS)
    domains = _parse_csv_arg(args.domains, DEFAULT_DOMAINS)
    variants = _parse_csv_arg(args.variants, DEFAULT_VARIANTS)
    reranking_generations = _parse_int_csv_arg(
        args.reranking_generations, [args.num_generations]
    )
    reranking_generations = [g for g in reranking_generations if g > 0]
    if not reranking_generations:
        raise ValueError("At least one positive value is required in --reranking-generations")

    tasks = build_tasks(
        spec,
        ckpt=args.ckpt,
        output_root=args.output_root,
        models=models,
        domains=domains,
        variants=variants,
        guided_method=args.guided_method,
        airl_run_template=args.airl_run_template,
        sft_run_template=args.sft_run_template,
        grpo_run_template=args.grpo_run_template,
        airl_file_template=args.airl_file_template,
        sft_file_template=args.sft_file_template,
        grpo_file_template=args.grpo_file_template,
    )

    print(f"[INFO] Loaded spec: {args.config}")
    print(f"[INFO] Domains: {', '.join(domains)}")
    print(f"[INFO] Models: {', '.join(models)}")
    print(f"[INFO] Variants: {', '.join(variants)}")
    print(f"[INFO] Reranking Generations: {', '.join(map(str, reranking_generations))}")
    print(f"[INFO] Selector Mode: {args.selector_mode}")
    print(f"[INFO] Starting {len(tasks)} tasks...")
    print(
        f"[INFO] Mode: {'DEBUG (sequential)' if args.debug else f'PARALLEL ({args.workers} workers)'}"
    )

    failed = execute_tasks(
        tasks,
        workers=args.workers or 1,
        debug=args.debug,
        make_token_figs=not args.no_token_figs,
        num_generations=args.num_generations,
        reranking_generations=reranking_generations,
        selector_mode=args.selector_mode,
    )

    print("\n" + "=" * 30)
    if failed:
        print(f"[SUMMARY] {len(failed)} tasks failed/missing:")
        for msg in failed:
            print(f" - {msg}")
    else:
        print("[SUMMARY] All plots generated successfully.")
    print("=" * 30)


if __name__ == "__main__":
    main()
