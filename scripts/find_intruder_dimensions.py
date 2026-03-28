#!/usr/bin/env python
"""CLI entry point for intruder dimension analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm.auto import tqdm

from lora_finetune.intruder_dimensions import (
    DEFAULT_LLAMA_MODULE_REGEXES,
    AnalysisConfig,
    analyze_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find intruder dimensions in a fine-tuned checkpoint."
    )
    parser.add_argument("--base-model", required=True, help="Path to the base model directory.")
    parser.add_argument("--tuned", required=True, help="Path to a run or checkpoint directory.")
    parser.add_argument("--epsilon", type=float, default=0.5, help="Cosine similarity threshold.")
    parser.add_argument("--k", type=int, default=10, help="Top-k tuned singular vectors to inspect.")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device to use for SVD computation.",
    )
    parser.add_argument(
        "--module-regex",
        action="append",
        default=None,
        help="Regex selecting weight names to analyze. Can be provided multiple times.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of matrices analyzed.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    module_regexes = args.module_regex or list(DEFAULT_LLAMA_MODULE_REGEXES)
    config = AnalysisConfig(
        base_model_path=Path(args.base_model),
        tuned_path=Path(args.tuned),
        epsilon=args.epsilon,
        k=args.k,
        module_regexes=module_regexes,
        device=args.device,
        limit=args.limit,
    )
    progress_bar: tqdm | None = None

    def on_progress(event: dict) -> None:
        nonlocal progress_bar
        phase = event["phase"]
        total_matrices = event.get("total_matrices") or 0
        weight_name = event.get("weight_name")
        matrix_index = event.get("matrix_index")

        if phase == "start":
            progress_bar = tqdm(
                total=total_matrices,
                desc="Intruder analysis",
                unit="matrix",
                dynamic_ncols=True,
            )
            return

        if progress_bar is None:
            return

        if phase == "complete":
            total_intruders = event.get("total_intruders")
            progress_bar.set_postfix_str(f"complete total_intruders={total_intruders}")
            progress_bar.close()
            progress_bar = None
            return

        label = phase
        if matrix_index is not None and total_matrices:
            label = f"{phase} {matrix_index}/{total_matrices}"
        if weight_name:
            short_name = weight_name.replace("model.layers.", "L").replace(".weight", "")
            progress_bar.set_postfix_str(f"{label} {short_name}")
        else:
            progress_bar.set_postfix_str(label)

        if phase == "matrix_complete":
            progress_bar.update(1)

    report = analyze_model(config, progress_callback=on_progress)
    report_dict = report.to_dict()

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report_dict, indent=2, sort_keys=True))

    print(f"Resolved checkpoint: {report.resolved_checkpoint_path}")
    print(f"Tuned type: {report.tuned_type}")
    print(f"Matrices analyzed: {report.num_matrices}")
    print(f"Total intruders: {report.total_intruders}")
    print("Top matrices by intruder count:")
    top_results = sorted(
        report.results,
        key=lambda result: (-result.intruder_count, result.weight_name),
    )[:10]
    for result in top_results:
        print(
            f"  {result.intruder_count:>3}  {result.weight_name}  "
            f"shape={result.shape}  k={result.examined_k}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
