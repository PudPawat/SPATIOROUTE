#!/usr/bin/env python3
"""
VLM evaluation with routed preprompts (SpatioRoute-R or SpatioRoute-L output).

Example:
  python -m experiments.spatioroute.eval.routed \\
    --preprompt-json experiments/spatioroute/results/preprompts_r_test.json \\
    --split test \\
    --model-name Qwen/Qwen2-VL-2B-Instruct \\
    --output experiments/spatioroute/results/spatioroute_r_qwen2vl2b_test.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluate_sqa_preprompts import main as preprompt_eval_main  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SpatioRoute routed preprompt VLM evaluation")
    parser.add_argument("--preprompt-json", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--model-name", default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-dir", default="dataset/SQA")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--no-quantization", action="store_true")
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--question-types", nargs="+", default=None)
    parser.add_argument("--filter-scenes-json", default=None)
    parser.add_argument("--force-situation-merge", action="store_true")
    parser.add_argument("--omit-dataset-situation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    argv = [
        "evaluate_sqa_preprompts.py",
        "--preprompt-json",
        args.preprompt_json,
        "--split",
        args.split,
        "--model-name",
        args.model_name,
        "--output",
        args.output,
        "--dataset-dir",
        args.dataset_dir,
        "--max-frames",
        str(args.max_frames),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--save-interval",
        str(args.save_interval),
    ]
    if args.max_samples:
        argv.extend(["--max-samples", str(args.max_samples)])
    if args.load_in_4bit:
        argv.append("--load-in-4bit")
    if args.load_in_8bit:
        argv.append("--load-in-8bit")
    if args.no_quantization:
        argv.append("--no-quantization")
    if args.question_types:
        argv.extend(["--question-types", *args.question_types])
    if args.filter_scenes_json:
        argv.extend(["--filter-scenes-json", args.filter_scenes_json])
    if args.force_situation_merge:
        argv.append("--force-situation-merge")
    if args.omit_dataset_situation:
        argv.append("--omit-dataset-situation")

    sys.argv = argv
    preprompt_eval_main()


if __name__ == "__main__":
    main()
