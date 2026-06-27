#!/usr/bin/env python3
"""
Think-it-Twice / simple two-pass CoT baseline (paper comparison condition).

Example:
  python -m experiments.spatioroute.eval.cot \\
    --backend qwen2_2b \\
    --split test \\
    --output experiments/spatioroute/results/cot_qwen2vl2b_test.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluate_sqa_simple_cot import main as cot_main  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SpatioRoute CoT comparison: simple two-pass")
    parser.add_argument(
        "--backend",
        choices=("qwen2_2b", "qwen2_7b", "qwen25", "qwen3", "llama"),
        default="qwen2_2b",
    )
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-dir", default="dataset/SQA")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--question-types", nargs="+", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    argv = [
        "evaluate_sqa_simple_cot.py",
        "--backend",
        args.backend,
        "--split",
        args.split,
        "--output",
        args.output,
        "--dataset-dir",
        args.dataset_dir,
        "--max-frames",
        str(args.max_frames),
        "--save-interval",
        str(args.save_interval),
    ]
    if args.model_name:
        argv.extend(["--model-name", args.model_name])
    if args.max_samples:
        argv.extend(["--max-samples", str(args.max_samples)])
    if args.load_in_4bit:
        argv.append("--load-in-4bit")
    if args.load_in_8bit:
        argv.append("--load-in-8bit")
    if args.question_types:
        argv.extend(["--question-types", *args.question_types])

    sys.argv = argv
    cot_main()


if __name__ == "__main__":
    main()
