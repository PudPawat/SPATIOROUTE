#!/usr/bin/env python3
"""
Fixed-prompt baseline (no routing): one YAML template for all questions.

Maps to the paper's uniform-prompt baseline (e.g. scene_understanding).

Example:
  python -m experiments.spatioroute.eval.baseline \\
    --split test \\
    --model-name Qwen/Qwen2-VL-2B-Instruct \\
    --prompt-name scene_understanding \\
    --output experiments/spatioroute/results/baseline_qwen2vl2b_test.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluate_sqa import main as evaluate_sqa_main  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SpatioRoute baseline: fixed YAML prompt")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--model-name", default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument(
        "--prompt-name",
        default="scene_understanding",
        help="Key under prompt_config.yaml prompts (fixed for all samples)",
    )
    parser.add_argument(
        "--prompt-config",
        default="experiments/spatioroute/configs/prompt_config.yaml",
    )
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    argv = [
        "evaluate_sqa.py",
        "--split",
        args.split,
        "--model-name",
        args.model_name,
        "--prompt-config",
        args.prompt_config,
        "--prompt-name",
        args.prompt_name,
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

    sys.argv = argv
    evaluate_sqa_main()


if __name__ == "__main__":
    main()
