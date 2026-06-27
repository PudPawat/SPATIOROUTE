#!/usr/bin/env python3
"""
SpatioRoute-R: rule-based dynamic prompt routing (no LLM at routing time).

For each SQA3D sample:
  1. Classify question type (What / Is / How many / Can / Which / Others)
  2. Route to a YAML template (details_scene, step_by_step, …)
  3. Fill template with situation + question

Output JSON matches SpatioRoute-L so the same VLM evaluator can consume it.

Example:
  python -m experiments.spatioroute.preprompts.generate_rule \\
    --split test \\
    --output experiments/spatioroute/results/preprompts_r_test.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluate_sqa import load_dataset  # noqa: E402
from experiments.spatioroute._paths import DEFAULT_PROMPT_CONFIG  # noqa: E402
from experiments.spatioroute.preprompts.io import save_preprompt_bundle  # noqa: E402
from experiments.spatioroute.routing import (  # noqa: E402
    INSTRUCTION_STYLE_RULES,
    build_rule_preprompt,
    classify_question_type,
    compute_semantic_flags,
    load_prompt_templates,
    resolve_instruction_style,
    template_body_for_style,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SpatioRoute-R: rule-based preprompt generation")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--dataset-dir", default="dataset/SQA")
    parser.add_argument(
        "--prompt-config",
        default=str(DEFAULT_PROMPT_CONFIG),
        help="YAML templates (default: experiments/spatioroute/configs/prompt_config.yaml)",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--question-types", nargs="+", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    questions_path = dataset_dir / "sqa_task" / "balanced" / f"v1_balanced_questions_{args.split}_scannetv2.json"
    annotations_path = dataset_dir / "sqa_task" / "balanced" / f"v1_balanced_sqa_annotations_{args.split}_scannetv2.json"
    video_dir = dataset_dir / "video"
    for path in (questions_path, annotations_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    yaml_path = Path(args.prompt_config)
    templates = load_prompt_templates(yaml_path)
    samples = load_dataset(str(questions_path), str(annotations_path), str(video_dir))
    if args.question_types:
        from utils.question_type_filter import filter_samples_by_question_type

        samples = filter_samples_by_question_type(samples, args.question_types, match_any=True)
    if args.max_samples:
        samples = samples[: args.max_samples]

    results: List[Dict[str, Any]] = []
    for sample in tqdm(samples, desc="spatioroute-r"):
        question = sample.get("question") or ""
        situation = sample.get("situation") or ""
        qtype = classify_question_type(question)
        style = resolve_instruction_style(qtype)
        template = template_body_for_style(templates, style)
        prompt = build_rule_preprompt(template, situation=situation, question=question)
        results.append(
            {
                "question_id": int(sample["question_id"]),
                "scene_id": sample.get("scene_id"),
                "question_type": qtype,
                "instruction_style": style,
                "semantic_flags": compute_semantic_flags(question, qtype),
                "question": question,
                "situation": situation,
                "generated_prompt": prompt,
            }
        )

    meta = {
        "generator": "experiments.spatioroute.preprompts.generate_rule",
        "method": "SpatioRoute-R",
        "split": args.split,
        "instruction_style_rules": INSTRUCTION_STYLE_RULES,
        "prompt_config_ref": str(yaml_path.resolve()),
    }
    out_path = Path(args.output)
    save_preprompt_bundle(out_path, meta, results)
    print(f"Wrote {len(results)} preprompts → {out_path}")


if __name__ == "__main__":
    main()
