#!/usr/bin/env python3
"""
SQA evaluation like ``evaluate_sqa.py``, but the text sent to the VLM per sample is
``generated_prompt`` from a preprompt JSON (e.g. ``generated_vlm_preprompts_test_1.json``).

The preprompt file shape matches ``generate_vlm_preprompts.py`` output:
``{ "meta": {...}, "results": [ { "question_id", "generated_prompt", ... } ], "by_question_id": {...} }``.

The VLM receives ``generated_prompt`` as the ``question`` text (``prompt_template=None``). Merged
samples set ``vlm_prompt`` to that same string (after optional situation prepend), and
``evaluate_sqa.evaluate`` writes it again on each result row so the output JSON always records the
exact user text sent to the VLM. If the
preprompt does not already embed the dataset **situation**, the script **prepends** a
``Situation:`` block (verbatim from SQA) so the VLM always has observer grounding. Use
``--force-situation-merge`` to prepend even when a substring match thinks it is already present.
Use ``--omit-dataset-situation`` to skip that prepend entirely (merge treats situation as empty:
``vlm_situation_merge.situation_empty`` is true); the sample row still keeps ``situation`` from the
dataset for logging.

``merge_vlm_preprompt_json.py`` only merges preprompt JSON files by ``question_id``; it does not
change how situations are prepended (that is all in this script).

Example:
  python evaluate_sqa_preprompts.py \\
    --preprompt-json generated_vlm_preprompts_test_1.json \\
    --split test \\
    --model-name Qwen/Qwen2-VL-2B-Instruct \\
    --output sqa_test_results_with_preprompts.json \\
    --save-interval 50
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from evaluate_sqa import evaluate, load_dataset
from utils.evaluation_result_meta import build_evaluation_settings
from utils.evaluation_result_meta import normalize_question_id
from videolm import VideoLM


def load_preprompt_map(path: str) -> Tuple[Dict[int, str], Dict[str, Any]]:
    """
    Build question_id -> generated_prompt from preprompt JSON.
    Skips empty prompts and rows whose prompt starts with ERROR:.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta = data.get("meta") or {}
    out: Dict[int, str] = {}

    for row in data.get("results") or []:
        qid = row.get("question_id")
        if qid is None:
            continue
        qid_n = int(normalize_question_id(qid))
        gp = (row.get("generated_prompt") or "").strip()
        if not gp or gp.upper().startswith("ERROR:"):
            continue
        out[qid_n] = gp

    for key, val in (data.get("by_question_id") or {}).items():
        qid_n = int(normalize_question_id(key))
        if qid_n in out:
            continue
        if isinstance(val, dict):
            gp = (val.get("generated_prompt") or "").strip()
        else:
            gp = str(val).strip()
        if gp and not gp.upper().startswith("ERROR:"):
            out[qid_n] = gp

    return out, meta


def _situation_embedded_in_prompt(preprompt: str, situation: str) -> bool:
    """Heuristic: preprompt already carries the observer situation text (or a Situation: block)."""
    sit = (situation or "").strip()
    if not sit:
        return True
    p = preprompt.replace("\r\n", "\n")
    if sit in p:
        return True
    sit_fold = " ".join(sit.split())
    p_fold = " ".join(p.split())
    if len(sit_fold) >= 16 and sit_fold in p_fold:
        return True
    if re.search(r"(?mi)^Situation:\s*$", p):
        return False
    if re.search(r"(?mi)^Situation:\s*\S", p) and sit_fold[:40] in p_fold:
        return True
    return False


def merge_situation_with_preprompt(
    preprompt: str,
    situation: str,
    *,
    force: bool,
) -> Tuple[str, Dict[str, Any]]:
    """
    Return the string passed to the VLM as ``question``.

    If ``situation`` is non-empty and does not appear to be embedded in ``preprompt``,
    prepend a clear ``Situation:`` block so the model always sees observer context.
    """
    sit = (situation or "").strip()
    info: Dict[str, Any] = {
        "situation_empty": not bool(sit),
        "prepended_situation_block": False,
        "force": force,
    }
    p = (preprompt or "").strip()
    if not sit:
        return p, info
    if not force and _situation_embedded_in_prompt(p, sit):
        return p, info

    block = (
        "Use this observer situation together with the instructions that follow.\n\n"
        f"Situation:\n{sit}\n\n"
        "---\n\n"
    )
    info["prepended_situation_block"] = True
    return (block + p).strip(), info


def merge_preprompts_into_samples(
    samples: List[Dict],
    preprompt_map: Dict[int, str],
    *,
    force_situation_merge: bool,
    omit_dataset_situation: bool = False,
) -> Tuple[List[Dict], int]:
    """Attach ``original_question``, ``vlm_prompt``, and set ``question`` to preprompt (+ situation merge for VLM)."""
    merged: List[Dict] = []
    skipped = 0
    for s in samples:
        try:
            qid_int = int(normalize_question_id(s["question_id"]))
        except (TypeError, ValueError):
            skipped += 1
            continue
        if qid_int not in preprompt_map:
            skipped += 1
            continue
        row = dict(s)
        row["original_question"] = s["question"]
        gp = preprompt_map[qid_int]
        situation = "" if omit_dataset_situation else (s.get("situation") or "")
        vlm_question, sit_info = merge_situation_with_preprompt(
            gp,
            situation,
            force=force_situation_merge,
        )
        if omit_dataset_situation:
            sit_info = dict(sit_info)
            sit_info["dataset_situation_omitted_for_merge"] = True
        row["question"] = vlm_question
        row["vlm_prompt"] = vlm_question
        row["preprompt_raw"] = gp
        row["vlm_situation_merge"] = sit_info
        row["preprompt_source_question_id"] = qid_int
        merged.append(row)
    return merged, skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SQA eval using per-question generated_prompt from a preprompt JSON (like evaluate_sqa.py)"
    )
    parser.add_argument(
        "--preprompt-json",
        type=str,
        required=True,
        help="Path to JSON from generate_vlm_preprompts.py (contains results[].generated_prompt)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        choices=["train", "val", "test"],
        help="Dataset split (default: meta.split from preprompt JSON, else test)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2-VL-2B-Instruct",
        help="VideoLM model name or path",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--dataset-dir", type=str, default="dataset/SQA")
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--no-nlp-eval", action="store_true")
    parser.add_argument("--no-clear-cache", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument(
        "--question-types",
        type=str,
        nargs="+",
        default=None,
        help="Same as evaluate_sqa: filter by question-type tags",
    )
    parser.add_argument(
        "--filter-scenes-json",
        type=str,
        default=None,
        help="Same as evaluate_sqa: restrict to scene IDs listed in JSON",
    )
    parser.add_argument(
        "--strict-split",
        action="store_true",
        help="Exit with error if --split differs from preprompt meta.split",
    )
    parser.add_argument(
        "--force-situation-merge",
        action="store_true",
        help=(
            "Always prepend a Situation: block from the dataset before generated_prompt, "
            "even if the preprompt already appears to contain the situation text."
        ),
    )
    parser.add_argument(
        "--omit-dataset-situation",
        action="store_true",
        help=(
            "Do not pass the dataset observer situation into the prepend merge: treat it as empty "
            "so ``vlm_situation_merge.situation_empty`` is true and no Situation: block is added "
            "from annotations (VLM still sees ``generated_prompt`` only, which may embed situation inside)."
        ),
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=0,
        metavar="N",
        help=(
            "If N > 0 and --output is set, rewrite the output JSON every N completed samples "
            "(same as evaluate_sqa.py). 0 = save only at the end (and on CUDA OOM)."
        ),
    )
    args = parser.parse_args()

    preprompt_path = Path(args.preprompt_json).resolve()
    if not preprompt_path.is_file():
        raise FileNotFoundError(f"--preprompt-json not found: {preprompt_path}")

    preprompt_map, preprompt_meta = load_preprompt_map(str(preprompt_path))
    if not preprompt_map:
        raise SystemExit(f"No usable generated_prompt entries in {preprompt_path}")

    split = args.split
    if split is None:
        split = preprompt_meta.get("split") or "test"
        print(f"Using split from preprompt meta: {split}")

    meta_split = preprompt_meta.get("split")
    if args.strict_split and meta_split and meta_split != split:
        raise SystemExit(f"--split {split} != preprompt meta.split {meta_split}")

    dataset_dir = Path(args.dataset_dir)
    questions_path = dataset_dir / "sqa_task" / "balanced" / f"v1_balanced_questions_{split}_scannetv2.json"
    annotations_path = dataset_dir / "sqa_task" / "balanced" / f"v1_balanced_sqa_annotations_{split}_scannetv2.json"
    video_dir = dataset_dir / "video"

    if not questions_path.exists():
        raise FileNotFoundError(questions_path)
    if not annotations_path.exists():
        raise FileNotFoundError(annotations_path)
    if not video_dir.exists():
        raise FileNotFoundError(video_dir)

    samples = load_dataset(str(questions_path), str(annotations_path), str(video_dir))
    print(f"Loaded {len(samples)} dataset samples; preprompt map has {len(preprompt_map)} ids")

    # Optional filters (same logic as evaluate_sqa)
    if args.filter_scenes_json:
        if not os.path.exists(args.filter_scenes_json):
            raise FileNotFoundError(args.filter_scenes_json)
        with open(args.filter_scenes_json, "r") as f:
            filter_data = json.load(f)
        if "videos" in filter_data:
            scene_ids = [v["scene_id"] for v in filter_data["videos"]]
        elif isinstance(filter_data, list):
            scene_ids = filter_data
        elif "scene_ids" in filter_data:
            scene_ids = filter_data["scene_ids"]
        else:
            raise ValueError("Could not parse scene IDs from filter JSON")
        samples = [s for s in samples if s["scene_id"] in scene_ids]
        print(f"After scene filter: {len(samples)} samples")

    if args.question_types:
        from utils.question_type_filter import filter_samples_by_question_type

        samples = filter_samples_by_question_type(samples, args.question_types, match_any=True)
        print(f"After question-type filter: {len(samples)} samples")

    if args.omit_dataset_situation and args.force_situation_merge:
        print("Note: --omit-dataset-situation wins; there is no dataset situation to prepend.")

    samples, n_skip = merge_preprompts_into_samples(
        samples,
        preprompt_map,
        force_situation_merge=args.force_situation_merge,
        omit_dataset_situation=args.omit_dataset_situation,
    )
    n_prepended = sum(
        1 for s in samples if (s.get("vlm_situation_merge") or {}).get("prepended_situation_block")
    )
    print(
        f"Merged preprompts: {len(samples)} samples to evaluate ({n_skip} skipped: no preprompt for question_id); "
        f"Situation prepended for VLM on {n_prepended} samples"
        + (" (force on all non-empty situations)" if args.force_situation_merge else "")
        + ("; dataset situation omitted for prepend (--omit-dataset-situation)" if args.omit_dataset_situation else "")
    )

    if not samples:
        raise SystemExit("No samples left after merging preprompts. Check split and preprompt JSON overlap.")

    if args.load_in_4bit and args.load_in_8bit:
        raise ValueError("Cannot use both --load-in-4bit and --load-in-8bit")

    if torch.cuda.is_available() and not args.no_clear_cache:
        torch.cuda.empty_cache()
        gc.collect()

    print(f"\nInitializing VideoLM: {args.model_name}")
    # No YAML template: full preprompt is passed as `question` to the VLM.
    model = VideoLM(
        model_name=args.model_name,
        max_frames=args.max_frames,
        frame_size=(448, 448),
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        prompt_template=None,
    )

    if not args.output:
        model_suffix = args.model_name.split("/")[-1].replace("-", "_").lower()
        args.output = f"sqa_{split}_results_preprompts_{model_suffix}.json"

    save_interval = args.save_interval if args.save_interval > 0 else None
    evaluation_settings = build_evaluation_settings(
        script=os.path.basename(__file__),
        model_name=args.model_name,
        temperature=float(args.temperature),
        max_new_tokens=args.max_new_tokens,
        max_frames=args.max_frames,
        split=split,
        dataset_dir=args.dataset_dir,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        preprompt_json=str(preprompt_path),
        preprompt_meta=preprompt_meta,
        preprompt_force_situation_merge=args.force_situation_merge,
        preprompt_omit_dataset_situation=args.omit_dataset_situation,
        chain_of_thought=False,
        save_interval=save_interval,
    )

    results = evaluate(
        model=model,
        samples=samples,
        output_file=args.output,
        max_samples=args.max_samples,
        use_nlp_eval=not args.no_nlp_eval,
        clear_cache=not args.no_clear_cache,
        prompt_template=None,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        evaluation_settings=evaluation_settings,
        save_interval=save_interval,
    )

    print("\n" + "=" * 60)
    print("Evaluation Results (VLM input = generated_prompt per question)")
    print("=" * 60)
    print(f"Exact Match Accuracy: {results['accuracy']:.4f} ({results['correct']}/{results['total']})")
    print(f"Results file: {args.output} (each row includes vlm_prompt = text sent to the VLM)")
    print("=" * 60)


if __name__ == "__main__":
    main()
