"""
SQA evaluation with **simple** two-pass chain-of-thought:

  Pass 1: (question + situation + vision) → Answer₁ (full CoT + Answer: line)
  Pass 2: (question + situation + vision) + Answer₁ → Answer₂ (refined; metrics use this)

**Related scripts (standard spatial CoT, different prompt schedule):**

- ``evaluate_cot_qwen25.py`` — Qwen2.5-VL; pass 1: S+V → spatial reasoning; pass 2: +Q.
- ``evaluate_cot_qwen3.py`` — Qwen3-VL; same structure; adds checkpoint / resume / memory
  threshold flags for long runs.

This file keeps one CLI via ``--backend`` and uses ``utils/simple_cot_two_pass.py``.
Backends include **Qwen2-VL** (``qwen2_2b``, ``qwen2_7b``), Qwen2.5-VL, Qwen3-VL, and Llama Vision.

**Quantization:** use ``--qwen-quantization safe`` (default) to avoid bitsandbytes ``CB``
errors on some stacks, or ``--qwen-quantization cot`` to match ``evaluate_cot_qwen25.py`` /
``evaluate_cot_qwen3.py`` (8-bit unless overridden). Applies to all Qwen* backends here.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Optional, Set, Tuple

import torch

from evaluate_cot_qwen25 import check_qwen25_model, load_dataset
from evaluate_cot_qwen3 import check_qwen3_model
from evaluate_sqa_llama4 import check_llama4_model, resolve_llama_vision_model_id
from utils.question_type_filter import filter_samples_by_question_type, get_question_type_statistics
from utils.evaluation_result_meta import build_evaluation_settings, checkpoint_json_incomplete
from utils.simple_cot_two_pass import evaluate_simple_cot_two_pass
from videolm import VideoLM

BACKEND_DEFAULT_MODEL = {
    "qwen2_2b": "Qwen/Qwen2-VL-2B-Instruct",
    "qwen2_7b": "Qwen/Qwen2-VL-7B-Instruct",
    "qwen25": "Qwen/Qwen2.5-VL-3B-Instruct",
    "qwen3": "Qwen/Qwen3-VL-2B-Instruct",
    "llama": "meta-llama/Llama-3.2-11B-Vision-Instruct",
}


def check_qwen2_vl_model(model_name: str) -> bool:
    """True for original Qwen2-VL ids (not Qwen2.5-VL or Qwen3-VL)."""
    low = model_name.lower()
    if "qwen3" in low:
        return False
    if "2.5" in low or "qwen2_5" in low or "qwen2.5" in low:
        return False
    return "qwen2-vl" in low or ("qwen2" in low and "vl" in low and "2.5" not in low)


def _validate_model_for_backend(backend: str, model_name: str) -> None:
    ok = False
    if backend in ("qwen2_2b", "qwen2_7b"):
        ok = check_qwen2_vl_model(model_name)
        hint = "Use a Hugging Face Qwen2-VL Instruct id (e.g. Qwen/Qwen2-VL-2B-Instruct). For Qwen2.5 use --backend qwen25."
    elif backend == "qwen25":
        ok = check_qwen25_model(model_name)
        hint = "For Qwen2-VL use --backend qwen2_2b or qwen2_7b; for Qwen3-VL use --backend qwen3."
    elif backend == "qwen3":
        ok = check_qwen3_model(model_name)
        hint = "For Qwen2.5-VL use --backend qwen25; for Qwen2-VL use --backend qwen2_2b / qwen2_7b."
    else:
        ok = check_llama4_model(model_name)
        hint = "Use a Llama Vision Instruct id, or a Qwen --backend."

    if ok:
        return
    print(f"⚠️  Warning: Model '{model_name}' does not look like a typical {backend} model.")
    print(f"   {hint}")
    if input("   Continue anyway? (y/n): ").lower() != "y":
        raise SystemExit("Exiting.")


def _resolve_qwen_quantization_cot_defaults(args: argparse.Namespace) -> Tuple[bool, bool]:
    """
    Same quantization policy as ``evaluate_cot_qwen25.py`` / ``evaluate_cot_qwen3.py`` main():
    default 8-bit, ``--no-quantization`` for full precision, 4-bit with confirmation.
    """
    load_in_4bit = False
    load_in_8bit = True

    if args.no_quantization:
        load_in_4bit = False
        load_in_8bit = False
        print("⚠️  Running without quantization (requires more GPU memory)")
        print("   This is the most reliable option for Qwen-VL if quantization fails")
    elif args.load_in_4bit:
        load_in_4bit = True
        load_in_8bit = False
        print("⚠️  WARNING: Using 4-bit quantization with Qwen-VL may cause AssertionError!")
        print("   Recommended: Use --load-in-8bit instead (or omit flag, 8-bit is default)")
        if input("   Continue with 4-bit anyway? (y/n): ").lower() != "y":
            print("Switching to 8-bit quantization (default like evaluate_cot_qwen25/3)...")
            load_in_4bit = False
            load_in_8bit = True
    elif args.load_in_8bit:
        load_in_4bit = False
        load_in_8bit = True
        print("✓ Using 8-bit quantization")
        print("   ⚠️  Note: If you encounter 'CB' attribute error, use --no-quantization instead")

    return load_in_4bit, load_in_8bit


def _resolve_qwen_quantization_safe(args: argparse.Namespace) -> Tuple[bool, bool]:
    """
    Default for ``evaluate_sqa_simple_cot.py``: full precision unless user opts into 8-bit
    with ``--accept-8bit-risk`` (avoids common bitsandbytes ``CB`` errors on Qwen-VL).
    """
    if args.no_quantization:
        return False, False
    if args.load_in_4bit:
        print("⚠️  WARNING: 4-bit Qwen-VL often fails (vision stack / bitsandbytes). Prefer full precision.")
        if input("   Continue with 4-bit anyway? (y/n): ").lower() != "y":
            print("Using full precision (no quantization).")
            return False, False
        return True, False
    if args.load_in_8bit:
        if not getattr(args, "accept_8bit_risk", False):
            print(
                "\n❌ Refusing --load-in-8bit for Qwen-VL without --accept-8bit-risk.\n"
                "   On many setups (including common PyTorch 3.x + bitsandbytes paths) this raises:\n"
                "     AttributeError: 'Parameter' object has no attribute 'CB'\n"
                "   Or use the same defaults as evaluate_cot_qwen25.py:\n"
                "     --qwen-quantization cot\n\n"
                "   Run one of:\n"
                "     python evaluate_sqa_simple_cot.py --backend qwen25 --input-type video\n"
                "     … same … --load-in-8bit --accept-8bit-risk   # only if you know 8-bit works\n"
            )
            sys.exit(2)
        print(
            "⚠️  8-bit Qwen-VL with --accept-8bit-risk: expect CB errors if your stack is incompatible."
        )
        return False, True
    return False, False


def _resolve_qwen_quantization(args: argparse.Namespace) -> Tuple[bool, bool]:
    mode = getattr(args, "qwen_quantization", "safe")
    if mode == "cot":
        return _resolve_qwen_quantization_cot_defaults(args)
    return _resolve_qwen_quantization_safe(args)


def _resolve_llama_quantization(args: argparse.Namespace) -> Tuple[bool, bool]:
    if args.no_quantization:
        return False, False
    if args.load_in_4bit:
        print("⚠️  Using 4-bit quantization (may fail on some Llama Vision builds).")
        return True, False
    if args.load_in_8bit:
        return False, True
    return False, True


def _scene_filter_from_args(args: argparse.Namespace) -> Optional[Set[str]]:
    if args.filter_scenes_json:
        if not os.path.exists(args.filter_scenes_json):
            raise FileNotFoundError(f"Filter scenes JSON not found: {args.filter_scenes_json}")
        with open(args.filter_scenes_json, "r") as f:
            filter_data = json.load(f)
        if "videos" in filter_data:
            ids = [v["scene_id"] for v in filter_data["videos"]]
        elif isinstance(filter_data, list):
            ids = filter_data
        elif "scene_ids" in filter_data:
            ids = filter_data["scene_ids"]
        else:
            raise ValueError("Could not parse scene IDs (expected 'videos', list, or 'scene_ids').")
        print(f"Loaded {len(ids)} scene IDs from {args.filter_scenes_json}")
        s = set(ids)
        if args.scenes:
            s = s.union(set(args.scenes))
        return s
    if args.scenes:
        return set(args.scenes)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SQA simple two-pass CoT: (Q+S+V)→A₁; (Q+S+V)+A₁→A₂.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Qwen-VL: safe mode = full precision (like evaluate_cot_qwen25 + --no-quantization on broken stacks)
  python evaluate_sqa_simple_cot.py --backend qwen25 --input-type video
  # Same 8-bit default as evaluate_cot_qwen25.py / evaluate_cot_qwen3.py (may hit CB on some setups)
  python evaluate_sqa_simple_cot.py --backend qwen25 --input-type video --qwen-quantization cot
  python evaluate_sqa_simple_cot.py --backend qwen25 --input-type video --load-in-8bit --accept-8bit-risk
  python evaluate_sqa_simple_cot.py --backend qwen2_2b --input-type video
  python evaluate_sqa_simple_cot.py --backend qwen2_7b --input-type video
  python evaluate_sqa_simple_cot.py --backend qwen3 --input-type render --model-name Qwen/Qwen3-VL-4B-Instruct
  python evaluate_sqa_simple_cot.py --backend llama --input-type video --no-quantization

  # Custom temperature, output path, 8 frames, checkpoint every 10 samples
  python evaluate_sqa_simple_cot.py --backend qwen25 --input-type video --max-frames 8 \\
    --temperature 0.7 --output my_run.json --checkpoint-interval 10 --qwen-quantization cot

Standard spatial CoT (different prompts) lives in ``evaluate_cot_qwen25.py`` and
``evaluate_cot_qwen3.py``. Do not use YAML ``prompt_config`` here: prompts are fixed.
        """,
    )
    parser.add_argument(
        "--backend",
        type=str,
        required=True,
        choices=["qwen2_2b", "qwen2_7b", "qwen25", "qwen3", "llama"],
        help="Model family: Qwen2-VL 2B/7B, Qwen2.5-VL, Qwen3-VL, or Llama Vision",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="HF model id (defaults per --backend if omitted)",
    )
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--dataset-dir", type=str, default="dataset/SQA")
    parser.add_argument("--input-type", type=str, required=True, choices=["video", "render"])
    parser.add_argument("--render-dir", type=str, default="dataset/SQA/render")
    parser.add_argument("--scenes", type=str, nargs="+", default=None)
    parser.add_argument("--filter-scenes-json", type=str, default=None)
    parser.add_argument("--max-frames", type=int, default=4, help="Video frames (video input)")
    parser.add_argument("--max-render-views", type=int, default=5)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--no-nlp-eval", action="store_true")
    parser.add_argument("--no-clear-cache", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Qwen-VL: 8-bit via bitsandbytes (often raises CB errors; requires --accept-8bit-risk here).",
    )
    parser.add_argument(
        "--accept-8bit-risk",
        action="store_true",
        help="Allow --load-in-8bit for Qwen backends (otherwise the script exits before loading the model).",
    )
    parser.add_argument(
        "--no-quantization",
        action="store_true",
        help="Full precision (with --qwen-quantization cot, same as evaluate_cot_qwen25/3).",
    )
    parser.add_argument(
        "--qwen-quantization",
        type=str,
        choices=["safe", "cot"],
        default="safe",
        help=(
            "For Qwen backends (qwen2_2b, qwen2_7b, qwen25, qwen3). "
            "'safe': full precision by default; --load-in-8bit needs --accept-8bit-risk. "
            "'cot': same as evaluate_cot_qwen25.py / evaluate_cot_qwen3.py (8-bit default). "
            "Ignored for --backend llama."
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--question-types", type=str, nargs="+", default=None)
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=0,
        help="Save JSON every N rows appended to results (0 = off). OOM still writes partial output.",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Resume from this JSON. If omitted and --output looks like a checkpoint, auto-resume.",
    )

    args = parser.parse_args()

    model_name = args.model_name or BACKEND_DEFAULT_MODEL[args.backend]
    if args.backend == "llama":
        model_name = resolve_llama_vision_model_id(model_name)

    _validate_model_for_backend(args.backend, model_name)

    if args.backend in ("qwen2_2b", "qwen2_7b", "qwen25", "qwen3"):
        load_in_4bit, load_in_8bit = _resolve_qwen_quantization(args)
        if args.qwen_quantization == "cot":
            if not load_in_4bit and not load_in_8bit:
                print("✓ Qwen-VL: full precision (--qwen-quantization cot, like evaluate_cot_qwen25/3 + --no-quantization).")
            elif load_in_8bit:
                print("✓ Qwen-VL: 8-bit (--qwen-quantization cot; matches evaluate_cot_qwen25/3 defaults).")
        else:
            if not load_in_4bit and not load_in_8bit:
                print("✓ Qwen-VL: full precision (--qwen-quantization safe; avoids common 'CB' errors).")
            elif load_in_8bit:
                print("✓ Qwen-VL: 8-bit (--accept-8bit-risk with safe mode).")
    else:
        load_in_4bit, load_in_8bit = _resolve_llama_quantization(args)
        if args.no_quantization:
            print("⚠️  Llama Vision without quantization (more VRAM).")

    dataset_dir = Path(args.dataset_dir)
    questions_path = dataset_dir / "sqa_task" / "balanced" / f"v1_balanced_questions_{args.split}_scannetv2.json"
    annotations_path = dataset_dir / "sqa_task" / "balanced" / f"v1_balanced_sqa_annotations_{args.split}_scannetv2.json"
    video_dir = dataset_dir / "video"

    if not questions_path.exists():
        raise FileNotFoundError(f"Questions file not found: {questions_path}")
    if not annotations_path.exists():
        raise FileNotFoundError(f"Annotations file not found: {annotations_path}")
    if args.input_type == "video" and not video_dir.exists():
        raise FileNotFoundError(f"Video directory not found: {video_dir}")
    if args.input_type == "render":
        if not args.render_dir or not os.path.exists(args.render_dir):
            raise FileNotFoundError(f"Render directory missing: {args.render_dir}")

    print(f"\nLoading {args.split} split...")
    samples = load_dataset(str(questions_path), str(annotations_path), str(video_dir))
    print(f"Loaded {len(samples)} samples")

    scene_ids_to_filter = _scene_filter_from_args(args)
    if scene_ids_to_filter:
        n0 = len(samples)
        samples = [s for s in samples if s["scene_id"] in scene_ids_to_filter]
        print(f"Scene filter: {n0} → {len(samples)} samples")
        if not samples:
            print("No samples left after scene filter.")
            return

    if args.question_types:
        n0 = len(samples)
        samples = filter_samples_by_question_type(samples, args.question_types, match_any=True)
        print(f"Question-type filter: {n0} → {len(samples)} samples")
        stats = get_question_type_statistics(samples)
        for qtype, count in sorted(stats.items(), key=lambda x: x[1], reverse=True):
            print(f"  {qtype:15s}: {count:5d}")

    if torch.cuda.is_available() and not args.no_clear_cache:
        torch.cuda.empty_cache()
        gc.collect()

    max_frames_vm = args.max_frames if args.input_type == "video" else 8
    print(f"\nLoading model: {model_name}")
    print(f"  Quantization: {'4-bit' if load_in_4bit else '8-bit' if load_in_8bit else 'none'}")
    model = VideoLM(
        model_name=model_name,
        max_frames=max_frames_vm,
        frame_size=(448, 448),
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        prompt_template=None,
    )
    print(f"✓ Model on device: {model.device}")

    out_file = args.output
    if not out_file:
        if args.resume_from and os.path.exists(args.resume_from):
            out_file = args.resume_from
        else:
            suffix = model_name.split("/")[-1].replace("-", "_").lower()
            if args.filter_scenes_json:
                scenes_part = Path(args.filter_scenes_json).stem
            elif args.scenes:
                scenes_part = "_".join(args.scenes)
                if len(scenes_part) > 80:
                    scenes_part = scenes_part[:80] + "..."
            else:
                scenes_part = "all_scenes"
            out_file = f"sqa_{args.split}_simple_cot_{args.backend}_{args.input_type}_{suffix}_{scenes_part}.json"

    resume_from = None
    if args.resume_from:
        if not os.path.exists(args.resume_from):
            print(f"⚠️  Warning: Resume file not found: {args.resume_from}")
        else:
            resume_from = args.resume_from
    else:
        if os.path.exists(out_file):
            try:
                with open(out_file, "r") as f:
                    ck = json.load(f)
                if checkpoint_json_incomplete(ck):
                    print(f"\n✓ Found checkpoint: {out_file} — auto-resuming")
                    resume_from = out_file
                else:
                    print(f"\n⚠️  Output exists and looks complete: {out_file} (will overwrite)")
            except (json.JSONDecodeError, KeyError):
                print(f"\n⚠️  Could not read {out_file} as JSON; starting fresh (overwrite)")

    print("\n" + "=" * 60)
    print("Simple two-pass CoT (Q+S+V on both passes)")
    print("=" * 60)

    evaluation_settings = build_evaluation_settings(
        script=os.path.basename(__file__),
        model_name=model_name,
        temperature=float(args.temperature),
        max_new_tokens=args.max_new_tokens,
        max_frames=args.max_frames if args.input_type == "video" else None,
        split=args.split,
        dataset_dir=str(dataset_dir),
        backend=args.backend,
        chain_of_thought=True,
        cot_method="simple_cot_twopass",
        input_type=args.input_type,
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
        qwen_quantization=args.qwen_quantization,
        no_quantization=args.no_quantization,
        max_render_views=args.max_render_views if args.input_type == "render" else None,
        render_dir=args.render_dir if args.input_type == "render" else None,
        checkpoint_interval=args.checkpoint_interval if args.checkpoint_interval > 0 else None,
    )

    results = evaluate_simple_cot_two_pass(
        model=model,
        samples=samples,
        input_type=args.input_type,
        render_dir=args.render_dir if args.input_type == "render" else None,
        output_file=out_file,
        max_samples=args.max_samples,
        use_nlp_eval=not args.no_nlp_eval,
        clear_cache=not args.no_clear_cache,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        scenes_filter=sorted(scene_ids_to_filter) if scene_ids_to_filter else None,
        max_render_views=args.max_render_views if args.input_type == "render" else None,
        max_frames=args.max_frames if args.input_type == "video" else None,
        checkpoint_interval=args.checkpoint_interval if args.checkpoint_interval > 0 else None,
        resume_from=resume_from,
        evaluation_settings=evaluation_settings,
    )

    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Exact match: {results['accuracy']:.4f} ({results['correct']}/{results['total']})")
    for method, acc in results.get("accuracies", {}).items():
        if method == "exact_match":
            continue
        m = results["metrics"][method]
        print(f"  {method}: {acc:.4f} ({m['correct']}/{m['total']})")
    print("=" * 60)


if __name__ == "__main__":
    main()
