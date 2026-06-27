"""
Two-pass **simple** CoT: both passes use question + situation + vision.

Pass 1: (Q + S + V) → full output (reasoning + Answer: line)
Pass 2: (Q + S + V) + pass-1 output → refined full output; metrics use final Answer line.

For the **standard** two-pass CoT in this repo — pass 1: situation + images (spatial
reasoning only), pass 2: reasoning + question + images — see ``evaluate_cot_qwen25.py``
and ``evaluate_cot_qwen3.py``. Video frames and render resizing follow the same
pattern as those scripts (``VideoLM.video_processor.process_video``, LANCZOS to
``model.frame_size`` for renders).
"""

from __future__ import annotations

import gc
import json
import os
import signal
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from PIL import Image
from tqdm import tqdm

from evaluate_cot_qwen25 import exact_match, load_render_images
from utils.evaluation_result_meta import (
    attach_result_json_metadata,
    normalize_question_id,
)
from videolm import VideoLM
from videolm.evaluators import AnswerEvaluator


def build_simple_cot_pass1_prompt(situation: str, question: str, input_type: str) -> str:
    vis = "video frames" if input_type == "video" else "rendered views"
    return f"""You are analyzing a 3D indoor scene from {vis} and a situation description.

SITUATION:
{situation}

QUESTION:
{question}

The images above are {vis} of this scene.

Think step by step (chain of thought), then give your best answer on its own line at the end:
Answer: <short phrase>"""


def build_simple_cot_pass2_prompt(
    situation: str,
    question: str,
    pass1_output: str,
    input_type: str,
) -> str:
    vis = "video frames" if input_type == "video" else "rendered views"
    p1 = (pass1_output or "").strip()
    return f"""You already produced a first chain-of-thought answer. Refine it using the same situation, question, and {vis}.

SITUATION:
{situation}

QUESTION:
{question}

YOUR FIRST OUTPUT (reasoning + answer):
{p1}

The images above are the same {vis}.

Review and improve. Think step by step again if needed, then give your final answer on its own line:
Answer: <short phrase>"""


def extract_final_answer_line(text: str) -> str:
    if not text or not str(text).strip():
        return ""
    t = text.strip()
    low = t.lower()
    key = "answer:"
    idx = low.rfind(key)
    if idx != -1:
        rest = t[idx + len("answer:") :].strip()
        first = rest.splitlines()[0].strip()
        return first
    lines = [x.strip() for x in t.splitlines() if x.strip()]
    return lines[-1] if lines else t


def _load_simple_cot_checkpoint(path: str) -> Tuple[Dict[str, Any], Set[Any], Set[Any]]:
    if not os.path.exists(path):
        return {}, set(), set()
    print(f"\n📂 Loading checkpoint from: {path}")
    with open(path, "r") as f:
        data = json.load(f)
    processed: Set[Any] = set()
    oom_ids: Set[Any] = set()
    for r in data.get("results", []):
        qid = normalize_question_id(r.get("question_id"))
        if qid is None:
            continue
        if (
            r.get("predicted_answer") == "CUDA_OOM_ERROR"
            or r.get("error") == "CUDA_OUT_OF_MEMORY"
            or r.get("excluded_from_metrics", False)
        ):
            oom_ids.add(qid)
        else:
            processed.add(qid)
    print(f"   Found {len(data.get('results', []))} results; skip {len(processed)}, rerun OOM {len(oom_ids)}")
    return data, processed, oom_ids


def _save_simple_cot_checkpoint(
    output_file: str,
    *,
    results: List[Dict],
    metrics: Dict[str, Dict[str, int]],
    correct: int,
    total: int,
    samples_total: int,
    is_final: bool,
    evaluation_settings: Optional[Dict[str, Any]],
    input_type: str,
    render_dir: Optional[str],
    scenes_filter: Optional[List[str]],
    max_render_views: Optional[int],
    max_frames: Optional[int],
    model_name: str,
) -> None:
    if not output_file:
        return
    accuracies: Dict[str, float] = {}
    for method, counts in metrics.items():
        accuracies[method] = float(counts["correct"] / counts["total"]) if counts["total"] > 0 else 0.0
    accuracy = float(accuracies.get("exact_match", correct / total if total > 0 else 0.0))
    doc: Dict[str, Any] = {
        "accuracy": accuracy,
        "accuracies": accuracies,
        "correct": correct,
        "total": total,
        "samples_total": samples_total,
        "samples_remaining": samples_total - total,
        "metrics": metrics,
        "results": results,
        "method": "simple_cot_twopass",
        "input_type": input_type,
        "model_name": model_name,
        "render_dir": render_dir if input_type == "render" else None,
        "scenes_filter": scenes_filter,
        "max_render_views": max_render_views if input_type == "render" else None,
        "max_frames": max_frames if input_type == "video" else None,
        "is_checkpoint": not is_final,
        "checkpoint_time": datetime.now().isoformat(),
    }
    attach_result_json_metadata(doc, evaluation_settings)
    try:
        with open(output_file, "w") as f:
            json.dump(doc, f, indent=2)
        if not is_final:
            print(f"\n💾 Checkpoint saved: {len(results)}/{samples_total} samples in results")
    except Exception as e:
        print(f"⚠️  Warning: Failed to save checkpoint: {e}")


def evaluate_simple_cot_two_pass(
    model: VideoLM,
    samples: List[Dict],
    input_type: str = "video",
    render_dir: Optional[str] = None,
    output_file: Optional[str] = None,
    max_samples: Optional[int] = None,
    use_nlp_eval: bool = True,
    clear_cache: bool = True,
    max_new_tokens: int = 256,
    temperature: float = 0.3,
    scenes_filter: Optional[List[str]] = None,
    max_render_views: Optional[int] = None,
    max_frames: Optional[int] = None,
    checkpoint_interval: Optional[int] = None,
    resume_from: Optional[str] = None,
    evaluation_settings: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    samples_work: List[Dict] = list(samples)

    results: List[Dict] = []
    correct = 0
    total = 0
    metrics: Dict[str, Dict[str, int]] = {}

    if resume_from and os.path.exists(resume_from):
        ck, processed_ids, oom_ids = _load_simple_cot_checkpoint(resume_from)
        all_res = ck.get("results", [])
        results = [
            r for r in all_res
            if normalize_question_id(r.get("question_id")) not in oom_ids
        ]
        metrics = ck.get("metrics", metrics)
        correct = ck.get("correct", 0)
        total = ck.get("total", 0)
        samples_work = [
            s for s in samples_work
            if normalize_question_id(s.get("question_id")) not in processed_ids
        ]
        print(f"\n🔄 Resuming: {len(samples_work)} samples left to process")
    elif resume_from:
        print(f"⚠️  Warning: Checkpoint not found: {resume_from}. Starting fresh.")

    if max_samples:
        samples_work = samples_work[:max_samples]

    nlp_evaluator = AnswerEvaluator() if use_nlp_eval else None

    samples_total = len(samples_work) + total
    shutdown_requested = {"flag": False}

    def signal_handler(signum, frame):
        print("\n\n⚠️  Interrupt received. Saving progress...")
        shutdown_requested["flag"] = True

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    def maybe_save(is_final: bool) -> None:
        if not output_file:
            return
        _save_simple_cot_checkpoint(
            output_file,
            results=results,
            metrics=metrics,
            correct=correct,
            total=total,
            samples_total=samples_total,
            is_final=is_final,
            evaluation_settings=evaluation_settings,
            input_type=input_type,
            render_dir=render_dir,
            scenes_filter=scenes_filter,
            max_render_views=max_render_views,
            max_frames=max_frames,
            model_name=getattr(model, "model_name", "") or "",
        )

    iterator = tqdm(samples_work, desc="Simple CoT (2-pass)")
    for sample in iterator:
        if shutdown_requested["flag"]:
            print("\n⚠️  Shutting down gracefully...")
            maybe_save(is_final=False)
            break

        scene_id = sample["scene_id"]
        question_id = sample["question_id"]
        question = sample["question"]
        situation = sample.get("situation", "")
        gt_answer = sample["gt_answer"]

        try:
            if input_type == "video":
                video_path = sample["video_path"]
                if not os.path.exists(video_path):
                    print(f"Warning: Video not found: {video_path}")
                    continue
                frames = model.video_processor.process_video(video_path)
                if max_frames is not None and len(frames) > max_frames:
                    frames = frames[:max_frames]
                if not frames:
                    print(f"Warning: No frames extracted from {video_path}")
                    continue
                images_list = frames
            else:
                if not render_dir:
                    raise ValueError("render_dir required when input_type is 'render'")
                render_images = load_render_images(scene_id, render_dir, max_render_views)
                if not render_images:
                    print(f"Warning: No render images for {scene_id}")
                    continue
                images_list = []
                for img in render_images:
                    if img.size != model.frame_size:
                        img = img.resize(model.frame_size, Image.Resampling.LANCZOS)
                    images_list.append(img)

            p1 = build_simple_cot_pass1_prompt(situation, question, input_type)
            answer1_full = model.answer_question_from_images(
                images=images_list,
                question=p1,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            if clear_cache and torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            p2 = build_simple_cot_pass2_prompt(situation, question, answer1_full, input_type)
            answer2_full = model.answer_question_from_images(
                images=images_list,
                question=p2,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )

            predicted_answer = extract_final_answer_line(answer2_full)
            pred_pass1 = extract_final_answer_line(answer1_full)

            is_correct_exact = exact_match(predicted_answer, gt_answer)
            if is_correct_exact:
                correct += 1
            total += 1

            nlp_metrics: Dict[str, Any] = {}
            if nlp_evaluator:
                nlp_metrics = nlp_evaluator.evaluate(predicted_answer, gt_answer)

            metric_rows = [("exact_match", is_correct_exact)]
            if nlp_evaluator:
                metric_rows.extend(
                    [
                        ("semantic", bool(nlp_metrics.get("correct_semantic", False))),
                        ("bleu", bool(nlp_metrics.get("correct_bleu", False))),
                        ("rouge", bool(nlp_metrics.get("correct_rouge", False))),
                        ("fuzzy", bool(nlp_metrics.get("correct_fuzzy", False))),
                        ("contains", bool(nlp_metrics.get("correct_contains", False))),
                    ]
                )
            for method, is_correct_method in metric_rows:
                if method not in metrics:
                    metrics[method] = {"correct": 0, "total": 0}
                metrics[method]["total"] += 1
                if is_correct_method:
                    metrics[method]["correct"] += 1

            result: Dict[str, Any] = {
                **sample,
                "predicted_answer": predicted_answer,
                "predicted_answer_pass1": pred_pass1,
                "cot_pass1": (answer1_full or "").strip(),
                "cot_pass2": (answer2_full or "").strip(),
                "cot_reasoning": (answer1_full or "").strip(),
                "cot_refined_reasoning": (answer2_full or "").strip(),
                "cot_method": "simple_twopass",
                "input_type": input_type,
                "correct": is_correct_exact,
            }
            if nlp_evaluator:
                result.update(
                    {
                        "correct_semantic": nlp_metrics.get("correct_semantic", False),
                        "correct_bleu": nlp_metrics.get("correct_bleu", False),
                        "correct_rouge": nlp_metrics.get("correct_rouge", False),
                        "correct_fuzzy": nlp_metrics.get("correct_fuzzy", False),
                        "correct_contains": nlp_metrics.get("correct_contains", False),
                        "semantic_similarity": nlp_metrics.get("semantic_similarity", 0.0),
                        "bleu_score": nlp_metrics.get("bleu_score", 0.0),
                        "rouge1": nlp_metrics.get("rouge1", 0.0),
                        "rougeL": nlp_metrics.get("rougeL", 0.0),
                        "fuzzy_similarity": nlp_metrics.get("fuzzy_similarity", 0.0),
                        "contains_score": nlp_metrics.get("contains_score", 0.0),
                    }
                )
            results.append(result)
            if output_file and checkpoint_interval and len(results) % checkpoint_interval == 0:
                maybe_save(is_final=False)

            if clear_cache and torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

        except torch.cuda.OutOfMemoryError as e:
            print(f"\n❌ CUDA OOM on {scene_id}: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            if output_file:
                print(f"\n💾 Saving partial results to {output_file}...")
                maybe_save(is_final=False)
                try:
                    with open(output_file, "r") as f:
                        ck = json.load(f)
                    ck["error"] = "CUDA_OUT_OF_MEMORY"
                    ck["error_message"] = str(e)
                    ck["stopped_at_sample"] = len(results)
                    with open(output_file, "w") as f:
                        json.dump(ck, f, indent=2)
                except Exception:
                    pass
            raise RuntimeError(
                f"CUDA OOM at sample {len(results) + 1}/{samples_total}"
            ) from e

        except Exception as e:
            error_msg = str(e)
            print(f"Error processing {scene_id}: {error_msg}")
            if "out of memory" in error_msg.lower() and torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

            result = {
                **sample,
                "predicted_answer": f"ERROR: {error_msg}",
                "predicted_answer_pass1": "",
                "cot_pass1": "",
                "cot_pass2": "",
                "cot_reasoning": "",
                "cot_refined_reasoning": "",
                "cot_method": "simple_twopass",
                "input_type": input_type,
                "correct": False,
            }
            if nlp_evaluator:
                result.update(
                    {
                        "correct_semantic": False,
                        "correct_bleu": False,
                        "correct_rouge": False,
                        "correct_fuzzy": False,
                        "correct_contains": False,
                        "semantic_similarity": 0.0,
                        "bleu_score": 0.0,
                        "rouge1": 0.0,
                        "rougeL": 0.0,
                        "fuzzy_similarity": 0.0,
                        "contains_score": 0.0,
                    }
                )
            results.append(result)
            total += 1
            err_rows = [("exact_match", False)]
            if nlp_evaluator:
                err_rows.extend(
                    [
                        ("semantic", False),
                        ("bleu", False),
                        ("rouge", False),
                        ("fuzzy", False),
                        ("contains", False),
                    ]
                )
            for method, is_correct_method in err_rows:
                if method not in metrics:
                    metrics[method] = {"correct": 0, "total": 0}
                metrics[method]["total"] += 1
                if is_correct_method:
                    metrics[method]["correct"] += 1
            if output_file and checkpoint_interval and len(results) % checkpoint_interval == 0:
                maybe_save(is_final=False)
            if clear_cache and torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

    accuracies: Dict[str, float] = {}
    for method, counts in metrics.items():
        accuracies[method] = float(counts["correct"] / counts["total"]) if counts["total"] > 0 else 0.0

    accuracy = float(accuracies.get("exact_match", correct / total if total > 0 else 0.0))

    if output_file:
        maybe_save(is_final=True)
        print(f"\n✅ Final results saved to: {output_file}")

    return {
        "accuracy": accuracy,
        "accuracies": accuracies,
        "correct": correct,
        "total": total,
        "metrics": metrics,
        "results": results,
    }
