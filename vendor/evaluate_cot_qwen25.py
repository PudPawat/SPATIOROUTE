"""
Chain of Thought (CoT) evaluation script for SQA dataset
Specialized for Qwen2.5-VL models with optimized defaults

Two-pass reasoning: 
  1. First pass: Situation + images → think about position/direction
  2. Second pass: Reasoning from pass 1 + question + images → answer

Supports either Video OR 3D Render images (not both)
"""

import json
import os
import signal
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from tqdm import tqdm
import argparse
import torch
import gc
import yaml
from PIL import Image
import glob

from videolm import VideoLM
from videolm.multimodal_thinking import answer_from_images_with_thinking
from videolm.evaluators import AnswerEvaluator
from utils.question_type_filter import filter_samples_by_question_type, get_question_type_statistics
from utils.evaluation_result_meta import (
    attach_result_json_metadata,
    build_evaluation_settings,
    checkpoint_json_incomplete,
    normalize_question_id,
)


def check_qwen25_model(model_name: str) -> bool:
    """Check if model name is a Qwen2.5-VL model"""
    model_name_lower = model_name.lower()
    return "2.5" in model_name or "qwen2_5" in model_name_lower or "qwen2.5" in model_name_lower


def load_checkpoint(checkpoint_file: str) -> Tuple[Dict, Set[Any], Set[Any]]:
    if not os.path.exists(checkpoint_file):
        return {}, set(), set()

    print(f"\n📂 Loading checkpoint from: {checkpoint_file}")
    with open(checkpoint_file, "r") as f:
        checkpoint_data = json.load(f)

    existing_results = checkpoint_data.get("results", [])
    processed_ids: Set[Any] = set()
    oom_error_ids: Set[Any] = set()

    for r in existing_results:
        qid = normalize_question_id(r.get("question_id"))
        if qid is None:
            continue
        if (
            r.get("predicted_answer") == "CUDA_OOM_ERROR"
            or r.get("error") == "CUDA_OUT_OF_MEMORY"
            or r.get("excluded_from_metrics", False)
        ):
            oom_error_ids.add(qid)
        else:
            processed_ids.add(qid)

    print(f"   Found {len(existing_results)} results in checkpoint:")
    print(f"   - Successfully processed: {len(processed_ids)} (will be skipped)")
    print(f"   - OOM errors: {len(oom_error_ids)} (will be rerun)")

    return checkpoint_data, processed_ids, oom_error_ids


def save_cot_checkpoint(
    output_file: str,
    results: List[Dict],
    metrics: Dict,
    correct: int,
    total: int,
    samples_total: int,
    is_final: bool,
    evaluation_settings: Optional[Dict],
    input_type: str,
    render_dir: Optional[str],
    scenes_filter: Optional[List[str]],
    max_render_views: Optional[int],
    max_frames: Optional[int],
    model_name: str,
) -> None:
    if not output_file:
        return

    accuracies = {}
    for method, counts in metrics.items():
        if counts["total"] > 0:
            accuracies[method] = float(counts["correct"] / counts["total"])
        else:
            accuracies[method] = 0.0

    accuracy = float(accuracies.get("exact_match", correct / total if total > 0 else 0.0))

    checkpoint: Dict[str, Any] = {
        "accuracy": accuracy,
        "accuracies": accuracies,
        "correct": correct,
        "total": total,
        "samples_total": samples_total,
        "samples_remaining": samples_total - total,
        "metrics": metrics,
        "results": results,
        "is_checkpoint": not is_final,
        "checkpoint_time": datetime.now().isoformat(),
        "method": "cot",
        "input_type": input_type,
        "model_name": model_name,
        "render_dir": render_dir if input_type == "render" else None,
        "scenes_filter": scenes_filter,
        "max_render_views": max_render_views if input_type == "render" else None,
        "max_frames": max_frames if input_type == "video" else None,
    }
    attach_result_json_metadata(checkpoint, evaluation_settings)

    try:
        with open(output_file, "w") as f:
            json.dump(checkpoint, f, indent=2)
        if not is_final:
            print(f"\n💾 Checkpoint saved: {len(results)}/{samples_total} samples processed")
    except Exception as e:
        print(f"⚠️  Warning: Failed to save checkpoint: {e}")


def load_dataset(
    questions_path: str,
    annotations_path: str,
    video_dir: str
) -> List[Dict]:
    """
    Load SQA dataset
    
    Args:
        questions_path: Path to questions JSON file
        annotations_path: Path to annotations JSON file
        video_dir: Directory containing video files
        
    Returns:
        List of samples with question, answer, situation, and video path
    """
    # Load questions
    with open(questions_path, 'r') as f:
        questions_data = json.load(f)
        questions = questions_data.get('questions', questions_data)  # Handle both formats
    
    # Load annotations
    with open(annotations_path, 'r') as f:
        annotations_data = json.load(f)
        annotations = annotations_data.get('annotations', annotations_data)  # Handle both formats
    
    # Create mapping from question_id to annotation
    annotation_map = {ann['question_id']: ann for ann in annotations}
    
    # Combine questions with annotations
    samples = []
    for q in questions:
        question_id = q['question_id']
        scene_id = q['scene_id']
        
        # Find corresponding annotation
        if question_id in annotation_map:
            ann = annotation_map[question_id]
            video_path = os.path.join(video_dir, f"{scene_id}.mp4")
            
            # Get ground truth answer (first answer in the list)
            gt_answer = ann['answers'][0]['answer'].lower().strip()
            
            samples.append({
                'question_id': question_id,
                'scene_id': scene_id,
                'question': q['question'],
                'situation': q.get('situation', ''),
                'video_path': video_path,
                'gt_answer': gt_answer,
                'answer_type': ann.get('answer_type', 'unknown')
            })
    
    return samples


def load_render_images(scene_id: str, render_dir: str, max_views: Optional[int] = None) -> List[Image.Image]:
    """
    Load 3D rendered images for a scene
    
    Args:
        scene_id: Scene ID (e.g., 'scene0496_00')
        render_dir: Directory containing render images (e.g., 'dataset/SQA/render')
        max_views: Maximum number of views to load (None = load all)
        
    Returns:
        List of PIL Images (rendered views)
    """
    if not render_dir or not os.path.exists(render_dir):
        return []
    
    # Try to find render directory for this scene
    scene_render_dir = os.path.join(render_dir, scene_id)
    
    # If not found, try with _15 suffix (some scenes have more views)
    if not os.path.exists(scene_render_dir):
        scene_render_dir = os.path.join(render_dir, f"{scene_id}_15")
    
    if not os.path.exists(scene_render_dir):
        return []
    
    # Find render images: capture_XX first, then view_XX (legacy)
    view_files = sorted(glob.glob(os.path.join(scene_render_dir, "capture_*.png")))
    if not view_files:
        view_files = sorted(glob.glob(os.path.join(scene_render_dir, "capture_*.jpg")))
    if not view_files:
        view_files = sorted(glob.glob(os.path.join(scene_render_dir, "view_*.png")))
    if not view_files:
        view_files = sorted(glob.glob(os.path.join(scene_render_dir, "view_*.jpg")))
    
    if not view_files:
        return []
    
    # Limit number of views if specified
    if max_views:
        view_files = view_files[:max_views]
    
    # Load images
    render_images = []
    for view_file in view_files:
        try:
            img = Image.open(view_file).convert('RGB')
            render_images.append(img)
        except Exception as e:
            print(f"Warning: Could not load render image {view_file}: {e}")
            continue
    
    return render_images


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison"""
    answer = answer.lower().strip()
    # Remove common punctuation
    answer = answer.replace('.', '').replace(',', '').replace('!', '').replace('?', '')
    return answer


def exact_match(pred: str, gt: str) -> bool:
    """Check if prediction exactly matches ground truth"""
    pred_norm = normalize_answer(pred)
    gt_norm = normalize_answer(gt)
    return pred_norm == gt_norm


def evaluate_with_cot(
    model: VideoLM,
    samples: List[Dict],
    input_type: str = 'video',  # 'video' or 'render'
    render_dir: Optional[str] = None,
    output_file: str = None,
    max_samples: int = None,
    use_nlp_eval: bool = True,
    clear_cache: bool = True,
    max_new_tokens: int = 256,
    temperature: float = 0.3,
    prompt_template: Optional[str] = None,
    scenes_filter: Optional[List[str]] = None,
    max_render_views: Optional[int] = None,
    max_frames: Optional[int] = None,
    checkpoint_interval: Optional[int] = None,
    resume_from: Optional[str] = None,
    evaluation_settings: Optional[Dict[str, Any]] = None,
    thinking_mode: bool = False,
    top_p: float = 0.9,
    think_end_token_id: int = 151668,
) -> Dict:
    """
    Evaluate using Chain of Thought (CoT) reasoning with two passes
    
    Pass 1: Situation + images → reasoning about position/direction
    Pass 2: Reasoning from pass 1 + question + images → final answer

    With ``thinking_mode=True``, each pass uses ``answer_question_from_images_with_thinking``
    (Qwen3-style ``</think>`` split, or string fallback for Qwen2.5-VL thinking checkpoints).
    Pass 2 receives the **full** pass-1 decoded text in ``REASONING FROM FIRST ANALYSIS``.
    
    Args:
        model: VideoLM model instance
        samples: List of samples to evaluate
        input_type: 'video' or 'render' (which input to use)
        render_dir: Directory containing 3D render images (required if input_type='render')
        output_file: Optional path to save detailed results
        max_samples: Optional limit on number of samples to evaluate
        use_nlp_eval: Whether to use NLP-based evaluation
        clear_cache: Whether to clear GPU cache between samples
        max_new_tokens: Maximum tokens for generation
        temperature: Sampling temperature
        top_p: Nucleus top-p
        thinking_mode: Split thinking vs visible tail on both passes
        think_end_token_id: End-of-thinking token id (Qwen3-VL-Thinking)
        prompt_template: Optional prompt template
        scenes_filter: List of scene IDs to evaluate
        max_render_views: Maximum number of render views to use (None = use all)
        max_frames: Maximum number of video frames to use (None = use model default)
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Validate input type
    if input_type not in ['video', 'render']:
        raise ValueError(f"input_type must be 'video' or 'render', got '{input_type}'")
    
    if input_type == 'render' and not render_dir:
        raise ValueError("render_dir must be specified when input_type='render'")
    
    # Filter by scenes if specified
    if scenes_filter:
        print(f"\nFiltering by scenes: {scenes_filter}")
        original_count = len(samples)
        samples = [s for s in samples if s['scene_id'] in scenes_filter]
        print(f"Filtered from {original_count} to {len(samples)} samples")
        
        if len(samples) == 0:
            print("⚠️  Warning: No samples found for the specified scenes!")
            return {
                'accuracy': 0.0,
                'accuracies': {},
                'correct': 0,
                'total': 0,
                'metrics': {},
                'results': []
            }

    results: List[Dict] = []
    correct = 0
    total = 0
    metrics = {
        'exact_match': {'correct': 0, 'total': 0},
        'semantic': {'correct': 0, 'total': 0},
        'bleu': {'correct': 0, 'total': 0},
        'rouge': {'correct': 0, 'total': 0},
        'fuzzy': {'correct': 0, 'total': 0},
        'contains': {'correct': 0, 'total': 0}
    }

    if resume_from and os.path.exists(resume_from):
        checkpoint_data, processed_ids, oom_error_ids = load_checkpoint(resume_from)
        all_results = checkpoint_data.get("results", [])
        results = [
            r for r in all_results
            if normalize_question_id(r.get("question_id")) not in oom_error_ids
        ]
        metrics = checkpoint_data.get("metrics", metrics)
        correct = checkpoint_data.get("correct", 0)
        total = checkpoint_data.get("total", 0)
        samples = [
            s for s in samples
            if normalize_question_id(s.get("question_id")) not in processed_ids
        ]
        print(f"\n🔄 Resuming evaluation:")
        print(f"   - Skipped {len(processed_ids)} successfully processed samples")
        print(f"   - Rerunning {len(oom_error_ids)} samples that had OOM errors")
        print(f"   - Remaining samples to process: {len(samples)}")
    elif resume_from:
        print(f"⚠️  Warning: Checkpoint file not found: {resume_from}. Starting fresh.")

    if max_samples:
        samples = samples[:max_samples]

    nlp_evaluator = None
    if use_nlp_eval:
        print("Initializing NLP evaluators...")
        nlp_evaluator = AnswerEvaluator()

    if thinking_mode:
        print("\nThinking mode: two-pass CoT uses answer_question_from_images_with_thinking each pass.")

    print(f"\nEvaluating on {len(samples)} samples with Chain of Thought reasoning...")
    print(f"Input type: {input_type.upper()}")
    if input_type == 'render' and render_dir:
        print(f"Render directory: {render_dir}")
        if max_render_views:
            print(f"Maximum render views per scene: {max_render_views}")
    if input_type == 'video' and max_frames:
        print(f"Maximum video frames: {max_frames}")

    samples_total = len(samples) + total
    shutdown_requested = {"flag": False}

    def signal_handler(signum, frame):
        print("\n\n⚠️  Interrupt signal received. Saving progress...")
        shutdown_requested["flag"] = True

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    for idx, sample in enumerate(tqdm(samples, desc="Processing"), 1):
        if shutdown_requested["flag"]:
            print("\n⚠️  Shutting down gracefully...")
            if output_file:
                save_cot_checkpoint(
                    output_file=output_file,
                    results=results,
                    metrics=metrics,
                    correct=correct,
                    total=total,
                    samples_total=samples_total,
                    is_final=False,
                    evaluation_settings=evaluation_settings,
                    input_type=input_type,
                    render_dir=render_dir,
                    scenes_filter=scenes_filter,
                    max_render_views=max_render_views,
                    max_frames=max_frames,
                    model_name=getattr(model, "model_name", ""),
                )
            break

        video_path = sample['video_path']
        question = sample['question']
        situation = sample.get('situation', '')
        gt_answer = sample['gt_answer']
        scene_id = sample['scene_id']
        
        # Load images based on input type
        all_images = []
        input_info = {}
        
        if input_type == 'video':
            # Check if video exists
            if not os.path.exists(video_path):
                print(f"Warning: Video not found: {video_path}")
                result = {
                    **sample,
                    'predicted_answer': 'VIDEO_NOT_FOUND',
                    'cot_reasoning': '',
                    'thinking_mode': thinking_mode,
                    'cot_pass1_thinking': '',
                    'cot_pass1_after_think': '',
                    'cot_pass1_full': '',
                    'cot_pass2_thinking': '',
                    'cot_pass2_after_think': '',
                    'cot_pass2_full': '',
                    'input_type': input_type,
                    'correct': False
                }
                if nlp_evaluator:
                    result.update({
                        'correct_semantic': False,
                        'correct_bleu': False,
                        'correct_rouge': False,
                        'correct_fuzzy': False,
                        'correct_contains': False,
                        'semantic_similarity': 0.0,
                        'bleu_score': 0.0,
                        'rouge1': 0.0,
                        'rougeL': 0.0,
                        'fuzzy_similarity': 0.0,
                        'contains_score': 0.0
                    })
                results.append(result)
                total += 1
                continue
            
            # Load video frames
            video_images = model.video_processor.process_video(video_path)
            if max_frames and len(video_images) > max_frames:
                video_images = video_images[:max_frames]
            all_images = video_images
            input_info = {'video_path': video_path, 'frames_used': len(video_images)}
            
        else:  # input_type == 'render'
            # Load 3D render images
            render_images = load_render_images(scene_id, render_dir, max_render_views)
            if not render_images:
                print(f"Warning: No render images found for {scene_id}, skipping...")
                result = {
                    **sample,
                    'predicted_answer': 'NO_RENDER_IMAGES',
                    'cot_reasoning': '',
                    'thinking_mode': thinking_mode,
                    'cot_pass1_thinking': '',
                    'cot_pass1_after_think': '',
                    'cot_pass1_full': '',
                    'cot_pass2_thinking': '',
                    'cot_pass2_after_think': '',
                    'cot_pass2_full': '',
                    'input_type': input_type,
                    'render_views_used': 0,
                    'correct': False
                }
                if nlp_evaluator:
                    result.update({
                        'correct_semantic': False,
                        'correct_bleu': False,
                        'correct_rouge': False,
                        'correct_fuzzy': False,
                        'correct_contains': False,
                        'semantic_similarity': 0.0,
                        'bleu_score': 0.0,
                        'rouge1': 0.0,
                        'rougeL': 0.0,
                        'fuzzy_similarity': 0.0,
                        'contains_score': 0.0
                    })
                results.append(result)
                total += 1
                continue
            
            # Resize render images to match frame size
            resized_render_images = []
            for img in render_images:
                if img.size != model.frame_size:
                    img = img.resize(model.frame_size, Image.Resampling.LANCZOS)
                resized_render_images.append(img)
            all_images = resized_render_images
            input_info = {'render_views_used': len(all_images)}
            
            # Get paths for logging
            scene_render_dir = os.path.join(render_dir, scene_id)
            if not os.path.exists(scene_render_dir):
                scene_render_dir = os.path.join(render_dir, f"{scene_id}_15")
            if os.path.exists(scene_render_dir):
                view_files = sorted(glob.glob(os.path.join(scene_render_dir, "capture_*.png")))
                if not view_files:
                    view_files = sorted(glob.glob(os.path.join(scene_render_dir, "capture_*.jpg")))
                if not view_files:
                    view_files = sorted(glob.glob(os.path.join(scene_render_dir, "view_*.png")))
                if not view_files:
                    view_files = sorted(glob.glob(os.path.join(scene_render_dir, "view_*.jpg")))
                input_info['render_paths'] = view_files[:len(all_images)]
        
        # Safety limit: Qwen2-VL has strict limits on image count
        total_images = len(all_images)
        if total_images > 8:
            print(f"\n⚠️  Warning: Processing {total_images} images. This may cause OOM.")
            print(f"  Limiting to 8 images to prevent OOM.")
            all_images = all_images[:8]
            if input_type == 'video':
                input_info['frames_used'] = len(all_images)
            else:
                input_info['render_views_used'] = len(all_images)
        
        try:
            # ===== PASS 1: Situation + Images → Reasoning about Position/Direction =====
            if prompt_template and '{situation}' in prompt_template:
                # Use custom prompt template for pass 1
                pass1_prompt = prompt_template.format(situation=situation, question="")
            else:
                # Default pass 1 prompt: focus on understanding position/direction from situation
                if input_type == 'video':
                    pass1_prompt = f"""You are analyzing a 3D indoor scene. Look at the video frames and the situation description below.

SITUATION DESCRIPTION:
{situation}

VIDEO FRAMES:
The video frames above show the scene from the situation description.

TASK: Based on the situation description and the video frames, think carefully about:
1. The spatial layout and structure of the scene
2. The positions and locations of objects mentioned in the situation
3. The directions and orientations (e.g., which way is north, where is the camera facing)
4. The relationships between different objects and areas

Provide your reasoning about the position, direction, and spatial understanding of this scene:"""
                else:  # render
                    pass1_prompt = f"""You are analyzing a 3D indoor scene. Look at the 3D rendered views and the situation description below.

SITUATION DESCRIPTION:
{situation}

3D RENDERED VIEWS:
The images above show {len(all_images)} 3D rendered views of the scene from different camera angles.

TASK: Based on the situation description and the 3D rendered views, think carefully about:
1. The spatial layout and structure of the scene
2. The positions and locations of objects mentioned in the situation
3. The directions and orientations (e.g., which way is north, where is each camera facing)
4. The relationships between different objects and areas

Provide your reasoning about the position, direction, and spatial understanding of this scene:"""
            
            # Generate reasoning from pass 1
            if thinking_mode:
                p1_th, p1_after, p1_full = answer_from_images_with_thinking(
                    model,
                    all_images,
                    pass1_prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    think_end_token_id=think_end_token_id,
                )
                cot_reasoning = (p1_full or "").strip()
            else:
                p1_th = p1_after = ""
                p1_full = model.answer_question_from_images(
                    images=all_images,
                    question=pass1_prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                cot_reasoning = p1_full.strip()
                p1_full = cot_reasoning
            
            # Clear GPU cache between passes
            if clear_cache and torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            # ===== PASS 2: Reasoning + Question + Images → Final Answer =====
            if prompt_template and '{question}' in prompt_template:
                # Use custom prompt template for pass 2
                base_prompt = prompt_template.format(question=question, situation=situation)
                if input_type == 'video':
                    pass2_prompt = f"""SITUATION DESCRIPTION:
{situation}

REASONING FROM FIRST ANALYSIS:
{cot_reasoning}

VIDEO FRAMES:
The video frames above show the scene again.

{base_prompt}"""
                else:  # render
                    pass2_prompt = f"""SITUATION DESCRIPTION:
{situation}

REASONING FROM FIRST ANALYSIS:
{cot_reasoning}

3D RENDERED VIEWS:
The images above show the 3D rendered views again.

{base_prompt}"""
            else:
                # Default pass 2 prompt: use reasoning to answer the question
                if input_type == 'video':
                    pass2_prompt = f"""You are answering a question about a 3D indoor scene. You have already analyzed the scene in a first pass.

SITUATION DESCRIPTION:
{situation}

REASONING FROM FIRST ANALYSIS:
{cot_reasoning}

VIDEO FRAMES:
The video frames above show the scene again for reference.

QUESTION: {question}

Based on your previous reasoning about the spatial layout, positions, and directions, provide a direct and concise answer to the question:"""
                else:  # render
                    pass2_prompt = f"""You are answering a question about a 3D indoor scene. You have already analyzed the scene in a first pass.

SITUATION DESCRIPTION:
{situation}

REASONING FROM FIRST ANALYSIS:
{cot_reasoning}

3D RENDERED VIEWS:
The images above show the 3D rendered views again for reference.

QUESTION: {question}

Based on your previous reasoning about the spatial layout, positions, and directions, provide a direct and concise answer to the question:"""
            
            # Generate final answer from pass 2
            if thinking_mode:
                p2_th, p2_after, p2_full = answer_from_images_with_thinking(
                    model,
                    all_images,
                    pass2_prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    think_end_token_id=think_end_token_id,
                )
                answer = (p2_after or p2_full or "").strip()
                p2_full = (p2_full or "").strip()
            else:
                p2_th = p2_after = ""
                answer = model.answer_question_from_images(
                    images=all_images,
                    question=pass2_prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                ).strip()
                p2_full = answer
            # Remove common prefixes
            for prefix in ["Answer:", "The answer is:", "Answer is:", "ANSWER:"]:
                if answer.lower().startswith(prefix.lower()):
                    answer = answer[len(prefix):].strip()
            
            # Check exact match with ground truth
            is_correct_exact = exact_match(answer, gt_answer) if answer else False
            
            if is_correct_exact:
                correct += 1
            
            total += 1
            
            # Initialize result dict
            result = {
                **sample,
                'predicted_answer': answer,
                'cot_reasoning': cot_reasoning,
                'thinking_mode': thinking_mode,
                'cot_pass1_thinking': p1_th or '',
                'cot_pass1_after_think': p1_after or '',
                'cot_pass1_full': p1_full or '',
                'cot_pass2_thinking': p2_th or '',
                'cot_pass2_after_think': p2_after or '',
                'cot_pass2_full': p2_full or '',
                'input_type': input_type,
                **input_info,
                'correct': bool(is_correct_exact)
            }
            
            # Add NLP-based evaluations
            if nlp_evaluator and answer:
                nlp_results = nlp_evaluator.evaluate(answer, gt_answer)
                result.update(nlp_results)
                
                # Update metrics counters
                metrics['exact_match']['correct'] += int(is_correct_exact)
                metrics['exact_match']['total'] += 1
                
                if 'correct_semantic' in nlp_results:
                    metrics['semantic']['correct'] += int(bool(nlp_results['correct_semantic']))
                    metrics['semantic']['total'] += 1
                
                if 'correct_bleu' in nlp_results:
                    metrics['bleu']['correct'] += int(bool(nlp_results['correct_bleu']))
                    metrics['bleu']['total'] += 1
                
                if 'correct_rouge' in nlp_results:
                    metrics['rouge']['correct'] += int(bool(nlp_results['correct_rouge']))
                    metrics['rouge']['total'] += 1
                
                if 'correct_fuzzy' in nlp_results:
                    metrics['fuzzy']['correct'] += int(bool(nlp_results['correct_fuzzy']))
                    metrics['fuzzy']['total'] += 1
                
                if 'correct_contains' in nlp_results:
                    metrics['contains']['correct'] += int(bool(nlp_results['correct_contains']))
                    metrics['contains']['total'] += 1
            else:
                metrics['exact_match']['correct'] += int(is_correct_exact)
                metrics['exact_match']['total'] += 1
            
            results.append(result)

            if output_file and checkpoint_interval and idx % checkpoint_interval == 0:
                save_cot_checkpoint(
                    output_file=output_file,
                    results=results,
                    metrics=metrics,
                    correct=correct,
                    total=total,
                    samples_total=samples_total,
                    is_final=False,
                    evaluation_settings=evaluation_settings,
                    input_type=input_type,
                    render_dir=render_dir,
                    scenes_filter=scenes_filter,
                    max_render_views=max_render_views,
                    max_frames=max_frames,
                    model_name=getattr(model, "model_name", ""),
                )
            
            # Clear GPU memory after each sample
            if clear_cache and torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
        except torch.cuda.OutOfMemoryError as e:
            # CUDA OOM - clear cache and raise to stop execution
            print(f"\n❌ CUDA Out of Memory Error processing {scene_id}")
            print(f"Error: {str(e)}")
            
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                try:
                    total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                    print(f"GPU memory cleared. Free memory: {total_memory:.2f} GB")
                except Exception:
                    print("GPU memory cleared.")
            
            print("\n⚠️  Stopping evaluation due to CUDA OOM error.")
            print("Suggestions:")
            print("  - Use --load-in-4bit flag for quantization")
            print("  - Use smaller model (--model-name Qwen/Qwen2-VL-2B-Instruct)")
            if input_type == 'video':
                print("  - Reduce --max-frames (e.g., --max-frames 4)")
            else:
                print("  - Reduce --max-render-views (e.g., --max-render-views 3)")
            print("  - Process fewer samples at once (--max-samples)")
            
            if output_file:
                print(f"\n💾 Saving partial results to {output_file}...")
                save_cot_checkpoint(
                    output_file=output_file,
                    results=results,
                    metrics=metrics,
                    correct=correct,
                    total=total,
                    samples_total=samples_total,
                    is_final=False,
                    evaluation_settings=evaluation_settings,
                    input_type=input_type,
                    render_dir=render_dir,
                    scenes_filter=scenes_filter,
                    max_render_views=max_render_views,
                    max_frames=max_frames,
                    model_name=getattr(model, "model_name", ""),
                )
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
                f"CUDA Out of Memory. Evaluation stopped at sample {len(results) + 1}/{samples_total}"
            ) from e
            
        except Exception as e:
            error_msg = str(e)
            print(f"Error processing {scene_id}: {error_msg}")
            
            # Check if it's a memory-related error
            if "out of memory" in error_msg.lower() or "cuda" in error_msg.lower():
                print("\n⚠️  Memory-related error detected. Clearing GPU cache...")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                print("Consider using --load-in-8bit (recommended for Qwen2.5-VL), or reducing image count")
            
            # Check for 'CB' attribute errors (vision encoder quantization issues)
            if "'parameter' object has no attribute 'cb'" in error_msg.lower() or "has no attribute 'CB'" in error_msg.lower():
                print("\n❌ Qwen2.5-VL vision encoder quantization error detected!")
                print("The vision encoder (Conv3d layers) is incompatible with bitsandbytes quantization.")
                print("This error occurs even with 8-bit quantization because the vision encoder cannot be quantized.")
                print("\n💡 SOLUTION: Restart evaluation with --no-quantization flag")
                print("   Example: python evaluate_cot_qwen25.py --split test --input-type video --model-name Qwen/Qwen2.5-VL-3B-Instruct --no-quantization")
                print("\n   Note: This requires more GPU memory (~12-16 GB for 3B model)")
                print("   See QWEN25_FIXES.md for more details.")
            
            result = {
                **sample,
                'predicted_answer': f'ERROR: {error_msg}',
                'cot_reasoning': '',
                'thinking_mode': thinking_mode,
                'cot_pass1_thinking': '',
                'cot_pass1_after_think': '',
                'cot_pass1_full': '',
                'cot_pass2_thinking': '',
                'cot_pass2_after_think': '',
                'cot_pass2_full': '',
                'input_type': input_type,
                'correct': False
            }
            if nlp_evaluator:
                result.update({
                    'correct_semantic': False,
                    'correct_bleu': False,
                    'correct_rouge': False,
                    'correct_fuzzy': False,
                    'correct_contains': False,
                    'semantic_similarity': 0.0,
                    'bleu_score': 0.0,
                    'rouge1': 0.0,
                    'rougeL': 0.0,
                    'fuzzy_similarity': 0.0,
                    'contains_score': 0.0
                })
            results.append(result)
            total += 1
            
            # Clear GPU memory after error
            if clear_cache and torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
    
    # Calculate accuracies for all methods
    accuracies = {}
    for method, counts in metrics.items():
        if counts['total'] > 0:
            accuracies[method] = float(counts['correct'] / counts['total'])
        else:
            accuracies[method] = 0.0
    
    # Main accuracy (exact match)
    accuracy = float(accuracies.get('exact_match', correct / total if total > 0 else 0.0))
    
    if output_file:
        save_cot_checkpoint(
            output_file=output_file,
            results=results,
            metrics=metrics,
            correct=correct,
            total=total,
            samples_total=samples_total,
            is_final=True,
            evaluation_settings=evaluation_settings,
            input_type=input_type,
            render_dir=render_dir,
            scenes_filter=scenes_filter,
            max_render_views=max_render_views,
            max_frames=max_frames,
            model_name=getattr(model, "model_name", ""),
        )
        print(f"\n✅ Final results saved to: {output_file}")
    
    return {
        'accuracy': accuracy,
        'accuracies': accuracies,
        'correct': correct,
        'total': total,
        'metrics': metrics,
        'results': results
    }


def load_prompt_config(config_path: str, prompt_name: Optional[str] = None) -> Optional[str]:
    """
    Load prompt template from YAML config file
    
    Args:
        config_path: Path to YAML config file
        prompt_name: Name of prompt to use (defaults to default_prompt_name in config)
        
    Returns:
        Prompt template string with {question} and {situation} placeholders, or None if not found
    """
    if not os.path.exists(config_path):
        print(f"Warning: Prompt config file not found: {config_path}")
        return None
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Determine which prompt to use
        if prompt_name:
            # Use specified prompt name
            if prompt_name in config.get('prompts', {}):
                return config['prompts'][prompt_name]['template']
            else:
                print(f"Warning: Prompt '{prompt_name}' not found in config. Available prompts: {list(config.get('prompts', {}).keys())}")
                # Fall back to default
                return config.get('default_prompt')
        else:
            # Use default prompt name from config
            default_name = config.get('default_prompt_name', 'scene_understanding')
            if default_name in config.get('prompts', {}):
                return config['prompts'][default_name]['template']
            else:
                # Fall back to top-level default_prompt
                return config.get('default_prompt')
    except Exception as e:
        print(f"Error loading prompt config: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Chain of Thought (CoT) Evaluation on SQA dataset (Qwen2.5-VL optimized)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Qwen2.5-VL-3B with video input (8-bit quantization, recommended)
  python evaluate_cot_qwen25.py --split test --input-type video --model-name Qwen/Qwen2.5-VL-3B-Instruct
  
  # Qwen2.5-VL-7B with render input
  python evaluate_cot_qwen25.py --split test --input-type render --model-name Qwen/Qwen2.5-VL-7B-Instruct
  
  # Without quantization (requires more GPU memory)
  python evaluate_cot_qwen25.py --split test --input-type video --model-name Qwen/Qwen2.5-VL-3B-Instruct --no-quantization

  # Checkpoint every 10 samples (same output path; auto-resume if file looks incomplete)
  python evaluate_cot_qwen25.py --split test --input-type video --checkpoint-interval 10 --output run_cot.json

Note: Qwen2.5-VL models have compatibility issues with 4-bit quantization.
      This script defaults to 8-bit quantization. Use --no-quantization for most reliable results.
        """
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['train', 'val', 'test'],
        help='Dataset split to evaluate on'
    )
    parser.add_argument(
        '--dataset-dir',
        type=str,
        default='dataset/SQA',
        help='Root directory of SQA dataset'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='Qwen/Qwen2.5-VL-3B-Instruct',
        help='Qwen2.5-VL model name or path (default: Qwen/Qwen2.5-VL-3B-Instruct)'
    )
    parser.add_argument(
        '--input-type',
        type=str,
        choices=['video', 'render'],
        required=True,
        help='Input type: "video" for video frames, "render" for 3D rendered images'
    )
    parser.add_argument(
        '--render-dir',
        type=str,
        default='dataset/SQA/render',
        help='Directory containing 3D render images (required if --input-type=render)'
    )
    parser.add_argument(
        '--scenes',
        type=str,
        nargs='+',
        default=None,
        help='List of scene IDs to evaluate (e.g., scene0000_00 scene1234_00). Optional - if not provided, evaluates all scenes.'
    )
    parser.add_argument(
        '--filter-scenes-json',
        type=str,
        default=None,
        help='JSON file containing scene IDs to filter (e.g., video_frame_statistics_frames_under_50.json). Only evaluates videos/scenes listed in this file. Optional - if not provided, evaluates all scenes.'
    )
    parser.add_argument(
        '--max-frames',
        type=int,
        default=4,
        help='Maximum number of video frames to use (default: 4, only for --input-type=video)'
    )
    parser.add_argument(
        '--max-render-views',
        type=int,
        default=5,
        help='Maximum number of render views to use (default: 5, only for --input-type=render)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output JSON file path (auto-generated if not specified)'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum number of samples to evaluate (for testing)'
    )
    parser.add_argument(
        '--no-nlp-eval',
        action='store_true',
        help='Disable NLP-based evaluation metrics'
    )
    parser.add_argument(
        '--no-clear-cache',
        action='store_true',
        help='Disable GPU cache clearing between samples'
    )
    parser.add_argument(
        '--load-in-4bit',
        action='store_true',
        help='Load model in 4-bit quantization (⚠️ NOT RECOMMENDED for Qwen2.5-VL - may cause AssertionError)'
    )
    parser.add_argument(
        '--load-in-8bit',
        action='store_true',
        help='Load model in 8-bit quantization (RECOMMENDED for Qwen2.5-VL, default if no quantization flags specified)'
    )
    parser.add_argument(
        '--no-quantization',
        action='store_true',
        help='Disable quantization entirely (requires more GPU memory, most reliable for Qwen2.5-VL)'
    )
    parser.add_argument(
        '--max-new-tokens',
        type=int,
        default=256,
        help='Maximum number of tokens per pass (default: 256; with --thinking bumped to 8192 if still 256)'
    )
    parser.add_argument(
        '--thinking',
        action='store_true',
        help='Qwen thinking split on both CoT passes (VideoLM.answer_question_from_images_with_thinking; use with Qwen3-VL-Thinking or Qwen2.5-VL thinking-style checkpoints)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=None,
        help='Sampling temperature (default: 1.0 with --thinking, else 0.3)'
    )
    parser.add_argument(
        '--top-p',
        type=float,
        default=None,
        dest='top_p',
        help='Nucleus top-p (default: 0.95 with --thinking, else 0.9)'
    )
    parser.add_argument(
        '--question-types',
        type=str,
        nargs='+',
        default=None,
        help='Filter samples by question types (e.g., --question-types what where how)'
    )
    parser.add_argument(
        '--prompt-name',
        type=str,
        default=None,
        help='Name of prompt template to use from prompt_config.yaml'
    )
    parser.add_argument(
        '--prompt-config',
        type=str,
        default='prompt_config.yaml',
        help='Path to YAML file containing prompt templates'
    )
    parser.add_argument(
        '--checkpoint-interval',
        type=int,
        default=0,
        help='Save checkpoint every N samples in the current run (default: 0 = disabled).',
    )
    parser.add_argument(
        '--resume-from',
        type=str,
        default=None,
        help='JSON to resume from. If omitted and --output exists and looks like a checkpoint, auto-resume.',
    )

    args = parser.parse_args()

    gen_temperature = args.temperature if args.temperature is not None else (1.0 if args.thinking else 0.3)
    gen_top_p = args.top_p if args.top_p is not None else (0.95 if args.thinking else 0.9)
    max_new_tokens = args.max_new_tokens
    if args.thinking and max_new_tokens == 256:
        max_new_tokens = 8192
        print("Thinking mode: using --max-new-tokens 8192 (override with an explicit --max-new-tokens).")
    
    # Validate model name
    if not check_qwen25_model(args.model_name):
        print(f"⚠️  Warning: Model '{args.model_name}' does not appear to be a Qwen2.5-VL model.")
        print("   This script is optimized for Qwen2.5-VL models.")
        print("   For Qwen2-VL models, use evaluate_sqa_cot_2pass_qwen2_vl.py instead.")
        print("   For Qwen3-VL models, use evaluate_cot_qwen3.py instead.")
        response = input("   Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Exiting.")
            return
    
    # Handle quantization flags
    # Default: use 8-bit quantization (recommended for Qwen2.5-VL)
    # NOTE: Even 8-bit quantization may fail with 'CB' attribute error for vision encoder
    # If you encounter this error, use --no-quantization instead
    load_in_4bit = False
    load_in_8bit = True  # Default to 8-bit for Qwen2.5-VL
    
    if args.no_quantization:
        # User explicitly disabled quantization
        load_in_4bit = False
        load_in_8bit = False
        print("⚠️  Running without quantization (requires more GPU memory)")
        print("   This is the most reliable option for Qwen2.5-VL if quantization fails")
    elif args.load_in_4bit:
        # User explicitly requested 4-bit (not recommended for Qwen2.5-VL)
        load_in_4bit = True
        load_in_8bit = False
        print("⚠️  WARNING: Using 4-bit quantization with Qwen2.5-VL may cause AssertionError!")
        print("   See QWEN25_FIXES.md for details.")
        print("   Recommended: Use --load-in-8bit instead (or omit flag, 8-bit is default)")
        response = input("   Continue with 4-bit anyway? (y/n): ")
        if response.lower() != 'y':
            print("Switching to 8-bit quantization (default for Qwen2.5-VL)...")
            load_in_4bit = False
            load_in_8bit = True
    elif args.load_in_8bit:
        # User explicitly requested 8-bit (good choice, but may still have issues)
        load_in_4bit = False
        load_in_8bit = True
        print("✓ Using 8-bit quantization")
        print("   ⚠️  Note: If you encounter 'CB' attribute error, use --no-quantization instead")
    # else: already set to default (8-bit) above
    
    # Set up paths
    dataset_dir = Path(args.dataset_dir)
    questions_path = dataset_dir / 'sqa_task' / 'balanced' / f'v1_balanced_questions_{args.split}_scannetv2.json'
    annotations_path = dataset_dir / 'sqa_task' / 'balanced' / f'v1_balanced_sqa_annotations_{args.split}_scannetv2.json'
    video_dir = dataset_dir / 'video'
    
    # Check if files exist
    if not questions_path.exists():
        raise FileNotFoundError(f"Questions file not found: {questions_path}")
    if not annotations_path.exists():
        raise FileNotFoundError(f"Annotations file not found: {annotations_path}")
    if args.input_type == 'video' and not video_dir.exists():
        raise FileNotFoundError(f"Video directory not found: {video_dir}")
    if args.input_type == 'render':
        if not args.render_dir:
            raise ValueError("--render-dir must be specified when --input-type=render")
        if not os.path.exists(args.render_dir):
            raise FileNotFoundError(f"Render directory not found: {args.render_dir}")
    
    # Load dataset
    print(f"\nLoading {args.split} split...")
    samples = load_dataset(
        str(questions_path),
        str(annotations_path),
        str(video_dir)
    )
    print(f"Loaded {len(samples)} samples")
    
    # Determine scene IDs to filter by
    scene_ids_to_filter = None
    
    # Load scene IDs from JSON file if specified
    if args.filter_scenes_json:
        if not os.path.exists(args.filter_scenes_json):
            raise FileNotFoundError(f"Filter scenes JSON file not found: {args.filter_scenes_json}")
        
        print(f"\nLoading scene filter from: {args.filter_scenes_json}")
        with open(args.filter_scenes_json, 'r') as f:
            filter_data = json.load(f)
        
        # Extract scene IDs from the JSON
        # Handle both formats: direct list or videos array
        if 'videos' in filter_data:
            # Format from analyze_video_frames.py
            scene_ids_from_json = [v['scene_id'] for v in filter_data['videos']]
            threshold_info = f" (threshold: {filter_data.get('threshold', 'N/A')}, filter_type: {filter_data.get('filter_type', 'N/A')})"
        elif isinstance(filter_data, list):
            # Direct list of scene IDs
            scene_ids_from_json = filter_data
            threshold_info = ""
        elif 'scene_ids' in filter_data:
            # Alternative format
            scene_ids_from_json = filter_data['scene_ids']
            threshold_info = ""
        else:
            raise ValueError(f"Could not parse scene IDs from JSON file. Expected 'videos' array or 'scene_ids' list.")
        
        print(f"Found {len(scene_ids_from_json)} scene IDs in filter file{threshold_info}")
        scene_ids_to_filter = set(scene_ids_from_json)
        
        # Merge with --scenes if provided
        if args.scenes:
            scene_ids_to_filter = scene_ids_to_filter.union(set(args.scenes))
            print(f"Merged with --scenes argument: {len(scene_ids_to_filter)} total scene IDs")
    elif args.scenes:
        # Use only --scenes argument
        scene_ids_to_filter = set(args.scenes)
    else:
        # No scene filtering specified - evaluate all samples
        scene_ids_to_filter = None
        print("\nNo scene filter specified - evaluating all samples in the dataset")
    
    # Filter samples by scene IDs (if filter is provided)
    if scene_ids_to_filter:
        print(f"\nFiltering by {len(scene_ids_to_filter)} scene IDs...")
        original_count = len(samples)
        samples = [s for s in samples if s['scene_id'] in scene_ids_to_filter]
        print(f"Filtered from {original_count} to {len(samples)} samples")
        
        if len(samples) == 0:
            print("⚠️  Warning: No samples found matching the scene IDs!")
            print("   Make sure the scene IDs match the scene IDs in the dataset.")
            return
    
    # Filter by question type if specified
    if args.question_types:
        print(f"\nFiltering by question types: {args.question_types}")
        original_count = len(samples)
        samples = filter_samples_by_question_type(samples, args.question_types, match_any=True)
        print(f"Filtered from {original_count} to {len(samples)} samples")
        
        # Show statistics
        stats = get_question_type_statistics(samples)
        print("\nQuestion type distribution in filtered samples:")
        for qtype, count in sorted(stats.items(), key=lambda x: x[1], reverse=True):
            print(f"  {qtype:15s}: {count:5d} samples")
    
    # Check GPU/CUDA availability
    if torch.cuda.is_available():
        print(f"\n✓ CUDA is available")
        print(f"GPU Memory Status:")
        try:
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                allocated = torch.cuda.memory_allocated(i) / 1e9
                reserved = torch.cuda.memory_reserved(i) / 1e9
                total = props.total_memory / 1e9
                free = total - reserved
                
                print(f"  GPU {i} ({props.name}):")
                print(f"    Total: {total:.2f} GB | Allocated: {allocated:.2f} GB | Free: {free:.2f} GB")
            
            # Clear any existing cache
            if not args.no_clear_cache:
                print("\nClearing GPU cache before starting...")
                torch.cuda.empty_cache()
                gc.collect()
        except Exception as e:
            print(f"⚠️  Warning: Could not access GPU information: {e}")
            print("   Continuing with CPU mode...")
    else:
        print(f"\n⚠️  CUDA is not available - will run on CPU")
        print("   This will be much slower. Consider:")
        print("   - Installing NVIDIA drivers")
        print("   - Using a machine with GPU")
        print("   - Using cloud GPU services")
    
    # Initialize model
    print(f"\nInitializing model: {args.model_name}")
    print(f"  Quantization: {'4-bit' if load_in_4bit else '8-bit' if load_in_8bit else 'None (full precision)'}")
    try:
        model = VideoLM(
            model_name=args.model_name,
            max_frames=args.max_frames if args.input_type == 'video' else 8,  # Only used for video
            frame_size=(448, 448),
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit
        )
        print(f"✓ Model loaded successfully on device: {model.device}")
    except Exception as e:
        print(f"\n❌ Failed to load model: {e}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        raise
    
    # Load prompt template if specified
    prompt_template = None
    if args.prompt_name:
        print(f"\nLoading prompt template: {args.prompt_name}")
        prompt_template = load_prompt_config(args.prompt_config, args.prompt_name)
        if prompt_template:
            print(f"✓ Prompt template loaded")
        else:
            print(f"⚠ Warning: Could not load prompt template, using default")
    
    # Evaluate with Chain of Thought
    print(f"\n{'='*60}")
    print("Chain of Thought (CoT) Evaluation")
    print(f"{'='*60}")
    print(f"Model: {args.model_name}")
    print(f"Input Type: {args.input_type.upper()}")
    if args.input_type == 'video':
        print(f"Max video frames: {args.max_frames}")
        if args.max_frames > 8:
            print(f"⚠️  Warning: More than 8 frames may cause OOM!")
    else:
        print(f"Render Directory: {args.render_dir}")
        print(f"Max render views per scene: {args.max_render_views}")
        if args.max_render_views > 8:
            print(f"⚠️  Warning: More than 8 render views may cause OOM!")
    if scene_ids_to_filter:
        scenes_list = sorted(list(scene_ids_to_filter))
        print(f"Scenes to evaluate: {len(scenes_list)} scenes")
        if len(scenes_list) <= 10:
            print(f"  {', '.join(scenes_list)}")
        else:
            print(f"  {', '.join(scenes_list[:10])} ... and {len(scenes_list) - 10} more")
    else:
        print(f"Evaluating all scenes in the dataset")
    print(f"{'='*60}\n")
    
    # Generate output filename if not specified
    if not args.output:
        if args.resume_from and os.path.exists(args.resume_from):
            args.output = args.resume_from
        else:
            model_suffix = args.model_name.split('/')[-1].replace('-', '_').lower()
            if args.filter_scenes_json:
                filter_name = Path(args.filter_scenes_json).stem
                scenes_suffix = filter_name
            elif args.scenes:
                scenes_suffix = '_'.join(args.scenes)
                if len(scenes_suffix) > 100:
                    scenes_suffix = scenes_suffix[:100] + '...'
            else:
                scenes_suffix = 'all_scenes'
            think_suffix = '_thinking' if args.thinking else ''
            args.output = f'sqa_{args.split}_results_cot{think_suffix}_{args.input_type}_{model_suffix}_{scenes_suffix}.json'

    output_file = args.output
    resume_from = None
    if args.resume_from:
        if not os.path.exists(args.resume_from):
            print(f"⚠️  Warning: Resume file not found: {args.resume_from}")
            resume_from = None
        else:
            resume_from = args.resume_from
    else:
        if os.path.exists(output_file):
            try:
                with open(output_file, "r") as f:
                    checkpoint_data = json.load(f)
                if checkpoint_json_incomplete(checkpoint_data):
                    print(f"\n✓ Found existing checkpoint file: {output_file}")
                    print(f"   Total samples processed: {checkpoint_data.get('total', 0)}")
                    print(f"   Auto-resuming from checkpoint...")
                    resume_from = output_file
                else:
                    print(f"\n⚠️  Output file exists: {output_file}")
                    print("   It appears to be a complete results file (not a checkpoint).")
                    print("   Starting fresh evaluation (will overwrite existing file).")
                    print(f"   To resume explicitly, use: --resume-from {output_file}")
                    resume_from = None
            except (json.JSONDecodeError, KeyError):
                print(f"\n⚠️  Output file exists but could not be read as checkpoint: {output_file}")
                print("   Starting fresh evaluation (will overwrite existing file).")
                resume_from = None
        else:
            resume_from = None

    evaluation_settings = build_evaluation_settings(
        script=os.path.basename(__file__),
        model_name=args.model_name,
        temperature=float(gen_temperature),
        top_p=float(gen_top_p),
        max_new_tokens=max_new_tokens,
        split=args.split,
        dataset_dir=args.dataset_dir,
        input_type=args.input_type,
        chain_of_thought=True,
        thinking_mode=args.thinking,
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
        no_quantization=args.no_quantization,
        prompt_config=args.prompt_config,
        prompt_name=args.prompt_name,
        max_frames=args.max_frames if args.input_type == 'video' else None,
        max_render_views=args.max_render_views if args.input_type == 'render' else None,
        render_dir=args.render_dir if args.input_type == 'render' else None,
        checkpoint_interval=args.checkpoint_interval if args.checkpoint_interval > 0 else None,
    )
    
    results = evaluate_with_cot(
        model=model,
        samples=samples,
        input_type=args.input_type,
        render_dir=args.render_dir if args.input_type == 'render' else None,
        output_file=output_file,
        max_samples=args.max_samples,
        use_nlp_eval=not args.no_nlp_eval,
        clear_cache=not args.no_clear_cache,
        max_new_tokens=max_new_tokens,
        temperature=gen_temperature,
        top_p=gen_top_p,
        thinking_mode=args.thinking,
        prompt_template=prompt_template,
        scenes_filter=list(scene_ids_to_filter) if scene_ids_to_filter else None,
        max_render_views=args.max_render_views if args.input_type == 'render' else None,
        max_frames=args.max_frames if args.input_type == 'video' else None,
        checkpoint_interval=args.checkpoint_interval if args.checkpoint_interval > 0 else None,
        resume_from=resume_from,
        evaluation_settings=evaluation_settings,
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("Chain of Thought (CoT) Evaluation Results")
    print("=" * 60)
    print(f"Exact Match Accuracy: {results['accuracy']:.4f} ({results['correct']}/{results['total']})")
    
    if 'accuracies' in results:
        print("\nNLP-based Accuracies:")
        for method, acc in results['accuracies'].items():
            if method != 'exact_match':
                method_name = method.replace('_', ' ').title()
                counts = results['metrics'][method]
                print(f"  {method_name}: {acc:.4f} ({counts['correct']}/{counts['total']})")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
