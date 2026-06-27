"""
Zero-shot evaluation script for SQA (Scene Question Answering) dataset
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from tqdm import tqdm
import argparse
import torch
import gc
import yaml

from videolm import VideoLM
from videolm.evaluators import AnswerEvaluator
from videolm.prompt_utils import vlm_input_text
from utils.question_type_filter import filter_samples_by_question_type, get_question_type_statistics
from utils.evaluation_result_meta import attach_result_json_metadata, build_evaluation_settings


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
        List of samples with question, answer, and video path
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


def load_prompt_config(config_path: str, prompt_name: Optional[str] = None) -> Optional[str]:
    """
    Load prompt template from YAML config file
    
    Args:
        config_path: Path to YAML config file
        prompt_name: Name of prompt to use (defaults to default_prompt_name in config)
        
    Returns:
        Prompt template string with {question} placeholder, or None if not found
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


def _write_eval_results_json(
    output_file: str,
    results: List[Dict],
    correct: int,
    total: int,
    metrics: Dict[str, Dict[str, int]],
    evaluation_settings: Optional[Dict],
    extra: Optional[Dict] = None,
) -> None:
    """Write the standard result JSON shape (accuracy, metrics, results rows)."""
    accuracies: Dict[str, float] = {}
    for method, counts in metrics.items():
        if counts["total"] > 0:
            accuracies[method] = float(counts["correct"] / counts["total"])
        else:
            accuracies[method] = 0.0
    accuracy = float(accuracies.get("exact_match", correct / total if total > 0 else 0.0))
    payload: Dict[str, Any] = {
        "accuracy": accuracy,
        "accuracies": accuracies,
        "correct": correct,
        "total": total,
        "metrics": metrics,
        "results": results,
    }
    if extra:
        payload.update(extra)
    attach_result_json_metadata(payload, evaluation_settings)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def evaluate(
    model: VideoLM,
    samples: List[Dict],
    output_file: str = None,
    max_samples: int = None,
    use_nlp_eval: bool = True,
    clear_cache: bool = True,
    prompt_template: Optional[str] = None,
    max_new_tokens: int = 64,
    temperature: float = 0.3,
    evaluation_settings: Optional[Dict] = None,
    save_interval: Optional[int] = None,
) -> Dict:
    """
    Evaluate model on SQA dataset
    
    Args:
        model: VideoLM model instance
        samples: List of samples to evaluate
        output_file: Optional path to save detailed results
        max_samples: Optional limit on number of samples to evaluate
        save_interval: If set with ``output_file``, rewrite ``output_file`` every N completed
            samples (checkpoint). Omit or use ``None`` / ``<= 0`` to save only at the end
            (and on CUDA OOM partial save).
        
    Returns:
        Dictionary with evaluation metrics
    """
    if max_samples:
        samples = samples[:max_samples]
    
    results = []
    correct = 0
    total = 0
    
    # Initialize NLP evaluator if requested
    nlp_evaluator = None
    if use_nlp_eval:
        print("Initializing NLP evaluators...")
        nlp_evaluator = AnswerEvaluator()
    
    # Track metrics for all methods
    metrics = {
        'exact_match': {'correct': 0, 'total': 0},
        'semantic': {'correct': 0, 'total': 0},
        'bleu': {'correct': 0, 'total': 0},
        'rouge': {'correct': 0, 'total': 0},
        'fuzzy': {'correct': 0, 'total': 0},
        'contains': {'correct': 0, 'total': 0}
    }

    save_every = int(save_interval) if save_interval and int(save_interval) > 0 else 0

    def _checkpoint_if_due() -> None:
        if not output_file or not save_every:
            return
        n = len(results)
        if n > 0 and n % save_every == 0:
            _write_eval_results_json(
                output_file,
                results,
                correct,
                total,
                metrics,
                evaluation_settings,
                extra={"checkpoint": True, "completed_samples": n},
            )
            print(f"Checkpoint: saved {n} samples -> {output_file}")

    print(f"\nEvaluating on {len(samples)} samples...")
    if save_every:
        print(f"Checkpoint interval: every {save_every} completed samples -> {output_file}")
    
    for sample in tqdm(samples, desc="Processing"):
        video_path = sample['video_path']
        question = sample['question']
        gt_answer = sample['gt_answer']
        vlm_prompt_text = vlm_input_text(
            question,
            prompt_template=prompt_template,
            instance_prompt_template=model.prompt_template,
            situation=sample.get("situation"),
        )

        # Check if video exists
        if not os.path.exists(video_path):
            print(f"Warning: Video not found: {video_path}")
            result = {
                **sample,
                'predicted_answer': 'VIDEO_NOT_FOUND',
                'correct': False,
                'vlm_prompt': vlm_prompt_text,
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
            _checkpoint_if_due()
            continue
        
        try:
            # Get model prediction
            # Use method-level prompt_template if provided, otherwise use instance-level
            # Use temperature 0.3 instead of 0.1 to avoid numerical stability issues
            # Temperature 0.1 can cause "probability tensor contains inf/nan" errors on some GPUs
            predicted_answer = model.answer_question(
                video_path=video_path,
                question=question,
                max_new_tokens=max_new_tokens,  # Use parameter for concise answers
                temperature=temperature,
                prompt_template=prompt_template,
                situation=sample.get("situation"),
            )
            
            # Check exact match
            is_correct_exact = exact_match(predicted_answer, gt_answer)
            
            if is_correct_exact:
                correct += 1
            
            total += 1
            
            # Initialize result dict
            result = {
                **sample,
                'predicted_answer': predicted_answer,
                'correct': bool(is_correct_exact),  # Keep exact match as 'correct' for backward compatibility
                'vlm_prompt': vlm_prompt_text,
            }
            
            # Add NLP-based evaluations
            if nlp_evaluator:
                nlp_results = nlp_evaluator.evaluate(predicted_answer, gt_answer)
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
            _checkpoint_if_due()

            # Clear GPU memory after each sample to prevent OOM
            if clear_cache and torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
        except torch.cuda.OutOfMemoryError as e:
            # CUDA OOM - clear cache and raise to stop execution
            print(f"\n❌ CUDA Out of Memory Error processing {video_path}")
            print(f"Error: {str(e)}")
            
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                print(f"GPU memory cleared. Free memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            
            print("\n⚠️  Stopping evaluation due to CUDA OOM error.")
            print("Suggestions:")
            print("  - Use --load-in-4bit or --load-in-8bit flag for quantization")
            print("  - Use smaller model (--model-name Qwen/Qwen2-VL-2B-Instruct)")
            print("  - Reduce --max-frames (e.g., --max-frames 4)")
            print("  - Process fewer samples at once (--max-samples)")
            
            # Save partial results before exiting
            if output_file:
                print(f"\n💾 Saving partial results to {output_file}...")
                _write_eval_results_json(
                    output_file,
                    results,
                    correct,
                    total,
                    metrics,
                    evaluation_settings,
                    extra={
                        "error": "CUDA_OUT_OF_MEMORY",
                        "error_message": str(e),
                        "stopped_at_sample": len(results),
                    },
                )
            
            raise RuntimeError(f"CUDA Out of Memory. Evaluation stopped at sample {len(results) + 1}/{len(samples)}") from e
            
        except Exception as e:
            import traceback
            error_msg = str(e) if str(e) else repr(e)
            error_type = type(e).__name__
            full_traceback = traceback.format_exc()
            
            # Print detailed error for debugging
            print(f"Error processing {video_path}:")
            print(f"  Type: {error_type}")
            print(f"  Message: {error_msg}")
            if not error_msg or error_msg.strip() == "":
                print(f"  Full traceback:\n{full_traceback}")
            
            # Check if it's a memory-related error
            if "out of memory" in error_msg.lower() or "cuda" in error_msg.lower():
                print("\n⚠️  Memory-related error detected. Clearing GPU cache...")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                print("Consider using --load-in-4bit or --load-in-8bit, or reducing --max-frames")
            
            # Store more detailed error info
            error_info = f"{error_type}: {error_msg}" if error_msg else error_type
            result = {
                **sample,
                'predicted_answer': f'ERROR: {error_info}',
                'error_type': error_type,
                'error_message': error_msg,
                'correct': False,
                'vlm_prompt': vlm_prompt_text,
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
            _checkpoint_if_due()

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
    
    # Save detailed results if requested
    if output_file:
        _write_eval_results_json(
            output_file,
            results,
            correct,
            total,
            metrics,
            evaluation_settings,
        )
        print(f"\nDetailed results saved to: {output_file}")
    
    return {
        'accuracy': accuracy,
        'accuracies': accuracies,
        'correct': correct,
        'total': total,
        'metrics': metrics,
        'results': results
    }


def main():
    parser = argparse.ArgumentParser(description='Zero-shot evaluation on SQA dataset')
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['train', 'val', 'test'],
        help='Dataset split to evaluate on'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='Qwen/Qwen2-VL-2B-Instruct', # Qwen/Qwen2-VL-2B-Instruct, Qwen/Qwen2-VL-7B-Instruct, Qwen/Qwen2-VL-72B-Instruct
        help='Model name or path'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum number of samples to evaluate (for testing)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file to save detailed results'
    )
    parser.add_argument(
        '--dataset-dir',
        type=str,
        default='dataset/SQA',
        help='Directory containing SQA dataset'
    )
    parser.add_argument(
        '--max-frames',
        type=int,
        default=8,
        help='Maximum number of frames to extract from video'
    )
    parser.add_argument(
        '--load-in-4bit',
        action='store_true',
        help='Load model in 4-bit quantization'
    )
    parser.add_argument(
        '--load-in-8bit',
        action='store_true',
        help='Load model in 8-bit quantization (recommended for Qwen2.5-VL)'
    )
    parser.add_argument(
        '--no-nlp-eval',
        action='store_true',
        help='Disable NLP-based evaluation methods'
    )
    parser.add_argument(
        '--no-clear-cache',
        action='store_true',
        help='Disable automatic GPU cache clearing between samples'
    )
    parser.add_argument(
        '--prompt-config',
        type=str,
        default='prompt_config.yaml',
        help='Path to YAML file containing prompt templates'
    )
    parser.add_argument(
        '--prompt-name',
        type=str,
        default=None,
        help='Name of prompt template to use from config (defaults to default_prompt_name in config)'
    )
    parser.add_argument(
        '--max-new-tokens',
        type=int,
        default=64,
        help='Maximum number of tokens to generate (default: 64 for concise answers)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.3,
        help='Sampling temperature (default: 0.3; lower values are more deterministic)',
    )
    parser.add_argument(
        '--question-types',
        type=str,
        nargs='+',
        default=None,
        help='Filter samples by question types (e.g., --question-types what where how). If not specified, evaluates all types.'
    )
    parser.add_argument(
        '--filter-scenes-json',
        type=str,
        default=None,
        help='JSON file containing scene IDs to filter (e.g., video_frame_statistics_frames_under_50.json). Only evaluates videos/scenes listed in this file.'
    )
    parser.add_argument(
        '--save-interval',
        type=int,
        default=0,
        metavar='N',
        help='If N > 0 and --output is set, rewrite the output JSON every N completed samples (checkpoint). 0 = only save at the end (and on CUDA OOM).',
    )
    
    args = parser.parse_args()
    
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
    if not video_dir.exists():
        raise FileNotFoundError(f"Video directory not found: {video_dir}")
    
    # Load prompt template from config
    prompt_template = None
    if args.prompt_config:
        print(f"\nLoading prompt template from {args.prompt_config}...")
        prompt_template = load_prompt_config(args.prompt_config, args.prompt_name)
        if prompt_template:
            print("✓ Prompt template loaded successfully")
            # Show a preview (first 100 chars)
            preview = prompt_template[:100].replace('\n', ' ')
            print(f"  Preview: {preview}...")
        else:
            print("⚠ No prompt template loaded, using default question format")
    
    # Load dataset
    print(f"\nLoading {args.split} split...")
    samples = load_dataset(
        str(questions_path),
        str(annotations_path),
        str(video_dir)
    )
    print(f"Loaded {len(samples)} samples")
    
    # Filter by scenes from JSON file if specified
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
            scene_ids = [v['scene_id'] for v in filter_data['videos']]
            threshold_info = f" (threshold: {filter_data.get('threshold', 'N/A')}, filter_type: {filter_data.get('filter_type', 'N/A')})"
        elif isinstance(filter_data, list):
            # Direct list of scene IDs
            scene_ids = filter_data
            threshold_info = ""
        elif 'scene_ids' in filter_data:
            # Alternative format
            scene_ids = filter_data['scene_ids']
            threshold_info = ""
        else:
            raise ValueError(f"Could not parse scene IDs from JSON file. Expected 'videos' array or 'scene_ids' list.")
        
        print(f"Found {len(scene_ids)} scene IDs in filter file{threshold_info}")
        original_count = len(samples)
        samples = [s for s in samples if s['scene_id'] in scene_ids]
        print(f"Filtered from {original_count} to {len(samples)} samples")
        
        if len(samples) == 0:
            print("⚠️  Warning: No samples found matching the scene IDs in the filter file!")
            print("   Make sure the scene IDs in the JSON match the scene IDs in the dataset.")
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
    
    # Check GPU memory before starting
    if torch.cuda.is_available():
        print(f"\nGPU Memory Status:")
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
    
    # Initialize model
    print(f"\nInitializing model: {args.model_name}")
    
    # Check quantization flags
    if args.load_in_4bit and args.load_in_8bit:
        raise ValueError("Cannot use both --load-in-4bit and --load-in-8bit. Choose one.")
    
    model = VideoLM(
        model_name=args.model_name,
        max_frames=args.max_frames,
        frame_size=(448, 448),
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        prompt_template=prompt_template  # Set instance-level prompt template
    )
    
    # Generate output filename if not specified
    if not args.output:
        model_suffix = args.model_name.split('/')[-1].replace('-', '_').lower()
        if args.filter_scenes_json:
            # Include filter info in filename
            filter_name = Path(args.filter_scenes_json).stem
            output_file = f'sqa_{args.split}_results_{model_suffix}_{filter_name}.json'
        else:
            output_file = f'sqa_{args.split}_results_{model_suffix}.json'
    else:
        output_file = args.output

    save_interval = args.save_interval if args.save_interval > 0 else None
    evaluation_settings = build_evaluation_settings(
        script=os.path.basename(__file__),
        model_name=args.model_name,
        temperature=float(args.temperature),
        max_new_tokens=args.max_new_tokens,
        max_frames=args.max_frames,
        split=args.split,
        dataset_dir=args.dataset_dir,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        prompt_config=args.prompt_config,
        prompt_name=args.prompt_name,
        chain_of_thought=False,
        save_interval=save_interval,
    )
    
    # Evaluate
    results = evaluate(
        model=model,
        samples=samples,
        output_file=output_file,
        max_samples=args.max_samples,
        use_nlp_eval=not args.no_nlp_eval,
        clear_cache=not args.no_clear_cache,
        prompt_template=prompt_template,  # Pass as method parameter (can override instance-level)
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        evaluation_settings=evaluation_settings,
        save_interval=save_interval,
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("Evaluation Results")
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

