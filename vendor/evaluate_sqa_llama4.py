"""
Zero-shot evaluation script for SQA (Scene Question Answering) dataset
Specialized for Llama4-VL (Llama 3.2/3.3 Vision or Llama 4 Vision) models with optimized defaults
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import argparse
import torch
import gc
import yaml

from videolm import VideoLM
from videolm.evaluators import AnswerEvaluator
from utils.question_type_filter import filter_samples_by_question_type, get_question_type_statistics


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


def check_llama4_model(model_name: str) -> bool:
    """Check if model name is a Llama4-VL or Llama Vision model"""
    model_name_lower = model_name.lower()
    # Check for Llama 3.2/3.3/4 Vision models
    return (("llama" in model_name_lower and ("vision" in model_name_lower or "vl" in model_name_lower)) or
            ("llama-3.2" in model_name_lower and "vision" in model_name_lower) or
            ("llama-3.3" in model_name_lower and "vision" in model_name_lower) or
            ("llama-4" in model_name_lower and "vision" in model_name_lower) or
            ("meta-llama" in model_name_lower and ("vision" in model_name_lower or "vl" in model_name_lower)))


# Meta does not publish Llama-3.3-Vision-11B* on HF (404). 11B multimodal is Llama 3.2.
_LLAMA_11B_VISION_INSTRUCT = "meta-llama/Llama-3.2-11B-Vision-Instruct"


def resolve_llama_vision_model_id(model_name: str) -> str:
    """
    Map mistaken repo ids to a valid checkpoint.

    ``meta-llama/Llama-3.3-Vision-11B-V1.5`` and similar strings are not on Hugging Face.
    For ~11B instruction-tuned vision, use Llama 3.2 11B Vision Instruct.
    For Llama 3.3 multimodal at scale, use ``meta-llama/Llama-3.3-70B-Vision-Instruct`` (gated).
    """
    key = model_name.strip()
    aliases = {
        "meta-llama/Llama-3.3-Vision-11B-V1.5": _LLAMA_11B_VISION_INSTRUCT,
        "meta-llama/Llama-3.3-Vision-11B": _LLAMA_11B_VISION_INSTRUCT,
    }
    if key in aliases:
        print(
            f"\n⚠️  Repo '{key}' does not exist on Hugging Face (404).\n"
            f"   Meta has no published Llama 3.3 **11B** Vision model id.\n"
            f"   Switching to: {_LLAMA_11B_VISION_INSTRUCT}\n"
            f"   (For 3.3 multimodal, use meta-llama/Llama-3.3-70B-Vision-Instruct if you have access.)\n"
        )
        return aliases[key]
    low = key.lower()
    if "llama-3.3" in low and "vision" in low and "11b" in low and "70" not in low:
        print(
            f"\n⚠️  Llama 3.3 + 11B + Vision is not a valid hub id; Meta lists 11B vision under Llama 3.2.\n"
            f"   Switching to: {_LLAMA_11B_VISION_INSTRUCT}\n"
        )
        return _LLAMA_11B_VISION_INSTRUCT
    return model_name


def evaluate(
    model: VideoLM,
    samples: List[Dict],
    output_file: str = None,
    max_samples: int = None,
    use_nlp_eval: bool = True,
    clear_cache: bool = True,
    prompt_template: Optional[str] = None,
    max_new_tokens: int = 64
) -> Dict:
    """
    Evaluate model on SQA dataset
    
    Args:
        model: VideoLM model instance
        samples: List of samples to evaluate
        output_file: Optional path to save detailed results
        max_samples: Optional limit on number of samples to evaluate
        
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
    
    print(f"\nEvaluating on {len(samples)} samples...")
    
    for sample in tqdm(samples, desc="Processing"):
        video_path = sample['video_path']
        question = sample['question']
        gt_answer = sample['gt_answer']
        
        # Check if video exists
        if not os.path.exists(video_path):
            print(f"Warning: Video not found: {video_path}")
            result = {
                **sample,
                'predicted_answer': 'VIDEO_NOT_FOUND',
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
        
        try:
            # Get model prediction
            # Use method-level prompt_template if provided, otherwise use instance-level
            predicted_answer = model.answer_question(
                video_path=video_path,
                question=question,
                max_new_tokens=max_new_tokens,  # Use parameter for concise answers
                temperature=0.1,  # Lower temperature for more deterministic answers
                prompt_template=prompt_template
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
                'correct': bool(is_correct_exact)  # Keep exact match as 'correct' for backward compatibility
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
            print("  - Use --load-in-8bit or --load-in-4bit flag for quantization")
            print("  - Use smaller model")
            print("  - Reduce --max-frames (e.g., --max-frames 4)")
            print("  - Process fewer samples at once (--max-samples)")
            
            # Save partial results before exiting
            if output_file:
                print(f"\n💾 Saving partial results to {output_file}...")
                with open(output_file, 'w') as f:
                    json.dump({
                        'accuracy': correct / total if total > 0 else 0.0,
                        'accuracies': {method: counts['correct'] / counts['total'] if counts['total'] > 0 else 0.0 
                                     for method, counts in metrics.items()},
                        'correct': correct,
                        'total': total,
                        'metrics': metrics,
                        'results': results,
                        'error': 'CUDA_OUT_OF_MEMORY',
                        'error_message': str(e),
                        'stopped_at_sample': len(results)
                    }, f, indent=2)
            
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
                print("Consider using --load-in-8bit or --load-in-4bit, or reducing --max-frames")
            
            # Check for quantization errors
            if "assertionerror" in error_msg.lower() and "bitsandbytes" in error_msg.lower():
                print("\n⚠️  Quantization error detected!")
                print("Llama4-VL models may have compatibility issues with quantization.")
                print("Solution: Try --no-quantization or use a different quantization method")
                print("See LLAMA4_FIXES.md for more details.")
            
            # Check for 'CB' attribute errors (vision encoder quantization issues)
            if "'parameter' object has no attribute 'cb'" in error_msg.lower() or "has no attribute 'CB'" in error_msg.lower():
                print("\n⚠️  Llama4-VL vision encoder quantization error detected!")
                print("The vision encoder is not compatible with bitsandbytes quantization.")
                print("Solution: Use --no-quantization instead")
                print("See LLAMA4_FIXES.md for more details.")
            
            # Store more detailed error info
            error_info = f"{error_type}: {error_msg}" if error_msg else error_type
            result = {
                **sample,
                'predicted_answer': f'ERROR: {error_info}',
                'error_type': error_type,
                'error_message': error_msg,
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
    
    # Save detailed results if requested
    if output_file:
        with open(output_file, 'w') as f:
            json.dump({
                'accuracy': accuracy,
                'accuracies': accuracies,
                'correct': correct,
                'total': total,
                'metrics': metrics,
                'results': results
            }, f, indent=2)
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
    parser = argparse.ArgumentParser(
        description='Zero-shot evaluation on SQA dataset (Llama4-VL optimized)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Llama 3.2 11B Vision Instruct (recommended ~11B multimodal id on Hugging Face)
  python evaluate_sqa_llama4.py --split test --model-name meta-llama/Llama-3.2-11B-Vision-Instruct

  # Llama 3.3 multimodal on HF is 70B (gated): meta-llama/Llama-3.3-70B-Vision-Instruct
  
  # Without quantization (requires more GPU memory)
  python evaluate_sqa_llama4.py --split test --model-name meta-llama/Llama-3.2-11B-Vision-Instruct --no-quantization

Note: Llama4-VL models may have compatibility issues with quantization.
      This script defaults to 8-bit quantization. Use --no-quantization if quantization fails.
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
        '--model-name',
        type=str,
        default='meta-llama/Llama-3.2-11B-Vision-Instruct',
        help='Llama Vision HF id (default: meta-llama/Llama-3.2-11B-Vision-Instruct). '
             'Note: Llama-3.3-Vision-11B* is not on HF; this script remaps it to the 3.2 11B instruct model.'
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
        help='Load model in 4-bit quantization (may have compatibility issues)'
    )
    parser.add_argument(
        '--load-in-8bit',
        action='store_true',
        help='Load model in 8-bit quantization (RECOMMENDED for Llama4-VL, default if no quantization flags specified)'
    )
    parser.add_argument(
        '--no-quantization',
        action='store_true',
        help='Disable quantization entirely (requires more GPU memory, most reliable option)'
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
    
    args = parser.parse_args()
    args.model_name = resolve_llama_vision_model_id(args.model_name)
    
    # Validate model name
    if not check_llama4_model(args.model_name):
        print(f"⚠️  Warning: Model '{args.model_name}' does not appear to be a Llama Vision model.")
        print("   This script is optimized for Llama Vision models (Llama 3.2/3.3/4 Vision).")
        print("   For Qwen models, use evaluate_sqa.py, evaluate_sqa_qwen25.py, or evaluate_sqa_qwen3.py")
        response = input("   Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Exiting.")
            return
    
    # Handle quantization flags
    # Default: use 8-bit quantization (may have issues, but try first)
    # NOTE: Llama4-VL may have quantization compatibility issues
    # If you encounter 'CB' attribute error, use --no-quantization instead
    load_in_4bit = False
    load_in_8bit = True  # Default to 8-bit for Llama4-VL
    
    if args.no_quantization:
        # User explicitly disabled quantization
        load_in_4bit = False
        load_in_8bit = False
        print("⚠️  Running without quantization (requires more GPU memory)")
        print("   This is the most reliable option for Llama4-VL if quantization fails")
    elif args.load_in_4bit:
        # User explicitly requested 4-bit
        load_in_4bit = True
        load_in_8bit = False
        print("⚠️  Using 4-bit quantization")
        print("   Note: Llama4-VL may have compatibility issues with quantization")
        print("   If you encounter errors, try --no-quantization instead")
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
    print(f"  Quantization: {'8-bit (recommended)' if load_in_8bit else '4-bit (⚠️ may have issues)' if load_in_4bit else 'None (full precision, most reliable)'}")
    
    model = VideoLM(
        model_name=args.model_name,
        max_frames=args.max_frames,
        frame_size=(448, 448),
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        prompt_template=prompt_template  # Set instance-level prompt template
    )
    
    # Generate output filename if not specified
    if not args.output:
        model_suffix = args.model_name.split('/')[-1].replace('-', '_').lower()
        quant_suffix = '8bit' if load_in_8bit else '4bit' if load_in_4bit else 'fp'
        if args.filter_scenes_json:
            # Include filter info in filename
            filter_name = Path(args.filter_scenes_json).stem
            output_file = f'sqa_{args.split}_results_{model_suffix}_{quant_suffix}_{filter_name}.json'
        else:
            output_file = f'sqa_{args.split}_results_{model_suffix}_{quant_suffix}.json'
    else:
        output_file = args.output
    
    # Evaluate
    results = evaluate(
        model=model,
        samples=samples,
        output_file=output_file,
        max_samples=args.max_samples,
        use_nlp_eval=not args.no_nlp_eval,
        clear_cache=not args.no_clear_cache,
        prompt_template=prompt_template,  # Pass as method parameter (can override instance-level)
        max_new_tokens=args.max_new_tokens
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
