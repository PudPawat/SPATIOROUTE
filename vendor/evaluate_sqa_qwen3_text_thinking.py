"""
SQA evaluation using Qwen3 *text* thinking models (e.g. Qwen3-4B-Thinking-2507).

IMPORTANT: https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507 is a causal LM only — no vision.
This script feeds each sample's SITUATION text + QUESTION (no video). Results are NOT comparable
to full VideoLM SQA runs; use for ablations or text-only baselines.

Requires transformers>=4.51 (see model card). Parsing follows the official snippet (token id
151668 = end-of-thinking marker).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

from videolm.evaluators import AnswerEvaluator
from utils.question_type_filter import filter_samples_by_question_type, get_question_type_statistics
from utils.evaluation_result_meta import attach_result_json_metadata, build_evaluation_settings

# Official Qwen3 thinking end marker token id (see model card)
QWEN3_THINK_END_TOKEN_ID = 151668


def load_dataset(
    questions_path: str,
    annotations_path: str,
    video_dir: str,
) -> List[Dict]:
    with open(questions_path, "r") as f:
        questions_data = json.load(f)
        questions = questions_data.get("questions", questions_data)

    with open(annotations_path, "r") as f:
        annotations_data = json.load(f)
        annotations = annotations_data.get("annotations", annotations_data)

    annotation_map = {ann["question_id"]: ann for ann in annotations}
    samples = []
    for q in questions:
        qid = q["question_id"]
        scene_id = q["scene_id"]
        if qid not in annotation_map:
            continue
        ann = annotation_map[qid]
        video_path = os.path.join(video_dir, f"{scene_id}.mp4")
        gt_answer = ann["answers"][0]["answer"].lower().strip()
        samples.append(
            {
                "question_id": qid,
                "scene_id": scene_id,
                "question": q["question"],
                "situation": q.get("situation", ""),
                "video_path": video_path,
                "gt_answer": gt_answer,
                "answer_type": ann.get("answer_type", "unknown"),
            }
        )
    return samples


def normalize_answer(answer: str) -> str:
    answer = answer.lower().strip()
    for ch in ".!,?":
        answer = answer.replace(ch, "")
    return answer


def exact_match(pred: str, gt: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gt)


def build_user_prompt(situation: str, question: str) -> str:
    situation = (situation or "").strip()
    return (
        "You are answering a question about an indoor 3D scene. "
        "The following is a text description of the observer's situation (no images).\n\n"
        f"SITUATION:\n{situation if situation else '(not provided)'}\n\n"
        f"QUESTION:\n{question}\n\n"
        "Answer concisely when possible (short phrase or single word for benchmarks)."
    )


def split_thinking_from_tokens(
    tokenizer,
    output_ids: List[int],
) -> Tuple[str, str]:
    """Match Hugging Face Qwen3 thinking quickstart (last occurrence of think-end token)."""
    try:
        index = len(output_ids) - output_ids[::-1].index(QWEN3_THINK_END_TOKEN_ID)
    except ValueError:
        index = 0
    thinking = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    return thinking, content


def split_thinking_from_string(text: str) -> Tuple[str, str]:
    """Fallback if token id layout differs between tokenizer versions."""
    marker = "</think>"
    if marker not in text:
        return "", text.strip()
    idx = text.rfind(marker)
    thinking = text[:idx].strip()
    answer = text[idx + len(marker) :].strip()
    return thinking, answer


def load_model_and_tokenizer(
    model_name: str,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    kwargs = {"torch_dtype": "auto", "device_map": "auto", "trust_remote_code": True}
    if load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif load_in_8bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def generate_answer(
    model,
    tokenizer,
    user_text: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> Tuple[str, str, str]:
    messages = [{"role": "user", "content": user_text}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    gen_kwargs: Dict = {"max_new_tokens": max_new_tokens}
    if temperature > 0:
        gen_kwargs.update(
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
    else:
        gen_kwargs["do_sample"] = False
    generated_ids = model.generate(**model_inputs, **gen_kwargs)
    new_tokens = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
    full_decoded = tokenizer.decode(new_tokens, skip_special_tokens=False).strip()

    thinking_tok, content_tok = split_thinking_from_tokens(tokenizer, new_tokens)
    if not content_tok.strip() and "</think>" in full_decoded:
        thinking_str, content_str = split_thinking_from_string(full_decoded)
        return thinking_str, content_str, full_decoded
    return thinking_tok, content_tok, full_decoded


def evaluate_text_thinking(
    model,
    tokenizer,
    samples: List[Dict],
    output_file: Optional[str],
    max_samples: Optional[int],
    use_nlp_eval: bool,
    clear_cache: bool,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    evaluation_settings: Optional[Dict[str, Any]] = None,
) -> Dict:
    if max_samples:
        samples = samples[:max_samples]

    nlp_evaluator = AnswerEvaluator() if use_nlp_eval else None
    metrics = {
        "exact_match": {"correct": 0, "total": 0},
        "semantic": {"correct": 0, "total": 0},
        "bleu": {"correct": 0, "total": 0},
        "rouge": {"correct": 0, "total": 0},
        "fuzzy": {"correct": 0, "total": 0},
        "contains": {"correct": 0, "total": 0},
    }
    results: List[Dict] = []
    correct = 0
    total = 0

    for sample in tqdm(samples, desc="Qwen3 text thinking"):
        question = sample["question"]
        situation = sample.get("situation", "")
        gt = sample["gt_answer"]
        user_prompt = build_user_prompt(situation, question)

        try:
            thinking, predicted, raw_tail = generate_answer(
                model,
                tokenizer,
                user_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            is_ok = exact_match(predicted, gt)
            if is_ok:
                correct += 1
            total += 1

            row = {
                **sample,
                "input_type": "text_situation_only",
                "model_note": "Qwen3 thinking LM — no video; not comparable to VL SQA",
                "user_prompt": user_prompt,
                "thinking_reasoning": thinking,
                "predicted_answer": predicted,
                "raw_generation_tail": raw_tail[:2000] if len(raw_tail) > 2000 else raw_tail,
                "correct": bool(is_ok),
            }
            if nlp_evaluator:
                row.update(nlp_evaluator.evaluate(predicted, gt))
                metrics["exact_match"]["correct"] += int(is_ok)
                metrics["exact_match"]["total"] += 1
                for k in ("semantic", "bleu", "rouge", "fuzzy", "contains"):
                    ck = f"correct_{k}" if k != "contains" else "correct_contains"
                    if ck in row:
                        metrics[k]["correct"] += int(bool(row[ck]))
                        metrics[k]["total"] += 1
            else:
                metrics["exact_match"]["correct"] += int(is_ok)
                metrics["exact_match"]["total"] += 1

            results.append(row)
        except Exception as e:
            err = str(e)
            print(f"Error sample {sample.get('question_id')}: {err}")
            row = {
                **sample,
                "input_type": "text_situation_only",
                "predicted_answer": f"ERROR: {err}",
                "thinking_reasoning": "",
                "correct": False,
            }
            if nlp_evaluator:
                row.update(
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
            results.append(row)
            total += 1

        if clear_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    accuracies = {
        m: (metrics[m]["correct"] / metrics[m]["total"] if metrics[m]["total"] else 0.0)
        for m in metrics
    }
    accuracy = accuracies.get("exact_match", 0.0)
    out = {
        "accuracy": accuracy,
        "accuracies": accuracies,
        "correct": correct,
        "total": total,
        "metrics": metrics,
        "results": results,
        "meta": {
            "model": "Qwen3 text thinking (no vision)",
            "reference": "https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507",
        },
    }
    if output_file:
        attach_result_json_metadata(out, evaluation_settings)
        with open(output_file, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Saved: {output_file}")
    return out


def main():
    p = argparse.ArgumentParser(
        description="SQA with Qwen3-4B-Thinking (text-only: situation + question, no video)"
    )
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen3-4B-Thinking-2507",
        help="HF id, default Qwen3-4B-Thinking-2507",
    )
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--dataset-dir", type=str, default="dataset/SQA")
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Qwen3 thinking best-practice default 0.6 (see model card)",
    )
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--load-in-8bit", action="store_true")
    p.add_argument("--no-nlp-eval", action="store_true")
    p.add_argument("--no-clear-cache", action="store_true")
    p.add_argument("--question-types", type=str, nargs="+", default=None)
    args = p.parse_args()

    if args.load_in_4bit and args.load_in_8bit:
        raise SystemExit("Use only one of --load-in-4bit / --load-in-8bit")

    dataset_dir = Path(args.dataset_dir)
    questions_path = dataset_dir / "sqa_task" / "balanced" / f"v1_balanced_questions_{args.split}_scannetv2.json"
    annotations_path = (
        dataset_dir / "sqa_task" / "balanced" / f"v1_balanced_sqa_annotations_{args.split}_scannetv2.json"
    )
    video_dir = dataset_dir / "video"

    for path in (questions_path, annotations_path):
        if not path.exists():
            raise FileNotFoundError(path)

    samples = load_dataset(str(questions_path), str(annotations_path), str(video_dir))
    print(f"Loaded {len(samples)} samples (video paths ignored for generation)")

    if args.question_types:
        before = len(samples)
        samples = filter_samples_by_question_type(samples, args.question_types, match_any=True)
        print(f"Question-type filter: {before} -> {len(samples)}")
        stats = get_question_type_statistics(samples)
        for t, c in sorted(stats.items(), key=lambda x: -x[1]):
            print(f"  {t}: {c}")

    print(f"Loading {args.model_name} …")
    model, tokenizer = load_model_and_tokenizer(
        args.model_name,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
    )

    out_path = args.output
    if not out_path:
        safe = args.model_name.split("/")[-1].replace("-", "_").lower()
        out_path = f"sqa_{args.split}_results_qwen3_text_thinking_{safe}.json"

    evaluation_settings = build_evaluation_settings(
        script=os.path.basename(__file__),
        model_name=args.model_name,
        temperature=float(args.temperature),
        max_new_tokens=args.max_new_tokens,
        split=args.split,
        dataset_dir=args.dataset_dir,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        top_p=args.top_p,
        input_type="text_situation_only",
    )

    summary = evaluate_text_thinking(
        model,
        tokenizer,
        samples,
        output_file=out_path,
        max_samples=args.max_samples,
        use_nlp_eval=not args.no_nlp_eval,
        clear_cache=not args.no_clear_cache,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        evaluation_settings=evaluation_settings,
    )

    print("=" * 60)
    print(f"Exact match: {summary['accuracy']:.4f} ({summary['correct']}/{summary['total']})")
    for name, acc in summary.get("accuracies", {}).items():
        if name != "exact_match":
            c = summary["metrics"][name]
            print(f"  {name}: {acc:.4f} ({c['correct']}/{c['total']})")
    print("=" * 60)


if __name__ == "__main__":
    main()
