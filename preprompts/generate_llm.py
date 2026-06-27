#!/usr/bin/env python3
"""
SpatioRoute-L: LLM-driven dynamic prompt routing with few-shot examples.

Routing uses the same rule table as SpatioRoute-R, but a small **text-only** LLM
(default: Qwen/Qwen2.5-0.5B-Instruct) writes the final VLM prompt from:
  - question + situation (no video at routing time)
  - instruction_style chosen by rules
  - YAML template excerpt + few-shot demonstrations

Example:
  python -m experiments.spatioroute.preprompts.generate_llm \\
    --split test \\
    --output experiments/spatioroute/results/preprompts_l_test.json \\
    --max-samples 50
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluate_sqa import load_dataset  # noqa: E402
from evaluate_sqa_qwen3_text_thinking import (  # noqa: E402
    load_model_and_tokenizer,
    split_thinking_from_string,
    split_thinking_from_tokens,
)
from experiments.spatioroute._paths import (  # noqa: E402
    DEFAULT_FEW_SHOTS,
    DEFAULT_PROMPT_CONFIG,
    DEFAULT_SYSTEM_PROMPT,
)
from experiments.spatioroute.preprompts.io import (  # noqa: E402
    completed_prompt_row,
    load_existing_ids,
    save_preprompt_bundle,
)
from experiments.spatioroute.routing import (  # noqa: E402
    INSTRUCTION_STYLE_RULES,
    classify_question_type,
    compute_semantic_flags,
    format_semantics_instruction,
    load_prompt_templates,
    resolve_instruction_style,
    template_body_for_style,
)

MODEL_PRESETS: Dict[str, str] = {
    "qwen2-0.5b-instruct": "Qwen/Qwen2-0.5B-Instruct",
    "qwen2-1.5b-instruct": "Qwen/Qwen2-1.5B-Instruct",
    "qwen2-3b-instruct": "Qwen/Qwen2-3B-Instruct",
    "qwen2-7b-instruct": "Qwen/Qwen2-7B-Instruct",
    "qwen25-0.5b-instruct": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen25-1.5b-instruct": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen25-3b-instruct": "Qwen/Qwen2.5-3B-Instruct",
    "qwen25-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
}


def build_user_block(
    *,
    situation: str,
    question: str,
    sqa_category: str,
    instruction_style: str,
    semantic_flags: Dict[str, bool],
    style_template_snippet: str,
    yaml_ref: str,
    few_shots: str,
    yaml_max_chars: int,
) -> str:
    cap = max(256, yaml_max_chars)
    truncated = yaml_ref[:cap]
    tail = "\n... (truncated)" if len(yaml_ref) > cap else ""
    sem_block = format_semantics_instruction(semantic_flags)

    return f"""## Reference: full prompt_config.yaml (excerpt)
```yaml
{truncated}{tail}
```

## YAML template for instruction_style `{instruction_style}`

Your output must **follow this template’s layout** (copy the headings, bullets, numbered steps, and labels). Insert the real situation and question text where appropriate.

For **`step_by_step`**, match ``prompt_config.yaml`` exactly: numbered lines 1–3, then ``Situation:`` … then ``Question:`` … then ``Answer:``.

For **`details_scene`**, **`scene_understanding`**, and **`focus_instructions`**, still **add** ``Situation:`` and ``Question:`` with the Input text before the closing line, as in the few-shots.

```text
{style_template_snippet}
```

{few_shots}

---

## Your task for this sample

**Question Type (SQA):** {sqa_category}

**instruction_style** (chosen by code from Question Type): `{instruction_style}`
- What / How many / Which → details_scene
- Is → step_by_step
- Can → scene_understanding
- Others → focus_instructions

**Semantics (apply all that apply):**
{sem_block}

Input:
Situation: {situation.strip() if situation else "(none)"}
Question: {question.strip()}

Output:
"""


def resolve_thinking_strip(mode: str, model_name: str) -> bool:
    if mode == "on":
        return True
    if mode == "off":
        return False
    if mode == "auto":
        return "thinking" in model_name.lower()
    raise ValueError(mode)


@torch.inference_mode()
def generate_one(
    model,
    tokenizer,
    user_block: str,
    system_prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    thinking: bool,
) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_block},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    gen: Dict[str, Any] = {"max_new_tokens": max_new_tokens}
    if temperature > 0:
        gen.update(do_sample=True, temperature=temperature, top_p=top_p)
    else:
        gen["do_sample"] = False
    out = model.generate(**inputs, **gen)
    new_ids = out[0][inputs.input_ids.shape[1] :].tolist()
    full_decoded = tokenizer.decode(new_ids, skip_special_tokens=False).strip()

    if thinking:
        _, content = split_thinking_from_tokens(tokenizer, new_ids)
        think_end = "</" + "redacted_thinking" + ">"
        if not content.strip() and think_end in full_decoded:
            _, content = split_thinking_from_string(full_decoded)
        return content.strip() or full_decoded
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SpatioRoute-L: LLM preprompt generation with few-shots")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--dataset-dir", default="dataset/SQA")
    parser.add_argument("--model-preset", choices=sorted(MODEL_PRESETS.keys()), default=None)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt-config", default=str(DEFAULT_PROMPT_CONFIG))
    parser.add_argument("--few-shots-file", default=str(DEFAULT_FEW_SHOTS))
    parser.add_argument("--system-prompt-file", default=str(DEFAULT_SYSTEM_PROMPT))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--question-types", nargs="+", default=None)
    parser.add_argument("--thinking-strip", choices=("off", "auto", "on"), default="off")
    parser.add_argument("--yaml-reference-max-chars", type=int, default=12000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.model_preset:
        args.model_name = MODEL_PRESETS[args.model_preset]
    if args.load_in_4bit and args.load_in_8bit:
        raise SystemExit("Use only one of --load-in-4bit / --load-in-8bit")

    dataset_dir = Path(args.dataset_dir)
    questions_path = dataset_dir / "sqa_task" / "balanced" / f"v1_balanced_questions_{args.split}_scannetv2.json"
    annotations_path = dataset_dir / "sqa_task" / "balanced" / f"v1_balanced_sqa_annotations_{args.split}_scannetv2.json"
    video_dir = dataset_dir / "video"
    for path in (questions_path, annotations_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    yaml_path = Path(args.prompt_config)
    yaml_ref = yaml_path.read_text(encoding="utf-8") if yaml_path.is_file() else ""
    templates = load_prompt_templates(yaml_path)
    few_shots = Path(args.few_shots_file).read_text(encoding="utf-8").strip()
    system_prompt = Path(args.system_prompt_file).read_text(encoding="utf-8").strip()

    samples = load_dataset(str(questions_path), str(annotations_path), str(video_dir))
    if args.question_types:
        from utils.question_type_filter import filter_samples_by_question_type

        samples = filter_samples_by_question_type(samples, args.question_types, match_any=True)
    if args.max_samples:
        samples = samples[: args.max_samples]

    out_path = Path(args.output)
    done = load_existing_ids(out_path)
    if done:
        print(f"Resume: skipping {len(done)} question_ids already in {out_path}")

    thinking = resolve_thinking_strip(args.thinking_strip, args.model_name)
    print(f"Loading router LLM {args.model_name} (thinking_strip={thinking}) …")
    model, tokenizer = load_model_and_tokenizer(
        args.model_name,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
    )

    meta = {
        "generator": "experiments.spatioroute.preprompts.generate_llm",
        "method": "SpatioRoute-L",
        "model_name": args.model_name,
        "model_preset": args.model_preset,
        "split": args.split,
        "instruction_style_rules": INSTRUCTION_STYLE_RULES,
        "prompt_config_ref": str(yaml_path.resolve()),
        "few_shots_file": str(Path(args.few_shots_file).resolve()),
        "system_prompt_file": str(Path(args.system_prompt_file).resolve()),
        "yaml_reference_max_chars": args.yaml_reference_max_chars,
        "thinking_strip": thinking,
    }

    results: List[Dict[str, Any]] = []
    if out_path.is_file():
        import json

        try:
            results = list(json.loads(out_path.read_text(encoding="utf-8")).get("results") or [])
        except json.JSONDecodeError:
            results = []
    results = [row for row in results if completed_prompt_row(row)]
    seen = {int(row["question_id"]) for row in results if row.get("question_id") is not None}
    seen |= done

    n_new = 0
    for sample in tqdm(samples, desc="spatioroute-l"):
        qid = int(sample["question_id"])
        if qid in seen:
            continue
        question = sample.get("question") or ""
        situation = sample.get("situation") or ""
        qtype = classify_question_type(question)
        style = resolve_instruction_style(qtype)
        flags = compute_semantic_flags(question, qtype)
        style_snippet = template_body_for_style(templates, style)
        user_block = build_user_block(
            situation=situation,
            question=question,
            sqa_category=qtype,
            instruction_style=style,
            semantic_flags=flags,
            style_template_snippet=style_snippet,
            yaml_ref=yaml_ref,
            few_shots=few_shots,
            yaml_max_chars=args.yaml_reference_max_chars,
        )
        try:
            generated = generate_one(
                model,
                tokenizer,
                user_block,
                system_prompt,
                args.max_new_tokens,
                args.temperature,
                args.top_p,
                thinking,
            )
        except Exception as exc:
            generated = f"ERROR: {exc}"
        row = {
            "question_id": qid,
            "scene_id": sample.get("scene_id"),
            "question_type": qtype,
            "instruction_style": style,
            "semantic_flags": flags,
            "question": question,
            "situation": situation,
            "generated_prompt": generated.strip(),
        }
        results.append(row)
        seen.add(qid)
        n_new += 1
        if args.checkpoint_every and n_new % args.checkpoint_every == 0:
            save_preprompt_bundle(out_path, meta, results)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    save_preprompt_bundle(out_path, meta, results)
    print(f"Done. Wrote {len(results)} rows to {out_path} ({n_new} new this run)")


if __name__ == "__main__":
    main()
