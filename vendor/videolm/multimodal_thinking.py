"""
Multimodal image QA with Qwen3-VL-Thinking style split (think marker / </think>).

Used by ``VideoLM.answer_question_from_images_with_thinking`` and by CoT eval scripts
(``evaluate_cot_qwen25`` / ``evaluate_cot_qwen3``) so two-pass thinking works even when an
older editable install exposed a ``VideoLM`` class without that method.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image

from .prompt_utils import format_prompt_template


def split_qwen_thinking_string(text: str) -> Tuple[str, str]:
    """Split on closing think tag (Qwen3 / Qwen3-VL Thinking)."""
    marker = "</think>"
    if marker not in text:
        return "", text.strip()
    idx = text.rfind(marker)
    return text[:idx].strip(), text[idx + len(marker) :].strip()


def answer_from_images_with_thinking(
    vlm: Any,
    images: List[Image.Image],
    question: str,
    max_new_tokens: int = 8192,
    temperature: float = 1.0,
    top_p: float = 0.95,
    think_end_token_id: int = 151668,
    situation: Optional[str] = None,
) -> Tuple[str, str, str]:
    """
    ``vlm`` is a :class:`videolm.VideoLM` instance (duck-typed).

    Returns:
        (thinking_text, answer_after_think_marker, full_decoded_skip_special_true)
    """
    if vlm.prompt_template:
        formatted_question = format_prompt_template(
            vlm.prompt_template, question=question, situation=situation
        )
    else:
        formatted_question = question

    if vlm.is_clip:
        raise NotImplementedError(
            "answer_from_images_with_thinking requires a generative vision-language model."
        )

    if vlm.is_internvl and hasattr(vlm, "internvl_loader") and hasattr(vlm, "tokenizer") and hasattr(vlm, "image_processor"):
        try:
            generation_config = dict(
                num_beams=1,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0.0,
                temperature=temperature if temperature > 0.0 else None,
                top_p=top_p if temperature > 0.0 else None,
            )
            response = vlm.internvl_loader.chat(
                model=vlm.model,
                tokenizer=vlm.tokenizer,
                image_processor=vlm.image_processor,
                images=images[0] if len(images) == 1 else images,
                question=formatted_question,
                generation_config=generation_config,
            )
            text = response.strip()
            thinking_s, answer_s = split_qwen_thinking_string(text)
            return thinking_s, answer_s or text, text
        except Exception as e:
            print(f"   ⚠️  InternVL chat error: {str(e)[:200]}")
            print("   Falling back to standard multimodal processing...")

    if vlm.is_llava and hasattr(vlm, "llava_loader") and hasattr(vlm, "processor"):
        try:
            response = vlm.llava_loader.generate(
                model=vlm.model,
                processor=vlm.processor,
                images=images[0] if len(images) == 1 else images,
                question=formatted_question,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            text = response.strip()
            thinking_s, answer_s = split_qwen_thinking_string(text)
            return thinking_s, answer_s or text, text
        except Exception as e:
            print(f"   ⚠️  LLaVA generate error: {str(e)[:200]}")
            print("   Falling back to standard multimodal processing...")

    messages = [
        {
            "role": "user",
            "content": [{"type": "image", "image": img} for img in images]
            + [{"type": "text", "text": formatted_question}],
        }
    ]

    if hasattr(vlm.processor, "apply_chat_template"):
        text = vlm.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        text = formatted_question

    image_list = [item["image"] for item in messages[0]["content"] if item["type"] == "image"]

    inputs = vlm.processor(
        text=[text],
        images=image_list if image_list else None,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(vlm.device)

    gen_kwargs: Dict[str, Any] = {"max_new_tokens": max_new_tokens}
    if temperature > 0:
        gen_kwargs.update(
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
    else:
        gen_kwargs["do_sample"] = False

    if vlm.is_llama4:
        import transformers.generation.utils as gen_utils

        original_validate = gen_utils.GenerationMixin._validate_model_kwargs

        def patched_validate(self_m, model_kwargs):
            filtered_kwargs = {
                k: v
                for k, v in model_kwargs.items()
                if k not in ["pixel_values", "aspect_ratio_ids", "aspect_ratio_mask"]
            }
            return original_validate(self_m, filtered_kwargs)

        gen_utils.GenerationMixin._validate_model_kwargs = patched_validate
        try:
            with torch.no_grad():
                generated_ids = vlm.model.generate(**inputs, **gen_kwargs)
        finally:
            gen_utils.GenerationMixin._validate_model_kwargs = original_validate
    else:
        with torch.no_grad():
            generated_ids = vlm.model.generate(**inputs, **gen_kwargs)

    in_len = inputs.input_ids.shape[1]
    new_tokens = generated_ids[0][in_len:].tolist()

    tokenizer = getattr(vlm.processor, "tokenizer", None)
    if tokenizer is None:
        trimmed = [generated_ids[0][in_len:]]
        full_text = vlm.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        thinking, answer = split_qwen_thinking_string(full_text)
        return thinking, answer or full_text, full_text

    try:
        split_at = len(new_tokens) - new_tokens[::-1].index(think_end_token_id)
    except ValueError:
        split_at = 0

    thinking = tokenizer.decode(new_tokens[:split_at], skip_special_tokens=True).strip()
    answer = tokenizer.decode(new_tokens[split_at:], skip_special_tokens=True).strip()
    full_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    if not answer.strip():
        raw = tokenizer.decode(new_tokens, skip_special_tokens=False)
        thinking_s, answer_s = split_qwen_thinking_string(raw)
        if answer_s.strip():
            return thinking_s, answer_s.strip(), full_text or answer_s.strip()

    return thinking, answer.strip() or full_text, full_text
