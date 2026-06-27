"""Shared prompt template substitution for VideoLM (``{question}``, ``{situation}``)."""

from __future__ import annotations

from typing import Optional


def vlm_input_text(
    question: str,
    *,
    prompt_template: Optional[str],
    instance_prompt_template: Optional[str],
    situation: Optional[str] = None,
) -> str:
    """
    Same text ``VideoLM.answer_question`` uses as the user question after template fill.

    With no template, this is ``question`` unchanged (e.g. full preprompt merged with situation).
    """
    template_to_use = (
        prompt_template if prompt_template is not None else instance_prompt_template
    )
    if template_to_use:
        return format_prompt_template(
            template_to_use, question=question, situation=situation
        )
    return question


def format_prompt_template(
    template: str,
    *,
    question: str,
    situation: Optional[str] = None,
) -> str:
    """
    Fill ``{question}`` and ``{situation}`` placeholders.

    Templates that only use ``{question}`` ignore the extra key. Empty / missing
    situation becomes the literal ``(not provided)`` so YAML lines still read well.
    """
    sit = (situation or "").strip() or "(not provided)"
    return template.format_map({"question": question, "situation": sit})
