"""YAML template loading and rule-based preprompt assembly (SpatioRoute-R)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import yaml

from videolm.prompt_utils import format_prompt_template


def load_prompt_templates(path: Path) -> Dict[str, str]:
    """Map ``prompts.<name>.template`` strings from ``prompt_config.yaml``."""
    if not path.is_file():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    prompts = data.get("prompts") or {}
    out: Dict[str, str] = {}
    for key, value in prompts.items():
        if isinstance(value, dict) and isinstance(value.get("template"), str):
            out[str(key)] = value["template"].strip()
    if "details_scene" not in out and "detailed_scene" in out:
        out["details_scene"] = out["detailed_scene"]
    if "focus_instructions" not in out and "instruction_focused" in out:
        out["focus_instructions"] = out["instruction_focused"]
    return out


def template_body_for_style(templates: Dict[str, str], style: str) -> str:
    order: Tuple[str, ...]
    if style == "details_scene":
        order = ("details_scene", "detailed_scene")
    elif style == "focus_instructions":
        order = ("focus_instructions", "instruction_focused")
    else:
        order = (style,)
    for key in order:
        if key in templates:
            return templates[key]
    return f"(No template in YAML for style '{style}'. Add it under prompts:.)"


def build_rule_preprompt(
    template: str,
    *,
    situation: str,
    question: str,
) -> str:
    """
    Fill a YAML template with situation + question (SpatioRoute-R).

    Templates that only declare ``{question}`` get a ``Situation:`` block inserted
    before the ``Question:`` line, matching the few-shot layout used in SpatioRoute-L.
    """
    q = (question or "").strip()
    sit = (situation or "").strip()
    text = format_prompt_template(template, question=q, situation=sit)

    if sit and "Situation:" not in text:
        needle = f"Question: {q}"
        if needle in text:
            text = text.replace(
                needle,
                f"Situation: {sit}\n\nQuestion: {q}",
                1,
            )
        else:
            text = f"Situation: {sit}\n\n{text}"
    return text.strip()
