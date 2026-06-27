"""Map SQA3D question type → prompt template key (SpatioRoute-R routing table)."""

from __future__ import annotations

INSTRUCTION_STYLE_RULES = {
    "What, How many, Which": "details_scene",
    "Is": "step_by_step",
    "Can": "scene_understanding",
    "Others": "focus_instructions",
}


def resolve_instruction_style(sqa_category: str) -> str:
    """Return ``prompt_config.yaml`` ``prompts`` key for the given SQA category."""
    if sqa_category in ("What", "How many", "Which"):
        return "details_scene"
    if sqa_category == "Is":
        return "step_by_step"
    if sqa_category == "Can":
        return "scene_understanding"
    return "focus_instructions"
