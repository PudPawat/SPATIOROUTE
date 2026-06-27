"""Question-type routing for SpatioRoute-R and SpatioRoute-L."""

from experiments.spatioroute.routing.classify import CATEGORY_ORDER, classify_question_type
from experiments.spatioroute.routing.resolve import INSTRUCTION_STYLE_RULES, resolve_instruction_style
from experiments.spatioroute.routing.semantics import compute_semantic_flags, format_semantics_instruction
from experiments.spatioroute.routing.templates import (
    build_rule_preprompt,
    load_prompt_templates,
    template_body_for_style,
)

__all__ = [
    "CATEGORY_ORDER",
    "INSTRUCTION_STYLE_RULES",
    "build_rule_preprompt",
    "classify_question_type",
    "compute_semantic_flags",
    "format_semantics_instruction",
    "load_prompt_templates",
    "resolve_instruction_style",
    "template_body_for_style",
]
