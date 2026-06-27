"""Semantic flags fed to SpatioRoute-L (open vocab, directional, binary)."""

from __future__ import annotations

import re
from typing import Dict, List

_DIR_RE = re.compile(
    r"\b(left|right|front|behind|backward|ahead|forward|toward|towards|"
    r"beside|next to|nearest|farthest|across from|opposite|leftward|rightward)\b",
    re.I,
)


def _binary_sqa_type(sqa_category: str) -> bool:
    return sqa_category in ("Is", "Can")


def compute_semantic_flags(question: str, sqa_category: str) -> Dict[str, bool]:
    q = (question or "").strip().lower()
    directional = bool(_DIR_RE.search(question or ""))
    binary_lexical = _binary_sqa_type(sqa_category) or q.startswith(
        ("are ", "do ", "does ", "did ", "was ", "were ", "will ", "would ", "is there", "are there")
    )
    open_vocab = sqa_category in ("What", "How many", "Which") or (
        sqa_category == "Others" and not binary_lexical
    )
    return {
        "directional": directional,
        "binary_lexical": binary_lexical,
        "open_vocab": open_vocab,
    }


def format_semantics_instruction(flags: Dict[str, bool]) -> str:
    parts: List[str] = []
    if flags["open_vocab"]:
        parts.append(
            "OPEN_VOCAB: expect objects, colors, counts, or identities—ask the VLM for fine-grained visual evidence."
        )
    if flags["directional"]:
        parts.append(
            "DIRECTIONAL: question involves left/right/front/behind (etc.)—stress viewer-relative "
            "spatial, relational, and directional reasoning aligned with the Situation."
        )
    if flags["binary_lexical"]:
        parts.append(
            "BINARY: yes/no or true/false—require checking every clause of the question against the "
            "video; be strict about evidence."
        )
    return "\n".join(f"- {p}" for p in parts) if parts else "- (no extra semantics flags)"
