"""SQA3D coarse question-type classification (first-word heuristic)."""

from __future__ import annotations

CATEGORY_ORDER = ["What", "Is", "How many", "Can", "Which", "Others"]


def classify_question_type(question: str) -> str:
    """
    Classify a question into SQA3D paper categories.

    Categories: What, Is, How many, Can, Which, Others
    """
    text = (question or "").strip().lower()
    if text.startswith("what"):
        return "What"
    if text.startswith("is "):
        return "Is"
    if text.startswith("how many") or text.startswith("how much"):
        return "How many"
    if text.startswith("can"):
        return "Can"
    if text.startswith("which"):
        return "Which"
    return "Others"
