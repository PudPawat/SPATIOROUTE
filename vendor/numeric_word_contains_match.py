"""
Contains-style matching after **both** sides normalize standalone **digit tokens** to English words.

Examples (all should score as correct):

- ``gt_answer=1``, ``predicted_answer="there's only one"`` → GT becomes ``one``, pred unchanged, ``\\bone\\b`` in pred.
- ``gt_answer=1``, ``predicted_answer="there's only 1"`` → both use the word ``one``.
- ``gt_answer=one``, ``predicted_answer="there's only 1"`` → same after normalization.
- ``gt_answer=one``, ``predicted_answer="there's only one"`` → phrase / token match with **word boundaries** (avoids spurious ``one`` ⊂ ``only``-style errors on short junk substrings inside longer words).

Digit tokens use ``\\b`` so ``scene0050`` and ``3d`` are not rewritten.

For ``make_research_table.py`` use ``numeric_word_contains_match`` via ``--metric contains_numword``.
For lowercase-only matching (no digit→word), use ``--metric contains_lower`` or
``--metric contains_lower_score`` (mean of ``_standard_contains`` scores).
"""

from __future__ import annotations

import math
import re
from typing import Any, List, Tuple

# 0 .. 20 inclusive
_SMALL: List[str] = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty",
]

_TENS: dict[int, str] = {
    2: "twenty",
    3: "thirty",
    4: "forty",
    5: "fifty",
    6: "sixty",
    7: "seventy",
    8: "eighty",
    9: "ninety",
}

_STANDALONE_DIGITS = re.compile(r"\b([0-9]+)\b")

_MAX_SPELL = 9999


def int_to_english_words(n: int) -> str:
    """Spell non-negative ``n`` in English; fall back to str(n) if out of range."""
    if n < 0 or n > _MAX_SPELL:
        return str(n)
    if n <= 20:
        return _SMALL[n]
    if n < 100:
        t, o = divmod(n, 10)
        if o == 0:
            return _TENS[t]
        return f"{_TENS[t]} {_SMALL[o]}"
    if n < 1000:
        h, r = divmod(n, 100)
        if r == 0:
            return f"{_SMALL[h]} hundred"
        return f"{_SMALL[h]} hundred {int_to_english_words(r)}"
    k, r = divmod(n, 1000)
    if r == 0:
        return f"{int_to_english_words(k)} thousand"
    return f"{int_to_english_words(k)} thousand {int_to_english_words(r)}"


def digits_to_words_text(text: str) -> str:
    """Replace each standalone digit token with spelled-out English."""

    def _sub(m: re.Match) -> str:
        return int_to_english_words(int(m.group(1)))

    return _STANDALONE_DIGITS.sub(_sub, text)


def spell_answer_field(value: Any) -> str:
    """
    Normalize a single answer field for storage/export: JSON ints become words (``1`` → ``one``),
    strings get standalone digit tokens spelled (``\"2 chairs\"`` → ``\"two chairs\"``).
    Booleans are spelled as ``yes`` / ``no`` so they are not treated as integers.
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return int_to_english_words(int(value))
    if isinstance(value, float):
        if math.isfinite(value) and value == int(value):
            return int_to_english_words(int(value))
        return digits_to_words_text(str(value).strip())
    return digits_to_words_text(str(value).strip())


def normalize_for_contains_match(text: str) -> str:
    """Lowercase, collapse whitespace; hyphens → spaces so *twenty-one* aligns with *twenty one*."""
    text = text.lower().strip()
    text = text.replace("-", " ")
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_pair_with_digits_as_words(pred: str, gt: str) -> Tuple[str, str]:
    """Digit→word on both strings, then shared text normalization."""
    p = normalize_for_contains_match(digits_to_words_text(str(pred)))
    g = normalize_for_contains_match(digits_to_words_text(str(gt)))
    return p, g


def _strip_token_punct(w: str) -> str:
    return w.strip(".,;:!?\"'()[]`").lower()


def _gt_phrase_pattern(gt_norm: str) -> str:
    """Regex: GT words in order, each as a whole word (\\b…\\b), allowing flexible space between."""
    words = [_strip_token_punct(t) for t in gt_norm.split() if _strip_token_punct(t)]
    if not words:
        return ""
    return r"\s+".join(rf"\b{re.escape(w)}\b" for w in words)


def _standard_contains(pred_norm: str, gt_norm: str) -> Tuple[float, bool]:
    """
    GT must appear as **whole words** (in order), then token-overlap fallback with punctuation stripped.
    Avoids naive substring ``gt in pred`` (e.g. ``one`` inside unrelated words) and ``one.`` vs ``one``.
    """
    if not gt_norm:
        return 0.0, False

    phrase = _gt_phrase_pattern(gt_norm)
    if phrase and re.search(phrase, pred_norm):
        return 1.0, True

    gt_words = [_strip_token_punct(t) for t in gt_norm.split() if _strip_token_punct(t)]
    if not gt_words:
        return 0.0, False

    pred_words = [_strip_token_punct(t) for t in pred_norm.split() if _strip_token_punct(t)]
    pred_set = set(pred_words)
    matched = sum(1 for w in gt_words if w in pred_set)
    match_score = matched / len(gt_words)
    is_correct = match_score >= 0.8 or (len(gt_words) == 1 and match_score > 0)
    return float(match_score), bool(is_correct)


def contains_match_with_numeric_words(pred: str, gt: str) -> Tuple[float, bool]:
    """
    Normalize digits→words on **both** sides, then contains-style match with word boundaries.
    """
    try:
        pred_norm, gt_norm = normalize_pair_with_digits_as_words(pred, gt)
        return _standard_contains(pred_norm, gt_norm)
    except Exception:
        return 0.0, False


def json_row_correct_contains_numword(row: dict) -> bool:
    pred = row.get("predicted_answer") or row.get("prediction") or ""
    gt = row.get("gt_answer") or row.get("answer") or ""
    if gt is not None and not isinstance(gt, str):
        gt = str(gt)
    if pred is not None and not isinstance(pred, str):
        pred = str(pred)
    _, ok = contains_match_with_numeric_words(str(pred), str(gt))
    return bool(ok)


def contains_match_lowercase_only(pred: str, gt: str) -> Tuple[float, bool]:
    """
    Lowercase both sides (plus shared whitespace / hyphen normalization only).
    No digit→word step. Same word-boundary ``contains`` score and pass/fail as
    ``_standard_contains`` (phrase match or token-overlap threshold).
    """
    try:
        pred_norm = normalize_for_contains_match(str(pred))
        gt_norm = normalize_for_contains_match(str(gt))
        return _standard_contains(pred_norm, gt_norm)
    except Exception:
        return 0.0, False


def json_row_contains_score_lowercase(row: dict) -> float:
    """Float in [0, 1]: overlap / phrase-style contains score after lowercasing only."""
    pred = row.get("predicted_answer") or row.get("prediction") or ""
    gt = row.get("gt_answer") or row.get("answer") or ""
    if gt is not None and not isinstance(gt, str):
        gt = str(gt)
    if pred is not None and not isinstance(pred, str):
        pred = str(pred)
    score, _ = contains_match_lowercase_only(str(pred), str(gt))
    return float(score)


def json_row_correct_contains_lowercase(row: dict) -> bool:
    """Binary correct: same threshold as ``contains_match_lowercase_only`` second return value."""
    pred = row.get("predicted_answer") or row.get("prediction") or ""
    gt = row.get("gt_answer") or row.get("answer") or ""
    if gt is not None and not isinstance(gt, str):
        gt = str(gt)
    if pred is not None and not isinstance(pred, str):
        pred = str(pred)
    _, ok = contains_match_lowercase_only(str(pred), str(gt))
    return bool(ok)
