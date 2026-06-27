"""Shared JSON I/O for preprompt bundles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Set


def completed_prompt_row(row: Dict[str, Any]) -> bool:
    prompt = (row.get("generated_prompt") or "").strip()
    return bool(prompt) and not prompt.upper().startswith("ERROR:")


def load_existing_ids(path: Path) -> Set[int]:
    if not path.is_file():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return set()
    out: Set[int] = set()
    for row in data.get("results") or []:
        if not completed_prompt_row(row):
            continue
        qid = row.get("question_id")
        if qid is not None:
            out.add(int(qid))
    return out


def save_preprompt_bundle(path: Path, meta: Dict[str, Any], results: List[Dict[str, Any]]) -> None:
    by_id: Dict[str, Dict[str, str]] = {}
    for row in results:
        qid = row.get("question_id")
        if qid is None:
            continue
        key = str(int(qid) if isinstance(qid, (int, float)) else qid)
        by_id[key] = {"generated_prompt": row.get("generated_prompt", "")}
    doc = {"meta": meta, "results": results, "by_question_id": by_id}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(doc, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
