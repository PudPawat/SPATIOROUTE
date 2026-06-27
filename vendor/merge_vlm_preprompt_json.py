#!/usr/bin/env python3
"""
Merge multiple VLM preprompt JSON bundles (``meta`` + ``results`` + ``by_question_id``)
into one file — same shape as ``generate_vlm_preprompts.py`` / ``generate_vlm_preprompts_augmentor_style.py``.

Rows are keyed by ``question_id``. If the same id appears in more than one input, choose
how to resolve with ``--on-duplicate``.

Example:
  python merge_vlm_preprompt_json.py \\
    --output generated_vlm_preprompts_augmentor_style_test_merged.json \\
    generated_vlm_preprompts_augmentor_style_test.json \\
    generated_vlm_preprompts_augmentor_style_test_1500.json \\
    generated_vlm_preprompts_augmentor_style_test_2500.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _norm_qid(qid: Any) -> int:
    if isinstance(qid, bool):
        raise ValueError("question_id must not be bool")
    if isinstance(qid, int):
        return int(qid)
    if isinstance(qid, float) and qid == int(qid):
        return int(qid)
    s = str(qid).strip()
    if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
        return int(s)
    raise ValueError(f"Unsupported question_id: {qid!r}")


def _row_ok(row: Dict[str, Any]) -> bool:
    p = (row.get("generated_prompt") or "").strip()
    return bool(p) and not p.upper().startswith("ERROR:")


def _pick_row(
    existing: Dict[str, Any],
    incoming: Dict[str, Any],
    mode: str,
) -> Dict[str, Any]:
    if mode == "last":
        return dict(incoming)
    if mode == "first":
        return dict(existing)
    if mode == "prefer_non_error":
        if _row_ok(incoming) and not _row_ok(existing):
            return dict(incoming)
        if _row_ok(existing) and not _row_ok(incoming):
            return dict(existing)
        return dict(incoming)
    raise ValueError(mode)


def load_bundle(path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level JSON must be an object")
    meta = data.get("meta") or {}
    results = list(data.get("results") or [])
    return meta, results


def merge_bundles(
    paths: List[Path],
    *,
    on_duplicate: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    merged: Dict[int, Dict[str, Any]] = {}
    source_metas: List[Dict[str, Any]] = []

    for p in paths:
        meta, results = load_bundle(p)
        source_metas.append({"path": str(p.resolve()), "meta": meta, "n_results": len(results)})
        for row in results:
            if row.get("question_id") is None:
                continue
            iq = _norm_qid(row["question_id"])
            row = dict(row)
            row["question_id"] = iq
            if iq not in merged:
                merged[iq] = row
            else:
                merged[iq] = _pick_row(merged[iq], row, on_duplicate)

    out_results = [merged[iq] for iq in sorted(merged.keys())]

    by_question_id: Dict[str, Any] = {}
    for r in out_results:
        gp = (r.get("generated_prompt") or "").strip()
        if gp and not gp.upper().startswith("ERROR:"):
            by_question_id[str(r["question_id"])] = {"generated_prompt": gp}

    out_meta: Dict[str, Any] = {
        "generator": "merge_vlm_preprompt_json.py",
        "merged_sources": [str(p.resolve()) for p in paths],
        "merge_on_duplicate": on_duplicate,
        "n_merged_results": len(out_results),
        "source_files": source_metas,
    }
    return out_meta, out_results, by_question_id


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge preprompt JSON bundles by question_id.")
    ap.add_argument(
        "inputs",
        nargs="+",
        type=str,
        help="Input JSON files (processed in order; later files can override earlier for same id).",
    )
    ap.add_argument("--output", "-o", required=True, help="Merged output JSON path.")
    ap.add_argument(
        "--on-duplicate",
        choices=("prefer_non_error", "last", "first"),
        default="prefer_non_error",
        help=(
            "prefer_non_error: keep successful generated_prompt over ERROR; tie → last file wins. "
            "last: always use the row from the rightmost file. first: keep the leftmost file's row."
        ),
    )
    args = ap.parse_args()

    paths = [Path(p).resolve() for p in args.inputs]
    for p in paths:
        if not p.is_file():
            raise FileNotFoundError(p)

    meta, results, by_id = merge_bundles(paths, on_duplicate=args.on_duplicate)
    out = {"meta": meta, "results": results, "by_question_id": by_id}
    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path} ({len(results)} rows, {len(by_id)} entries in by_question_id)")


if __name__ == "__main__":
    main()
