"""Build ``evaluation_settings`` objects stored in SQA result JSON files."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional


def build_evaluation_settings(**kwargs: Any) -> Dict[str, Any]:
    """
    Return a dict suitable for ``result_json["evaluation_settings"]``.
    Drops keys whose value is None. Converts Path to str.
    """
    fields = dict(kwargs)
    out: Dict[str, Any] = {}
    for k, v in fields.items():
        if v is None:
            continue
        if isinstance(v, Path):
            out[k] = str(v)
        elif isinstance(v, (str, int, float, bool)):
            out[k] = v
        elif isinstance(v, (list, dict)):
            out[k] = v
        else:
            out[k] = str(v)
    return out


def attach_result_json_metadata(
    target: MutableMapping[str, Any],
    evaluation_settings: Optional[Mapping[str, Any]],
) -> None:
    """
    Write ``evaluation_settings`` onto ``result`` JSON and also copy
    ``model_name`` / ``temperature`` to the top level for quick inspection
    (e.g. ``jq .model_name``, legacy files without nested settings).
    """
    if not evaluation_settings:
        return
    target["evaluation_settings"] = dict(evaluation_settings)
    mn = evaluation_settings.get("model_name")
    if mn is not None:
        target["model_name"] = mn
    temp = evaluation_settings.get("temperature")
    if temp is not None:
        target["temperature"] = temp


def normalize_question_id(qid: Any) -> Any:
    """
    Stable key for matching ``question_id`` across JSON round-trips and Python types
    (e.g. int in RAM vs str in some dumps, numpy integers).
    """
    if qid is None:
        return None
    if isinstance(qid, bool):
        return qid
    if isinstance(qid, int):
        return int(qid)
    try:
        import numpy as np

        if isinstance(qid, np.integer):
            return int(qid)
    except ImportError:
        pass
    if isinstance(qid, float) and qid == int(qid):
        return int(qid)
    if isinstance(qid, str):
        s = qid.strip()
        if s.isdigit() or (s.startswith("-") and len(s) > 1 and s[1:].isdigit()):
            try:
                return int(s)
            except ValueError:
                pass
        return s
    return qid


def checkpoint_json_incomplete(ck: Mapping[str, Any]) -> bool:
    """True if a result JSON looks like a partial run that should be resumed."""
    if not isinstance(ck, Mapping) or "results" not in ck:
        return False
    if ck.get("is_checkpoint") is True:
        return True
    if ck.get("stopped_at_sample") is not None:
        return True
    if ck.get("error") == "CUDA_OUT_OF_MEMORY":
        return True
    st = ck.get("samples_total")
    tot = ck.get("total")
    if st is not None and tot is not None:
        try:
            if int(st) > int(tot):
                return True
        except (TypeError, ValueError):
            pass
    return False


def attach_evaluation_settings(target: MutableMapping[str, Any], **fields: Any) -> None:
    """Set or replace ``target['evaluation_settings']`` from non-None fields."""
    es = build_evaluation_settings(**fields)
    if es:
        target["evaluation_settings"] = es
