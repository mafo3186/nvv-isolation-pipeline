from __future__ import annotations

from typing import List
from pathlib import Path

from utils.io import read_json_with_status


def build_candidate_label(seg: dict) -> str:
    """
    Build candidate label exactly like json_nvv_to_audacity_labels():
    category (source): text
    """
    text = seg.get("text", "") or ""
    label = seg.get("category", "nvv") or "nvv"
    src = seg.get("source", "") or ""
    if src:
        label = f"{label} ({src})"
    if text:
        label = f"{label}: {text}"
    return label


def load_candidate_events_from_nvv_json(candidate_path: Path) -> List[dict]:
    """
    Load NVV candidate JSON and return list of canonical candidate event dicts.

    Output event keys:
        cand_event_id, cand_start_s, cand_end_s, cand_label
    """
    data, status = read_json_with_status(candidate_path)
    if status != "ok":
        raise ValueError(f"Cannot read candidate JSON ({status}): {candidate_path}")

    if not isinstance(data, dict):
        raise ValueError(f"Invalid candidate JSON schema (expected dict): {candidate_path}")

    segments = data.get("nvv", [])
    if not isinstance(segments, list):
        raise ValueError(f"Invalid candidate JSON schema (expected 'nvv' as list): {candidate_path}")

    events: List[dict] = []
    for seg in segments:
        if not isinstance(seg, dict):
            raise ValueError(f"Invalid NVV segment schema (expected dict): {candidate_path}")

        start = seg.get("start")
        end = seg.get("end")
        cand_event_id = seg.get("candidate_id", None)
        if start is None or end is None or cand_event_id is None:
            raise ValueError(f"Invalid NVV segment schema (missing start/end/candidate_id): {candidate_path}")

        try:
            start_f = float(start)
            end_f = float(end)
        except Exception as e:
            raise ValueError(f"Invalid NVV segment schema (non-numeric start/end): {candidate_path}") from e

        if end_f <= start_f:
            raise ValueError(f"Invalid NVV segment schema (end <= start): {candidate_path}")

        ev = {
            "cand_event_id": cand_event_id,
            "cand_start_s": start_f,
            "cand_end_s": end_f,
            "cand_label": build_candidate_label(seg),
        }
        events.append(ev)

    return events