from __future__ import annotations

from typing import Dict, List, Any, Optional
import pandas as pd


def parse_time_to_seconds(value: Any) -> Optional[float]:
    """
    Parse time cell into seconds (float).

    Supports:
        - numeric seconds (int/float)
        - "hh:mm:ss.sss"
        - "mm:ss.sss"
        - commas as decimal separators
        - extra colon "hh:mm:ss:sss" -> "hh:mm:ss.sss"
    Returns None if not parseable.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if pd.isna(value):
            return None
        return float(value)

    s = str(value).strip()
    if s == "" or s.lower() == "nan":
        return None

    s = s.replace(",", ".")
    parts = [p for p in s.split(":") if p != ""]

    try:
        if len(parts) == 3:
            h, m, sec = parts
        elif len(parts) == 2:
            h, m, sec = "0", parts[0], parts[1]
        elif len(parts) == 4:
            h, m, sec = parts[0], parts[1], f"{parts[2]}.{parts[3]}"
        else:
            return None

        total = (float(h) * 3600.0) + (float(m) * 60.0) + float(sec)
        return total
    except Exception:
        return None


def build_gt_dict(
    df_gt: pd.DataFrame,
    *,
    id_column: str = "video_id",
    ann_id_column: str = "ann_id",
    start_column: str = "start_s",
    end_column: str = "end_s",
    label_column: str = "vocalization_type",
) -> Dict[str, List[dict]]:
    """
    Build gt_dict[audio_id] -> list of GT event dicts with canonical keys.

    Output event keys:
        gt_event_id, gt_start_s, gt_end_s, gt_label
    """
    missing = [c for c in [id_column, ann_id_column, start_column, end_column, label_column] if c not in df_gt.columns]
    if missing:
        raise ValueError(f"GT dataframe missing required columns: {missing}")

    gt_dict: Dict[str, List[dict]] = {}

    for _, row in df_gt.iterrows():
        audio_id = str(row[id_column]).strip()
        if not audio_id:
            continue

        gt_event_id = str(row[ann_id_column]).strip()
        gt_label = str(row[label_column]).strip()

        gt_start = parse_time_to_seconds(row[start_column])
        gt_end = parse_time_to_seconds(row[end_column])

        if gt_start is None or gt_end is None:
            continue
        if gt_end <= gt_start:
            continue

        ev = {
            "gt_event_id": gt_event_id,
            "gt_start_s": float(gt_start),
            "gt_end_s": float(gt_end),
            "gt_label": gt_label,
        }
        gt_dict.setdefault(audio_id, []).append(ev)

    return gt_dict