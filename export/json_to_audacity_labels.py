#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions to convert annotation JSONs (ASR or VAD)
to Audacity-compatible label text files (.txt).
"""

import json
from pathlib import Path
from utils.io import ensure_dir, read_json_with_status


# --- ASR JSON → Audacity Labels ---
def json_asr_to_audacity_labels(json_path: Path, label_path: Path):
    """
    Converts ASR transcript JSON (segmented or flattened)
    into Audacity label format (.txt).

    Works with both legacy and current structures:
      {
        "sampling_rate": 44100,
        "segments": [...]
      }
    or plain lists/dicts of segments.

    Keeps None timestamps wherever possible, but substitutes
    with reasonable VAD/segment-based fallbacks when available.

    Fallback strategy:
      • If both start and end are None → try seg_start/seg_end
      • If start missing:
            - first chunk → seg_start
            - later chunks → previous end
      • If end missing → seg_end if available, else = start
      • Ensures monotonic time order (no backwards overlaps)

    Semantics:
      - Missing/broken JSON => do NOT write a label file
      - Valid-but-empty inputs => write an empty label file
    """
    data, status = read_json_with_status(json_path)
    if status == "missing":
        print(f"⚠️ Missing ASR JSON: {json_path.name} -> no label written")
        return
    if status == "error" or data is None:
        print(f"⚠️ Broken ASR JSON: {json_path.name} -> no label written")
        return

    # unwrap new structure if present
    if isinstance(data, dict) and "segments" in data:
        data = data["segments"]

    lines = []

    # Case 1: flattened ASR (single dict with "chunks")
    if isinstance(data, dict) and "chunks" in data:
        chunks = data.get("chunks", [])
        if chunks is None:
            print(f"⚠️ Broken ASR JSON structure in {json_path.name} (chunks is None) -> no label written")
            return
        if not isinstance(chunks, list):
            print(f"⚠️ Broken ASR JSON structure in {json_path.name} (chunks not a list) -> no label written")
            return

        last_end = 0.0
        for i, ch in enumerate(chunks):
            if not isinstance(ch, dict):
                print(f"⚠️ Broken ASR JSON structure in {json_path.name} (chunk not a dict) -> no label written")
                return

            text = str(ch.get("text", "")).strip().replace("\n", " ")
            ts = ch.get("timestamp")
            start, end = (None, None)

            if ts and isinstance(ts, (list, tuple)):
                start = ts[0]
                end = ts[1] if len(ts) > 1 else None

            # handle missing timestamps gracefully
            if start is None and end is None:
                continue
            elif start is None:
                start = last_end
                text += " [NO_START_PREV]"
            elif end is None:
                end = start
                text += " [NO_END]"

            if end < start:
                end = start
                text += " [FIXED_ORDER]"

            lines.append(f"{float(start):.3f}\t{float(end):.3f}\t{text}")
            last_end = float(end)

    # Case 2: segmented ASR (with VAD)
    elif isinstance(data, list):
        for seg in data:
            if not isinstance(seg, dict):
                print(f"⚠️ Broken ASR JSON structure in {json_path.name} (segment not a dict) -> no label written")
                return

            seg_start = seg.get("segment_start", 0.0)
            seg_end = seg.get("segment_end", None)
            chunks = seg.get("chunks", [])

            if chunks is None:
                print(f"⚠️ Broken ASR JSON structure in {json_path.name} (chunks is None) -> no label written")
                return
            if not isinstance(chunks, list):
                print(f"⚠️ Broken ASR JSON structure in {json_path.name} (chunks not a list) -> no label written")
                return

            last_end = float(seg_start) if isinstance(seg_start, (int, float)) else 0.0

            for i, ch in enumerate(chunks):
                if not isinstance(ch, dict):
                    print(f"⚠️ Broken ASR JSON structure in {json_path.name} (chunk not a dict) -> no label written")
                    return

                text = str(ch.get("text", "")).strip().replace("\n", " ")
                ts = ch.get("timestamp")
                start, end = (None, None)

                if ts and isinstance(ts, (list, tuple)):
                    start = ts[0]
                    end = ts[1] if len(ts) > 1 else None

                # --- Fallbacks ---
                if start is None and end is None:
                    if seg_start is not None and seg_end is not None:
                        start, end = seg_start, seg_end
                        text += " [SEGMENT_FALLBACK]"
                    else:
                        continue
                if start is None:
                    if i == 0:
                        start = seg_start
                        text += " [NO_START_SEG]"
                    else:
                        start = last_end
                        text += " [NO_START_PREV]"
                if end is None:
                    if seg_end is not None:
                        end = seg_end
                        text += " [NO_END_SEG]"
                    else:
                        end = start
                        text += " [NO_END_FALLBACK]"

                if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
                    print(f"⚠️ Broken ASR JSON structure in {json_path.name} (non-numeric timestamps) -> no label written")
                    return

                start = float(start)
                end = float(end)

                if end < start:
                    end = start
                    text += " [FIXED_ORDER]"

                lines.append(f"{start:.3f}\t{end:.3f}\t{text}")
                last_end = end

    else:
        print(f"⚠️ Unknown ASR JSON structure in {json_path.name} -> no label written")
        return

    if not lines:
        print(f"⚠️ No valid ASR timestamps found in {json_path.name} -> writing empty label file")

    ensure_dir(label_path.parent)
    with open(label_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"✓ Exported ASR labels → {label_path.name}, {len(lines)} segments") 

# --- VAD JSON → Audacity Labels (seconds-based) ---
def json_vad_to_audacity_labels(json_path: Path, label_path: Path):
    """
    Converts a VAD JSON (with segments in seconds) to Audacity label format.

    Expected structure:
        {
          "segments": [
            {"start": 0.054, "end": 0.893},
            {"start": 1.304, "end": 2.742}
          ]
        }

    Semantics:
      - Missing/broken JSON => do NOT write a label file
      - Valid-but-empty inputs => write an empty label file
    """
    data, status = read_json_with_status(json_path)
    if status == "missing":
        print(f"⚠️ Missing VAD JSON: {json_path.name} -> no label written")
        return
    if status == "error" or data is None:
        print(f"⚠️ Broken VAD JSON: {json_path.name} -> no label written")
        return

    if not isinstance(data, dict):
        print(f"⚠️ Broken VAD JSON structure in {json_path.name} (expected dict) -> no label written")
        return

    segments = data.get("segments", [])
    if segments is None or not isinstance(segments, list):
        print(f"⚠️ Broken VAD JSON structure in {json_path.name} (segments must be a list) -> no label written")
        return

    if not segments:
        print(f"⚠️ No segments found in {json_path.name} -> writing empty label file")

    lines = []
    for i, seg in enumerate(segments, start=1):
        if not isinstance(seg, dict):
            print(f"⚠️ Broken VAD JSON structure in {json_path.name} (segment not a dict) -> no label written")
            return

        start = seg.get("start")
        end = seg.get("end")
        if start is not None and end is not None:
            # values are already in seconds, no division by SR
            try:
                lines.append(f"{float(start):.3f}\t{float(end):.3f}\tvoice_{i:02d}")
            except (TypeError, ValueError):
                print(f"⚠️ Broken VAD JSON timestamps in {json_path.name} -> no label written")
                return

    if not lines and segments:
        print(f"⚠️ No valid VAD timestamps found in {json_path.name} -> writing empty label file")

    ensure_dir(label_path.parent)
    with open(label_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"✓ Exported VAD labels → {label_path.name}, {len(lines)} segments")


# --- NVV candidates JSON → Audacity Labels ---

def json_nvv_to_audacity_labels(nvv_segments, out_path):
    """
    Export NVV candidates (start/end in seconds) to Audacity label file.
    Includes category and source if available.
    """
    lines = []
    for seg in nvv_segments:
        start = seg.get("start")
        end = seg.get("end")
        text = seg.get("text", "")
        label = seg.get("category", "nvv")
        src = seg.get("source", "")
        if src:
            label = f"{label} ({src})"
        if text:
            label = f"{label}: {text}"
        if start is None or end is None:
            continue
        lines.append(f"{start:.3f}\t{end:.3f}\t{label}")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"✓ Exported NVV candidates → {out_path.name}, {len(lines)} segments")