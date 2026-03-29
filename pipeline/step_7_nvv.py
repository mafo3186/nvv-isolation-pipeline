#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NVV Pipeline – Step 7: NVV Candidate Extraction (strict, file-driven)
---------------------------------------------------------------------
Detects NVV candidates within or without VAD segments.
• Keeps non_word / filler / oov / unknown as NVV candidates.
• Works with or without VAD-mask (vad_mask="no").

File-driven processing:
- Iterates existing NLP artifacts only (annotations/nlp/*_nlp.json).
- Parses (vad_mask, asr_audio_in) from filename via parse_sources_from_audio_id_filename().
- Supports vad_mask == "no" (no-VAD mode): evaluates NLP only (no VAD gating, no VAD gaps).

Strict error semantics:
- NLP files are processed only if they exist (file-driven).
- JSON broken/invalid => raise
- Valid-but-empty inputs => write empty NVV artifact {"nvv": []}
- force=False => cached skip (but validates existing NVV JSON is readable)

Outputs:
- annotations/nvv/<stem>_nvv.json with nvv-candidate-segments
  where <stem> mirrors NLP stem: <audio_id>_<vad_mask>_vad_<asr_audio_in>_asr_nlp
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from intervaltree import IntervalTree

from config.constants import (
    EXT_JSON,
    KEY_ANNOTATIONS,
    KEY_FIELD_PATH,
    KEY_NLP,
    KEY_NVV,
    KEY_STEP_7,
    KEY_VAD,
    VAD_MASKS,
    VAD_NO,
    AUDIO_DERIVATIVES,
)

from utils.io import (
    ensure_dir,
    read_json_with_status,
    write_json,
    audio_dir_metadata_path,
)
from utils.parsing import (
    parse_vad_and_asr_identifier_from_audio_id_filename,
    derive_combo_key,
)
from pipeline.pipeline_workspace_runner import setup_workspace_run
from metadata.metadata import mark_step
#toDo: use path factory

# Helpers (pure functions)

def _consolidate_duplicates(
    chunks: List[Dict[str, Any]],
    dedup_overlap_ratio: float,
    dedup_time_tol_s: float,
) -> List[Dict[str, Any]]:
    """
    Merge near-identical duplicates (same text, overlap >= dedup_overlap_ratio).

    Args:
        chunks: list of chunk dicts with start/end/text keys
        dedup_overlap_ratio: minimum overlap ratio to merge near-identical duplicates
        dedup_time_tol_s: time tolerance for near-identical start/end deduplication
    """
    if not chunks:
        return []
    chunks = sorted(chunks, key=lambda x: (x["start"], x["end"]))
    consolidated: List[Dict[str, Any]] = []
    for c in chunks:
        if not consolidated:
            consolidated.append(c)
            continue

        prev = consolidated[-1]
        same_text = str(c.get("text", "")).strip() == str(prev.get("text", "")).strip()
        overlap = min(c["end"], prev["end"]) - max(c["start"], prev["start"])
        dur = max(c["end"] - c["start"], prev["end"] - prev["start"])
        overlap_ratio = overlap / dur if dur > 0 else 0.0

        if same_text and (
            overlap_ratio >= dedup_overlap_ratio
            or (abs(c["start"] - prev["start"]) < dedup_time_tol_s and abs(c["end"] - prev["end"]) < dedup_time_tol_s)
        ):
            prev["start"] = min(prev["start"], c["start"])
            prev["end"] = max(prev["end"], c["end"])
        else:
            consolidated.append(c)

    return consolidated


def _filter_by_duration(
    segments: List[Dict[str, Any]],
    min_duration: float,
    max_duration: Optional[float],
) -> List[Dict[str, Any]]:
    """
    Filter segments by duration (seconds).
    """
    out: List[Dict[str, Any]] = []
    for seg in segments:
        dur = float(seg["end"]) - float(seg["start"])
        if dur < 0:
            continue
        if dur >= float(min_duration) and (max_duration is None or dur <= float(max_duration)):
            out.append(seg)
    return out


def _subtract_intervals(base: List[Dict[str, float]], sub: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    """
    Subtract intervals (sub) from base.
    base: [{"start": s, "end": e}] (seconds)
    sub:  objects containing "start"/"end" (seconds)
    """
    if not base:
        return []

    tree = IntervalTree.from_tuples((float(b["start"]), float(b["end"])) for b in base)
    for s in sub:
        ss = float(s["start"])
        ee = float(s["end"])
        if ee <= ss:
            continue
        tree.chop(ss, ee)

    return [{"start": iv.begin, "end": iv.end} for iv in sorted(tree)]


def _find_vad_gaps(
    vad_segments: List[Dict[str, float]],
    chunks_to_cover: List[Dict[str, Any]],
    min_duration: float,
) -> List[Dict[str, Any]]:
    """
    Identify gaps inside VAD base where chunks_to_cover produced no chunks.

    Note:
        This is used on vad_base (= VAD minus excluded categories).
        Therefore, chunks_to_cover should be consistent with your goal.
        If you want "VAD minus words, then investigate rest", pass kept (non-excluded) chunks.
    """
    if not vad_segments:
        return []

    chunk_tree = IntervalTree.from_tuples((float(c["start"]), float(c["end"])) for c in chunks_to_cover)
    gaps: List[Dict[str, Any]] = []

    for v in vad_segments:
        remaining = IntervalTree.from_tuples([(float(v["start"]), float(v["end"]))])
        for a in chunk_tree:
            remaining.chop(a.begin, a.end)
        for iv in sorted(remaining):
            if (iv.end - iv.begin) >= float(min_duration):
                gaps.append(
                    {
                        "start": iv.begin,
                        "end": iv.end,
                        "class": "nvv_candidate",
                        "source": "vad_gap",
                    }
                )

    return gaps


def _clip_segments_to_vad(
    segments: List[Dict[str, Any]],
    vad_segments: List[Dict[str, float]],
    padding: float,
) -> List[Dict[str, Any]]:
    """
    Clip segments to intersection with VAD segments (strict gate).
    Segment may be split if it overlaps multiple VAD intervals.
    """
    if not segments:
        return []
    if not vad_segments:
        return []
    if padding < 0.0:
        raise ValueError("padding must be >= 0.0")

    vads = sorted(vad_segments, key=lambda x: (x["start"], x["end"]))
    segs = sorted(segments, key=lambda x: (x["start"], x["end"]))

    out: List[Dict[str, Any]] = []

    for seg in segs:
        s0 = float(seg["start"])
        e0 = float(seg["end"])
        if e0 <= s0:
            continue

        for vad in vads:
            vs = float(vad["start"])
            ve = float(vad["end"])
            if ve <= vs:
                continue

            if e0 <= vs or s0 >= ve:
                continue

            start = max(s0, vs)
            end = min(e0, ve)

            if s0 < vs:
                start = max(s0, vs - padding)
            if e0 > ve:
                end = min(e0, ve + padding)

            if end <= start:
                continue

            clipped = dict(seg)
            clipped["start"] = start
            clipped["end"] = end
            out.append(clipped)

    return out


def _add_candidate_ids(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Add consecutive numeric IDs as 'candidate_id' starting at 1.
    """
    out: List[Dict[str, Any]] = []
    for i, seg in enumerate(segments, start=1):
        s = dict(seg)
        s["candidate_id"] = i
        out.append(s)
    return out


# IO + validation

def _load_vad_segments_required(meta: Dict[str, Any], vad_mask: str) -> List[Dict[str, float]]:
    """
    Resolve VAD path via metadata (mapping only), then load/validate VAD JSON.

    Args:
        meta: per-audio metadata.json content
        vad_mask: one of VAD_MASKS (must not be "no" here)

    Returns:
        List of {"start": float, "end": float} (may be empty)

    Raises:
        FileNotFoundError / ValueError on missing/broken/invalid VAD.
    """
    vad_entry = (meta.get(KEY_ANNOTATIONS, {}).get(KEY_VAD, {}) or {}).get(vad_mask) or {}
    vad_path_str = str(vad_entry.get(KEY_FIELD_PATH, "") or "")
    if not vad_path_str:
        raise FileNotFoundError(f"Missing VAD metadata entry for vad_mask='{vad_mask}' (no path).")

    vad_path = Path(vad_path_str)
    data, st = read_json_with_status(vad_path)
    if st == "missing":
        raise FileNotFoundError(f"Missing VAD file for vad_mask='{vad_mask}': {vad_path}")
    if st != "ok" or data is None:
        raise ValueError(f"Broken VAD JSON (status='{st}') at: {vad_path}")

    segments = data.get("segments", [])
    if segments is None or not isinstance(segments, list):
        raise ValueError(f"Invalid VAD JSON structure at: {vad_path} (segments must be list).")

    out: List[Dict[str, float]] = []
    for s in segments:
        if not isinstance(s, dict):
            raise ValueError(f"Invalid VAD segment type at: {vad_path}")
        start = s.get("start")
        end = s.get("end")
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            raise ValueError(f"Invalid VAD segment values at: {vad_path}")
        start_f = float(start)
        end_f = float(end)
        if end_f > start_f:
            out.append({"start": start_f, "end": end_f})

    return out


def _extract_chunks_from_nlp(nlp_path: Path) -> List[Dict[str, Any]]:
    """
    Flatten chunks from NLP JSON (timestamps in seconds).

    Returns:
        List of chunk dicts (may be empty).

    Raises:
        ValueError on broken JSON or invalid structure.
    """
    data, st = read_json_with_status(nlp_path)
    if st != "ok" or data is None:
        raise ValueError(f"Broken NLP JSON (status='{st}') at: {nlp_path}")

    segments = data.get("segments", [])
    if segments is None or not isinstance(segments, list):
        raise ValueError(f"Invalid NLP structure at: {nlp_path} (segments must be list).")

    chunks_out: List[Dict[str, Any]] = []

    for seg in segments:
        if not isinstance(seg, dict):
            raise ValueError(f"Invalid NLP segment type at: {nlp_path}")

        seg_chunks = seg.get("chunks", [])
        if seg_chunks is None:
            seg_chunks = []
        if not isinstance(seg_chunks, list):
            raise ValueError(f"Invalid NLP segment.chunks type at: {nlp_path}")

        for ch in seg_chunks:
            if not isinstance(ch, dict):
                raise ValueError(f"Invalid NLP chunk type at: {nlp_path}")

            ts = ch.get("timestamp", None)
            if ts is None:
                continue
            if not isinstance(ts, (list, tuple)) or len(ts) != 2:
                raise ValueError(f"Invalid NLP chunk.timestamp at: {nlp_path}")

            start, end = ts
            if start is None or end is None:
                continue
            if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
                raise ValueError(f"Invalid NLP chunk.timestamp values at: {nlp_path}")

            s = float(start)
            e = float(end)
            if e <= s:
                continue

            chunks_out.append(
                {
                    "start": s,
                    "end": e,
                    "text": str(ch.get("text", "")).strip(),
                    "category": str(ch.get("category", "unknown")),
                    "source": KEY_NLP,
                }
            )

    return chunks_out

# --- Single file processing ---

def process_single_nvv(
    *,
    nlp_path: Path,
    audio_id_dir: Path,
    audio_id: str,
    exclude_categories: List[str],
    min_duration: float,
    max_duration: Optional[float],
    vad_gate_padding: float,
    dedup_overlap_ratio: float,
    dedup_time_tol_s: float,
    force: bool,
) -> None:
    """
    Process exactly one NLP artifact into one NVV artifact.

    Args:
        nlp_path: Path to *_nlp.json
        audio_id_dir: per_audio/<audio_id> directory
        audio_id: audio_id_dir.name
        exclude_categories: categories to exclude from analysis (e.g. ["word"])
        min_duration: minimum NVV duration in seconds
        max_duration: optional maximum NVV duration in seconds
        vad_gate_padding: optional padding when clipping kept segments to VAD
        dedup_overlap_ratio: minimum overlap ratio to merge near-identical duplicate chunks
        dedup_time_tol_s: time tolerance for near-identical start/end deduplication
        force: overwrite existing NVV artifacts
    """
    vad_mask, asr_audio_in = parse_vad_and_asr_identifier_from_audio_id_filename(audio_id, nlp_path.stem)

    if vad_mask not in VAD_MASKS:
        raise ValueError(f"Invalid vad_mask parsed from NLP filename: '{vad_mask}' ({nlp_path.name})")
    if asr_audio_in not in AUDIO_DERIVATIVES:
        raise ValueError(f"Invalid asr_audio_in parsed from NLP filename: '{asr_audio_in}' ({nlp_path.name})")

    meta_path = audio_dir_metadata_path(audio_id_dir)
    meta, st = read_json_with_status(meta_path)
    if st != "ok" or meta is None:
        raise RuntimeError(f"Cannot read metadata at: {meta_path}")

    out_dir = audio_id_dir / KEY_ANNOTATIONS / KEY_NVV
    ensure_dir(out_dir)

    stem = nlp_path.stem
    out_path = (out_dir / f"{stem}_{KEY_NVV}{EXT_JSON}").resolve()
    combo_key = derive_combo_key(stem, audio_id)

    # Cached reuse
    if out_path.exists() and not force:
        existing, st_out = read_json_with_status(out_path)
        if st_out != "ok" or existing is None:
            raise ValueError(f"Broken existing NVV JSON at: {out_path}")

        segs = existing.get(KEY_NVV, [])
        if segs is None:
            segs = []
        if not isinstance(segs, list):
            raise ValueError(f"Invalid existing NVV JSON structure at: {out_path} (nvv must be list).")

        meta.setdefault(KEY_ANNOTATIONS, {}).setdefault(KEY_NVV, {})
        meta[KEY_ANNOTATIONS][KEY_NVV][combo_key] = {
            KEY_FIELD_PATH: str(out_path),
            "vad_mask": vad_mask,
            "asr_audio_in": asr_audio_in,
            "exclude_categories": list(exclude_categories),
            "min_duration": float(min_duration),
            "max_duration": None if max_duration is None else float(max_duration),
            "vad_gate_padding": float(vad_gate_padding),
            "segments": len(segs),
            "status": "ok",
        }
        write_json(meta_path, meta)
        return

    # Load NLP (always required)
    all_chunks = _extract_chunks_from_nlp(nlp_path)

    # Valid empty NLP => valid empty NVV
    if len(all_chunks) == 0:
        write_json(out_path, {KEY_NVV: []})
        meta.setdefault(KEY_ANNOTATIONS, {}).setdefault(KEY_NVV, {})
        meta[KEY_ANNOTATIONS][KEY_NVV][combo_key] = {
            KEY_FIELD_PATH: str(out_path),
            "vad_mask": vad_mask,
            "asr_audio_in": asr_audio_in,
            "exclude_categories": list(exclude_categories),
            "min_duration": float(min_duration),
            "max_duration": None if max_duration is None else float(max_duration),
            "vad_gate_padding": float(vad_gate_padding),
            "segments": 0,
            "status": "ok",
        }
        write_json(meta_path, meta)
        return

    all_chunks = _consolidate_duplicates(all_chunks, dedup_overlap_ratio=dedup_overlap_ratio, dedup_time_tol_s=dedup_time_tol_s)

    excl_set = set(exclude_categories)
    excluded = [c for c in all_chunks if c.get("category") in excl_set]
    kept = [c for c in all_chunks if c.get("category") not in excl_set]

    # no-VAD mode: evaluate NLP only (no clipping, no gaps)
    if vad_mask == VAD_NO:
        nvv_segments = _filter_by_duration(kept, min_duration=min_duration, max_duration=max_duration)
        nvv_segments = sorted(nvv_segments, key=lambda x: (x["start"], x["end"]))
        nvv_segments = _add_candidate_ids(nvv_segments)

        write_json(out_path, {KEY_NVV: nvv_segments})

        meta.setdefault(KEY_ANNOTATIONS, {}).setdefault(KEY_NVV, {})
        meta[KEY_ANNOTATIONS][KEY_NVV][combo_key] = {
            KEY_FIELD_PATH: str(out_path),
            "vad_mask": vad_mask,
            "asr_audio_in": asr_audio_in,
            "exclude_categories": list(exclude_categories),
            "min_duration": float(min_duration),
            "max_duration": None if max_duration is None else float(max_duration),
            "vad_gate_padding": float(vad_gate_padding),
            "segments": len(nvv_segments),
            "status": "ok",
        }
        write_json(meta_path, meta)
        return

    # VAD-required mode (strict)
    vad_segments = _load_vad_segments_required(meta, vad_mask=vad_mask)

    # Valid empty VAD => valid empty NVV (VAD gates candidates)
    if len(vad_segments) == 0:
        write_json(out_path, {KEY_NVV: []})
        meta.setdefault(KEY_ANNOTATIONS, {}).setdefault(KEY_NVV, {})
        meta[KEY_ANNOTATIONS][KEY_NVV][combo_key] = {
            KEY_FIELD_PATH: str(out_path),
            "vad_mask": vad_mask,
            "asr_audio_in": asr_audio_in,
            "exclude_categories": list(exclude_categories),
            "min_duration": float(min_duration),
            "max_duration": None if max_duration is None else float(max_duration),
            "vad_gate_padding": float(vad_gate_padding),
            "segments": 0,
            "status": "ok",
        }
        write_json(meta_path, meta)
        return

    # Strict gate: keep only what overlaps with VAD (optionally padded)
    kept_clipped = _clip_segments_to_vad(kept, vad_segments=vad_segments, padding=vad_gate_padding)

    # Remove excluded speech (e.g., words) from VAD base
    vad_base = _subtract_intervals(vad_segments, excluded) if excluded else vad_segments

    # Find gaps inside reduced VAD base where there are no kept (non-excluded) chunks
    vad_gaps = _find_vad_gaps(vad_base, chunks_to_cover=kept_clipped, min_duration=min_duration)

    nvv_segments = kept_clipped + vad_gaps
    nvv_segments = _filter_by_duration(nvv_segments, min_duration=min_duration, max_duration=max_duration)
    nvv_segments = sorted(nvv_segments, key=lambda x: (x["start"], x["end"]))
    nvv_segments = _add_candidate_ids(nvv_segments)

    write_json(out_path, {KEY_NVV: nvv_segments})

    meta.setdefault(KEY_ANNOTATIONS, {}).setdefault(KEY_NVV, {})
    meta[KEY_ANNOTATIONS][KEY_NVV][combo_key] = {
        KEY_FIELD_PATH: str(out_path),
        "vad_mask": vad_mask,
        "asr_audio_in": asr_audio_in,
        "exclude_categories": list(exclude_categories),
        "min_duration": float(min_duration),
        "max_duration": None if max_duration is None else float(max_duration),
        "vad_gate_padding": float(vad_gate_padding),
        "segments": len(nvv_segments),
        "status": "ok",
    }
    write_json(meta_path, meta)
    print(f"✓ Step 7: Saved NVV segments for VAD-mask {vad_mask} / ASR-Audio Input {asr_audio_in} →  at: {out_path}")
    print(f"NVV-segments: {len(nvv_segments)}. From NLP: {len(all_chunks)} (excluded: {len(excluded)}, kept: {len(kept)}) "
)

# --- Workspace runner ---

def run_step_7_nvv(
    workspace: Path | str,
    *,
    exclude_categories: List[str],
    min_duration: float,
    max_duration: Optional[float],
    vad_masks_in: List[str],
    asr_audios_in: List[str],
    vad_gate_padding: float,
    dedup_overlap_ratio: float,
    dedup_time_tol_s: float,
    force: bool = False,
) -> None:
    """
    Workspace runner for Step 7.

    Args:
        workspace: processed workspace root
        exclude_categories: categories excluded from analysis (e.g. ["word"])
        min_duration: minimum NVV duration (seconds)
        max_duration: optional maximum NVV duration (seconds)
        vad_masks_in: filter list for vad masks (must be in VAD_MASKS; can include "no")
        asr_audios_in: filter list for ASR input audios (must be in AUDIO_DERIVATIVES)
        vad_gate_padding: optional padding when clipping kept segments to VAD
        dedup_overlap_ratio: minimum overlap ratio to merge near-identical duplicate chunks
        dedup_time_tol_s: time tolerance for near-identical start/end deduplication
        force: overwrite existing NVV outputs
    """
    setup = setup_workspace_run(
        workspace=workspace,
        input_dir=None,
        device="cpu",  # not used here, but required by runner
        force=force,
        require_metadata=True,
    )

    audio_id_dirs = setup["audio_id_dirs"]
    if not audio_id_dirs:
        print("No per_audio folders with metadata found. Run previous steps first.")
        return

    # Validate filters (fail fast)
    for v in vad_masks_in:
        if v not in VAD_MASKS:
            raise ValueError(f"Invalid vad_mask '{v}' in run_step_7_nvv(). Must be in VAD_MASKS.")
    for a in asr_audios_in:
        if a not in AUDIO_DERIVATIVES:
            raise ValueError(f"Invalid asr_audio_in '{a}' in run_step_7_nvv(). Must be in AUDIO_DERIVATIVES.")

    print(f"Step 7 (NVV) – workspace='{Path(workspace).name}'")
    print(f"  vad_masks_in: {vad_masks_in}")
    print(f"  asr_audios_in: {asr_audios_in}")
    print(f"  exclude_categories: {exclude_categories}")
    print(f"  min_duration: {min_duration}")
    print(f"  max_duration: {max_duration}")
    print(f"  vad_gate_padding: {vad_gate_padding}")
    print(f"  force: {force}")

    for audio_id_dir in audio_id_dirs:
        t0 = time.time()
        audio_id = audio_id_dir.name

        nlp_dir = audio_id_dir / KEY_ANNOTATIONS / KEY_NLP
        if not nlp_dir.exists():
            continue

        nlp_files = sorted([p for p in nlp_dir.glob(f"*_{KEY_NLP}{EXT_JSON}") if "_log" not in p.name])
        if not nlp_files:
            continue

        processed = 0

        for nlp_path in nlp_files:
            vad_mask, asr_audio_in = parse_vad_and_asr_identifier_from_audio_id_filename(audio_id, nlp_path.stem)

            if vad_mask not in vad_masks_in:
                continue
            if asr_audio_in not in asr_audios_in:
                continue

            processed += 1
            process_single_nvv(
                nlp_path=nlp_path,
                audio_id_dir=audio_id_dir,
                audio_id=audio_id,
                exclude_categories=exclude_categories,
                min_duration=min_duration,
                max_duration=max_duration,
                vad_gate_padding=vad_gate_padding,
                dedup_overlap_ratio=dedup_overlap_ratio,
                dedup_time_tol_s=dedup_time_tol_s,
                force=force,
            )

        # Mark step done once per audio_id_dir
        meta_path = audio_dir_metadata_path(audio_id_dir)
        meta, st = read_json_with_status(meta_path)
        if st != "ok" or meta is None:
            raise RuntimeError(f"Cannot re-read metadata at {meta_path}")

        mark_step(
            meta,
            KEY_STEP_7,
            "done",
            t0,
            {
                "processed_files": processed,
                "vad_masks_in": list(vad_masks_in),
                "asr_audios_in": list(asr_audios_in),
                "exclude_categories": list(exclude_categories),
                "min_duration": float(min_duration),
                "max_duration": None if max_duration is None else float(max_duration),
                "vad_gate_padding": float(vad_gate_padding),
            },
        )
        write_json(meta_path, meta)

    print("✅ Step 7 completed.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Step 7: NVV Candidate Extraction")    
    parser.add_argument("workspace", type=str, help="Path to processed workspace root.")
    parser.add_argument("--exclude_categories", nargs="*", default=["word"], help="Categories to exclude from NVV analysis (default: ['word'])")
    parser.add_argument("--min_duration", type=float, default=0.2, help="Minimum NVV segment duration in seconds (default: 0.2)")
    parser.add_argument("--max_duration", type=float, default=None, help="Optional maximum NVV segment duration in seconds (default: None)")
    parser.add_argument("--vad_masks_in", nargs="*", default=VAD_MASKS, help=f"Filter list for VAD masks (default: all in VAD_MASKS={VAD_MASKS})")      
    parser.add_argument("--asr_audios_in", nargs="*", default=AUDIO_DERIVATIVES, help=f"Filter list for ASR input audios (default: all in AUDIO_DERIVATIVES={AUDIO_DERIVATIVES})")
    parser.add_argument("--vad_gate_padding", type=float, default=0.0, help="Optional padding in seconds when clipping to VAD (default: 0.0)")
    parser.add_argument("--dedup_overlap_ratio", type=float, default=0.7, help="Minimum overlap ratio to merge near-identical duplicate chunks (default: 0.7)")
    parser.add_argument("--dedup_time_tol_s", type=float, default=0.05, help="Time tolerance for near-identical start/end deduplication (default: 0.05)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing NVV outputs (default: False)")
    args = parser.parse_args()  
    run_step_7_nvv(
        workspace=args.workspace,
        exclude_categories=args.exclude_categories,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        vad_masks_in=args.vad_masks_in,
        asr_audios_in=args.asr_audios_in,
        vad_gate_padding=args.vad_gate_padding,
        dedup_overlap_ratio=args.dedup_overlap_ratio,
        dedup_time_tol_s=args.dedup_time_tol_s,
        force=args.force,
    )