#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract clips from pipeline artifacts (file-scan based JSON selection).

JSON sources:
- mode="nvv"   -> per_audio/<audio_id>/annotations/nvv/*_nvv.json
- mode="words" -> per_audio/<audio_id>/annotations/nlp/*_nlp.json (extract category=="word" chunks)

Audio resolution (metadata-based, pragmatic):
- Prefer meta["audios"][asr_audio_in]["path"] for ALL asr_audio_in tokens, including "original"
- Backward compatible fallback for original: meta["file"]["path"]

Outputs:
- workspace/global/clips/<mode>/<json_stem>_<mode>_<i:03d>.wav
- with optional sub_dir:
  workspace/global/clips/<sub_dir>/<mode>/<json_stem>_<mode>_<i:03d>.wav
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import soundfile as sf

from config.constants import (
    KEY_ANNOTATIONS,
    KEY_AUDIO_FILES,
    KEY_CLIPS,
    KEY_FILE,
    KEY_GLOBAL,
    KEY_NLP,
    KEY_NVV,
    KEY_PER_AUDIO,
)

from utils.io import (
    ensure_dir,
    is_audio_id_dir,
    audio_dir_metadata_path,
    read_json,
    read_json_with_status,
)
from utils.parsing import (
    parse_vad_and_asr_identifier_from_audio_id_filename,
)


def load_audio(audio_path: Path) -> Tuple[Any, int]:
    """Load audio file and return (signal, sr)."""
    if not audio_path.exists():
        raise FileNotFoundError(str(audio_path))
    data, sr = sf.read(str(audio_path), always_2d=False)
    return data, sr


def save_clip(y: Any, sr: int, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), y, sr, subtype="PCM_16")


def _resolve_audio_path(meta: Dict[str, Any], asr_audio_in: str, audio_id: str) -> Optional[Path]:
    """
    Resolve the audio path for clipping based on asr_audio_in token.

    Priority:
    1) meta["audios"][asr_audio_in]["path"]  (works also for "original" in your newer metadata)
    2) if asr_audio_in == "original": meta["file"]["path"] (backward compat)

    Returns None if not resolvable or file missing.
    """
    p = (meta.get(KEY_AUDIO_FILES, {}) or {}).get(asr_audio_in, {}).get("path")
    if p:
        audio_path = Path(p)
        if audio_path.exists():
            return audio_path
        print(f"⚠️ Audio path missing on disk for {audio_id} ({asr_audio_in}): {audio_path}")
        return None

    if asr_audio_in == "original":
        p2 = (meta.get(KEY_FILE, {}) or {}).get("path")
        if p2:
            audio_path = Path(p2)
            if audio_path.exists():
                return audio_path
            print(f"⚠️ Original fallback path missing on disk for {audio_id}: {audio_path}")
            return None

    print(f"⚠️ Missing audio path in metadata for {audio_id}: asr_audio_in='{asr_audio_in}'")
    return None


def _extract_segments_from_nvv(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract NVV segments list from NVV JSON dict."""
    segs = data.get(KEY_NVV, [])
    return segs if isinstance(segs, list) else []


def _extract_segments_from_nlp_words(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract word chunks from NLP JSON dict.
    Uses category=="word" and timestamp=[start,end].
    """
    out: List[Dict[str, Any]] = []
    segments = data.get("segments", []) or []
    if not isinstance(segments, list):
        return out

    for seg in segments:
        if not isinstance(seg, dict):
            continue
        chunks = seg.get("chunks", []) or []
        if not isinstance(chunks, list):
            continue

        for ch in chunks:
            if not isinstance(ch, dict):
                continue
            if ch.get("category") != "word":
                continue
            ts = ch.get("timestamp")
            if not ts or not isinstance(ts, (list, tuple)) or len(ts) != 2:
                continue
            start, end = ts
            out.append({"start": start, "end": end, "text": ch.get("text", "")})

    return out


def _export_clips_from_segments(
    *,
    y: Any,
    sr: int,
    segments: List[Dict[str, Any]],
    clips_root: Path,
    base_name: str,
    mode: str,
    jf_name: str,
    force: bool,
) -> int:
    written = 0

    for i, seg in enumerate(segments, start=1):
        cand_id = seg.get("candidate_id")
        if isinstance(cand_id, int):
            index = cand_id
        else:
            index= i
        start, end = seg.get("start"), seg.get("end")
        if start is None or end is None:
            print(f"⚠️ Invalid timestamp in {jf_name}: {seg}")
            continue

        try:
            start_f = float(start)
            end_f = float(end)
        except Exception:
            print(f"⚠️ Non-numeric timestamp in {jf_name}: {seg}")
            continue

        start_samp = int(start_f * sr)
        end_samp = int(end_f * sr)
        if end_samp <= start_samp:
            continue

        out_path = clips_root / f"{base_name}_{mode}_{index:03d}.wav"
        if out_path.exists() and not force:
            continue

        clip = y[start_samp:end_samp]
        if clip is None or len(clip) == 0:
            print(f"⚠️ Skip empty clip extracted for {jf_name}: {seg}")
            continue
        save_clip(clip, sr, out_path)
        written += 1

    return written


def export_clips(
    workspace: Path | str,
    *,
    mode: str = "nvv",
    vad_masks: Optional[List[str]] = None,
    asr_audio_ins: Optional[List[str]] = None,
    sub_dir: Optional[str] = None,
    force: bool = False,
) -> None:
    """
    Extract clips for all matching JSON artifacts in workspace.

    Args:
        workspace: workspace root containing per_audio/ and global/
        mode: "nvv" or "words"
        vad_masks: optional filter list for vad_mask token
        asr_audio_ins: optional filter list for asr_audio_in token
        sub_dir: optional sub directory under global/clips/
        force: overwrite existing clips
    """
    ws = Path(workspace).resolve()
    per_audio = ws / KEY_PER_AUDIO
    if not per_audio.exists():
        print(f"❌ per_audio not found: {per_audio}")
        return

    if sub_dir:
        clips_root = ws / KEY_GLOBAL / KEY_CLIPS / sub_dir / mode
    else:
        clips_root = ws / KEY_GLOBAL / KEY_CLIPS / mode
    ensure_dir(clips_root)

    audio_dirs = sorted([p for p in per_audio.iterdir() if p.is_dir()])
    if not audio_dirs:
        print(f"⚠️ No audio_id folders found in: {per_audio}")
        return

    total_json = 0
    total_clips = 0
    skipped_no_meta = 0
    skipped_no_audio = 0

    for audio_id_dir in audio_dirs:
        if not is_audio_id_dir(audio_id_dir):
            continue

        audio_id = audio_id_dir.name
        meta_path = audio_dir_metadata_path(audio_id_dir)
        meta, st = read_json_with_status(meta_path)
        if st != "ok" or meta is None:
            skipped_no_meta += 1
            continue

        ann_root = audio_id_dir / KEY_ANNOTATIONS
        if not ann_root.exists():
            continue

        if mode == "nvv":
            ann_dir = ann_root / KEY_NVV
            pattern = f"*_{KEY_NVV}.json"
        elif mode == "words":
            ann_dir = ann_root / KEY_NLP
            pattern = f"*_{KEY_NLP}.json"
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if not ann_dir.exists():
            continue

        json_files = sorted([p for p in ann_dir.glob(pattern) if "_log" not in p.name])
        if not json_files:
            continue

        for jf in json_files:
            try:
                vad_mask, asr_audio_in = parse_vad_and_asr_identifier_from_audio_id_filename(audio_id, jf.stem)
            except Exception:
                continue

            if vad_masks is not None and vad_mask not in vad_masks:
                continue
            if asr_audio_ins is not None and asr_audio_in not in asr_audio_ins:
                continue

            audio_path = _resolve_audio_path(meta, asr_audio_in, audio_id)
            if not audio_path:
                skipped_no_audio += 1
                continue

            data = read_json(jf) or {}
            if mode == "nvv":
                segments = _extract_segments_from_nvv(data)
            else:
                segments = _extract_segments_from_nlp_words(data)

            if not segments:
                continue

            y, sr = load_audio(audio_path)
            written = _export_clips_from_segments(
                y=y,
                sr=sr,
                segments=segments,
                clips_root=clips_root,
                base_name=jf.stem,
                mode=mode,
                jf_name=jf.name,
                force=force,
            )

            if written > 0:
                total_json += 1
                total_clips += written

    print(
        f"✅ export_clips done for workspace='{ws.name}'. "
        f"mode={mode}, processed_json={total_json}, written_clips={total_clips}, "
        f"skipped_no_meta={skipped_no_meta}, skipped_no_audio={skipped_no_audio}"
    )