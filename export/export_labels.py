#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export Audacity label files from pipeline artifacts (file-scan based).

Supports:
- VAD labels: per_audio/<audio_id>/annotations/vad/*_vad.json  -> per_audio/<audio_id>/labels/vad/*.txt
- ASR labels: per_audio/<audio_id>/annotations/asr/*_asr*.json -> per_audio/<audio_id>/labels/asr/*.txt
- NVV labels: per_audio/<audio_id>/annotations/nvv/*_nvv.json  -> per_audio/<audio_id>/labels/nvv/*.txt

Semantics (pragmatic export tool):
- Missing/broken JSON => skip (continue)
- Valid-but-empty JSON => write empty label file
- Metadata update is OPTIONAL: only if <audio_id>_metadata.json exists.

Filters (optional):
- vad_masks: list[str]    (applied to VAD/ASR/NVV via filename tokens)
- asr_audio_ins: list[str] (applied to ASR/NVV via filename tokens)

Note:
- For ASR/NVV token parsing, we reuse parse_sources_from_audio_id_filename().
- VAD filenames do not include ASR token; we parse vad_mask with a small helper.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config.constants import (
    EXT_TXT,
    KEY_ANNOTATIONS,
    KEY_ASR,
    KEY_LABELS,
    KEY_METADATA,
    KEY_NVV,
    KEY_PER_AUDIO,
    KEY_VAD,
)

from utils.io import (
    ensure_dir,
    read_json,
    read_json_with_status,
    write_json,
    audio_dir_metadata_path,
    is_audio_id_dir,
    to_relative_path,
)
from utils.parsing import (
    parse_vad_and_asr_identifier_from_audio_id_filename,
)

from export.json_to_audacity_labels import (
    json_asr_to_audacity_labels,
    json_vad_to_audacity_labels,
    json_nvv_to_audacity_labels,
)


def _parse_vad_mask_from_vad_filename(audio_id: str, filename: str) -> str:
    """
    Parse vad_mask from VAD json filename:
        <audio_id>_<vad_mask>_vad.json
    """
    name = filename
    if name.endswith(".json"):
        name = name[:-5]

    prefix = f"{audio_id}_"
    suffix = "_vad"
    if not name.startswith(prefix) or not name.endswith(suffix):
        raise ValueError(f"Cannot parse vad_mask from VAD filename: {filename}")

    return name[len(prefix):-len(suffix)]


def _should_export_combo(
    *,
    vad_mask: Optional[str],
    asr_audio_in: Optional[str],
    vad_masks: Optional[List[str]],
    asr_audio_ins: Optional[List[str]],
) -> bool:
    """Return True if the (vad_mask, asr_audio_in) matches optional filters."""
    if vad_masks is not None and vad_mask is not None and vad_mask not in vad_masks:
        return False
    if asr_audio_ins is not None and asr_audio_in is not None and asr_audio_in not in asr_audio_ins:
        return False
    return True


def _update_metadata_with_label_if_present(
    *,
    audio_id_dir: Path,
    label_path: Path,
    source: str,
    generated_from: Path,
    vad_mask: Optional[str],
    asr_audio_in: Optional[str],
    project_root: Path,
) -> None:
    """
    Optional metadata update:
    - if <audio_id>_metadata.json exists, add entry under meta["labels"][source][label_filename]
    - does nothing if metadata is missing

    Args:
        audio_id_dir: per_audio/<audio_id>
        label_path: written label file (.txt)
        source: one of {"vad","asr","nvv"}
        generated_from: JSON artifact that produced the label
        vad_mask: token parsed from filename
        asr_audio_in: token parsed from filename (ASR/NVV); None for VAD
        project_root: Configured project root for relative path storage.
    """
    meta_path = audio_dir_metadata_path(audio_id_dir)
    if not meta_path.exists():
        return

    meta = read_json(meta_path) or {}
    meta.setdefault(KEY_LABELS, {}).setdefault(source, {})

    entry: Dict[str, Any] = {
        "path": to_relative_path(label_path, project_root),
        "generated_from": to_relative_path(generated_from, project_root),
    }
    if vad_mask is not None:
        entry["vad_mask"] = vad_mask
    if asr_audio_in is not None:
        entry["asr_audio_in"] = asr_audio_in

    meta[KEY_LABELS][source][label_path.name] = entry
    write_json(meta_path, meta)


def export_labels(
    workspace: Path | str,
    project_root: Path | str,
    *,
    export_vad: bool = True,
    export_asr: bool = True,
    export_nvv: bool = True,
    vad_masks: Optional[List[str]] = None,
    asr_audio_ins: Optional[List[str]] = None,
    force: bool = False,
) -> None:
    """
    Export Audacity labels from VAD/ASR/NVV artifacts in a workspace.

    Args:
        workspace: workspace root containing per_audio/
        project_root: Configured project root for relative path storage.
        export_vad: export VAD label files
        export_asr: export ASR label files
        export_nvv: export NVV label files
        vad_masks: optional filter list for vad_mask tokens
        asr_audio_ins: optional filter list for asr_audio_in tokens (ASR/NVV)
        force: overwrite existing label files
    """
    ws = Path(workspace).resolve()
    project_root = Path(project_root)
    per_audio = ws / KEY_PER_AUDIO
    if not per_audio.exists():
        print(f"❌ per_audio not found: {per_audio}")
        return

    audio_dirs = sorted([p for p in per_audio.iterdir() if p.is_dir()])
    if not audio_dirs:
        print(f"⚠️ No audio_id folders found in: {per_audio}")
        return

    total_written = 0
    total_skipped = 0
    total_broken = 0

    for audio_id_dir in audio_dirs:
        if not is_audio_id_dir(audio_id_dir):
            continue

        audio_id = audio_id_dir.name
        ann_root = audio_id_dir / KEY_ANNOTATIONS
        if not ann_root.exists():
            continue

        if export_vad:
            vad_dir = ann_root / KEY_VAD
            if vad_dir.exists():
                label_dir = audio_id_dir / KEY_LABELS / KEY_VAD
                ensure_dir(label_dir)

                for vad_json in sorted(vad_dir.glob(f"*_{KEY_VAD}.json")):
                    try:
                        vad_mask = _parse_vad_mask_from_vad_filename(audio_id, vad_json.name)
                    except Exception:
                        total_skipped += 1
                        continue

                    if not _should_export_combo(
                        vad_mask=vad_mask,
                        asr_audio_in=None,
                        vad_masks=vad_masks,
                        asr_audio_ins=None,
                    ):
                        continue

                    label_path = label_dir / f"{vad_json.stem}{EXT_TXT}"
                    if label_path.exists() and not force:
                        total_skipped += 1
                        print(f"⚠️ Label already exists and force=False: {label_path} -> skipping")
                        continue

                    data, status = read_json_with_status(vad_json)
                    if status != "ok" or data is None:
                        total_broken += 1
                        continue

                    try:
                        json_vad_to_audacity_labels(vad_json, label_path)
                        if not label_path.exists():
                            total_skipped += 1
                            continue
                        _update_metadata_with_label_if_present(
                            audio_id_dir=audio_id_dir,
                            label_path=label_path,
                            source=KEY_VAD,
                            generated_from=vad_json,
                            vad_mask=vad_mask,
                            asr_audio_in=None,
                            project_root=project_root,
                        )
                        total_written += 1
                    except Exception:
                        total_broken += 1

        if export_asr:
            asr_dir = ann_root / KEY_ASR
            if asr_dir.exists():
                label_dir = audio_id_dir / KEY_LABELS / KEY_ASR
                ensure_dir(label_dir)

                for asr_json in sorted(asr_dir.glob(f"*_{KEY_ASR}*.json")):
                    try:
                        vad_mask, asr_audio_in = parse_vad_and_asr_identifier_from_audio_id_filename(audio_id, asr_json.stem)
                    except Exception:
                        total_skipped += 1
                        continue

                    if not _should_export_combo(
                        vad_mask=vad_mask,
                        asr_audio_in=asr_audio_in,
                        vad_masks=vad_masks,
                        asr_audio_ins=asr_audio_ins,
                    ):
                        continue

                    label_path = label_dir / f"{asr_json.stem}{EXT_TXT}"
                    if label_path.exists() and not force:
                        total_skipped += 1
                        print(f"⚠️ Label already exists and force=False: {label_path} -> skipping")
                        continue

                    data, status = read_json_with_status(asr_json)
                    if status != "ok" or data is None:
                        total_broken += 1
                        continue

                    try:
                        json_asr_to_audacity_labels(asr_json, label_path)
                        if not label_path.exists():
                            total_skipped += 1
                            continue
                        _update_metadata_with_label_if_present(
                            audio_id_dir=audio_id_dir,
                            label_path=label_path,
                            source=KEY_ASR,
                            generated_from=asr_json,
                            vad_mask=vad_mask,
                            asr_audio_in=asr_audio_in,
                            project_root=project_root,
                        )
                        total_written += 1
                    except Exception:
                        total_broken += 1

        if export_nvv:
            nvv_dir = ann_root / KEY_NVV
            if nvv_dir.exists():
                label_dir = audio_id_dir / KEY_LABELS / KEY_NVV
                ensure_dir(label_dir)

                for nvv_json in sorted(nvv_dir.glob(f"*_{KEY_NVV}.json")):
                    try:
                        vad_mask, asr_audio_in = parse_vad_and_asr_identifier_from_audio_id_filename(audio_id, nvv_json.stem)
                    except Exception:
                        total_skipped += 1
                        continue

                    if not _should_export_combo(
                        vad_mask=vad_mask,
                        asr_audio_in=asr_audio_in,
                        vad_masks=vad_masks,
                        asr_audio_ins=asr_audio_ins,
                    ):
                        continue

                    label_path = label_dir / f"{nvv_json.stem}{EXT_TXT}"
                    if label_path.exists() and not force:
                        total_skipped += 1
                        print(f"⚠️ Label already exists and force=False: {label_path} -> skipping")
                        continue

                    data, status = read_json_with_status(nvv_json)
                    if status != "ok" or data is None:
                        total_broken += 1
                        continue

                    if not isinstance(data, dict):
                        total_broken += 1
                        continue

                    segments = data.get(KEY_NVV, [])
                    if segments is None or not isinstance(segments, list):
                        total_broken += 1
                        continue

                    try:
                        # Converter expects the list of segments
                        json_nvv_to_audacity_labels(segments, label_path)
                        if not label_path.exists():
                            total_skipped += 1
                            continue
                        _update_metadata_with_label_if_present(
                            audio_id_dir=audio_id_dir,
                            label_path=label_path,
                            source=KEY_NVV,
                            generated_from=nvv_json,
                            vad_mask=vad_mask,
                            asr_audio_in=asr_audio_in,
                            project_root=project_root,
                        )
                        total_written += 1
                    except Exception:
                        total_broken += 1

    print(
        f"✅ export_labels done for workspace='{ws.name}'. "
        f"written={total_written}, skipped={total_skipped}, broken={total_broken}"
    )