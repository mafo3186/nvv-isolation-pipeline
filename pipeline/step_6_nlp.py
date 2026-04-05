#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NVV Pipeline – Step 6: NLP (spaCy Lexical Filter)
-------------------------------------------------
Classifies ASR-segmented transcriptions into
[word | filler | non_word | oov | unknown].
• Loads spaCy model once per workspace
• Processes existing ASR artifacts only (file-driven A1)
• Writes:
    annotations/nlp/<audio_id>_<vad_mask>_vad_<asr_audio_in>_asr_nlp.json
    annotations/nlp/<audio_id>_<vad_mask>_vad_<asr_audio_in>_asr_nlp_log.json
• Updates metadata.json:
    meta["annotations"]["nlp"][combo_key] = {...}

Evaluation Semantics / Error Handling:
- Missing ASR => ignored (not found, not processed)
- Broken ASR JSON / invalid structure => raise
- segments=[] => valid empty NLP artifact
- force=False + existing NLP => refresh metadata only
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List
from collections import defaultdict

from config.constants import (
    KEY_NLP,
    KEY_ASR,
    KEY_ANNOTATIONS,
    KEY_FIELD_PATH,
    KEY_STEP_6,
    EXT_JSON,
)

#toDo: use factory path functions and pass workspace instead of audio_id_dir, get paths. 

from utils.io import (
    ensure_dir,
    read_json_with_status,
    write_json,
    audio_dir_metadata_path,
    to_relative_path,
)
from utils.parsing import (
    derive_combo_key,
    parse_vad_and_asr_identifier_from_audio_id_filename,
)

from pipeline.pipeline_workspace_runner import setup_workspace_run
from metadata.metadata import mark_step
from pipeline.spacy_nlp import analyze_asr_text, init_spacy_en


# --- ASR Validation Helpers ---

def _parse_asr_segments(asr_data: Any) -> List[Dict[str, Any]]:
    if asr_data is None:
        raise ValueError("ASR JSON is None.")

    if isinstance(asr_data, dict):
        if "segments" not in asr_data:
            raise ValueError("ASR JSON missing 'segments' key.")
        segments = asr_data["segments"]
        if not isinstance(segments, list):
            raise ValueError("ASR JSON 'segments' must be a list.")
        return segments

    if isinstance(asr_data, list):
        return asr_data

    raise ValueError(f"Unsupported ASR JSON type: {type(asr_data).__name__}")


def _safe_get_chunks(seg: Any) -> List[Dict[str, Any]]:
    if not isinstance(seg, dict):
        raise ValueError(f"ASR segment must be dict, got {type(seg).__name__}")
    chunks = seg.get("chunks", [])
    if chunks is None:
        return []
    if not isinstance(chunks, list):
        raise ValueError("ASR segment 'chunks' must be list.")
    return chunks


# --- Core NLP Computation ---

def _compute_nlp(asr_path: Path, nlp_model):
    t0 = time.time()

    asr_data, st = read_json_with_status(asr_path)
    if st != "ok" or asr_data is None:
        raise ValueError(f"Broken ASR JSON at: {asr_path}")

    segments = _parse_asr_segments(asr_data)

    n_chunks = defaultdict(int) # n_chunks per category have falues like: {"word": 10, "filler": 5, ...}
    token_lists = defaultdict(list)
    token_info: Dict[str, Any] = {}

    for seg in segments:
        chunks = _safe_get_chunks(seg)

        for ch in chunks:
            if not isinstance(ch, dict):
                raise ValueError("ASR chunk must be dict.")

            text = str(ch.get("text", "")).strip()
            if not text:
                ch["category"] = "unknown"
                continue

            chunk_cat, info, token_pairs = analyze_asr_text(nlp_model, text)
            ch["category"] = chunk_cat
            n_chunks[chunk_cat] += 1

            for t_text, t_cat in token_pairs:
                token_lists[t_cat].append(t_text)

            for tok, meta in info.items():
                token_info[tok] = meta

    # Output paths
    annotations_dir = asr_path.parent.parent
    nlp_dir = annotations_dir / KEY_NLP
    ensure_dir(nlp_dir)

    stem = asr_path.stem
    nlp_path = nlp_dir / f"{stem}_{KEY_NLP}{EXT_JSON}"
    log_path = nlp_dir / f"{stem}_{KEY_NLP}_log{EXT_JSON}"

    write_json(nlp_path, {"segments": segments})
    write_json(
        log_path,
        {
            "n_chunks": dict(n_chunks),
            "token": {k: sorted(set(v)) for k, v in token_lists.items()},
            "token_info": token_info,
        },
    )

    return {
        "nlp_path": nlp_path,
        "log_path": log_path,
        "segments": len(segments),
        "n_chunks": dict(n_chunks),
        "time_s": round(time.time() - t0, 3),
    }


def process_single_nlp(
    asr_path: Path | str,
    audio_id_dir: Path | str,
    audio_id: str,
    nlp_model,
    project_root: Path,
    *,
    force: bool = False,
) -> Dict[str, Any]:
    """
    Process one ASR artifact with a pre-loaded spaCy model and update metadata.json.

    Args:
        asr_path: Path to the ASR JSON artifact.
        audio_id_dir: per_audio/<audio_id> directory.
        audio_id: audio_id_dir.name.
        nlp_model: Loaded spaCy model.
        project_root: Configured project root for relative path storage.
        force: Overwrite existing NLP outputs.

    Returns:
        A small result dict for logging/debugging.
    """
    asr_path = Path(asr_path)
    audio_id_dir = Path(audio_id_dir)

    combo_key = derive_combo_key(asr_path.stem, audio_id)
    vad_mask, asr_audio_in = parse_vad_and_asr_identifier_from_audio_id_filename(audio_id, asr_path.stem)

    
    meta_path = audio_dir_metadata_path(audio_id_dir)

    annotations_dir = audio_id_dir / KEY_ANNOTATIONS
    nlp_dir = annotations_dir / KEY_NLP
    stem = asr_path.stem
    nlp_path = nlp_dir / f"{stem}_{KEY_NLP}{EXT_JSON}"
    log_path = nlp_dir / f"{stem}_{KEY_NLP}_log{EXT_JSON}"

    # --- Refresh only ---
    if nlp_path.exists() and log_path.exists() and not force:
        nlp_data, st1 = read_json_with_status(nlp_path)
        log_data, st2 = read_json_with_status(log_path)
        if st1 != "ok" or st2 != "ok":
            raise ValueError(f"Broken NLP artifacts at {nlp_path}")

        seg_obj = nlp_data.get("segments", [])
        n_chunks = log_data.get("n_chunks", {})
    else:
        result = _compute_nlp(asr_path, nlp_model)
        seg_obj = result["segments"]  # currently len(segments) from _compute_nlp
        n_chunks = result["n_chunks"]
        nlp_path = result["nlp_path"]
        log_path = result["log_path"]

    # Normalize to a consistent count in metadata
    segments_count = len(seg_obj) if isinstance(seg_obj, list) else int(seg_obj)

    model_meta = {
        "spacy_model_name": nlp_model.meta.get("name"),
        "spacy_model_version": nlp_model.meta.get("version"),
        "spacy_model_lang": nlp_model.meta.get("lang"),
    }

    # --- Update metadata ---
    meta, st = read_json_with_status(meta_path)
    if st != "ok" or meta is None:
        raise RuntimeError(f"Cannot read metadata at {meta_path}")

    meta.setdefault(KEY_ANNOTATIONS, {}).setdefault(KEY_NLP, {})
    meta[KEY_ANNOTATIONS][KEY_NLP][combo_key] = {
        KEY_FIELD_PATH: to_relative_path(nlp_path, project_root),
        "log_path": to_relative_path(log_path, project_root),
        "vad_mask": vad_mask,
        "asr_audio_in": asr_audio_in,
        "segments": segments_count,
        "n_chunks": n_chunks,
        "status": "ok",
        **model_meta,
    }

    write_json(meta_path, meta)
    print(f"✓ Step 6 NLP: Saved {segments_count} segments for VAD-mask {vad_mask} / ASR-Audio Input {asr_audio_in} →  at: {nlp_path}")

    return {
        "combo_key": combo_key,
        "nlp_path": nlp_path,
        "log_path": log_path,
        "segments": segments_count,
        "n_chunks": n_chunks,
    }


# --- Workspace Runner ---

def run_step_6_nlp(
    workspace: Path | str,
    spacy_model: str,
    project_root: Path | str,
    *,
    auto_download: bool = True,
    force: bool = False,
) -> None:
    """
    Workspace runner for Step 6.

    Args:
        workspace: processed workspace root
        spacy_model: spaCy model name
        project_root: Configured project root for relative path storage.
        auto_download: Auto-download missing spaCy model.
        force: Overwrite existing results.
    """
    project_root = Path(project_root)

    setup = setup_workspace_run(
        workspace=workspace,
        input_dir=None,
        device="cpu",  # not relevant for spaCy, but required by runner
        force=force,
        require_metadata=True,
    )

    audio_id_dirs = setup["audio_id_dirs"]

    if not audio_id_dirs:
        print("⚠️  No per_audio folders found.")
        return

    # --- Load spaCy ONCE ---
    nlp = init_spacy_en(model_name=spacy_model, auto_download=auto_download)

    model_meta = {
        "spacy_model_name": nlp.meta.get("name"),
        "spacy_model_version": nlp.meta.get("version"),
        "spacy_model_lang": nlp.meta.get("lang"),
    }

    print(f"🚀 Step 6 (NLP) – Model: {model_meta}")

    for audio_id_dir in audio_id_dirs:
        t0 = time.time()
        audio_id = audio_id_dir.name
        meta_path = audio_dir_metadata_path(audio_id_dir)

        asr_dir = audio_id_dir / KEY_ANNOTATIONS / KEY_ASR
        if not asr_dir.exists():
            continue

        asr_files = sorted(asr_dir.glob(f"*_{KEY_ASR}{EXT_JSON}"))

        for asr_path in asr_files:
            process_single_nlp(
                asr_path=asr_path,
                audio_id_dir=audio_id_dir,
                audio_id=audio_id,
                nlp_model=nlp,
                project_root=project_root,
                force=force,
            )
            
    
        # --- Mark Step Done ---
        meta, st = read_json_with_status(meta_path)
        if st != "ok" or meta is None:
            raise RuntimeError(f"Cannot re-read metadata at {meta_path}")

        mark_step(
            meta,
            KEY_STEP_6,
            "done",
            t0,
            {
                "spacy_model": spacy_model,
                **model_meta,
            },
        )
        write_json(meta_path, meta)

    print("✅ Step 6 completed.")



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Step 6 (NLP) over workspace.")
    parser.add_argument("--workspace", required=True, help="Processed workspace (contains metadata.json)")
    parser.add_argument("--spacy-model", default="en_core_web_sm", help="spaCy model name")
    parser.add_argument("--project-root", required=True, dest="project_root", help="Project root for relative path storage")
    parser.add_argument("--auto-download", action="store_true", help="Auto-download missing spaCy model")
    parser.add_argument("--force", action="store_true", help="Overwrite existing results")

    args = parser.parse_args()

    run_step_6_nlp(
        workspace=Path(args.workspace),
        spacy_model=args.spacy_model,
        project_root=args.project_root,
        auto_download=args.auto_download,
        force=args.force,
    )