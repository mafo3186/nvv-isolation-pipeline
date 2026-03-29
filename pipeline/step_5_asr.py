#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# --- STEP 5: ASR (CrisperWhisper) ---
"""
NVV Pipeline – Step 5: Automatic Speech Recognition (CrisperWhisper)
--------------------------------------------------------------------
• Loads CrisperWhisper once per workspace (init_asr_model)
• Transcribes any asr_audio_in (AUDIO_DERIVATIVES)
• Optionally uses vad_mask for gating (VAD_MASKS, including "no")
• Writes outputs per clip in:
      per_audio/<audio_id>/annotations/asr/<audio_id>_<vad_mask>_vad_<asr_audio_in>_asr.json
• Updates metadata.json:
      meta["annotations"]["asr"][combo_key] = {...}

Error philosophy (evaluation-safe):
  - missing inputs (audio or vad) => no output JSON, but write metadata entry status="missing"
  - broken inputs (cannot read VAD JSON, cannot load audio) => raise (pipeline must be fixed)
  - empty results are valid artifacts => write JSON with segments=[] and status="ok"
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import warnings

import torch
import torchaudio
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers.models.whisper import generation_whisper

from utils.io import ensure_dir, read_json_with_status, write_json, audio_dir_metadata_path
from utils.parsing import create_combo_key
from pipeline.pipeline_workspace_runner import setup_workspace_run
from utils.detect_device import detect_device
from metadata.metadata import mark_step, reset_metadata_group #toDo: reset_metadata_group should be used in case of force=True but not step already done, but currently not implemented
from config.constants import (
    EXT_JSON,
    KEY_ANNOTATIONS,
    KEY_ASR,
    KEY_VAD,
    KEY_AUDIO_FILES,
    KEY_FIELD_PATH,
    KEY_STEP_5,
    AUDIO_DERIVATIVES,
    VAD_MASKS,
    VAD_NO,
)


# --- Model Initialization ---
def init_asr_model(utils_path: Path, device: str = "auto", chunk_length_s: int = 30, batch_size: int = 1):
    """
    Load CrisperWhisper model once and return:
      (pipe, adjust_func, torch_dtype, used_device)

    Args:
        utils_path: Path to utils module (for patched whisper utils).
        device: "auto" | "cuda" | "cpu"
        chunk_length_s: CrisperWhisper chunk length in seconds.
        batch_size: Inference batch size.
    """
    used_device = device if device in ["cuda", "cpu"] else detect_device()

    utils_path = Path(utils_path).resolve()
    utils_path_str = os.path.abspath(str(utils_path))
    if utils_path_str not in sys.path:
        sys.path.insert(0, utils_path_str)

    from pipeline.crisperwhisper_utils import (
        _adjust_pauses_for_hf_pipeline_output,
        _patched_extract_token_timestamps,
    )

    # Patch Whisper timestamp extraction globally 
    generation_whisper.WhisperGenerationMixin._extract_token_timestamps = _patched_extract_token_timestamps

    model_id = "nyrahealth/CrisperWhisper"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        use_cache=False,
        attn_implementation="eager",
    ).to(used_device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=chunk_length_s,
        batch_size=batch_size,
        return_timestamps="word",
        torch_dtype=torch_dtype,
        device=used_device,
    )
    return pipe, _adjust_pauses_for_hf_pipeline_output, torch_dtype, used_device


# --- Helper: Offset Timestamps ---
def _offset_chunks_global_timestamps(result: Dict[str, Any], offset_s: float) -> Dict[str, Any]:
    """
    Adds a global offset (in seconds) to all chunk timestamps.
    Keeps None end timestamps without replacement.
    """
    chunks = result.get("chunks", [])
    out_chunks = []
    for ch in chunks:
        ts = ch.get("timestamp", None)
        if isinstance(ts, (list, tuple)) and ts and ts[0] is not None:
            start = ts[0] + offset_s
            end = (ts[1] + offset_s) if len(ts) > 1 and ts[1] is not None else None
            ch = {**ch, "timestamp": (start, end)}
        out_chunks.append(ch)
    result["chunks"] = out_chunks
    return result


# --- Transcription Core ---
def transcribe_single_audio(
    audio_path: Path,
    vad_segments: Optional[List[Dict[str, float]]],
    pipe,
    adjust_func,
) -> Tuple[List[Dict[str, Any]], int, Optional[int]]:
    """
    Transcribe one audio file with optional VAD segmentation.

    Args:
        audio_path: Input wav path.
        vad_segments:
            - None => no VAD gating (full file)
            - []   => VAD requested but no segments (valid empty artifact)
            - list of segments => run ASR per segment
        pipe: Initialized HF pipeline.
        adjust_func: Post-processing function for HF output.

    Returns:
        transcripts:
            List of dicts (one per segment) OR [] for empty results.
        sampling_rate:
            Sampling rate used by torchaudio.load.
        vad_segments_skipped:
            - None for no-VAD run
            - 0 if VAD run and no segments were skipped
            - N if N segments were skipped due to invalid boundaries
    """
    wav, sr = torchaudio.load(str(audio_path))

    # --- Ensure mono ---
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    wav_np = wav.squeeze().numpy()
    total_duration = len(wav_np) / sr if sr > 0 else 0.0

    transcripts: List[Dict[str, Any]] = []

    # --- Case 1: No-VAD run (full file) ---
    if vad_segments is None:
        input_dict = {"array": wav_np, "sampling_rate": sr}
        raw_out = pipe(input_dict)
        adjusted = adjust_func(raw_out)
        adjusted = _offset_chunks_global_timestamps(adjusted, offset_s=0.0)
        adjusted["segment_start"] = 0.0
        adjusted["segment_end"] = float(total_duration)
        transcripts.append(adjusted)
        return transcripts, sr, None

    # --- Case 2: VAD requested, but zero VAD segments => valid empty artifact ---
    if len(vad_segments) == 0:
        return [], sr, 0

    # --- Case 3: VAD requested with segments ---
    skipped = 0

    for seg in vad_segments:
        start = seg.get("start")
        end = seg.get("end")

        if start is None or end is None:
            skipped += 1
            continue

        s_samp = int(float(start) * sr)
        e_samp = int(float(end) * sr)

        if e_samp <= s_samp or e_samp > len(wav_np):
            skipped += 1
            continue

        audio_slice = wav_np[s_samp:e_samp]
        if audio_slice.size == 0:
            skipped += 1
            continue

        input_dict = {"array": audio_slice, "sampling_rate": sr}
        raw_out = pipe(input_dict)
        adjusted = adjust_func(raw_out)

        adjusted = _offset_chunks_global_timestamps(adjusted, offset_s=float(start))
        adjusted["segment_start"] = float(start)
        adjusted["segment_end"] = float(end)

        transcripts.append(adjusted)

    return transcripts, sr, skipped


# --- Single clip + combo ---
def transcribe_single_audio_vad_masked(
    audio_id_dir: Path,
    asr_audio_in: str,
    vad_mask: str,
    pipe,
    adjust_func,
    device: str,
    force: bool = False,
) -> None:
    """
    Transcribe exactly one (vad_mask, asr_audio_in) combo for one clip folder.

    Missing policy:
      - missing audio input => write metadata status="missing", no json output
      - missing VAD (when vad_mask != "no") => write metadata status="missing", no json output

    Broken policy:
      - broken VAD json => raise
      - torchaudio load / ASR errors => raise
      - existing output json but broken => raise
    """
    meta_path = audio_dir_metadata_path(audio_id_dir)
    meta_data, meta_status = read_json_with_status(meta_path)
    if meta_status != "ok" or meta_data is None:
        raise RuntimeError(f"Step 5 requires a readable metadata.json, got status='{meta_status}' at: {meta_path}")

    meta = meta_data

    # Validate inputs against Single Source of Truth
    if asr_audio_in not in AUDIO_DERIVATIVES:
        raise ValueError(f"Invalid asr_audio_in='{asr_audio_in}'. Must be in AUDIO_DERIVATIVES.")
    if vad_mask not in VAD_MASKS:
        raise ValueError(f"Invalid vad_mask='{vad_mask}'. Must be in VAD_MASKS.")

    combo_key = create_combo_key(vad_mask=vad_mask, asr_audio_in=asr_audio_in)

    out_dir = audio_id_dir / KEY_ANNOTATIONS / KEY_ASR
    ensure_dir(out_dir)

    audio_id = audio_id_dir.name
    out_json = (out_dir / f"{audio_id}_{combo_key}{EXT_JSON}").resolve()

    def _write_asr_metadata(status: str, *, path: Optional[Path], extra: Optional[Dict[str, Any]] = None) -> None:
        meta.setdefault(KEY_ANNOTATIONS, {}).setdefault(KEY_ASR, {})
        entry: Dict[str, Any] = {
            "path": str(path) if path is not None else None,
            "vad_mask": vad_mask,
            "asr_audio_in": asr_audio_in,
            "device": device,
            "status": status,
        }
        if extra:
            entry.update(extra)
        meta[KEY_ANNOTATIONS][KEY_ASR][combo_key] = entry
        write_json(meta_path, meta)

    # --- Resolve input wav from metadata (Single Truth) ---
    audio_info = meta.get(KEY_AUDIO_FILES, {}).get(asr_audio_in, {}) or {}
    wav_path_str = audio_info.get(KEY_FIELD_PATH, "")
    wav_path = Path(wav_path_str) if wav_path_str else None

    if wav_path is None or not wav_path.exists():
        _write_asr_metadata("missing", path=None, extra={"error_msg": f"Missing input audio: {wav_path_str}"})
        return

    # --- Resolve VAD segments ---
    vad_segments: Optional[List[Dict[str, float]]] = None
    if vad_mask != VAD_NO:
        vad_entry = (meta.get(KEY_ANNOTATIONS, {}).get(KEY_VAD, {}) or {}).get(vad_mask)
        vad_path_str = (vad_entry or {}).get(KEY_FIELD_PATH, "")
        vad_path = Path(vad_path_str) if vad_path_str else None

        if vad_path is None or not vad_path.exists():
            _write_asr_metadata("missing", path=None, extra={"error_msg": f"Missing VAD annotation: {vad_path_str}"})
            return

        vad_data, vad_status = read_json_with_status(vad_path)
        if vad_status == "missing":
            _write_asr_metadata("missing", path=None, extra={"error_msg": f"Missing VAD annotation: {vad_path}"})
            return
        if vad_status == "error" or vad_data is None:
            raise ValueError(f"Broken VAD JSON (status='{vad_status}') at: {vad_path}")

        # IMPORTANT: keep [] as-is (no fallback)
        vad_segments = vad_data.get("segments", [])
        if vad_segments is None:
            vad_segments = []

    # --- Reuse existing output if present and not force ---
    if out_json.exists() and not force:
        existing, st = read_json_with_status(out_json)
        if st == "error" or existing is None:
            raise ValueError(f"Broken ASR JSON output exists at: {out_json}")
        # Minimal re-derive counts for rebuilt metadata
        segments = existing.get("segments", []) if isinstance(existing, dict) else []
        if segments is None:
            segments = []
        word_chunks = sum(len(seg.get("chunks", [])) for seg in segments)
        sampling_rate = existing.get("sampling_rate", None)
        _write_asr_metadata(
            "ok",
            path=out_json,
            extra={
                "sampling_rate": sampling_rate,
                "vad_segments": len(vad_segments) if vad_segments is not None else None,
                "vad_segments_skipped": None,
                "asr_segments": len(segments),
                "word_chunks": word_chunks,
            },
        )
        return

    # --- Run ASR ---
    try:
        transcripts, sr, vad_segments_skipped = transcribe_single_audio(
            audio_path=wav_path,
            vad_segments=vad_segments,
            pipe=pipe,
            adjust_func=adjust_func,
        )
    except Exception as e:
        print(f"ASR failed for {audio_id} combo={combo_key}: {e}")
        raise

    asr_output = {
        "sampling_rate": int(sr),
        "segments": transcripts,
    }

    try:
        write_json(out_json, asr_output)
    except Exception as e:
        raise RuntimeError(f"Failed to write ASR JSON for {audio_id} combo={combo_key}: {e}") from e

    word_chunks = sum(len(seg.get("chunks", [])) for seg in transcripts)

    _write_asr_metadata(
        "ok",
        path=out_json,
        extra={
            "sampling_rate": int(sr),
            "vad_segments": len(vad_segments) if vad_segments is not None else None,
            "vad_segments_skipped": vad_segments_skipped,
            "asr_segments": len(transcripts),
            "word_chunks": word_chunks,
        },
    )

    print(f"✓ Step 5 ASR: Saved. {len(transcripts) if transcripts is not None else 0} ASR-segments, {len(vad_segments) if vad_segments is not None else 0} VAD-segments (skipped: {vad_segments_skipped}) at: {out_json}")


# --- Workspace Runner (model loaded once) ---

def run_step_5_asr(
    workspace: Path | str,
    utils_path: Path | str,
    vad_masks: List[str],
    asr_audios_in: List[str],
    asr_chunk_length_s: int,
    asr_batch_size: int,
    device: str = "auto",
    force: bool = False,
) -> None:
    """
    Workspace runner for Step 5.

    Args:
        workspace: processed workspace root
        utils_path: Path to CrisperWhisper utils module.
        vad_masks: list of vad masks to run (must be in VAD_MASKS)
        asr_audios_in: list of ASR input audio derivatives (must be in AUDIO_DERIVATIVES)
        asr_chunk_length_s: CrisperWhisper chunk length in seconds
        asr_batch_size: Inference batch size
        device: torch device string
        force: overwrite existing outputs

    Notes:
      - missing inputs => metadata status="missing", continue
      - broken inputs => raise
    """
    setup = setup_workspace_run(
        workspace=workspace,
        input_dir=None,
        device=device,
        force=force,
        require_metadata=True,
    )

    used_device = setup["device"]
    audio_id_dirs = setup["audio_id_dirs"]

    if not audio_id_dirs:
        print("⚠️  No per_audio folders with metadata found. Run previous steps first.")
        return

    # Validate lists (fail fast)
    for v in vad_masks:
        if v not in VAD_MASKS:
            raise ValueError(f"Invalid vad_mask '{v}' in run_step_5_asr(). Must be in VAD_MASKS.")
    for a in asr_audios_in:
        if a not in AUDIO_DERIVATIVES:
            raise ValueError(f"Invalid asr_audio_in '{a}' in run_step_5_asr(). Must be in AUDIO_DERIVATIVES.")

    # --- Load ASR model ONCE ---
    pipe, adjust_func, _, used_device = init_asr_model(Path(utils_path), device=used_device, chunk_length_s=asr_chunk_length_s, batch_size=asr_batch_size)

    print(f"🚀 Step 5 (ASR) for workspace='{Path(workspace).name}'")
    print(f"   • Device: {used_device}")
    print(f"   • Clips: {len(audio_id_dirs)}")
    print(f"   • vad_masks: {vad_masks}")
    print(f"   • asr_audios_in: {asr_audios_in}")

    for audio_id_dir in audio_id_dirs:
        t0 = time.time()
        meta_path = audio_dir_metadata_path(audio_id_dir)
        # toDo: reset metadata group if force = True or step = done

        for vad_mask in vad_masks:
            for asr_audio_in in asr_audios_in:
                transcribe_single_audio_vad_masked(
                    audio_id_dir=audio_id_dir,
                    asr_audio_in=asr_audio_in,
                    vad_mask=vad_mask,
                    pipe=pipe,
                    adjust_func=adjust_func,
                    device=used_device,
                    force=setup["force"],
                )

        # --- Log Step ---
        meta_after, st = read_json_with_status(meta_path)
        if st != "ok" or meta_after is None:
            raise RuntimeError(f"Failed to re-read metadata after ASR for: {meta_path}")

        mark_step(
            meta_after,
            KEY_STEP_5,
            "done",
            t0,
            {
                "vad_masks": vad_masks,
                "asr_audios_in": asr_audios_in,
                "device": used_device,
            },
        )
        write_json(meta_path, meta_after)

    print("✅ Step 5 completed.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Step 5 (ASR) over a workspace.")
    parser.add_argument("--workspace", type=str, required=True, help="Path to the workspace directory.")
    parser.add_argument("--utils_path", type=str, required=True, help="Path to the utils module (for patched whisper utils).")
    parser.add_argument("--vad_masks", type=str, nargs="+", required=True, help="List of vad_masks to run (e.g., 'no', 'vad1', 'vad2').")
    parser.add_argument("--asr_audios_in", type=str, nargs="+", required=True, help="List of asr_audio_in to run (must be in AUDIO_DERIVATIVES).")
    parser.add_argument("--asr_chunk_length_s", type=int, default=30, help="CrisperWhisper chunk length in seconds.")
    parser.add_argument("--asr_batch_size", type=int, default=1, help="Inference batch size.")
    parser.add_argument("--device", type=str, default="auto", help="Device to use for ASR (e.g., 'auto', 'cuda', 'cpu').")
    parser.add_argument("--force", action="store_true", help="Force re-run even if outputs exist.")
    args = parser.parse_args()
    run_step_5_asr(
        workspace=args.workspace,
        utils_path=args.utils_path,
        vad_masks=args.vad_masks,
        asr_audios_in=args.asr_audios_in,
        asr_chunk_length_s=args.asr_chunk_length_s,
        asr_batch_size=args.asr_batch_size,
        device=args.device,
        force=args.force,
    )   