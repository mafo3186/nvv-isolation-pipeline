#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# --- STEP 4: HYBRID VAD (Silero) ---
"""
NVV Pipeline – Step 4: Voice Activity Detection (Hybrid)
--------------------------------------------------------------
  • Detects speech-like regions with high recall
  • Uses Silero-VAD + energy-based boundary refinement
  • Saves annotations as separate JSON in "annotations/vad" folder per audio_derivative
  • Updates metadata.json under ["annotations"]["vad"][audio_derivative] with reference + parameters
  • audio derivative selectable (from config / runner)
--------------------------------------------------------------
"""

import time
from pathlib import Path

import numpy as np
import torchaudio
from silero_vad import get_speech_timestamps, load_silero_vad

from utils.io import ensure_dir, read_json, write_json
from pipeline.pipeline_workspace_runner import setup_workspace_run
from metadata.metadata import audio_dir_metadata_path, mark_step
from utils.detect_device import detect_device
from utils.io import to_relative_path, resolve_metadata_path
from config.constants import (
    EXT_JSON,
    KEY_FIELD_SR,
    KEY_FIELD_PATH,
    KEY_VAD,
    KEY_STEP_4,
    KEY_AUDIO_FILES,
    KEY_ANNOTATIONS,
    AUDIO_ORIGINAL,
    AUDIO_DERIVATIVES,
)
from config.params import (
    SILERO_SAMPLING_RATE,
    VAD_ENERGY_REL_THRESHOLD,
    VAD_SMOOTHING_WINDOW,
    VAD_EXPAND_PRE,
    VAD_EXPAND_POST,
    VAD_EXPAND_STEP,
)




# --- Model helper (load once) ---

def load_step_4_vad_model(device: str = "auto"):
    """
    Load Silero-VAD model once.
    Returns: (vad_model, used_device)
    """
    used_device = detect_device(device) if device == "auto" else device
    vad_model = load_silero_vad()
    try:
        vad_model.to(used_device)
    except Exception:
        # If the wrapper/model doesn't support .to(), we still keep used_device for logging.
        pass
    vad_model.eval()
    return vad_model, used_device


# --- Single audio processing ---

def vad_single_audio(
    audio_id_dir: Path,
    audio: str,
    vad_model,
    device: str,
    vad_threshold: float,
    vad_min_speech_ms: int,
    vad_min_silence_ms: int,
    vad_pad_ms: int,
    project_root: Path,
    force: bool = False,
):
    """
    Step 4: Voice Activity Detection (Hybrid Silero + Energy Refinement)

    Args:
        audio_id_dir: per_audio/<audio_id> directory
        audio: audio derivative name (must be in AUDIO_DERIVATIVES)
        vad_model: loaded Silero-VAD model
        device: torch device string
        vad_threshold: Silero speech probability threshold
        vad_min_speech_ms: minimum accepted speech segment length (ms)
        vad_min_silence_ms: minimum silence gap to split segments (ms)
        vad_pad_ms: context padding around each detected segment (ms)
        project_root: Configured project root for relative path storage and resolution.
        force: overwrite existing outputs

    Error philosophy:
      - Step 4 is pre-evaluation critical.
      - If a requested audio_derivative is missing although previous steps should have created it -> RAISE.
      - Valid-but-empty audio => still write a VAD file with segments=[] (evaluable artifact).
    """

    # --- Setup ---
    t0 = time.time()
    meta_path = audio_dir_metadata_path(audio_id_dir)
    meta = read_json(meta_path)

    # --- Validate audio_derivative (Single Truth) ---
    if audio not in AUDIO_DERIVATIVES:
        raise ValueError(
            f"Invalid audio_derivative='{audio}'. Must be one of AUDIO_DERIVATIVES: {AUDIO_DERIVATIVES}"
        )

    # --- Check if this exact audio_derivative already processed ---
    # NOTE: Step-4 can be run multiple times for different audio_derivatives,
    # so we cache per audio_derivative in annotations.vad, not only in steps log.
    prev_entry = (meta.get(KEY_ANNOTATIONS, {}).get(KEY_VAD, {}) or {}).get(audio)
    if prev_entry and not force:
        prev_path_str = prev_entry.get(KEY_FIELD_PATH, "")
        prev_path = resolve_metadata_path(prev_path_str, project_root) if prev_path_str else None
        if prev_path and prev_path.exists():
            print(f"↪ {audio_id_dir.name}: Step 4 already done for {audio} (cached)")
            return meta
        
    # --- Prepare output folder ---
    vad_dir = audio_id_dir / KEY_ANNOTATIONS / KEY_VAD
    ensure_dir(vad_dir)

    # --- Resolve input wav path from metadata (Single Truth) ---
    audio_info = meta.get(KEY_AUDIO_FILES, {}).get(audio, {})
    wav_path_str = audio_info.get(KEY_FIELD_PATH, "")
    wav_path = resolve_metadata_path(wav_path_str, project_root) if wav_path_str else None

    if wav_path is None or not wav_path.exists():
        raise FileNotFoundError(
            f"❌ Step 4 input not found for audio_derivative='{audio}'. "
            f"Expected metadata['{KEY_AUDIO_FILES}']['{audio}']['{KEY_FIELD_PATH}'] -> existing file, got: {wav_path_str}"
        )

    # --- Load audio ---
    wav, sr = torchaudio.load(wav_path)

    # --- Ensure mono ---
    if wav.shape[0] > 1:
        print(f"⚠️  {audio_id_dir.name}: Input has {wav.shape[0]} channels, mixing down to mono.")
        wav = wav.mean(dim=0, keepdim=True)

    # --- Resample to Silero native rate ---
    if sr != SILERO_SAMPLING_RATE:
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=SILERO_SAMPLING_RATE)
        sr = SILERO_SAMPLING_RATE

    # --- Move audio tensor to device (if possible) ---
    try:
        wav_dev = wav.to(device)
    except Exception:
        wav_dev = wav

    # --- Step 1: Base Detection (Silero-VAD) ---
    speech_timestamps = get_speech_timestamps(
        wav_dev,
        vad_model,
        sampling_rate=sr,
        threshold=vad_threshold,
        min_speech_duration_ms=vad_min_speech_ms,
        min_silence_duration_ms=vad_min_silence_ms,
        speech_pad_ms=vad_pad_ms,
    )

    # --- Step 2: Energy-based Boundary Refinement ---
    # refinement is done on CPU numpy
    wav_cpu = wav.squeeze(0).detach().cpu().numpy()
    energy = np.abs(wav_cpu)
    smooth = np.convolve(
        energy,
        np.ones(VAD_SMOOTHING_WINDOW) / VAD_SMOOTHING_WINDOW,
        mode="same",
    )
    mean_e = float(np.mean(smooth)) if smooth.size else 0.0
    energy_threshold = VAD_ENERGY_REL_THRESHOLD * mean_e

    n_samples = len(wav_cpu)

    for seg in speech_timestamps:
        start = int(seg["start"])
        end = int(seg["end"])

        start = max(0, start - int(VAD_EXPAND_PRE * sr))
        end = min(n_samples, end + int(VAD_EXPAND_POST * sr))

        step_size = max(1, int(VAD_EXPAND_STEP * sr))

        while start > 0 and smooth[start] > energy_threshold:
            start = max(0, start - step_size)
        while end < n_samples - 1 and smooth[end] > energy_threshold:
            end = min(n_samples - 1, end + step_size)

        # --- Convert from samples to seconds ---
        seg["start_s"] = round(start / sr, 6)
        seg["end_s"] = round(end / sr, 6)

    # --- Replace sample indices with second-based times ---
    segments_in_seconds = [{"start": seg["start_s"], "end": seg["end_s"]} for seg in speech_timestamps]

    # --- 3. Save Annotation JSON (agglutinated filename) ---
    # agglutinate on wav stem
    # special-case: original should still be clearly identifiable in the filename
    if audio == AUDIO_ORIGINAL and not wav_path.stem.endswith(f"_{AUDIO_ORIGINAL}"):
        vad_filename = f"{wav_path.stem}_{AUDIO_ORIGINAL}_{KEY_VAD}{EXT_JSON}"
    else:
        vad_filename = f"{wav_path.stem}_{KEY_VAD}{EXT_JSON}"

    vad_path = (vad_dir / vad_filename).resolve()
    ensure_dir(vad_path.parent)

    vad_data = {
        "audio_derivative": audio,
        "sampling_rate": sr,
        "segment_count": len(segments_in_seconds),
        "segments": segments_in_seconds,
        "parameters": {
            "model": "Silero-VAD",
            "threshold": vad_threshold,
            "pad_ms": vad_pad_ms,
            "min_speech_ms": vad_min_speech_ms,
            "min_silence_ms": vad_min_silence_ms,
            "energy_threshold_rel": VAD_ENERGY_REL_THRESHOLD,
            "smoothing_window": VAD_SMOOTHING_WINDOW,
            "expand_pre_s": VAD_EXPAND_PRE,
            "expand_post_s": VAD_EXPAND_POST,
            "expand_step_s": VAD_EXPAND_STEP,
            "device": device,
        },
    }

    write_json(vad_path, vad_data)
    print(f"✓ Step 4: Saved VAD annotations for {audio} → {vad_path.name}")

    # --- 4. Update Metadata (Single Truth mapping by audio_derivative) ---
    meta.setdefault(KEY_ANNOTATIONS, {}).setdefault(KEY_VAD, {})
    meta[KEY_ANNOTATIONS][KEY_VAD][audio] = {
        KEY_FIELD_PATH: to_relative_path(vad_path, project_root),
        "model": "Silero-VAD",
        KEY_FIELD_SR: sr,
        "segment_count": len(segments_in_seconds),
        "device": device,
    }

    # --- 5. Log Step ---
    mark_step(
        meta,
        KEY_STEP_4,
        "done",
        t0,
        {
            "audio_derivative": audio,
            "segments": len(segments_in_seconds),
        },
    )

    write_json(meta_path, meta)
    print(f"✓ Updated metadata for VAD ({audio}) → {meta_path.name}")

    return meta


# --- Batch / workspace runner ---

def run_step_4_vad(
    workspace: Path | str,
    audio_derivatives: list[str],
    vad_threshold: float,
    vad_min_speech_ms: int,
    vad_min_silence_ms: int,
    vad_pad_ms: int,
    project_root: Path | str,
    device: str = "auto",
    force: bool = False,
) -> None:
    """
    Workspace runner for Step 4.

    Args:
        workspace: processed workspace root
        audio_derivatives: audio derivatives to run VAD on
        vad_threshold: Silero speech probability threshold
        vad_min_speech_ms: minimum accepted speech segment length (ms)
        vad_min_silence_ms: minimum silence gap to split segments (ms)
        vad_pad_ms: context padding around each detected segment (ms)
        project_root: Configured project root for relative path storage and resolution.
        device: torch device string
        force: overwrite existing outputs
    """

    project_root = Path(project_root)

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

    # Validate audio_derivatives against Single Truth (fail fast)
    for a in audio_derivatives:
        if a not in AUDIO_DERIVATIVES:
            raise ValueError(f"Invalid audio derivative '{a}' in run_step_4_vad(). Must be in AUDIO_DERIVATIVES.")

    # --- Load Silero model ONCE ---
    vad_model, used_device = load_step_4_vad_model(used_device)

    print(f"🚀 Step 4 (VAD) for workspace='{Path(workspace).name}'")
    print(f"   • Device: {used_device}")
    print(f"   • Clips: {len(audio_id_dirs)}")
    print(f"   • Audio derivatives: {audio_derivatives}")

    for audio_id_dir in audio_id_dirs:
        for audio_derivative in audio_derivatives:
            vad_single_audio(
                audio_id_dir=audio_id_dir,
                audio=audio_derivative,
                vad_model=vad_model,
                device=used_device,
                vad_threshold=vad_threshold,
                vad_min_speech_ms=vad_min_speech_ms,
                vad_min_silence_ms=vad_min_silence_ms,
                vad_pad_ms=vad_pad_ms,
                project_root=project_root,
                force=setup["force"],
            )

    print("✅ Step 4 completed.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Step 4 (VAD) over a workspace.")
    parser.add_argument("--workspace", required=True, help="Output workspace root directory")
    parser.add_argument(
        "--audio-derivatives",
        required=True,
        help="Comma-separated list of audio derivatives to run VAD on (must be in AUDIO_DERIVATIVES)",
    )
    parser.add_argument("--vad_threshold", type=float, default=0.20, help="Silero speech probability threshold")
    parser.add_argument("--vad_min_speech_ms", type=int, default=75, help="Minimum speech segment length (ms)")
    parser.add_argument("--vad_min_silence_ms", type=int, default=75, help="Minimum silence gap (ms)")
    parser.add_argument("--vad_pad_ms", type=int, default=50, help="Context padding around each segment (ms)")
    parser.add_argument("--project-root", required=True, dest="project_root", help="Project root for relative path storage")
    parser.add_argument("--device", default="auto", help="Device to use: auto | cuda | cpu")
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    args = parser.parse_args()

    audio_derivatives = [a.strip() for a in args.audio_derivatives.split(",") if a.strip()]

    run_step_4_vad(
        workspace=args.workspace,
        audio_derivatives=audio_derivatives,
        vad_threshold=args.vad_threshold,
        vad_min_speech_ms=args.vad_min_speech_ms,
        vad_min_silence_ms=args.vad_min_silence_ms,
        vad_pad_ms=args.vad_pad_ms,
        project_root=args.project_root,
        device=args.device,
        force=args.force,
    )