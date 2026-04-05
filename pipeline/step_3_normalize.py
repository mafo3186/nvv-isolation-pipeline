#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# --- STEP 3: NORMALIZATION (Analysis Prep) ---

"""
NVV Pipeline – Step 3: Audio Normalization
------------------------------------------
Purpose:
    Prepare separated audio derivatives (vocals, background)
    for all downstream analysis steps (VAD, ASR, NVV).

Rationale:
    • Converts all derivatives to mono for consistent processing
    • Resamples to a fixed analysis rate (24 kHz)
      → balances quality vs. computation cost
      → compatible with later steps (Silero 16 kHz, Whisper 16–24 kHz)
    • RMS loudness normalization to TARGET_DBFS ± LIMIT_DB
      → prevents volume-based bias in VAD or ASR
    • Saves normalized "_norm" variants per derivative
      (e.g., "<stem>_norm.wav")

Output:
    - Mono, normalized WAV files (24 kHz)
    - Metadata updated with SR, channels, normalization gain
"""

import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from utils.io import ensure_dir, read_json, write_json
from pipeline.pipeline_workspace_runner import setup_workspace_run
from metadata.metadata import audio_dir_metadata_path, set_metadata_audio, mark_step
from utils.io import resolve_metadata_path
from config.constants import (
    KEY_STEP_3,
    EXT_WAV,
    KEY_AUDIO_FILES,
    KEY_NORM,
    KEY_FIELD_PATH,
    KEY_FIELD_SR,
    KEY_FIELD_CHANNELS,
)
from config.params import (
    STEP3_TARGET_DBFS,
    STEP3_LIMIT_DB,
    STEP3_ANALYSIS_SAMPLING_RATE,
    STEP3_AUDIO_INPUT,
    STEP3_WAV_SUBTYPE,
    STEP3_APPLY_PEAK_SAFETY_NORMALIZE,
    STEP3_RAISE_ON_MISSING_INPUT,
)


# --- Helper Function ---

def loudness_normalize(
    y: np.ndarray,
    target_dbfs: float = STEP3_TARGET_DBFS,
    limit_db: float = STEP3_LIMIT_DB,
) -> tuple[np.ndarray, float]:
    """
    RMS-based loudness normalization towards target dBFS with gain limit.

    Returns:
        (y_norm, gain_db)
    """
    if y.size == 0 or np.allclose(y, 0):
        return y, 0.0

    rms = float(np.sqrt(np.mean(y ** 2)))
    if np.isnan(rms) or rms <= 0.0:
        return y, 0.0

    current_db = 20 * np.log10(rms)
    gain_db = float(np.clip(target_dbfs - current_db, -limit_db, limit_db))
    factor = 10 ** (gain_db / 20)

    return y * factor, gain_db


# --- Single File Processing ---

def normalize_single_audio(
    audio_id_dir: Path,
    project_root: Path,
    device: str = "auto",   # accepted + logged for symmetry/traceability
    force: bool = False,
):
    """
    Step 3: Normalization for Analysis (VAD/ASR)
      - Mono, STEP3_ANALYSIS_SAMPLING_RATE Hz
      - RMS normalization to TARGET_DBFS ± LIMIT_DB
      - Agglutinates suffix chain (e.g. "_std_vocals" → "_std_vocals_norm.wav")

    Args:
        audio_id_dir: per_audio/<audio_id> directory.
        project_root: Configured project root for relative path storage and resolution.
        device: Device flag (logged for traceability).
        force: Overwrite existing outputs.

    Error philosophy:
      - Default: raise on missing expected inputs (broken pipeline state),
        because Step 3 is pre-evaluation critical.
    """
    t0 = time.time()

    meta_path = audio_dir_metadata_path(audio_id_dir)
    meta = read_json(meta_path)

    # Skip if already processed
    if meta.get("steps", {}).get(KEY_STEP_3, {}).get("status") == "done" and not force:
        print(f"↪ {audio_id_dir.name}: Step 3 already done (cached)")
        return meta

    print(f"▶ Step 3: Normalizing separated tracks for {audio_id_dir.name}...")

    # Ensure output folder
    audio_dir = audio_id_dir / KEY_AUDIO_FILES
    ensure_dir(audio_dir)

    # Process each configured derivative
    for audio_derivative in STEP3_AUDIO_INPUT:
        entry = meta.get(KEY_AUDIO_FILES, {}).get(audio_derivative, {})
        in_path_str = entry.get(KEY_FIELD_PATH, "")
        in_path = resolve_metadata_path(in_path_str, project_root) if in_path_str else None

        if in_path is None or not in_path.exists():
            msg = (
                f"❌ Step 3 missing input for '{audio_derivative}'. "
                f"Expected metadata['{KEY_AUDIO_FILES}']['{audio_derivative}']['{KEY_FIELD_PATH}'] "
                f"to point to an existing file, got: {in_path_str}"
            )
            if STEP3_RAISE_ON_MISSING_INPUT:
                raise FileNotFoundError(msg)
            print(f"⚠️  {msg} (skipping)")
            continue

        # Load as mono (analysis standard)
        y, sr = librosa.load(in_path, sr=None, mono=True)

        # Resample
        if sr != STEP3_ANALYSIS_SAMPLING_RATE:
            y = librosa.resample(y=y, orig_sr=sr, target_sr=STEP3_ANALYSIS_SAMPLING_RATE)
            sr = STEP3_ANALYSIS_SAMPLING_RATE

        # Loudness normalize (RMS)
        y, gain_db = loudness_normalize(y, target_dbfs=STEP3_TARGET_DBFS, limit_db=STEP3_LIMIT_DB)

        # Optional safety normalize (peak)
        if STEP3_APPLY_PEAK_SAFETY_NORMALIZE:
            peak = float(np.max(np.abs(y))) if y.size else 0.0
            if peak > 0.0:
                y = np.clip(y / peak, -1, 1)

        # Save normalized file (agglutinated suffix chain)
        stem = in_path.stem
        out_path = audio_dir / f"{stem}_{KEY_NORM}{EXT_WAV}"
        sf.write(out_path, y, sr, subtype=STEP3_WAV_SUBTYPE)

        print(f"   • Normalized {in_path.name} ({gain_db:+.2f} dB) → {out_path.name}")

        # Update metadata: key becomes e.g. "std_vocals_norm" / "std_background_norm"
        set_metadata_audio(meta, f"{audio_derivative}_{KEY_NORM}", out_path, sr, 1, project_root)

    # Log Step (device is recorded for traceability even if Step 3 is CPU work)
    mark_step(
        meta,
        KEY_STEP_3,
        "done",
        t0,
        {
            KEY_FIELD_SR: STEP3_ANALYSIS_SAMPLING_RATE,
            KEY_FIELD_CHANNELS: 1,
            "device": device,
            "target_dbfs": STEP3_TARGET_DBFS,
            "limit_db": STEP3_LIMIT_DB,
            "peak_safety_normalize": STEP3_APPLY_PEAK_SAFETY_NORMALIZE,
            "inputs": STEP3_AUDIO_INPUT,
        },
    )

    write_json(meta_path, meta)
    print("✓ Normalized files saved.")
    return meta


# --- Batch / Workspace Runner ---

def run_step_3_normalize(
    workspace: Path | str,
    project_root: Path | str,
    device: str = "auto",
    force: bool = False,
) -> None:
    """
    Batch runner for Step 3.

    Runs Step 3 over all per_audio/<audio_id>/ folders that have metadata.
    Uses setup_workspace_run to resolve/standardize device handling (symmetry).

    Args:
        workspace: processed workspace root
        project_root: Configured project root for relative path storage and resolution.
        device: Device flag.
        force: Overwrite existing outputs.
    """
    project_root = Path(project_root)

    setup = setup_workspace_run(
        workspace=workspace,
        device=device,
        force=force,
        input_dir=None,
        require_metadata=True,
    )

    audio_id_dirs = setup["audio_id_dirs"]
    used_device = setup["device"]

    if not audio_id_dirs:
        print("⚠️  No per_audio folders with metadata found. Run previous steps first.")
        return

    print(f"🚀 Step 3 (Normalize) for workspace='{Path(workspace).name}'")
    print(f"   • Device (logged): {used_device}")
    print(f"   • Clips: {len(audio_id_dirs)}")
    print(f"   • Analysis SR: {STEP3_ANALYSIS_SAMPLING_RATE}")
    print(f"   • Inputs: {STEP3_AUDIO_INPUT}")
    print(f"   • Raise on missing input: {STEP3_RAISE_ON_MISSING_INPUT}")

    for audio_id_dir in audio_id_dirs:
        normalize_single_audio(
            audio_id_dir=audio_id_dir,
            project_root=project_root,
            device=used_device,
            force=setup["force"],
        )

    print("✅ Step 3 completed.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Step 3 (normalization) over a workspace.")
    parser.add_argument("--workspace", required=True, help="Output workspace root directory")
    parser.add_argument("--project-root", required=True, dest="project_root", help="Project root for relative path storage")
    parser.add_argument("--device", default="auto", help="Device flag (resolved + logged via setup)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    args = parser.parse_args()

    run_step_3_normalize(
        workspace=args.workspace,
        project_root=args.project_root,
        device=args.device,
        force=args.force,
    )