"""
extract_and_copy_annotated_wavs.py

This module exports a single wrapper function:

    extract_and_copy_annotated_wavs(
        excel_path,
        dataset_root,
        output_folder_name
    )

It:
- extracts unique audio IDs from cleaned Excel
- auto-detects all audio_wave* folders inside dataset_root 
- copies WAVs as <audio_id>.wav into output folder
- writes missing_ids.txt

No Excel cleaning, no label export — these happen in the Notebook.
"""

from pathlib import Path
import pandas as pd
import shutil

from utils.parsing import extract_unique_ids



# --- Helper: auto-detect wave directories ---
def auto_collect_wave_dirs(dataset_root: Path):
    """
    Find all directories inside dataset_root that contain "audio_wave".
    """
    wave_dirs = [
        p for p in dataset_root.iterdir()
        if p.is_dir() and "audio_wave" in p.name
    ]
    return sorted(wave_dirs, key=lambda p: p.name)


# --- Helper: find single WAV file ---
def find_wav_file(wave_dirs, ytid: str):
    for d in wave_dirs:
        p = d / f"{ytid}.wav"
        if p.exists():
            return p
    return None


# --- Helper: copy WAV files ---
def copy_wavs(ytids, wave_dirs, out_folder):
    out_folder.mkdir(parents=True, exist_ok=True)

    copied = 0
    missing = []

    for ytid in ytids:
        src = find_wav_file(wave_dirs, ytid)
        if src:
            shutil.copy2(src, out_folder / f"{ytid}.wav")
            copied += 1
        else:
            missing.append(ytid)

    if missing:
        (out_folder / "missing_ids.txt").write_text("\n".join(missing))

    return copied, missing


# --- MAIN WRAPPER ---
def extract_and_copy_annotated_wavs_from_vocals_project_structure(
    excel_path: Path,
    dataset_root: Path,
    output_folder_path: Path,
    id_column: str = "video_id",
    verbose: bool = True,
):
    """
    Wrapper that:
    - extracts unique audio IDs from cleaned excel
    - auto-detects audio_wave* folders
    - copies WAVs
    - writes missing_ids.txt
    """

    # 1. Extract IDs
    audio_ids = extract_unique_ids(excel_path, id_column=id_column)
    if verbose:
        print(f"✔ Extracted {len(audio_ids)} audio IDs from: {excel_path.name}")

    # 2. Detect wave dirs
    wave_dirs = auto_collect_wave_dirs(dataset_root)
    if verbose:
        print(f"✔ Found {len(wave_dirs)} audio_wave folders:")
        for w in wave_dirs:
            print("  •", w.name)

    # 3. Copy WAVs
    output_folder = Path(output_folder_path)
    copied, missing = copy_wavs(audio_ids, wave_dirs, output_folder)

    if verbose:
        print(f"✔ Copied WAVs: {copied}")
        if missing:
            print(f"⚠ Missing {len(missing)} files → {output_folder/'missing_ids.txt'}")

    return {
        "cleaned_excel": excel_path,
        "wave_dirs": wave_dirs,
        "audio_ids": audio_ids,
        "copied": copied,
        "missing": missing,
        "output_dir": output_folder,
    }
