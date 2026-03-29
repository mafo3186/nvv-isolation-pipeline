#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yt_wav_download.py
------------------
Reads a list of YouTube video IDs from a text file (or directly from this file),
removes duplicates and downloads the best audio as WAV for each ID.
Requires: yt-dlp + ffmpeg
"""

import subprocess
from pathlib import Path

# --- Settings ---
INPUT_FILE = "video_ids.txt"  # text file with IDs (one per line or with 'video_id' header)
OUTPUT_DIR = Path("yt_wav")  # target folder for WAVs

# --- Setup ---
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def read_ids(file_path: Path):
    """Reads IDs, removes header and duplicates."""
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    # remove 'video_id' header if present
    ids = [l for l in lines if l.lower() != "video_id"]
    return sorted(set(ids))

def download_wav(video_id: str):
    """Downloads a single WAV file from YouTube if it does not already exist.
    Streams full yt-dlp output live and logs errors separately.
    """
    wav_path = OUTPUT_DIR / f"{video_id}.wav"

    # file already present?
    if wav_path.exists():
        print(f"⏩ Skipping {video_id}, WAV already exists.")
        return

    url = f"https://www.youtube.com/watch?v={video_id}"

    cmd = [
        "yt-dlp",
        "--extractor-args", "youtube:player_client=android",
        "-f", "bestaudio/best",
        "--extract-audio",
        "--audio-format", "wav",
        "--audio-quality", "0",
        "--ignore-errors",
        "-o", f"{OUTPUT_DIR}/{video_id}.%(ext)s",
        url,
    ]

    print(f"🎧 Downloading {video_id} ...")

    # stream yt-dlp output live
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    for line in process.stdout:
        print(line, end="")  # line-by-line live log

    process.wait()

    if process.returncode != 0:
        print(f"❌ Error for {video_id}")
        with open("failed_ids.txt", "a", encoding="utf-8") as log:
            log.write(f"{video_id}\n")
            log.flush()

def main():
    if not Path(INPUT_FILE).exists():
        print(f"❌ File {INPUT_FILE} not found.")
        return

    ids = read_ids(Path(INPUT_FILE))
    print(f"📄 {len(ids)} unique IDs found.")

    try:
        for idx, vid in enumerate(ids, 1):
            print(f"\n[{idx}/{len(ids)}] -> {vid}")
            download_wav(vid)
    except KeyboardInterrupt:
        print("\n🛑 Aborted by user – current downloads stopped.")

    print("\n✅ Done! All WAV files are located in:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()


# execution example
# python yt_wav_download.py
