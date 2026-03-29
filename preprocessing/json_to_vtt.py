import json
import os

def json_to_vtt(json_path: str, vtt_path: str = None):
    """
    Converts a transcript JSON with 'chunks' into a WebVTT file.
    
    Args:
        json_path (str): path to the JSON file with transcript
        vtt_path (str, optional): output path for the VTT file.
                                  If None, same base name as json_path.
    """
    # load JSON
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    chunks = data.get("chunks", [])
    if not chunks:
        raise ValueError("No 'chunks' found in JSON")

    # determine output filename
    if vtt_path is None:
        base, _ = os.path.splitext(json_path)
        vtt_path = base + ".vtt"

    # write VTT
    with open(vtt_path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for i, chunk in enumerate(chunks, start=1):
            start, end = chunk["timestamp"]

            # format timestamps to VTT format (hh:mm:ss.mmm)
            def fmt(t):
                h = int(t // 3600)
                m = int((t % 3600) // 60)
                s = int(t % 60)
                ms = int((t - int(t)) * 1000)
                return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

            line = f"{i}\n{fmt(start)} --> {fmt(end)}\n{chunk['text']}\n\n"
            f.write(line)

    print(f"✅ VTT saved to: {vtt_path}")
    return vtt_path

# example usage
# json_to_vtt("path/to/transcript.json")
# json_to_vtt(".../processed/transcripts/VOCALS_pilot_data_v2_transcript/0xavu1xwQKg_transcript.json")

