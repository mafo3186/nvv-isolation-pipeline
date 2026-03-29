from __future__ import annotations
from pathlib import Path


def json_segments_to_audacity_labels(segments, out_path: Path):
    """
    Exportiert eine Liste von {start, end}-Segmenten als Audacity-kompatible Labeldatei (.txt).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for idx, seg in enumerate(segments, start=1):
            start = float(seg["start"])
            end = float(seg["end"])
            label = f"NVV_candidate_{idx}"
            f.write(f"{start:.6f}\t{end:.6f}\t{label}\n")

    print(f"🎧 Audacity labels saved: {out_path.name}")
