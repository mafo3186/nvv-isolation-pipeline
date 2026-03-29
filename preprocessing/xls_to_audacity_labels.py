#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xls_to_audacity_labels.py
---------------------------------------------------------
Converts a VOCALS annotation spreadsheet (.xls / .xlsx)
into Audacity-compatible label files (.txt) — one per video_id.

Format (per row):
<start_time_in_seconds>    <end_time_in_seconds>    <vocalization_type>

Example:
0.400   0.770   sharp inhale
0.850   1.670   ooo sound
"""

import argparse
import pandas as pd
import datetime
from pathlib import Path


def xls_to_audacity_labels(
    input_path: str | Path,
    output_dir: str | Path = "audacity_labels",
    start_col: str = "start_s",
    end_col: str = "end_s",
    label_col: str = "vocalization_type",
    id_col: str = "video_id",
) -> None:
    """
    Converts an Excel annotation spreadsheet to Audacity label files.

    Parameters
    ----------
    input_path : str | Path
        Path to the input file (.xls or .xlsx)
    output_dir : str | Path, optional
        Target directory for exported label files
    start_col, end_col, label_col, id_col : str, optional
        Column names for start time, end time, label text and video ID
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # --- load Excel with automatic header detection ---
    def read_excel_with_flexible_header(path, expected_cols):
        """
        Reads Excel file robustly by trying multiple possible header rows.
        Tries rows 0, 1 and 2 as header until expected columns are found.
        """
        for header_row in [0, 1, 2]:
            try:
                df = pd.read_excel(path, header=header_row)
                # normalize column names
                df.columns = df.columns.str.strip().str.lower()
                expected_lower = [c.lower() for c in expected_cols]
                # check if at least one expected column exists
                if all(c in df.columns for c in expected_lower):
                    print(f"📄 Header found in row {header_row + 1}")
                    return df
            except Exception as e:
                print(f"⚠️ Error reading with header={header_row}: {e}")
        raise ValueError(f"❌ No valid header row found in {path}")

    # expected columns derived from function parameters
    expected_cols = [id_col, start_col, end_col, label_col]

    # read Excel with flexible header
    df = read_excel_with_flexible_header(input_path, expected_cols)

    # --- type cleanup: time columns always as float ---
    def to_seconds(x):
        """Tries to robustly convert arbitrary Excel time formats to float seconds."""
        if pd.isna(x):
            return None

        # 1️⃣ number → use directly
        if isinstance(x, (float, int)):
            return float(x)

        # 2️⃣ timedelta → total_seconds()
        if isinstance(x, datetime.timedelta):
            return x.total_seconds()

        # 3️⃣ time or datetime → convert to seconds
        if isinstance(x, datetime.time):
            return x.hour * 3600 + x.minute * 60 + x.second + x.microsecond / 1e6
        if isinstance(x, datetime.datetime):
            return x.hour * 3600 + x.minute * 60 + x.second + x.microsecond / 1e6

        # 4️⃣ string formats
        if isinstance(x, str):
            x = x.strip().replace(",", ".")
            if ":" in x:
                try:
                    parts = [float(p) for p in x.split(":")]
                    if len(parts) == 3:
                        return parts[0] * 3600 + parts[1] * 60 + parts[2]
                except ValueError:
                    pass
            try:
                return float(x)
            except ValueError:
                return None

        return None

    df[start_col] = df[start_col].apply(to_seconds)
    df[end_col] = df[end_col].apply(to_seconds)

    # cleanup
    df = df.dropna(subset=[start_col, end_col, label_col])

    # --- grouping and export ---
    for vid, group in df.groupby(id_col):
        group = group.sort_values(start_col)
        label_file = output_dir / f"{vid}.txt"

        lines = []
        for _, row in group.iterrows():
            start = float(row[start_col])
            end = float(row[end_col])
            label = str(row[label_col]).strip()
            lines.append(f"{start:.3f}\t{end:.3f}\t{label}")

        label_file.write_text("\n".join(lines), encoding="utf-8")
        print(f"✅ {label_file.name}: {len(group)} labels exported")

    print(f"\n🎧 Done! All files are located in: {output_dir.resolve()}")


# --- CLI entry point ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Converts VOCALS annotations (.xls/.xlsx) to Audacity label files"
    )
    parser.add_argument("--input", "-i", required=True, help="Path to the input Excel file (.xls/.xlsx)")
    parser.add_argument("--output", "-o", default="audacity_labels", help="Target directory for label files")

    args = parser.parse_args()
    xls_to_audacity_labels(args.input, args.output)
