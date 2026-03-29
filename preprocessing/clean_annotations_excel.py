#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
clean_annotations_excel.py
---------------------------------------------------------
Cleans VOCALS annotation spreadsheets (.xls / .xlsx)
and overwrites start/end times in the format hh:mm:ss.sss.

• Detects time formats: hh:mm:ss, mm:ss, , . :
• Replaces commas → dots, adds missing hours, fixes extra colons
• Automatically detects header row (line 1–3)
• Preserves all original columns
• Adds parse_status and parse_note columns at the end
---------------------------------------------------------
"""

import pandas as pd
from pathlib import Path
import datetime


# --- Helper: convert seconds to hh:mm:ss.sss ---
def seconds_to_timestamp_str(sec: float) -> str:
    if sec is None or pd.isna(sec):
        return ""
    try:
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s = sec % 60
        return f"{h:02d}:{m:02d}:{s:06.3f}"
    except Exception:
        return ""


# --- Helper: flexible header detection ---
def read_excel_with_flexible_header(path: Path, expected_cols: list[str]) -> pd.DataFrame:
    """
    Reads an Excel file robustly by trying header rows 0–2.
    Detects which row actually contains the column headers.
    """
    for header_row in [0, 1, 2]:
        try:
            df = pd.read_excel(path, header=header_row)
            df.columns = df.columns.str.strip().str.lower()
            expected_lower = [c.lower() for c in expected_cols]
            if all(c in df.columns for c in expected_lower):
                print(f"📄 Header detected in row {header_row + 1}")
                return df
        except Exception as e:
            print(f"⚠️ Error reading with header={header_row}: {e}")
    raise ValueError(f"❌ No valid header row found in {path}")


# --- Main cleaning function ---
def clean_annotations_excel(
    input_path: str | Path,
    output_path: str | Path,
    start_col: str = "start_s",
    end_col: str = "end_s",
    id_col: str = "video_id",
    label_col: str = "vocalization_type",
) -> Path:
    """
    Cleans and normalizes the annotation spreadsheet:
    - Parses and fixes time formats
    - Writes corrected start/end times in hh:mm:ss.sss
    - Appends columns parse_status and parse_note
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"📘 Loading file: {input_path.name}")
    expected_cols = [id_col, start_col, end_col, label_col]
    df = read_excel_with_flexible_header(input_path, expected_cols)

    warnings = []

    def is_empty_cell(value) -> bool:
        if pd.isna(value):
            return True
        return str(value).strip() == ""

    def parse_single_time(value):
        """Parse various (single) time formats into seconds (float). Returns (value, note, status)."""
        if pd.isna(value):
            return None, "empty", "error"
        
        if isinstance(value, (int, float)):
            return float(value), "seconds_numeric", "changed"

        original = str(value).strip()
        if original == "":
            return None, "empty", "error"

        s = original.replace(",", ".")
        notes = []

        try:
            parts = s.split(":")
            parts = [p for p in parts if p != ""]

            if len(parts) == 3:  # hh:mm:ss.sss
                h, m, s_val = parts
            elif len(parts) == 2:  # mm:ss.sss
                h, m, s_val = 0, parts[0], parts[1]
                notes.append("added hh=0")
            elif len(parts) == 4:  # hh:mm:ss:sss
                h, m, s_val = parts[0], parts[1], f"{parts[2]}.{parts[3]}"
                notes.append("extra colon fixed")
            else:
                warnings.append(f"⚠️ Unknown time format: '{value}'")
                return None, "unknown format", "error"

            total = (float(h) * 3600) + (float(m) * 60) + float(s_val)
            if "," in original:
                notes.append("comma→dot")

            note = ", ".join(notes) if notes else "ok"
            status = "changed" if notes else "ok"
            return total, note, status

        except Exception as e:
            warnings.append(f"⚠️ Error parsing '{value}': {e}")
            return None, str(e), "error"
        
    def try_split_two_timestamps(value):
        """
        Try to split a cell that contains two timestamps into (t1, t2).
        Uses parse_single_time() for validation.
        Returns (t1, t2) or (None, None) if not safely splittable.
        """
        if pd.isna(value):
            return None, None
        raw = str(value).strip()
        if raw == "":
            return None, None

        # Split on ANY whitespace (handles unicode spaces)
        tokens = raw.split()
        if len(tokens) < 2:
            return None, None

        t1, t2 = tokens[0], tokens[1]

        p1 = parse_single_time(t1)[0]
        p2 = parse_single_time(t2)[0]
        if p1 is None or p2 is None:
            return None, None
        if not (p1 < p2):
            return None, None

        return t1, t2
    

    df["_row_fix_applied"] = False
    df["_row_fix_note"] = ""

    # function to run splitting logic on both start and end columns and un update or return note/status
    def handle_combined_timestamps(row):
        for col in [start_col, end_col]:
            if is_empty_cell(row[col]):
                other_col = end_col if col == start_col else start_col
                if not is_empty_cell(row[other_col]):
                    t1, t2 = try_split_two_timestamps(row[other_col])
                    if t1 and t2:
                        row[start_col] = t1
                        row[end_col] = t2
                        row["_row_fix_applied"] = True
                        row["_row_fix_note"] = "row-repair: split combined timestamps"
                        return row
        return row
    
    # handle combined timestamps before parsing individual times 
    df = df.apply(handle_combined_timestamps, axis=1)

    # Convert time columns 
    start_parsed = df[start_col].apply(parse_single_time)
    end_parsed = df[end_col].apply(parse_single_time)

    # Write corrected timestamps
    df[start_col] = start_parsed.apply(lambda x: seconds_to_timestamp_str(x[0]))
    df[end_col] = end_parsed.apply(lambda x: seconds_to_timestamp_str(x[0]))

    # Determine parse_status and parse_note based on parsing results and any row fixes applied
    df["parse_status"] = [
        "error" if ("error" in (s1, s2)) else
        ("changed" if (("changed" in (s1, s2)) or row_fix_applied) else "ok")
        for s1, s2, row_fix_applied in zip(
            start_parsed.apply(lambda x: x[2]),
            end_parsed.apply(lambda x: x[2]),
            df["_row_fix_applied"]
        )
    ]
    df["parse_note"] = [
        ", ".join(
            sorted(set([n for n in (n1, n2, row_fix_note) if n not in (None, "", "ok")]))
        ) or "ok"
        for n1, n2, row_fix_note in zip(
            start_parsed.apply(lambda x: x[1]),
            end_parsed.apply(lambda x: x[1]),
            df["_row_fix_note"]
        )
    ]
    df.drop(columns=["_row_fix_applied", "_row_fix_note"], inplace=True)

    # Save cleaned file
    df.to_excel(output_path, index=False)

    # Summary
    total_ok = (df["parse_status"] == "ok").sum()
    total_changed = (df["parse_status"] == "changed").sum()
    total_error = (df["parse_status"] == "error").sum()
    print(f"✅ Cleaned file saved as: {output_path.resolve()}")
    print(f"📊 Summary: {total_ok} ok · {total_changed} changed · {total_error} error")

    if warnings:
        print(f"📋 {len(warnings)} warnings (examples):")
        for w in warnings[:10]:
            print("   ", w)
        if len(warnings) > 10:
            print(f"   … ({len(warnings)-10} more)")

    return output_path


# CLI
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Cleans VOCALS annotation spreadsheets (.xls/.xlsx).")
    parser.add_argument("--input", "-i", required=True, help="Path to the Excel file (.xls/.xlsx)")
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Full output file path for cleaned Excel (must match YAML cleaned_excel_rel_path)."
    )

    args = parser.parse_args()

    clean_annotations_excel(
        input_path=args.input,
        output_path=args.output,
        id_col=args.id_col,
    )
