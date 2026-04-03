from __future__ import annotations

from pathlib import Path
import pandas as pd
from typing import Mapping

VALID_MODES = {"full_gt", "part_gt"}


def validate_mode(mode: str) -> None:
    """
    Validate evaluation mode.

    Args:
        mode: Evaluation mode string.
    """
    if mode not in VALID_MODES:
        raise ValueError(f"Invalid mode '{mode}'. Expected one of {VALID_MODES}.")
    

def _atomic_replace(src: Path, dst: Path) -> None:
    """Atomically replace dst with src."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    src.replace(dst)


def write_csv_atomic(df: pd.DataFrame, out_path: Path) -> None:
    """
    Write CSV atomically. Always overwrites.
    """
    out_path = Path(out_path)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    df.to_csv(tmp_path, index=False)
    _atomic_replace(tmp_path, out_path)


def write_xlsx_atomic(
    detailed_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    out_path: Path,
    *,
    detailed_sheet: str = "detailed",
    summary_sheet: str = "summary",
) -> None:
    """
    Write XLSX atomically with two sheets. Always overwrites.
    """
    out_path = Path(out_path)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    with pd.ExcelWriter(tmp_path, engine="openpyxl") as writer:
        detailed_df.to_excel(writer, sheet_name=detailed_sheet, index=False)
        summary_df.to_excel(writer, sheet_name=summary_sheet, index=False)

    _atomic_replace(tmp_path, out_path)


def _auto_cast_int_like_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Automatically cast columns to pandas nullable Int64 if all non-null values
    are integer-valued.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with integer-like numeric columns cast to Int64.
    """
    result = df.copy()

    for col in result.columns:
        s = pd.to_numeric(result[col], errors="coerce")

        # Skip columns with no numeric values
        if s.notna().sum() == 0:
            continue

        # Cast only if all non-null numeric values are integers
        if (s.dropna() % 1 == 0).all():
            result[col] = s.astype("Int64")

    return result


def load_csv_or_fail(path: Path) -> pd.DataFrame:
    """
    Load a CSV file and fail if it is missing or empty.

    Args:
        path: CSV path.

    Returns:
        Loaded DataFrame.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Required CSV artifact not found: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"CSV artifact exists but is empty: {path}")

    df = _auto_cast_int_like_columns(df)

    return df


def write_multi_sheet_xlsx_atomic(
    sheets: Mapping[str, pd.DataFrame | Path],
    out_path: Path,
) -> None:
    """
    Write an XLSX file atomically with an arbitrary number of sheets.

    Each value in ``sheets`` may be either:
    - a pandas DataFrame
    - a Path to a CSV file, which will be loaded via load_csv_or_fail()

    Args:
        sheets: Mapping sheet_name -> DataFrame or CSV path.
        out_path: Target XLSX path.
    """
    out_path = Path(out_path)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    with pd.ExcelWriter(tmp_path, engine="openpyxl") as writer:
        for sheet_name, value in sheets.items():
            if isinstance(value, pd.DataFrame):
                df = value
            else:
                df = load_csv_or_fail(Path(value))

            df.to_excel(
                writer,
                sheet_name=str(sheet_name)[:31],
                index=False,
            )

    _atomic_replace(tmp_path, out_path)