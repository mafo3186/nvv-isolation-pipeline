#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Optional, Tuple, Union, List

import pandas as pd


PathLikeSeq = Union[Iterable[Path], Mapping[str, Path]]


def merge_gt_excels(
    excel_paths: PathLikeSeq,
    *,
    out_path: Optional[Path] = None,
    validate_columns: bool = True,
    fail_on_duplicates: bool = True,
    verbose: bool = False,
) -> Tuple[pd.DataFrame, Optional[Path]]:
    """
    Read and merge multiple cleaned GT Excel files into a single DataFrame,
    optionally validating schema and writing the merged result to disk.

    Arguments:
        excel_paths:
            Iterable of Paths or a dict mapping names to Paths.
        out_path:
            If provided, write the merged DataFrame to this Excel file.
            If None, no file is written.
        validate_columns:
            If True, enforce identical column names and order across inputs.
        fail_on_duplicates:
            If True, raise if duplicate rows exist in the merged DataFrame.
            Duplicate check is performed on full rows (all columns).
        verbose:
            If True, print basic loading info for each file.

    Returns:
        (merged_df, written_path)
        written_path is None if out_path is None.
    """
    # Normalize input to a list[Path]
    if isinstance(excel_paths, dict):
        paths = [Path(p) for p in excel_paths.values()]
    else:
        paths = [Path(p) for p in excel_paths]

    if not paths:
        raise RuntimeError("No GT excel files provided (excel_paths empty).")

    dfs = []
    for p in paths:
        df = pd.read_excel(p)
        if verbose:
            print(f"Loaded GT file: {p.name} | rows: {len(df)}")
        dfs.append(df)

    if validate_columns:
        cols0 = list(dfs[0].columns)
        for i, df in enumerate(dfs[1:], start=1):
            if list(df.columns) != cols0:
                raise ValueError(
                    "Cannot merge: column mismatch between cleaned excels. "
                    f"First columns={cols0}, file#{i} columns={list(df.columns)}"
                )

    merged = pd.concat(dfs, ignore_index=True)

    if fail_on_duplicates and merged.duplicated().any():
        raise ValueError("Cannot merge: duplicate rows detected in merged GT.")

    written_path: Optional[Path] = None
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_excel(out_path, index=False)
        written_path = out_path

    return merged, written_path

def read_and_merge_gt_excels(cleaned_excel_paths: List[Path]) -> pd.DataFrame:
    dfs: List[pd.DataFrame] = []
    for p in cleaned_excel_paths:
        p = Path(p)
        df = pd.read_excel(p)
        print(f"Loaded GT file: {p.name} | rows: {len(df)}")
        dfs.append(df)

    if not dfs:
        raise RuntimeError("No GT excel files provided (gt_truth_paths empty).")

    return pd.concat(dfs, ignore_index=True)

