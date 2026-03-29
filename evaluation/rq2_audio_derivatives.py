from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from evaluation.eval_io import write_csv_atomic
from config.path_factory import (
    get_global_combo_ranking_csv_path,
    get_rq2_audio_derivatives_csv_path,
)


RQ3_REQUIRED_COLUMNS_BASE = {
    "mode",
    "combo_key",
    "vad_mask",
    "asr_audio_in",
    "macro_mean_recall",
    "macro_mean_mean_dice_eos_tp",
    "macro_mean_n_cand",
    "macro_mean_fp",
}

RQ3_REQUIRED_COLUMNS_FULL_GT = {
    "macro_mean_f1",
}


def _map_audio_derivative_group(asr_audio_in: str) -> str:
    """
    Map asr_audio_in to the configured derivative group.

    Arguments:
        asr_audio_in: ASR input audio derivative name.

    Returns:
        Group name for RQ3 aggregation.

    Raises:
        ValueError: If the derivative is unknown.
    """
    value = str(asr_audio_in)

    if value in {"original", "std"}:
        return "original_like"
    if value in {"std_vocals", "std_vocals_norm"}:
        return "vocals_like"
    if value in {"std_background", "std_background_norm"}:
        return "background_like"

    raise ValueError(f"Unknown asr_audio_in for RQ3 grouping: {value!r}")


def _validate_ranking_columns(df: pd.DataFrame, mode: str, path: Path) -> None:
    """
    Validate required columns for RQ3 ranking input.

    Arguments:
        df: Ranking dataframe.
        mode: Evaluation mode.
        path: Input CSV path.
    """
    required = set(RQ3_REQUIRED_COLUMNS_BASE)
    if mode == "full_gt":
        required |= RQ3_REQUIRED_COLUMNS_FULL_GT

    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"{path.name} missing required columns: {missing}")


def _select_best_row(group_df: pd.DataFrame, mode: str) -> pd.Series:
    """
    Select the best config row inside one derivative group.

    Arguments:
        group_df: Group-specific ranking dataframe.
        mode: Evaluation mode.

    Returns:
        Best row according to the mode-specific primary metric.
    """
    if mode == "full_gt":
        sort_col = "macro_mean_f1"
    elif mode == "part_gt":
        sort_col = "macro_mean_recall"
    else:
        raise ValueError(f"mode must be 'full_gt' or 'part_gt', got: {mode}")

    if sort_col not in group_df.columns:
        raise KeyError(f"Missing required primary metric column: {sort_col}")

    best_idx = group_df[sort_col].astype(float).idxmax()
    return group_df.loc[best_idx]


def _aggregate_group(group_df: pd.DataFrame, mode: str, group_name: str) -> Dict[str, object]:
    """
    Aggregate one derivative group for RQ3.

    Arguments:
        group_df: Ranking rows belonging to one derivative group.
        mode: Evaluation mode.
        group_name: Group label.

    Returns:
        One output row as a dict.
    """
    best_row = _select_best_row(group_df, mode=mode)

    row: Dict[str, object] = {
        "mode": str(mode),
        "audio_derivative_group": str(group_name),
        "n_configs": int(group_df.shape[0]),
        "best_combo_key": str(best_row["combo_key"]),
        "best_vad_mask": str(best_row["vad_mask"]),
        "best_asr_audio_in": str(best_row["asr_audio_in"]),
        "macro_mean_recall": float(group_df["macro_mean_recall"].astype(float).mean()),
        "macro_mean_mean_dice_eos_tp": float(group_df["macro_mean_mean_dice_eos_tp"].astype(float).mean()),
        "best_macro_mean_recall": float(best_row["macro_mean_recall"]),
        "best_macro_mean_mean_dice_eos_tp": float(best_row["macro_mean_mean_dice_eos_tp"]),
        "macro_mean_n_cand": float(group_df["macro_mean_n_cand"].astype(float).mean()),
        "macro_mean_fp": float(group_df["macro_mean_fp"].astype(float).mean()),
    }

    if mode == "full_gt":
        row["macro_mean_f1"] = float(group_df["macro_mean_f1"].astype(float).mean())
        row["best_macro_mean_f1"] = float(best_row["macro_mean_f1"])

    return row


def rank_audio_derivatives(
    *,
    evaluation_dir: str | Path,
    mode: str,
    write_file: bool = True,
) -> pd.DataFrame:
    """
    Create the RQ3 audio-derivative aggregation from global combo ranking.

    Reads:
        <evaluation_dir>/<mode>/global_combo_ranking_<mode>.csv

    Writes:
        <evaluation_dir>/<mode>/rq2_config_audio_derivatives_<mode>.csv

    Arguments:
        evaluation_dir: Workspace evaluation directory.
        mode: "full_gt" or "part_gt".
        write_file: Whether to write the output CSV.

    Returns:
        Aggregated RQ3 dataframe.
    """
    if mode not in ("full_gt", "part_gt"):
        raise ValueError(f"mode must be 'full_gt' or 'part_gt', got: {mode}")

    evaluation_dir = Path(evaluation_dir)
    in_path = get_global_combo_ranking_csv_path(evaluation_dir, mode)

    if not in_path.exists():
        raise FileNotFoundError(f"Missing required ranking file: {in_path}")

    df = pd.read_csv(in_path)
    if df.empty:
        raise RuntimeError(f"{in_path.name} contains 0 rows: {in_path}")

    _validate_ranking_columns(df, mode=mode, path=in_path)

    df = df[df["mode"].astype(str) == str(mode)].copy()
    if df.empty:
        raise RuntimeError(f"{in_path.name} contains no rows for mode={mode!r}")

    df["audio_derivative_group"] = df["asr_audio_in"].astype(str).map(_map_audio_derivative_group)

    ordered_groups: List[str] = [
        "original_like",
        "vocals_like",
        "background_like",
        "all_derivatives",
    ]

    rows: List[Dict[str, object]] = []

    for group_name in ordered_groups:
        if group_name == "all_derivatives":
            group_df = df.copy()
        else:
            group_df = df[df["audio_derivative_group"] == group_name].copy()

        if group_df.empty:
            continue

        rows.append(_aggregate_group(group_df, mode=mode, group_name=group_name))

    out_df = pd.DataFrame(rows)

    base_columns = [
        "mode",
        "audio_derivative_group",
        "n_configs",
        "best_combo_key",
        "best_vad_mask",
        "best_asr_audio_in",
        "macro_mean_recall",
        "macro_mean_mean_dice_eos_tp",
        "best_macro_mean_recall",
        "best_macro_mean_mean_dice_eos_tp",
        "macro_mean_n_cand",
        "macro_mean_fp",
    ]

    if mode == "full_gt":
        columns = base_columns[:6] + [
            "macro_mean_f1",
            "best_macro_mean_f1",
        ] + base_columns[6:]
    else:
        columns = base_columns

    out_df = out_df.reindex(columns=columns)

    if write_file:
        out_path = get_rq2_audio_derivatives_csv_path(evaluation_dir, mode)
        write_csv_atomic(out_df, out_path)

    return out_df