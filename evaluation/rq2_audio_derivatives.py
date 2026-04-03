from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from evaluation.eval_io import write_csv_atomic
from config.path_factory import (
    get_global_combo_ranking_csv_path,
    get_rq2_audio_derivatives_csv_path,
)
from evaluation.analysis_metrics import (
    get_ordered_metric_names,
    get_metric_sort_ascending,
)


RQ2_REQUIRED_COLUMNS_BASE = {
    "mode",
    "combo_key",
    "vad_mask",
    "asr_audio_in",
    "macro_mean_recall",
    "macro_mean_dice_eos_recall",
    "macro_mean_mean_dice_eos_tp",
    "macro_mean_n_cand",
    "macro_mean_fp",
    "macro_mean_insertion_rate",
    "macro_mean_deletion_rate",
    "macro_mean_error_rate",
}

RQ2_REQUIRED_COLUMNS_FULL_GT = {
    "macro_mean_f1",
}


def _map_audio_derivative_group(asr_audio_in: str) -> str:
    """
    Map asr_audio_in to the configured derivative group.

    Arguments:
        asr_audio_in: ASR input audio derivative name.

    Returns:
        Group name for RQ2 aggregation.

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

    raise ValueError(f"Unknown asr_audio_in for RQ2 grouping: {value!r}")


def _validate_ranking_columns(df: pd.DataFrame, mode: str, path: Path) -> None:
    """
    Validate required columns for RQ2 ranking input.

    Arguments:
        df: Ranking dataframe.
        mode: Evaluation mode.
        path: Input CSV path.
    """
    required = set(RQ2_REQUIRED_COLUMNS_BASE)
    if mode == "full_gt":
        required |= RQ2_REQUIRED_COLUMNS_FULL_GT

    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"{path.name} missing required columns: {missing}")


def _select_best_row(group_df: pd.DataFrame, mode: str) -> pd.Series:
    """
    Select the best config row inside one derivative group.

    Uses the canonical metric order and sort directions from
    evaluation.analysis_metrics.

    Args:
        group_df: Group-specific ranking dataframe.
        mode: Evaluation mode.

    Returns:
        Best row according to the canonical ranking logic.
    """
    metric_cols = get_ordered_metric_names(mode)

    missing = [col for col in metric_cols if col not in group_df.columns]
    if missing:
        raise KeyError(
            f"Missing required ranking columns for derivative-group selection: {missing}"
        )

    ascending = [get_metric_sort_ascending(col) for col in metric_cols]

    sorted_df = group_df.sort_values(
        by=metric_cols,
        ascending=ascending,
        na_position="last",
    ).reset_index(drop=True)

    if sorted_df.empty:
        raise RuntimeError("Cannot select best row from an empty derivative group.")

    return sorted_df.iloc[0]


def _aggregate_group(group_df: pd.DataFrame, mode: str, group_name: str) -> Dict[str, object]:
    """
    Aggregate one derivative group for RQ2.

    Args:
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
        "macro_mean_dice_eos_recall": float(group_df["macro_mean_dice_eos_recall"].astype(float).mean()),
        "macro_mean_mean_dice_eos_tp": float(group_df["macro_mean_mean_dice_eos_tp"].astype(float).mean()),
        "best_macro_mean_recall": float(best_row["macro_mean_recall"]),
        "best_macro_mean_dice_eos_recall": float(best_row["macro_mean_dice_eos_recall"]),
        "best_macro_mean_mean_dice_eos_tp": float(best_row["macro_mean_mean_dice_eos_tp"]),
        "macro_mean_n_cand": float(group_df["macro_mean_n_cand"].astype(float).mean()),
        "macro_mean_fp": float(group_df["macro_mean_fp"].astype(float).mean()),
        "macro_mean_insertion_rate": float(group_df["macro_mean_insertion_rate"].astype(float).mean()),
        "best_macro_mean_insertion_rate": float(best_row["macro_mean_insertion_rate"]),
        "macro_mean_deletion_rate": float(group_df["macro_mean_deletion_rate"].astype(float).mean()),
        "best_macro_mean_deletion_rate": float(best_row["macro_mean_deletion_rate"]),
        "macro_mean_error_rate": float(group_df["macro_mean_error_rate"].astype(float).mean()),
        "best_macro_mean_error_rate": float(best_row["macro_mean_error_rate"]),
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
    Create the RQ2 audio-derivative aggregation from global combo ranking.

    Reads:
        <evaluation_dir>/<mode>/global_combo_ranking_<mode>.csv

    Writes:
        <evaluation_dir>/<mode>/rq2_config_audio_derivatives_<mode>.csv

    Arguments:
        evaluation_dir: Workspace evaluation directory.
        mode: "full_gt" or "part_gt".
        write_file: Whether to write the output CSV.

    Returns:
        Aggregated RQ2 dataframe.
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
        "macro_mean_dice_eos_recall",
        "macro_mean_mean_dice_eos_tp",
        "macro_mean_insertion_rate",
        "macro_mean_deletion_rate",
        "macro_mean_error_rate",
        "best_macro_mean_recall",
        "best_macro_mean_dice_eos_recall",
        "best_macro_mean_mean_dice_eos_tp",
        "best_macro_mean_insertion_rate",
        "best_macro_mean_deletion_rate",
        "best_macro_mean_error_rate",
        "macro_mean_n_cand",
        "macro_mean_fp",
    ]

    if mode == "full_gt":
        columns = base_columns[:6] + [
            "macro_mean_f1",
        ] + base_columns[6:12] + [
            "best_macro_mean_f1",
        ] + base_columns[12:]
    else:
        columns = base_columns

    out_df = out_df.reindex(columns=columns)

    if write_file:
        out_path = get_rq2_audio_derivatives_csv_path(evaluation_dir, mode)
        write_csv_atomic(out_df, out_path)

    return out_df