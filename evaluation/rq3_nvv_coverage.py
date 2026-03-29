from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from evaluation.eval_io import write_csv_atomic
from evaluation.eval_metrics import recall
from config.path_factory import (
    get_pipeline_capability_nvv_events_csv_path,
    get_rq3_nvv_coverage_csv_path,
    get_eval_mode_dir,
)


RQ4_REQUIRED_COLUMNS = {
    "mode",
    "audio_id",
    "gt_event_id",
    "gt_label",
    "cand_event_id",
    "cand_label",
    "status",
    "dice_eos",
    "overlap_s",
}

RQ4_FULL_GT_COLUMNS = [
    "mode",
    "label",
    "n_gt_events",
    "tp",
    "fn",
    "recall",
    "mean_dice_eos",
    "mean_overlap_s",
    "fp",
]

RQ4_PART_GT_COLUMNS = [
    "mode",
    "audio_id",
    "gt_event_id",
    "gt_label",
    "cand_event_id",
    "cand_label",
    "status",
    "dice_eos",
    "overlap_s",
]


def _validate_columns(df: pd.DataFrame, required: set[str], path: Path) -> None:
    """
    Validate required columns for RQ4 input.

    Arguments:
        df: Input dataframe.
        required: Required column names.
        path: Input CSV path.
    """
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"{path.name} missing required columns: {missing}")


def _compute_full_gt_coverage(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """
    Compute aggregated RQ3 coverage table for full_gt from final pipeline event rows.

    Arguments:
        df: pipeline_capability_full_gt_nvv_events dataframe.
        mode: Evaluation mode.

    Returns:
        Aggregated coverage dataframe including FP summary row.
    """
    gt_df = df[df["status"].isin(["hit", "miss"])].copy()
    if gt_df.empty:
        raise RuntimeError(f"pipeline_capability_{mode}_nvv_events.csv contains no GT hit/miss rows.")

    rows: List[Dict[str, object]] = []

    for label, label_df in gt_df.groupby("gt_label", dropna=True):
        tp = int((label_df["status"] == "hit").sum())
        fn = int((label_df["status"] == "miss").sum())
        n_gt_events = int(tp + fn)

        hit_df = label_df[label_df["status"] == "hit"].copy()

        mean_dice_eos = (
            float(hit_df["dice_eos"].astype(float).mean())
            if not hit_df.empty
            else 0.0
        )
        mean_overlap_s = (
            float(hit_df["overlap_s"].astype(float).mean())
            if not hit_df.empty
            else 0.0
        )

        rows.append(
            {
                "mode": str(mode),
                "label": str(label),
                "n_gt_events": int(n_gt_events),
                "tp": int(tp),
                "fn": int(fn),
                "recall": float(recall(tp, n_gt_events)),
                "mean_dice_eos": float(mean_dice_eos),
                "mean_overlap_s": float(mean_overlap_s),
                "fp": pd.NA,
            }
        )

    fp_count = int((df["status"] == "insertion").sum())
    rows.append(
        {
            "mode": str(mode),
            "label": "__FP__",
            "n_gt_events": pd.NA,
            "tp": pd.NA,
            "fn": pd.NA,
            "recall": pd.NA,
            "mean_dice_eos": pd.NA,
            "mean_overlap_s": pd.NA,
            "fp": int(fp_count),
        }
    )

    out_df = pd.DataFrame(rows, columns=RQ4_FULL_GT_COLUMNS)
    out_df["label"] = out_df["label"].astype("string")
    return out_df


def _compute_part_gt_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep final pipeline event rows for part_gt semantic analysis.

    Arguments:
        df: pipeline_capability_part_gt_nvv_events dataframe.

    Returns:
        Event-level RQ3 dataframe for part_gt.
    """
    out_df = df.copy()

    out_df["mode"] = out_df["mode"].astype("string")
    out_df["audio_id"] = out_df["audio_id"].astype("string")
    out_df["gt_event_id"] = out_df["gt_event_id"].astype("string")
    out_df["gt_label"] = out_df["gt_label"].astype("string")
    out_df["cand_event_id"] = out_df["cand_event_id"].astype("string")
    out_df["cand_label"] = out_df["cand_label"].astype("string")
    out_df["status"] = out_df["status"].astype("string")

    return out_df.reindex(columns=RQ4_PART_GT_COLUMNS)


def compute_nvv_coverage(
    *,
    evaluation_dir: str | Path,
    mode: str,
    write_file: bool = True,
) -> pd.DataFrame:
    """
    Create the RQ3 NVV coverage artifact from final pipeline event rows.

    Reads:
        <evaluation_dir>/<mode>/pipeline_capability_<mode>_nvv_events.csv

    Writes:
        <evaluation_dir>/<mode>/rq3_nvv_coverage_<mode>.csv

    Arguments:
        evaluation_dir: Workspace evaluation directory.
        mode: "full_gt" or "part_gt".
        write_file: Whether to write the output CSV.

    Returns:
        RQ4 dataframe for the requested mode.
    """
    if mode not in ("full_gt", "part_gt"):
        raise ValueError(f"mode must be 'full_gt' or 'part_gt', got: {mode}")

    evaluation_dir = Path(evaluation_dir)
    in_path = get_pipeline_capability_nvv_events_csv_path(evaluation_dir, mode)

    if not in_path.exists():
        raise FileNotFoundError(f"Missing required file: {in_path}")

    df = pd.read_csv(in_path)
    if df.empty:
        raise RuntimeError(f"{in_path.name} contains 0 rows: {in_path}")

    _validate_columns(df, RQ4_REQUIRED_COLUMNS, in_path)

    if mode == "full_gt":
        out_df = _compute_full_gt_coverage(df, mode=mode)
    else:
        out_df = _compute_part_gt_coverage(df)

    if write_file:
        out_path = get_rq3_nvv_coverage_csv_path(evaluation_dir, mode)
        write_csv_atomic(out_df, out_path)

    return out_df