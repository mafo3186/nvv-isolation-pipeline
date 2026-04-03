from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from evaluation.eval_io import write_csv_atomic
from evaluation.eval_metrics import recall, precision, f1, insertion_rate, deletion_rate
from config.path_factory import (
    get_pipeline_capability_nvv_events_csv_path,
    get_rq3_nvv_coverage_label_csv_path,
    get_rq3_nvv_coverage_global_csv_path,
)


RQ3_REQUIRED_COLUMNS = {
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

RQ3_LABEL_FULL_GT_COLUMNS = [
    "mode",
    "label",
    "n_gt_events",
    "tp",
    "fn",
    "recall",
    "dice_eos_recall",
    "mean_dice_eos_tp",
    "mean_overlap_s",
]

RQ3_LABEL_PART_GT_COLUMNS = [
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

RQ3_GLOBAL_FULL_GT_COLUMNS = [
    "mode",
    "n_gt_events_total",
    "tp_total",
    "fn_total",
    "insertions_total",
    "f1",
    "recall",
    "dice_eos_recall",
    "mean_dice_eos_tp",
    "insertion_rate",
    "deletion_rate",
]

RQ3_GLOBAL_PART_GT_COLUMNS = [
    "mode",
    "n_gt_events_total",
    "tp_total",
    "fn_total",
    "insertions_total",
    "recall",
    "dice_eos_recall",
    "mean_dice_eos_tp",
    "insertion_rate",
    "deletion_rate",
]


def _validate_columns(df: pd.DataFrame, required: set[str], path: Path) -> None:
    """
    Validate required columns for RQ3 input.

    Arguments:
        df: Input dataframe.
        required: Required column names.
        path: Input CSV path.
    """
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"{path.name} missing required columns: {missing}")


def _compute_full_gt_coverage_label(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """
    Compute aggregated RQ3 coverage table for full_gt from final pipeline event rows.

    Arguments:
        df: pipeline_capability_full_gt_nvv_events dataframe.
        mode: Evaluation mode.

    Returns:
        Aggregated coverage dataframe by label.
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

        mean_dice_eos_tp = (
            float(hit_df["dice_eos"].astype(float).mean())
            if not hit_df.empty
            else 0.0
        )

        dice_eos_recall = (
            float(hit_df["dice_eos"].astype(float).sum()) / float(n_gt_events)
            if n_gt_events > 0
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
                "dice_eos_recall": float(dice_eos_recall),
                "mean_dice_eos_tp": float(mean_dice_eos_tp),
                "mean_overlap_s": float(mean_overlap_s),
            }
        )

    out_df = pd.DataFrame(rows, columns=RQ3_LABEL_FULL_GT_COLUMNS)
    out_df["label"] = out_df["label"].astype("string")
    return out_df


def _compute_part_gt_coverage_label(df: pd.DataFrame) -> pd.DataFrame:
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

    return out_df.reindex(columns=RQ3_LABEL_PART_GT_COLUMNS)


def _compute_global_coverage(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """
    Compute global RQ3 coverage summary from final pipeline event rows.

    Arguments:
        df: pipeline_capability_<mode>_nvv_events dataframe.
        mode: Evaluation mode.

    Returns:
        One-row global coverage dataframe.
    """
    tp_total = int((df["status"] == "hit").sum())
    fn_total = int((df["status"] == "miss").sum())
    insertions_total = int((df["status"] == "insertion").sum())
    n_gt_events_total = int(tp_total + fn_total)

    hit_df = df[df["status"] == "hit"].copy()

    mean_dice_eos_tp = (
        float(hit_df["dice_eos"].astype(float).mean())
        if not hit_df.empty
        else 0.0
    )

    dice_eos_recall = (
        float(hit_df["dice_eos"].astype(float).sum()) / float(n_gt_events_total)
        if n_gt_events_total > 0
        else 0.0
    )

    recall_value = float(recall(tp_total, n_gt_events_total))
    insertion_rate_value = float(insertion_rate(insertions_total, n_gt_events_total))
    deletion_rate_value = float(deletion_rate(fn_total, n_gt_events_total))

    if mode == "full_gt":
        precision_value = float(precision(tp_total, tp_total + insertions_total))
        f1_value = float(f1(precision_value, recall_value))

        out_df = pd.DataFrame(
            [
                {
                    "mode": str(mode),
                    "n_gt_events_total": int(n_gt_events_total),
                    "tp_total": int(tp_total),
                    "fn_total": int(fn_total),
                    "insertions_total": int(insertions_total),
                    "f1": float(f1_value),
                    "recall": float(recall_value),
                    "dice_eos_recall": float(dice_eos_recall),
                    "mean_dice_eos_tp": float(mean_dice_eos_tp),
                    "insertion_rate": float(insertion_rate_value),
                    "deletion_rate": float(deletion_rate_value),
                }
            ],
            columns=RQ3_GLOBAL_FULL_GT_COLUMNS,
        )
        return out_df

    if mode == "part_gt":
        out_df = pd.DataFrame(
            [
                {
                    "mode": str(mode),
                    "n_gt_events_total": int(n_gt_events_total),
                    "tp_total": int(tp_total),
                    "fn_total": int(fn_total),
                    "insertions_total": int(insertions_total),
                    "recall": float(recall_value),
                    "dice_eos_recall": float(dice_eos_recall),
                    "mean_dice_eos_tp": float(mean_dice_eos_tp),
                    "insertion_rate": float(insertion_rate_value),
                    "deletion_rate": float(deletion_rate_value),
                }
            ],
            columns=RQ3_GLOBAL_PART_GT_COLUMNS,
        )
        return out_df

    raise ValueError(f"Unsupported mode: {mode}")


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
        <evaluation_dir>/<mode>/rq3_nvv_coverage_label_<mode>.csv
        <evaluation_dir>/<mode>/rq3_nvv_coverage_global_<mode>.csv

    Arguments:
        evaluation_dir: Workspace evaluation directory.
        mode: "full_gt" or "part_gt".
        write_file: Whether to write the output CSVs.

    Returns:
        RQ3 label/event dataframe for the requested mode.
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

    _validate_columns(df, RQ3_REQUIRED_COLUMNS, in_path)

    if mode == "full_gt":
        label_df = _compute_full_gt_coverage_label(df, mode=mode)
    else:
        label_df = _compute_part_gt_coverage_label(df)

    global_df = _compute_global_coverage(df, mode=mode)

    if write_file:
        label_out_path = get_rq3_nvv_coverage_label_csv_path(evaluation_dir, mode)
        global_out_path = get_rq3_nvv_coverage_global_csv_path(evaluation_dir, mode)

        write_csv_atomic(label_df, label_out_path)
        write_csv_atomic(global_df, global_out_path)

    return label_df