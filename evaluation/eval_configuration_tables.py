from __future__ import annotations

from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd

from evaluation.eval_metrics import (
    overlap_seconds,
    dice_event_overlap_score,
    MatchCounts,
    full_gt_metrics,
    partial_gt_metrics,
)


DETAILED_COLUMNS = [
    "audio_id",
    "vad_mask",
    "asr_audio_in",
    "nvv_file",
    "combo_key",
    "gt_event_id",
    "gt_start_s",
    "gt_end_s",
    "gt_label",
    "match_type",
    "cand_event_id",
    "cand_start_s",
    "cand_end_s",
    "cand_label",
    "dice_eos",
    "overlap_s",
]

SUMMARY_COLUMNS = [
    "audio_id",
    "vad_mask",
    "asr_audio_in",
    "nvv_file",
    "combo_key",
    "mode",
    "n_gt",
    "n_cand",
    "tp",
    "fn",
    "fp",
    "mean_dice_eos_tp",
    "dice_eos_recall",
    "mean_overlap_s_tp",
    "insertion_rate",
    "deletion_rate",
    "error_rate",
    "precision",
    "recall",
    "f1",
]


def build_detailed_rows_from_gt_cand_pairs(
    *,
    audio_id: str,
    vad_mask: str,
    asr_audio_in: str,
    nvv_file: str,
    combo_key: str,
    gt_events: List[dict],
    cand_events: List[dict],
    gt_cand_pairs: List[Tuple[int, int]],
    gt_id_key: str = "gt_event_id",
    gt_onset_key: str = "gt_start_s",
    gt_offset_key: str = "gt_end_s",
    gt_label_key: str = "gt_label",
    cand_id_key: str = "cand_event_id",
    cand_onset_key: str = "cand_start_s",
    cand_offset_key: str = "cand_end_s",
    cand_label_key: str = "cand_label",
) -> pd.DataFrame:
    """
    Build detailed rows for one track from optimal 1:1 gt_cand_pairs.

    Produces TP rows from gt_cand_pairs, plus FN rows (unmatched GT) and
    FP rows (unmatched candidates). No IO, only list/dict lookups
    FN: dice_eos=0.0, overlap_s=0.0
    FP: dice_eos=NaN, overlap_s=NaN (undefined without GT reference)
    
    Arguments:
        audio_id: Audio identifier.
        vad_mask: VAD mask identifier.
        asr_audio_in: ASR input identifier.
        nvv_file: Candidate filename.
        combo_key: Combined track identifier.
        gt_events: GT event list.
        cand_events: Candidate event list.
        gt_cand_pairs: Matched GT/candidate index pairs.
        gt_id_key: GT event id key.
        gt_onset_key: GT onset key.
        gt_offset_key: GT offset key.
        gt_label_key: GT label key.
        cand_id_key: Candidate event id key.
        cand_onset_key: Candidate onset key.
        cand_offset_key: Candidate offset key.
        cand_label_key: Candidate label key.

    Returns:
        Detailed TP/FN/FP dataframe for one candidate track.
    """
    matched_gt = {gi for gi, _ in gt_cand_pairs}
    matched_cand = {ci for _, ci in gt_cand_pairs}

    rows: List[Dict[str, Any]] = []

    def base_fields() -> Dict[str, Any]:
        return {
            "audio_id": audio_id,
            "vad_mask": vad_mask,
            "asr_audio_in": asr_audio_in,
            "nvv_file": nvv_file,
            "combo_key": combo_key,
        }

    # TP rows
    for gi, ci in gt_cand_pairs:
        gt = gt_events[gi]
        cand = cand_events[ci]

        gt_start = float(gt[gt_onset_key])
        gt_end = float(gt[gt_offset_key])
        cand_start = float(cand[cand_onset_key])
        cand_end = float(cand[cand_offset_key])

        overlap = overlap_seconds(gt_start, gt_end, cand_start, cand_end)
        dice = dice_event_overlap_score(gt_start, gt_end, cand_start, cand_end)

        row = {
            **base_fields(),
            "gt_event_id": gt.get(gt_id_key, np.nan),
            "gt_start_s": gt_start,
            "gt_end_s": gt_end,
            "gt_label": gt.get(gt_label_key, np.nan),
            "match_type": "TP",
            "cand_event_id": cand.get(cand_id_key, np.nan),
            "cand_start_s": cand_start,
            "cand_end_s": cand_end,
            "cand_label": cand.get(cand_label_key, np.nan),
            "dice_eos": float(dice),
            "overlap_s": float(overlap),
        }
        rows.append(row)

    # FN rows (unmatched GT)
    for gi, gt in enumerate(gt_events):
        if gi in matched_gt:
            continue

        gt_start = float(gt[gt_onset_key])
        gt_end = float(gt[gt_offset_key])

        row = {
            **base_fields(),
            "gt_event_id": gt.get(gt_id_key, np.nan),
            "gt_start_s": gt_start,
            "gt_end_s": gt_end,
            "gt_label": gt.get(gt_label_key, np.nan),
            "match_type": "FN",
            "cand_event_id": np.nan,
            "cand_start_s": np.nan,
            "cand_end_s": np.nan,
            "cand_label": np.nan,
            "dice_eos": 0.0,
            "overlap_s": 0.0,
        }
        rows.append(row)

    # FP rows (unmatched candidates)
    for ci, cand in enumerate(cand_events):
        if ci in matched_cand:
            continue

        cand_start = float(cand[cand_onset_key])
        cand_end = float(cand[cand_offset_key])

        row = {
            **base_fields(),
            "gt_event_id": np.nan,
            "gt_start_s": np.nan,
            "gt_end_s": np.nan,
            "gt_label": np.nan,
            "match_type": "FP",
            "cand_event_id": cand.get(cand_id_key, np.nan),
            "cand_start_s": cand_start,
            "cand_end_s": cand_end,
            "cand_label": cand.get(cand_label_key, np.nan),
            "dice_eos": np.nan,
            "overlap_s": np.nan,
        }
        rows.append(row)

    return pd.DataFrame(rows, columns=DETAILED_COLUMNS)


def compute_summary_row_from_detailed(
    *,
    detailed_df: pd.DataFrame,
    counts: MatchCounts,
    mode: str,
    audio_id: str,
    vad_mask: str,
    asr_audio_in: str,
    combo_key: str,
    nvv_file: str,
) -> pd.DataFrame:
    """
    Compute one summary row for one track from detailed_df + MatchCounts.

    Reporting-specific overlap metrics are derived from detailed_df.
    Classic count-based metrics are derived from evaluation_metrics.py.

    Arguments:
        detailed_df: Detailed TP/FN/FP dataframe for one track.
        counts: Match count container.
        mode: "full_gt" or "part_gt".
        audio_id: Audio identifier.
        vad_mask: VAD mask identifier.
        asr_audio_in: ASR input identifier.
        combo_key: Combined track identifier.
        nvv_file: Candidate filename.

    Returns:
        One-row summary dataframe.
    """
    if mode not in ("full_gt", "part_gt"):
        raise ValueError(f"mode must be 'full_gt' or 'part_gt', got: {mode}")

    tp_mask = detailed_df["match_type"] == "TP"
    gt_mask = detailed_df["match_type"].isin(["TP", "FN"])

    mean_dice_tp = float(detailed_df.loc[tp_mask, "dice_eos"].mean()) if counts.tp > 0 else 0.0
    mean_overlap_tp = float(detailed_df.loc[tp_mask, "overlap_s"].mean()) if counts.tp > 0 else 0.0

    if counts.n_gt > 0:
        dice_eos_recall = (
            float(detailed_df.loc[gt_mask, "dice_eos"].fillna(0.0).sum()) / float(counts.n_gt)
        )
    else:
        dice_eos_recall = 0.0

    if mode == "full_gt":
        classic = full_gt_metrics(counts)
        precision_val = classic["precision"]
        recall_val = classic["recall"]
        f1_val = classic["f1"]
        insertion_rate_val = classic["insertion_rate"]
        deletion_rate_val = classic["deletion_rate"]
        error_rate_val = classic["error_rate"]
    else:
        partial = partial_gt_metrics(counts)
        precision_val = np.nan
        recall_val = partial["recall"]
        f1_val = np.nan
        insertion_rate_val = partial["insertion_rate"]
        deletion_rate_val = partial["deletion_rate"]
        error_rate_val = np.nan  


    row = {
        "audio_id": audio_id,
        "vad_mask": vad_mask,
        "asr_audio_in": asr_audio_in,
        "nvv_file": nvv_file,
        "combo_key": combo_key,
        "mode": mode,
        "n_gt": counts.n_gt,
        "n_cand": counts.n_cand,
        "tp": counts.tp,
        "fn": counts.fn,
        "fp": counts.fp,
        "mean_dice_eos_tp": mean_dice_tp,
        "dice_eos_recall": dice_eos_recall,
        "mean_overlap_s_tp": mean_overlap_tp,
        "insertion_rate": insertion_rate_val,
        "deletion_rate": deletion_rate_val,
        "error_rate": error_rate_val,
        "precision": precision_val,
        "recall": recall_val,
        "f1": f1_val,
    }

    return pd.DataFrame([row], columns=SUMMARY_COLUMNS)