from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

from utils.parsing import (
    parse_vad_and_asr_identifier_from_audio_id_filename,
    derive_combo_key,
)
from evaluation.eval_event_matching import match_events_optimal
from evaluation.eval_configuration_tables import (
    build_detailed_rows_from_gt_cand_pairs,
    compute_summary_row_from_detailed,
)
from evaluation.eval_adapter_candidates import load_candidate_events_from_nvv_json


def evaluate_single_candidate_file(
    *,
    audio_id: str,
    candidate_path: Path,
    gt_dict: Dict[str, List[dict]],
    mode: str,
    t_collar: float,
    percentage_of_length: float,
    evaluate_onset: bool,
    evaluate_offset: bool,
    match_labels: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate one *_nvv.json candidate file against GT for one audio_id.

    This function does no Excel IO. GT is passed in via gt_dict.

    Returns:
        detailed_df: TP/FN/FP rows for this track
        summary_df:  one-row summary for this track
    """
    if mode not in ("full_gt", "part_gt"):
        raise ValueError(f"mode must be 'full_gt' or 'part_gt', got: {mode}")

    nvv_file = candidate_path.name
    nvv_file_stem = candidate_path.stem
    vad_mask, asr_audio_in = parse_vad_and_asr_identifier_from_audio_id_filename(audio_id, nvv_file)
    if vad_mask and asr_audio_in:
        combo_key = derive_combo_key(nvv_file_stem, audio_id)

    gt_events = gt_dict.get(audio_id, [])
    cand_events = load_candidate_events_from_nvv_json(candidate_path)

    counts, gt_cand_pairs = match_events_optimal(
        gt_events,
        cand_events,
        match_labels=match_labels,
        evaluate_onset=evaluate_onset,
        evaluate_offset=evaluate_offset,
        t_collar=t_collar,
        percentage_of_length=percentage_of_length,
        gt_onset_key="gt_start_s",
        gt_offset_key="gt_end_s",
        gt_label_key="gt_label",
        gt_id_key="gt_event_id",
        cand_onset_key="cand_start_s",
        cand_offset_key="cand_end_s",
        cand_label_key="cand_label",
        cand_id_key="cand_event_id",
    )

    detailed_df = build_detailed_rows_from_gt_cand_pairs(
        audio_id=audio_id,
        vad_mask=vad_mask,
        asr_audio_in=asr_audio_in,
        combo_key=combo_key,
        nvv_file=nvv_file,
        gt_events=gt_events,
        cand_events=cand_events,
        gt_cand_pairs=gt_cand_pairs,
        gt_id_key="gt_event_id",
        gt_onset_key="gt_start_s",
        gt_offset_key="gt_end_s",
        gt_label_key="gt_label",
        cand_id_key="cand_event_id",
        cand_onset_key="cand_start_s",
        cand_offset_key="cand_end_s",
        cand_label_key="cand_label",
    )

    summary_df = compute_summary_row_from_detailed(
        detailed_df=detailed_df,
        counts=counts,
        mode=mode,
        audio_id=audio_id,
        vad_mask=vad_mask,
        asr_audio_in=asr_audio_in,
        combo_key=combo_key,
        nvv_file=nvv_file,
    )

    return detailed_df, summary_df