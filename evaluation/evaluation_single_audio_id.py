from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd

from evaluation.evaluation_single_candidate_file import evaluate_single_candidate_file
from config.path_factory import get_nvv_dir


def evaluate_single_audio_id(
    *,
    workspace_root: Path,
    audio_id: str,
    gt_dict: Dict[str, List[dict]],
    mode: str,
    t_collar: float,
    percentage_of_length: float,
    evaluate_onset: bool,
    evaluate_offset: bool,
    match_labels: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate all candidate files for one audio_id.

    Candidate folder convention:
        workspace_root/per_audio/<audio_id>/annotations/nvv/*.json

    Returns:
        detailed_df: all TP/FN/FP rows across all candidate files of this audio_id
        summary_df:  one summary row per candidate file (track)
    """
    nvv_dir = get_nvv_dir(workspace_root, audio_id)

    candidate_paths = sorted(nvv_dir.glob("*.json"))

    detailed_parts: List[pd.DataFrame] = []
    summary_parts: List[pd.DataFrame] = []

    for candidate_path in candidate_paths:
        detailed_df, summary_df = evaluate_single_candidate_file(
            audio_id=audio_id,
            candidate_path=candidate_path,
            gt_dict=gt_dict,
            mode=mode,
            t_collar=t_collar,
            percentage_of_length=percentage_of_length,
            evaluate_onset=evaluate_onset,
            evaluate_offset=evaluate_offset,
            match_labels=match_labels,
        )
        detailed_parts.append(detailed_df)
        summary_parts.append(summary_df)

    if detailed_parts:
        detailed_all = pd.concat(detailed_parts, ignore_index=True)
    else:
        detailed_all = pd.DataFrame()

    if summary_parts:
        summary_all = pd.concat(summary_parts, ignore_index=True)
    else:
        summary_all = pd.DataFrame()

    return detailed_all, summary_all