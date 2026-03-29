from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

from config.path_factory import (
    get_per_audio_evaluation_dir,
    get_per_audio_evaluation_mode_dir,
    get_per_audio_detailed_csv_path,
    get_per_audio_summary_csv_path,
    get_per_audio_evaluation_xlsx_path,
)
from evaluation.eval_io import write_csv_atomic, write_xlsx_atomic
from evaluation.evaluation_single_audio_id import evaluate_single_audio_id


def write_audio_id_results_atomically(
    *,
    workspace_root: Path,
    audio_id: str,
    mode: str,
    detailed_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    write_csv: bool = True,
    write_xlsx: bool = True,
) -> None:
    mode_dir = get_per_audio_evaluation_mode_dir(workspace_root, audio_id, mode)
    mode_dir.mkdir(parents=True, exist_ok=True)

    if write_csv:
        write_csv_atomic(
            detailed_df,
            get_per_audio_detailed_csv_path(workspace_root, audio_id, mode),
        )
        write_csv_atomic(
            summary_df,
            get_per_audio_summary_csv_path(workspace_root, audio_id, mode),
        )

    if write_xlsx:
        write_xlsx_atomic(
            detailed_df=detailed_df,
            summary_df=summary_df,
            out_path=get_per_audio_evaluation_xlsx_path(workspace_root, audio_id, mode),
        )


def evaluate_workspace(
    *,
    workspace_root: Path,
    evaluable_ids: List[str],
    gt_dict: Dict[str, List[dict]],
    mode: str,
    t_collar: float,
    percentage_of_length: float,
    evaluate_onset: bool,
    evaluate_offset: bool,
    match_labels: bool = False,
    write_per_audio: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate all evaluable audio_ids in one workspace root.

    Writes per-audio evaluation results to:
        <workspace_root>/<KEY_PER_AUDIO>/<audio_id>/<KEY_EVALUATION>/<mode>/
            <audio_id>_detailed_<mode>.csv
            <audio_id>_summary_<mode>.csv
            <audio_id>_evaluation_<mode>.xlsx

    Returns:
        (detailed_all, summary_all) concatenated across ids (in-memory only).
    """
    workspace_root = Path(workspace_root)

    detailed_parts: List[pd.DataFrame] = []
    summary_parts: List[pd.DataFrame] = []

    for audio_id in evaluable_ids:
        detailed_df, summary_df = evaluate_single_audio_id(
            workspace_root=workspace_root,
            audio_id=audio_id,
            gt_dict=gt_dict,
            mode=mode,
            t_collar=t_collar,
            percentage_of_length=percentage_of_length,
            evaluate_onset=evaluate_onset,
            evaluate_offset=evaluate_offset,
            match_labels=match_labels,
        )

        if write_per_audio:
            write_audio_id_results_atomically(
                workspace_root=workspace_root,
                audio_id=audio_id,
                mode=mode,
                detailed_df=detailed_df,
                summary_df=summary_df,
                write_csv=True,
                write_xlsx=True,
            )

        detailed_parts.append(detailed_df)
        summary_parts.append(summary_df)

    detailed_all = pd.concat(detailed_parts, ignore_index=True) if detailed_parts else pd.DataFrame()
    summary_all = pd.concat(summary_parts, ignore_index=True) if summary_parts else pd.DataFrame()

    return detailed_all, summary_all