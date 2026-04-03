#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run the full evaluation pipeline for NVV.

This includes:
- 0) Building the GT dict from cleaned excels
- 1) Evaluating each workspace against GT (detailed + summary)
- 2) Global combo-ranking of nvv annotation configuration sets (tracks) (per dataset)
- 3) Selection of best k (full-GT) or selected set (part-GT)
- 4) Plotting greedy forward selection curves (full-GT)
- 5) Union evaluation for selected set
- 6) Exporting clips for selected set
- 7) Final pipeline quality evaluation for selected set

CLI usage:
    python run_evaluation.py --config ./config/config.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from config.load_config import load_config
from config.path_factory import (
    get_workspace_paths,
    get_eval_mode_dir,
    get_detailed_all_csv_path,
    get_summary_all_csv_path,
    get_global_evaluation_xlsx_path,
    get_selected_set_clips_sub_dir,
)

from preprocessing.merge_gt_excels import merge_gt_excels
from utils.parsing import collect_evaluable_audio_ids
from utils.io import print_header

from evaluation.eval_adapter_gt import build_gt_dict
from evaluation.eval_io import write_csv_atomic, write_xlsx_atomic
from evaluation.evaluation_workspace import evaluate_workspace
from export.export_clips import export_clips

from evaluation.plot_greedy_forward_selection_curves import plot_curves_for_dataset

from evaluation.rq1_pipeline_capability import run_pipeline_capability_evaluation
from evaluation.rq2_configuration_ranking_single import run_single_configuration_ranking
from evaluation.rq2_configuration_best_k_selection import (
    load_global_best_k_set,
    run_best_k_selection_for_dataset,
)
from evaluation.rq2_configuration_combination_selected_set import run_best_k_union_evaluation
from evaluation.rq2_audio_derivatives import rank_audio_derivatives
from evaluation.rq3_nvv_coverage import compute_nvv_coverage

from evaluation.rq_results_workspace import collect_rq_results_from_artifacts, write_rq_results

from config.params import (
    EVAL_T_COLLAR,
    EVAL_PERCENTAGE_OF_LENGTH,
    EVAL_K_MAX,
    EVAL_DELTA_F1_STOP,
    EVAL_TOP_N,
    EVAL_STOP_ON_NON_IMPROVEMENT,
    EVAL_DEDUP_EPS_S,
    EVAL_PRELOAD,
    EVAL_VERBOSE_MISSING,
    EVAL_EVALUATE_ONSET,
    EVAL_EVALUATE_OFFSET,
    EVAL_MATCH_LABELS,
    EVAL_CLIP_MODES,
    EVAL_FORCE,
    EVAL_K_OVERRIDE,
)


# --- Steps ---

def step_0_build_gt_dict(*, cleaned_excel_paths: List[Path]) -> Dict[str, List[dict]]:
    """
    Step 0: Merge cleaned GT excels and build GT dict.
    """
    df_gt, _ = merge_gt_excels(
        excel_paths=cleaned_excel_paths,
        out_path=None,
        validate_columns=True,
    )
    print("\nMerged GT rows:", len(df_gt))
    print("GT columns:", list(df_gt.columns))

    gt_dict = build_gt_dict(
        df_gt,
        id_column="video_id",
        ann_id_column="ann_id",
        start_column="start_s",
        end_column="end_s",
        label_column="vocalization_type",
    )

    print("\nGT audio_ids:", len(gt_dict))
    if gt_dict:
        some_id = next(iter(gt_dict.keys()))
        print("Example audio_id:", some_id, "n_events:", len(gt_dict[some_id]))
        print("Example event:", gt_dict[some_id][0])

    return gt_dict


def step_1_workspace_evaluation(
    *,
    ds,
    ws,
    gt_mode: str,
    gt_dict: Dict[str, List[dict]],
    evaluable_ids: List[str],
) -> None:
    """
    Step 1: Workspace evaluation (detailed + summary + xlsx).
    """
    mode_dir = get_eval_mode_dir(Path(ws.evaluation), gt_mode)
    mode_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "-" * 50)
    print(f"Dataset: {ds.name}")
    print("Workspace:", ws.workspace)
    print("Evaluation:", ws.evaluation)
    print("Mode dir:", mode_dir)
    print("-" * 50)

    print(f"Evaluable IDs: {len(evaluable_ids)}")

    detailed_all, summary_all = evaluate_workspace(
        workspace_root=ws.workspace,
        evaluable_ids=evaluable_ids,
        gt_dict=gt_dict,
        mode=gt_mode,
        t_collar=EVAL_T_COLLAR,
        percentage_of_length=EVAL_PERCENTAGE_OF_LENGTH,
        evaluate_onset=True,
        evaluate_offset=True,
        match_labels=False,
        write_per_audio=True,
    )

    print("Finished dataset:", ds.name)
    print("Detailed rows:", detailed_all.shape)
    print("Summary rows:", summary_all.shape)

    write_csv_atomic(detailed_all, get_detailed_all_csv_path(ws.evaluation, gt_mode))
    write_csv_atomic(summary_all, get_summary_all_csv_path(ws.evaluation, gt_mode))

    write_xlsx_atomic(
        detailed_df=detailed_all,
        summary_df=summary_all,
        out_path=get_global_evaluation_xlsx_path(ws.evaluation, gt_mode),
        detailed_sheet="Detailed",
        summary_sheet="Summary",
    )


def step_2_global_combo_ranking(*, ws, gt_mode: str) -> pd.DataFrame:
    """
    Step 2: Global combo ranking per dataset.
    """
    ranking_df = run_single_configuration_ranking(
        evaluation_dir=ws.evaluation,
        mode=gt_mode,
        top_n=EVAL_TOP_N,
        write_xlsx_sheet=True,
    )

    print("\nRanking (head=3):")
    print(ranking_df.head(3).to_string(index=False))
    return ranking_df


def step_2b_rq2b_audio_derivatives(*, ws, gt_mode: str) -> pd.DataFrame:
    """
    Step 2b: RQ2b audio derivative aggregation from global combo ranking.

    Arguments:
        ws: Workspace path bundle.
        gt_mode: "full_gt" or "part_gt".

    Returns:
        RQ2b aggregated dataframe.
    """
    df = rank_audio_derivatives(
        evaluation_dir=ws.evaluation,
        mode=gt_mode,
        write_file=True,
    )

    print("\nRQ2b audio derivatives:")
    print(df.to_string(index=False))
    return df


def step_3_best_k_selection(
    *,
    ds,
    ws_by_dataset,
    evaluable_by_dataset: Dict[str, List[str]],
    gt_dict: Dict[str, List[dict]],
    gt_mode: str,
    part_gt_additional_selected_set: List[str],
) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], pd.DataFrame]:
    """
    Step 3: Best-k / selected-set construction per dataset.
    """
    trace_df, f1_vs_k_df, best_set_df = run_best_k_selection_for_dataset(
        dataset=ds,
        ws_by_dataset=ws_by_dataset,
        evaluable_by_dataset=evaluable_by_dataset,
        gt_dict=gt_dict,
        mode=gt_mode,
        top_n=EVAL_TOP_N,
        k_max=EVAL_K_MAX,
        delta_f1_stop=EVAL_DELTA_F1_STOP,
        stop_on_non_improvement=EVAL_STOP_ON_NON_IMPROVEMENT,
        dedup_eps_s=EVAL_DEDUP_EPS_S,
        preload=EVAL_PRELOAD,
        verbose_missing=EVAL_VERBOSE_MISSING,
        t_collar=EVAL_T_COLLAR,
        percentage_of_length=EVAL_PERCENTAGE_OF_LENGTH,
        evaluate_onset=EVAL_EVALUATE_ONSET,
        evaluate_offset=EVAL_EVALUATE_OFFSET,
        match_labels=EVAL_MATCH_LABELS,
        part_gt_additional_selected_set=part_gt_additional_selected_set,
    )
    return trace_df, f1_vs_k_df, best_set_df


def step_4_plot_curves(*, ds, gt_mode: str, ws_by_dataset) -> None:
    """
    Step 4: Plot greedy forward selection curves (may skip internally for part_gt).
    """
    plot_curves_for_dataset(ds.name, gt_mode, ws_by_dataset)


def step_5_union_evaluation(
    *,
    ws,
    gt_dict: Dict[str, List[dict]],
    gt_mode: str,
    evaluable_ids: List[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Step 5: Union evaluation for selected set.
    """
    per_audio_df, summary_df, set_df = run_best_k_union_evaluation(
        workspace_root=ws.workspace,
        evaluation_dir=ws.evaluation,
        gt_dict=gt_dict,
        mode=gt_mode,
        evaluable_ids=evaluable_ids,
        k_override=EVAL_K_OVERRIDE,
        dedup_eps_s=EVAL_DEDUP_EPS_S,
        preload=EVAL_PRELOAD,
        verbose_missing=EVAL_VERBOSE_MISSING,
        t_collar=EVAL_T_COLLAR,
        percentage_of_length=EVAL_PERCENTAGE_OF_LENGTH,
        evaluate_onset=EVAL_EVALUATE_ONSET,
        evaluate_offset=EVAL_EVALUATE_OFFSET,
        match_labels=EVAL_MATCH_LABELS,
        write_xlsx_sheet=True,
    )

    print("\nUnion summary:")
    print(summary_df.to_string(index=False))

    return per_audio_df, summary_df, set_df


def step_6a_load_selected_set(*, ws, gt_mode: str) -> pd.DataFrame:
    """
    Step 6a: Load selected set for dataset.
    """
    best_k_df = load_global_best_k_set(ws.evaluation, mode=gt_mode)
    print(best_k_df.to_string(index=False))
    print("Tracks:", best_k_df["combo_key"].tolist())
    return best_k_df


def step_6b_export_clips(*, ds, ws, gt_mode: str, best_k_df: pd.DataFrame) -> None:
    """
    Step 6b: Export clips for selected set (dataset).

    Note: (toDo)
        Current export uses only the first row metadata as selector input.
        This affects clip export behavior only, not evaluation metrics.
    """
    if best_k_df is None or best_k_df.empty:
        print("No selected set found. Skipping clip export.")
        return

    required_cols = {"combo_key", "vad_mask", "asr_audio_in"}
    missing = sorted(required_cols - set(best_k_df.columns))
    if missing:
        raise KeyError(
            f"[{ds.name}] global_best_k_set_{gt_mode}.csv missing columns {missing}. "
            f"Found: {list(best_k_df.columns)}"
        )

    top = best_k_df.iloc[0]
    best_vad_mask = top["vad_mask"]
    best_asr_audio_in = top["asr_audio_in"]

    for clip_mode in EVAL_CLIP_MODES:
        export_clips(
            workspace=ws.workspace,
            mode=clip_mode,
            vad_masks=[best_vad_mask] if best_vad_mask is not None else None,
            asr_audio_ins=[best_asr_audio_in] if best_asr_audio_in is not None else None,
            sub_dir=get_selected_set_clips_sub_dir(gt_mode),
            force=EVAL_FORCE,
        )


def step_7_pipeline_capability(
    *,
    ws,
    gt_dict: Dict[str, List[dict]],
    gt_mode: str,
    evaluable_ids: List[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Step 7: Final pipeline capability evaluation.
    """
    summary_df, per_audio_df = run_pipeline_capability_evaluation(
        workspace_root=ws.workspace,
        evaluation_dir=ws.evaluation,
        gt_dict=gt_dict,
        mode=gt_mode,
        evaluable_ids=evaluable_ids,
        dedup_eps_s=EVAL_DEDUP_EPS_S,
        preload=EVAL_PRELOAD,
        verbose_missing=EVAL_VERBOSE_MISSING,
        t_collar=EVAL_T_COLLAR,
        percentage_of_length=EVAL_PERCENTAGE_OF_LENGTH,
        evaluate_onset=EVAL_EVALUATE_ONSET,
        evaluate_offset=EVAL_EVALUATE_OFFSET,
        match_labels=EVAL_MATCH_LABELS,
        write_files=True,
    )

    print("\nSummary (head=10):")
    print(summary_df.head(10).to_string(index=False))

    print("\nPer-audio (head=10):")
    print(per_audio_df.head(10).to_string(index=False))

    return summary_df, per_audio_df


def step_7b_rq3_nvv_coverage(*, ws, gt_mode: str) -> pd.DataFrame:
    """
    Step 7b: RQ3 NVV coverage artifact from detailed evaluation output.

    Arguments:
        ws: Workspace path bundle.
        gt_mode: "full_gt" or "part_gt".

    Returns:
        RQ3 dataframe.
    """
    df = compute_nvv_coverage(
        evaluation_dir=ws.evaluation,
        mode=gt_mode,
        write_file=True,
    )

    print("\nRQ3 NVV coverage:")
    print(df.head(20).to_string(index=False))
    return df


def step_8_write_rq_results(
    *,
    ws,
    gt_mode: str,
) -> dict[str, Path]:
    """
    Step 8: Build and write research question result tables.
    """
    print("\n" + "-" * 50)
    print("Write RQ result tables")
    print("Evaluation dir:", ws.evaluation)
    print("Mode:", gt_mode)
    print("-" * 50)

    results = collect_rq_results_from_artifacts(
        evaluation_dir=ws.evaluation,
        mode=gt_mode,
    )

    written = write_rq_results(
        evaluation_dir=ws.evaluation,
        mode=gt_mode,
        results=results,
    )

    print("Written RQ artifacts:")
    for key, path in written.items():
        print(f"  {key}: {path}")

    return written


# --- Evaluation ---

def run_evaluation_for_dataset(
    *,
    config,
    ds,
    ws,
    gt_mode: str,
    cleaned_excel_paths: List[Path],
    id_column: str,
    gt_dict: Dict[str, List[dict]],
    ws_by_dataset,
) -> None:
    """
    Run evaluation Steps 1–8 for a single dataset.
    """
    print_header(
        title=f"▶ Collect evaluable IDs — dataset={ds.name}",
        subtitle=f"Workspace: {ws.workspace}",
    )

    evaluable_ids = collect_evaluable_audio_ids(
        workspace_dir=ws.workspace,
        cleaned_excel_paths=cleaned_excel_paths,
        id_column=id_column,
        verbose=True,
    )

    print(f"✅ {ds.name}: {len(evaluable_ids)} evaluable IDs")

    evaluable_by_dataset = {ds.name: evaluable_ids}

    print_header(
        title=f"▶ Workspace Evaluation (detailed + summary) — dataset={ds.name}",
        subtitle=f"GT_MODE={gt_mode}",
    )
    step_1_workspace_evaluation(
        ds=ds,
        ws=ws,
        gt_mode=gt_mode,
        gt_dict=gt_dict,
        evaluable_ids=evaluable_ids,
    )

    print_header(
        title=f"▶ Global Combo Ranking - dataset={ds.name}",
        subtitle=f"GT_MODE={gt_mode}, Evaluation dir: {ws.evaluation}",
    )
    step_2_global_combo_ranking(
        ws=ws,
        gt_mode=gt_mode,
    )

    print_header(
        title=f"▶ RQ2b Audio Derivative Aggregation - dataset={ds.name}",
        subtitle=f"GT_MODE={gt_mode}, Evaluation dir: {ws.evaluation}",
    )
    step_2b_rq2b_audio_derivatives(
        ws=ws,
        gt_mode=gt_mode,
    )

    print_header(
        title=f"▶ Best-k / selected-set construction - dataset={ds.name}",
        subtitle=f"GT_MODE={gt_mode}, Evaluation dir: {ws.evaluation}",
    )
    step_3_best_k_selection(
        ds=ds,
        ws_by_dataset=ws_by_dataset,
        evaluable_by_dataset=evaluable_by_dataset,
        gt_dict=gt_dict,
        gt_mode=gt_mode,
        part_gt_additional_selected_set=config.evaluation.part_gt_additional_selected_set,
    )

    print_header(
        title="▶ Plot Greedy Forward Selection Curves (full-GT only)",
        subtitle=f"dataset={ds.name}",
    )
    step_4_plot_curves(
        ds=ds,
        gt_mode=gt_mode,
        ws_by_dataset=ws_by_dataset,
    )

    print_header(
        title=f"▶ UNION Evaluation — dataset={ds.name}",
        subtitle=f"Mode: {gt_mode}",
    )
    step_5_union_evaluation(
        ws=ws,
        gt_dict=gt_dict,
        gt_mode=gt_mode,
        evaluable_ids=evaluable_ids,
    )

    print_header(
        title=f"▶ Load and check best k / selected set — dataset={ds.name}",
        subtitle=f"Mode: {gt_mode}",
    )
    best_k_df = step_6a_load_selected_set(
        ws=ws,
        gt_mode=gt_mode,
    )

    print_header(
        title=f"▶ Export clips for best k / selected set — dataset={ds.name}",
        subtitle=f"Mode: {gt_mode}",
    )
    step_6b_export_clips(
        ds=ds,
        ws=ws,
        gt_mode=gt_mode,
        best_k_df=best_k_df,
    )

    print_header(
        title=f"▶ Final Pipeline Capability — dataset={ds.name}",
        subtitle=f"Mode: {gt_mode}",
    )
    step_7_pipeline_capability(
        ws=ws,
        gt_dict=gt_dict,
        gt_mode=gt_mode,
        evaluable_ids=evaluable_ids,
    )

    print_header(
        title=f"▶ RQ3 NVV Coverage — dataset={ds.name}",
        subtitle=f"Mode: {gt_mode}",
    )
    step_7b_rq3_nvv_coverage(
        ws=ws,
        gt_mode=gt_mode,
    )

    print_header(
        title=f"▶ Write RQ result tables — dataset={ds.name}",
        subtitle=f"Mode: {gt_mode}",
    )
    step_8_write_rq_results(
        ws=ws,
        gt_mode=gt_mode,
    )


def run_evaluation_from_config(config) -> None:
    """
    Run the evaluation pipeline using only the loaded unified config object.
    """
    print_header("▶ Setup Evaluation Configuration")

    eval_cfg = config.evaluation
    gt_mode: str = eval_cfg.gt_mode
    gt_units = eval_cfg.gt_units
    cleaned_excel_paths: List[Path] = list(eval_cfg.gt_truth_paths)

    if gt_mode not in {"full_gt", "part_gt"}:
        raise ValueError(f"Invalid evaluation.gt_mode: {gt_mode}")

    id_column = gt_units[0].id_column if gt_units else "video_id"

    eval_datasets = list(config.datasets)

    ws_by_dataset = {ds.name: get_workspace_paths(ds.workspace) for ds in eval_datasets}

    print("GT_MODE:", gt_mode)
    print("Ground Truth cleaned excels:")
    for p in cleaned_excel_paths:
        print("  -", p)

    print_header("▶ Build GT-Dictionary")
    gt_dict = step_0_build_gt_dict(cleaned_excel_paths=cleaned_excel_paths)

    for ds in eval_datasets:
        ws = ws_by_dataset[ds.name]
        run_evaluation_for_dataset(
            config=config,
            ds=ds,
            ws=ws,
            gt_mode=gt_mode,
            cleaned_excel_paths=cleaned_excel_paths,
            id_column=id_column,
            gt_dict=gt_dict,
            ws_by_dataset=ws_by_dataset,
        )

    print("\n✅ All datasets processed.")


def main() -> None:
    """
    CLI entry point: load config and run evaluation.
    """
    p = argparse.ArgumentParser(description="Run NVV evaluation (full_gt or part_gt)")
    p.add_argument(
        "--config",
        required=True,
        type=str,
        help="Path to config.yaml (e.g., ./config/config.yaml)",
    )
    args = p.parse_args()

    config_path = Path(args.config).resolve()

    config = load_config(config_path)
    run_evaluation_from_config(config)


if __name__ == "__main__":
    main()


# --- EXAMPLE USAGE ---
# (nvv_isolation_pipeline) cd <project_root>
# python run_evaluation.py --config ./config/config.yaml