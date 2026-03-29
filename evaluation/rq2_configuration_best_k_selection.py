from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json

import pandas as pd

from evaluation.eval_union import evaluate_union_for_audio_id, list_audio_ids_from_workspace
from evaluation.eval_adapter_candidates import load_candidate_events_from_nvv_json
from evaluation.eval_io import write_csv_atomic
from config.params import (
    EVAL_T_COLLAR,
    EVAL_PERCENTAGE_OF_LENGTH,
    EVAL_K_MAX,
    EVAL_DELTA_F1_STOP,
)
from utils.io import print_header
from config.path_factory import (
    get_eval_mode_dir,
    get_global_combo_ranking_csv_path,
    get_global_best_k_set_csv_path,
    get_global_best_k_trace_csv_path,
    get_global_f1_vs_k_csv_path,
)


RANKING_REQUIRED_COLUMNS = {
    "mode",
    "combo_key",
    "vad_mask",
    "asr_audio_in",
}


def _read_combo_ranking_csv(evaluation_dir: Path, mode: str) -> pd.DataFrame:
    """
    Read global_combo_ranking_<mode>.csv with basic validation.

    Arguments:
        evaluation_dir: workspace/global/evaluation
        mode: "full_gt" or "part_gt"
    """
    path = get_global_combo_ranking_csv_path(Path(evaluation_dir), mode)

    if not path.exists():
        raise FileNotFoundError(f"Missing required ranking file: {path}")

    df = pd.read_csv(path)
    if df.shape[0] == 0:
        raise RuntimeError(f"global_combo_ranking_{mode}.csv contains 0 rows: {path}")

    missing = sorted(RANKING_REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise KeyError(f"global_combo_ranking_{mode}.csv missing columns: {missing}")

    return df


def load_global_best_k_set(evaluation_dir: str | Path, mode: str) -> pd.DataFrame:
    """
    Load global_best_k_set_<mode>.csv and return it as a DataFrame.

    Naming is internal:
        <evaluation_dir>/<mode>/global_best_k_set_<mode>.csv
    """
    if mode not in ("full_gt", "part_gt"):
        raise ValueError(f"mode must be 'full_gt' or 'part_gt', got: {mode}")

    path = get_global_best_k_set_csv_path(Path(evaluation_dir), mode)

    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise RuntimeError(f"{path.name} contains 0 rows: {path}")

    # Keep intended order if present
    if "rank_in_set" in df.columns:
        df = df.sort_values("rank_in_set")
    elif "rank" in df.columns:
        df = df.sort_values("rank")

    return df.reset_index(drop=True)


def select_top_n_part_gt_set(*, evaluation_dir: str | Path, top_n: int = 1) -> pd.DataFrame:
    """
    Build a 'selected set' for part_gt from global_combo_ranking_part_gt.
    Writes: <evaluation_dir>/part_gt/global_best_k_set_part_gt.csv

    Args:
        evaluation_dir: Workspace evaluation directory.
        top_n: Number of top-ranked combos to include.

    Returns:
        set_df: Selected set as DataFrame.
    """
    mode = "part_gt"
    evaluation_dir = Path(evaluation_dir)

    ranking_path = get_global_combo_ranking_csv_path(evaluation_dir, mode)
    if not ranking_path.exists():
        raise FileNotFoundError(f"Missing ranking: {ranking_path}")

    ranking = pd.read_csv(ranking_path)
    if ranking.empty:
        raise RuntimeError(f"Empty ranking: {ranking_path}")

    if top_n <= 0:
        raise ValueError(f"top_n must be > 0, got: {top_n}")

    top = ranking.head(int(top_n)).copy()

    required = {"combo_key", "vad_mask", "asr_audio_in"}
    missing = sorted(required - set(top.columns))
    if missing:
        raise KeyError(f"{ranking_path.name} missing columns: {missing}")

    combo_keys = top["combo_key"].astype(str).tolist()

    set_df = top.loc[:, ["combo_key", "vad_mask", "asr_audio_in"]].copy()
    set_df.insert(0, "mode", mode)
    set_df.insert(1, "best_k", int(len(combo_keys)))
    set_df.insert(2, "rank_in_set", list(range(1, len(combo_keys) + 1)))
    set_df.insert(3, "selected_set_json", json.dumps(combo_keys))

    out_path = get_global_best_k_set_csv_path(evaluation_dir, mode)
    write_csv_atomic(set_df, out_path)
    return set_df


def run_best_k_selection_for_dataset(
    *,
    dataset: object,
    ws_by_dataset: Dict[str, object],
    evaluable_by_dataset: Dict[str, List[str]],
    gt_dict: Dict[str, List[dict]],
    mode: str,
    top_n: int,
    k_max: int,
    delta_f1_stop: float,
    stop_on_non_improvement: bool,
    dedup_eps_s: float,
    preload: bool,
    verbose_missing: bool,
    t_collar: float,
    percentage_of_length: float,
    evaluate_onset: bool,
    evaluate_offset: bool,
    match_labels: bool,
) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], pd.DataFrame]:
    """
    Create the "selected set" per dataset.

    - full_gt: greedy forward best-k selection
    - part_gt: deterministic top-n selection from ranking

    Args:
        datasets: Dataset config objects.
        ws_by_dataset: Mapping dataset_name -> workspace_paths.
        evaluable_by_dataset: Mapping dataset_name -> list of evaluable ids.
        gt_dict: Ground truth dict.
        mode: "full_gt" or "part_gt".
        top_n: Top-N combos considered in greedy stage / ranking stage.
        k_max: Max k for greedy selection.
        delta_f1_stop: Early stopping threshold for greedy selection.
        stop_on_non_improvement: Stop if no improvement.
        dedup_eps_s: Dedup tolerance in seconds.
        preload: Preload artifacts for speed.
        verbose_missing: Print missing file details.
        t_collar: Event collar (seconds).
        percentage_of_length: Percentage collar factor.
        evaluate_onset: Evaluate onset matching.
        evaluate_offset: Evaluate offset matching.
        match_labels: Match labels/types or ignore labels.
    """
    if mode == "full_gt":
        dataset_name = dataset.name
        ws = ws_by_dataset[dataset_name]
        evaluable_ids = evaluable_by_dataset[dataset_name]

        print("\n" + "=" * 70)
        print(f"▶ Best-k Selection — dataset={dataset_name}")
        print(f"Mode: {mode}")
        print("=" * 70)

        trace_df, f1_vs_k_df, best_set_df = greedy_forward_best_k_selection_full_gt(
            workspace_root=ws.workspace,
            evaluation_dir=ws.evaluation,
            gt_dict=gt_dict,
            evaluable_ids=evaluable_ids,
            top_n=top_n,
            k_max=k_max,
            delta_f1_stop=delta_f1_stop,
            stop_on_non_improvement=stop_on_non_improvement,
            dedup_eps_s=dedup_eps_s,
            preload=preload,
            verbose_missing=verbose_missing,
            t_collar=t_collar,
            percentage_of_length=percentage_of_length,
            evaluate_onset=evaluate_onset,
            evaluate_offset=evaluate_offset,
            match_labels=match_labels,
        )

        print("\nbest_set_df:")
        print(best_set_df.to_string(index=False))

        return trace_df, f1_vs_k_df, best_set_df

    elif mode == "part_gt":
        dataset_name = dataset.name
        ws = ws_by_dataset[dataset_name]

        print_header(
            title=f"▶ Part-GT Top-N selection — dataset={dataset_name}", 
            subtitle=f"Mode: {mode}"
        )

        best_set_df = select_top_n_part_gt_set(evaluation_dir=ws.evaluation, top_n=1)

        print("\nselected set:")
        print(best_set_df.to_string(index=False))
        return None, None, best_set_df

    else:
        raise ValueError(f"Unknown mode: {mode}")


def greedy_forward_best_k_selection_full_gt(
    *,
    workspace_root: Path,
    evaluation_dir: Path,
    gt_dict: Dict[str, List[dict]],
    evaluable_ids: Optional[List[str]] = None,
    top_n: Optional[int] = None,
    k_max: int = EVAL_K_MAX,
    delta_f1_stop: float = EVAL_DELTA_F1_STOP,
    stop_on_non_improvement: bool = True,
    dedup_eps_s: Optional[float] = None,
    preload: bool = True,
    verbose_missing: bool = True,
    t_collar: float = EVAL_T_COLLAR,
    percentage_of_length: float = EVAL_PERCENTAGE_OF_LENGTH,
    evaluate_onset: bool = True,
    evaluate_offset: bool = True,
    match_labels: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Greedy forward best-k selection optimizing macro mean F1 over audio_ids (full_gt only).

    Inputs:
        - <evaluation_dir>/full_gt/global_combo_ranking_full_gt.csv
        - per_audio/<audio_id>/annotations/nvv/<audio_id>_<combo_key>.json

    Outputs (written internally):
        - <evaluation_dir>/full_gt/global_best_k_trace_full_gt.csv
        - <evaluation_dir>/full_gt/global_f1_vs_k_full_gt.csv
        - <evaluation_dir>/full_gt/global_best_k_set_full_gt.csv

    STRICT:
        - Missing candidate files raise FileNotFoundError (enforced by eval_union.evaluate_union_for_audio_id).

    Arguments:
        stop_on_non_improvement: If True, stop when best delta_f1 <= delta_f1_stop.
            If False, continue until k_max (or pool exhausted) and select best_k by argmax F1 on the path.
        verbose_missing: retained for interface compatibility; missing files raise (strict).
    """
    mode = "full_gt"

    workspace_root = Path(workspace_root)
    evaluation_dir = Path(evaluation_dir)
    mode_dir = get_eval_mode_dir(evaluation_dir, mode)
    mode_dir.mkdir(parents=True, exist_ok=True)

    # kept for interface compatibility; strict behavior now raises on missing files.
    _ = bool(verbose_missing)

    if evaluable_ids is None:
        evaluable_ids = list_audio_ids_from_workspace(workspace_root)
    evaluable_ids = list(evaluable_ids)

    if k_max <= 0:
        raise ValueError(f"k_max must be > 0, got: {k_max}")

    ranking = _read_combo_ranking_csv(evaluation_dir, mode=mode)

    # Safety: keep only requested mode (should already be true if files are mode-scoped)
    ranking = ranking[ranking["mode"].astype(str) == mode].copy()
    if ranking.shape[0] == 0:
        raise RuntimeError(f"global_combo_ranking_{mode}.csv contains no rows for mode={mode}")

    if top_n is not None:
        if int(top_n) <= 0:
            raise ValueError(f"top_n must be > 0, got: {top_n}")
        ranking = ranking.head(int(top_n)).copy()

    pool_combo_keys = ranking["combo_key"].astype(str).tolist()
    pool_vad_mask = ranking["vad_mask"].tolist()
    pool_asr_audio_in = ranking["asr_audio_in"].tolist()

    match_params = {
        "match_labels": bool(match_labels),
        "evaluate_onset": bool(evaluate_onset),
        "evaluate_offset": bool(evaluate_offset),
        "t_collar": float(t_collar),
        "percentage_of_length": float(percentage_of_length),
    }

    cache: Optional[Dict[Tuple[str, str], List[dict]]] = {} if preload else None

    selected: List[str] = []
    selected_meta: List[Dict[str, Any]] = []
    trace_rows: List[Dict[str, Any]] = []
    curve_rows: List[Dict[str, Any]] = []

    def macro_over_audio_ids(combo_keys_in_order: List[str]) -> Dict[str, float]:
        vals = []
        for aid in evaluable_ids:
            vals.append(
                evaluate_union_for_audio_id(
                    audio_id=aid,
                    workspace_root=workspace_root,
                    gt_dict=gt_dict,
                    combo_keys_in_order=combo_keys_in_order,
                    load_candidate_events_fn=load_candidate_events_from_nvv_json,
                    cache=cache,
                    dedup_eps_s=dedup_eps_s,
                    match_params=match_params,
                )
            )
        df = pd.DataFrame(vals)
        return {k: float(df[k].mean()) for k in df.columns}

    def _meta_for_ck(ck: str) -> Dict[str, Any]:
        idx = pool_combo_keys.index(ck)
        return {
            "combo_key": ck,
            "vad_mask": pool_vad_mask[idx],
            "asr_audio_in": pool_asr_audio_in[idx],
        }

    # Step 1: best single track (evaluate all singles; correct by definition)
    best_ck: Optional[str] = None
    best_single_metrics: Optional[Dict[str, float]] = None
    best_single_f1 = float("-inf")

    for ck in pool_combo_keys:
        metrics = macro_over_audio_ids([ck])
        f1 = float(metrics["f1"])
        if f1 > best_single_f1:
            best_single_f1 = f1
            best_ck = ck
            best_single_metrics = metrics

    if best_ck is None or best_single_metrics is None:
        raise RuntimeError("Could not select best single track from ranking pool.")

    selected.append(best_ck)
    selected_meta.append(_meta_for_ck(best_ck))

    selected_set_json = json.dumps(selected)

    trace_rows.append(
        {
            "mode": mode,
            "k": 1,
            **_meta_for_ck(best_ck),
            "macro_mean_f1": best_single_metrics["f1"],
            "macro_mean_recall": best_single_metrics["recall"],
            "macro_mean_precision": best_single_metrics["precision"],
            "macro_mean_mean_dice_eos_tp": best_single_metrics["mean_dice_eos_tp"],
            "macro_mean_dice_eos_recall": best_single_metrics["dice_eos_recall"],
            "macro_mean_fp": best_single_metrics["fp"],
            "delta_f1": best_single_metrics["f1"],
            "selected_set_json": selected_set_json,
        }
    )

    curve_rows.append(
        {
            "mode": mode,
            "k": 1,
            "macro_mean_f1": best_single_metrics["f1"],
            "macro_mean_recall": best_single_metrics["recall"],
            "macro_mean_precision": best_single_metrics["precision"],
            "macro_mean_mean_dice_eos_tp": best_single_metrics["mean_dice_eos_tp"],
            "macro_mean_dice_eos_recall": best_single_metrics["dice_eos_recall"],
            "macro_mean_fp": best_single_metrics["fp"],
            "selected_set_json": selected_set_json,
        }
    )

    current_f1 = float(best_single_metrics["f1"])
    remaining = [ck for ck in pool_combo_keys if ck != best_ck]

    for k in range(2, k_max + 1):
        best_delta = float("-inf")
        best_candidate_ck: Optional[str] = None
        best_metrics_for_candidate: Optional[Dict[str, float]] = None

        for ck in remaining:
            metrics = macro_over_audio_ids(selected + [ck])
            delta = float(metrics["f1"]) - float(current_f1)
            if delta > best_delta:
                best_delta = delta
                best_candidate_ck = ck
                best_metrics_for_candidate = metrics

        if best_candidate_ck is None or best_metrics_for_candidate is None:
            break

        if stop_on_non_improvement and best_delta <= float(delta_f1_stop):
            break

        selected.append(best_candidate_ck)
        selected_meta.append(_meta_for_ck(best_candidate_ck))
        remaining = [ck for ck in remaining if ck != best_candidate_ck]

        current_f1 = float(best_metrics_for_candidate["f1"])
        selected_set_json = json.dumps(selected)

        trace_rows.append(
            {
                "mode": mode,
                "k": k,
                **_meta_for_ck(best_candidate_ck),
                "macro_mean_f1": best_metrics_for_candidate["f1"],
                "macro_mean_recall": best_metrics_for_candidate["recall"],
                "macro_mean_precision": best_metrics_for_candidate["precision"],
                "macro_mean_mean_dice_eos_tp": best_metrics_for_candidate["mean_dice_eos_tp"],
                "macro_mean_dice_eos_recall": best_metrics_for_candidate["dice_eos_recall"],
                "macro_mean_fp": best_metrics_for_candidate["fp"],
                "delta_f1": best_delta,
                "selected_set_json": selected_set_json,
            }
        )

        curve_rows.append(
            {
                "mode": mode,
                "k": k,
                "macro_mean_f1": best_metrics_for_candidate["f1"],
                "macro_mean_recall": best_metrics_for_candidate["recall"],
                "macro_mean_precision": best_metrics_for_candidate["precision"],
                "macro_mean_mean_dice_eos_tp": best_metrics_for_candidate["mean_dice_eos_tp"],
                "macro_mean_dice_eos_recall": best_metrics_for_candidate["dice_eos_recall"],
                "macro_mean_fp": best_metrics_for_candidate["fp"],
                "selected_set_json": selected_set_json,
            }
        )

        if not remaining:
            break

    trace_df = pd.DataFrame(trace_rows)
    curve_df = pd.DataFrame(curve_rows)

    if curve_df.shape[0] == 0:
        raise RuntimeError("Best-k selection produced no curve rows.")

    curve_f1 = curve_df["macro_mean_f1"].astype(float)
    best_k_idx = int(curve_f1.idxmax())
    best_k = int(curve_df.loc[best_k_idx, "k"])

    best_selected = selected[:best_k]
    best_selected_meta = selected_meta[:best_k]

    best_set_df = pd.DataFrame(best_selected_meta)
    best_set_df.insert(0, "mode", mode)
    best_set_df.insert(1, "best_k", best_k)
    best_set_df.insert(2, "rank_in_set", list(range(1, len(best_selected_meta) + 1)))
    best_set_df.insert(3, "selected_set_json", json.dumps(best_selected))

    curve_df = curve_df.copy()
    curve_df["best_k"] = best_k

    write_csv_atomic(trace_df, get_global_best_k_trace_csv_path(evaluation_dir, mode))
    write_csv_atomic(curve_df, get_global_f1_vs_k_csv_path(evaluation_dir, mode))
    write_csv_atomic(best_set_df, get_global_best_k_set_csv_path(evaluation_dir, mode))

    return trace_df, curve_df, best_set_df