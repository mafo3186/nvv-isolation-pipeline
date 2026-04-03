from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json

import pandas as pd

from config.constants import KEY_PER_AUDIO
from utils.io import ensure_dir
from evaluation.eval_event_matching import match_events_optimal
from evaluation.eval_metrics import (
    overlap_seconds,
    dice_event_overlap_score,
    full_gt_metrics,
    partial_gt_metrics,
)
from evaluation.eval_adapter_candidates import load_candidate_events_from_nvv_json
from evaluation.eval_io import write_csv_atomic, write_xlsx_atomic
from config.params import EVAL_T_COLLAR, EVAL_PERCENTAGE_OF_LENGTH
from config.path_factory import (
    get_global_best_k_set_csv_path,
    get_eval_mode_dir,
    get_pipeline_capability_summary_csv_path,
    get_pipeline_capability_per_audio_csv_path,
    get_pipeline_capability_xlsx_path,
    get_pipeline_capability_nvv_events_csv_path,
    get_nvv_json_path_from_combo_key,
)


# --- IO helpers ---

def _list_audio_ids_from_workspace(workspace_root: Path) -> List[str]:
    per_audio = Path(workspace_root) / KEY_PER_AUDIO
    if not per_audio.exists():
        return []
    return sorted([p.name for p in per_audio.iterdir() if p.is_dir()])


def _read_best_k_set_csv(evaluation_dir: Path, mode: str) -> Tuple[int, List[str], str]:
    """
    Read global_best_k_set_<mode>.csv produced by best-k selection and return:
      best_k, combo_keys_in_order, selected_set_json (string)

    Naming (internal):
      <evaluation_dir>/<mode>/global_best_k_set_<mode>.csv
    """
    if mode not in ("full_gt", "part_gt"):
        raise ValueError(f"mode must be 'full_gt' or 'part_gt', got: {mode}")

    path = get_global_best_k_set_csv_path(Path(evaluation_dir), mode)
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")

    df = pd.read_csv(path)
    if df.shape[0] == 0:
        raise RuntimeError(f"{path.name} contains 0 rows.")

    # Safety: keep only requested mode if present
    if "mode" in df.columns:
        df = df[df["mode"].astype(str) == str(mode)].copy()
        if df.shape[0] == 0:
            raise RuntimeError(f"{path.name} contains no rows for mode={mode!r}")

    # selected_set_json: take first non-null
    if "selected_set_json" not in df.columns:
        raise KeyError(f"{path.name} missing required column: selected_set_json")

    sjson = None
    for v in df["selected_set_json"].tolist():
        if isinstance(v, str) and v.strip():
            sjson = v
            break
    if sjson is None:
        raise RuntimeError(f"{path.name}: selected_set_json is empty.")

    # parse list of combo keys (selection order)
    try:
        combo_keys_in_order = json.loads(sjson)
    except Exception as e:
        raise RuntimeError(f"Failed to parse selected_set_json in {path.name}: {e}")

    if not isinstance(combo_keys_in_order, list) or not all(isinstance(x, str) for x in combo_keys_in_order):
        raise RuntimeError(f"{path.name}: selected_set_json must be a JSON list[str]. Got: {type(combo_keys_in_order)}")

    # best_k: prefer explicit column, else len(list)
    best_k = None
    if "best_k" in df.columns:
        # pick first finite
        for v in df["best_k"].tolist():
            try:
                fv = int(v)
                if fv > 0:
                    best_k = fv
                    break
            except Exception:
                continue
    if best_k is None:
        best_k = int(len(combo_keys_in_order))

    # ensure we respect best_k prefix
    combo_keys_in_order = combo_keys_in_order[:best_k]
    return best_k, combo_keys_in_order, sjson


# --- Core eval helpers ---

def _dedup_events_keep_first(
    *,
    events: List[dict],
    dedup_eps_s: Optional[float],
    onset_key: str = "cand_start_s",
    offset_key: str = "cand_end_s",
) -> List[dict]:
    """
    Deduplicate events by near-exact (start,end) within eps, keeping the first occurrence.
    """
    if dedup_eps_s is None or dedup_eps_s <= 0:
        return events

    kept: List[dict] = []
    eps = float(dedup_eps_s)

    for ev in events:
        try:
            s = float(ev[onset_key])
            e = float(ev[offset_key])
        except Exception:
            continue

        is_dup = False
        for k in kept:
            try:
                ks = float(k[onset_key])
                ke = float(k[offset_key])
            except Exception:
                continue
            if abs(s - ks) <= eps and abs(e - ke) <= eps:
                is_dup = True
                break

        if not is_dup:
            kept.append(ev)

    return kept


def _compute_metrics_from_pairs_full_gt(
    *,
    gt_events: List[dict],
    cand_events: List[dict],
    gt_cand_pairs: List[Tuple[int, int]],
    counts: Any,
) -> Dict[str, float]:
    """
    Full-GT: TP/FP/FN + Precision/Recall/F1 + EOS (dice + overlap).
    """
    tp = int(counts.tp)
    n_gt = int(counts.n_gt)

    # mean dice / overlap over TP pairs
    if tp > 0:
        dices: List[float] = []
        overlaps: List[float] = []
        for gi, ci in gt_cand_pairs:
            gt = gt_events[gi]
            cand = cand_events[ci]
            gs = float(gt["gt_start_s"])
            ge = float(gt["gt_end_s"])
            cs = float(cand["cand_start_s"])
            ce = float(cand["cand_end_s"])
            overlaps.append(float(overlap_seconds(gs, ge, cs, ce)))
            dices.append(float(dice_event_overlap_score(gs, ge, cs, ce)))
        mean_dice_tp = float(sum(dices) / len(dices))
        mean_overlap_tp = float(sum(overlaps) / len(overlaps))
    else:
        mean_dice_tp = 0.0
        mean_overlap_tp = 0.0

    # dice_eos_recall: sum(dice per GT) / n_gt (FN contributes 0)
    if n_gt > 0 and tp > 0:
        dice_sum = 0.0
        for gi, ci in gt_cand_pairs:
            gt = gt_events[gi]
            cand = cand_events[ci]
            gs = float(gt["gt_start_s"])
            ge = float(gt["gt_end_s"])
            cs = float(cand["cand_start_s"])
            ce = float(cand["cand_end_s"])
            dice_sum += float(dice_event_overlap_score(gs, ge, cs, ce))
        dice_eos_recall = float(dice_sum / float(n_gt))
    else:
        dice_eos_recall = 0.0

    classic = full_gt_metrics(counts)

    return {
        "n_gt": float(counts.n_gt),
        "n_cand": float(counts.n_cand),
        "tp": float(counts.tp),
        "fn": float(counts.fn),
        "fp": float(counts.fp),
        "precision": float(classic["precision"]),
        "recall": float(classic["recall"]),
        "f1": float(classic["f1"]),
        "insertion_rate": float(classic["insertion_rate"]),
        "deletion_rate": float(classic["deletion_rate"]),
        "error_rate": float(classic["error_rate"]),
        "mean_dice_eos_tp": float(mean_dice_tp),
        "dice_eos_recall": float(dice_eos_recall),
        "mean_overlap_s_tp": float(mean_overlap_tp),
    }


def _compute_metrics_from_pairs_part_gt(
    *,
    gt_events: List[dict],
    cand_events: List[dict],
    gt_cand_pairs: List[Tuple[int, int]],
    counts: Any,
) -> Dict[str, float]:
    """
    Partial-GT: recall-oriented reporting with overlap metrics.
    """
    tp = int(counts.tp)
    n_gt = int(counts.n_gt)

    # EOS recall (same definition)
    if n_gt > 0 and tp > 0:
        dice_sum = 0.0
        for gi, ci in gt_cand_pairs:
            gt = gt_events[gi]
            cand = cand_events[ci]
            gs = float(gt["gt_start_s"])
            ge = float(gt["gt_end_s"])
            cs = float(cand["cand_start_s"])
            ce = float(cand["cand_end_s"])
            dice_sum += float(dice_event_overlap_score(gs, ge, cs, ce))
        dice_eos_recall = float(dice_sum / float(n_gt))
    else:
        dice_eos_recall = 0.0
    # still provide mean overlap/dice on TP for interpretability

    if tp > 0:
        dices: List[float] = []
        overlaps: List[float] = []
        for gi, ci in gt_cand_pairs:
            gt = gt_events[gi]
            cand = cand_events[ci]
            gs = float(gt["gt_start_s"])
            ge = float(gt["gt_end_s"])
            cs = float(cand["cand_start_s"])
            ce = float(cand["cand_end_s"])
            overlaps.append(float(overlap_seconds(gs, ge, cs, ce)))
            dices.append(float(dice_event_overlap_score(gs, ge, cs, ce)))
        mean_dice_tp = float(sum(dices) / len(dices))
        mean_overlap_tp = float(sum(overlaps) / len(overlaps))
    else:
        mean_dice_tp = 0.0
        mean_overlap_tp = 0.0

    partial = partial_gt_metrics(counts)

    return {
        "n_gt": float(counts.n_gt),
        "n_cand": float(counts.n_cand),
        "tp": float(counts.tp),
        "fn": float(counts.fn),
        "fp": float(counts.fp),
        "precision": float("nan"),
        "recall": float(partial["recall"]),
        "f1": float("nan"),
        "insertion_rate": float(partial["insertion_rate"]),
        "deletion_rate": float(partial["deletion_rate"]),
        "error_rate": float("nan"),
        "mean_dice_eos_tp": float(mean_dice_tp),
        "dice_eos_recall": float(dice_eos_recall),
        "mean_overlap_s_tp": float(mean_overlap_tp),
    }


def _build_nvv_event_rows(
    *,
    mode: str,
    audio_id: str,
    gt_events: List[dict],
    cand_events: List[dict],
    gt_cand_pairs: List[Tuple[int, int]],
) -> List[Dict[str, Any]]:
    """
    Create event-level rows for pipeline-quality NVV logging.
    """
    rows: List[Dict[str, Any]] = []

    matched_gt = {gi for gi, _ in gt_cand_pairs}
    matched_cand = {ci for _, ci in gt_cand_pairs}

    # hits
    for gi, ci in gt_cand_pairs:
        gt = gt_events[gi]
        cand = cand_events[ci]

        gs = float(gt["gt_start_s"])
        ge = float(gt["gt_end_s"])
        cs = float(cand["cand_start_s"])
        ce = float(cand["cand_end_s"])

        rows.append(
            {
                "mode": str(mode),
                "audio_id": audio_id,
                "gt_event_id": gt.get("gt_event_id"),
                "gt_label": gt.get("gt_label"),
                "cand_event_id": cand.get("cand_event_id"),
                "cand_label": cand.get("cand_label"),
                "status": "hit",
                "dice_eos": float(dice_event_overlap_score(gs, ge, cs, ce)),
                "overlap_s": float(overlap_seconds(gs, ge, cs, ce)),
            }
        )

    # misses
    for gi, gt in enumerate(gt_events):
        if gi in matched_gt:
            continue

        rows.append(
            {
                "mode": str(mode),
                "audio_id": audio_id,
                "gt_event_id": gt.get("gt_event_id"),
                "gt_label": gt.get("gt_label"),
                "cand_event_id": None,
                "cand_label": None,
                "status": "miss",
                "dice_eos": 0.0,
                "overlap_s": 0.0,
            }
        )

    # insertions
    for ci, cand in enumerate(cand_events):
        if ci in matched_cand:
            continue

        rows.append(
            {
                "mode": str(mode),
                "audio_id": audio_id,
                "gt_event_id": None,
                "gt_label": None,
                "cand_event_id": cand.get("cand_event_id"),
                "cand_label": cand.get("cand_label"),
                "status": "insertion",
                "dice_eos": None,
                "overlap_s": None,
            }
        )

    return rows


def _evaluate_union_for_audio_id(
    *,
    mode: str,
    audio_id: str,
    workspace_root: Path,
    gt_dict: Dict[str, List[dict]],
    combo_keys_in_order: List[str],
    cache: Optional[Dict[Tuple[str, str], List[dict]]],
    dedup_eps_s: Optional[float],
    match_params: Dict[str, Any],
    verbose_missing: bool = False,
) -> Dict[str, float]:
    gt_events = gt_dict.get(audio_id, [])
    union_events: List[dict] = []

    for ck in combo_keys_in_order:
        key = (audio_id, ck)
        if cache is not None and key in cache:
            events = cache[key]
        else:
            path = get_nvv_json_path_from_combo_key(workspace=workspace_root, audio_id=audio_id, combo_key=ck)
            if not path.exists():
                if verbose_missing:
                    print(f"⚠ Missing candidate file: {path}")
                events = []
            else:
                events = load_candidate_events_from_nvv_json(path)
            if cache is not None:
                cache[key] = events

        union_events.extend(events)
        if dedup_eps_s is not None and dedup_eps_s > 0:
            union_events = _dedup_events_keep_first(events=union_events, dedup_eps_s=dedup_eps_s)

    counts, gt_cand_pairs = match_events_optimal(
        gt_events,
        union_events,
        gt_onset_key="gt_start_s",
        gt_offset_key="gt_end_s",
        gt_label_key="gt_label",
        gt_id_key="gt_event_id",
        cand_onset_key="cand_start_s",
        cand_offset_key="cand_end_s",
        cand_label_key="cand_label",
        cand_id_key="cand_event_id",
        **match_params,
    )

    if str(mode) == "full_gt":
        metrics = _compute_metrics_from_pairs_full_gt(
            gt_events=gt_events,
            cand_events=union_events,
            gt_cand_pairs=gt_cand_pairs,
            counts=counts,
        )
    elif str(mode) == "part_gt":
        metrics = _compute_metrics_from_pairs_part_gt(
            gt_events=gt_events,
            cand_events=union_events,
            gt_cand_pairs=gt_cand_pairs,
            counts=counts,
        )
    else:
        raise ValueError(f"Unknown mode: {mode!r} (expected 'full_gt' or 'part_gt')")

    nvv_event_rows = _build_nvv_event_rows(
        mode=mode,
        audio_id=audio_id,
        gt_events=gt_events,
        cand_events=union_events,
        gt_cand_pairs=gt_cand_pairs,
    )
    metrics["nvv_event_rows"] = nvv_event_rows

    return metrics


# --- Public API ---

def run_pipeline_capability_evaluation(
    *,
    workspace_root: str | Path,
    evaluation_dir: str | Path,
    gt_dict: Dict[str, List[dict]],
    mode: str,
    evaluable_ids: Optional[List[str]] = None,
    dedup_eps_s: Optional[float] = None,
    preload: bool = True,
    verbose_missing: bool = False,
    t_collar: float = EVAL_T_COLLAR,
    percentage_of_length: float = EVAL_PERCENTAGE_OF_LENGTH,
    evaluate_onset: bool = True,
    evaluate_offset: bool = True,
    match_labels: bool = False,
    write_files: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stage-2 evaluation ("Pipeline Quality") using the selected best-k set.

    Reads:
      <evaluation_dir>/<mode>/global_best_k_set_<mode>.csv
    Uses:  per_audio/<audio_id>/annotations/nvv/<audio_id>_<combo_key>.json

    Writes (if write_files=True):
      <evaluation_dir>/<mode>/pipeline_capability_<mode>_summary.csv
      <evaluation_dir>/<mode>/pipeline_capability_<mode>_per_audio.csv
      <evaluation_dir>/<mode>/_pipeline_capability_<mode>.xlsx (tabs: Summary, PerAudio)
      <evaluation_dir>/<mode>/pipeline_capability_<mode>_nvv_events.csv (detailed event-level log for NVV analysis)

    Returns:
      summary_df (1 row), per_audio_df (N rows)
    """
    if mode not in ("full_gt", "part_gt"):
        raise ValueError(f"mode must be 'full_gt' or 'part_gt', got: {mode}")

    workspace_root = Path(workspace_root)
    evaluation_dir = Path(evaluation_dir)
    mode_dir = get_eval_mode_dir(evaluation_dir, mode)
    ensure_dir(mode_dir)

    if evaluable_ids is None:
        evaluable_ids = _list_audio_ids_from_workspace(workspace_root)
    evaluable_ids = list(evaluable_ids)

    best_k, combo_keys_in_order, selected_set_json = _read_best_k_set_csv(mode_dir.parent, mode=str(mode))

    match_params = {
        "match_labels": bool(match_labels),
        "evaluate_onset": bool(evaluate_onset),
        "evaluate_offset": bool(evaluate_offset),
        "t_collar": float(t_collar),
        "percentage_of_length": float(percentage_of_length),
    }

    cache: Optional[Dict[Tuple[str, str], List[dict]]] = {} if preload else None

    per_rows: List[Dict[str, Any]] = []
    nvv_event_rows_all: List[Dict[str, Any]] = []
    for aid in evaluable_ids:
        m = _evaluate_union_for_audio_id(
            mode=str(mode),
            audio_id=aid,
            workspace_root=workspace_root,
            gt_dict=gt_dict,
            combo_keys_in_order=combo_keys_in_order,
            cache=cache,
            dedup_eps_s=dedup_eps_s,
            match_params=match_params,
            verbose_missing=verbose_missing,
        )

        nvv_event_rows = m.pop("nvv_event_rows")
        nvv_event_rows_all.extend(nvv_event_rows)

        per_rows.append(
            {
                "mode": str(mode),
                "audio_id": aid,
                "best_k": int(best_k),
                "dedup_eps_s": float(dedup_eps_s) if (dedup_eps_s is not None and dedup_eps_s > 0) else float("nan"),
                "n_selected_tracks": int(len(combo_keys_in_order)),
                "selected_set_json": selected_set_json,
                **m,
            }
        )

    per_audio_df = pd.DataFrame(per_rows)
    nvv_events_df = pd.DataFrame(nvv_event_rows_all)

    # macro means (include zeros, keep NaNs where intentional)
    def _nanmean(series: pd.Series) -> float:
        # keep NaNs out for macro means, but if all NaN -> NaN
        if series.dropna().shape[0] == 0:
            return float("nan")
        return float(series.dropna().mean())

    macro = {}
    for col in [
        "n_gt", "n_cand", "tp", "fn", "fp",
        "insertion_rate", "deletion_rate", "error_rate",
        "precision", "recall", "f1",
        "mean_dice_eos_tp", "dice_eos_recall", "mean_overlap_s_tp",
    ]:
        if col in per_audio_df.columns:
            macro[f"macro_mean_{col}"] = _nanmean(per_audio_df[col])

    summary_row = {
        "mode": str(mode),
        "best_k": int(best_k),
        "n_audio_ids": int(len(evaluable_ids)),
        "dedup_eps_s": float(dedup_eps_s) if (dedup_eps_s is not None and dedup_eps_s > 0) else float("nan"),
        "n_selected_tracks": int(len(combo_keys_in_order)),
        "selected_set_json": selected_set_json,
        **macro,
    }
    summary_df = pd.DataFrame([summary_row])

    if write_files:
        out_summary = get_pipeline_capability_summary_csv_path(evaluation_dir, mode)
        out_per = get_pipeline_capability_per_audio_csv_path(evaluation_dir, mode)
        out_xlsx = get_pipeline_capability_xlsx_path(evaluation_dir, mode)
        out_nvv_events = get_pipeline_capability_nvv_events_csv_path(evaluation_dir, mode)

        write_csv_atomic(df=summary_df, out_path=out_summary)
        write_csv_atomic(df=per_audio_df, out_path=out_per)
        write_csv_atomic(df=nvv_events_df, out_path=out_nvv_events)

        write_xlsx_atomic(
            detailed_df=per_audio_df,
            summary_df=summary_df,
            out_path=out_xlsx,
            detailed_sheet="PerAudio",
            summary_sheet="Summary",
        )

    return summary_df, per_audio_df