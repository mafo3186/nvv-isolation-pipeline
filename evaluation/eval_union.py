from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from config.path_factory import (
    get_workspace_paths,
    get_nvv_json_path_from_combo_key,
)
from evaluation.eval_event_matching import match_events_optimal
from evaluation.eval_metrics import overlap_seconds, dice_event_overlap_score


def list_audio_ids_from_workspace(workspace_root: Path) -> List[str]:
    """
    List audio_ids from workspace_root/per_audio/* directories.

    Args:
        workspace_root: Workspace root directory.

    Returns:
        Sorted list of per-audio directory names (audio_ids).
    """
    per_audio = get_workspace_paths(workspace_root).per_audio
    if not per_audio.exists():
        return []
    return sorted([p.name for p in per_audio.iterdir() if p.is_dir()])


def dedup_events_keep_first(
    *,
    events: List[dict],
    dedup_eps_s: Optional[float],
    onset_key: str = "cand_start_s",
    offset_key: str = "cand_end_s",
) -> List[dict]:
    """
    Deduplicate events by near-exact (start,end) within eps, keeping the first occurrence.

    Args:
        events: Candidate events list.
        dedup_eps_s: If None or <=0 -> no dedup.
        onset_key: Key for event onset in seconds.
        offset_key: Key for event offset in seconds.

    Returns:
        Deduplicated list of events.
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


def compute_metrics_from_pairs(
    *,
    gt_events: List[dict],
    cand_events: List[dict],
    gt_cand_pairs: List[Tuple[int, int]],
    counts: Any,
) -> Dict[str, float]:
    """
    Compute summary metrics for one union set from matching outputs.

    Notes:
        - Dice/overlap are computed over TP pairs only.
        - dice_eos_recall = sum(dice over matched GT rows) / n_gt (FN contributes 0).
        - Precision/recall/f1 are computed in the "full GT" style:
          if no candidates -> precision=0; if no GT -> recall=0.

    Args:
        gt_events: GT events list (canonical keys: gt_start_s, gt_end_s).
        cand_events: Candidate events list (canonical keys: cand_start_s, cand_end_s).
        gt_cand_pairs: List of (gt_idx, cand_idx).
        counts: MatchCounts-like object returned by match_events_optimal.

    Returns:
        Dict of numeric metrics (floats).
    """
    tp = int(counts.tp)
    fn = int(counts.fn)
    fp = int(counts.fp)
    n_gt = int(counts.n_gt)
    n_cand = int(counts.n_cand)

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

    precision = (tp / float(n_cand)) if n_cand > 0 else 0.0
    recall = (tp / float(n_gt)) if n_gt > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "n_gt": float(n_gt),
        "n_cand": float(n_cand),
        "tp": float(tp),
        "fn": float(fn),
        "fp": float(fp),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mean_dice_eos_tp": float(mean_dice_tp),
        "dice_eos_recall": float(dice_eos_recall),
        "mean_overlap_s_tp": float(mean_overlap_tp),
    }


def evaluate_union_for_audio_id(
    *,
    audio_id: str,
    workspace_root: Path,
    gt_dict: Dict[str, List[dict]],
    combo_keys_in_order: List[str],
    load_candidate_events_fn: Any,
    cache: Optional[Dict[Tuple[str, str], List[dict]]],
    dedup_eps_s: Optional[float],
    match_params: Dict[str, Any],
) -> Dict[str, float]:
    """
    Evaluate union(combo_keys_in_order) for one audio_id.

    STRICT behavior:
        - Missing candidate file => raises FileNotFoundError.
        - Candidate parsing errors should be handled inside load_candidate_events_fn
          (it may raise; we do not swallow it here).

    Args:
        audio_id: Audio ID.
        workspace_root: Workspace root.
        gt_dict: Ground truth dict audio_id -> events.
        combo_keys_in_order: Selected combo keys (order defines dedup precedence).
        load_candidate_events_fn: Callable(Path) -> List[dict] with canonical candidate keys.
        cache: Optional dict cache[(audio_id, combo_key)] -> events.
        dedup_eps_s: Dedup epsilon, optional.
        match_params: Forwarded to match_events_optimal.

    Returns:
        Dict of metrics for this audio_id (floats).
    """
    gt_events = gt_dict.get(audio_id, [])
    union_events: List[dict] = []

    for ck in combo_keys_in_order:
        key = (audio_id, ck)

        if cache is not None and key in cache:
            events = cache[key]
        else:
            path = get_nvv_json_path_from_combo_key(
                workspace=workspace_root,
                audio_id=audio_id,
                combo_key=ck,
            )
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing candidate file for union evaluation: "
                    f"audio_id={audio_id}, combo_key={ck}, path={path}"
                )

            events = load_candidate_events_fn(path)

            if cache is not None:
                cache[key] = events

        union_events.extend(events)
        if dedup_eps_s is not None and dedup_eps_s > 0:
            union_events = dedup_events_keep_first(
                events=union_events,
                dedup_eps_s=dedup_eps_s,
            )

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

    return compute_metrics_from_pairs(
        gt_events=gt_events,
        cand_events=union_events,
        gt_cand_pairs=gt_cand_pairs,
        counts=counts,
    )