"""
Event Matching (SED-style optimal matching with collars)

This module implements optimal 1:1 event matching equivalent to the
event-based matching logic used in:

    Mesaros, A., Heittola, T., & Virtanen, T. (2016).
    Metrics for Polyphonic Sound Event Detection.
    Applied Sciences, 6(6), 162.
    https://doi.org/10.3390/app6060162

and the implementation found in:

    sed_eval (TUT-ARG)
    https://github.com/TUT-ARG/sed_eval

Specifically reproduced:
    • Event-based hit validation (onset collar + offset collar rule)
    • Maximum cardinality bipartite matching
    • Derivation of TP / FP / FN from optimal 1:1 matching

Not included:
    • Segment-based metrics
    • Substitution handling (label): irrelevant for single-class NVV but kept as option
    • Any additional reporting utilities

This implementation is intentionally minimal and tailored for
single-class NVV evaluation.
"""
from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Any
import math

from evaluation.eval_metrics import MatchCounts
from config.params import EVAL_T_COLLAR, EVAL_PERCENTAGE_OF_LENGTH

def validate_onset(gt: dict, cand: dict, *, gt_onset_key: str, cand_onset_key: str, t_collar: float) -> bool:
    """sed_eval-style onset validation: |gt_onset - cand_onset| <= t_collar"""
    return math.fabs(float(gt[gt_onset_key]) - float(cand[cand_onset_key])) <= t_collar

# Note: sed_eval applies the percentage_of_length tolerance only to the offset condition.
# This follows the DCASE event-based evaluation definition (Mesaros et al., 2016).
# For NVVs, this asymmetry (fixed onset collar vs. length-dependent offset collar) is under discussion.
def validate_offset(
    gt: dict,
    cand: dict,
    *,
    gt_onset_key: str,
    gt_offset_key: str,
    cand_offset_key: str,
    t_collar: float,
    percentage_of_length: float,
) -> bool:
    """
    sed_eval-style offset validation:
        |gt_offset - cand_offset| <= max(t_collar, percentage_of_length * gt_length)
    """
    gt_on = float(gt[gt_onset_key])
    gt_off = float(gt[gt_offset_key])
    cand_off = float(cand[cand_offset_key])

    gt_len = gt_off - gt_on
    if gt_len <= 0:
        return False

    return math.fabs(gt_off - cand_off) <= max(t_collar, percentage_of_length * gt_len)


def _is_hit(
    gt: dict,
    cand: dict,
    *,
    gt_onset_key: str,
    gt_offset_key: str,
    gt_label_key: str,
    cand_onset_key: str,
    cand_offset_key: str,
    cand_label_key: str,
    match_labels: Optional[bool],
    evaluate_onset: bool,
    evaluate_offset: bool,
    t_collar: float,
    percentage_of_length: float,
) -> bool:
    """Check if candidate event matches GT event under sed_eval-style hit conditions.   
        - match_labels: if True, event_label must also match (optional for single-class NVV)
        - evaluate_onset: if True, apply onset collar validation
        - evaluate_offset: if True, apply offset collar validation
    """
    if match_labels is True:
        if str(gt.get(gt_label_key, "")) != str(cand.get(cand_label_key, "")):
            return False

    if evaluate_onset and not validate_onset(
        gt, cand, gt_onset_key=gt_onset_key, cand_onset_key=cand_onset_key, t_collar=t_collar
    ):
        return False

    if evaluate_offset and not validate_offset(
        gt,
        cand,
        gt_onset_key=gt_onset_key,
        gt_offset_key=gt_offset_key,
        cand_offset_key=cand_offset_key,
        t_collar=t_collar,
        percentage_of_length=percentage_of_length,
    ):
        return False

    return True


def bipartite_match(graph: Dict[int, List[int]]) -> Dict[int, int]:
    """
    Maximum cardinality matching in a bipartite graph.
    (Function is borrowed from sed_eval.util.event_matching.bipartite_match: https://github.com/TUT-ARG/sed_eval/blob/master/sed_eval/util/event_matching.py)
    Args:
        graph:
            cand_idx -> list of gt_idx that this candidate can match.
        returns:
            gt_idx -> cand_idx (for the optimal 1:1 matching)
    """
    matching: Dict[int, int] = {}

    # greedy init
    for u in graph:
        for v in graph[u]:
            if v not in matching:
                matching[v] = u
                break

    while True:
        preds: Dict[int, List[int]] = {}
        unmatched: List[int] = []
        pred = {u: unmatched for u in graph}

        for v in list(matching.keys()):
            u = matching[v]
            if u in pred:
                del pred[u]

        layer = list(pred)

        while layer and not unmatched:
            new_layer: Dict[int, List[int]] = {}
            for u in layer:
                for v in graph[u]:
                    if v not in preds:
                        new_layer.setdefault(v, []).append(u)

            layer = []
            for v in new_layer:
                preds[v] = new_layer[v]

                if v in matching:
                    layer.append(matching[v])
                    pred[matching[v]] = v
                else:
                    unmatched.append(v)

        if not unmatched:
            return matching

        def recurse(v: int) -> bool:
            if v in preds:
                L = preds[v]
                del preds[v]
                for u in L:
                    if u in pred:
                        pu = pred[u]
                        del pred[u]
                        if pu is unmatched or recurse(pu):
                            matching[v] = u
                            return True
            return False

        for v in unmatched:
            recurse(v)


def pairs_idx_to_ids(
    pairs: List[Tuple[int, int]],
    gt_events: List[dict],
    cand_events: List[dict],
    *,
    gt_id_key: str = "gt_event_id",
    cand_id_key: str = "cand_event_id",
) -> List[Tuple[Any, Any]]:
    """
    Convert index pairs (gt_idx, cand_idx) to ID pairs (gt_event_id, cand_event_id).
    No IO, only dict lookups.
    """
    out: List[Tuple[Any, Any]] = []
    for gi, ci in pairs:
        gt_id = gt_events[gi].get(gt_id_key)
        cand_id = cand_events[ci].get(cand_id_key)
        out.append((gt_id, cand_id))
    return out

def match_events_optimal(
    gt_events: List[dict],
    cand_events: List[dict],
    *,
    gt_onset_key: str = "gt_start_s",
    gt_offset_key: str = "gt_end_s",
    gt_label_key: str = "gt_label",
    gt_id_key: str = "gt_event_id",
    cand_onset_key: str = "cand_start_s",
    cand_offset_key: str = "cand_end_s",
    cand_label_key: str = "cand_label",
    cand_id_key: str = "cand_event_id",
    match_labels: Optional[bool] = None,
    evaluate_onset: bool = True,
    evaluate_offset: bool = True,
    t_collar: float = EVAL_T_COLLAR,
    percentage_of_length: float = EVAL_PERCENTAGE_OF_LENGTH,
) -> Tuple[MatchCounts, List[Tuple[int, int]]]:
    """
    Optimal 1:1 matching under collar rules.

    Returns:
        MatchCounts
        pairs: list of (gt_index of gt_events, cand_index of cand_events) for matched pairs (TPs)
    """
    n_gt = len(gt_events)
    n_cand = len(cand_events)

    graph: Dict[int, List[int]] = {}

    for ci, cand in enumerate(cand_events):
        hits: List[int] = []
        for gi, gt in enumerate(gt_events):
            if _is_hit(
                gt,
                cand,
                gt_onset_key=gt_onset_key,
                gt_offset_key=gt_offset_key,
                gt_label_key=gt_label_key,
                cand_onset_key=cand_onset_key,
                cand_offset_key=cand_offset_key,
                cand_label_key=cand_label_key,
                match_labels=match_labels,
                evaluate_onset=evaluate_onset,
                evaluate_offset=evaluate_offset,
                t_collar=t_collar,
                percentage_of_length=percentage_of_length,
            ):
                hits.append(gi)
        if hits:
            graph[ci] = hits

    gt_to_cand = bipartite_match(graph)
    gt_cand_pairs = sorted((gi, ci) for gi, ci in gt_to_cand.items())

    tp = len(gt_cand_pairs)
    fp = n_cand - tp
    fn = n_gt - tp

    return MatchCounts(tp=tp, fp=fp, fn=fn, n_gt=n_gt, n_cand=n_cand), gt_cand_pairs