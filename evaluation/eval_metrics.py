from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class MatchCounts:
    """
    Neutral counts derived from 1:1 matching between GT events and candidate events.

    Args:
        tp: Number of matched pairs (true positives).
        fp: Number of unmatched candidate events (insertions).
        fn: Number of unmatched GT events (deletions).
        n_gt: Total number of GT events.
        n_cand: Total number of candidate events.
    """
    tp: int
    fp: int
    fn: int
    n_gt: int
    n_cand: int


# Atomic score functions

def precision(tp: int, n_cand: int) -> float:
    """Precision = TP / Nsys. Returns 0.0 if n_cand == 0."""
    return 0.0 if n_cand == 0 else tp / float(n_cand)


def recall(tp: int, n_gt: int) -> float:
    """Recall = TP / Nref. Returns 0.0 if n_gt == 0."""
    return 0.0 if n_gt == 0 else tp / float(n_gt)


def f1(p: float, r: float) -> float:
    """F1 = 2PR/(P+R). Returns 0.0 if P+R == 0."""
    return 0.0 if (p + r) <= 0.0 else (2.0 * p * r) / (p + r)


def insertion_rate(fp: int, n_gt: int) -> float:
    """
    Insertion rate (DCASE / SED naming) = FP / Ngt.
    Returns 0.0 if n_gt == 0.
    """
    return 0.0 if n_gt == 0 else fp / float(n_gt)


def deletion_rate(fn: int, n_gt: int) -> float:
    """
    Deletion rate (DCASE / SED naming) = FN / Ngt.
    Returns 0.0 if n_gt == 0.
    """
    return 0.0 if n_gt == 0 else fn / float(n_gt)


def error_rate(fp: int, fn: int, n_gt: int) -> float:
    """
    Event-based error rate without substitutions (single-class NVV):
    ER = (FP + FN) / Ngt.
    Returns 0.0 if n_gt == 0.
    """
    return 0.0 if n_gt == 0 else (fp + fn) / float(n_gt)


# Overlap helpers + Dice overlap

def overlap_seconds(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Return overlap duration (sec) between two intervals."""
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def has_overlap(a_start: float, a_end: float, b_start: float, b_end: float, min_ms: float = 10.0) -> bool:
    """Check overlap ≥ min_ms milliseconds."""
    return overlap_seconds(a_start, a_end, b_start, b_end) >= (min_ms / 1000.0)


def dice_event_overlap_score(a_start: float, a_end: float, b_start: float, b_end: float, ndigits: int = 3) -> float:
    """
    Sørensen–Dice / event overlap score between two time-interval events.

    Formula:
        Dice = 2 * overlap / (dur_a + dur_b)

    where:
        overlap = max(0, min(a_end, b_end) - max(a_start, b_start))
        dur_a   = max(0, a_end - a_start)
        dur_b   = max(0, b_end - b_start)

    Notes:
        - Returns 0.0 when there is no overlap or when one duration is zero.
        - Output is in [0.0, 1.0] and rounded to ndigits.

    Args:
        a_start/a_end: Interval A in seconds.
        b_start/b_end: Interval B in seconds.
        ndigits: Rounding digits.

    Returns:
        Dice overlap score (float).
    """
    overlap = overlap_seconds(a_start, a_end, b_start, b_end)
    dur_a = max(0.0, a_end - a_start)
    dur_b = max(0.0, b_end - b_start)

    if overlap <= 0.0 or dur_a == 0.0 or dur_b == 0.0:
        return 0.0

    return round((2.0 * overlap) / (dur_a + dur_b), ndigits)

# toDo: move mean_dice_eos_tp dice_eos_recall and mean_overlap_s_tp here from eval_union.py

# Small collectors (optional, but convenient)

def full_gt_metrics(c: MatchCounts) -> dict:
    """
    Full-GT report: precision/recall/f1 + insertion/deletion/error rates.
    """
    p = precision(c.tp, c.n_cand)
    r = recall(c.tp, c.n_gt)
    return {
        "tp": c.tp,
        "fp": c.fp,
        "fn": c.fn,
        "n_gt": c.n_gt,
        "n_cand": c.n_cand,
        "precision": p,
        "recall": r,
        "f1": f1(p, r),
        "insertion_rate": insertion_rate(c.fp, c.n_gt),
        "deletion_rate": deletion_rate(c.fn, c.n_gt),
        "error_rate": error_rate(c.fp, c.fn, c.n_gt),
    }


def partial_gt_metrics(c: MatchCounts) -> dict:
    """
    Partial-GT report: recall + deletions, and still report insertions as a neutral count/rate.
    (No precision/F1 interpretation.)
    """
    return {
        "tp": c.tp,
        "insertions": c.fp,
        "deletions": c.fn,
        "n_gt": c.n_gt,
        "n_cand": c.n_cand,
        "recall": recall(c.tp, c.n_gt),
        "deletion_rate": deletion_rate(c.fn, c.n_gt),
        "insertion_rate": insertion_rate(c.fp, c.n_gt),
    }