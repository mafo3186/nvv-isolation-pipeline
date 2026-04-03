from __future__ import annotations

from evaluation.eval_io import validate_mode


ORDERED_METRICS_BY_MODE = {
    "full_gt": [
        "macro_mean_f1",
        "macro_mean_recall",
        "macro_mean_dice_eos_recall",
        "macro_mean_mean_dice_eos_tp",
        "macro_mean_insertion_rate",
    ],
    "part_gt": [
        "macro_mean_recall",
        "macro_mean_dice_eos_recall",
        "macro_mean_mean_dice_eos_tp",
        "macro_mean_insertion_rate",
    ],
}

METRIC_SORT_ASCENDING = {
    "macro_mean_f1": False,
    "macro_mean_recall": False,
    "macro_mean_dice_eos_recall": False,
    "macro_mean_mean_dice_eos_tp": False,
    "macro_mean_insertion_rate": True,
}

METRIC_LABELS = {
    "macro_mean_f1": "F1",
    "macro_mean_recall": "Recall",
    "macro_mean_dice_eos_recall": "EOS Recall",
    "macro_mean_mean_dice_eos_tp": "Mean EOS TP",
    "macro_mean_insertion_rate": "Insertion Rate",
    "macro_mean_deletion_rate": "Deletion Rate",
    "macro_mean_error_rate": "Error Rate",
    "macro_mean_fp": "FP",
    "macro_mean_n_cand": "n_cand",
}


def get_ordered_metric_names(mode: str) -> list[str]:
    """
    Return ordered metric names for ranking and display.

    Args:
        mode: Evaluation mode ("full_gt" or "part_gt").

    Returns:
        Ordered metric column names from most to least important.
    """
    validate_mode(mode)
    return list(ORDERED_METRICS_BY_MODE[mode])


def get_metric_sort_ascending(metric_name: str) -> bool:
    return METRIC_SORT_ASCENDING[metric_name]


def get_metric_label(metric_name: str) -> str:
    """
    Return a compact display label for a metric column.

    Args:
        metric_name: Internal metric column name.

    Returns:
        Human-readable label.
    """
    return METRIC_LABELS.get(metric_name, metric_name)