from __future__ import annotations

from typing import Optional
import pandas as pd


from evaluation.eval_io import validate_mode

def _filter_dataset(df: pd.DataFrame, dataset_name: Optional[str]) -> pd.DataFrame:
    """
    Optionally filter a DataFrame by dataset_name.

    Args:
        df: Input DataFrame.
        dataset_name: Dataset name to keep. If None, no filtering is applied.

    Returns:
        Filtered copy of the input DataFrame.
    """
    result = df.copy()

    if dataset_name is None:
        return result

    if "dataset_name" not in result.columns:
        raise KeyError("Expected column 'dataset_name' for dataset filtering.")

    return result[result["dataset_name"] == dataset_name].copy()


def get_primary_metric_name(mode: str, rq: str = "rq1") -> str:
    """
    Return the primary metric name for a given RQ and mode.

    Args:
        mode: Evaluation mode ("full_gt" or "part_gt").
        rq: Research question key.

    Returns:
        Column name of the primary metric.
    """
    validate_mode(mode)

    if rq == "rq1":
        return "macro_mean_f1" if mode == "full_gt" else "macro_mean_recall"

    if rq == "rq2a":
        return "macro_mean_f1" if mode == "full_gt" else "macro_mean_recall"

    if rq == "rq3":
        return "macro_mean_f1" if mode == "full_gt" else "macro_mean_recall"

    raise ValueError(f"Unsupported rq='{rq}'.")


def get_secondary_metric_names(mode: str, rq: str = "rq1") -> list[str]:
    """
    Return secondary sort metrics for a given RQ and mode.

    Args:
        mode: Evaluation mode ("full_gt" or "part_gt").
        rq: Research question key.

    Returns:
        Ordered list of secondary metric column names.
    """
    validate_mode(mode)

    if rq in {"rq1", "rq2a"}:
        if mode == "full_gt":
            return ["macro_mean_recall", "macro_mean_mean_dice_eos_tp"]
        return ["macro_mean_dice_eos_recall", "macro_mean_mean_dice_eos_tp"]

    if rq == "rq3":
        if mode == "full_gt":
            return ["macro_mean_recall", "macro_mean_mean_dice_eos_tp"]
        return ["macro_mean_mean_dice_eos_tp"]

    raise ValueError(f"Unsupported rq='{rq}'.")


def _sort_by_metrics(
    df: pd.DataFrame,
    metric_cols: list[str],
) -> pd.DataFrame:
    """
    Sort a DataFrame descending by the provided metric columns.

    Args:
        df: Input DataFrame.
        metric_cols: Ordered metric columns.

    Returns:
        Sorted DataFrame copy.
    """
    missing = [c for c in metric_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required metric columns: {missing}")

    return df.sort_values(by=metric_cols, ascending=[False] * len(metric_cols)).reset_index(drop=True)


def get_top_runs(
    df_rq1: pd.DataFrame,
    mode: str,
    dataset_name: Optional[str] = None,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Return the best RQ1 runs for one mode and optionally one dataset.

    Args:
        df_rq1: Experiment- or workspace-level RQ1 DataFrame.
        mode: Evaluation mode.
        dataset_name: Optional dataset filter.
        top_n: Number of rows to return.

    Returns:
        Sorted top-n DataFrame.
    """
    validate_mode(mode)

    df = df_rq1.copy()

    if "mode" not in df.columns:
        raise KeyError("Expected column 'mode' in RQ1 DataFrame.")

    df = df[df["mode"] == mode].copy()
    df = _filter_dataset(df, dataset_name)

    metric_cols = [get_primary_metric_name(mode, "rq1"), *get_secondary_metric_names(mode, "rq1")]
    df = _sort_by_metrics(df, metric_cols)

    return df.head(top_n).reset_index(drop=True)


def get_top_region_runs(
    df_rq1: pd.DataFrame,
    mode: str,
    dataset_name: Optional[str] = None,
    score_fraction: float = 0.95,
) -> pd.DataFrame:
    """
    Keep all RQ1 runs within a fraction of the best primary score.

    Args:
        df_rq1: Experiment- or workspace-level RQ1 DataFrame.
        mode: Evaluation mode.
        dataset_name: Optional dataset filter.
        score_fraction: Fraction of best score to keep, e.g. 0.95.

    Returns:
        Filtered DataFrame with helper score columns.
    """
    validate_mode(mode)

    if not (0 < score_fraction <= 1):
        raise ValueError("score_fraction must be in the range (0, 1].")

    df = df_rq1.copy()

    if "mode" not in df.columns:
        raise KeyError("Expected column 'mode' in RQ1 DataFrame.")

    df = df[df["mode"] == mode].copy()
    df = _filter_dataset(df, dataset_name)

    primary_metric = get_primary_metric_name(mode, "rq1")
    if primary_metric not in df.columns:
        raise KeyError(f"Expected column '{primary_metric}' in RQ1 DataFrame.")

    best_score = df[primary_metric].max()
    cutoff = best_score * score_fraction

    result = df[df[primary_metric] >= cutoff].copy()
    result["primary_score"] = result[primary_metric]
    result["score_fraction_of_best"] = result[primary_metric] / best_score if best_score != 0 else 0.0

    metric_cols = [primary_metric, *get_secondary_metric_names(mode, "rq1")]
    return _sort_by_metrics(result, metric_cols)


def get_parameter_value_summary(
    df_top_region: pd.DataFrame,
    param_name: str,
    primary_metric: str,
) -> pd.DataFrame:
    """
    Summarize one parameter across a top-region subset.

    Args:
        df_top_region: DataFrame returned by get_top_region_runs().
        param_name: Parameter column to summarize.
        primary_metric: Primary score column.

    Returns:
        Grouped summary by parameter value.
    """
    if param_name not in df_top_region.columns:
        raise KeyError(f"Expected parameter column '{param_name}'.")

    if primary_metric not in df_top_region.columns:
        raise KeyError(f"Expected metric column '{primary_metric}'.")

    summary = (
        df_top_region
        .groupby(param_name, dropna=False)
        .agg(
            n_runs=("run_id", "count") if "run_id" in df_top_region.columns else (primary_metric, "count"),
            mean_primary_score=(primary_metric, "mean"),
            median_primary_score=(primary_metric, "median"),
            min_primary_score=(primary_metric, "min"),
            max_primary_score=(primary_metric, "max"),
        )
        .reset_index()
    )

    total = len(df_top_region)
    summary["share_of_top_region"] = summary["n_runs"] / total if total > 0 else 0.0

    return summary.sort_values(
        by=["mean_primary_score", "max_primary_score", "n_runs"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def get_parameter_pair_summary(
    df_top_region: pd.DataFrame,
    param_x: str,
    param_y: str,
    primary_metric: str,
) -> pd.DataFrame:
    """
    Summarize parameter pairs across a top-region subset.

    Args:
        df_top_region: DataFrame returned by get_top_region_runs().
        param_x: First parameter column.
        param_y: Second parameter column.
        primary_metric: Primary score column.

    Returns:
        Grouped summary by parameter pair.
    """
    for col in [param_x, param_y, primary_metric]:
        if col not in df_top_region.columns:
            raise KeyError(f"Expected column '{col}'.")

    summary = (
        df_top_region
        .groupby([param_x, param_y], dropna=False)
        .agg(
            n_runs=("run_id", "count") if "run_id" in df_top_region.columns else (primary_metric, "count"),
            mean_primary_score=(primary_metric, "mean"),
            median_primary_score=(primary_metric, "median"),
            best_primary_score=(primary_metric, "max"),
        )
        .reset_index()
    )

    return summary.sort_values(
        by=["mean_primary_score", "best_primary_score", "n_runs"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def get_combo_key_summary(
    df_rq2a: pd.DataFrame,
    mode: str,
    dataset_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Summarize combo_key performance across runs.

    Args:
        df_rq2a: Experiment- or workspace-level RQ2a DataFrame.
        mode: Evaluation mode.
        dataset_name: Optional dataset filter.

    Returns:
        Summary DataFrame per combo_key.
    """
    validate_mode(mode)

    df = df_rq2a.copy()

    if "mode" not in df.columns:
        raise KeyError("Expected column 'mode' in RQ2a DataFrame.")

    required = ["combo_key", "vad_mask", "asr_audio_in", "rank_within_run"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required RQ2a columns: {missing}")

    df = df[df["mode"] == mode].copy()
    df = _filter_dataset(df, dataset_name)

    primary_metric = get_primary_metric_name(mode, "rq2a")
    secondary_metrics = get_secondary_metric_names(mode, "rq2a")

    agg_spec = {
        "vad_mask": ("vad_mask", "first"),
        "asr_audio_in": ("asr_audio_in", "first"),
        "n_rows": ("combo_key", "count"),
        "mean_rank": ("rank_within_run", "mean"),
        "median_rank": ("rank_within_run", "median"),
        "best_rank": ("rank_within_run", "min"),
        "worst_rank": ("rank_within_run", "max"),
        "top1_count": ("rank_within_run", lambda s: int((s == 1).sum())),
        "top3_count": ("rank_within_run", lambda s: int((s <= 3).sum())),
        f"mean_{primary_metric}": (primary_metric, "mean"),
        f"best_{primary_metric}": (primary_metric, "max"),
        f"worst_{primary_metric}": (primary_metric, "min"),
    }

    for metric in secondary_metrics:
        if metric in df.columns:
            agg_spec[f"mean_{metric}"] = (metric, "mean")
            agg_spec[f"best_{metric}"] = (metric, "max")
            agg_spec[f"worst_{metric}"] = (metric, "min")

    if "macro_mean_n_cand" in df.columns:
        agg_spec["mean_macro_mean_n_cand"] = ("macro_mean_n_cand", "mean")

    summary = df.groupby("combo_key", dropna=False).agg(**agg_spec).reset_index()

    total_runs = df["run_id"].nunique() if "run_id" in df.columns else len(df)
    summary["top1_share"] = summary["top1_count"] / total_runs if total_runs > 0 else 0.0
    summary["top3_share"] = summary["top3_count"] / total_runs if total_runs > 0 else 0.0
    summary["same_source_pair"] = summary["vad_mask"] == summary["asr_audio_in"]

    return summary.sort_values(
        by=["mean_rank", f"mean_{primary_metric}"],
        ascending=[True, False],
    ).reset_index(drop=True)


def get_audio_derivative_group_summary(
    df_rq3: pd.DataFrame,
    mode: str,
    dataset_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Summarize RQ3 audio derivative groups.

    Args:
        df_rq3: Experiment- or workspace-level RQ3 DataFrame.
        mode: Evaluation mode.
        dataset_name: Optional dataset filter.

    Returns:
        Sorted RQ3 group summary.
    """
    validate_mode(mode)

    df = df_rq3.copy()

    if "mode" not in df.columns:
        raise KeyError("Expected column 'mode' in RQ3 DataFrame.")

    if "audio_derivative_group" not in df.columns:
        raise KeyError("Expected column 'audio_derivative_group' in RQ3 DataFrame.")

    df = df[df["mode"] == mode].copy()
    df = _filter_dataset(df, dataset_name)

    primary_metric = get_primary_metric_name(mode, "rq3")
    metric_cols = [c for c in [
        primary_metric,
        "best_macro_mean_f1",
        "best_macro_mean_recall",
        "macro_mean_mean_dice_eos_tp",
        "best_macro_mean_mean_dice_eos_tp",
        "macro_mean_n_cand",
        "macro_mean_fp",
    ] if c in df.columns]

    sort_cols = []
    if mode == "full_gt":
        if "best_macro_mean_f1" in df.columns:
            sort_cols.append("best_macro_mean_f1")
        if "macro_mean_f1" in df.columns:
            sort_cols.append("macro_mean_f1")
    else:
        if "best_macro_mean_recall" in df.columns:
            sort_cols.append("best_macro_mean_recall")
        if "macro_mean_recall" in df.columns:
            sort_cols.append("macro_mean_recall")

    keep_cols = [
        c for c in [
            "audio_derivative_group",
            "n_configs",
            "best_combo_key",
            "best_vad_mask",
            "best_asr_audio_in",
            *metric_cols,
        ] if c in df.columns
    ]

    result = df[keep_cols].copy()

    if sort_cols:
        result = result.sort_values(by=sort_cols, ascending=[False] * len(sort_cols))

    return result.reset_index(drop=True)