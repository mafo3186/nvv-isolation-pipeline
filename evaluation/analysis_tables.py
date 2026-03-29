from __future__ import annotations

from typing import Iterable, Optional, Any
import pandas as pd
from evaluation.eval_io import validate_mode


def _require_columns(df: pd.DataFrame, required: list[str], *, label: str) -> None:
    """
    Validate required columns.

    Args:
        df: Input DataFrame.
        required: Required columns.
        label: Error context label.
    """
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in {label}: {missing}")


def _filter_dataset(df: pd.DataFrame, dataset_name: Optional[str]) -> pd.DataFrame:
    """
    Optionally filter a DataFrame by dataset_name.

    Args:
        df: Input DataFrame.
        dataset_name: Dataset filter.

    Returns:
        Filtered DataFrame copy.
    """
    result = df.copy()

    if dataset_name is None:
        return result

    if "dataset_name" not in result.columns:
        raise KeyError("Expected column 'dataset_name' for dataset filtering.")

    return result[result["dataset_name"] == dataset_name].copy()


def _sort_by_metrics(df: pd.DataFrame, metric_cols: list[str]) -> pd.DataFrame:
    """
    Sort descending by metric columns.

    Args:
        df: Input DataFrame.
        metric_cols: Ordered metric columns.

    Returns:
        Sorted DataFrame.
    """
    _require_columns(df, metric_cols, label="metric sort")
    return df.sort_values(
        by=metric_cols,
        ascending=[False] * len(metric_cols),
    ).reset_index(drop=True)


def _value_to_text(value: Any) -> str:
    """
    Convert a scalar value to a stable text label.

    Args:
        value: Input scalar value.

    Returns:
        String representation.
    """
    if pd.isna(value):
        return "None"
    return str(value)


def _sorted_unique_as_text(series: pd.Series) -> str:
    """
    Convert unique series values to a stable, comma-separated text.

    Args:
        series: Input pandas Series.

    Returns:
        Comma-separated string.
    """
    values = list(series.drop_duplicates().tolist())
    values = sorted(values, key=lambda x: (pd.isna(x), str(x)))
    return ", ".join(_value_to_text(v) for v in values)


def _append_setting_columns(
    df: pd.DataFrame,
    *,
    dataset_name: str,
    mode: str,
) -> pd.DataFrame:
    """
    Add standard setting columns.

    Args:
        df: Input DataFrame.
        dataset_name: Dataset identifier.
        mode: Evaluation mode.

    Returns:
        DataFrame with inserted setting columns.
    """
    result = df.copy()
    setting = f"{dataset_name} | {mode}"

    result.insert(0, "setting", setting)
    result.insert(1, "dataset_name", dataset_name)
    result.insert(2, "mode", mode)

    return result


def get_primary_metric_name(mode: str, rq: str = "rq1") -> str:
    """
    Return the primary metric column for a given RQ and mode.

    Args:
        mode: Evaluation mode.
        rq: Research question key.

    Returns:
        Primary metric column name.
    """
    validate_mode(mode)

    if rq in {"rq1", "rq2a"}:
        return "macro_mean_f1" if mode == "full_gt" else "macro_mean_recall"

    if rq == "rq3":
        return "macro_mean_f1" if mode == "full_gt" else "macro_mean_recall"

    raise ValueError(f"Unsupported rq='{rq}'.")


def get_secondary_metric_names(mode: str, rq: str = "rq1") -> list[str]:
    """
    Return ordered secondary metrics for sorting and interpretation.

    Args:
        mode: Evaluation mode.
        rq: Research question key.

    Returns:
        Ordered secondary metric column names.
    """
    validate_mode(mode)

    if rq in {"rq1", "rq2a"}:
        if mode == "full_gt":
            return [
                "macro_mean_recall", 
                "macro_mean_dice_eos_recall", 
                "macro_mean_mean_dice_eos_tp"]
        return [
            "macro_mean_dice_eos_recall", 
            "macro_mean_mean_dice_eos_tp"
        ]

    if rq == "rq3":
        if mode == "full_gt":
            return [
                "macro_mean_recall", 
                "macro_mean_dice_eos_recall", 
                "macro_mean_mean_dice_eos_tp"
            ]
        return [
            "macro_mean_dice_eos_recall", 
            "macro_mean_mean_dice_eos_tp"
        ]

    raise ValueError(f"Unsupported rq='{rq}'.")


def get_top_runs(
    df_rq1: pd.DataFrame,
    *,
    mode: str,
    dataset_name: Optional[str] = None,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Return the best RQ1 rows for one mode and optional dataset.

    Args:
        df_rq1: RQ1 DataFrame.
        mode: Evaluation mode.
        dataset_name: Optional dataset filter.
        top_n: Number of rows to return.

    Returns:
        Top-n sorted DataFrame.
    """
    validate_mode(mode)
    _require_columns(df_rq1, ["mode"], label="RQ1")

    df = df_rq1[df_rq1["mode"] == mode].copy()
    df = _filter_dataset(df, dataset_name)

    metric_cols = [
        get_primary_metric_name(mode, "rq1"),
        *get_secondary_metric_names(mode, "rq1"),
    ]

    return _sort_by_metrics(df, metric_cols).head(top_n).reset_index(drop=True)


def get_top_region_runs(
    df_rq1: pd.DataFrame,
    *,
    mode: str,
    dataset_name: Optional[str] = None,
    score_fraction: float = 0.95,
) -> pd.DataFrame:
    """
    Keep all RQ1 rows within a fraction of the best primary score.

    Args:
        df_rq1: RQ1 DataFrame.
        mode: Evaluation mode.
        dataset_name: Optional dataset filter.
        score_fraction: Fraction of best score to keep.

    Returns:
        Top-region DataFrame with helper score columns.
    """
    validate_mode(mode)
    if not (0 < score_fraction <= 1):
        raise ValueError("score_fraction must be in the range (0, 1].")

    _require_columns(df_rq1, ["mode"], label="RQ1")

    df = df_rq1[df_rq1["mode"] == mode].copy()
    df = _filter_dataset(df, dataset_name)

    primary_metric = get_primary_metric_name(mode, "rq1")
    _require_columns(df, [primary_metric], label="RQ1")

    best_score = df[primary_metric].max()
    cutoff = best_score * score_fraction

    result = df[df[primary_metric] >= cutoff].copy()
    result["primary_metric"] = primary_metric
    result["primary_score"] = result[primary_metric]
    result["best_score"] = best_score
    result["cutoff_score"] = cutoff
    result["score_fraction_of_best"] = (
        result[primary_metric] / best_score if best_score != 0 else 0.0
    )

    metric_cols = [primary_metric, *get_secondary_metric_names(mode, "rq1")]
    return _sort_by_metrics(result, metric_cols)


def get_parameter_value_summary(
    df_top_region: pd.DataFrame,
    *,
    param_name: str,
    primary_metric: str,
) -> pd.DataFrame:
    """
    Summarize one parameter across a top-region subset.

    Args:
        df_top_region: DataFrame returned by get_top_region_runs().
        param_name: Parameter column.
        primary_metric: Primary score column.

    Returns:
        Grouped summary by parameter value.
    """
    _require_columns(
        df_top_region,
        [param_name, primary_metric],
        label="parameter value summary",
    )

    count_col = "run_id" if "run_id" in df_top_region.columns else primary_metric

    summary = (
        df_top_region
        .groupby(param_name, dropna=False)
        .agg(
            n_runs=(count_col, "count"),
            mean_primary_score=(primary_metric, "mean"),
            median_primary_score=(primary_metric, "median"),
            min_primary_score=(primary_metric, "min"),
            max_primary_score=(primary_metric, "max"),
        )
        .reset_index()
    )

    total = len(df_top_region)
    summary["share_of_top_region"] = summary["n_runs"] / total if total > 0 else 0.0
    summary["param_value_text"] = summary[param_name].apply(_value_to_text)

    return summary.sort_values(
        by=["mean_primary_score", "max_primary_score", "n_runs"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def get_parameter_pair_summary(
    df_top_region: pd.DataFrame,
    *,
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
    _require_columns(
        df_top_region,
        [param_x, param_y, primary_metric],
        label="parameter pair summary",
    )

    count_col = "run_id" if "run_id" in df_top_region.columns else primary_metric

    summary = (
        df_top_region
        .groupby([param_x, param_y], dropna=False)
        .agg(
            n_runs=(count_col, "count"),
            mean_primary_score=(primary_metric, "mean"),
            median_primary_score=(primary_metric, "median"),
            best_primary_score=(primary_metric, "max"),
        )
        .reset_index()
    )

    summary["param_x_text"] = summary[param_x].apply(_value_to_text)
    summary["param_y_text"] = summary[param_y].apply(_value_to_text)

    return summary.sort_values(
        by=["mean_primary_score", "best_primary_score", "n_runs"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def get_combo_key_summary(
    df_rq2a: pd.DataFrame,
    *,
    mode: str,
    dataset_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Summarize combo_key performance across runs.

    Args:
        df_rq2a: RQ2a ranking DataFrame.
        mode: Evaluation mode.
        dataset_name: Optional dataset filter.

    Returns:
        Summary DataFrame per combo_key.
    """
    validate_mode(mode)
    _require_columns(
        df_rq2a,
        ["mode", "combo_key", "vad_mask", "asr_audio_in", "rank_within_run"],
        label="RQ2a",
    )

    df = df_rq2a[df_rq2a["mode"] == mode].copy()
    df = _filter_dataset(df, dataset_name)

    primary_metric = get_primary_metric_name(mode, "rq2a")
    secondary_metrics = get_secondary_metric_names(mode, "rq2a")

    agg_spec: dict[str, tuple[str, Any]] = {
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
        agg_spec["best_macro_mean_n_cand"] = ("macro_mean_n_cand", "max")
        agg_spec["worst_macro_mean_n_cand"] = ("macro_mean_n_cand", "min")

    summary = df.groupby("combo_key", dropna=False).agg(**agg_spec).reset_index()

    if "run_id" in df.columns:
        denom = df["run_id"].nunique()
    else:
        denom = len(df)

    summary["top1_share"] = summary["top1_count"] / denom if denom > 0 else 0.0
    summary["top3_share"] = summary["top3_count"] / denom if denom > 0 else 0.0
    summary["same_source_pair"] = summary["vad_mask"] == summary["asr_audio_in"]

    return summary.sort_values(
        by=["mean_rank", f"mean_{primary_metric}"],
        ascending=[True, False],
    ).reset_index(drop=True)


def get_audio_derivative_group_summary(
    df_rq3: pd.DataFrame,
    *,
    mode: str,
    dataset_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Summarize RQ3 derivative groups.

    Args:
        df_rq3: RQ3 DataFrame.
        mode: Evaluation mode.
        dataset_name: Optional dataset filter.

    Returns:
        Sorted derivative-group summary.
    """
    validate_mode(mode)
    _require_columns(df_rq3, ["mode", "audio_derivative_group"], label="RQ3")

    df = df_rq3[df_rq3["mode"] == mode].copy()
    df = _filter_dataset(df, dataset_name)

    keep_cols = [
        col for col in [
            "audio_derivative_group",
            "n_configs",
            "best_combo_key",
            "best_vad_mask",
            "best_asr_audio_in",
            "macro_mean_f1",
            "best_macro_mean_f1",
            "macro_mean_recall",
            "best_macro_mean_recall",
            "macro_mean_mean_dice_eos_tp",
            "best_macro_mean_mean_dice_eos_tp",
            "macro_mean_n_cand",
            "macro_mean_fp",
        ]
        if col in df.columns
    ]

    result = df[keep_cols].copy()

    sort_cols: list[str] = []
    if mode == "full_gt":
        if "best_macro_mean_f1" in result.columns:
            sort_cols.append("best_macro_mean_f1")
        if "macro_mean_f1" in result.columns:
            sort_cols.append("macro_mean_f1")
    else:
        if "best_macro_mean_recall" in result.columns:
            sort_cols.append("best_macro_mean_recall")
        if "macro_mean_recall" in result.columns:
            sort_cols.append("macro_mean_recall")

    if sort_cols:
        result = result.sort_values(
            by=sort_cols,
            ascending=[False] * len(sort_cols),
        )

    return result.reset_index(drop=True)


def _rq1_for_comparison(df_rq1: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the per-run comparison rows from the RQ1 DataFrame.

    The canonical RQ1 DataFrame has a 'system' column with values:
    'baseline', 'best_single', 'best_selected_set'. For per-run analysis
    (top-runs, top-region) only the 'best_selected_set' row is relevant.

    If the DataFrame has no 'system' column (legacy format), it is returned as-is.

    Args:
        df_rq1: RQ1 DataFrame as returned by get_rq_results() or get_results_experiment().

    Returns:
        Per-run comparison DataFrame filtered to best_selected_set rows.
    """
    if "system" not in df_rq1.columns:
        return df_rq1.copy()
    return df_rq1[df_rq1["system"] == "best_selected_set"].copy()


def collect_comparison_views(
    *,
    results: dict[str, pd.DataFrame],
    dataset_names: Iterable[str],
    modes: Iterable[str],
    score_fraction: float = 0.95,
    top_n_runs: int = 10,
    param_names: Optional[list[str]] = None,
    param_pairs: Optional[list[tuple[str, str]]] = None,
    combo_top_n: int = 10,
) -> dict[str, Any]:
    """
    Collect comparison-ready views across dataset/mode settings.

    Args:
        results: Dict returned by get_results_experiment() or get_rq_results().
        dataset_names: Dataset names to compare.
        modes: Modes to compare.
        score_fraction: Top-region fraction of best score.
        top_n_runs: Number of top runs retained per setting.
        param_names: Parameters to summarize.
        param_pairs: Parameter pairs to summarize.
        combo_top_n: Number of combo rows retained per setting.

    Returns:
        Dict with detailed per-setting views and combined comparison tables.
    """
    if param_names is None:
        param_names = [
            "vad_threshold",
            "vad_min_silence_ms",
            "max_duration",
            "dedup_overlap_ratio",
        ]

    if param_pairs is None:
        param_pairs = [
            ("vad_threshold", "vad_min_silence_ms"),
        ]

    per_setting: dict[str, dict[str, Any]] = {}
    overview_rows: list[dict[str, Any]] = []
    param_region_rows: list[dict[str, Any]] = []
    param_value_details: list[pd.DataFrame] = []
    pair_best_rows: list[dict[str, Any]] = []
    pair_value_details: list[pd.DataFrame] = []
    derivative_rows: list[pd.DataFrame] = []
    combo_rows: list[pd.DataFrame] = []

    for dataset_name in dataset_names:
        for mode in modes:
            primary_metric = get_primary_metric_name(mode, "rq1")
            setting = f"{dataset_name} | {mode}"

            top_runs = get_top_runs(
                _rq1_for_comparison(results["rq1"]),
                mode=mode,
                dataset_name=dataset_name,
                top_n=top_n_runs,
            )

            top_region = get_top_region_runs(
                _rq1_for_comparison(results["rq1"]),
                mode=mode,
                dataset_name=dataset_name,
                score_fraction=score_fraction,
            )

            best_score = float(top_region["best_score"].iloc[0]) if len(top_region) else float("nan")
            cutoff_score = float(top_region["cutoff_score"].iloc[0]) if len(top_region) else float("nan")

            rq1_comp = _rq1_for_comparison(results["rq1"])
            total_mask = (
                (rq1_comp["dataset_name"] == dataset_name)
                & (rq1_comp["mode"] == mode)
            ) if "dataset_name" in rq1_comp.columns else (rq1_comp["mode"] == mode)

            overview_rows.append(
                {
                    "setting": setting,
                    "dataset_name": dataset_name,
                    "mode": mode,
                    "primary_metric": primary_metric,
                    "best_score": best_score,
                    "cutoff_score": cutoff_score,
                    "n_top_region_runs": len(top_region),
                    "n_total_runs": int(total_mask.sum()),
                }
            )

            setting_param_summaries: dict[str, pd.DataFrame] = {}
            for param_name in param_names:
                param_summary = get_parameter_value_summary(
                    top_region,
                    param_name=param_name,
                    primary_metric=primary_metric,
                )
                param_summary = _append_setting_columns(
                    param_summary,
                    dataset_name=dataset_name,
                    mode=mode,
                )
                param_summary.insert(3, "parameter", param_name)
                setting_param_summaries[param_name] = param_summary
                param_value_details.append(param_summary)

                best_values = param_summary.loc[
                    param_summary["mean_primary_score"] == param_summary["mean_primary_score"].max(),
                    param_name,
                ] if len(param_summary) else pd.Series(dtype=object)

                param_region_rows.append(
                    {
                        "setting": setting,
                        "dataset_name": dataset_name,
                        "mode": mode,
                        "parameter": param_name,
                        "primary_metric": primary_metric,
                        "values_in_top_region": _sorted_unique_as_text(top_region[param_name]) if len(top_region) else "",
                        "best_values_by_mean": _sorted_unique_as_text(best_values) if len(param_summary) else "",
                        "n_distinct_values": int(top_region[param_name].nunique(dropna=False)) if len(top_region) else 0,
                    }
                )

            setting_pair_summaries: dict[str, pd.DataFrame] = {}
            for param_x, param_y in param_pairs:
                pair_key = f"{param_x}__{param_y}"

                pair_summary = get_parameter_pair_summary(
                    top_region,
                    param_x=param_x,
                    param_y=param_y,
                    primary_metric=primary_metric,
                )
                pair_summary = _append_setting_columns(
                    pair_summary,
                    dataset_name=dataset_name,
                    mode=mode,
                )
                pair_summary.insert(3, "pair_key", pair_key)
                setting_pair_summaries[pair_key] = pair_summary
                pair_value_details.append(pair_summary)

                if len(pair_summary):
                    best_pair = pair_summary.iloc[0]
                    pair_best_rows.append(
                        {
                            "setting": setting,
                            "dataset_name": dataset_name,
                            "mode": mode,
                            "pair_key": pair_key,
                            "primary_metric": primary_metric,
                            param_x: best_pair[param_x],
                            param_y: best_pair[param_y],
                            "best_pair_text": f"{_value_to_text(best_pair[param_x])} | {_value_to_text(best_pair[param_y])}",
                            "mean_primary_score": best_pair["mean_primary_score"],
                            "best_primary_score": best_pair["best_primary_score"],
                            "n_runs": best_pair["n_runs"],
                        }
                    )

            combo_summary = get_combo_key_summary(
                results["rq2a_single"],
                mode=mode,
                dataset_name=dataset_name,
            )
            combo_summary = _append_setting_columns(
                combo_summary,
                dataset_name=dataset_name,
                mode=mode,
            )
            combo_rows.append(combo_summary.head(combo_top_n))

            derivative_summary = get_audio_derivative_group_summary(
                results["rq2b"],
                mode=mode,
                dataset_name=dataset_name,
            )
            derivative_summary = _append_setting_columns(
                derivative_summary,
                dataset_name=dataset_name,
                mode=mode,
            )
            derivative_rows.append(derivative_summary)

            per_setting[setting] = {
                "top_runs": top_runs,
                "top_region": top_region,
                "param_summaries": setting_param_summaries,
                "pair_summaries": setting_pair_summaries,
                "combo_summary": combo_summary,
                "derivative_summary": derivative_summary,
            }

    return {
        "per_setting": per_setting,
        "setting_overview": pd.DataFrame(overview_rows),
        "parameter_region_summary": pd.DataFrame(param_region_rows),
        "parameter_value_details": (
            pd.concat(param_value_details, ignore_index=True)
            if param_value_details else pd.DataFrame()
        ),
        "pair_best_summary": pd.DataFrame(pair_best_rows),
        "pair_value_details": (
            pd.concat(pair_value_details, ignore_index=True)
            if pair_value_details else pd.DataFrame()
        ),
        "derivative_comparison": (
            pd.concat(derivative_rows, ignore_index=True)
            if derivative_rows else pd.DataFrame()
        ),
        "combo_summary_by_setting": (
            pd.concat(combo_rows, ignore_index=True)
            if combo_rows else pd.DataFrame()
        ),
    }

def _concat_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Concatenate DataFrames safely.

    Args:
        frames: List of DataFrames.

    Returns:
        Concatenated DataFrame or empty DataFrame.
    """
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

def combine_comparison_views(
    views_by_experiment: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """
    Combine multiple comparison-view bundles into one shared structure.

    Args:
        views_by_experiment: Mapping experiment_label -> views dict
            returned by collect_comparison_views().

    Returns:
        Combined views dict with the same top-level keys as a single
        collect_comparison_views() output.
    """
    per_setting_combined: dict[str, dict[str, Any]] = {}

    setting_overview_frames: list[pd.DataFrame] = []
    parameter_region_frames: list[pd.DataFrame] = []
    parameter_value_frames: list[pd.DataFrame] = []
    pair_best_frames: list[pd.DataFrame] = []
    pair_value_frames: list[pd.DataFrame] = []
    derivative_frames: list[pd.DataFrame] = []
    combo_frames: list[pd.DataFrame] = []

    for views in views_by_experiment.values():
        if "per_setting" in views:
            per_setting_combined.update(views["per_setting"])

        if "setting_overview" in views:
            setting_overview_frames.append(views["setting_overview"])

        if "parameter_region_summary" in views:
            parameter_region_frames.append(views["parameter_region_summary"])

        if "parameter_value_details" in views:
            parameter_value_frames.append(views["parameter_value_details"])

        if "pair_best_summary" in views:
            pair_best_frames.append(views["pair_best_summary"])

        if "pair_value_details" in views:
            pair_value_frames.append(views["pair_value_details"])

        if "derivative_comparison" in views:
            derivative_frames.append(views["derivative_comparison"])

        if "combo_summary_by_setting" in views:
            combo_frames.append(views["combo_summary_by_setting"])

    return {
        "per_setting": per_setting_combined,
        "setting_overview": _concat_frames(setting_overview_frames),
        "parameter_region_summary": _concat_frames(parameter_region_frames),
        "parameter_value_details": _concat_frames(parameter_value_frames),
        "pair_best_summary": _concat_frames(pair_best_frames),
        "pair_value_details": _concat_frames(pair_value_frames),
        "derivative_comparison": _concat_frames(derivative_frames),
        "combo_summary_by_setting": _concat_frames(combo_frames),
    }


def build_top_region_parameter_values_matrix(
    parameter_region_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Pivot compact parameter-region comparison into a matrix.

    Args:
        parameter_region_summary: Output from collect_comparison_views().

    Returns:
        Pivoted matrix DataFrame.
    """
    _require_columns(
        parameter_region_summary,
        ["parameter", "setting", "values_in_top_region"],
        label="parameter region summary",
    )

    return (
        parameter_region_summary
        .pivot_table(
            index="parameter",
            columns="setting",
            values="values_in_top_region",
            aggfunc="first",
        )
        .reset_index()
    )


def build_derivative_matrix(
    derivative_comparison: pd.DataFrame,
    *,
    value_col: str,
) -> pd.DataFrame:
    """
    Pivot derivative comparison into a matrix.

    Args:
        derivative_comparison: Output from collect_comparison_views().
        value_col: Column to pivot, e.g. 'macro_mean_recall'.

    Returns:
        Pivoted derivative matrix.
    """
    _require_columns(
        derivative_comparison,
        ["audio_derivative_group", "setting", value_col],
        label="derivative comparison",
    )

    return (
        derivative_comparison
        .pivot_table(
            index="audio_derivative_group",
            columns="setting",
            values=value_col,
            aggfunc="first",
        )
        .reset_index()
    )

def build_setting_summary_table(
    setting_overview: pd.DataFrame,
    *,
    score_fraction: float,
) -> dict[str, pd.DataFrame]:
    """
    Build thesis-friendly setting summary tables split by mode,
    with dynamic cutoff column naming based on score_fraction.

    Args:
        setting_overview: Table from collect_comparison_views().
        score_fraction: Fraction used to define top-region cutoff (e.g. 0.95).

    Returns:
        {
            "full_gt": DataFrame,
            "part_gt": DataFrame,
        }
    """
    _require_columns(
        setting_overview,
        [
            "setting",
            "dataset_name",
            "mode",
            "best_score",
            "cutoff_score",
            "n_top_region_runs",
        ],
        label="setting overview",
    )

    # --- build cutoff label ---
    cutoff_pct = score_fraction * 100
    if float(cutoff_pct).is_integer():
        cutoff_pct_str = f"{int(cutoff_pct)}pct"
    else:
        cutoff_pct_str = f"{cutoff_pct:g}pct"

    rows_by_mode: dict[str, list[dict[str, Any]]] = {
        "full_gt": [],
        "part_gt": [],
    }

    for _, row in setting_overview.iterrows():
        mode = row["mode"]

        base_row = {
            "setting": row["setting"],
            "dataset_name": row["dataset_name"],
            "n_top_region_runs": row["n_top_region_runs"],
        }

        if mode == "full_gt":
            base_row["best_f1"] = row["best_score"]
            base_row[f"f1_cutoff_{cutoff_pct_str}"] = row["cutoff_score"]

        elif mode == "part_gt":
            base_row["best_recall"] = row["best_score"]
            base_row[f"recall_cutoff_{cutoff_pct_str}"] = row["cutoff_score"]

        rows_by_mode[mode].append(base_row)

    return {
        mode: pd.DataFrame(rows)
        for mode, rows in rows_by_mode.items()
    }


def build_best_all_screened_values_matrix(
    results_rq1: pd.DataFrame,
    *,
    dataset_names: list[str],
    modes: list[str],
) -> dict[str, pd.DataFrame]:
    """
    Build best configuration tables split by mode.

    Returns:
        {
            "full_gt": DataFrame,
            "part_gt": DataFrame,
        }
    """
    param_cols = [
        "vad_threshold",
        "vad_min_silence_ms",
        "max_duration",
        "dedup_overlap_ratio",
    ]

    rows_by_mode: dict[str, list[dict[str, Any]]] = {
        "full_gt": [],
        "part_gt": [],
    }

    for dataset_name in dataset_names:
        for mode in modes:
            top_run = get_top_runs(
                results_rq1,
                mode=mode,
                dataset_name=dataset_name,
                top_n=1,
            )

            if top_run.empty:
                continue

            row = top_run.iloc[0].to_dict()

            base_row = {
                "setting": f"{dataset_name} | {mode}",
                "dataset_name": dataset_name,
                "best_k": row.get("best_k"),
                "selected_set_json": row.get("selected_set_json"),
            }

            if mode == "full_gt":
                base_row["f1"] = row.get("macro_mean_f1")
                base_row["recall"] = row.get("macro_mean_recall")
                base_row["eos_tp"] = row.get("macro_mean_mean_dice_eos_tp")

            elif mode == "part_gt":
                base_row["recall"] = row.get("macro_mean_recall")
                base_row["eos_tp"] = row.get("macro_mean_mean_dice_eos_tp")
                base_row["eos_recall"] = row.get("macro_mean_dice_eos_recall")

            for col in param_cols:
                if col in row:
                    base_row[col] = row.get(col)

            if "run_id" in row:
                base_row["run_id"] = row.get("run_id")

            rows_by_mode[mode].append(base_row)

    return {
        mode: pd.DataFrame(rows)
        for mode, rows in rows_by_mode.items()
    }


def build_top_region_parameter_values_matrix(
    parameter_region_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a thesis-friendly matrix of parameter values that appear in the
    top region for each setting.

    Args:
        parameter_region_summary: Output from collect_comparison_views().

    Returns:
        Pivoted parameter-value matrix.
    """
    _require_columns(
        parameter_region_summary,
        ["parameter", "setting", "values_in_top_region"],
        label="parameter region summary",
    )

    return (
        parameter_region_summary
        .pivot_table(
            index="parameter",
            columns="setting",
            values="values_in_top_region",
            aggfunc="first",
        )
        .reset_index()
    )

def build_top_k_runs_table(
    per_setting: dict[str, dict[str, pd.DataFrame]],
    *,
    top_k: int = 10,
) -> pd.DataFrame:
    """
    Build a compact top-k runs table across all settings.

    Args:
        per_setting: Mapping returned in views["per_setting"].
        top_k: Number of top-ranked runs per setting.

    Returns:
        Combined table with one row per ranked run and setting.
    """
    rows: list[dict[str, Any]] = []

    for setting_name, setting_dict in per_setting.items():
        if "top_runs" not in setting_dict:
            continue

        top_runs = setting_dict["top_runs"].copy()
        if top_runs.empty:
            continue

        top_runs = top_runs.head(top_k).reset_index(drop=True)

        if "mode" not in top_runs.columns:
            raise KeyError(f"Missing 'mode' column in top_runs for setting '{setting_name}'.")

        mode = str(top_runs["mode"].iloc[0])
        primary_metric_name = get_primary_metric_name(mode, "rq1")

        secondary_metrics = get_secondary_metric_names(mode, "rq1")
        secondary_metric_name = secondary_metrics[0] if len(secondary_metrics) > 0 else None
        tertiary_metric_name = secondary_metrics[1] if len(secondary_metrics) > 1 else None

        for idx, row in top_runs.iterrows():
            rows.append(
                {
                    "setting": setting_name,
                    "rank": idx + 1,
                    "primary_metric_name": primary_metric_name,
                    "primary_score": row.get(primary_metric_name),
                    "secondary_metric_name": secondary_metric_name,
                    "secondary_score": row.get(secondary_metric_name) if secondary_metric_name else None,
                    "tertiary_metric_name": tertiary_metric_name,
                    "tertiary_score": row.get(tertiary_metric_name) if tertiary_metric_name else None,
                    "vad_threshold": row.get("vad_threshold"),
                    "vad_min_silence_ms": row.get("vad_min_silence_ms"),
                    "max_duration": row.get("max_duration"),
                    "dedup_overlap_ratio": row.get("dedup_overlap_ratio"),
                    "run_id": row.get("run_id"),
                }
            )

    return pd.DataFrame(rows)

def _format_setting_label(dataset_name: str, mode: str) -> str:
    """
    Build a compact setting label for thesis tables.

    Args:
        dataset_name: Dataset identifier.
        mode: Evaluation mode.

    Returns:
        Compact setting label.
    """
    return f"{dataset_name} | {mode}"


def _safe_metric_delta(best_value: Any, baseline_value: Any) -> Any:
    """
    Compute delta if both values are numeric, else return None.

    Args:
        best_value: Best-selected-set value.
        baseline_value: Baseline value.

    Returns:
        Numeric delta or None.
    """
    if pd.isna(best_value) or pd.isna(baseline_value):
        return None
    return best_value - baseline_value


def _build_rq1_table_for_mode(df_mode: pd.DataFrame, *, mode: str) -> pd.DataFrame:
    """
    Build one thesis-ready RQ1 capability table for a single mode.

    Expected input rows per setting:
    - baseline
    - best_single
    - best_selected_set

    Final output rows per setting:
    - Baseline
    - Best Selected Set
    - Δ vs Baseline

    Args:
        df_mode: RQ1 DataFrame filtered to one mode.
        mode: Evaluation mode.

    Returns:
        Thesis-ready RQ1 table for one mode.
    """
    if df_mode.empty:
        return pd.DataFrame()

    required = ["dataset_name", "mode", "system"]
    _require_columns(df_mode, required, label="RQ1 capability input")

    if mode == "full_gt":
        metric_map = {
            "macro_mean_f1": "F1",
            "macro_mean_recall": "Recall",
            "macro_mean_dice_eos_recall": "EOS Recall", #toDo: insert!
            "macro_mean_mean_dice_eos_tp": "Mean EOS TP",
        }
    else:
        metric_map = {
            "macro_mean_recall": "Recall",
            "macro_mean_dice_eos_recall": "EOS Recall", #toDo: insert
            "macro_mean_mean_dice_eos_tp": "Mean EOS TP",
            #"insertion_rate": "Insertion Rate", #toDo: insert in csv
        }

    metric_cols = [col for col in metric_map if col in df_mode.columns]

    rows: list[dict[str, Any]] = []

    for dataset_name in df_mode["dataset_name"].drop_duplicates().tolist():
        df_setting = df_mode[df_mode["dataset_name"] == dataset_name].copy()

        baseline_df = df_setting[df_setting["system"] == "baseline"]
        best_df = df_setting[df_setting["system"] == "best_selected_set"]

        if baseline_df.empty or best_df.empty:
            continue

        baseline_row = baseline_df.iloc[0]
        best_row = best_df.iloc[0]

        setting_label = _format_setting_label(dataset_name, mode)

        baseline_entry = {
            "Setting": setting_label,
            "System": "Baseline",
        }
        for col in metric_cols:
            baseline_entry[metric_map[col]] = baseline_row.get(col)

        best_entry = {
            "Setting": setting_label,
            "System": "Best Selected Set",
        }
        for col in metric_cols:
            best_entry[metric_map[col]] = best_row.get(col)

        delta_entry = {
            "Setting": setting_label,
            "System": "Δ vs Baseline",
        }
        for col in metric_cols:
            delta_entry[metric_map[col]] = _safe_metric_delta(
                best_row.get(col),
                baseline_row.get(col),
            )

        rows.extend([baseline_entry, best_entry, delta_entry])

    return pd.DataFrame(rows)


def build_rq1_capability_tables(results_rq1: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Build thesis-ready RQ1 capability tables split by mode.

    This function intentionally keeps only the core RQ1 comparison:
    - Baseline
    - Best Selected Set
    - Delta vs Baseline

    It ignores auxiliary columns such as precision/fp that may exist in the
    exported artifact but are not part of the thesis RQ1 table.

    Args:
        results_rq1: Concatenated canonical RQ1 DataFrame across settings.

    Returns:
        {
            "full_gt": DataFrame,
            "part_gt": DataFrame,
        }
    """
    _require_columns(results_rq1, ["mode", "dataset_name", "system"], label="RQ1")

    out: dict[str, pd.DataFrame] = {}

    for mode in ["full_gt", "part_gt"]:
        df_mode = results_rq1[results_rq1["mode"] == mode].copy()
        out[mode] = _build_rq1_table_for_mode(df_mode, mode=mode)

    return out

def build_rq2a_single_ranking_tables(
    ranking_single: pd.DataFrame,
    *,
    top_k: int = 5,
) -> dict[str, pd.DataFrame]:
    """
    Build thesis-ready RQ2a single-ranking tables, one table per setting.

    Args:
        ranking_single: Concatenated RQ2a single-ranking DataFrame across settings.
        top_k: Number of top-ranked rows to keep per setting.

    Returns:
        Dict mapping setting -> DataFrame.
    """
    _require_columns(
        ranking_single,
        [
            "setting",
            "mode",
            "vad_mask",
            "asr_audio_in",
            "rank_within_run",
        ],
        label="RQ2a single ranking",
    )

    results: dict[str, pd.DataFrame] = {}

    for setting in ranking_single["setting"].drop_duplicates().tolist():
        df_setting = ranking_single[ranking_single["setting"] == setting].copy()
        if df_setting.empty:
            continue

        mode = str(df_setting["mode"].iloc[0])

        base_cols = {
            "rank_within_run": "Rank",
            "vad_mask": "VAD Mask",
            "asr_audio_in": "ASR Audio Input",
        }

        if mode == "full_gt":
            metric_cols = {
                "macro_mean_f1": "F1",
                "macro_mean_recall": "Recall",
                "macro_mean_dice_eos_recall": "EOS Recall",
                "macro_mean_mean_dice_eos_tp": "Mean EOS TP",
            }
            sort_cols = [
                "rank_within_run",
                "macro_mean_f1",
                "macro_mean_recall",
                "macro_mean_dice_eos_recall",
                "macro_mean_mean_dice_eos_tp",
            ]
            ascending = [True, False, False, False, False]

        elif mode == "part_gt":
            metric_cols = {
                "macro_mean_recall": "Recall",
                "macro_mean_dice_eos_recall": "EOS Recall",
                "macro_mean_mean_dice_eos_tp": "Mean EOS TP",
            }
            sort_cols = [
                "rank_within_run",
                "macro_mean_recall",
                "macro_mean_dice_eos_recall",
                "macro_mean_mean_dice_eos_tp",
            ]
            ascending = [True, False, False, False]

        else:
            continue

        keep_cols = list(base_cols.keys()) + [c for c in metric_cols if c in df_setting.columns]
        df_out = df_setting[keep_cols].copy()
        df_out = df_out.rename(columns={**base_cols, **metric_cols})

        df_out = df_setting.sort_values(
            by=[c for c in sort_cols if c in df_setting.columns],
            ascending=ascending[:len([c for c in sort_cols if c in df_setting.columns])],
        ).reset_index(drop=True)

        keep_cols = list(base_cols.keys()) + [c for c in metric_cols if c in df_out.columns]
        df_out = df_out[keep_cols].copy()
        df_out = df_out.rename(columns={**base_cols, **metric_cols})

        if top_k is not None:
            df_out = df_out.head(top_k).reset_index(drop=True)

        results[setting] = df_out

    return results

def build_rq2a_selected_set_tables(
    ranking_single: pd.DataFrame,
    ranking_selected_set: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Build thesis-ready RQ2a selected-set comparison tables, one table per setting.

    For each setting, compares:
    - Best Single
    - Best Selected Set
    - Delta vs Best Single

    Args:
        ranking_single: Concatenated RQ2a single-ranking DataFrame across settings.
        ranking_selected_set: Concatenated RQ2a selected-set DataFrame across settings.

    Returns:
        Dict mapping setting -> DataFrame.
    """
    _require_columns(
        ranking_single,
        [
            "setting",
            "mode",
            "rank_within_run",
            "vad_mask",
            "asr_audio_in",
        ],
        label="RQ2a single ranking",
    )
    _require_columns(
        ranking_selected_set,
        [
            "setting",
            "mode",
            "best_k",
            "selected_set_json",
        ],
        label="RQ2a selected set",
    )

    results: dict[str, pd.DataFrame] = {}

    common_settings = [
        s for s in ranking_selected_set["setting"].drop_duplicates().tolist()
        if s in ranking_single["setting"].drop_duplicates().tolist()
    ]

    for setting in common_settings:
        df_single = ranking_single[ranking_single["setting"] == setting].copy()
        df_set = ranking_selected_set[ranking_selected_set["setting"] == setting].copy()

        if df_single.empty or df_set.empty:
            continue

        mode = str(df_set["mode"].iloc[0])

        best_single = df_single.sort_values("rank_within_run", ascending=True).iloc[0]
        best_set = df_set.iloc[0]

        best_single_config = f"{best_single.get('vad_mask', '')} + {best_single.get('asr_audio_in', '')}"

        rows: list[dict[str, Any]] = []

        if mode == "full_gt":
            metric_pairs = [
                ("macro_mean_f1", "F1"),
                ("macro_mean_recall", "Recall"),
                ("macro_mean_dice_eos_recall", "EOS Recall"),
                ("macro_mean_mean_dice_eos_tp", "Mean EOS TP"),
            ]
        elif mode == "part_gt":
            metric_pairs = [
                ("macro_mean_recall", "Recall"),
                ("macro_mean_dice_eos_recall", "EOS Recall"),
                ("macro_mean_mean_dice_eos_tp", "Mean EOS TP"),
            ]
        else:
            continue

        row_single = {
            "Setting": setting,
            "System": "Best Single",
            "k": 1,
            "Config / Selected Set": best_single_config,
        }
        for src, dst in metric_pairs:
            row_single[dst] = best_single.get(src)
        rows.append(row_single)

        row_set = {
            "Setting": setting,
            "System": "Best Selected Set",
            "k": best_set.get("best_k"),
            "Config / Selected Set": best_set.get("selected_set_json"),
        }
        for src, dst in metric_pairs:
            row_set[dst] = best_set.get(src)
        rows.append(row_set)

        row_delta = {
            "Setting": setting,
            "System": "Δ vs Best Single",
            "k": None,
            "Config / Selected Set": None,
        }
        for src, dst in metric_pairs:
            a = best_set.get(src)
            b = best_single.get(src)
            
        if pd.isna(a) or pd.isna(b):
            row_delta[dst] = None
        else:
            delta = a - b
            row_delta[dst] = 0.0 if abs(delta) < 1e-12 else delta

        rows.append(row_delta)

        results[setting] = pd.DataFrame(rows)

    return results

def build_rq2b_derivative_tables(derivative_comparison: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Build thesis-ready RQ2b audio-derivative tables split by mode.

    The reported metric values are group-level means.
    The 'Best Config' column is included only as contextual information about
    the strongest configuration within each derivative group.

    Args:
        derivative_comparison: Combined derivative comparison table.

    Returns:
        {
            "full_gt": DataFrame,
            "part_gt": DataFrame,
        }
    """
    _require_columns(
        derivative_comparison,
        [
            "setting",
            "dataset_name",
            "mode",
            "audio_derivative_group",
            "best_vad_mask",
            "best_asr_audio_in",
        ],
        label="RQ2b derivative comparison",
    )

    rows_by_mode: dict[str, list[dict[str, Any]]] = {
        "full_gt": [],
        "part_gt": [],
    }

    for _, row in derivative_comparison.iterrows():
        mode = row["mode"]

        best_config = f"{row.get('best_vad_mask', '')} + {row.get('best_asr_audio_in', '')}"

        base_row = {
            "Setting": row["setting"],
            "Audio Derivative": row["audio_derivative_group"],
        }

        if mode == "full_gt":
            base_row["F1"] = row.get("macro_mean_f1")
            base_row["Recall"] = row.get("macro_mean_recall")
            #base_row["EOS Recall"] = row.get("macro_mean_dice_eos_recall") # toDo: insert in csv
            base_row["Mean EOS TP"] = row.get("macro_mean_mean_dice_eos_tp")

        elif mode == "part_gt":
            base_row["Recall"] = row.get("macro_mean_recall")
            #base_row["EOS Recall"] = row.get("macro_mean_dice_eos_recall") #toDo: insert in csv
            base_row["Mean EOS TP"] = row.get("macro_mean_mean_dice_eos_tp")
            #base_row["Insertion Rate"] = row.get("insertion_rate") #toDo: insert in csv

        else:
            continue

        # 👉 Best Config ans Ende
        base_row["Best Config (VAD Mask + ASR Audio Input)"] = best_config

        rows_by_mode[mode].append(base_row)

    out = {
        "full_gt": pd.DataFrame(rows_by_mode["full_gt"]),
        "part_gt": pd.DataFrame(rows_by_mode["part_gt"]),
    }

    setting_order = [
        "nvs38k_EN | full_gt",
        "nvs38k_EN | part_gt",
        "VOCAL_RA1 | part_gt",
        "VOCAL_RA2 | part_gt",
    ]

    derivative_order = [
        "original_like",
        "vocals_like",
        "background_like",
        "all_derivatives",
    ]

    if not out["full_gt"].empty:
        out["full_gt"]["Setting"] = pd.Categorical(
            out["full_gt"]["Setting"],
            categories=setting_order,
            ordered=True,
        )
        out["full_gt"]["Audio Derivative"] = pd.Categorical(
            out["full_gt"]["Audio Derivative"],
            categories=derivative_order,
            ordered=True,
        )
        out["full_gt"] = out["full_gt"].sort_values(
            by=["Setting", "Audio Derivative"],
            ascending=[True, True],
        ).reset_index(drop=True)

    if not out["part_gt"].empty:
        out["part_gt"]["Setting"] = pd.Categorical(
            out["part_gt"]["Setting"],
            categories=setting_order,
            ordered=True,
        )
        out["part_gt"]["Audio Derivative"] = pd.Categorical(
            out["part_gt"]["Audio Derivative"],
            categories=derivative_order,
            ordered=True,
        )
        out["part_gt"] = out["part_gt"].sort_values(
            by=["Setting", "Audio Derivative"],
            ascending=[True, True],
        ).reset_index(drop=True)

    return out


def build_rq3_full_gt_tables(df_rq3: pd.DataFrame) -> dict[str, pd.DataFrame]:
    _require_columns(df_rq3, ["mode", "dataset_name", "label"], label="RQ3 full_gt")

    df_full = df_rq3[df_rq3["mode"] == "full_gt"].copy()
    if df_full.empty:
        return {}

    results: dict[str, pd.DataFrame] = {}

    for dataset_name in df_full["dataset_name"].drop_duplicates().tolist():
        df_ds = df_full[df_full["dataset_name"] == dataset_name].copy()
        setting = f"{dataset_name} | full_gt"
        df_ds["Setting"] = setting

        # Split normal label rows and special FP row
        fp_mask = df_ds["label"].astype(str) == "__FP__"
        df_fp = df_ds[fp_mask].copy()
        df_labels = df_ds[~fp_mask].copy()

        rename_map = {
            "label": "Label",
            "n_gt_events": "n_gt",
            "tp": "n_tp",
            "fn": "n_fn",
            #"f1": "F1", #toDo: insert in csv
            "recall": "Recall",
            "mean_dice_eos": "EOS",
            #"mean_dice_eos_tp": "Mean EOS TP", #toDo: insert in csv
            "mean_overlap_s": "Mean Overlap (s)",
        }
        df_labels = df_labels.rename(columns=rename_map)

        ordered_cols = [
            "Setting",
            "Label",
            "n_gt",
            "n_tp",
            "n_fn",
            "Recall",
            "EOS",
            "Mean Overlap (s)",
        ]
        ordered_cols = [c for c in ordered_cols if c in df_labels.columns]
        df_labels = df_labels[ordered_cols].copy()

        df_labels = df_labels.sort_values(
            by=["Recall", "EOS"],
            ascending=[False, False],
        ).reset_index(drop=True)

        # ---- add FP as Insertion as new row/column ----
        if not df_fp.empty and "fp" in df_fp.columns:
            fp_value = df_fp["fp"].iloc[0]

            fp_row = {
                "Setting": setting,
                "Label": "__FP__",
                "n_gt": None,
                "n_tp": None,
                "n_fn": None,
                "Recall": None,
                "EOS": None,
                "Mean Overlap (s)": None,
                "Insertions": fp_value,
            }

            # FP-Spalte nur für diese Zeile hinzufügen
            df_labels["Insertions"] = None
            df_fp_row = pd.DataFrame([fp_row])

            df_out = pd.concat([df_labels, df_fp_row], ignore_index=True)
        else:
            df_out = df_labels.copy()

        results[setting] = df_out.reset_index(drop=True)

    return results


def build_rq3_part_gt_tables(df_rq3: pd.DataFrame) -> dict[str, pd.DataFrame]:
    #toDo: normalize column names for gt/cand label across csvs to avoid this complexity in the table-building code
    """
    Build thesis-ready RQ3 part_gt event-level tables, one table per dataset/setting.

    Expected input:
        Event-level RQ3 rows for mode == "part_gt".

    Returns:
        Dict mapping setting -> DataFrame.
    """
    _require_columns(df_rq3, ["mode", "dataset_name"], label="RQ3 part_gt")

    df_part = df_rq3[df_rq3["mode"] == "part_gt"].copy()
    if df_part.empty:
        return {}

    results: dict[str, pd.DataFrame] = {}

    for dataset_name in df_part["dataset_name"].drop_duplicates().tolist():
        df_ds = df_part[df_part["dataset_name"] == dataset_name].copy()
        setting = f"{dataset_name} | part_gt"
        df_ds["Setting"] = setting

        gt_label_col = None
        cand_label_col = None

        for candidate in ["gt_label", "label_gt"]:
            if candidate in df_ds.columns:
                gt_label_col = candidate
                break

        for candidate in ["cand_label", "label_cand"]:
            if candidate in df_ds.columns:
                cand_label_col = candidate
                break

        rename_map = {
            "audio_id": "Audio ID",
            "gt_event_id": "GT Event ID",
            "cand_event_id": "Cand Event ID",
            "status": "Event Type",
            "dice_eos": "EOS",
            "overlap_s": "Overlap (s)",
        }

        if gt_label_col is not None:
            rename_map[gt_label_col] = "Label GT"
        if cand_label_col is not None:
            rename_map[cand_label_col] = "Label Cand"

        df_ds = df_ds.rename(columns=rename_map)

        if "Event Type" in df_ds.columns:
            df_ds["Event Type"] = df_ds["Event Type"].replace(
                {
                    "hit": "TP",
                    "miss": "FN",
                    "insertion": "Insertion",
                }
            )

        keep_cols = [
            "Setting",
            "Label GT",
            "Label Cand",
            "Event Type",
            "EOS",
            "Overlap (s)",
            "Audio ID",
            "GT Event ID",
            "Cand Event ID",
        ]
        keep_cols = [c for c in keep_cols if c in df_ds.columns]
        df_ds = df_ds[keep_cols].copy()

        # Sort primarily by event type (TP, FN, Insertion), then by EOS descending.
        # Helper column stays internal and is dropped before output.
        if "Event Type" in df_ds.columns:
            event_order_map = {
                "TP": 0,
                "FN": 1,
                "Insertion": 2,
            }
            df_ds["_event_order"] = df_ds["Event Type"].map(event_order_map).fillna(99)

            sort_cols = ["_event_order"]
            ascending = [True]

            if "EOS" in df_ds.columns:
                sort_cols.append("EOS")
                ascending.append(False)

            if "Audio ID" in df_ds.columns:
                sort_cols.append("Audio ID")
                ascending.append(True)

            if "GT Event ID" in df_ds.columns:
                sort_cols.append("GT Event ID")
                ascending.append(True)

            if "Cand Event ID" in df_ds.columns:
                sort_cols.append("Cand Event ID")
                ascending.append(True)

            df_ds = df_ds.sort_values(by=sort_cols, ascending=ascending).reset_index(drop=True)
            df_ds = df_ds.drop(columns="_event_order")

        results[setting] = df_ds.reset_index(drop=True)

    return results