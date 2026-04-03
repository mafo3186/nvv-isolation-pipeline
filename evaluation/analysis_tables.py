from __future__ import annotations

from typing import Iterable, Optional, Any
import pandas as pd
from evaluation.analysis_metrics import (
    get_ordered_metric_names,
    get_metric_sort_ascending,
    get_metric_label,
)
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
    Sort by canonical metric columns using canonical sort directions.

    Args:
        df: Input DataFrame.
        metric_cols: Ordered metric columns.

    Returns:
        Sorted DataFrame.
    """
    _require_columns(df, metric_cols, label="metric sort")

    ascending = [get_metric_sort_ascending(col) for col in metric_cols]

    return df.sort_values(
        by=metric_cols,
        ascending=ascending,
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

    metric_cols = get_ordered_metric_names(mode)

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

    metric_cols = get_ordered_metric_names(mode)
    primary_metric = metric_cols[0]
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

    ordered_metrics = get_ordered_metric_names(mode)
    sort_metric = ordered_metrics[0]

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
    }

    for metric in ordered_metrics:
        if metric in df.columns:
            agg_spec[f"mean_{metric}"] = (metric, "mean")
            if get_metric_sort_ascending(metric):
                agg_spec[f"best_{metric}"] = (metric, "min")
                agg_spec[f"worst_{metric}"] = (metric, "max")
            else:
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
        by=["mean_rank", f"mean_{sort_metric}"],
        ascending=[True, get_metric_sort_ascending(sort_metric)],
    ).reset_index(drop=True)


def get_audio_derivative_group_summary(
    df_rq2b: pd.DataFrame,
    *,
    mode: str,
    dataset_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Summarize RQ2b derivative groups.

    Args:
        df_rq2b: RQ2b DataFrame.
        mode: Evaluation mode.
        dataset_name: Optional dataset filter.

    Returns:
        Sorted derivative-group summary.
    """
    validate_mode(mode)
    _require_columns(df_rq2b, ["mode", "audio_derivative_group"], label="RQ2b")

    df = df_rq2b[df_rq2b["mode"] == mode].copy()
    df = _filter_dataset(df, dataset_name)

    keep_cols = [
        "audio_derivative_group",
        "n_configs",
        "best_combo_key",
        "best_vad_mask",
        "best_asr_audio_in",
    ]

    ordered_metrics = get_ordered_metric_names(mode)

    for metric in ordered_metrics:
        if metric in df.columns:
            keep_cols.append(metric)

        best_metric = f"best_{metric}"
        if best_metric in df.columns:
            keep_cols.append(best_metric)

    for extra_col in ["macro_mean_n_cand", "macro_mean_fp"]:
        if extra_col in df.columns:
            keep_cols.append(extra_col)

    result = df[keep_cols].copy()

    sort_cols: list[str] = []
    ascending: list[bool] = []

    for metric in ordered_metrics:
        best_metric = f"best_{metric}"
        if best_metric in result.columns:
            sort_cols.append(best_metric)
            ascending.append(get_metric_sort_ascending(metric))

    if sort_cols:
        result = result.sort_values(
            by=sort_cols,
            ascending=ascending,
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
            metric_cols = get_ordered_metric_names(mode)
            primary_metric = metric_cols[0]
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
            ordered_metrics = get_ordered_metric_names(mode)

            base_row = {
                "setting": f"{dataset_name} | {mode}",
                "dataset_name": dataset_name,
                "best_k": row.get("best_k"),
                "selected_set_json": row.get("selected_set_json"),
            }

            for col in param_cols:
                if col in row:
                    base_row[col] = row.get(col)

            for metric_name in ordered_metrics:
                if metric_name in row:
                    base_row[metric_name] = row.get(metric_name)

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

    param_cols = [
        "vad_threshold",
        "vad_min_silence_ms",
        "max_duration",
        "dedup_overlap_ratio",
    ]

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
        ordered_metrics = get_ordered_metric_names(mode)

        dataset_name = (
            str(top_runs["dataset_name"].iloc[0])
            if "dataset_name" in top_runs.columns
            else setting_name.split(" | ")[0]
        )

        for idx, row in top_runs.iterrows():
            row_dict: dict[str, Any] = {
                "rank": idx + 1,
                "setting": setting_name,
                "dataset_name": dataset_name,
                "mode": mode,
            }

            for col in param_cols:
                row_dict[col] = row.get(col)

            for metric_name in ordered_metrics:
                if metric_name in row.index:
                    row_dict[metric_name] = row.get(metric_name)

            row_dict["run_id"] = row.get("run_id")
            rows.append(row_dict)

    result = pd.DataFrame(rows)

    if result.empty:
        return result

    ordered_cols = [
        "rank",
        "setting",
        "dataset_name",
        "mode",
        *param_cols,
        *[
            metric_name
            for mode_name in ["full_gt", "part_gt"]
            for metric_name in get_ordered_metric_names(mode_name)
            if metric_name in result.columns
        ],
        "run_id",
    ]

    ordered_cols = [
        col
        for i, col in enumerate(ordered_cols)
        if col in result.columns and col not in ordered_cols[:i]
    ]

    return result[ordered_cols]


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

    ordered_metrics = get_ordered_metric_names(mode)
    metric_pairs = [(metric, get_metric_label(metric)) for metric in ordered_metrics if metric in df_mode.columns]

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
        for src, dst in metric_pairs:
            baseline_entry[dst] = baseline_row.get(src)

        best_entry = {
            "Setting": setting_label,
            "System": "Best Selected Set",
        }
        for src, dst in metric_pairs:
            best_entry[dst] = best_row.get(src)

        delta_entry = {
            "Setting": setting_label,
            "System": "Δ vs Baseline",
        }
        for src, dst in metric_pairs:
            delta_entry[dst] = _safe_metric_delta(
                best_row.get(src),
                baseline_row.get(src),
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
        ordered_metrics = get_ordered_metric_names(mode)

        base_cols = {
            "rank_within_run": "Rank",
            "vad_mask": "VAD Mask",
            "asr_audio_in": "ASR Audio Input",
        }
        metric_cols = {
            metric_name: get_metric_label(metric_name)
            for metric_name in ordered_metrics
            if metric_name in df_setting.columns
        }

        sort_cols = ["rank_within_run"]
        ascending = [True]

        for metric_name in ordered_metrics:
            if metric_name in df_setting.columns:
                sort_cols.append(metric_name)
                ascending.append(get_metric_sort_ascending(metric_name))

        df_out = df_setting.sort_values(
            by=sort_cols,
            ascending=ascending,
        ).reset_index(drop=True)

        keep_cols = list(base_cols.keys()) + list(metric_cols.keys())
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
        ordered_metrics = get_ordered_metric_names(mode)

        sort_cols = ["rank_within_run"]
        ascending = [True]

        for metric_name in ordered_metrics:
            if metric_name in df_single.columns:
                sort_cols.append(metric_name)
                ascending.append(get_metric_sort_ascending(metric_name))

        best_single = df_single.sort_values(
            by=sort_cols,
            ascending=ascending,
        ).reset_index(drop=True).iloc[0]

        best_set = df_set.iloc[0]
        best_single_config = f"{best_single.get('vad_mask', '')} + {best_single.get('asr_audio_in', '')}"

        metric_pairs = [
            (metric_name, get_metric_label(metric_name))
            for metric_name in ordered_metrics
            if metric_name in df_single.columns and metric_name in df_set.columns
        ]

        best_single_k = 1
        best_set_k = int(best_set.get("best_k")) if pd.notna(best_set.get("best_k")) else None
        delta_k = (best_set_k - best_single_k) if best_set_k is not None else None

        rows: list[dict[str, Any]] = []

        row_single = {
            "Setting": setting,
            "System": "Best Single",
            "k": best_single_k,
            "Config / Selected Set": best_single_config,
        }
        for src, dst in metric_pairs:
            row_single[dst] = best_single.get(src)
        rows.append(row_single)

        row_set = {
            "Setting": setting,
            "System": "Best Selected Set",
            "k": best_set_k,
            "Config / Selected Set": best_set.get("selected_set_json"),
        }
        for src, dst in metric_pairs:
            row_set[dst] = best_set.get(src)
        rows.append(row_set)

        row_delta = {
            "Setting": setting,
            "System": "Δ vs Best Single",
            "k": delta_k,
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

        df_out = pd.DataFrame(rows)

        if "k" in df_out.columns:
            df_out["k"] = df_out["k"].apply(
                lambda x: "" if pd.isna(x) else str(int(x))
            )

        results[setting] = df_out

    return results

def _derive_audio_derivative_group(asr_audio_in: str) -> str:
    """
    Map ASR input audio to audio-derivative group.

    Args:
        asr_audio_in: ASR input audio name.

    Returns:
        Audio derivative group label.
    """
    value = str(asr_audio_in)

    if value in {"original", "std"}:
        return "original_like"
    if value in {"std_vocals", "std_vocals_norm"}:
        return "vocals_like"
    if value in {"std_background", "std_background_norm"}:
        return "background_like"

    return "unknown"


def inspect_top_n_rq2a_configs_by_group(
    ranking_single: pd.DataFrame,
    *,
    mode: str,
    group_by: str,
    top_n: int = 2,
    setting_order: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Inspect top-n RQ2a single configurations per setting and group.

    Supported group_by values:
    - "audio_derivative_group"
    - "vad_mask"

    Args:
        ranking_single: Concatenated RQ2a single-ranking DataFrame.
        mode: Evaluation mode.
        group_by: Grouping column or derived grouping key.
        top_n: Number of top rows to keep per setting/group.
        setting_order: Optional explicit setting order.

    Returns:
        Filtered and sorted inspection DataFrame.
    """
    validate_mode(mode)
    if top_n <= 0:
        raise ValueError(f"top_n must be > 0, got: {top_n}")

    _require_columns(
        ranking_single,
        ["setting", "mode", "combo_key", "vad_mask", "asr_audio_in"],
        label="RQ2a single ranking inspection",
    )

    df = ranking_single[ranking_single["mode"] == mode].copy()
    if df.empty:
        return pd.DataFrame()

    if group_by == "audio_derivative_group":
        df[group_by] = df["asr_audio_in"].astype(str).apply(_derive_audio_derivative_group)
    else:
        if group_by not in df.columns:
            raise KeyError(f"Missing grouping column: {group_by}")

    ordered_metrics = get_ordered_metric_names(mode)

    sort_cols = ["setting", group_by]
    ascending = [True, True]

    for metric_name in ordered_metrics:
        if metric_name in df.columns:
            sort_cols.append(metric_name)
            ascending.append(get_metric_sort_ascending(metric_name))

    if "rank_within_run" in df.columns:
        sort_cols.append("rank_within_run")
        ascending.append(True)

    df = df.sort_values(
        by=sort_cols,
        ascending=ascending,
    )

    df = (
        df.groupby(["setting", group_by], as_index=False, group_keys=False)
        .head(top_n)
        .copy()
    )

    if setting_order is not None and "setting" in df.columns:
        df["setting"] = pd.Categorical(
            df["setting"],
            categories=setting_order,
            ordered=True,
        )
        df = df.sort_values(
            by=["setting", group_by],
            ascending=[True, True],
        )

    keep_cols = [
        "setting",
        group_by,
        "combo_key",
        "vad_mask",
        "asr_audio_in",
    ] + [metric_name for metric_name in ordered_metrics if metric_name in df.columns]

    if "rank_within_run" in df.columns:
        keep_cols.append("rank_within_run")

    return df[keep_cols].reset_index(drop=True)

def build_rq2b_derivative_tables(
    derivative_comparison: pd.DataFrame,
    *,
    setting_order: Optional[list[str]] = None,
) -> dict[str, pd.DataFrame]:
    """
    Build thesis-ready RQ2b audio-derivative tables split by mode.

    The reported metric values are group-level means.
    The 'Best Config' column is included only as contextual information about
    the strongest configuration within each derivative group.

    Args:
        derivative_comparison: Combined derivative comparison table.
        setting_order: Optional explicit setting order from notebook specs.

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
        mode = str(row["mode"])
        ordered_metrics = get_ordered_metric_names(mode)

        best_config = f"{row.get('best_vad_mask', '')} + {row.get('best_asr_audio_in', '')}"

        base_row = {
            "Setting": row["setting"],
            "Audio Derivative": row["audio_derivative_group"],
        }

        for metric_name in ordered_metrics:
            if metric_name in row.index:
                base_row[get_metric_label(metric_name)] = row.get(metric_name)

        # Best Config ans Ende
        base_row["Best Config (VAD Mask + ASR Audio Input)"] = best_config

        if mode in rows_by_mode:
            rows_by_mode[mode].append(base_row)

    out = {
        "full_gt": pd.DataFrame(rows_by_mode["full_gt"]),
        "part_gt": pd.DataFrame(rows_by_mode["part_gt"]),
    }

    derivative_order = [
        "original_like",
        "vocals_like",
        "background_like",
        "all_derivatives",
    ]

    for mode_key in ["full_gt", "part_gt"]:
        if out[mode_key].empty:
            continue

        if setting_order is not None:
            out[mode_key]["Setting"] = pd.Categorical(
                out[mode_key]["Setting"],
                categories=setting_order,
                ordered=True,
            )

        out[mode_key]["Audio Derivative"] = pd.Categorical(
            out[mode_key]["Audio Derivative"],
            categories=derivative_order,
            ordered=True,
        )

        sort_cols = ["Audio Derivative"]
        ascending = [True]

        if setting_order is not None:
            sort_cols = ["Setting", "Audio Derivative"]
            ascending = [True, True]

        out[mode_key] = out[mode_key].sort_values(
            by=sort_cols,
            ascending=ascending,
        ).reset_index(drop=True)

    return out

def build_rq2b_vad_mask_tables(
    ranking_single: pd.DataFrame,
    *,
    setting_order: Optional[list[str]] = None,
) -> dict[str, pd.DataFrame]:
    """
    Build inspection tables for RQ2 grouped by VAD mask.

    Args:
        ranking_single: Concatenated RQ2a single-ranking DataFrame.
        setting_order: Optional explicit setting order from notebook specs.

    Returns:
        {
            "full_gt": DataFrame,
            "part_gt": DataFrame,
        }
    """
    _require_columns(
        ranking_single,
        ["setting", "mode", "vad_mask", "combo_key", "asr_audio_in"],
        label="RQ2b VAD mask table input",
    )

    out: dict[str, pd.DataFrame] = {}

    for mode in ["full_gt", "part_gt"]:
        df_mode = ranking_single[ranking_single["mode"] == mode].copy()
        if df_mode.empty:
            out[mode] = pd.DataFrame()
            continue

        ordered_metrics = get_ordered_metric_names(mode)

        group_cols = ["setting", "vad_mask"]
        agg_dict: dict[str, tuple[str, str]] = {}

        for metric_name in ordered_metrics:
            if metric_name in df_mode.columns:
                agg_dict[metric_name] = (metric_name, "mean")

        df_out = (
            df_mode.groupby(group_cols, dropna=False)
            .agg(**agg_dict)
            .reset_index()
        )

        rename_map = {"setting": "Setting", "vad_mask": "VAD Mask"}
        for metric_name in ordered_metrics:
            if metric_name in df_out.columns:
                rename_map[metric_name] = get_metric_label(metric_name)

        df_out = df_out.rename(columns=rename_map)

        if setting_order is not None and "Setting" in df_out.columns:
            df_out["Setting"] = pd.Categorical(
                df_out["Setting"],
                categories=setting_order,
                ordered=True,
            )

        df_out = df_out.sort_values(
            by=["Setting", "VAD Mask"] if "Setting" in df_out.columns else ["VAD Mask"],
            ascending=[True, True] if "Setting" in df_out.columns else [True],
        ).reset_index(drop=True)

        out[mode] = df_out

    return out


def build_rq3_full_gt_label_tables(df_rq3_label: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Build thesis-ready RQ3 full_gt label coverage tables.

    Args:
        df_rq3_label: Concatenated RQ3 label artifact across settings.

    Returns:
        Dict mapping setting -> formatted label coverage table.
    """
    _require_columns(
        df_rq3_label,
        [
            "mode",
            "dataset_name",
            "label",
            "n_gt_events",
            "tp",
            "fn",
            "recall",
            "dice_eos_recall",
            "mean_dice_eos_tp",
            "mean_overlap_s",
        ],
        label="RQ3 full_gt label coverage",
    )

    df_full = df_rq3_label[df_rq3_label["mode"] == "full_gt"].copy()
    if df_full.empty:
        return {}

    results: dict[str, pd.DataFrame] = {}

    for dataset_name in df_full["dataset_name"].drop_duplicates().tolist():
        df_ds = df_full[df_full["dataset_name"] == dataset_name].copy()
        setting = f"{dataset_name} | full_gt"

        df_ds.insert(0, "Setting", setting)

        df_ds = df_ds.rename(
            columns={
                "label": "Label",
                "n_gt_events": "n_gt_events",
                "tp": "tp",
                "fn": "fn",
                "recall": "Recall",
                "dice_eos_recall": "EOS Recall",
                "mean_dice_eos_tp": "Mean EOS TP",
                "mean_overlap_s": "Mean Overlap (s)",
            }
        )

        ordered_cols = [
            "Setting",
            "Label",
            "n_gt_events",
            "tp",
            "fn",
            "Recall",
            "EOS Recall",
            "Mean EOS TP",
            "Mean Overlap (s)",
        ]
        df_ds = df_ds[ordered_cols].copy()

        df_ds = df_ds.sort_values(
            by=["Recall", "EOS Recall", "Mean EOS TP", "Label"],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)

        results[setting] = df_ds

    return results


def build_rq3_part_gt_event_tables(df_rq3_label: pd.DataFrame) -> dict[str, pd.DataFrame]:
    #toDo: normalize column names for gt/cand label across csvs to avoid this complexity in the table-building code
    """
    Build thesis-ready RQ3 part_gt event tables.

    Args:
        df_rq3_label: Concatenated RQ3 label artifact across settings.
            For part_gt this artifact is the event list.

    Returns:
        Dict mapping setting -> formatted event table.
    """
    _require_columns(
        df_rq3_label,
        [
            "mode",
            "dataset_name",
            "audio_id",
            "gt_event_id",
            "gt_label",
            "cand_event_id",
            "cand_label",
            "status",
            "dice_eos",
            "overlap_s",
        ],
        label="RQ3 part_gt event list",
    )

    df_part = df_rq3_label[df_rq3_label["mode"] == "part_gt"].copy()
    if df_part.empty:
        return {}

    results: dict[str, pd.DataFrame] = {}

    for dataset_name in df_part["dataset_name"].drop_duplicates().tolist():
        df_ds = df_part[df_part["dataset_name"] == dataset_name].copy()
        setting = f"{dataset_name} | part_gt"

        df_ds.insert(0, "Setting", setting)

        df_ds = df_ds.rename(
            columns={
                "audio_id": "Audio ID",
                "gt_event_id": "GT Event ID",
                "gt_label": "GT Label",
                "cand_event_id": "Cand Event ID",
                "cand_label": "Cand Label",
                "status": "Status",
                "dice_eos": "EOS",
                "overlap_s": "Overlap (s)",
            }
        )

        if "Status" in df_ds.columns:
            df_ds["Status"] = df_ds["Status"].replace(
                {
                    "hit": "Hit",
                    "miss": "Miss",
                    "insertion": "Insertion",
                }
            )

        ordered_cols = [
            "Setting",
            "Audio ID",
            "GT Event ID",
            "GT Label",
            "Cand Event ID",
            "Cand Label",
            "Status",
            "EOS",
            "Overlap (s)",
        ]
        df_ds = df_ds[ordered_cols].copy()

        status_order = {"Hit": 0, "Miss": 1, "Insertion": 2}
        df_ds["_status_order"] = df_ds["Status"].map(status_order).fillna(99)

        sort_cols = ["_status_order"]
        ascending = [True]

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
        df_ds = df_ds.drop(columns="_status_order")

        results[setting] = df_ds

    return results


def build_rq3_global_tables(df_rq3_global: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Build thesis-ready RQ3 global tables split by mode.

    Args:
        df_rq3_global: Concatenated RQ3 global artifact across settings.

    Returns:
        Dict with keys:
            "full_gt"
            "part_gt"
    """
    _require_columns(
        df_rq3_global,
        [
            "mode",
            "dataset_name",
            "n_gt_events_total",
            "tp_total",
            "fn_total",
            "insertions_total",
            "recall",
            "dice_eos_recall",
            "mean_dice_eos_tp",
            "insertion_rate",
            "deletion_rate",
        ],
        label="RQ3 global",
    )

    results: dict[str, pd.DataFrame] = {}

    df_full = df_rq3_global[df_rq3_global["mode"] == "full_gt"].copy()
    if not df_full.empty:
        if "f1" not in df_full.columns:
            raise KeyError("Expected column 'f1' in full_gt RQ3 global artifact.")

        df_full.insert(
            0,
            "Setting",
            df_full["dataset_name"].astype(str) + " | " + df_full["mode"].astype(str),
        )

        df_full = df_full.rename(
            columns={
                "n_gt_events_total": "n_gt_events_total",
                "tp_total": "tp_total",
                "fn_total": "fn_total",
                "insertions_total": "insertions_total",
                "f1": "F1",
                "recall": "Recall",
                "dice_eos_recall": "EOS Recall",
                "mean_dice_eos_tp": "Mean EOS TP",
                "insertion_rate": "Insertion Rate",
                "deletion_rate": "Deletion Rate",
            }
        )

        ordered_cols = [
            "Setting",
            "n_gt_events_total",
            "tp_total",
            "fn_total",
            "insertions_total",
            "F1",
            "Recall",
            "EOS Recall",
            "Mean EOS TP",
            "Insertion Rate",
            "Deletion Rate",
        ]
        df_full = df_full[ordered_cols].copy()
        results["full_gt"] = df_full.reset_index(drop=True)

    df_part = df_rq3_global[df_rq3_global["mode"] == "part_gt"].copy()
    if not df_part.empty:
        df_part.insert(
            0,
            "Setting",
            df_part["dataset_name"].astype(str) + " | " + df_part["mode"].astype(str),
        )

        df_part = df_part.rename(
            columns={
                "n_gt_events_total": "n_gt_events_total",
                "tp_total": "tp_total",
                "fn_total": "fn_total",
                "insertions_total": "insertions_total",
                "recall": "Recall",
                "dice_eos_recall": "EOS Recall",
                "mean_dice_eos_tp": "Mean EOS TP",
                "insertion_rate": "Insertion Rate",
                "deletion_rate": "Deletion Rate",
            }
        )

        ordered_cols = [
            "Setting",
            "n_gt_events_total",
            "tp_total",
            "fn_total",
            "insertions_total",
            "Recall",
            "EOS Recall",
            "Mean EOS TP",
            "Insertion Rate",
            "Deletion Rate",
        ]
        df_part = df_part[ordered_cols].copy()
        results["part_gt"] = df_part.reset_index(drop=True)

    return results