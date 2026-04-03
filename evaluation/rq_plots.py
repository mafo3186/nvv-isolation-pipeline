import math
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from evaluation.analysis_metrics import get_metric_label, get_ordered_metric_names

# --- RQ1 ---

def plot_rq1_full_gt_grouped_bars(rq1_full_gt: pd.DataFrame) -> plt.Figure:
    """
    Plot grouped bars for RQ1 full_gt: Baseline vs Best Selected Set across metrics.

    Args:
        rq1_full_gt: RQ1 full_gt table.

    Returns:
        Matplotlib figure.
    """
    df = rq1_full_gt.copy()
    df = df[df["System"].isin(["Baseline", "Best Selected Set"])].copy()

    metrics = [c for c in ["F1", "Recall", "Mean EOS TP"] if c in df.columns]
    systems = ["Baseline", "Best Selected Set"]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(metrics))
    width = 0.35

    for idx, system in enumerate(systems):
        row = df[df["System"] == system].iloc[0]
        values = [row[m] for m in metrics]
        ax.bar(x + (idx - 0.5) * width, values, width=width, label=system)

    ax.set_title("RQ1 Capability – Full-GT")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    fig.tight_layout()
    return fig

def plot_rq1_part_gt_grouped_bars(
    rq1_part_gt: pd.DataFrame,
    *,
    metric: str = "Recall",
    setting_order: Optional[list[str]] = None,
) -> plt.Figure:
    """
    Plot grouped bars for RQ1 part_gt across settings for one metric.

    Args:
        rq1_part_gt: RQ1 part_gt table.
        metric: Metric column to plot.
        setting_order: Explicit setting order from the analysis bundle.

    Returns:
        Matplotlib figure.
    """
    if metric not in rq1_part_gt.columns:
        raise KeyError(f"Missing metric column '{metric}'.")

    df = rq1_part_gt.copy()
    df = df[df["System"].isin(["Baseline", "Best Selected Set"])].copy()

    if setting_order is None:
        raise ValueError("plot_rq1_part_gt_grouped_bars() requires setting_order.")

    settings = [s for s in setting_order if s in df["Setting"].unique()]
    systems = ["Baseline", "Best Selected Set"]

    fig, ax = plt.subplots(figsize=(max(6, 2.8 * len(settings)), 5))
    x = np.arange(len(settings))
    width = 0.35

    for idx, system in enumerate(systems):
        values = []
        for setting in settings:
            subset = df[(df["Setting"] == setting) & (df["System"] == system)]
            if subset.empty:
                values.append(np.nan)
            else:
                values.append(subset.iloc[0][metric])
        ax.bar(x + (idx - 0.5) * width, values, width=width, label=system)

    ax.set_title(f"RQ1 Capability – Part-GT ({metric})")
    ax.set_xlabel("Setting")
    ax.set_ylabel(metric)
    ax.set_xticks(x)
    ax.set_xticklabels(settings, rotation=20)
    ax.legend()
    fig.tight_layout()
    return fig

# --- RQ2a ---

def plot_rq2a_rank_vs_score(
    ranking_single: pd.DataFrame,
    *,
    mode: str,
    score_col: str,
    top_k: int = 10,
    setting_order: Optional[list[str]] = None,
) -> plt.Figure:
    """
    Plot rank vs score with one line per setting.

    Args:
        ranking_single: Concatenated RQ2a single-ranking DataFrame.
        mode: "full_gt" or "part_gt".
        score_col: Metric column to plot.
        top_k: Number of ranks to show.
        setting_order: Explicit setting order from the analysis bundle.

    Returns:
        Matplotlib figure.
    """
    df = ranking_single.copy()
    df = df[df["mode"] == mode].copy()

    if score_col not in df.columns:
        raise KeyError(f"Missing score column '{score_col}'.")

    if setting_order is None:
        raise ValueError("plot_rq2a_rank_vs_score() requires setting_order.")

    settings = [s for s in setting_order if s in df["setting"].unique()]

    fig, ax = plt.subplots(figsize=(9, 5))

    for setting in settings:
        df_setting = df[df["setting"] == setting].copy()
        df_setting = df_setting.sort_values("rank_within_run", ascending=True).head(top_k)

        ax.plot(
            df_setting["rank_within_run"],
            df_setting[score_col],
            marker="o",
            label=setting,
        )

    ax.set_title(f"RQ2a Configuration Ranking –  Single Best – {get_metric_label(score_col)}")
    ax.set_xlabel("Rank")
    ax.set_ylabel(get_metric_label(score_col))
    ax.legend()
    fig.tight_layout()
    return fig

# --- RQ2b ---

def _derivative_order() -> list[str]:
    """
    Return the fixed derivative group order.

    Returns:
        Ordered list of derivative groups.
    """
    return [
        "original_like",
        "vocals_like",
        "background_like",
        "all_derivatives",
    ]


def _derive_audio_derivative_group(asr_audio_in: str) -> str:
    """
    Map ASR audio input to derivative group.

    Args:
        asr_audio_in: ASR audio derivative key.

    Returns:
        Derivative group label.
    """
    if asr_audio_in in {"original", "std"}:
        return "original_like"
    if asr_audio_in in {"std_vocals", "std_vocals_norm"}:
        return "vocals_like"
    if asr_audio_in in {"std_background", "std_background_norm"}:
        return "background_like"
    return "unknown"

def plot_rq2b_boxplot_with_points(
    ranking_single: pd.DataFrame,
    *,
    mode: str,
    score_col: str,
    top_k: Optional[int] = None,
    setting_order: Optional[list[str]] = None,
    jitter: float = 0.05,
) -> plt.Figure:
    """
    Plot RQ2b boxplots with points per setting using single-config results.

    Each subplot shows the score distribution across audio derivative groups.
    The "all_derivatives" group contains all configs of the setting.

    Args:
        ranking_single: Concatenated RQ2a single-ranking DataFrame.
        mode: "full_gt" or "part_gt".
        score_col: Metric column to plot.
        top_k: Optional limit per setting after sorting by rank.
        setting_order: Explicit setting order from the analysis bundle.
        jitter: Horizontal jitter for points.

    Returns:
        Matplotlib figure.
    """
    df = ranking_single.copy()
    df = df[df["mode"] == mode].copy()

    if score_col not in df.columns:
        raise KeyError(f"Missing score column '{score_col}'.")

    if "asr_audio_in" not in df.columns:
        raise KeyError("Missing column 'asr_audio_in'.")

    if "rank_within_run" not in df.columns:
        raise KeyError("Missing column 'rank_within_run'.")

    if setting_order is None:
        raise ValueError("plot_rq2b_boxplot_with_points() requires setting_order.")

    df["audio_derivative_group"] = df["asr_audio_in"].apply(_derive_audio_derivative_group)

    if top_k is not None:
        df = (
            df.sort_values(["setting", "rank_within_run"], ascending=[True, True])
            .groupby("setting", as_index=False, group_keys=False)
            .head(top_k)
            .copy()
        )

    settings = [s for s in setting_order if s in df["setting"].unique()]
    n_panels = len(settings)

    if n_panels == 0:
        raise ValueError("No settings available for the selected mode.")

    ncols = 2
    nrows = math.ceil(n_panels / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 4.5 * nrows))
    axes = np.array(axes).reshape(-1)

    group_order = _derivative_order()
    score_label = get_metric_label(score_col)

    rng = np.random.default_rng(42)

    for ax, setting in zip(axes, settings):
        df_setting = df[df["setting"] == setting].copy()

        grouped_values = []
        for group in group_order:
            if group == "all_derivatives":
                values = df_setting[score_col].dropna().tolist()
            else:
                values = df_setting.loc[
                    df_setting["audio_derivative_group"] == group,
                    score_col,
                ].dropna().tolist()
            grouped_values.append(values)

        ax.boxplot(grouped_values, labels=group_order)

        for idx, values in enumerate(grouped_values, start=1):
            if not values:
                continue
            x = rng.normal(loc=idx, scale=jitter, size=len(values))
            ax.scatter(x, values, alpha=0.85)

        ax.set_title(f"RQ2b Configuration – Audio Derivatives – {setting}")
        ax.set_xlabel("Audio Derivative Group")
        ax.set_ylabel(score_label)
        ax.tick_params(axis="x", rotation=20)

    for ax in axes[n_panels:]:
        ax.set_visible(False)

    fig.tight_layout()
    return fig

def plot_rq2b_vad_mask_boxplot_with_points(
    ranking_single: pd.DataFrame,
    *,
    mode: str,
    score_col: str,
    top_k: Optional[int] = None,
    setting_order: Optional[list[str]] = None,
    jitter: float = 0.05,
) -> plt.Figure:
    """
    Plot RQ2b boxplots with points per setting using VAD masks.

    Args:
        ranking_single: Concatenated RQ2a single-ranking DataFrame.
        mode: "full_gt" or "part_gt".
        score_col: Metric column to plot.
        top_k: Optional limit per setting after sorting by rank.
        setting_order: Explicit setting order from the analysis bundle.
        jitter: Horizontal jitter for points.

    Returns:
        Matplotlib figure.
    """
    df = ranking_single.copy()
    df = df[df["mode"] == mode].copy()

    if score_col not in df.columns:
        raise KeyError(f"Missing score column '{score_col}'.")

    if "vad_mask" not in df.columns:
        raise KeyError("Missing column 'vad_mask'.")

    if "rank_within_run" not in df.columns:
        raise KeyError("Missing column 'rank_within_run'.")

    if setting_order is None:
        raise ValueError("plot_rq2b_vad_mask_boxplot_with_points() requires setting_order.")

    if top_k is not None:
        df = (
            df.sort_values(["setting", "rank_within_run"], ascending=[True, True])
            .groupby("setting", as_index=False, group_keys=False)
            .head(top_k)
            .copy()
        )

    settings = [s for s in setting_order if s in df["setting"].unique()]
    n_panels = len(settings)

    if n_panels == 0:
        raise ValueError("No settings available for the selected mode.")

    ncols = 2
    nrows = math.ceil(n_panels / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 4.5 * nrows))
    axes = np.array(axes).reshape(-1)

    score_label = get_metric_label(score_col)
    rng = np.random.default_rng(42)

    for ax, setting in zip(axes, settings):
        df_setting = df[df["setting"] == setting].copy()
        mask_order = sorted(df_setting["vad_mask"].dropna().astype(str).unique().tolist())

        grouped_values = []
        for mask_name in mask_order:
            values = df_setting.loc[
                df_setting["vad_mask"].astype(str) == mask_name,
                score_col,
            ].dropna().tolist()
            grouped_values.append(values)

        ax.boxplot(grouped_values, labels=mask_order)

        for idx, values in enumerate(grouped_values, start=1):
            if not values:
                continue
            x = rng.normal(loc=idx, scale=jitter, size=len(values))
            ax.scatter(x, values, alpha=0.85)

        ax.set_title(f"RQ2b Configuration – VAD Masks – {setting}")
        ax.set_xlabel("VAD Mask")
        ax.set_ylabel(score_label)
        ax.tick_params(axis="x", rotation=20)

    for ax in axes[n_panels:]:
        ax.set_visible(False)

    fig.tight_layout()
    return fig

# --- RQ3 ---

def plot_rq3_label_coverage(
    df_rq3_full_gt_label: pd.DataFrame,
    *,
    setting: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Plot RQ3 full_gt label coverage as stacked TP/FN counts.

    Args:
        df_rq3_full_gt_label: One formatted full_gt label coverage table.
        setting: Optional setting label for the plot title.
            If None, the function tries to infer it from the "Setting" column.
        ax: Optional matplotlib Axes.

    Returns:
        Matplotlib Figure.
    """
    df = df_rq3_full_gt_label.copy()

    required_cols = {"Label", "tp", "fn"}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns for label coverage plot: {missing}")

    if setting is None:
        if "Setting" in df.columns and not df["Setting"].dropna().empty:
            setting = str(df["Setting"].dropna().iloc[0])
        else:
            setting = "full_gt"

    # Sort by total GT count
    df["_total"] = df["tp"] + df["fn"]
    df = df.sort_values(by=["_total", "Label"], ascending=[False, True])

    labels = df["Label"].tolist()
    tp = df["tp"].astype(float).values
    fn = df["fn"].astype(float).values

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.figure

    x = np.arange(len(labels))

    ax.bar(x, tp, label="TP")
    ax.bar(x, fn, bottom=tp, label="FN")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Number of events")
    ax.set_title(f"RQ3 Label Coverage (Counts) – {setting}")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    return fig


def plot_rq3_label_quality(
    df_rq3_full_gt_label: pd.DataFrame,
    *,
    setting: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Plot RQ3 full_gt label-wise quality metrics as grouped bars.

    Metrics:
        - Recall
        - EOS Recall
        - Mean EOS TP

    Args:
        df_rq3_full_gt_label: One formatted full_gt label coverage table.
        setting: Optional setting label for the plot title.
            If None, the function tries to infer it from the "Setting" column.
        ax: Optional matplotlib Axes.

    Returns:
        Matplotlib Figure.
    """
    df = df_rq3_full_gt_label.copy()

    required_cols = {"Label", "Recall", "EOS Recall", "Mean EOS TP"}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns for label quality plot: {missing}")

    if setting is None:
        if "Setting" in df.columns and not df["Setting"].dropna().empty:
            setting = str(df["Setting"].dropna().iloc[0])
        else:
            setting = "full_gt"

    # Sort by Recall, then EOS Recall, then Mean EOS TP
    df = df.sort_values(
        by=["Recall", "EOS Recall", "Mean EOS TP", "Label"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)

    labels = df["Label"].tolist()
    recall_vals = df["Recall"].astype(float).values
    eos_recall_vals = df["EOS Recall"].astype(float).values
    mean_eos_tp_vals = df["Mean EOS TP"].astype(float).values

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.figure

    x = np.arange(len(labels))
    width = 0.25

    ax.bar(x - width, recall_vals, width=width, label="Recall", color="tab:blue")
    ax.bar(x, eos_recall_vals, width=width, label="EOS Recall", color="tab:green")
    ax.bar(x + width, mean_eos_tp_vals, width=width, label="Mean EOS TP", color="tab:purple")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(f"RQ3 Label Quality – {setting}")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    return fig


def plot_rq3_global_recall_comparison(
    df_rq3_global: pd.DataFrame,
    *,
    setting_order: Optional[list[str]] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Plot global Recall comparison across settings.

    Args:
        df_rq3_global: Concatenated rq3_global DataFrame across settings.
        setting_order: Optional list defining the order of settings on x-axis.
        ax: Optional matplotlib Axes.

    Returns:
        Matplotlib Figure.
    """
    df = df_rq3_global.copy()

    required_cols = {"dataset_name", "mode", "recall"}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns for global recall plot: {missing}")

    # Build setting label
    df["Setting"] = df["dataset_name"].astype(str) + " | " + df["mode"].astype(str)

    # Order settings
    if setting_order is not None:
        df["Setting"] = pd.Categorical(df["Setting"], categories=setting_order, ordered=True)
        df = df.sort_values("Setting")
    else:
        df = df.sort_values("Setting")

    settings = df["Setting"].tolist()
    recall_vals = df["recall"].astype(float).values

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.figure

    x = np.arange(len(settings))

    ax.bar(x, recall_vals)

    ax.set_xticks(x)
    ax.set_xticklabels(settings, rotation=45, ha="right")

    ax.set_ylabel("Recall")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("RQ3 Global Recall Comparison across Settings")

    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    return fig

def plot_rq3_binary_coverage(
    rq3_full_gt_tables: dict[str, pd.DataFrame],
    *,
    setting: str,
) -> plt.Figure:
    """
    Legacy plot - may be deleted in the future.
    Plot binary aggregated coverage for one full_gt setting.

    TP and FN are summed across all labels.
    FP/Insertions are shown as a text annotation.

    Args:
        rq3_full_gt_tables: Dict mapping setting -> full_gt coverage table.
        setting: Setting key to plot.

    Returns:
        Matplotlib figure.
    """
    if setting not in rq3_full_gt_tables:
        raise KeyError(f"Setting '{setting}' not found.")

    df = rq3_full_gt_tables[setting].copy()

    fp_mask = df["Label"].astype(str).isin(["__FP__", "FP", "Insertions"])
    df_fp = df[fp_mask].copy()
    df_labels = df[~fp_mask].copy()

    tp_total = df_labels["n_tp"].fillna(0).sum()
    fn_total = df_labels["n_fn"].fillna(0).sum()

    insertions = None
    if not df_fp.empty:
        if "FP" in df_fp.columns:
            insertions = df_fp["FP"].dropna().iloc[0] if df_fp["FP"].dropna().size else None
        elif "Insertions" in df_fp.columns:
            insertions = df_fp["Insertions"].dropna().iloc[0] if df_fp["Insertions"].dropna().size else None

    fig, ax = plt.subplots(figsize=(5, 5))

    ax.bar(["Coverage"], [tp_total], label="TP")
    ax.bar(["Coverage"], [fn_total], bottom=[tp_total], label="FN")

    ax.set_title(f"RQ3 Binary Coverage – {setting}")
    ax.set_ylabel("Count")
    ax.legend()

    if insertions is not None:
        ax.text(
            0.99,
            0.98,
            f"Insertions: {int(insertions)}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    fig.tight_layout()
    return fig
