import math
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from evaluation.analysis_metrics import get_metric_label, get_ordered_metric_names


# --- Barplot Wrapper ---

def plot_grouped_bars_all_settings(
    full_df: pd.DataFrame,
    part_df: pd.DataFrame,
    *,
    setting_order: Optional[list[str]] = None,
    systems: Optional[list[str]] = None,
    colors: Optional[dict[str, str]] = None,
    full_metrics: Optional[list[str]] = None,
    part_metrics: Optional[list[str]] = None,
    y_label: str = "Score",
    y_lim: tuple[float, float] = (0, 0.35),
    font_size: int = 16,
    bar_width: float = 0.28,
    panel_width: float = 4.2,
    panel_height: float = 4.8,
    x_margin: float = 0.20,
    legend_bbox_y: float = 1.02,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Plot one horizontal grouped-bar overview figure with one panel per setting.

    Full-GT panels use `full_metrics`.
    Part-GT panels use `part_metrics`.

    Args:
        full_df: Formatted full_gt table.
        part_df: Formatted part_gt table.
        setting_order: Explicit global setting order.
        systems: Systems to compare, in plot order.
        colors: Mapping system -> color.
        full_metrics: Metrics shown for full_gt panels.
        part_metrics: Metrics shown for part_gt panels.
        y_label: Shared y-axis label.
        y_lim: Shared y-axis limits.
        font_size: Base font size.
        bar_width: Bar width.
        panel_width: Width per subplot panel.
        panel_height: Figure height.
        x_margin: Horizontal subplot margin.
        legend_bbox_y: Vertical anchor of shared legend.
        title: Optional figure title.

    Returns:
        Matplotlib figure.
    """
    if systems is None:
        systems = ["Baseline", "Best Single", "Best Selected Set"]

    default_palette = {
        "Baseline": "#7A8594",
        "Best Single": "#719AD4",
        "Best Selected Set": "#0655CB",
    }

    if colors is None:
        colors = {}
        palette_values = list(default_palette.values())
        for idx, system in enumerate(systems):
            if system in default_palette:
                colors[system] = default_palette[system]
            else:
                colors[system] = palette_values[idx % len(palette_values)]

    if full_metrics is None:
        full_metrics = ["F1", "Recall", "EOS Recall"]

    if part_metrics is None:
        part_metrics = ["Recall", "EOS Recall"]

    full_df = full_df.copy()
    part_df = part_df.copy()

    full_df = full_df[full_df["System"].isin(systems)].copy()
    part_df = part_df[part_df["System"].isin(systems)].copy()

    full_settings = full_df["Setting"].dropna().astype(str).unique().tolist()

    if setting_order is None:
        part_settings = part_df["Setting"].dropna().astype(str).unique().tolist()
    else:
        part_settings = [
            s for s in setting_order
            if s in part_df["Setting"].dropna().astype(str).unique().tolist()
        ]

    all_panels: list[tuple[str, pd.DataFrame, list[str]]] = []

    for setting in full_settings:
        all_panels.append(
            (
                setting,
                full_df[full_df["Setting"] == setting].copy(),
                [m for m in full_metrics if m in full_df.columns],
            )
        )

    for setting in part_settings:
        all_panels.append(
            (
                setting,
                part_df[part_df["Setting"] == setting].copy(),
                [m for m in part_metrics if m in part_df.columns],
            )
        )

    n_panels = len(all_panels)
    if n_panels == 0:
        raise ValueError("No settings available for plotting.")

    fig, axes = plt.subplots(
        nrows=1,
        ncols=n_panels,
        figsize=(panel_width * n_panels, panel_height),
        sharey=True,
    )

    if n_panels == 1:
        axes = [axes]

    legend_handles = None
    legend_labels = None

    for ax, (setting, df_setting, metrics) in zip(axes, all_panels):
        x = np.arange(len(metrics)) * 1.0

        local_handles = []
        local_labels = []

        # symmetric positioning for any number of systems
        center_offset = (len(systems) - 1) / 2

        for idx, system in enumerate(systems):
            row = df_setting[df_setting["System"] == system]

            if row.empty:
                values = [np.nan] * len(metrics)
            else:
                row = row.iloc[0]
                values = [row[m] for m in metrics]

            bars = ax.bar(
                x + (idx - center_offset) * bar_width,
                values,
                width=bar_width,
                color=colors[system],
                label=system,
            )

            if legend_handles is None:
                local_handles.append(bars[0])
                local_labels.append(system)

        if legend_handles is None:
            legend_handles = local_handles
            legend_labels = local_labels

        ax.margins(x=x_margin)
        ax.set_title(setting, fontsize=font_size)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=font_size)
        ax.set_ylim(*y_lim)
        ax.tick_params(axis="y", labelsize=font_size)
        ax.grid(axis="y", linestyle="--", alpha=0.35)

    axes[0].set_ylabel(y_label, fontsize=font_size)

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=len(systems),
        frameon=False,
        bbox_to_anchor=(0.5, legend_bbox_y),
        fontsize=font_size,
    )

    if title is not None:
        fig.suptitle(title, fontsize=font_size)
        fig.tight_layout(rect=[0, 0, 1, 0.90])
    else:
        fig.tight_layout(rect=[0, 0, 1, 0.92])

    return fig


# --- RQ1 Capability ---
def plot_rq1_all_settings_grouped_bars(
    rq1_full_gt: pd.DataFrame,
    rq1_part_gt: pd.DataFrame,
    *,
    setting_order: Optional[list[str]] = None,
    colors: Optional[dict[str, str]] = None,
) -> plt.Figure:
    """
    Plot RQ1 grouped bars across all settings.
    """
    return plot_grouped_bars_all_settings(
        rq1_full_gt,
        rq1_part_gt,
        setting_order=setting_order,
        systems=["Baseline", "Best Selected Set"],
        colors=colors,
        full_metrics=["F1", "Recall", "EOS Recall"],
        part_metrics=["Recall", "EOS Recall"],
        y_label="Score",
    )


def plot_rq2a_selected_set_all_settings_grouped_bars(
    rq2_full_gt: pd.DataFrame,
    rq2_part_gt: pd.DataFrame,
    *,
    setting_order: Optional[list[str]] = None,
    colors: Optional[dict[str, str]] = None,
) -> plt.Figure:
    """
    Plot RQ2a selected-set grouped bars across all settings.
    """
    return plot_grouped_bars_all_settings(
        rq2_full_gt,
        rq2_part_gt,
        setting_order=setting_order,
        systems=["Best Single", "Best Selected Set"],
        colors=colors,
        full_metrics=["F1", "Recall", "EOS Recall"],
        part_metrics=["Recall", "EOS Recall"],
        y_label="Score",
    )

# --- legacy!!! RQ1 Capability ---

def plot_rq1_full_gt_grouped_bars(rq1_full_gt: pd.DataFrame) -> plt.Figure:
    """
    Plot grouped bars for RQ1 full_gt: Baseline vs Best Selected Set across metrics.

    Metrics shown:
        - F1
        - Recall
        - EOS Recall

    Args:
        rq1_full_gt: RQ1 full_gt table.

    Returns:
        Matplotlib figure.
    """
    df = rq1_full_gt.copy()
    df = df[df["System"].isin(["Baseline", "Best Selected Set"])].copy()

    metrics = [c for c in ["F1", "Recall", "EOS Recall"] if c in df.columns]
    systems = ["Baseline", "Best Selected Set"]

    colors = {
        "Baseline": "#7A8594",
        "Best Selected Set": "#0655CB",
    }

    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    # Slightly larger spacing between metric groups
    x = np.arange(len(metrics)) * 1.0
    width = 0.2

    for idx, system in enumerate(systems):
        row = df[df["System"] == system].iloc[0]
        values = [row[m] for m in metrics]
        ax.bar(
            x + (idx - 0.5) * width,
            values,
            width=width,
            label=system,
            color=colors[system],
            #hatch=hatches[system],
            #edgecolor="black",
            #linewidth=0.6,
        )
        

    ax.margins(x=0.15)
    ax.set_title("NVS-38K_EN | full_gt")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 0.35)
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    fig.tight_layout()
    return fig


def plot_rq1_part_gt_grouped_bars(
    rq1_part_gt: pd.DataFrame,
    *,
    setting_order: Optional[list[str]] = None,
) -> plt.Figure:
    """
    Plot grouped bars for RQ1 part_gt as three subplots, one per setting.

    Metrics shown per subplot:
        - Recall
        - EOS Recall

    Args:
        rq1_part_gt: RQ1 part_gt table.
        setting_order: Explicit setting order from the analysis bundle.

    Returns:
        Matplotlib figure.
    """
    required_metrics = ["Recall", "EOS Recall"]
    missing = [m for m in required_metrics if m not in rq1_part_gt.columns]
    if missing:
        raise KeyError(f"Missing metric columns: {missing}")

    df = rq1_part_gt.copy()
    df = df[df["System"].isin(["Baseline", "Best Selected Set"])].copy()

    if setting_order is None:
        raise ValueError("plot_rq1_part_gt_grouped_bars() requires setting_order.")

    settings = [s for s in setting_order if s in df["Setting"].unique()]
    systems = ["Baseline", "Best Selected Set"]

    colors = {
        "Baseline": "#7A8594",
        "Best Selected Set": "#0655CB",
    }

    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(settings),
        figsize=(4.4 * len(settings), 4.5),
        sharey=True,
    )

    if len(settings) == 1:
        axes = [axes]

    metrics = ["Recall", "EOS Recall"]
    width = 0.2

    for ax, setting in zip(axes, settings):
        x = np.arange(len(metrics)) * 1.0

        for idx, system in enumerate(systems):
            subset = df[(df["Setting"] == setting) & (df["System"] == system)]
            if subset.empty:
                values = [np.nan] * len(metrics)
            else:
                row = subset.iloc[0]
                values = [row[m] for m in metrics]

            ax.bar(
                x + (idx - 0.5) * width,
                values,
                width=width,
                label=system,
                color=colors[system],
                #hatch=hatches[system],
                #edgecolor="black",
                #linewidth=0.6,
            )
            
        ax.margins(x=0.15)
        ax.set_title(setting)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 0.35)
        ax.grid(axis="y", linestyle="--", alpha=0.35)

    axes[0].set_ylabel("Score")
    axes[0].legend(frameon=False)

    fig.tight_layout()
    return fig



# --- RQ2a Ranking ---

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

    font_size = 16
    colors = {
        "NVS-38K_EN | full_gt": "#5A189A",   # Indigo Velvet
        "NVS-38K_EN | part_gt": "#5A189A",   # Indigo Velvet
        "VOCAL_RA1 | part_gt": "#1C9800",    # forest green 
        "VOCAL_RA2 | part_gt":  "#70E000",    # radioactive grass
    }

    for setting in settings:
        df_setting = df[df["setting"] == setting].copy()
        df_setting = df_setting.sort_values("rank_within_run", ascending=True).head(top_k)

        ax.plot(
            df_setting["rank_within_run"],
            df_setting[score_col],
            marker="o",
            label=setting,
            color=colors.get(setting, None),
            markersize=5,
            linewidth=1.5,
        )

    ax.set_title(f"Single Configuration Ranking – {mode} – {get_metric_label(score_col)}", fontsize=font_size)
    ax.set_xlabel("Rank", fontsize=font_size)
    # set x-ticks at integer ranks, with a step of 5 for readability
    ax.tick_params(axis="y", labelsize=(font_size*0.9))
    ax.tick_params(axis="x", labelsize=(font_size*0.9))
    ax.set_xticks(range(0, top_k, 5))
    ax.set_ylabel(get_metric_label(score_col), fontsize=font_size)
    ax.legend(fontsize=font_size)
    fig.tight_layout()
    return fig


# --- RQ2a Complimentary Combination ---
def plot_rq2a_f1_vs_k(
    df_f1_vs_k: pd.DataFrame,
    *,
    setting: str,
) -> plt.Figure:
    """
    Plot greedy forward selection curve (F1 vs k).

    Args:
        df_f1_vs_k: DataFrame with columns ["k", "macro_mean_f1"].
        setting: Setting label for title.

    Returns:
        Matplotlib figure.
    """
    required_cols = ["k", "macro_mean_f1"]
    missing = [c for c in required_cols if c not in df_f1_vs_k.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    df = df_f1_vs_k.copy().sort_values("k")

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    font_size = 16
    colors = {
        "NVS-38K_EN | full_gt": "#5A189A",
    }

    # extract best_k
    best_k = None
    if "best_k" in df.columns and df["best_k"].notna().any():
        best_k = int(df["best_k"].dropna().iloc[0])

    if best_k is not None:
        y_best = df.loc[df["k"] == best_k, "macro_mean_f1"].values[0]

    ax.scatter(best_k, y_best, s=80, zorder=3)

    ax.text(
        best_k,
        y_best + 0.003,
        f"  k={best_k}",
        va="bottom",
        ha="left",
        fontsize=font_size * 0.8,
    )

    ax.axvline(best_k, linestyle="--", linewidth=1, alpha=0.5)
    
    ax.plot(
        df["k"],
        df["macro_mean_f1"],
        marker="o",
        label=setting,
        color=colors.get(setting, None),
        markersize=5,
        linewidth=1.5,
    )

    ax.set_ylabel("F1", fontsize=font_size)
    y_max = df["macro_mean_f1"].max()

    if "best_k" in df.columns:
        best_row = df[df["best_k"] == True]
        if not best_row.empty:
            x = best_row["k"].iloc[0]
            y = best_row["macro_mean_f1"].iloc[0]
            ax.scatter([x], [y], zorder=3)
            ax.text(x, y, f"  k={int(x)}", va="bottom")

    ax.set_title(f"Greedy Forward Selection", fontsize=font_size)
    ax.set_xlabel("k - number of combined configurations", fontsize=font_size)
    ax.set_xticks(range(0, int(df["k"].max()) + 1, 5))
    ax.tick_params(axis="y", labelsize=(font_size*0.9))
    ax.tick_params(axis="x", labelsize=(font_size*0.9))
    ax.set_ylim(top=y_max * 1.08)
    ax.legend(fontsize=font_size)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout()
    return fig

# --- RQ2b Audio Derivative Groups---

def _derivative_group_order() -> list[str]:
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
        ax.tick_params(axis="y", labelsize=16)
    for ax in axes[n_panels:]:
        ax.set_visible(False)

    fig.tight_layout()
    return fig

# --- RQ2b new boxplots and heatmaps ---
def _rq2b_vad_mask_order() -> list[str]:
    """
    Return the fixed VAD mask order used in RQ2b plots.

    Returns:
        Ordered list of VAD mask names.
    """
    return [
        "no",
        "original",
        "std",
        "std_vocals",
        "std_vocals_norm",
        "std_background",
        "std_background_norm",
    ]


def _rq2b_asr_audio_order() -> list[str]:
    """
    Return the fixed ASR audio input order used in RQ2b plots.

    Returns:
        Ordered list of ASR audio derivative names.
    """
    return [
        "original",
        "std",
        "std_vocals",
        "std_vocals_norm",
        "std_background",
        "std_background_norm",
    ]


def plot_rq2b_boxplots_by_setting(
    ranking_single: pd.DataFrame,
    *,
    mode: str,
    score_col: str,
    top_k: Optional[int] = None,
    setting_order: Optional[list[str]] = None,
    jitter: float = 0.05,
    figsize: tuple[float, float] = (7.5, 4.8),
) -> dict[str, plt.Figure]:
    """
    Plot one RQ2b derivative-group boxplot per setting.

    Each figure shows the score distribution across audio derivative groups:
    - original_like
    - vocals_like
    - background_like
    - all_derivatives

    Args:
        ranking_single: Concatenated RQ2a single-ranking DataFrame.
        mode: "full_gt" or "part_gt".
        score_col: Metric column to plot.
        top_k: Optional limit per setting after sorting by rank.
        setting_order: Explicit setting order from the analysis bundle.
        jitter: Horizontal jitter for points.
        figsize: Figure size per setting.

    Returns:
        Dict mapping setting -> matplotlib figure.
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
        raise ValueError("plot_rq2b_boxplots_by_setting() requires setting_order.")

    df["audio_derivative_group"] = df["asr_audio_in"].apply(_derive_audio_derivative_group)

    if top_k is not None:
        df = (
            df.sort_values(["setting", "rank_within_run"], ascending=[True, True])
            .groupby("setting", as_index=False, group_keys=False)
            .head(top_k)
            .copy()
        )

    settings = [s for s in setting_order if s in df["setting"].unique()]
    if not settings:
        raise ValueError("No settings available for the selected mode.")

    group_order = _derivative_group_order()
    score_label = get_metric_label(score_col)
    rng = np.random.default_rng(42)

    figures: dict[str, plt.Figure] = {}
    font_size = 16

    for setting in settings:
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

        fig, ax = plt.subplots(figsize=figsize)
        ax.boxplot(grouped_values, labels=group_order)

        for idx, values in enumerate(grouped_values, start=1):
            if not values:
                continue
            x = rng.normal(loc=idx, scale=jitter, size=len(values))
            ax.scatter(x, values, alpha=0.85)

        ax.set_title(f"{setting}  ", fontsize=font_size, pad=15)
        ax.set_xlabel("ASR Audio Input (Derivative Group)", fontsize=font_size)
        ax.set_ylabel(score_label, fontsize=font_size)
        ax.tick_params(axis="x", rotation=20, labelsize=(font_size*0.9))
        ax.tick_params(axis="y", labelsize=(font_size*0.9))
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        fig.tight_layout()
        figures[setting] = fig

    return figures

# verification of distribution within group - not used in thesis
def plot_rq2b_boxplots_by_asr_audio_input_by_setting(
    ranking_single: pd.DataFrame,
    *,
    mode: str,
    score_col: str,
    top_k: Optional[int] = None,
    setting_order: Optional[list[str]] = None,
    jitter: float = 0.05,
    figsize: tuple[float, float] = (7.5, 4.8),
) -> dict[str, plt.Figure]:
    """
    Plot one RQ2b boxplot per setting using individual ASR audio inputs.

    Each figure shows the score distribution across the original ASR input
    derivatives:
    - original
    - std
    - std_vocals
    - std_vocals_norm
    - std_background
    - std_background_norm

    Args:
        ranking_single: Concatenated RQ2a single-ranking DataFrame.
        mode: "full_gt" or "part_gt".
        score_col: Metric column to plot.
        top_k: Optional limit per setting after sorting by rank.
        setting_order: Explicit setting order from the analysis bundle.
        jitter: Horizontal jitter for points.
        figsize: Figure size per setting.

    Returns:
        Dict mapping setting -> matplotlib figure.
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
        raise ValueError("plot_rq2b_boxplots_by_asr_audio_input_setting() requires setting_order.")

    if top_k is not None:
        df = (
            df.sort_values(["setting", "rank_within_run"], ascending=[True, True])
            .groupby("setting", as_index=False, group_keys=False)
            .head(top_k)
            .copy()
        )

    settings = [s for s in setting_order if s in df["setting"].unique()]
    if not settings:
        raise ValueError("No settings available for the selected mode.")

    asr_order = _rq2b_asr_audio_order()
    score_label = get_metric_label(score_col)
    rng = np.random.default_rng(42)

    figures: dict[str, plt.Figure] = {}
    font_size = 16

    for setting in settings:
        df_setting = df[df["setting"] == setting].copy()

        grouped_values = []
        for asr_audio in asr_order:
            values = df_setting.loc[
                df_setting["asr_audio_in"].astype(str) == asr_audio,
                score_col,
            ].dropna().tolist()
            grouped_values.append(values)

        fig, ax = plt.subplots(figsize=figsize)
        ax.boxplot(grouped_values, labels=asr_order)

        for idx, values in enumerate(grouped_values, start=1):
            if not values:
                continue
            x = rng.normal(loc=idx, scale=jitter, size=len(values))
            ax.scatter(x, values, alpha=0.85)

        ax.set_title(
            f"Configuration – ASR Audio Inputs – {setting}",
            fontsize=font_size,
            pad=15,
        )
        ax.set_xlabel("ASR Audio Input", fontsize=font_size)
        ax.set_ylabel(score_label, fontsize=font_size)
        ax.tick_params(axis="x", rotation=20, labelsize=(font_size * 0.9))
        ax.tick_params(axis="y", labelsize=(font_size * 0.9))
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        fig.tight_layout()
        figures[setting] = fig

    return figures

# visualization of interaction between VAD mask and ASR audio input 
def plot_rq2b_heatmaps_by_setting(
    ranking_single: pd.DataFrame,
    *,
    mode: str,
    score_col: str,
    setting_order: Optional[list[str]] = None,
    agg: str = "mean",
    annot: bool = True,
    annot_fontsize: int = 16,
    figsize: tuple[float, float] = (8.5, 5.8),
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> dict[str, plt.Figure]:
    """
    Plot one VAD-mask × ASR-audio heatmap per setting.

    The heatmap shows one cell per configuration:
    rows = VAD mask
    cols = ASR audio input
    values = selected metric

    Args:
        ranking_single: Concatenated RQ2a single-ranking DataFrame.
        mode: "full_gt" or "part_gt".
        score_col: Metric column to plot.
        setting_order: Explicit setting order from the analysis bundle.
        agg: Aggregation used in pivot table. Supported: "mean", "median", "max".
        annot: Whether to print numeric values inside cells.
        annot_fontsize: Font size for cell annotations.
        figsize: Figure size per setting.
        cmap: Matplotlib colormap name.
        vmin: Optional fixed lower color limit.
        vmax: Optional fixed upper color limit.

    Returns:
        Dict mapping setting -> matplotlib figure.
    """
    df = ranking_single.copy()
    df = df[df["mode"] == mode].copy()

    required_cols = {"setting", "vad_mask", "asr_audio_in", score_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns for heatmap plot: {sorted(missing)}")

    if setting_order is None:
        raise ValueError("plot_rq2b_heatmaps_by_setting() requires setting_order.")

    if agg not in {"mean", "median", "max"}:
        raise ValueError("agg must be one of {'mean', 'median', 'max'}.")

    settings = [s for s in setting_order if s in df["setting"].unique()]
    if not settings:
        raise ValueError("No settings available for the selected mode.")

    vad_order = _rq2b_vad_mask_order()
    asr_order = _rq2b_asr_audio_order()
    score_label = get_metric_label(score_col)
    
    figures: dict[str, plt.Figure] = {}

    for setting in settings:
        df_setting = df[df["setting"] == setting].copy()

        pivot = pd.pivot_table(
            df_setting,
            index="vad_mask",
            columns="asr_audio_in",
            values=score_col,
            aggfunc=agg,
        )

        pivot = pivot.reindex(index=vad_order, columns=asr_order)

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(
            pivot.values,
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

        ax.set_title(f"{setting} ", fontsize=annot_fontsize, pad=15)
        ax.set_xlabel("ASR Audio Input", fontsize=annot_fontsize)
        ax.set_ylabel("VAD Mask", fontsize=annot_fontsize)

        ax.set_xticks(np.arange(len(asr_order)))
        ax.set_xticklabels(asr_order, rotation=25, ha="right", fontsize=(annot_fontsize * 0.9))
        ax.set_yticks(np.arange(len(vad_order)))
        ax.set_yticklabels(vad_order, fontsize=(annot_fontsize * 0.9))

        # Draw cell borders
        ax.set_xticks(np.arange(-0.5, len(asr_order), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(vad_order), 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=1)
        ax.tick_params(which="minor", bottom=False, left=False)

        if annot:
            for row_idx in range(len(vad_order)):
                for col_idx in range(len(asr_order)):
                    value = pivot.iloc[row_idx, col_idx]
                    if pd.isna(value):
                        text = "–"
                    else:
                        text = f"{value:.3f}"

                    ax.text(
                        col_idx,
                        row_idx,
                        text,
                        ha="center",
                        va="center",
                        color="white" if pd.notna(value) else "black",
                        fontsize=annot_fontsize,
                    )

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(score_label, fontsize=annot_fontsize)

        fig.tight_layout()
        figures[setting] = fig

    return figures

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
