from __future__ import annotations

from math import ceil
from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt


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


def _prepare_axes(n_panels: int, ncols: int = 2):
    """
    Create subplot grid.

    Args:
        n_panels: Number of panels.
        ncols: Number of columns.

    Returns:
        (fig, axes_list)
    """
    nrows = ceil(n_panels / ncols)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(6 * ncols, 4 * nrows),
        squeeze=False,
    )
    flat_axes = list(axes.ravel())

    for ax in flat_axes[n_panels:]:
        ax.set_visible(False)

    return fig, flat_axes[:n_panels]


def plot_setting_overview(
    setting_overview: pd.DataFrame,
    *,
    score_col: str = "best_score",
    title: Optional[str] = None,
):
    """
    Plot a compact bar chart of setting-level scores.

    Args:
        setting_overview: Setting overview table.
        score_col: Score column to plot.
        title: Optional figure title.

    Returns:
        Matplotlib figure.
    """
    _require_columns(setting_overview, ["setting", score_col], label="setting overview")

    df = setting_overview.copy().sort_values(by=score_col, ascending=False)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(df["setting"], df[score_col])

    ax.set_xlabel("Setting")
    ax.set_ylabel(score_col)
    ax.set_title(title or f"Setting overview: {score_col}")
    ax.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    return fig


def plot_parameter_value_comparison(
    parameter_value_details: pd.DataFrame,
    *,
    parameter: str,
    value_col: str = "mean_primary_score",
    panel_by: str = "setting",
    title: Optional[str] = None,
):
    """
    Plot parameter-value comparison as a small multiple chart.

    Args:
        parameter_value_details: Detailed parameter table from collect_comparison_views().
        parameter: Parameter name to plot.
        value_col: Metric column to plot.
        panel_by: Facet column. Usually 'setting', 'dataset_name', or 'mode'.
        title: Optional figure title.

    Returns:
        Matplotlib figure.
    """
    _require_columns(
        parameter_value_details,
        ["parameter", panel_by, "param_value_text", value_col],
        label="parameter value details",
    )

    df = parameter_value_details[parameter_value_details["parameter"] == parameter].copy()
    if df.empty:
        raise ValueError(f"No rows found for parameter '{parameter}'.")

    panels = list(df[panel_by].drop_duplicates())
    fig, axes = _prepare_axes(len(panels), ncols=2)

    for ax, panel_value in zip(axes, panels):
        part = df[df[panel_by] == panel_value].copy()
        part = part.sort_values(by="param_value_text")
        ax.bar(part["param_value_text"], part[value_col])

        ax.set_title(str(panel_value))
        ax.set_xlabel(parameter)
        ax.set_ylabel(value_col)
        ax.tick_params(axis="x", rotation=45)

    if title is not None:
        fig.suptitle(title)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
    else:
        fig.tight_layout()

    return fig


def plot_parameter_pair_heatmaps(
    pair_value_details: pd.DataFrame,
    *,
    pair_key: str,
    value_col: str = "mean_primary_score",
    panel_by: str = "setting",
    title: Optional[str] = None,
):
    """
    Plot heatmaps for a parameter pair across settings or other facets.

    Args:
        pair_value_details: Detailed pair table from collect_comparison_views().
        pair_key: Pair identifier, e.g. 'vad_threshold__vad_min_silence_ms'.
        value_col: Cell value column.
        panel_by: Facet column. Usually 'setting', 'dataset_name', or 'mode'.
        title: Optional figure title.

    Returns:
        Matplotlib figure.
    """
    _require_columns(
        pair_value_details,
        ["pair_key", panel_by, "param_x_text", "param_y_text", value_col],
        label="pair value details",
    )

    df = pair_value_details[pair_value_details["pair_key"] == pair_key].copy()
    if df.empty:
        raise ValueError(f"No rows found for pair_key '{pair_key}'.")

    panels = list(df[panel_by].drop_duplicates())
    fig, axes = _prepare_axes(len(panels), ncols=2)

    for ax, panel_value in zip(axes, panels):
        part = df[df[panel_by] == panel_value].copy()

        x_labels = sorted(part["param_x_text"].drop_duplicates().tolist())
        y_labels = sorted(part["param_y_text"].drop_duplicates().tolist())

        pivot = (
            part
            .pivot_table(
                index="param_y_text",
                columns="param_x_text",
                values=value_col,
                aggfunc="mean",
            )
            .reindex(index=y_labels, columns=x_labels)
        )

        img = ax.imshow(pivot.values, aspect="auto")
        ax.set_title(str(panel_value))
        ax.set_xlabel(pair_key.split("__")[0])
        ax.set_ylabel(pair_key.split("__")[1])
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=45)
        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels)

        for row_idx in range(pivot.shape[0]):
            for col_idx in range(pivot.shape[1]):
                value = pivot.iloc[row_idx, col_idx]
                if pd.notna(value):
                    ax.text(
                        col_idx,
                        row_idx,
                        f"{value:.3f}",
                        ha="center",
                        va="center",
                    )

        fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)

    if title is not None:
        fig.suptitle(title)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
    else:
        fig.tight_layout()

    return fig


def plot_derivative_group_comparison(
    derivative_comparison: pd.DataFrame,
    *,
    value_col: str,
    panel_by: str = "setting",
    title: Optional[str] = None,
):
    """
    Plot derivative-group comparison as small multiples.

    Args:
        derivative_comparison: Derivative comparison table.
        value_col: Metric column to plot.
        panel_by: Facet column. Usually 'setting', 'dataset_name', or 'mode'.
        title: Optional figure title.

    Returns:
        Matplotlib figure.
    """
    _require_columns(
        derivative_comparison,
        [panel_by, "audio_derivative_group", value_col],
        label="derivative comparison",
    )

    df = derivative_comparison.copy()
    panels = list(df[panel_by].drop_duplicates())
    fig, axes = _prepare_axes(len(panels), ncols=2)

    for ax, panel_value in zip(axes, panels):
        part = df[df[panel_by] == panel_value].copy()
        part = part.sort_values(by=value_col, ascending=False)

        ax.bar(part["audio_derivative_group"], part[value_col])
        ax.set_title(str(panel_value))
        ax.set_xlabel("audio_derivative_group")
        ax.set_ylabel(value_col)
        ax.tick_params(axis="x", rotation=45)

    if title is not None:
        fig.suptitle(title)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
    else:
        fig.tight_layout()

    return fig


def plot_four_panel_parameter_summary(
    parameter_value_details: pd.DataFrame,
    *,
    parameter: str,
    value_col: str = "mean_primary_score",
    title: Optional[str] = None,
):
    """
    Convenience wrapper for a 2x2 parameter plot across four settings.

    Args:
        parameter_value_details: Detailed parameter table.
        parameter: Parameter name to plot.
        value_col: Metric column to plot.
        title: Optional figure title.

    Returns:
        Matplotlib figure.
    """
    return plot_parameter_value_comparison(
        parameter_value_details,
        parameter=parameter,
        value_col=value_col,
        panel_by="setting",
        title=title or f"{parameter} across settings",
    )

def plot_top_k_runs_per_setting(
    per_setting: dict[str, dict[str, pd.DataFrame]],
    *,
    top_k: int = 10,
    show_param_text: bool = True,
    title: Optional[str] = None,
):
    """
    Plot the top-k RQ1 runs for each setting as small multiples.

    Each panel uses the setting-specific primary metric:
    - full_gt: macro_mean_f1
    - part_gt: macro_mean_recall

    Args:
        per_setting: Mapping returned in views["per_setting"] by
            collect_comparison_views().
        top_k: Number of top-ranked runs to display per setting.
        show_param_text: Whether to annotate bars with compact parameter text.
        title: Optional figure title.

    Returns:
        Matplotlib figure.
    """
    if not per_setting:
        raise ValueError("per_setting is empty.")

    setting_items = list(per_setting.items())
    fig, axes = _prepare_axes(len(setting_items), ncols=2)

    for ax, (setting_name, setting_dict) in zip(axes, setting_items):
        if "top_runs" not in setting_dict:
            raise KeyError(f"Missing 'top_runs' for setting '{setting_name}'.")

        top_runs = setting_dict["top_runs"].copy()
        if top_runs.empty:
            ax.set_title(setting_name)
            ax.text(0.5, 0.5, "No runs available", ha="center", va="center")
            ax.set_axis_off()
            continue

        plot_df = top_runs.head(top_k).copy()

        if "mode" not in plot_df.columns:
            raise KeyError(f"Missing 'mode' column in top_runs for '{setting_name}'.")

        mode = str(plot_df["mode"].iloc[0])
        if mode == "full_gt":
            score_col = "macro_mean_f1"
        elif mode == "part_gt":
            score_col = "macro_mean_recall"
        else:
            raise ValueError(f"Unsupported mode '{mode}' in setting '{setting_name}'.")

        if score_col not in plot_df.columns:
            raise KeyError(
                f"Missing score column '{score_col}' in top_runs for '{setting_name}'."
            )

        plot_df = plot_df.reset_index(drop=True)
        plot_df["rank"] = range(1, len(plot_df) + 1)

        ax.bar(plot_df["rank"].astype(str), plot_df[score_col])

        ax.set_title(f"{setting_name}\nmetric: {score_col}")
        ax.set_xlabel("Run rank")
        ax.set_ylabel(score_col)

        if show_param_text:
            for _, row in plot_df.iterrows():
                param_text = (
                    f"thr={row.get('vad_threshold', 'NA')}\n"
                    f"sil={row.get('vad_min_silence_ms', 'NA')}\n"
                    f"max={row.get('max_duration', 'NA')}\n"
                    f"dedup={row.get('dedup_overlap_ratio', 'NA')}"
                )

                ax.text(
                    row["rank"] - 1,  # bar center in categorical order
                    row[score_col],
                    param_text,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    if title is not None:
        fig.suptitle(title)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
    else:
        fig.tight_layout()

    return fig