from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from config.path_factory import get_global_f1_vs_k_csv_path

def plot_curves_for_dataset(
        dataset_name: str,
        mode: str,
        ws_by_dataset: dict[str, Path],
        num_cols_list: Optional[list[str]] = ["k", "macro_mean_f1", "macro_mean_recall", "macro_mean_precision", "macro_mean_fp"]
    ) -> None:
    """
    Plot greedy-forward-selection curves (F1/Recall/Precision/FP vs k) for one dataset.

    Arguments:
        dataset_name: Name of the dataset in eval_datasets / ws_by_dataset.
        num_cols_list: List of column names to plot.
    """
    if mode != "full_gt":
        print("Skipping curve plotting (only runs for full_gt).")
        return
    else:
    
        if dataset_name not in ws_by_dataset:
            raise KeyError(f"Dataset '{dataset_name}' not found in ws_by_dataset.")

        ws = ws_by_dataset[dataset_name]

        csv_path = get_global_f1_vs_k_csv_path(ws.evaluation, mode)
        if not csv_path.exists():
            raise FileNotFoundError(f"[{dataset_name}] Missing file: {csv_path}")

        df = pd.read_csv(csv_path)

        # Coerce numerics
        for c in num_cols_list:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # best_k (optional)
        best_k = None
        if "best_k" in df.columns and df["best_k"].notna().any():
            try:
                best_k = int(pd.to_numeric(df["best_k"].dropna().iloc[0], errors="coerce"))
            except Exception:
                best_k = None

        def plot_curve(y_col: str, title: str, y_label: str) -> None:
            """
            Plot a metric curve over k from the greedy forward selection CSV.

            Arguments:
                y_col: Column name in df to plot on y-axis.
                title: Plot title.
                y_label: Y-axis label.
            """
            if y_col not in df.columns:
                raise KeyError(
                    f"[{dataset_name}] Column '{y_col}' not found in df. Available: {list(df.columns)}"
                )

            plt.figure()
            plt.plot(df["k"], df[y_col])
            plt.xlabel("k")
            plt.ylabel(y_label)
            plt.title(f"{dataset_name} — {title}")

            if best_k is not None:
                row = df.loc[df["k"] == best_k]
                if not row.empty:
                    x = float(row["k"].iloc[0])
                    y = float(row[y_col].iloc[0])
                    plt.scatter([x], [y])
                    plt.text(x, y, f" best_k={best_k}", va="bottom", ha="left")

            plt.show()

        plot_curve("macro_mean_f1", "Greedy forward selection: F1 vs k", "macro_mean_f1")
        plot_curve("macro_mean_recall", "Greedy forward selection: Recall vs k", "macro_mean_recall")
        plot_curve("macro_mean_precision", "Greedy forward selection: Precision vs k", "macro_mean_precision")
        plot_curve("macro_mean_fp", "Greedy forward selection: FP vs k (macro mean)", "macro_mean_fp")