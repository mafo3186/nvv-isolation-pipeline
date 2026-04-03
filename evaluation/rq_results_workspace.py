from pathlib import Path
import pandas as pd

from config.path_factory import (
    get_rq_output_dir,
    get_pipeline_capability_summary_csv_path,
    get_global_combo_ranking_csv_path,
    get_global_best_k_union_summary_csv_path,
    get_rq2_audio_derivatives_csv_path,
    get_rq3_nvv_coverage_label_csv_path,
    get_rq3_nvv_coverage_global_csv_path,
    get_rq1_pipeline_capability_csv_path,
    get_rq2_config_ranking_single_csv_path,
    get_rq2_config_ranking_selected_set_csv_path,
    get_rq2_config_audio_derivatives_rq_csv_path,
    get_rq3_nvv_coverage_label_rq_csv_path,
    get_rq3_nvv_coverage_global_rq_csv_path,
    get_rq_results_xlsx_path,
)
from evaluation.eval_io import write_csv_atomic, load_csv_or_fail, validate_mode, write_multi_sheet_xlsx_atomic
from utils.io import ensure_dir


def _rq_output_dir(evaluation_dir: Path, mode: str) -> Path:
    """
    Return the RQ output directory and create it if needed.

    Args:
        evaluation_dir: Dataset evaluation root (workspace/global/evaluation).
        mode: Evaluation mode.

    Returns:
        Output directory path.
    """
    out_dir = get_rq_output_dir(Path(evaluation_dir), mode)
    ensure_dir(out_dir)
    return out_dir


def collect_rq1_pipeline_capability_result(
    evaluation_dir: Path,
    mode: str,
) -> pd.DataFrame:
    """
    Load the final pipeline capability summary artifact.

    Args:
        evaluation_dir: Dataset evaluation root (workspace/global/evaluation).
        mode: Evaluation mode ("full_gt" or "part_gt").

    Returns:
        DataFrame containing the pipeline capability summary.
    """
    validate_mode(mode)

    src = get_pipeline_capability_summary_csv_path(Path(evaluation_dir), mode)
    return load_csv_or_fail(src)


def collect_rq2a_config_ranking_result(
    evaluation_dir: Path,
    mode: str,
) -> pd.DataFrame:
    """
    Load the annotation configuration ranking artifact for RQ2a and add an
    explicit rank column.

    Args:
        evaluation_dir: Dataset evaluation root (workspace/global/evaluation).
        mode: Evaluation mode ("full_gt" or "part_gt").

    Returns:
        DataFrame containing the config ranking with rank_within_run.
    """
    validate_mode(mode)

    src = get_global_combo_ranking_csv_path(Path(evaluation_dir), mode)
    df = load_csv_or_fail(src).copy()

    df = df.reset_index(drop=True)
    df["rank_within_run"] = df.index + 1

    return df


def collect_rq2a_selected_set_result(
    evaluation_dir: Path,
    mode: str,
) -> pd.DataFrame:
    """
    Load the selected-set union result artifact (RQ2a) and add delta columns against
    the best single configuration for all shared macro_mean_* metrics.

    Args:
        evaluation_dir: Dataset evaluation root (workspace/global/evaluation).
        mode: Evaluation mode ("full_gt" or "part_gt").

    Returns:
        DataFrame containing the union result with delta columns.
    """
    validate_mode(mode)

    union_path = get_global_best_k_union_summary_csv_path(Path(evaluation_dir), mode)
    ranking_path = get_global_combo_ranking_csv_path(Path(evaluation_dir), mode)

    union_df = load_csv_or_fail(union_path).copy()
    ranking_df = load_csv_or_fail(ranking_path).copy()

    if union_df.shape[0] == 0:
        raise RuntimeError(f"Union artifact contains 0 rows: {union_path}")
    if ranking_df.shape[0] == 0:
        raise RuntimeError(f"Ranking artifact contains 0 rows: {ranking_path}")

    best_single_row = ranking_df.iloc[0]

    union_metric_cols = [c for c in union_df.columns if c.startswith("macro_mean_")]
    ranking_metric_cols = [c for c in ranking_df.columns if c.startswith("macro_mean_")]
    shared_metric_cols = sorted(set(union_metric_cols) & set(ranking_metric_cols))

    for metric_col in shared_metric_cols:
        union_value = pd.to_numeric(union_df[metric_col], errors="coerce")
        best_single_value = pd.to_numeric(
            pd.Series([best_single_row[metric_col]] * len(union_df)),
            errors="coerce",
        )
        union_df[f"delta_vs_best_single_{metric_col}"] = union_value - best_single_value

    return union_df


def collect_rq2b_audio_derivatives_result(
    evaluation_dir: Path,
    mode: str,
) -> pd.DataFrame:
    """
    Load the RQ2b audio derivative aggregation artifact.

    Args:
        evaluation_dir: Dataset evaluation root (workspace/global/evaluation).
        mode: Evaluation mode ("full_gt" or "part_gt").

    Returns:
        DataFrame containing the RQ2b audio derivative result.
    """
    validate_mode(mode)

    src = get_rq2_audio_derivatives_csv_path(Path(evaluation_dir), mode)
    return load_csv_or_fail(src)


def collect_rq3_nvv_coverage_label_result(
    evaluation_dir: Path,
    mode: str,
) -> pd.DataFrame:
    """
    Load the RQ3 NVV label/event coverage artifact.

    Args:
        evaluation_dir: Dataset evaluation root (workspace/global/evaluation).
        mode: Evaluation mode ("full_gt" or "part_gt").

    Returns:
        DataFrame containing the RQ3 label/event coverage result.
    """
    validate_mode(mode)

    src = get_rq3_nvv_coverage_label_csv_path(Path(evaluation_dir), mode)
    return load_csv_or_fail(src)


def collect_rq3_nvv_coverage_global_result(
    evaluation_dir: Path,
    mode: str,
) -> pd.DataFrame:
    """
    Load the RQ3 NVV global coverage artifact.

    Args:
        evaluation_dir: Dataset evaluation root (workspace/global/evaluation).
        mode: Evaluation mode ("full_gt" or "part_gt").

    Returns:
        DataFrame containing the RQ3 global coverage result.
    """
    validate_mode(mode)

    src = get_rq3_nvv_coverage_global_csv_path(Path(evaluation_dir), mode)
    return load_csv_or_fail(src)


def _build_rq1_comparison(
    ranking_df: pd.DataFrame,
    cap_summary_df: pd.DataFrame,
    mode: str,
) -> pd.DataFrame:
    """
    Build the canonical RQ1 pipeline capability comparison table.

    Combines three system rows:
      - baseline:          vad_mask == 'original' and asr_audio_in == 'original'
      - best_single:       top-ranked row in the single-config ranking
      - best_selected_set: union of the selected set from capability summary

    Args:
        ranking_df: Global combo ranking DataFrame (sorted by primary metric).
        cap_summary_df: Pipeline capability summary DataFrame.
        mode: Evaluation mode.

    Returns:
        Comparison DataFrame with a 'system' identifier column.

    Raises:
        RuntimeError: If no baseline row is found in the ranking.
    """
    ranking_metric_cols = {c for c in ranking_df.columns if c.startswith("macro_mean_")}
    cap_metric_cols = {c for c in cap_summary_df.columns if c.startswith("macro_mean_")}
    metric_cols = sorted(ranking_metric_cols | cap_metric_cols)

    # Baseline: identify by vad_mask and asr_audio_in, not by combo_key
    baseline_mask = (
        (ranking_df["vad_mask"].astype(str) == "original")
        & (ranking_df["asr_audio_in"].astype(str) == "original")
    )
    baseline_rows = ranking_df[baseline_mask]
    if baseline_rows.empty:
        raise RuntimeError(
            "No baseline row found in ranking "
            "(vad_mask='original', asr_audio_in='original')."
        )
    baseline_row = baseline_rows.iloc[0]

    best_single_row = ranking_df.iloc[0]
    cap_row = cap_summary_df.iloc[0]

    rows = []

    for system, row, is_ranking_row in [
        ("baseline", baseline_row, True),
        ("best_single", best_single_row, True),
        ("best_selected_set", cap_row, False),
    ]:
        entry: dict = {"system": system, "mode": mode}

        if is_ranking_row:
            entry["combo_key"] = str(row.get("combo_key", ""))
            entry["vad_mask"] = str(row.get("vad_mask", ""))
            entry["asr_audio_in"] = str(row.get("asr_audio_in", ""))
            entry["best_k"] = float("nan")
            entry["n_selected_tracks"] = float("nan")
            entry["selected_set_json"] = ""
        else:
            entry["combo_key"] = ""
            entry["vad_mask"] = ""
            entry["asr_audio_in"] = ""
            entry["best_k"] = row.get("best_k", float("nan"))
            entry["n_selected_tracks"] = row.get("n_selected_tracks", float("nan"))
            entry["selected_set_json"] = str(row.get("selected_set_json", ""))

        for col in metric_cols:
            entry[col] = row.get(col, float("nan"))

        rows.append(entry)

    return pd.DataFrame(rows)


def collect_rq1_canonical_comparison_result(
    evaluation_dir: Path,
    mode: str,
) -> pd.DataFrame:
    """
    Build the canonical RQ1 pipeline capability comparison artifact.

    Compares baseline, best_single, and best_selected_set in one table.

    Sources:
      - baseline and best_single: from global_combo_ranking_<mode>.csv
      - best_selected_set: from pipeline_capability_<mode>_summary.csv

    Args:
        evaluation_dir: Dataset evaluation root (workspace/global/evaluation).
        mode: Evaluation mode ("full_gt" or "part_gt").

    Returns:
        Comparison DataFrame with 'system' column identifying each row.
    """
    validate_mode(mode)

    ranking_df = load_csv_or_fail(get_global_combo_ranking_csv_path(Path(evaluation_dir), mode)).copy()
    cap_df = load_csv_or_fail(get_pipeline_capability_summary_csv_path(Path(evaluation_dir), mode)).copy()

    return _build_rq1_comparison(ranking_df=ranking_df, cap_summary_df=cap_df, mode=mode)


def collect_rq_results_from_artifacts(
    evaluation_dir: Path,
    mode: str,
) -> dict[str, pd.DataFrame]:
    """
    Collect all RQ result tables for one dataset/mode.

    Returns a dict with keys matching the final RQ structure:
      rq1               - canonical comparison (baseline / best_single / best_selected_set)
      rq2a_single       - single-configuration ranking
      rq2a_selected_set - selected-set / union analysis
      rq2b              - audio derivative aggregation
      rq3_label         - NVV label/event coverage analysis
      rq3_global        - NVV global coverage analysis

    Args:
        evaluation_dir: Dataset evaluation root (workspace/global/evaluation).
        mode: Evaluation mode ("full_gt" or "part_gt").

    Returns:
        Dict with RQ result DataFrames.
    """
    validate_mode(mode)

    return {
        "rq1": collect_rq1_canonical_comparison_result(evaluation_dir=evaluation_dir, mode=mode),
        "rq2a_single": collect_rq2a_config_ranking_result(evaluation_dir=evaluation_dir, mode=mode),
        "rq2a_selected_set": collect_rq2a_selected_set_result(evaluation_dir=evaluation_dir, mode=mode),
        "rq2b": collect_rq2b_audio_derivatives_result(evaluation_dir=evaluation_dir, mode=mode),
        "rq3_label": collect_rq3_nvv_coverage_label_result(evaluation_dir=evaluation_dir, mode=mode),
        "rq3_global": collect_rq3_nvv_coverage_global_result(evaluation_dir=evaluation_dir, mode=mode),
    }


def write_rq_results(
    evaluation_dir: Path,
    mode: str,
    results: dict[str, pd.DataFrame],
) -> dict[str, Path]:
    """
    Write RQ result tables to evaluation/<mode>/research_questions/.

    Canonical artifact names:
      rq1_pipeline_capability_<mode>.csv
      rq2_config_ranking_single_<mode>.csv
      rq2_config_ranking_selected_set_<mode>.csv
      rq2_config_audio_derivatives_<mode>.csv
      rq3_nvv_coverage_label_<mode>.csv
      rq3_nvv_coverage_global_<mode>.csv

    Args:
        evaluation_dir: Dataset evaluation root (workspace/global/evaluation).
        mode: Evaluation mode ("full_gt" or "part_gt").
        results: Dict returned by collect_rq_results_from_artifacts().

    Returns:
        Dict mapping result keys to written file paths.
    """
    validate_mode(mode)

    out_dir = _rq_output_dir(evaluation_dir, mode)

    mapping = {
        "rq1": get_rq1_pipeline_capability_csv_path(Path(evaluation_dir), mode),
        "rq2a_single": get_rq2_config_ranking_single_csv_path(Path(evaluation_dir), mode),
        "rq2a_selected_set": get_rq2_config_ranking_selected_set_csv_path(Path(evaluation_dir), mode),
        "rq2b": get_rq2_config_audio_derivatives_rq_csv_path(Path(evaluation_dir), mode),
        "rq3_label": get_rq3_nvv_coverage_label_rq_csv_path(Path(evaluation_dir), mode),
        "rq3_global": get_rq3_nvv_coverage_global_rq_csv_path(Path(evaluation_dir), mode),
    }

    written: dict[str, Path] = {}

    for key, out_path in mapping.items():
        if key not in results:
            raise KeyError(f"Missing RQ result '{key}' in results dict.")
        write_csv_atomic(results[key], out_path)
        written[key] = out_path

    xlsx_path = get_rq_results_xlsx_path(Path(evaluation_dir), mode)

    write_multi_sheet_xlsx_atomic(
        sheets={
            "rq1": results["rq1"],
            "rq2a_single": results["rq2a_single"],
            "rq2a_selected_set": results["rq2a_selected_set"],
            "rq2b": results["rq2b"],
            "rq3_label": results["rq3_label"],
            "rq3_global": results["rq3_global"],
        },
        out_path=xlsx_path,
    )
    written["xlsx"] = xlsx_path
 

    return written