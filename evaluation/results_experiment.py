from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from config.path_factory import (
    get_global_evaluation_dir,
    get_experiment_rq_output_dir,
    get_run_json_path,
    get_experiment_rq1_csv_path,
    get_experiment_rq2a_single_csv_path,
    get_experiment_rq2a_selected_set_csv_path,
    get_experiment_rq2b_csv_path,
    get_experiment_rq3_csv_path,
)
from evaluation.eval_io import write_csv_atomic
from evaluation.rq_results_workspace import collect_rq_results_from_artifacts
from evaluation.eval_io import validate_mode
from utils.io import ensure_dir, load_yaml


#toDo: Refactor to use utils/io.py read_json() and load_json() instead of duplicating JSON loading logic here.
def _load_json(path: Path) -> Any:
    """
    Load a JSON file.

    Args:
        path: JSON path.

    Returns:
        Parsed JSON object.
    """
    if not path.exists():
        raise FileNotFoundError(f"Required JSON file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _get_nested_value(data: dict[str, Any], dotted_key: str) -> Any:
    """
    Resolve a dotted key path inside a nested dict.

    Args:
        data: Nested dict.
        dotted_key: Dot-notation key path.

    Returns:
        Resolved value.
    """
    current: Any = data
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(f"Missing key '{dotted_key}' in nested structure.")
        current = current[part]
    return current


def _grid_keys_and_aliases(grid_keys: list[str]) -> list[tuple[str, str]]:
    """
    Build short aliases for grid keys.

    Args:
        grid_keys: Dot-notation grid keys.

    Returns:
        List of (grid_key, alias) tuples.
    """
    aliases: list[tuple[str, str]] = []
    for key in grid_keys:
        alias = key.split(".")[-1]
        aliases.append((key, alias))
    return aliases


def _find_resolved_config_paths(experiment_root: Path) -> list[Path]:
    """
    Find resolved config YAML files directly below the experiment root.

    Args:
        experiment_root: Experiment root directory.

    Returns:
        Sorted list of resolved config YAML paths.
    """
    config_paths = sorted(experiment_root.rglob("*_config.yaml"))
    if not config_paths:
        raise FileNotFoundError(
            f"No resolved config files ('*_config.yaml') found in {experiment_root}"
        )
    return config_paths


def _extract_run_metadata(
    resolved_cfg: dict[str, Any],
    run_json: dict[str, Any],
    key_aliases: list[tuple[str, str]],
) -> dict[str, Any]:
    """
    Extract run metadata and varied parameter values.

    Args:
        resolved_cfg: Parsed resolved config YAML.
        run_json: Parsed run.json content.
        key_aliases: List of (grid_key, alias) tuples.

    Returns:
        Flat metadata dict.
    """
    meta = {
        "run_id": run_json.get("run_id"),
        "dataset_name": run_json.get("dataset_name"),
        "output_rel": run_json.get("output_rel"),
        "created_at": run_json.get("created_at"),
    }

    for dotted_key, alias in key_aliases:
        meta[alias] = _get_nested_value(resolved_cfg, dotted_key)

    return meta


def _prepend_metadata(df: pd.DataFrame, meta: dict[str, Any]) -> pd.DataFrame:
    """
    Prepend metadata columns to a result DataFrame.

    Args:
        df: Source DataFrame.
        meta: Flat metadata dict.

    Returns:
        DataFrame with metadata inserted first.
    """
    result = df.copy()

    for key, value in reversed(list(meta.items())):
        result.insert(0, key, value)

    return result


def _experiment_output_dir(experiment_root: Path, mode: str) -> Path:
    """
    Return the experiment-level output directory for research question results.

    Args:
        experiment_root: Experiment root.
        mode: Evaluation mode.

    Returns:
        Output directory path.
    """
    out_dir = get_experiment_rq_output_dir(Path(experiment_root), mode)
    ensure_dir(out_dir)
    return out_dir


def get_results_experiment(
    experiment_root: Path,
    grid_keys: list[str],
    mode: str,
    top_k_rq2a_per_run: int = 3,
) -> dict[str, pd.DataFrame]:
    """
    Collect experiment-level RQ result tables across all runs.

    Args:
        experiment_root: Experiment root directory.
        grid_keys: Dot-notation keys varied in the experiment.
        mode: Evaluation mode ("full_gt" or "part_gt").
        top_k_rq2a_per_run: Number of top-ranked RQ2a single rows kept per run.

    Returns:
        Dict with experiment-level RQ result tables.
    """
    validate_mode(mode)

    experiment_root = Path(experiment_root)
    processed_root = experiment_root.parent

    resolved_cfg_paths = _find_resolved_config_paths(experiment_root)
    key_aliases = _grid_keys_and_aliases(grid_keys)

    rq1_frames: list[pd.DataFrame] = []
    rq2a_single_frames: list[pd.DataFrame] = []
    rq2a_selected_set_frames: list[pd.DataFrame] = []
    rq2b_frames: list[pd.DataFrame] = []
    rq3_frames: list[pd.DataFrame] = []

    for resolved_cfg_path in resolved_cfg_paths:
        resolved_cfg = load_yaml(resolved_cfg_path)

        datasets = resolved_cfg.get("workspace", {}).get("datasets", [])
        if not datasets:
            raise ValueError(f"No workspace.datasets found in {resolved_cfg_path}")

        for ds_entry in datasets:
            output_rel = ds_entry.get("output_rel")
            if not output_rel:
                raise KeyError(f"Missing workspace.datasets[].output_rel in {resolved_cfg_path}")

            run_dir = processed_root / output_rel
            if not run_dir.exists():
                raise FileNotFoundError(f"Run directory not found: {run_dir}")

            run_json = _load_json(get_run_json_path(run_dir))
            meta = _extract_run_metadata(
                resolved_cfg=resolved_cfg,
                run_json=run_json,
                key_aliases=key_aliases,
            )

            evaluation_dir = get_global_evaluation_dir(run_dir)
            rq_results = collect_rq_results_from_artifacts(
                evaluation_dir=evaluation_dir,
                mode=mode,
            )

            rq1_df = _prepend_metadata(rq_results["rq1"], meta)
            rq2a_selected_set_df = _prepend_metadata(rq_results["rq2a_selected_set"], meta)

            rq2a_single_df = rq_results["rq2a_single"].copy()
            if "rank_within_run" not in rq2a_single_df.columns:
                raise KeyError("Expected column 'rank_within_run' in RQ2a single result.")
            if top_k_rq2a_per_run is not None:
                rq2a_single_df = rq2a_single_df[
                    rq2a_single_df["rank_within_run"] <= top_k_rq2a_per_run
                ]

            rq2a_single_df = _prepend_metadata(rq2a_single_df, meta)
            rq2b_df = _prepend_metadata(rq_results["rq2b"], meta)
            rq3_df = _prepend_metadata(rq_results["rq3"], meta)

            rq1_frames.append(rq1_df)
            rq2a_single_frames.append(rq2a_single_df)
            rq2a_selected_set_frames.append(rq2a_selected_set_df)
            rq2b_frames.append(rq2b_df)
            rq3_frames.append(rq3_df)

    return {
        "rq1": pd.concat(rq1_frames, ignore_index=True),
        "rq2a_single": pd.concat(rq2a_single_frames, ignore_index=True),
        "rq2a_selected_set": pd.concat(rq2a_selected_set_frames, ignore_index=True),
        "rq2b": pd.concat(rq2b_frames, ignore_index=True),
        "rq3": pd.concat(rq3_frames, ignore_index=True),
    }


def write_results_experiment(
    experiment_root: Path,
    mode: str,
    results: dict[str, pd.DataFrame],
) -> dict[str, Path]:
    """
    Write experiment-level RQ result tables.

    Args:
        experiment_root: Experiment root.
        mode: Evaluation mode.
        results: Dict returned by get_results_experiment().

    Returns:
        Dict mapping result keys to written file paths.
    """
    validate_mode(mode)
    

    out_dir = _experiment_output_dir(experiment_root, mode)

    mapping = {
        "rq1": get_experiment_rq1_csv_path(Path(experiment_root), mode),
        "rq2a_single": get_experiment_rq2a_single_csv_path(Path(experiment_root), mode),
        "rq2a_selected_set": get_experiment_rq2a_selected_set_csv_path(Path(experiment_root), mode),
        "rq2b": get_experiment_rq2b_csv_path(Path(experiment_root), mode),
        "rq3": get_experiment_rq3_csv_path(Path(experiment_root), mode),
    }

    written: dict[str, Path] = {}

    for key, out_path in mapping.items():
        if key not in results:
            raise KeyError(f"Missing experiment result '{key}' in results dict.")
        write_csv_atomic(results[key], out_path)
        written[key] = out_path

    return written


def run_results_experiment(
    experiment_root: Path,
    grid_keys: list[str],
    mode: str,
    top_k_rq2a_per_run: int | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Build and write experiment-level RQ result tables.

    Args:
        experiment_root: Experiment root directory.
        grid_keys: Dot-notation keys varied in the experiment.
        mode: Evaluation mode ("full_gt" or "part_gt").
        top_k_rq2a_per_run: Number of top-ranked RQ2a single rows kept per run.
            If None, all rows are kept.

    Returns:
        Dict with experiment-level RQ result DataFrames.
    """
    results = get_results_experiment(
        experiment_root=experiment_root,
        grid_keys=grid_keys,
        mode=mode,
        top_k_rq2a_per_run=top_k_rq2a_per_run,
    )

    written = write_results_experiment(
        experiment_root=experiment_root,
        mode=mode,
        results=results,
    )

    print("\nWritten experiment RQ artifacts:")
    for key, path in written.items():
        print(f"  {key}: {path}")

    return results