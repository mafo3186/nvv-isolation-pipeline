from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, List

from config.path_factory import (
    get_datasets,
    get_workspace_paths,
    get_experiment_run_root,
    get_rq1_pipeline_capability_csv_path,
    get_rq2_config_ranking_single_csv_path,
    get_rq2_config_ranking_selected_set_csv_path,
    get_rq2_config_audio_derivatives_rq_csv_path,
    get_rq3_nvv_coverage_label_rq_csv_path,
    get_rq3_nvv_coverage_global_rq_csv_path,
)
from evaluation.results_experiment import get_results_experiment
from utils.io import load_yaml
from evaluation.eval_io import validate_mode, load_csv_or_fail

from evaluation.analysis_tables import (
    collect_comparison_views,
    combine_comparison_views,
)

def build_specs_from_config(
    *,
    cfg_path: Path,
    modes: List[str],
) -> List[dict]:
    """
    Build analysis specs from a config YAML.

    Each dataset defined in workspace.datasets will be expanded
    into one spec per requested mode.

    Args:
        cfg_path: Path to config YAML.
        modes: List of modes, e.g. ["full_gt", "part_gt"].

    Returns:
        List of spec dicts compatible with load_and_compare_workspaces().
    """
    cfg = load_yaml(cfg_path)

    datasets = cfg.get("workspace", {}).get("datasets", [])
    if not datasets:
        raise ValueError(f"No workspace.datasets found in {cfg_path}")

    specs = []

    for ds in datasets:
        dataset_name = ds.get("name")
        if not dataset_name:
            raise KeyError(f"Missing dataset name in {cfg_path}")

        for mode in modes:
            validate_mode(mode)

            specs.append(
                {
                    "label": f"{dataset_name} | {mode}",
                    "cfg_path": Path(cfg_path),
                    "dataset_name": dataset_name,
                    "mode": mode,
                }
            )

    return specs


def load_rq_results(
    *,
    evaluation_dir: Path,
    mode: str,
) -> dict[str, Any]:
    """
    Load final research-question result tables from research_questions/.

    This loader treats the exported RQ CSV files as the canonical interface
    for downstream notebook analysis.

    Args:
        evaluation_dir: Dataset evaluation root (workspace/global/evaluation).
        mode: Evaluation mode.

    Returns:
        Dict with the final RQ result DataFrames.
    """
    validate_mode(mode)
    evaluation_dir = Path(evaluation_dir)

    paths = {
        "rq1": get_rq1_pipeline_capability_csv_path(evaluation_dir, mode),
        "rq2a_single": get_rq2_config_ranking_single_csv_path(evaluation_dir, mode),
        "rq2a_selected_set": get_rq2_config_ranking_selected_set_csv_path(evaluation_dir, mode),
        "rq2b": get_rq2_config_audio_derivatives_rq_csv_path(evaluation_dir, mode),
        "rq3_label": get_rq3_nvv_coverage_label_rq_csv_path(evaluation_dir, mode),
        "rq3_global": get_rq3_nvv_coverage_global_rq_csv_path(evaluation_dir, mode),
    }

    return {
        key: load_csv_or_fail(path)
        for key, path in paths.items()
    }

# --- Workspace Result Loader ---

def load_workspace_results_from_config(
    *,
    cfg_path: Path,
    mode: str,
    dataset_name: Optional[str] = None,
) -> dict[str, Any]:
    """
    Load workspace-level RQ result tables from a config YAML.

    Args:
        cfg_path: Config YAML path.
        mode: Evaluation mode.
        dataset_name: Optional dataset filter. If None, all datasets are loaded.

    Returns:
        Bundle with per-dataset workspace results.
    """
    validate_mode(mode)

    datasets = get_datasets(cfg_path)
    if dataset_name is not None:
        datasets = [ds for ds in datasets if ds.name == dataset_name]

    if not datasets:
        raise ValueError("No matching datasets found for workspace loading.")

    results_by_dataset: dict[str, dict[str, Any]] = {}

    for ds in datasets:
        ws_paths = get_workspace_paths(ds.workspace)
        evaluation_dir = ws_paths.evaluation

        results_by_dataset[ds.name] = {
            "dataset_name": ds.name,
            "workspace": ds.workspace,
            "evaluation_dir": evaluation_dir,
            "results": load_rq_results(
                evaluation_dir=evaluation_dir,
                mode=mode,
            ),
        }

    return {
        "scope": "workspace",
        "mode": mode,
        "cfg_path": Path(cfg_path),
        "results_by_dataset": results_by_dataset,
    }


def load_and_compare_workspaces(
    *,
    specs: List[dict],
    score_fraction: float = 0.95,
    top_n_runs: int = 10,
    param_names: Optional[list[str]] = None,
    param_pairs: Optional[list[tuple[str, str]]] = None,
    combo_top_n: int = 10,
) -> dict[str, Any]:
    """
    Load multiple workspace runs (no experiment context),
    build comparison views, and combine them.

    Args:
        specs: List of spec dicts:
            {
                "label": str,
                "cfg_path": Path,
                "dataset_name": str,
                "mode": str,
            }

    Returns:
        Dict with:
            - bundles_by_spec
            - views_by_spec
            - combined_views
    """
    bundles_by_spec: dict[str, dict[str, Any]] = {}
    views_by_spec: dict[str, dict[str, Any]] = {}

    setting_order = [
        f"{str(spec['dataset_name'])} | {str(spec['mode'])}"
        for spec in specs
    ]

    for spec in specs:
        label = str(spec["label"])
        cfg_path = Path(spec["cfg_path"])
        dataset_name = str(spec["dataset_name"])
        mode = str(spec["mode"])

        validate_mode(mode)

        # --- Load workspace results ---
        bundle = load_workspace_results_from_config(
            cfg_path=cfg_path,
            mode=mode,
            dataset_name=dataset_name,
        )

        results = bundle["results_by_dataset"][dataset_name]["results"]

        setting = f"{dataset_name} | {mode}"

        results_with_meta = {}
        for key, df in results.items():
            df_copy = df.copy()

            # Insert dataset_name if missing
            if "dataset_name" not in df_copy.columns:
                df_copy.insert(0, "dataset_name", dataset_name)

            # Insert mode if missing (defensive)
            if "mode" not in df_copy.columns:
                df_copy.insert(1, "mode", mode)

            # Insert setting (useful for plots later)
            if "setting" not in df_copy.columns:
                df_copy.insert(0, "setting", setting)

            results_with_meta[key] = df_copy

        # --- Build comparison views ---
        views = collect_comparison_views(
            results=results_with_meta,
            dataset_names=[dataset_name],
            modes=[mode],
            score_fraction=score_fraction,
            top_n_runs=top_n_runs,
            param_names=param_names,
            param_pairs=param_pairs,
            combo_top_n=combo_top_n,
        )

        bundles_by_spec[label] = {
            "bundle": bundle,
            "results": results_with_meta,
        }

        views_by_spec[label] = views

    combined_views = combine_comparison_views(views_by_spec)

    return {
        "bundles_by_spec": bundles_by_spec,
        "views_by_spec": views_by_spec,
        "combined_views": combined_views,
        "setting_order": setting_order,
    }


# --- Experiment Result Loader ---

def load_experiment_results_from_yaml(
    *,
    cfg_path: Path,
    experiment_yaml_path: Path,
    mode: str,
    top_k_rq2a_per_run: Optional[int] = None,
) -> dict[str, Any]:
    """
    Load experiment-level RQ result tables from an experiment YAML.

    Args:
        cfg_path: Base config YAML path.
        experiment_yaml_path: Experiment definition YAML path.
        mode: Evaluation mode.
        top_k_rq2a_per_run: Optional top-k filter for RQ2a.
            None keeps all rows.

    Returns:
        Bundle with experiment metadata and result DataFrames.
    """
    validate_mode(mode)

    exp = load_yaml(experiment_yaml_path)
    grid = exp.get("grid", {})

    if not isinstance(grid, dict):
        raise TypeError("'grid' must be a dict in experiment YAML.")

    experiment_run_root = get_experiment_run_root(
        cfg_path=cfg_path,
        experiment_yaml_path=experiment_yaml_path,
    )

    results = get_results_experiment(
        experiment_root=experiment_run_root,
        grid_keys=list(grid.keys()),
        mode=mode,
        top_k_rq2a_per_run=top_k_rq2a_per_run,
    )

    return {
        "scope": "experiment",
        "mode": mode,
        "cfg_path": Path(cfg_path),
        "experiment_yaml_path": Path(experiment_yaml_path),
        "experiment_name": str(exp.get("experiment", Path(experiment_yaml_path).stem)),
        "experiment_run_root": experiment_run_root,
        "grid": grid,
        "grid_keys": list(grid.keys()),
        "results": results,
    }


def load_and_compare_experiments(
    *,
    experiment_specs: list[dict[str, Any]],
    experiment_yaml_path: Path,
    score_fraction: float = 0.95,
    top_n_runs: int = 10,
    param_names: Optional[list[str]] = None,
    param_pairs: Optional[list[tuple[str, str]]] = None,
    combo_top_n: int = 10,
    top_k_rq2a_per_run: Optional[int] = None,
) -> dict[str, Any]:
    """
    Load multiple experiment runs from separate config files, build comparison
    views for each, and combine them into one shared analysis structure.

    Expected spec structure:
        {
            "label": "nvs38k_EN_10_categories | full_gt",
            "cfg_path": Path(...),
            "dataset_name": "nvs38k_EN_10_categories",
            "mode": "full_gt",
        }

    Args:
        experiment_specs: List of experiment spec dicts.
        experiment_yaml_path: Shared experiment YAML path.
        score_fraction: Fraction of best score for top-region filtering.
        top_n_runs: Number of top runs to keep per setting.
        param_names: Parameter names to summarize.
        param_pairs: Parameter pairs to summarize.
        combo_top_n: Number of combo rows to keep per setting.
        top_k_rq2a_per_run: Optional top-k filter for RQ2a.
            None keeps all rows.

    Returns:
        Dict with:
            - bundles_by_experiment
            - views_by_experiment
            - combined_views
    """
    bundles_by_experiment: dict[str, dict[str, Any]] = {}
    views_by_experiment: dict[str, dict[str, Any]] = {}
 
    setting_order = [
        f"{str(spec['dataset_name'])} | {str(spec['mode'])}"
        for spec in experiment_specs
    ]

    for spec in experiment_specs:
        label = str(spec["label"])
        cfg_path = Path(spec["cfg_path"])
        dataset_name = str(spec["dataset_name"])
        mode = str(spec["mode"])

        validate_mode(mode)

        bundle = load_experiment_results_from_yaml(
            cfg_path=cfg_path,
            experiment_yaml_path=experiment_yaml_path,
            mode=mode,
            top_k_rq2a_per_run=top_k_rq2a_per_run,
        )

        results = bundle["results"]
        setting = f"{dataset_name} | {mode}"

        results_with_meta = {}
        for key, df in results.items():
            df_copy = df.copy()

            if "dataset_name" not in df_copy.columns:
                df_copy.insert(0, "dataset_name", dataset_name)

            if "mode" not in df_copy.columns:
                df_copy.insert(1, "mode", mode)

            if "setting" not in df_copy.columns:
                df_copy.insert(0, "setting", setting)

            results_with_meta[key] = df_copy

        views = collect_comparison_views(
            results=results_with_meta,
            dataset_names=[dataset_name],
            modes=[mode],
            score_fraction=score_fraction,
            top_n_runs=top_n_runs,
            param_names=param_names,
            param_pairs=param_pairs,
            combo_top_n=combo_top_n,
        )

        bundles_by_experiment[label] = {
            **bundle,
            "results": results_with_meta,
        }
        views_by_experiment[label] = views

    combined_views = combine_comparison_views(views_by_experiment)

    return {
        "bundles_by_experiment": bundles_by_experiment,
        "views_by_experiment": views_by_experiment,
        "combined_views": combined_views,
        "setting_order": setting_order,
    }