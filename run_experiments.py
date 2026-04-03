#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment runner for NVV pipeline parameter screening and grid search.

Loads a base config.yaml and an experiment definition YAML. The experiment
YAML specifies a ``grid:`` section mapping dot-notation parameter paths to
lists of values. The runner expands this grid via cartesian product; each
combination becomes one run. For every run:
  1. Deep-merges the generated override dict into the base config.
  2. Keeps processed_root unchanged and writes outputs into an experiment
     subfolder below that root.
  3. Computes a deterministic run hash per dataset and embeds it in each
     dataset's workspace folder name (output_rel), together with the run index.
  4. Writes the resolved config.yaml into the corresponding experiment folder.
  5. Executes the full pipeline followed by evaluation.

Experiment YAML format:

    experiment: screening_v1

    grid:
      pipeline.4_vad.vad_threshold: [0.15, 0.35]
      pipeline.4_vad.vad_min_silence_ms: [50, 300]
      pipeline.7_nvv.min_duration: [0.2, 0.6]
      pipeline.7_nvv.max_duration: [null, 3.0]
      pipeline.7_nvv.dedup_overlap_ratio: [0.6, 0.8]

Output structure (relative to the base processed_root):
    <processed_root>/<experiment_name>/
        <parent_output_rel>/
            001_<dataset_name>_<run_hash>/per_audio/...
            001_<dataset_name>_<run_hash>/global/...
            002_<dataset_name>_<run_hash>/...
            ...
            configs/
                <gt_mode>/
                    001_config.yaml
                    002_config.yaml
                    ...
            evaluation/

CLI usage:
    python run_experiments.py \\
        --config ./config/config.yaml \\
        --experiment ./experiments/screening_v1.yaml
"""

from __future__ import annotations

import subprocess
import sys
import argparse
import copy
import itertools
import re
import tempfile
from pathlib import Path
from typing import Any, Optional

import yaml

from config.load_config import load_config
from config.path_factory import get_project_paths
from metadata.run_tracking import compute_run_hash
from evaluation.results_experiment import run_results_experiment



# Only alphanumeric chars, underscores, and hyphens are valid in names used as
# directory components; reject anything else to prevent path traversal.
_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9_\-]+$")

VALID_STAGES = {
    "pipeline",
    "workspace_evaluation",
    "experiment_results",
}

# --- YAML helpers ---
#toDo use utils/load_yaml
def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file and return a dict (empty dict if file is empty)."""
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    """Write a dict to a YAML file, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


# --- Input validation ---

def _validate_name(name: str, label: str) -> None:
    """
    Raise ValueError if name contains characters that are unsafe as directory
    components (e.g. path separators, dots, spaces).

    Args:
        name: The name string to validate.
        label: Human-readable label used in the error message (e.g. "run name").
    """
    if not _SAFE_NAME_RE.match(name):
        raise ValueError(
            f"Invalid {label} '{name}': only alphanumeric characters, "
            "underscores, and hyphens are allowed."
        )
    
def _parse_stages(stages_raw: str | None) -> list[str]:
    """
    Parse and validate the stages argument.

    Args:
        stages_raw: Comma-separated stage string or None.

    Returns:
        List of validated stage names in input order.
    """
    if stages_raw is None:
        return ["pipeline", "workspace_evaluation", "experiment_results"]

    stages = [s.strip() for s in stages_raw.split(",") if s.strip()]
    if not stages:
        raise ValueError("Argument --stages was provided but no valid stage names were found.")

    invalid = [s for s in stages if s not in VALID_STAGES]
    if invalid:
        raise ValueError(
            f"Invalid stages: {invalid}. Allowed values: {sorted(VALID_STAGES)}"
        )

    return stages


# --- Config merging ---

def _set_nested(d: dict[str, Any], keys: list[str], value: Any) -> None:
    """
    Set a value in a nested dict using a list of string keys.

    Intermediate dicts are created as needed. If an intermediate key exists
    but holds a non-dict value it is replaced with an empty dict; this should
    not occur in normal grid expansion because all grid keys address leaf
    parameters, not intermediate containers.

    Args:
        d: The dict to modify in-place.
        keys: Ordered list of keys forming the path to the target.
        value: The value to store at the final key.
    """
    for key in keys[:-1]:
        if not isinstance(d.get(key), dict):
            d[key] = {}
        d = d[key]
    d[keys[-1]] = value


def _expand_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """
    Expand a flat parameter grid into a list of nested override dicts.

    Each key in ``grid`` is a dot-notation path (e.g.
    ``"pipeline.4_vad.vad_threshold"``) and each value is a list of candidate
    values. The function computes the cartesian product of all value lists and
    converts each combination into a nested dict compatible with ``_deep_merge``.

    An empty ``grid`` returns a list with one empty dict (a single no-override
    run). In practice this case is prevented by the caller, which raises
    ValueError before calling this function when ``grid`` is missing.

    Args:
        grid: Mapping of dot-notation parameter path to list of values.

    Returns:
        List of nested override dicts, one per parameter combination.
        Length equals the product of all value-list lengths.
    """
    if not grid:
        return [{}]
    keys = list(grid.keys())
    value_lists = [grid[k] for k in keys]
    result = []
    for combo in itertools.product(*value_lists):
        overrides: dict[str, Any] = {}
        for key, value in zip(keys, combo):
            _set_nested(overrides, key.split("."), value)
        result.append(overrides)
    return result


# --- Config merging (deep merge) ---

def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively merge overrides into a deep copy of base.

    Dicts are merged recursively; all other types (including None/null) are
    replaced by the override value. The original base dict is not modified.

    Args:
        base: Base configuration dict.
        overrides: Override values to apply.

    Returns:
        A new dict with overrides applied on top of base.
    """
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


# --- Run hash computation ---

def _compute_dataset_hashes(run_cfg: dict[str, Any]) -> dict[str, str]:
    """
    Compute deterministic run hashes for all datasets in run_cfg.

    Writes run_cfg to a temporary file, loads it via load_config, then calls
    compute_run_hash for each dataset. The hash depends only on the dataset
    identity (name, input path) and pipeline parameters — NOT on output_rel —
    so it can be computed before the final workspace path is decided.

    Args:
        run_cfg: Merged config dict with project.root set to an absolute path.

    Returns:
        Mapping from dataset name to 16-char hex run hash string.
    """
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", encoding="utf-8", delete=False
        ) as tmp:
            tmp_path = Path(tmp.name)
            yaml.safe_dump(run_cfg, tmp, default_flow_style=False, allow_unicode=True, sort_keys=False)
        tmp_config = load_config(tmp_path)
        return {ds.name: compute_run_hash(tmp_config, ds) for ds in tmp_config.datasets}
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


# --- Experiment runner ---

def run_experiments(
    config_path: Path,
    experiment_path: Path,
    *,
    stages: list[str],
) -> None:
    """
    Execute all runs defined in an experiment YAML file.

    The experiment YAML must contain a ``grid:`` section mapping dot-notation
    parameter paths to lists of values. The cartesian product of all value lists
    is expanded into individual runs, each assigned an auto-generated name
    (``001``, ``002``, …).

    For each run:
    - The base config is deep-merged with the generated parameter overrides.
    - The processed_root remains unchanged.
    - A deterministic run hash is computed for each dataset and embedded in
      that dataset's workspace folder name (output_rel).
    - The resolved config.yaml is written into the experiment folder.
    - The full pipeline and evaluation are executed.

    The experiment base directory is placed below the base processed_root as
    ``<processed_root>/<experiment_name>``.

    Args:
        config_path: Resolved path to the base config.yaml.
        experiment_path: Resolved path to the experiment definition YAML.
        stages: List of stages to execute.
    """
    
    config_path = Path(config_path).resolve()
    experiment_path = Path(experiment_path).resolve()

    base_cfg = _load_yaml(config_path)
    exp = _load_yaml(experiment_path)

    experiment_name = str(exp.get("experiment", experiment_path.stem))
    _validate_name(experiment_name, "experiment name")

    grid = exp.get("grid", {})
    if not grid:
        raise ValueError(f"No 'grid:' section defined in {experiment_path}")

    # Expand grid into one override dict per parameter combination.
    combinations = _expand_grid(grid)
    # Auto-generate run names: run_01, run_02, ...
    runs = [
        {"name": f"run_{i + 1:02d}", "overrides": overrides}
        for i, overrides in enumerate(combinations)
    ]

    # Resolve project paths from the base config.
    project = get_project_paths(config_path)
    project_root = project.project_root
    base_processed_root = project.processed_root
    experiment_base = base_processed_root / experiment_name

    print(f"\n▶ Experiment: {experiment_name}")
    print(f"  Base config:    {config_path}")
    print(f"  Processed root: {base_processed_root}")
    print(f"  Experiment dir: {experiment_base}")
    print(f"  Grid params:    {list(grid.keys())}")
    print(f"  Runs:           {len(runs)}\n")

    run_level_stages = {"pipeline", "workspace_evaluation"}
    execute_run_loop = any(stage in run_level_stages for stage in stages)

    if execute_run_loop:
        for run_index, run in enumerate(runs, start=1):
            run_name = str(run["name"])
            overrides = run["overrides"]

            # Deep-merge run overrides into the base config.
            run_cfg = _deep_merge(base_cfg, overrides)

            # Fix project.root to an absolute path so the resolved config.yaml
            # is loadable regardless of where it is stored on disk.
            run_cfg["project"]["root"] = str(project_root)

            # Keep processed_root unchanged exactly as defined in the base YAML.

            # Compute run hash per dataset (hash excludes output_rel, so it can be
            # computed before the final workspace folder name is decided).
            dataset_hashes = _compute_dataset_hashes(run_cfg)

            config_parent_rel: Path | None = None

            # Embed the run hash in each dataset's workspace folder name so that
            # every unique parameter combination maps to a distinct directory.
            for ds_entry in run_cfg["workspace"]["datasets"]:
                ds_name = str(ds_entry["name"])
                run_hash = dataset_hashes[ds_name]
                rel_path = Path(ds_entry["output_rel"])

                if config_parent_rel is None:
                    config_parent_rel = rel_path.parent

                if str(rel_path.parent) == ".":
                    ds_entry["output_rel"] = str(
                        Path(experiment_name) / f"{run_index:03d}_{ds_name}_{run_hash}"
                    )
                else:
                    ds_entry["output_rel"] = str(
                        Path(experiment_name) / rel_path.parent / f"{run_index:03d}_{ds_name}_{run_hash}"
                    )

            if config_parent_rel is None:
                raise ValueError("workspace.datasets is empty or missing in YAML.")
            
            gt_mode = str(run_cfg["evaluation"]["gt_mode"])

            # Write fully resolved config.yaml into the experiment folder.
            if str(config_parent_rel) == ".":
                resolved_cfg_path = (
                    experiment_base 
                    /"configs"
                    / gt_mode
                    / f"{run_index:03d}_config.yaml"
                )
            else:
                resolved_cfg_path = (
                    experiment_base
                    / config_parent_rel
                    / "configs"
                    / gt_mode
                    / f"{run_index:03d}_config.yaml"
                )

            _write_yaml(resolved_cfg_path, run_cfg)

            print(f"\n{'=' * 60}")
            print(f"▶ Run: {run_name}")
            print(f"  Resolved config: {resolved_cfg_path}")
            print(f"  Overrides:       {overrides}")
            for ds_entry in run_cfg["workspace"]["datasets"]:
                print(f"  Workspace [{ds_entry['name']}]: {ds_entry['output_rel']}")
            print(f"{'=' * 60}\n")


            if "pipeline" in stages:
                subprocess.run(
                    [sys.executable, "run_pipeline.py", "--config", str(resolved_cfg_path)],
                    check=True,
                    cwd=str(project_root),
                )

            if "workspace_evaluation" in stages:
                subprocess.run(
                    [sys.executable, "run_evaluation.py", "--config", str(resolved_cfg_path)],
                    check=True,
                    cwd=str(project_root),
                )

            print(f"\n✅ Run {run_name} completed.")

    if "experiment_results" in stages:
        print("\n▶ Running experiment results summary...")
        gt_mode = str(base_cfg["evaluation"]["gt_mode"])

        run_results_experiment(
            experiment_root=experiment_base,
            grid_keys=list(grid.keys()),
            mode=gt_mode,
            top_k_rq2a_per_run=3,
        )

        print("✅ Experiment results written.")

    print(f"\n✅ Experiment '{experiment_name}' completed. All {len(runs)} run(s) finished.")


def main() -> None:
    """
    CLI entry point.
    """
    p = argparse.ArgumentParser(
        description="Run NVV experiment stages (pipeline, workspace evaluation, experiment results)."
    )
    p.add_argument(
        "--config",
        required=True,
        type=str,
        help="Path to base config.yaml (e.g., ./config/config.yaml)",
    )
    p.add_argument(
        "--experiment",
        required=True,
        type=str,
        help="Path to experiment definition YAML (e.g., ./experiments/screening_v1.yaml)",
    )
    p.add_argument(
        "--stages",
        required=False,
        type=str,
        default=None,
        help=(
            "Comma-separated stages to execute. "
            "Allowed: pipeline,workspace_evaluation,experiment_results. "
            "Default: all."
        ),
    )

    args = p.parse_args()

    stages = _parse_stages(args.stages)

    run_experiments(
        config_path=Path(args.config).resolve(),
        experiment_path=Path(args.experiment).resolve(),
        stages=stages,
    )


if __name__ == "__main__":
    main()

# --- EXAMPLE USAGE ---
# Step 1: activate environment (pipeline_eval)
# Step 2: cd <project_root>
# Step 3: run the experiment runner, e.g.:
# python run_experiments.py \
#     --config ./config/config.yaml \
#     --experiment ./experiments/screening/param_screening_v1.yaml
#
# To run only specific stages, use the --stages argument with a comma-separated list of stages:
# python run_experiments.py \   
#     --config ./config/config.yaml \
#     --experiment ./experiments/screening/param_screening_v1.yaml \
#     --stages workspace_evaluation,experiment_results