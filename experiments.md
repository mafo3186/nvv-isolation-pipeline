# Experiment Framework

This document describes how to run parameter screening and grid search
experiments for the NVV pipeline.

---

## Overview

The experiment framework allows running the NVV pipeline with multiple
parameter configurations automatically. Each run:

* uses the base `config/config.yaml` as its starting point,
* applies parameter overrides generated from the experiment `grid:`,
* stores all outputs (pipeline artifacts + evaluation results) in a dedicated
  run directory,
* writes a fully resolved `config.yaml` to the experiment directory for reproducibility.

An experiment YAML defines a `grid:` section with dot-notation parameter paths
and value lists. The runner expands these via a **cartesian product** — each
combination becomes one run, automatically named (indexed) `001`, `002`, etc.

Both **screening** (coarse, wide parameter range) and **grid search** (fine,
targeted parameter range) experiments use the same runner. Only the experiment
definition YAML changes.

---

## Output Structure

```
data/processed/<processed_root>/
    <experiment_name>/
        <dataset_parent_rel>/ # optional
            001_config.yaml
            001_<dataset_name>_<run_hash>/
                per_audio/        ← pipeline step outputs per audio file
                global/
                    evaluation/   ← evaluation results
                    clips/        ← extracted audio clips
            002_config.yaml
            002_<dataset_name>_<run_hash>/
                ...
```

For each run:

* a resolved config file (`001_config.yaml`, `002_config.yaml`, …) is written
  into the experiment directory,
* one workspace directory is created **per dataset**.

The `<run_hash>` is a 16-character deterministic SHA-256 digest computed from
the dataset identity and pipeline parameters (excluding the output path itself).
This means the same parameter combination always maps to the same workspace,
enabling safe resume of interrupted runs.

---

## Quick Start

### 1. Activate the environment

```bash
conda activate pipeline_eval
cd <project_root>
```

### 2. Run a screening or grid search experiment defined as yaml

```bash
python run_experiments.py --config ./experiments/screening/config_param_screening_nvs38k_full_gt.yaml --experiment ./experiments/screening/param_screening_v1.yaml
```

### 3. Run a grid search experiment

```bash
python run_experiments.py \
    --config ./experiments/<your_experiment>/your_experiment_config.yaml \
    --experiment ./experiments/grid_search_v1.yaml
```

---

## Experiment Definition Format

An experiment YAML defines a name and a `grid:` section. Each entry in `grid:`
is a dot-notation path into the config structure mapped to a list of candidate
values. The runner computes the cartesian product of all value lists and executes
one pipeline run per combination.

```yaml
experiment: screening_v1

grid:
  pipeline.4_vad.vad_threshold: [0.15, 0.20, 0.25]
  pipeline.4_vad.vad_min_silence_ms: [50, 75, 150]
  pipeline.7_nvv.max_duration: [null, 6.5]
  pipeline.7_nvv.dedup_overlap_ratio: [0.5, 0.7, 0.9]

```

This example defines 3 × 3 × 2 × 3 = 54 runs.

**Rules:**

* Dot-notation keys mirror the nested structure of `config.yaml`
  (e.g. `pipeline.4_vad.vad_threshold` → `config["pipeline"]["4_vad"]["vad_threshold"]`).
* Only the parameters listed in `grid:` are overridden; all others inherit
  from the base config.
* `null` is valid for `max_duration` (no upper duration limit).
* Run names (`001`, `002`, …) are assigned automatically in cartesian-product order.

---

## Tunable Parameters

The following parameters are recommended for screening and grid search.

### Step 4 – Voice Activity Detection (VAD)

| Parameter            | Type    | Description                                     | Screening range |
| -------------------- | ------- | ----------------------------------------------- | --------------- |
| `vad_threshold`      | `float` | Silero VAD probability threshold (0–1)          | `[0.15, 0.25]`  |
| `vad_min_silence_ms` | `int`   | Minimum silence duration to split segments (ms) | `[50, 150]`     |

### Step 7 – Non-Vocal Vocalization Detection (NVV)

| Parameter             | Type           | Description                                          | Screening range |
| --------------------- | -------------- | ---------------------------------------------------- | --------------- |
| `max_duration`        | `float / null` | Maximum candidate duration in seconds (`null` = off) | `[null, 6.5]`   |
| `dedup_overlap_ratio` | `float`        | IoU threshold for deduplication (0–1)                | `[0.5, 0.9]`    |

All other parameters (e.g. `vad_min_speech_ms`, `vad_pad_ms`, ASR settings,
NLP model) are inherited unchanged from the base config.

---

## Provided Experiment Definitions

| File                              | Type        | Runs | Description                                       |
| --------------------------------- | ----------- | ---- | ------------------------------------------------- |
| `experiments/screening/param_screening_v1.yaml`   | Screening   | 54   | Full cartesian product of wide parameter extremes |
| tbd | Grid search | tbd   | Fine sweep around the default config              |

---

## Creating a New Experiment

1. Copy `experiments/screening_v1.yaml` to a new file, e.g.
   `experiments/my_experiment.yaml`.
2. Set a unique `experiment:` name (used as the output directory name).
3. Define the `grid:` with the desired parameter paths and value lists.
4. Run it with:

```bash
python run_experiments.py --config ./experiments/<experiment_name>config_<my_experiment>.yaml --experiment ./experiments/<experiment_name>/<my_experiment_grid>.yaml
```

The total number of runs equals the product of all value-list lengths. Keep
this manageable (e.g. 2–3 values per parameter) when doing an initial screening.

### Run only selected stages
Default is running all stages. The runner supports partial execution using the --stages argument. 
Allowed stages are:

| Stage                  | Description                                                           |
| ---------------------- | --------------------------------------------------------------------- |
| `pipeline`             | Runs the NVV pipeline for each parameter configuration                |
| `workspace_evaluation` | Runs `run_evaluation.py` for each workspace                           |
| `experiment_results`   | Aggregates results across runs and generates experiment-level outputs |

Example: run pipeline and workspace evaluation but skip experiment-level aggregation.

```bash
python run_experiments.py --config ./experiments/<experiment_name>/config_<my_experiment>.yaml --experiment ./experiments/<experiment_name>my_experiment_grid.yaml --stages pipeline,workspace_evaluation
```
---

## Reproducibility

Each run directory contains a fully resolved `config.yaml` with:

* an absolute `project.root` path,
* the exact parameter values used for that run,
* all other base config settings unchanged.

The pipeline's run tracking (`run.json`, `runs_index.json`) is also written per
run as usual, capturing the pipeline config hash for later verification.
