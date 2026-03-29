#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run evaluation preprocessing for GT annotation data.

This script mirrors the style of run_pipeline.py:
- Loads config (CLI) or accepts a config object (notebook)
- Ensures GT directories exist
- Cleans all configured GT Excel files
- Prints Excel inspection per GT unit
- Exports Audacity label files
- Optionally copies annotated VOCALS WAV files
- Merges cleaned GT Excel files at the end

Notebook usage:
    from config.load_config import load_config
    from run_preprocessing import run_preprocessing_from_config

    config = load_config("./config/config.yaml")
    run_preprocessing_from_config(config, copy_vocals=False)

CLI usage:
    python run_preprocessing.py --config ./config/config.yaml
    python run_preprocessing.py --config ./config/config.yaml --copy-vocals
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from config.load_config import (
    ensure_workspace,
    load_config,
    print_config,
)
from config.path_factory import (
    ensure_gt_dirs,
    get_gt_merged_cleaned_excel_path,
    print_paths,
)
from preprocessing.clean_annotations_excel import clean_annotations_excel
from preprocessing.extract_and_copy_annotated_wavs_from_vocals import (
    extract_and_copy_annotated_wavs_from_vocals_project_structure,
)
from preprocessing.merge_gt_excels import merge_gt_excels
from preprocessing.xls_to_audacity_labels import xls_to_audacity_labels

from utils.io import print_header



def print_excel_inspection(
    *,
    original_excel: Path,
    cleaned_excel: Path,
    unit_name: str,
) -> None:
    """
    Print a column-level comparison between raw and cleaned GT Excel files.

    Arguments:
        original_excel: Path to the raw Excel file.
        cleaned_excel: Path to the cleaned Excel file.
        unit_name: GT unit name for logging.
    """

    print_header(f"▶ COLUMN CHECK — {original_excel.name} ({unit_name})")

    df_orig = pd.read_excel(original_excel)
    df_clean = pd.read_excel(cleaned_excel)

    cols_orig = df_orig.columns.tolist()
    cols_clean = df_clean.columns.tolist()

    print("📘 Original Columns:")
    print(cols_orig)

    print("\n📗 Cleaned Columns:")
    print(cols_clean)

    missing_in_clean = [c for c in cols_orig if c not in cols_clean]
    added_in_clean = [c for c in cols_clean if c not in cols_orig]

    if missing_in_clean:
        print("\n⚠️ Removed during cleaning:", missing_in_clean)
    if added_in_clean:
        print("⚠️ Added during cleaning:", added_in_clean)

    print("\nDone.")


def run_preprocessing_from_config(
    config: object,
    *,
    copy_vocals: bool = False,
) -> None:
    """
    Run GT preprocessing in notebook-style orchestration.

    Arguments:
        config: Loaded AppConfig object from load_config().
        copy_vocals: If True, copy annotated VOCALS WAV files.
    """
    ensure_workspace(config)
    ensure_gt_dirs(config.cfg_path, project=config.project)

    print_config(config)
    print_paths(config.cfg_path)

    gt_units = config.evaluation.gt_units
    cleaned_excels: dict[str, Path] = {}

    for unit in gt_units:

        print_header(f"▶ GT Preprocessing: cleaning excel — {unit.name} ({unit.raw_excel_path.name})")

        cleaned_excel_path = clean_annotations_excel(
            input_path=unit.raw_excel_path,
            output_path=unit.cleaned_excel_path,
            id_col=unit.id_column,
        )
        cleaned_excel_path = Path(cleaned_excel_path)
        cleaned_excels[unit.name] = cleaned_excel_path

        print(f"Cleaned Excel saved → {cleaned_excel_path}")

        print_excel_inspection(
            original_excel=unit.raw_excel_path,
            cleaned_excel=cleaned_excel_path,
            unit_name=unit.name,
        )

        print_header(f"▶ STEP 2: Export Labels — {unit.name}")

        unit.labels_export_dir.mkdir(parents=True, exist_ok=True)

        xls_to_audacity_labels(
            input_path=cleaned_excel_path,
            output_dir=unit.labels_export_dir,
        )

        print(f"✔ Audacity labels exported → {unit.labels_export_dir}")

        if copy_vocals:
            if unit.vocals_dataset_root is None or unit.vocals_subset_copy_dir is None:
                print(f"⏭ Skipping VOCALS copy for {unit.name} (no VOCALS subset configuration)")
            else:
                print_header(f"▶ STEP 4: Copy WAVs — {unit.name}")

                unit.vocals_subset_copy_dir.mkdir(parents=True, exist_ok=True)

                result = extract_and_copy_annotated_wavs_from_vocals_project_structure(
                    excel_path=cleaned_excel_path,
                    dataset_root=unit.vocals_dataset_root,
                    output_folder_path=unit.vocals_subset_copy_dir,
                    id_column=unit.id_column,
                    verbose=True,
                )

                total_requested = len(result["audio_ids"])
                copied_count = result["copied"]
                missing_ids = result["missing"]

                print("\n📁 Output folder:", result["output_dir"])
                print("📊 Requested:", total_requested)
                print("✔ Copied:", copied_count)
                print("❌ Missing:", len(missing_ids))

                if missing_ids:
                    print("\nMissing IDs (first 20):")
                    for mid in missing_ids[:20]:
                        print("  ", mid)
                    if len(missing_ids) > 20:
                        print(f"  ... ({len(missing_ids) - 20} more)")

    print("\n✅ Per-unit GT preprocessing completed for all GT units.")

    merged_gt_path = get_gt_merged_cleaned_excel_path(
        cfg_path=config.cfg_path,
        project=config.project,
    )
    if merged_gt_path is not None:
        print(f"\n✅ Merging cleaned GT Excel files into one → {merged_gt_path}")
    else:
        print("Skipping GT merge (no merged GT path configured)")


    print_header("Merging cleaned GT Excel files into one")

    _, merged_path = merge_gt_excels(
        cleaned_excels,
        out_path=merged_gt_path,
    )

    print(f"Merged GT Excel saved → {merged_path}")
    print("\n✅ GT merge completed.")

    if not copy_vocals:
        print("\nℹ️ VOCALS subset copy skipped (--copy-vocals not set).")


def main() -> None:
    """
    CLI entry point: load config and run preprocessing.
    """
    p = argparse.ArgumentParser(description="Run GT preprocessing for NVV evaluation")
    p.add_argument(
        "--config",
        required=True,
        type=str,
        help="Path to config.yaml (e.g., ./config/config.yaml)",
    )
    p.add_argument(
        "--copy-vocals",
        action="store_true",
        help="If set, copy annotated VOCALS WAV files into configured subset folders.",
    )
    args = p.parse_args()

    config_path = Path(args.config).resolve()
    config = load_config(config_path)

    run_preprocessing_from_config(
        config,
        copy_vocals=args.copy_vocals,
    )


if __name__ == "__main__":
    main()


# --- EXAMPLE USAGE ---
# calling in CLI:
# Step 1: activate environment
# Step 2: go to project root (e.g., cd ~/thesis/repos/master_thesis/)
# Step 3: run preprocessing, e.g.:
# python run_preprocessing.py --config ./config/config.yaml
# python run_preprocessing.py --config ./config/config.yaml --copy-vocals