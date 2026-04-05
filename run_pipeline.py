#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run the full NVV pipeline (Steps 1–7) plus exports (labels + exploration clips).

This script is designed to mirror the user's notebook runner:
- Loads config (CLI) or accepts a config object (notebook)
- Runs steps 1..7 for all datasets with the same print-outs
- Exports VAD labels, ASR labels, NVV labels (file-scan based)
- Extracts exploration clips for all configured modes and (vad_mask, asr_audio_in) combos

Notebook usage:
    from run_pipeline import run_pipeline_from_config
    run_pipeline_from_config(config)

CLI usage:
    python run_pipeline.py --config ./config/config.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from config.load_config import (
    ensure_workspace,
    print_config,
)

from pipeline.step_1_standardize import run_step_1_std
from pipeline.step_2_separate import run_step_2_separate
from pipeline.step_3_normalize import run_step_3_normalize
from pipeline.step_4_vad import run_step_4_vad
from pipeline.step_5_asr import run_step_5_asr
from pipeline.step_6_nlp import run_step_6_nlp
from pipeline.step_7_nvv import run_step_7_nvv

from export.export_labels import export_labels
from export.export_clips import export_clips

from metadata.run_tracking import write_run
from config.path_factory import get_exploration_clips_sub_dir, default_uvr_model_path, default_asr_utils_path


def run_pipeline_from_config(
    config: object,
    *,
    model_path: Optional[Path] = None,
    asr_utils_path: Optional[Path] = None,
) -> None:
    """
    Run the full pipeline exactly like the notebook runner.

    Args:
        config: Loaded pipeline config object (from load_pipeline_config()).
        model_path: Optional UVR model path for Step 2 (defaults to models/UVR-MDX-NET-Inst_3.onnx).
        asr_utils_path: Optional CrisperWhisper utils path for Step 5.
    """
    ensure_workspace(config)
    print_config(config)

    # Register/verify run identity for each dataset workspace.
    for ds in config.datasets:
        run_data = write_run(config, ds, force=config.runtime.force)
        print(f"▶ run_id [{ds.name}]: {run_data['run_id']}")

    if model_path is None:
        model_path = default_uvr_model_path(config)
    if asr_utils_path is None:
        asr_utils_path = default_asr_utils_path(config)

    print(f"Using UVR model from: {model_path}")
    print(f"Using ASR utils from: {asr_utils_path}")

    # Step 1
    for ds in config.datasets:
        print(f"\n🚀 Step 1 Standardization for {ds.name}, Input: {ds.input_dir}, Output: {ds.workspace}.")
        run_step_1_std(
            input_dir=ds.input_dir,
            workspace=ds.workspace,
            project_root=config.project.project_root,
            device=config.runtime.device,
            force=config.runtime.force,
        )
    print("\n✅ Step 1 Standardization completed for all datasets.")

    # Step 2
    for ds in config.datasets:
        print(f"\n🚀 Step 2 Source Separation for {ds.name} in {ds.workspace}.")
        run_step_2_separate(
            workspace=ds.workspace,
            model_path=model_path,
            project_root=config.project.project_root,
            device=config.runtime.device,
            force=config.runtime.force,
        )
    print("\n✅ Step 2 Source Separation completed for all datasets.")

    # Step 3
    for ds in config.datasets:
        print(f"\n🚀 Step 3 for {ds.name} in {ds.workspace}.")
        run_step_3_normalize(
            workspace=ds.workspace,
            project_root=config.project.project_root,
            device=config.runtime.device,
            force=config.runtime.force,
        )
    print("\n✅ Step 3 Normalization completed for all datasets.")

    # Step 4
    for ds in config.datasets:
        print(f"\n🚀 Step 4 Voice Activity Detection for {ds.name} in {ds.workspace}.")
        run_step_4_vad(
            workspace=ds.workspace,
            audio_derivatives=config.step_4_vad.vad_audios_in,
            vad_threshold=config.step_4_vad.vad_threshold,
            vad_min_speech_ms=config.step_4_vad.vad_min_speech_ms,
            vad_min_silence_ms=config.step_4_vad.vad_min_silence_ms,
            vad_pad_ms=config.step_4_vad.vad_pad_ms,
            project_root=config.project.project_root,
            device=config.runtime.device,
            force=config.runtime.force,
        )
        print("\n✅ Step 4 VAD completed for all datasets.")

    # Export VAD labels
    for ds in config.datasets:
        workspace = ds.workspace
        print(f"\n🚀 Exporting VAD labels for {ds.name}: {workspace}")
        export_labels(
            workspace=workspace,
            project_root=config.project.project_root,
            export_vad=True,
            export_asr=False,
            export_nvv=False,
            vad_masks=None,
            asr_audio_ins=None,
            force=False,
        )
    print("\n✅ VAD label export completed for all datasets.")

    # Step 5
    for ds in config.datasets:
        print(f"\n🚀 Step 5 ASR for {ds.name} in {ds.workspace}.")
        run_step_5_asr(
            workspace=ds.workspace,
            utils_path=asr_utils_path,
            vad_masks=config.step_5_asr.vad_masks_in,
            asr_audios_in=config.step_5_asr.asr_audios_in,
            asr_chunk_length_s=config.step_5_asr.asr_chunk_length_s,
            asr_batch_size=config.step_5_asr.asr_batch_size,
            project_root=config.project.project_root,
            device=config.runtime.device,
            force=config.runtime.force,
        )
    print("\n✅ Step 5 ASR completed for all datasets.")

    # Export ASR labels
    for ds in config.datasets:
        print(f"\n🚀 Exporting ASR labels for {ds.name}: {ds.workspace}")
        export_labels(
            workspace=ds.workspace,
            project_root=config.project.project_root,
            export_vad=False,
            export_asr=True,
            export_nvv=False,
            vad_masks=None,
            asr_audio_ins=None,
            force=True,
        )
    print("\n✅ ASR label export completed for all datasets.")

    # Step 6
    for ds in config.datasets:
        print(f"\n🚀 Step 6 (NLP) for {ds.name} in {ds.workspace}.")
        run_step_6_nlp(
            workspace=ds.workspace,
            spacy_model=config.step_6_nlp.spacy_model,
            project_root=config.project.project_root,
            auto_download=True,
            force=config.runtime.force,
        )
        print("\n✅ Step 6 NLP completed for all datasets.")

    # Step 7
    for ds in config.datasets:
        print(f"\n🚀 Step 7 (NVV) for {ds.name} in {ds.workspace}.")
        run_step_7_nvv(
            workspace=ds.workspace,
            exclude_categories=config.step_7_nvv.exclude_categories,
            min_duration=config.step_7_nvv.min_duration,
            max_duration=config.step_7_nvv.max_duration,
            vad_masks_in=config.step_7_nvv.vad_masks_in,
            asr_audios_in=config.step_7_nvv.asr_audios_in,
            vad_gate_padding=config.step_7_nvv.vad_gate_padding,
            dedup_overlap_ratio=config.step_7_nvv.dedup_overlap_ratio,
            dedup_time_tol_s=config.step_7_nvv.dedup_time_tol_s,
            project_root=config.project.project_root,
            force=config.runtime.force,
        )
    print("\n✅ Step 7 NVV completed for all datasets.")

    # Export NVV labels
    for ds in config.datasets:
        workspace = ds.workspace
        print(f"\n🚀 Exporting NVV labels for {ds.name}: {workspace}")
        export_labels(
            workspace=workspace,
            project_root=config.project.project_root,
            export_vad=False,
            export_asr=False,
            export_nvv=True,
            vad_masks=None,
            asr_audio_ins=None,
            force=False,
        )
    print("\n✅ NVV label export completed for all datasets.")

    # Exploration clips
    gt_mode =  config.evaluation.gt_mode
    subfolder = "exploration"
    for ds in config.datasets:
        print(f"\n🚀 Exporting NVV clips for {ds.name}: {ds.workspace}")

        for clip_mode in config.export.clips:
            for vad_mask in config.step_7_nvv.vad_masks_in:
                for asr_audio_in in config.step_7_nvv.asr_audios_in:
                    export_clips(
                        workspace=ds.workspace,
                        project_root=config.project.project_root,
                        mode=clip_mode,
                        vad_masks=[vad_mask],
                        asr_audio_ins=[asr_audio_in],
                        sub_dir=get_exploration_clips_sub_dir(subfolder, gt_mode, vad_mask, asr_audio_in),
                        force=False,
                    )
    print("\n✅ NVV clip export completed for all datasets.")


def main() -> None:
    """
    CLI entry point: load config and run the pipeline.
    """
    p = argparse.ArgumentParser(description="Run NVV pipeline (Steps 1–7) plus exports")
    p.add_argument(
        "--config",
        required=True,
        type=str,
        help="Path to config.yaml (e.g., ./config/config.yaml)",
    )
    args = p.parse_args()

    config_path = Path(args.config).resolve()

    from config.load_config import load_config
    config = load_config(config_path)

    run_pipeline_from_config(config)


if __name__ == "__main__":
    main()

# --- EXAMPLE USAGE ---
# calling in CLI:
# Step 1: activate environment (pipeline_eval)
# Step 2: go to project root (e.g., cd ~/thesis/repos/master_thesis/)
# Step 3: run the pipeline with config e. g.:
# (pipeline_eval) <project_root_path> python run_pipeline.py --config ./config/config.yaml