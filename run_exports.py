#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Root runner for exports (labels + clips).

CLI Examples:
- Export everything (labels + clips):
  python run_exports.py --workspace ./data/processed/my_ws

- Only NVV labels:
  python run_exports.py --workspace ./data/processed/my_ws --labels --nvv

- Only clips (words):
  python run_exports.py --workspace ./data/processed/my_ws --clips --clip-mode words

- Filter by tokens:
  python run_exports.py --workspace ./data/processed/my_ws --labels --nvv --vad-mask no --asr-audio-in std_vocals_norm

Config-based CLI (optional):
  python run_exports.py --config ./config/config.yaml --subfolder exploration

Notebook usage:
  from run_exports import run_exports_from_config
  run_exports_from_config(config, subfolder="exploration")
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from export.export_labels import export_labels
from export.export_clips import export_clips


def _opt_list(values: Optional[List[str]]) -> Optional[List[str]]:
    """Return None if empty, else the list."""
    if not values:
        return None
    return values


def run_exports(
    workspace: Path | str,
    *,
    labels: bool = True,
    clips: bool = True,
    export_vad: bool = True,
    export_asr: bool = True,
    export_nvv: bool = True,
    clip_mode: str = "nvv",
    sub_dir: Optional[str] = None,
    vad_masks: Optional[List[str]] = None,
    asr_audio_ins: Optional[List[str]] = None,
    force: bool = False,
) -> None:
    """
    Run exports on a single workspace.

    Args:
        workspace: Workspace root (contains per_audio/ and global/).
        labels: If True, export labels.
        clips: If True, export clips.
        export_vad: If True, export VAD labels.
        export_asr: If True, export ASR labels.
        export_nvv: If True, export NVV labels.
        clip_mode: "nvv" or "words".
        sub_dir: Optional sub-dir under global/clips/<sub_dir>/<mode>/...
        vad_masks: Optional filter list.
        asr_audio_ins: Optional filter list.
        force: Overwrite existing outputs if True.
    """
    ws = Path(workspace).resolve()

    # Common case: if both toggles are False, do nothing.
    if not labels and not clips:
        return

    if labels:
        export_labels(
            workspace=ws,
            export_vad=export_vad,
            export_asr=export_asr,
            export_nvv=export_nvv,
            vad_masks=vad_masks,
            asr_audio_ins=asr_audio_ins,
            force=force,
        )

    if clips:
        export_clips(
            workspace=ws,
            mode=clip_mode,
            vad_masks=vad_masks,
            asr_audio_ins=asr_audio_ins,
            sub_dir=sub_dir,
            force=force,
        )


def run_exports_from_config(
    config: object,
    *,
    do_labels: bool = True,
    do_clips: bool = True,
    subfolder: str = "exploration",
    force_labels: bool = False,
    force_clips: bool = False,
) -> None:
    """
    Run exports for all datasets defined in config, without notebook-side loops.

    This mirrors your current notebook usage:
    - Labels: export VAD+ASR+NVV with file-scan based discovery
    - Clips: export exploration clips for each (mode, vad_mask, asr_audio_in)

    Args:
        config: Loaded pipeline config object with config.datasets, config.export.clips, config.step_7_nvv.
        do_labels: Export labels for all datasets if True.
        do_clips: Export clips for all datasets if True.
        subfolder: Subfolder prefix under global/clips/ for exploration outputs.
        force_labels: Overwrite existing label files if True.
        force_clips: Overwrite existing clip files if True.
    """
    # 1) Labels (all three)
    if do_labels:
        for dataset in config.datasets:
            workspace = dataset.workspace
            print(f"\n🚀 Exporting labels (VAD+ASR+NVV) for {dataset.name}: {workspace}")

            run_exports(
                workspace,
                labels=True,
                clips=False,
                export_vad=True,
                export_asr=True,
                export_nvv=True,
                vad_masks=None,
                asr_audio_ins=None,
                force=force_labels,
            )

        print("\n✅ Label export completed for all datasets.")

    # 2) Clips (exploration, per combo)
    if do_clips:
        for dataset in config.datasets:
            print(f"\n🚀 Exporting clips for {dataset.name}: {dataset.workspace}")

            for mode in config.export.clips:
                for vad_mask in config.step_7_nvv.vad_masks_in:
                    for asr_audio_in in config.step_7_nvv.asr_audios_in:
                        run_exports(
                            dataset.workspace,
                            labels=False,
                            clips=True,
                            clip_mode=mode,
                            vad_masks=[vad_mask],
                            asr_audio_ins=[asr_audio_in],
                            sub_dir=f"{subfolder}/{vad_mask}_vad__{asr_audio_in}_asr",
                            force=force_clips,
                        )

        print("\n✅ Clip export completed for all datasets.")


def main() -> None:
    """
    CLI entry point.
    Supports:
    - workspace-based run (your current script behavior)
    - optional config-based run (like run_pipeline), if you provide --config
    """
    p = argparse.ArgumentParser(description="Run exports (labels + clips)")

    # Mode A: workspace-based
    p.add_argument("--workspace", required=False, type=str, help="Workspace root (contains per_audio/ and global/)")

    # Mode B: config-based (optional)
    p.add_argument("--config", required=False, type=str, help="Path to config.yaml (runs all datasets)")
    p.add_argument("--subfolder", default="exploration", type=str, help="Subfolder under global/clips/ for exploration")

    # Main toggles
    p.add_argument("--labels", action="store_true", help="Export labels")
    p.add_argument("--clips", action="store_true", help="Export clips")

    # Labels toggles
    p.add_argument("--vad", action="store_true", help="Export VAD labels")
    p.add_argument("--asr", action="store_true", help="Export ASR labels")
    p.add_argument("--nvv", action="store_true", help="Export NVV labels")

    # Clips options
    p.add_argument("--clip-mode", choices=["nvv", "words"], default="nvv", help="Clip extraction mode")
    p.add_argument("--sub-dir", default=None, help="Optional sub-dir under global/clips/<sub_dir>/<mode>/... (workspace-mode)")

    # Filters (optional, workspace-mode and config-mode single-workspace runs)
    p.add_argument("--vad-mask", dest="vad_masks", action="append", default=[], help="Filter: vad_mask (repeatable)")
    p.add_argument("--asr-audio-in", dest="asr_audio_ins", action="append", default=[], help="Filter: asr_audio_in (repeatable)")

    # Overwrite
    p.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    p.add_argument("--force-labels", action="store_true", help="Overwrite existing label files (config-mode)")
    p.add_argument("--force-clips", action="store_true", help="Overwrite existing clip files (config-mode)")

    args = p.parse_args()

    # Decide mode
    if args.config:
        # Config-based run over all datasets (no notebook loops)
        from config.load_config import load_config

        cfg = load_config(Path(args.config).resolve())

        # If neither labels nor clips explicitly set: do both
        do_labels = args.labels or (not args.labels and not args.clips)
        do_clips = args.clips or (not args.labels and not args.clips)

        run_exports_from_config(
            cfg,
            do_labels=do_labels,
            do_clips=do_clips,
            subfolder=args.subfolder,
            force_labels=bool(args.force_labels),
            force_clips=bool(args.force_clips),
        )
        return

    # Workspace-based run (original behavior)
    if not args.workspace:
        raise SystemExit("Provide either --workspace or --config.")

    # If neither labels nor clips explicitly set: do both (the common case "export everything")
    do_labels = args.labels or (not args.labels and not args.clips)
    do_clips = args.clips or (not args.labels and not args.clips)

    # If labels are requested but no specific label types chosen: export all three
    export_vad = args.vad or (do_labels and not (args.vad or args.asr or args.nvv))
    export_asr = args.asr or (do_labels and not (args.vad or args.asr or args.nvv))
    export_nvv = args.nvv or (do_labels and not (args.vad or args.asr or args.nvv))

    vad_masks = _opt_list(args.vad_masks)
    asr_audio_ins = _opt_list(args.asr_audio_ins)

    run_exports(
        args.workspace,
        labels=do_labels,
        clips=do_clips,
        export_vad=export_vad,
        export_asr=export_asr,
        export_nvv=export_nvv,
        clip_mode=args.clip_mode,
        sub_dir=args.sub_dir,
        vad_masks=vad_masks,
        asr_audio_ins=asr_audio_ins,
        force=args.force,
    )


if __name__ == "__main__":
    main()