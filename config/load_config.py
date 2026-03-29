from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence
import yaml

from config.constants import (
    KEY_STEP_4,
    KEY_STEP_5,
    KEY_STEP_6,
    KEY_STEP_7,
    AUDIO_DERIVATIVES,
    VAD_MASKS,
)

from config.path_factory import (
    ProjectPaths,
    DatasetPaths,
    GtExcelUnitPaths,
    get_project_paths,
    get_datasets,
    ensure_workspace_dirs,
    get_workspace_paths,
    get_gt_units,
    get_gt_truth_excel_paths,
)


# Typed config dataclasses

@dataclass(frozen=True)
class RuntimeConfig:
    device: str
    force: bool


@dataclass(frozen=True)
class Step4VADConfig:
    vad_audios_in: list[str]
    vad_threshold: float
    vad_min_speech_ms: int
    vad_min_silence_ms: int
    vad_pad_ms: int


@dataclass(frozen=True)
class Step5ASRConfig:
    vad_masks_in: list[str]
    asr_audios_in: list[str]
    asr_chunk_length_s: int
    asr_batch_size: int


@dataclass(frozen=True)
class Step6NLPConfig:
    spacy_model: str


@dataclass(frozen=True)
class Step7NVVConfig:
    exclude_categories: list[str]
    min_duration: float
    max_duration: Optional[float]
    vad_masks_in: list[str]
    asr_audios_in: list[str]
    vad_gate_padding: float
    dedup_overlap_ratio: float
    dedup_time_tol_s: float


@dataclass(frozen=True)
class ExportConfig:
    clips: list[str]
    labels: list[str]


@dataclass(frozen=True)
class EvaluationConfig:
    gt_mode: str
    gt_units: list[GtExcelUnitPaths]
    gt_truth_paths: list[Path]


@dataclass(frozen=True)
class AppConfig:
    """
    Unified config for pipeline + evaluation.

    Args:
        cfg_path: Resolved path to the YAML config file.
        project: Resolved project roots.
        runtime: Runtime settings.
        datasets: Resolved dataset paths.
        step_4_vad: Step 4 config.
        step_5_asr: Step 5 config.
        step_6_nlp: Step 6 config.
        step_7_nvv: Step 7 config.
        export: Export config.
        evaluation: Evaluation config.
    """

    cfg_path: Path
    project: ProjectPaths
    runtime: RuntimeConfig
    datasets: list[DatasetPaths]
    step_4_vad: Step4VADConfig
    step_5_asr: Step5ASRConfig
    step_6_nlp: Step6NLPConfig
    step_7_nvv: Step7NVVConfig
    export: ExportConfig
    evaluation: EvaluationConfig

    def print_datasets(self) -> None:
        for ds in self.datasets:
            print(f"\n▶ Dataset: {ds.name}\n  Input: {ds.input_dir}\n  Workspace: {ds.workspace}")


# YAML helpers

def _load_yaml(cfg_path: Path) -> dict[str, Any]:
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _require_str(section: dict[str, Any], key: str, *, field_path: str) -> str:
    value = section.get(key, None)
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f'"{field_path}" must be a non-empty string.')
    return str(value)


def _require_float_or_none(section: dict[str, Any], key: str, *, field_path: str) -> Optional[float]:
    if key not in section:
        raise TypeError(f'"{field_path}" is required (use null explicitly if intended).')
    value = section.get(key)
    return None if value is None else float(value)


def _require_float(section: dict[str, Any], key: str, *, field_path: str) -> float:
    if key not in section:
        raise TypeError(f'"{field_path}" is required.')
    value = section.get(key)
    if value is None:
        raise TypeError(f'"{field_path}" must be a number (null is not allowed).')
    return float(value)


def _require_int(section: dict[str, Any], key: str, *, field_path: str) -> int:
    if key not in section:
        raise TypeError(f'"{field_path}" is required.')
    value = section.get(key)
    if value is None:
        raise TypeError(f'"{field_path}" must be an integer (null is not allowed).')
    return int(value)


def _require_list_of_str(section: dict[str, Any], key: str, *, field_path: str) -> list[str]:
    if key not in section:
        raise TypeError(f'"{field_path}" is required (use [] explicitly if intended).')
    value = section.get(key)
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        raise TypeError(f'"{field_path}" must be a list of strings.')
    return [str(x) for x in value]


def _as_list_of_str(value: Any, *, field_path: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        raise TypeError(f'"{field_path}" must be a list of strings.')
    return [str(x) for x in value]


def _resolve_sources(
    *,
    section: dict[str, Any],
    key: str,
    allowed: Sequence[str],
    default_all: Sequence[str],
    field_path: str,
) -> list[str]:
    """
    Resolve source selections with notebook/CLI-friendly semantics.

    Rules:
    - missing key => default_all
    - value == "all" (case-insensitive) => default_all
    - list[str] => validated list
    - YAML null => not supported (error)

    Args:
        section: YAML mapping for the relevant subsection.
        key: Field key to resolve.
        allowed: Allowed tokens.
        default_all: Tokens used when "all" or missing.
        field_path: Full field path for error messages.

    Returns:
        List of resolved tokens.
    """
    if key not in section:
        return list(default_all)

    value = section.get(key)
    if value is None:
        raise TypeError(f'"{field_path}" must be a list of strings or "all" (YAML null is not supported).')

    if isinstance(value, str):
        if value.strip().lower() == "all":
            return list(default_all)
        raise TypeError(f'"{field_path}" must be a list of strings or "all".')

    items = _as_list_of_str(value, field_path=field_path)

    invalid = [x for x in items if x not in allowed]
    if invalid:
        raise ValueError(f'Invalid values in "{field_path}": {invalid}. Allowed: {list(allowed)}')

    return items


def load_config(cfg_path: str | Path) -> AppConfig:
    """
    Load unified NVV config (pipeline + evaluation) from YAML.

    Args:
        cfg_path: Path to config.yaml.

    Returns:
        AppConfig: Loaded and resolved configuration object.
    """
    cfg_path = Path(cfg_path).resolve()
    cfg = _load_yaml(cfg_path)

    project = get_project_paths(cfg_path)
    datasets = get_datasets(cfg_path, project=project)

    runtime_cfg = cfg.get("runtime", {}) or {}
    runtime = RuntimeConfig(
        device=str(runtime_cfg.get("device", "auto")),
        force=bool(runtime_cfg.get("force", False)),
    )

    pipe = cfg.get("pipeline", {}) or {}
    if not isinstance(pipe, dict):
        raise TypeError('"pipeline" must be a mapping.')

    p4 = pipe.get(KEY_STEP_4, {}) or {}
    p5 = pipe.get(KEY_STEP_5, {}) or {}
    p6 = pipe.get(KEY_STEP_6, {}) or {}
    p7 = pipe.get(KEY_STEP_7, {}) or {}

    if not isinstance(p4, dict) or not isinstance(p5, dict) or not isinstance(p6, dict) or not isinstance(p7, dict):
        raise TypeError('"pipeline.<step>" sections must be mappings.')

    step_4_vad = Step4VADConfig(
        vad_audios_in=_resolve_sources(
            section=p4,
            key="vad_audios_in",
            allowed=AUDIO_DERIVATIVES,
            default_all=AUDIO_DERIVATIVES,
            field_path=f"{KEY_STEP_4}.vad_audios_in",
        ),
        vad_threshold=_require_float(p4, "vad_threshold", field_path=f"{KEY_STEP_4}.vad_threshold"),
        vad_min_speech_ms=_require_int(p4, "vad_min_speech_ms", field_path=f"{KEY_STEP_4}.vad_min_speech_ms"),
        vad_min_silence_ms=_require_int(p4, "vad_min_silence_ms", field_path=f"{KEY_STEP_4}.vad_min_silence_ms"),
        vad_pad_ms=_require_int(p4, "vad_pad_ms", field_path=f"{KEY_STEP_4}.vad_pad_ms"),
    )

    step_5_asr = Step5ASRConfig(
        vad_masks_in=_resolve_sources(
            section=p5,
            key="vad_masks_in",
            allowed=VAD_MASKS,
            default_all=VAD_MASKS,
            field_path=f"{KEY_STEP_5}.vad_masks_in",
        ),
        asr_audios_in=_resolve_sources(
            section=p5,
            key="asr_audios_in",
            allowed=AUDIO_DERIVATIVES,
            default_all=AUDIO_DERIVATIVES,
            field_path=f"{KEY_STEP_5}.asr_audios_in",
        ),
        asr_chunk_length_s=_require_int(p5, "asr_chunk_length_s", field_path=f"{KEY_STEP_5}.asr_chunk_length_s"),
        asr_batch_size=_require_int(p5, "asr_batch_size", field_path=f"{KEY_STEP_5}.asr_batch_size"),
    )

    step_6_nlp = Step6NLPConfig(
        spacy_model=_require_str(p6, "spacy_model", field_path=f"{KEY_STEP_6}.spacy_model"),
    )

    exclude_categories = _require_list_of_str(
        p7,
        "exclude_categories",
        field_path=f"{KEY_STEP_7}.exclude_categories",
    )

    step_7_nvv = Step7NVVConfig(
        exclude_categories=exclude_categories,
        min_duration=_require_float(p7, "min_duration", field_path=f"{KEY_STEP_7}.min_duration"),
        max_duration=_require_float_or_none(p7, "max_duration", field_path=f"{KEY_STEP_7}.max_duration"),
        vad_masks_in=_resolve_sources(
            section=p7,
            key="vad_masks_in",
            allowed=VAD_MASKS,
            default_all=VAD_MASKS,
            field_path=f"{KEY_STEP_7}.vad_masks_in",
        ),
        asr_audios_in=_resolve_sources(
            section=p7,
            key="asr_audios_in",
            allowed=AUDIO_DERIVATIVES,
            default_all=AUDIO_DERIVATIVES,
            field_path=f"{KEY_STEP_7}.asr_audios_in",
        ),
        vad_gate_padding=_require_float(p7, "vad_gate_padding", field_path=f"{KEY_STEP_7}.vad_gate_padding"),
        dedup_overlap_ratio=_require_float(p7, "dedup_overlap_ratio", field_path=f"{KEY_STEP_7}.dedup_overlap_ratio"),
        dedup_time_tol_s=_require_float(p7, "dedup_time_tol_s", field_path=f"{KEY_STEP_7}.dedup_time_tol_s"),
    )

    export_cfg = cfg.get("export", {}) or {}
    export = ExportConfig(
        clips=[str(x) for x in export_cfg.get("clips", [])],
        labels=[str(x) for x in export_cfg.get("labels", [])],
    )

    eval_section = cfg.get("evaluation", {}) or {}
    gt_mode = eval_section.get("gt_mode")
    if gt_mode not in {"full_gt", "part_gt"}:
        raise ValueError(f'Invalid evaluation.gt_mode: {gt_mode}')

    evaluation = EvaluationConfig(
        gt_mode=str(gt_mode),
        gt_units=get_gt_units(cfg_path, project=project),
        gt_truth_paths=get_gt_truth_excel_paths(cfg_path, project=project),
    )

    return AppConfig(
        cfg_path=cfg_path,
        project=project,
        runtime=runtime,
        datasets=datasets,
        step_4_vad=step_4_vad,
        step_5_asr=step_5_asr,
        step_6_nlp=step_6_nlp,
        step_7_nvv=step_7_nvv,
        export=export,
        evaluation=evaluation,
    )


def ensure_workspace(cfg: AppConfig) -> None:
    """
    Create workspace folders in a safe, code-consistent way.

    Args:
        cfg: Loaded AppConfig.
    """
    ensure_workspace_dirs(cfg.datasets)


def print_config(cfg: AppConfig) -> None:
    """
    Print a human-readable snapshot of the loaded config.

    Args:
        cfg: Loaded AppConfig.
    """
    print("\n" + "=" * 80)
    print("NVV CONFIG PRINT (PIPELINE + EVALUATION)")
    print("=" * 80)

    print("\n[PROJECT]")
    print(f"  cfg_path:       {cfg.cfg_path}")
    print(f"  project_root:   {cfg.project.project_root}")
    print(f"  raw_root:       {cfg.project.raw_root}")
    print(f"  processed_root: {cfg.project.processed_root}")

    print("\n[RUNTIME]")
    print(f"  device: {cfg.runtime.device}")
    print(f"  force:  {cfg.runtime.force}")

    print("\n[DATASETS]")
    for i, ds in enumerate(cfg.datasets, start=1):
        ws = get_workspace_paths(ds.workspace)
        print(f"  ({i}) name:      {ds.name}")
        print(f"      input_dir:  {ds.input_dir}")
        print(f"      workspace:  {ds.workspace}")
        print(f"      per_audio:  {ws.per_audio}")
        print(f"      global:     {ws.global_dir}")
        print(f"      clips:      {ws.clips}")
        print(f"      evaluation: {ws.evaluation}")

    print("\n[PIPELINE]")
    print(f"  {KEY_STEP_4}:")
    print(f"    vad_audios_in:     {cfg.step_4_vad.vad_audios_in}")
    print(f"    vad_threshold:     {cfg.step_4_vad.vad_threshold}")
    print(f"    vad_min_speech_ms: {cfg.step_4_vad.vad_min_speech_ms}")
    print(f"    vad_min_silence_ms:{cfg.step_4_vad.vad_min_silence_ms}")
    print(f"    vad_pad_ms:        {cfg.step_4_vad.vad_pad_ms}")

    print(f"  {KEY_STEP_5}:")
    print(f"    vad_masks_in:      {cfg.step_5_asr.vad_masks_in}")
    print(f"    asr_audios_in:     {cfg.step_5_asr.asr_audios_in}")
    print(f"    asr_chunk_length_s:{cfg.step_5_asr.asr_chunk_length_s}")
    print(f"    asr_batch_size:    {cfg.step_5_asr.asr_batch_size}")

    print(f"  {KEY_STEP_6}:")
    print(f"    spacy_model:   {cfg.step_6_nlp.spacy_model}")

    print(f"  {KEY_STEP_7}:")
    print(f"    exclude_categories:  {cfg.step_7_nvv.exclude_categories}")
    print(f"    min_duration:        {cfg.step_7_nvv.min_duration}")
    print(f"    max_duration:        {cfg.step_7_nvv.max_duration}")
    print(f"    vad_masks_in:        {cfg.step_7_nvv.vad_masks_in}")
    print(f"    asr_audios_in:       {cfg.step_7_nvv.asr_audios_in}")
    print(f"    vad_gate_padding:    {cfg.step_7_nvv.vad_gate_padding}")
    print(f"    dedup_overlap_ratio: {cfg.step_7_nvv.dedup_overlap_ratio}")
    print(f"    dedup_time_tol_s:    {cfg.step_7_nvv.dedup_time_tol_s}")

    print("\n[EXPORT]")
    print(f"  clips:  {cfg.export.clips}")
    print(f"  labels: {cfg.export.labels}")

    print("\n[EVALUATION]")
    print(f"  gt_mode: {cfg.evaluation.gt_mode}")
    print(f"  gt_units: {[u.name for u in cfg.evaluation.gt_units]}")
    print("  gt_truth_paths:")
    for p in cfg.evaluation.gt_truth_paths:
        print(f"    - {p}")

    print("\n" + "=" * 80 + "\n")