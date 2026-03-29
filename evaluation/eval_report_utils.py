from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
from IPython.display import display, Markdown


@dataclass(frozen=True)
class DetectabilityReportConfig:
    mode: Optional[str] = "full_gt"       # None = alle
    weak_recall_threshold: float = 0.2

    # default: compact, no per-audio tables
    show_per_audio_tables: bool = False

    # which IDs should receive detail tables at all?
    # "undetectable" | "weak" | "all"
    per_audio_scope: str = "undetectable"

    # Hard cap, damit 300 IDs nie explodieren
    per_audio_max_ids: int = 10

    # pro ID: wie viele Tracks anzeigen
    per_audio_top_n_tracks: int = 12

    show_track_hit_summary: bool = True

    # column names (for easy renaming)
    audio_id_col: str = "audio_id"
    combo_key_col: str = "combo_key"
    vad_mask_col: str = "vad_mask"
    asr_audio_in_col: str = "asr_audio_in"
    tp_col: str = "tp"
    fn_col: str = "fn"
    fp_col: str = "fp"
    precision_col: str = "precision"
    recall_col: str = "recall"
    f1_col: str = "f1"


def _filter_mode(df: pd.DataFrame, mode: Optional[str]) -> pd.DataFrame:
    if mode is None or "mode" not in df.columns:
        return df
    return df[df["mode"].astype(str) == str(mode)].copy()


def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    return out


def _audio_debug_table(
    summary_all: pd.DataFrame,
    *,
    audio_id: str,
    cfg: DetectabilityReportConfig,
) -> pd.DataFrame:
    df = summary_all.copy()
    df = _filter_mode(df, cfg.mode)
    df = df[df[cfg.audio_id_col].astype(str) == str(audio_id)].copy()

    df = _coerce_numeric(
        df,
        [cfg.tp_col, cfg.fn_col, cfg.fp_col, cfg.precision_col, cfg.recall_col, cfg.f1_col],
    )

    # sort = "best candidates" first
    df = df.sort_values(
        by=[cfg.f1_col, cfg.tp_col, cfg.recall_col, cfg.precision_col, cfg.fp_col],
        ascending=[False, False, False, False, True],
    )

    keep_cols = [
        cfg.vad_mask_col,
        cfg.asr_audio_in_col,
        cfg.combo_key_col,
        cfg.tp_col, cfg.fn_col, cfg.fp_col,
        cfg.precision_col, cfg.recall_col, cfg.f1_col,
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    return df[keep_cols].reset_index(drop=True)


def _generate_interpretation_text(
    *,
    n_audio: int,
    n_combos: int,
    n_detectable: int,
    n_undetectable: int,
) -> str:
    """
    Generate structured interpretation text for notebook or export.
    """

    text = f"""
        ## What this report answers

        ### 1) Audio-level detectability

        - For each audio_id, we check whether ANY track (combo_key) achieves TP>0.
        - If tp_max == 0, that audio_id is currently "undetectable" for this pipeline/config,
        given the current ground truth.

        ### 2) Track-level behavior
        - For each combo_key, we count how many audio_ids it hits (TP>0 at least once).
        - We report:
        • mean metrics across ALL audio_ids (macro mean)
        • mean metrics only on hits
        
        ---

        ## Key numbers (this run)

        - audio_ids evaluated: {n_audio}
        - candidate tracks (combo_keys) present: {n_combos}
        - detectable audio_ids (TP>0 in any track): {n_detectable}
        - undetectable audio_ids (TP==0 for all tracks): {n_undetectable}

        ---

        ## How to interpret common patterns

        ### If many audio_ids are undetectable (tp_max == 0)
        • Either the pipeline is missing those events (model/VAD/NLP limitations),  
        • Or the ground-truth is incomplete / mismatched to what the pipeline emits,  
        • Or candidate timing/labels are incompatible with the matching configuration
        (collars, onset/offset rules).

        ### If several top tracks hit exactly the SAME audio_ids
        • There is little/no complementarity between tracks on this dataset.  
        • Greedy best-k union will not improve (plateau at k=1).

        ### If union adds tracks but FP increases while TP does not
        • Added tracks contribute mostly false positives.  
        • Best-k should remain small (often k=1).

        ### If you subjectively hear NVVs but TP remains 0
        • Strong indication of missing/partial GT  
        • OR systematic mismatch in boundary alignment vs matching rule.

        ---

        ## Recommended follow-up

        - Inspect undetectable audio_ids:
        • listen to candidates vs GT timestamps  
        • check FN vs missing GT  

        - Compare best combo per audio_id:
        • heterogeneity supports track fusion  
        • identical winners suggest redundancy
        """

    return text.strip()


def build_detectability_report(
    summary_all: pd.DataFrame,
    *,
    cfg: Optional[DetectabilityReportConfig] = None,
) -> Dict[str, object]:
    """
    Build + DISPLAY a formatted notebook report.

    Returns a dict with:
      - overview_df
      - per_audio_agg_df
      - undetectable_df
      - weak_df
      - track_hit_df (optional)
      - debug_tables (dict[audio_id] -> df)
    """
    cfg = cfg or DetectabilityReportConfig()

    # --- Prepare / validate ---
    df = summary_all.copy()
    df = _filter_mode(df, cfg.mode)

    required = {
        cfg.audio_id_col,
        cfg.combo_key_col,
        cfg.tp_col,
        cfg.fp_col,
        cfg.fn_col,
        cfg.recall_col,
        cfg.f1_col,
        cfg.vad_mask_col,
        cfg.asr_audio_in_col,
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"summary_all is missing required columns: {missing}")

    df = _coerce_numeric(df, [cfg.tp_col, cfg.fn_col, cfg.fp_col, cfg.precision_col, cfg.recall_col, cfg.f1_col])

    # hit flag: TP >= 1
    df["hit"] = df[cfg.tp_col] > 0

    # --- Per-audio aggregation (max over tracks) ---
    per_audio_agg = (
        df.groupby(cfg.audio_id_col, as_index=False)
          .agg(
              tp_max=(cfg.tp_col, "max"),
              recall_max=(cfg.recall_col, "max"),
              f1_max=(cfg.f1_col, "max"),
              fp_sum=(cfg.fp_col, "sum"),
              fn_sum=(cfg.fn_col, "sum"),
              n_tracks=(cfg.combo_key_col, "nunique"),
          )
    )
    per_audio_agg["undetectable"] = per_audio_agg["tp_max"] <= 0
    per_audio_agg["weak"] = per_audio_agg["recall_max"] < float(cfg.weak_recall_threshold)

    undetectable_df = per_audio_agg[per_audio_agg["undetectable"]].copy()
    weak_df = per_audio_agg[per_audio_agg["weak"]].copy()

    # --- Overview ---
    n_audio = int(per_audio_agg.shape[0])
    n_undetectable = int(undetectable_df.shape[0])
    n_weak = int(weak_df.shape[0])
    n_detectable = int((per_audio_agg["undetectable"] == False).sum())
    n_combos = int(df[cfg.combo_key_col].nunique())
    interpretation_text = _generate_interpretation_text(
        n_audio=n_audio,
        n_combos=n_combos,
        n_detectable=n_detectable,
        n_undetectable=n_undetectable,
    )

    # interpretability: undetectable with fp_sum>0 suggests GT/matching gap; fp_sum==0 suggests pipeline miss
    undet_fp_gt_gap = int((undetectable_df["fp_sum"] > 0).sum()) if n_undetectable else 0
    undet_no_cand = int((undetectable_df["fp_sum"] <= 0).sum()) if n_undetectable else 0

    overview_df = pd.DataFrame(
        [
            {"metric": "mode", "value": cfg.mode if cfg.mode is not None else "ALL"},
            {"metric": "n_audio_ids", "value": n_audio},
            {"metric": "n_undetectable (tp_max==0)", "value": n_undetectable},
            {"metric": "n_detectable", "value": n_detectable},
            {"metric": "n_weak (recall_max < threshold)", "value": n_weak},
            {"metric": "weak_recall_threshold", "value": float(cfg.weak_recall_threshold)},
            {"metric": "undetectable_with_fp>0 (candidates exist, but never match)", "value": undet_fp_gt_gap},
            {"metric": "undetectable_with_fp==0 (no candidates at all)", "value": undet_no_cand},
        ]
    )

    # --- Track hit summary (optional) ---
    track_hit_df = None
    if cfg.show_track_hit_summary:
        track_hit_df = (
            df.groupby(cfg.combo_key_col, as_index=False)
              .agg(
                  n_audio_ids_with_hit=("hit", lambda s: int(s.sum())),
                  mean_f1_all=(cfg.f1_col, "mean"),
                  mean_f1_on_hits=(
                      cfg.f1_col,
                      lambda s: float(s[df.loc[s.index, "hit"]].mean()) if (df.loc[s.index, "hit"].any()) else 0.0,
                  ),
              )
              .sort_values(by=["n_audio_ids_with_hit", "mean_f1_all"], ascending=[False, False])
              .reset_index(drop=True)
        )

    # --- Per-audio debug tables (optional, capped) ---
    debug_tables: Dict[str, pd.DataFrame] = {}

    def _pick_debug_ids() -> List[str]:
        if not cfg.show_per_audio_tables:
            return []

        scope = str(cfg.per_audio_scope).lower().strip()
        if scope not in {"undetectable", "weak", "all"}:
            raise ValueError("per_audio_scope must be one of: 'undetectable', 'weak', 'all'")

        if scope == "undetectable":
            base = undetectable_df.copy()
            # "most informative": those with FP>0 first (possible GT mismatch), then high FN
            base = base.sort_values(["fp_sum", "fn_sum"], ascending=[False, False])
            ids = base[cfg.audio_id_col].astype(str).tolist()

        elif scope == "weak":
            base = weak_df.copy()
            # weakest first
            base = base.sort_values(["recall_max", "tp_max"], ascending=[True, True])
            ids = base[cfg.audio_id_col].astype(str).tolist()

        else:  # all
            base = per_audio_agg.copy()
            base = base.sort_values(["recall_max", "tp_max"], ascending=[True, True])
            ids = base[cfg.audio_id_col].astype(str).tolist()

        return ids[: int(cfg.per_audio_max_ids)]

    debug_ids = _pick_debug_ids()
    if cfg.show_per_audio_tables and debug_ids:
        for aid in debug_ids:
            tbl = _audio_debug_table(df, audio_id=aid, cfg=cfg).head(int(cfg.per_audio_top_n_tracks))
            debug_tables[str(aid)] = tbl

    # --- DISPLAY (formatted) ---
    title = f"Detectability Report — mode={cfg.mode}" if cfg.mode is not None else "Detectability Report — mode=ALL"
    display(Markdown(f"## {title}"))

    display(Markdown("### Overview"))
    display(overview_df)
    display(Markdown(f"Interpretation: {interpretation_text}"))

    display(Markdown("### Per-audio aggregation (max over tracks) — head"))
    display(per_audio_agg.sort_values(["undetectable", "weak", "recall_max"], ascending=[False, False, True]).head(15))

    display(Markdown("### Undetectable audio_ids (tp_max==0)"))
    if n_undetectable == 0:
        display(Markdown("✅ None"))
    else:
        # nicer ordering: show ones with FP (possible GT gap) first
        display(undetectable_df.sort_values(["fp_sum", "fn_sum"], ascending=[False, False]).reset_index(drop=True))

    display(Markdown(f"### Weak audio_ids (recall_max < {cfg.weak_recall_threshold})"))
    if n_weak == 0:
        display(Markdown("✅ None"))
    else:
        display(weak_df.sort_values(["recall_max", "tp_max"], ascending=[True, True]).reset_index(drop=True))

    if cfg.show_track_hit_summary:
        display(Markdown("### Track hit summary (how many audio_ids have at least one TP per track) — top 25"))
        display(track_hit_df.head(25))

    if cfg.show_per_audio_tables and debug_tables:
        display(Markdown(f"### Debug tables (capped to {cfg.per_audio_max_ids} audio_ids; scope={cfg.per_audio_scope})"))
        for aid, tbl in debug_tables.items():
            # hint from per_audio_agg
            row = per_audio_agg.loc[per_audio_agg[cfg.audio_id_col].astype(str) == str(aid)]
            fp_any = float(row["fp_sum"].iloc[0]) > 0 if len(row) else False
            hint = (
                "Candidates exist (FP>0) but never match GT → possible timing mismatch / incomplete GT / collars."
                if fp_any
                else "No candidates in any track → likely pipeline miss (VAD/ASR/NLP gating)."
            )
            display(Markdown(f"**audio_id = `{aid}`**  \n_{hint}_"))
            display(tbl)

    return {
        "overview_df": overview_df,
        "per_audio_agg_df": per_audio_agg,
        "undetectable_df": undetectable_df,
        "weak_df": weak_df,
        "track_hit_df": track_hit_df,
        "debug_tables": debug_tables,
    }