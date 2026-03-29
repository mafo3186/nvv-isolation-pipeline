# --- NVV Pipeline – Tunable Parameters ---
#
# All parametrizable / algorithmically relevant values are
# defined here. Import from this module
# wherever a value influences pipeline behaviour or output quality.

from config.constants import AUDIO_STD_VOCALS, AUDIO_STD_BACKGROUND


# --- Step-1 Parameters ---

STEP1_WAV_SUBTYPE = "PCM_16"          # Output WAV subtype for standardized file


# --- Step-2 Parameters ---

STEP2_TARGET_SR = 44100               # Target sampling rate for source separation (UVR input)
STEP2_WAV_SUBTYPE = "PCM_16"          # Output WAV subtype for separated files

# UVR / MDX-Net Predictor params
# Higher DIM_F / N_FFT = more frequency resolution (slower); lower = faster but less precise
STEP2_MODEL_DIM_F = 3072
STEP2_MODEL_DIM_T = 8
STEP2_MODEL_N_FFT = 6144
STEP2_MODEL_CHUNKS = 10              # Number of chunks to split audio for inference
STEP2_MODEL_MARGIN = STEP2_TARGET_SR # Margin (samples) around chunk boundaries; 1 s at 44100 Hz
STEP2_MODEL_DENOISE = False          # Apply denoising in UVR predictor; True = slower, cleaner


# --- Step-3 Parameters ---

STEP3_TARGET_DBFS = -20.0            # RMS loudness target in dBFS; lower = quieter output
STEP3_LIMIT_DB = 3.0                 # Maximum gain applied (dB); limits over/under-amplification
STEP3_ANALYSIS_SAMPLING_RATE = 24000 # Output sample rate for normalized files; balances quality vs. speed
STEP3_WAV_SUBTYPE = "PCM_16"         # Output WAV subtype for normalized files

STEP3_AUDIO_INPUT = [AUDIO_STD_VOCALS, AUDIO_STD_BACKGROUND]  # Audio derivatives to normalize

# Safety peak normalization after RMS gain; True = prevents clipping
STEP3_APPLY_PEAK_SAFETY_NORMALIZE = True

# Missing-input policy:
#   True  => raise if expected input audio is missing (strict / evaluation-safe)
#   False => skip missing sources (lenient)
STEP3_RAISE_ON_MISSING_INPUT = True


# --- Step-4 Parameters ---

SILERO_SAMPLING_RATE = 16000         # Silero-VAD native rate; audio is resampled to this before VAD

# Energy-based boundary refinement (remain in params.py)
VAD_SMOOTHING_WINDOW = 400           # Moving-average window size (samples) for energy smoothing
VAD_ENERGY_REL_THRESHOLD = 0.3       # Fraction of mean energy below which a sample is considered silence
VAD_EXPAND_PRE = 0.015               # Pre-extension before segment boundary (seconds)
VAD_EXPAND_POST = 0.015              # Post-extension after segment boundary (seconds)
VAD_EXPAND_STEP = 0.005              # Step size (seconds) during iterative boundary expansion

EVAL_T_COLLAR = 0.2                  # Onset/offset collar (seconds) for event matching; lower = stricter
EVAL_PERCENTAGE_OF_LENGTH = 0.2      # Offset tolerance as fraction of GT event length; lower = stricter
EVAL_K_MAX = 42                      # Maximum k for greedy best-k selection (max = 42 datasets in current evaluation)

# --- Best-k params ---
EVAL_STOP_ON_NON_IMPROVEMENT = False # see all, or stop when F1 doesn't improve
EVAL_DELTA_F1_STOP = 1e-3            # Minimum F1 improvement to continue greedy forward selection,  no influence if STOP_ON_NON_IMPROVEMENT=False, else: minimum improvement to continue
EVAL_TOP_N = None                    # optional: 20, else None
EVAL_PERCENTAGE_OF_LENGTH = 0.2      # Offset tolerance as fraction of GT event length; lower = stricter
EVAL_DEDUP_EPS_S = 0.01              # or None (off). Deduplication time tolerance (seconds) when identifying identical candidate events across datasets for best-k selection;
EVAL_PRELOAD = True                 
EVAL_VERBOSE_MISSING = True 

# --- Matching params ---
EVAL_EVALUATE_ONSET = True           # Whether to evaluate onset detection; True = include onset in matching and metrics, False = ignore onset (offset-only evaluation)
EVAL_EVALUATE_OFFSET = True          # Whether to evaluate offset detection; True = include offset in matching and metrics, False = ignore offset (onset-only evaluation)
EVAL_MATCH_LABELS = False            # Whether to require matching labels for a valid match; True = only match events with the same label, False = ignore labels during matching (but still report them in metrics breakdown)

EVAL_CLIP_MODES = ["nvv"]            # e.g. eval_cfg.export_clips or similar later
EVAL_FORCE = False                   # Force re-evaluation even if results already exist; True = overwrite existing results, False = skip if results exist   

EVAL_K_OVERRIDE = None               # Optional: set to an int to override best-k selection with a fixed k value for all datasets; None = use greedy selection
EVAL_WEAK_RECALL_THRESHOLD = 0.2     # Minimum recall to consider a GT event "weakly detected" in the report; lower = more GT events counted as weakly detected