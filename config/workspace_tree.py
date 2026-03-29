"""
Visible workspace tree model.

This module documents the intended workspace structure from workspace level
downwards. It is declarative only and does not resolve paths itself.
"""

from config.constants import (
    KEY_PER_AUDIO,
    KEY_GLOBAL,
    KEY_CLIPS,
    KEY_EVALUATION,
    KEY_AUDIO_FILES,
    KEY_ANNOTATIONS,
    KEY_LABELS,
    KEY_AUDIO_ID,
    KEY_RQ,
    KEY_VAD,
    KEY_ASR,
    KEY_NLP,
    KEY_NVV,
    KEY_METADATA,
)

WORKSPACE_TREE = {
    KEY_PER_AUDIO: {
        f"{{{KEY_AUDIO_ID}}}": {
            KEY_AUDIO_FILES: {},
            KEY_ANNOTATIONS: {
                KEY_VAD: {},
                KEY_ASR: {},
                KEY_NLP: {},
                KEY_NVV: {},
            },
            KEY_LABELS: {
                KEY_VAD: {},
                KEY_ASR: {},
                KEY_NVV: {},
            },
        }
    },
    KEY_GLOBAL: {
        KEY_CLIPS: {
            "{gt_mode}": {},
        },
        KEY_EVALUATION: {
            "{gt_mode}": {
                KEY_RQ: {},
            }
        },
    },
    KEY_METADATA: {},
}