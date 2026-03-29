#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
spaCy-based lexical classification for Step 6 (Speechmask)
-----------------------------------------------------------
Provides:
- init_spacy_en()       → loads English spaCy model
- classify_token()      → assigns token category
- classify_chunk()      → determines chunk category from tokens
- analyze_asr_text()    → runs full analysis on a chunk text
"""

import re
import spacy
from wordfreq import zipf_frequency


# --- Regex Helpers ---

# accepts [UH], [UM]., [BREATH]! etc.
BRACKET_WITH_TRAIL_PUNCT_RE = re.compile(r"^\s*\[[^\]]+\]\s*[\.\!\?,;:]*\s*$")
# detects marker anywhere in text (for mixed cases)
BRACKET_ANYWHERE_RE = re.compile(r"\[[^\]]+\]")

def is_bracket_chunk(text: str) -> bool:
    """True if chunk consists only of [MARKER] – optionally followed by punctuation."""
    return bool(BRACKET_WITH_TRAIL_PUNCT_RE.match(text.strip()))

def contains_bracket(text: str) -> bool:
    """True if any [MARKER] sequence appears anywhere in the text (for priority in mixed cases)."""
    return bool(BRACKET_ANYWHERE_RE.search(text))


# --- Init spaCy ---

def init_spacy_en(model_name: str = "en_core_web_lg", auto_download: bool = True):
    """
    Load a lightweight English spaCy model safely.
    - model_name: e.g. "en_core_web_sm"
    - auto_download: download model if not installed
    Returns spaCy Language object.
    """
    from spacy.cli import download

    print(f"🔹 Loading spaCy model '{model_name}' ...")
    try:
        nlp = spacy.load(model_name, disable=["ner", "parser"])
    except OSError:
        if auto_download:
            print(f"⚠️  Model '{model_name}' not found, downloading...")
            download(model_name)
            nlp = spacy.load(model_name, disable=["ner", "parser"])
        else:
            raise RuntimeError(
                f"Model '{model_name}' not found and auto_download=False."
            )

    nlp.max_length = 2_000_000
    if not nlp.vocab.vectors:
        print("⚠️ Warning: spaCy model has no word vectors. 'is_oov' will always be True!")

    print("✅ spaCy model loaded successfully.")
    return nlp


# --- Token Classification ---

def classify_token(tok) -> str:
    """
    Classify a spaCy token into one of:
    'non_word', 'filler', 'word', 'oov', 'unknown'

    Notes:
    - 'INTJ' (Interjections) → filler
    - 'X' (unclassifiable, often sounds like 'uh', 'hmm', etc.) → non_word
    - 'SYM' and 'PUNCT' → ignored (should not be processed)
    - covers fine-grained tags like AFX, FW, MD, XX
    """
    text = tok.text.strip()

    # --- ignore punctuation and symbols completely ---
    if tok.pos_ in {"PUNCT", "SYM"} or not text:
        return "ignore"

    # --- explicit Whisper marker like [UH] (incl. trailing punct) ---
    if BRACKET_WITH_TRAIL_PUNCT_RE.match(text):
        return "non_word"

    # --- interjection / filler ---
    if tok.pos_ == "INTJ":
        return "filler"

    # --- prefixes like 'pre-' or 'anti-' (AFX) ---
    if tok.tag_ == "AFX":
        return "word"

    # --- modal verbs (can, should, will...) ---
    if tok.tag_ == "MD":
        return "word"

    # --- foreign words or mixed script (FW) ---
    if tok.tag_ == "FW":
        return "oov"

    # --- unclassified garbage tokens (XX) ---
    if tok.tag_ == "XX":
        return "non_word"

    # --- sound-like, unknown, or foreign tokens ---
    if tok.pos_ == "X":
        return "non_word"

    # --- named entity or stopword ⇒ word ---
    if tok.is_stop or (hasattr(tok, "ent_type_") and tok.ent_type_):
        return "word"

    # --- numerical words also counted as word ---
    if tok.pos_ == "NUM":
        return "word"

    # --- lexically valid alphabetic words ---
    if tok.is_alpha and not tok.is_oov and tok.pos_ in {
        "NOUN", "VERB", "ADJ", "ADV", "PROPN", "PRON",
        "DET", "AUX", "ADP", "CCONJ", "SCONJ", "NUM"
    }:
        return "word"

    # misclassified proper nouns (e.g. 'hmm', 'öhr')
    if tok.pos_ == "PROPN" and tok.is_oov and tok.text.islower():
        return "non_word"

    # --- remaining OOV alphabetic items ---
    if tok.is_oov and tok.is_alpha:
        return "oov"

    # --- fallback ---
    return "unknown"



def classify_chunk(token_cats) -> str:
    """
    Determine overall chunk category by priority:
    non_word > filler > word > oov > unknown
    """
    if not token_cats:
        return "unknown"
    for cat in ["non_word", "filler", "word", "oov", "unknown"]:
        if cat in token_cats:
            return cat
    return "unknown"


# --- Chunk Text Analysis ---

def analyze_asr_text(nlp, text: str):
    """
    Analyze one ASR chunk text with spaCy.
    Returns (chunk_category, token_info_dict, token_pairs_list)
      token_pairs = [(token_text, category), ...]
    """
    text = text.strip()
    text = text.strip("-–.,!?;: ") # remove leading/trailing punctuation

    token_info = {}
    token_pairs = []

    # early exit for pure bracket markers
    if is_bracket_chunk(text):
        return "non_word", {}, [(text.strip(), "non_word")]

    doc = nlp(text)
    token_cats = []

    for tok in doc:
        cat = classify_token(tok)

        # --- skip ignored tokens (PUNCT, SYM, empty) ---
        if cat == "ignore":
            continue

        token_cats.append(cat)
        token_pairs.append((tok.text.lower(), cat))

        # --- gather token info including Zipf frequency ---
        zipf_val = zipf_frequency(tok.text.lower(), "en")
        token_info[tok.text.lower()] = {
            "pos": tok.pos_,
            "lemma": tok.lemma_,
            "oov": tok.is_oov,
            "zipf": round(zipf_val, 3),
        }

    # priority: if [...] appears anywhere → non_word
    if contains_bracket(text):
        token_cats.append("non_word")

    chunk_cat = classify_chunk(token_cats)
    return chunk_cat, token_info, token_pairs

