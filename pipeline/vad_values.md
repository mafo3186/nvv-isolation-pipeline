# VAD Parameter Versions

## Legende
- **SILERO_SAMPLING_RATE**: Zielabtastrate für Silero (fest vorgegeben)
- **VAD_THRESHOLD**: Sensitivitätsschwelle (je niedriger, desto empfindlicher)
- **VAD_MIN_SPEECH_MS**: Mindestdauer eines Sprachsegments (in ms)
- **VAD_MIN_SILENCE_MS**: Mindestdauer, damit eine Pause als Trennung zählt (in ms)
- **VAD_PAD_MS**: Padding je Seite eines VAD-Segments (in ms)
- **VAD_SMOOTHING_WINDOW**: Fenstergröße für Energie-Glättung
- **VAD_ENERGY_REL_THRESHOLD**: relative Energie-Schwelle (bezogen auf Mittelwert)
- **VAD_EXPAND_PRE/POST**: Erweiterung des Segments vor/nach dem Zentrum (in Sekunden)
- **VAD_EXPAND_STEP**: Schrittweite beim Energie-basierten Ausdehnen (in Sekunden)

---

## V1 – Ursprüngliche Version
```
SILERO_SAMPLING_RATE = 16000

VAD_THRESHOLD = 0.3
VAD_MIN_SPEECH_MS = 75
VAD_MIN_SILENCE_MS = 75
VAD_PAD_MS = 50

VAD_SMOOTHING_WINDOW = 400
VAD_ENERGY_REL_THRESHOLD = 0.4
VAD_EXPAND_PRE = 0.01
VAD_EXPAND_POST = 0.01
VAD_EXPAND_STEP = 0.01
```

---

## V2 – Erste modifizierte Version
```
SILERO_SAMPLING_RATE = 16000

VAD_THRESHOLD = 0.2
VAD_MIN_SPEECH_MS = 50
VAD_MIN_SILENCE_MS = 50
VAD_PAD_MS = 75

VAD_SMOOTHING_WINDOW = 400
VAD_ENERGY_REL_THRESHOLD = 0.2
VAD_EXPAND_PRE = 0.02
VAD_EXPAND_POST = 0.02
VAD_EXPAND_STEP = 0.005
```

---

## V3 – Aktuelle Version
```
SILERO_SAMPLING_RATE = 16000

VAD_THRESHOLD = 0.2
VAD_MIN_SPEECH_MS = 50
VAD_MIN_SILENCE_MS = 50
VAD_PAD_MS = 40

VAD_SMOOTHING_WINDOW = 400
VAD_ENERGY_REL_THRESHOLD = 0.2
VAD_EXPAND_PRE = 0.015
VAD_EXPAND_POST = 0.015
VAD_EXPAND_STEP = 0.005
```

## V4 - näher an V1
```
# ---------- Audio defaults ----------
SILERO_SAMPLING_RATE = 16000  # Silero-VAD native rate

# ---------- Silero-VAD parameters ----------
VAD_THRESHOLD = 0.25              # Lower = more sensitive (high recall)
VAD_MIN_SPEECH_MS = 75           # Minimal accepted speech segment length (ms)
VAD_MIN_SILENCE_MS = 75          # Minimal silence gap (ms)
VAD_PAD_MS = 50                  # Context padding (ms) around detected speech

# ---------- Energy-based refinement ----------
VAD_SMOOTHING_WINDOW = 400       # Moving average window for smoothing
VAD_ENERGY_REL_THRESHOLD = 0.3   # Relative to mean energy to consider "speech"
VAD_EXPAND_PRE = 0.015            # Seconds: pre-extension before segment
VAD_EXPAND_POST = 0.015           # Seconds: post-extension after segment
VAD_EXPAND_STEP = 0.005           # Step size (seconds) during boundary refinement
```
