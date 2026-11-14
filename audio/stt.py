"""Local speech-to-text via mlx-whisper."""
from __future__ import annotations

import numpy as np

import config
import mlx_whisper

MIN_SECONDS = 0.3
# Whisper's well-known output on silence/noise.
HALLUCINATIONS = {"thank you.", "thank you", "you", "thanks for watching", "thanks for watching!", "thanks for watching."}


def transcribe(audio: np.ndarray) -> str:
    audio = np.asarray(audio, dtype=np.float32)
    if len(audio) < MIN_SECONDS * config.SAMPLE_RATE:
        return ""
    result = mlx_whisper.transcribe(audio, path_or_hf_repo=config.WHISPER_MODEL, language="en")
    text = result.get("text", "").strip()
    if text.lower() in HALLUCINATIONS or not any(c.isalnum() for c in text):
        return ""
    return text
