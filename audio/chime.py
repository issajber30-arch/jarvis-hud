"""Soft rising two-tone chime that confirms Jarvis is listening."""
from __future__ import annotations

import numpy as np


def chime_wave(rate: int = 24000) -> np.ndarray:
    n = int(0.14 * rate)
    freq = np.where(np.arange(n) < n // 2, 660.0, 990.0)
    wave = np.sin(np.cumsum(2 * np.pi * freq / rate))  # phase-continuous: no click at the step
    fade = int(0.01 * rate)
    env = np.ones(n)
    env[:fade] = np.linspace(0, 1, fade)
    env[-fade:] = np.linspace(1, 0, fade)
    return (0.22 * wave * env).astype(np.float32)


def play_chime(rate: int = 24000) -> None:
    try:
        import sounddevice as sd
        sd.play(chime_wave(rate), rate)  # non-blocking
    except Exception:  # no output device must never break listening
        pass
