"""Right-rail voice panel: live mic level, noise floor, clarity, spectrum and what Jarvis last heard."""
from __future__ import annotations

from collections import deque

import numpy as np
from rich.text import Text
from textual.widget import Widget

import config
from contracts import AudioTelemetryFrame
from hud.telemetry_widget import _bar, _graph

FLOOR_DB, CEIL_DB = -80.0, 0.0
BLOCKS_PER_S = config.SAMPLE_RATE / config.BLOCK_SIZE  # 12.5 mic frames a second


def _db(rms: float) -> float:
    return max(20 * np.log10(max(rms, 1e-9)), FLOOR_DB)


def _pct(db: float) -> float:
    return (db - FLOOR_DB) / (CEIL_DB - FLOOR_DB) * 100


class VoiceMeter:
    """Pure signal maths, kept apart from drawing so it can be tested directly."""

    def __init__(self) -> None:
        self.level_db = FLOOR_DB
        self.frames = 0
        self.vad = False
        self.fft = np.zeros(64, np.float32)
        self._noise: deque[float] = deque(maxlen=int(5 * BLOCKS_PER_S))   # last ~5 s of non-speech
        self._speech: deque[float] = deque(maxlen=int(2 * BLOCKS_PER_S))  # last ~2 s of speech
        self.history: deque[float] = deque(maxlen=int(8 * BLOCKS_PER_S))

    def push(self, f: AudioTelemetryFrame) -> None:
        self.frames += 1
        self.level_db, self.vad, self.fft = _db(f.rms), f.vad, f.fft
        (self._speech if f.vad else self._noise).append(self.level_db)
        self.history.append(_pct(self.level_db))

    @property
    def noise_db(self) -> float:
        # 20th percentile of quiet moments: ignores brief bumps, tracks the steady room sound
        return float(np.percentile(self._noise, 20)) if self._noise else -60.0

    @property
    def speech_db(self) -> float | None:
        return float(np.median(self._speech)) if self._speech else None

    def clarity(self) -> tuple[str, float | None]:
        if self.speech_db is None:
            return "WAITING", None
        snr = self.speech_db - self.noise_db
        label = "EXCELLENT" if snr >= 20 else "GOOD" if snr >= 12 else "FAIR" if snr >= 6 else "POOR"
        return label, snr

    def warnings(self) -> list[str]:
        out = []
        if self.noise_db > -35:
            out.append("NOISY ROOM")
        if self.level_db > -3:
            out.append("TOO LOUD")
        return out


class VoiceWidget(Widget):
    DEFAULT_CSS = "VoiceWidget { border: round #6B3F0E; padding: 0 1; height: 15; }"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.border_title = "VOICE"
        self.meter = VoiceMeter()
        self.wake = 0.0
        self.last: tuple[str, float, float] | None = None
        self.off_reason: str | None = None
        self._dirty = False

    def on_mount(self) -> None:
        # Repaint at most 10x/s and only when new mic data arrived; idle panels cost nothing per frame.
        self.set_interval(0.1, self._repaint_if_dirty)

    def _repaint_if_dirty(self) -> None:
        if self._dirty:
            self._dirty = False
            self.refresh()

    def push(self, frame: AudioTelemetryFrame, wake: float | None = None) -> None:
        self.meter.push(frame)
        self._dirty = True
        if wake is not None:
            self.wake = max(float(wake), self.wake * 0.85)  # hold peaks briefly so they are readable

    def heard(self, text: str, audio_s: float, stt_s: float) -> None:
        self.last = (text, audio_s, stt_s)
        self.refresh()

    def mic_off(self, reason: str) -> None:
        self.off_reason = reason
        self.refresh()

    def render(self) -> Text:
        hi, dim, glow, warn = config.HUD_PRIMARY, config.HUD_DIM, config.HUD_GLOW, config.HUD_ERROR
        t = Text(no_wrap=True, overflow="crop")
        if self.off_reason:
            return t.append(f"MIC OFF · {self.off_reason}", style=dim)
        m = self.meter
        w = max(self.size.width, 30)
        bar_w = w - 16

        t.append("● HEARING SPEECH" if m.vad else "○ QUIET         ", style=glow if m.vad else dim)
        pips = round(min(self.wake, 1.0) * 5)
        t.append("   WAKE ", style=dim)
        t.append("▮" * pips + "▯" * (5 - pips) + "\n", style=glow if self.wake >= config.WAKE_THRESHOLD else hi)

        for label, db in (("LEVEL", m.level_db), ("NOISE", m.noise_db)):
            t.append(f"{label:<6} ", style=dim)
            t.append(_bar(_pct(db), bar_w), style=warn if label == "LEVEL" and db > -3 else hi)
            t.append(f"{db:5.0f} dB\n", style=hi)

        label, snr = m.clarity()
        t.append("CLARITY ", style=dim)
        t.append(label, style=f"bold {warn if label == 'POOR' else glow}")
        t.append(f" · {snr:.0f} dB ABOVE NOISE\n" if snr is not None else " · SAY SOMETHING\n", style=dim)
        flags = m.warnings()
        t.append(("⚠ " + " · ".join(flags)) if flags else "", style=warn)
        t.append("\n")

        graph_w = w - 2
        bins = np.interp(np.linspace(0, len(m.fft) - 1, graph_w), np.arange(len(m.fft)), m.fft)
        for row in _graph(bins, graph_w, 3, 1.0):
            t.append(row + "\n", style=hi)
        for row in _graph(m.history, graph_w, 2, 100):
            t.append(row + "\n", style=dim)

        t.append("HEARD ", style=dim)
        if self.last:
            text, audio_s, stt_s = self.last
            t.append(f'"{text.upper()}"', style=glow)
            t.append(f" · {audio_s:.1f} S · STT {stt_s:.2f} S", style=dim)
        else:
            t.append("—", style=dim)
        return t
