"""Mic capture: wake word / push-to-talk start an utterance, Silero VAD ends it."""
from __future__ import annotations

import threading
import time
from typing import Callable

import numpy as np

import config
from contracts import AudioTelemetryFrame, ListenerEvent

_EDGES = np.unique(np.geomspace(1, 640, 65).astype(int))  # rfft of 1280 samples has 641 bins


def fft_frame(block: np.ndarray, vad: bool) -> AudioTelemetryFrame:
    """64 log-spaced spectrum bins, 0..1 on a fixed -90..0 dB scale."""
    block = np.asarray(block, dtype=np.float32)
    n = max(len(block), 1)
    mag = np.abs(np.fft.rfft(block * np.hanning(n))) / n * 4
    edges = np.geomspace(1, max(len(mag) - 1, 2), 65).astype(int)
    edges[-1] = max(edges[-1], edges[-2] + 1)
    bins = np.array([mag[a:max(b, a + 1)].mean() for a, b in zip(edges[:-1], edges[1:])])
    fft = np.clip((20 * np.log10(bins + 1e-9) + 90) / 90, 0, 1).astype(np.float32)
    rms = float(np.sqrt(np.mean(block**2))) if len(block) else 0.0
    return AudioTelemetryFrame(rms=rms, fft=fft, vad=bool(vad), ts=time.time())


class Listener:
    def __init__(self, on_event: Callable[[ListenerEvent], None],
                 on_audio: Callable[[AudioTelemetryFrame], None], stream_factory=None,
                 wake_fn=None, vad_fn=None):
        self._emit, self._on_audio = on_event, on_audio
        self._factory = stream_factory
        self._wake_fn, self._vad_fn = wake_fn, vad_fn
        self._lock = threading.RLock()
        self._stream = None
        self._speaking = False
        self._followup_until = 0.0
        self._mode: str | None = None  # None (idle) | "wake" | "ptt" | "followup"
        self._buf: list[np.ndarray] = []
        self._heard = False
        self._silence_ms = 0.0
        self._block_ms = config.BLOCK_SIZE * 1000 / config.SAMPLE_RATE

    # --- lifecycle ---
    def start(self):
        if self._wake_fn is None:
            self._wake_fn = self._default_wake()
        if self._vad_fn is None:
            self._vad_fn = self._default_vad()
        if self._factory is None:
            import sounddevice as sd
            self._factory = sd.InputStream
        self._stream = self._factory(samplerate=config.SAMPLE_RATE, blocksize=config.BLOCK_SIZE,
                                     channels=1, dtype="float32", callback=self._callback)
        self._stream.start()

    def stop(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    @staticmethod
    def _default_wake():
        import sys
        import types
        from ai_edge_litert import interpreter
        # openWakeWord imports tflite_runtime (no arm64 wheel) and silently falls back to broken ONNX;
        # ai-edge-litert is a drop-in, so alias it.
        pkg = types.ModuleType("tflite_runtime")
        pkg.interpreter = interpreter
        sys.modules.setdefault("tflite_runtime", pkg)
        sys.modules.setdefault("tflite_runtime.interpreter", interpreter)
        from openwakeword.model import Model
        model = Model(wakeword_models=[config.WAKE_MODEL], inference_framework=config.WAKE_FRAMEWORK)

        def score(block):
            pcm = (np.clip(block, -1, 1) * 32767).astype(np.int16)
            return max(model.predict(pcm).values(), default=0.0)
        assert "onnx" not in type(next(iter(model.models.values()))).__module__, "wake model fell back to ONNX"
        score.reset = model.reset
        return score

    @staticmethod
    def _default_vad():
        import torch
        from silero_vad import load_silero_vad
        model = load_silero_vad()

        def prob(chunk):
            return float(model(torch.from_numpy(np.ascontiguousarray(chunk)), config.SAMPLE_RATE).item())
        prob.reset = model.reset_states
        return prob

    # --- controls ---
    def set_speaking(self, speaking: bool):
        self._speaking = speaking

    def open_followup(self, seconds: float):
        self._followup_until = time.monotonic() + seconds

    def ptt_down(self):
        with self._lock:
            if self._speaking:
                self._emit(ListenerEvent("barge_in"))
            self._emit(ListenerEvent("ptt_start"))
            self._begin("ptt")

    def ptt_up(self):
        with self._lock:
            if self._mode != "ptt":
                return
            self._emit(ListenerEvent("ptt_end"))
            self._finish()

    # --- audio path ---
    def _callback(self, indata, frames, time_info, status):
        self._on_block(indata[:, 0].copy())

    def _vad(self, block) -> float:
        # Silero wants 512-sample chunks; a 1280 block holds two.
        return max(self._vad_fn(block[i:i + 512]) for i in (0, 512))

    def _begin(self, mode: str, first: np.ndarray | None = None):
        self._mode, self._buf, self._heard, self._silence_ms = mode, [], False, 0.0
        if first is not None:
            self._buf.append(first)
        for fn in (self._vad_fn, self._wake_fn):
            getattr(fn, "reset", lambda: None)()

    def _finish(self):
        audio = np.concatenate(self._buf).astype(np.float32) if self._buf else np.zeros(0, np.float32)
        self._mode, self._buf = None, []
        if len(audio):
            self._emit(ListenerEvent("utterance", audio))

    def _on_block(self, block: np.ndarray):
        with self._lock:
            speech = self._vad(block) >= config.VAD_THRESHOLD
            self._on_audio(fft_frame(block, speech))
            if self._mode is None:
                self._idle(block, speech)
                return
            self._buf.append(block)
            n = sum(len(b) for b in self._buf)
            if n >= config.MAX_UTTERANCE_S * config.SAMPLE_RATE:
                if self._mode == "ptt":
                    self._emit(ListenerEvent("ptt_end"))
                self._finish()
            elif self._mode != "ptt":
                if speech:
                    self._heard, self._silence_ms = True, 0.0
                else:
                    self._silence_ms += self._block_ms
                    limit = config.SILENCE_END_MS if self._heard else 5000  # no speech after wake: give up
                    if self._silence_ms >= limit:
                        self._finish() if self._heard else self._abandon()

    def _abandon(self):
        self._mode, self._buf = None, []

    def _idle(self, block, speech):
        thr = config.WAKE_THRESHOLD_SPEAKING if self._speaking else config.WAKE_THRESHOLD
        if self._wake_fn(block) >= thr:
            if self._speaking:
                self._emit(ListenerEvent("barge_in"))
            self._emit(ListenerEvent("wake"))
            self._begin("wake")
        elif speech and time.monotonic() < self._followup_until:
            self._followup_until = 0.0
            self._begin("followup", block)
            self._heard = True
