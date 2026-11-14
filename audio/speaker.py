"""Kokoro TTS playback: synthesise sentence N+1 while N plays; stop() silences within one 10 ms block."""
from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncIterator, Callable

import numpy as np

import config
from audio.listener import fft_frame
from contracts import AudioTelemetryFrame

TTS_RATE = 24000
BLOCK = TTS_RATE // 100  # 10 ms
_MODEL_ID = "mlx-community/Kokoro-82M-bf16"


def clone_voice_available() -> bool:
    return (config.REFERENCE_VOICE_DIR / "jarvis_ref.wav").exists()


class Speaker:
    def __init__(self, on_audio: Callable[[AudioTelemetryFrame], None], stream_factory=None, synth=None):
        self._on_audio = on_audio
        self._factory = stream_factory
        self._synth_fn = synth
        self._model = None
        self._stop = threading.Event()
        self._stream = None
        self._speaking = False
        self._tts_pool = ThreadPoolExecutor(1)  # MLX: one synthesis at a time
        self._play_pool = ThreadPoolExecutor(1)

    @property
    def is_speaking(self) -> bool:
        return self._speaking

    def stop(self):
        self._stop.set()
        stream = self._stream
        if stream is not None:
            stream.abort()

    def _synth(self, text: str) -> np.ndarray:
        if self._synth_fn is not None:
            return self._synth_fn(text)
        if self._model is None:
            from mlx_audio.tts.utils import load_model
            self._model = load_model(_MODEL_ID)
        parts = [np.asarray(r.audio, dtype=np.float32).reshape(-1)
                 for r in self._model.generate(text=text, voice=config.TTS_VOICE,
                                               speed=config.TTS_SPEED, lang_code=config.TTS_LANG)]
        return np.concatenate(parts) if parts else np.zeros(0, np.float32)

    def _play(self, audio: np.ndarray) -> bool:
        for i in range(0, len(audio), BLOCK):
            if self._stop.is_set():
                return False
            block = audio[i:i + BLOCK]
            self._stream.write(block.reshape(-1, 1))
            if (i // BLOCK) % 4 == 0:
                self._on_audio(fft_frame(block, True))
        return not self._stop.is_set()

    async def speak(self, sentences: AsyncIterator[str] | list[str]) -> bool:
        loop = asyncio.get_running_loop()
        self._stop.clear()
        if self._factory is None:
            import sounddevice as sd
            self._factory = lambda: sd.OutputStream(samplerate=TTS_RATE, channels=1, dtype="float32",
                                                    blocksize=BLOCK)
        self._stream = self._factory()
        self._stream.start()
        self._speaking = True
        queue: asyncio.Queue = asyncio.Queue()

        async def player() -> bool:
            ok = True
            while (fut := await queue.get()) is not None:
                if not ok or self._stop.is_set():
                    ok = False
                    fut.cancel()
                    continue
                audio = await fut
                ok = await loop.run_in_executor(self._play_pool, self._play, audio)
            return ok

        play_task = asyncio.create_task(player())
        try:
            async def _it(items):
                for s in items:
                    yield s
            source = _it(sentences) if isinstance(sentences, list) else sentences
            async for text in source:
                if self._stop.is_set():
                    break
                if text.strip():
                    queue.put_nowait(loop.run_in_executor(self._tts_pool, self._synth, text))
        finally:
            queue.put_nowait(None)
            ok = await play_task
            self._speaking = False
            stream, self._stream = self._stream, None
            if not self._stop.is_set():
                # close() alone discards buffered audio and clips the sentence ending; stop() drains it first
                await loop.run_in_executor(self._play_pool, stream.stop)
            stream.close()
        return ok and not self._stop.is_set()
