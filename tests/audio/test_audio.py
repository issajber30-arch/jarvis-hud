import asyncio
import time
from unittest.mock import patch

import numpy as np

import config
from audio import stt
from audio.listener import Listener, fft_frame
from audio.speaker import Speaker
from contracts import AudioTelemetryFrame

SR = config.SAMPLE_RATE


def test_fft_frame_shape():
    t = np.arange(config.BLOCK_SIZE) / SR
    f = fft_frame(np.sin(2 * np.pi * 440 * t).astype(np.float32), True)
    assert isinstance(f, AudioTelemetryFrame) and f.vad is True
    assert f.fft.shape == (64,) and 0 <= f.fft.min() and f.fft.max() <= 1 and f.rms > 0.5
    assert fft_frame(np.zeros(1280, np.float32), False).rms == 0


def test_stt_filters_hallucinations():
    loud = np.ones(SR, np.float32) * 0.1
    with patch("audio.stt.mlx_whisper.transcribe", return_value={"text": " Thank you."}) as m:
        assert stt.transcribe(loud) == ""
        m.return_value = {"text": " ... "}
        assert stt.transcribe(loud) == ""
        m.return_value = {"text": " Open Safari. "}
        assert stt.transcribe(loud) == "Open Safari."
    with patch("audio.stt.mlx_whisper.transcribe") as m:
        assert stt.transcribe(np.zeros(int(0.1 * SR), np.float32)) == ""
        m.assert_not_called()


class FakeOut:
    def __init__(self):
        self.written = 0
        self.aborted = False

    def start(self): pass
    def close(self): pass
    def abort(self): self.aborted = True
    def stop(self): pass

    def write(self, block):
        time.sleep(len(block) / 24000)  # real-time device
        self.written += len(block)


def test_speaker_stop_latency():
    out = FakeOut()
    sp = Speaker(lambda f: None, stream_factory=lambda: out,
                 synth=lambda text: np.zeros(3 * 24000, np.float32))
    stopped = {}

    async def run():
        task = asyncio.create_task(sp.speak(["A long sentence."]))
        await asyncio.sleep(0.3)
        assert sp.is_speaking
        t0 = time.perf_counter()
        sp.stop()
        stopped["stop_call_ms"] = (time.perf_counter() - t0) * 1000
        res = await task
        stopped["exit_ms"] = (time.perf_counter() - t0) * 1000
        return res

    assert asyncio.run(run()) is False
    assert stopped["stop_call_ms"] < 20 and stopped["exit_ms"] < 20, stopped
    assert out.aborted and out.written < 24000 and not sp.is_speaking


def test_speaker_plays_all_and_pipelines():
    out = FakeOut()
    calls = []
    sp = Speaker(lambda f: None, stream_factory=lambda: out,
                 synth=lambda t: calls.append(t) or np.zeros(2400, np.float32))

    async def gen():
        for s in ["One.", "Two."]:
            yield s

    assert asyncio.run(sp.speak(gen())) is True
    assert calls == ["One.", "Two."] and out.written == 4800


class FakeStream:
    def __init__(self, **kw): self.kw = kw
    def start(self): pass
    def stop(self): pass
    def close(self): pass


def make_listener(events, wake=0.0):
    l = Listener(events.append, lambda f: None, stream_factory=lambda **kw: FakeStream(**kw),
                 wake_fn=lambda b: wake, vad_fn=lambda c: 0.9 if np.abs(c).max() > 0.05 else 0.0)
    return l


def tone(n=1280):
    return (0.3 * np.sin(2 * np.pi * 440 * np.arange(n) / SR)).astype(np.float32)


def test_listener_ptt_utterance():
    ev = []
    l = make_listener(ev)
    l.start()
    l.ptt_down()
    for _ in range(5):
        l._on_block(tone())
    l.ptt_up()
    kinds = [e.kind for e in ev]
    assert kinds == ["ptt_start", "ptt_end", "utterance"]
    assert ev[-1].audio.dtype == np.float32 and len(ev[-1].audio) >= 5 * 1280


def test_listener_barge_in_while_speaking():
    ev = []
    l = make_listener(ev)
    l.start()
    l.set_speaking(True)
    l.ptt_down()
    assert [e.kind for e in ev][:2] == ["barge_in", "ptt_start"]


def test_listener_wake_threshold_and_vad_end():
    ev = []
    l = make_listener(ev, wake=0.6)
    l.start()
    l.set_speaking(True)
    l._on_block(tone())  # 0.6 < 0.8 while speaking
    assert ev == []
    l.set_speaking(False)
    l._on_block(tone())
    assert [e.kind for e in ev] == ["wake"]
    l._wake_fn = lambda b: 0.0
    for _ in range(3):
        l._on_block(tone())
    for _ in range(12):  # 960 ms of silence > 700 ms
        l._on_block(np.zeros(1280, np.float32))
    assert [e.kind for e in ev] == ["wake", "utterance"]


def test_listener_max_utterance_and_empty_ptt():
    ev = []
    l = make_listener(ev)
    l.start()
    l.ptt_down()
    l.ptt_up()  # nothing captured
    assert [e.kind for e in ev] == ["ptt_start", "ptt_end"]
    ev.clear()
    l._on_block(tone())
    l._wake_fn = lambda b: 1.0
    l._on_block(tone())
    l._wake_fn = lambda b: 0.0
    for _ in range(int(config.MAX_UTTERANCE_S * SR / 1280) + 2):
        l._on_block(tone())
    assert [e.kind for e in ev] == ["wake", "utterance"]


def test_followup_window_starts_capture_on_speech():
    ev = []
    l = make_listener(ev)
    l.start()
    l.open_followup(5)
    l._on_block(tone())
    for _ in range(12):
        l._on_block(np.zeros(1280, np.float32))
    assert [e.kind for e in ev] == ["utterance"]


class RecordingOut:
    def __init__(self): self.calls = []
    def start(self): self.calls.append("start")
    def write(self, block): self.calls.append("write"); return False
    def stop(self): self.calls.append("stop")
    def abort(self): self.calls.append("abort")
    def close(self): self.calls.append("close")


async def test_speaker_drains_before_close():
    # Regression: close() discards buffered audio, clipping every sentence's ending (heard as crackle).
    import numpy as np
    from audio.speaker import Speaker
    out = RecordingOut()
    sp = Speaker(on_audio=lambda f: None, stream_factory=lambda: out, synth=lambda t: np.zeros(2400, np.float32))
    assert await sp.speak(["Hello."])
    assert out.calls.index("stop") < out.calls.index("close")
