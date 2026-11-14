import time
from unittest.mock import PropertyMock, patch

import numpy as np
from textual.geometry import Size

from contracts import AudioTelemetryFrame, HUDState
from hud.core_widget import CoreWidget, _SHELL_SPIN, heartbeat

SIZE = Size(120, 55)


def lit(c: CoreWidget) -> int:
    with patch.object(CoreWidget, "size", new_callable=PropertyMock, return_value=SIZE):
        bits, _ = c._layers(SIZE.width, SIZE.height)
    return int((bits > 0).sum())


def glow_cells(c: CoreWidget) -> int:
    with patch.object(CoreWidget, "size", new_callable=PropertyMock, return_value=SIZE):
        _, layer = c._layers(SIZE.width, SIZE.height)
    return int((layer == 3).sum())


def test_heartbeat_is_a_double_pulse():
    ph = np.linspace(0, 1, 400, endpoint=False)
    hb = heartbeat(ph)
    peaks = [i for i in range(1, 399) if hb[i] > hb[i - 1] and hb[i] >= hb[i + 1] and hb[i] > 0.3]
    assert len(peaks) == 2, peaks           # lub + dub
    assert hb[peaks[0]] > hb[peaks[1]]       # dub is softer
    assert hb[200] < 0.05                    # rest between beats


def test_heart_rate_follows_state():
    c = CoreWidget()
    assert c.bpm == 60
    c.state = HUDState.THINKING
    assert c.bpm > 80
    c.state = HUDState.ERROR
    assert c.bpm >= 100


def test_all_shells_spin_right():
    assert (_SHELL_SPIN > 0).all()


def test_core_beats_in_size():
    c = CoreWidget()
    c._beat = 0.0
    on = lit(c)
    c._beat = 0.5  # rest
    off = lit(c)
    assert on > off


def test_frame_rate_is_always_full_and_evenly_paced():
    # Owner asked for "super smooth": a heartbeat steps visibly at 15 fps, so no battery downshift.
    c = CoreWidget()
    assert c.target_fps == 30
    _ = c.state  # Textual runs the state watcher once on first read (one extra repaint at startup)
    ticks = []
    c.refresh = lambda *a, **k: ticks.append(1)
    for _ in range(30):
        c.tick(1 / 30)
    assert len(ticks) == 30


def test_parameters_glide_not_jump():
    c = CoreWidget()
    c.state = HUDState.LISTENING        # shells ease inward
    c.tick(1 / 30)
    first = c.shell_scale
    assert 0.97 < first < 1.0           # moved a little, not snapped to the target
    for _ in range(60):
        c.tick(1 / 30)
    assert c.shell_scale < 0.93


def test_heart_rate_change_keeps_rhythm_continuous():
    c = CoreWidget()
    c._beat = 0.4
    c.state = HUDState.THINKING
    c.tick(1 / 30)
    assert 0.4 < c._beat < 0.5          # phase carries on from where it was, just faster


def test_boot_assembles_shells():
    c = CoreWidget()
    full = lit(c)
    c.set_boot_progress(0.25)
    assert lit(c) < full * 0.5
    c.set_boot_progress(1.0)
    assert lit(c) == full


def test_thinking_scan_band_glows():
    c = CoreWidget()
    idle = glow_cells(c)
    c.state = HUDState.THINKING
    assert glow_cells(c) > idle + 20


def test_listening_shows_voice_ring():
    c = CoreWidget()
    c.state = HUDState.LISTENING
    quiet = lit(c)
    c.push_audio(AudioTelemetryFrame(rms=0.2, fft=np.ones(64, np.float32), vad=True, ts=0.0))
    assert lit(c) > quiet


def test_confirm_ring_drains():
    c = CoreWidget()
    c.state = HUDState.CONFIRM
    start = len(c._confirm_ring())
    c._state_t0 = time.monotonic() - 5  # half of the 10 s window gone
    half = len(c._confirm_ring())
    assert 0.35 * start < half < 0.65 * start


def test_error_bursts_outward():
    c = CoreWidget()
    c.state = HUDState.ERROR
    c.tick(1 / 30)
    assert c.shell_scale > 1.1
    c.state = HUDState.IDLE
    for _ in range(90):
        c.tick(1 / 30)
    assert abs(c.shell_scale - 1.0) < 0.05
