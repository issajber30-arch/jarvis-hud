import numpy as np

from audio.chime import chime_wave, play_chime
from tests.test_main import make_app


def test_chime_wave_shape():
    w = chime_wave(24000)
    assert w.dtype == np.float32 and abs(len(w) - int(0.14 * 24000)) <= 2
    assert 0.1 < np.abs(w).max() <= 0.25
    n = int(0.01 * 24000)
    assert abs(w[0]) < 1e-3 and abs(w[-1]) < 1e-2 and np.abs(w[:n]).max() < np.abs(w[n:2 * n]).max() + 0.25
    # rises: more zero crossings in the second half
    zc = lambda x: int((np.diff(np.sign(x)) != 0).sum())
    assert zc(w[len(w) // 2:]) > zc(w[: len(w) // 2])


def test_play_chime_never_raises(monkeypatch):
    import sounddevice
    monkeypatch.setattr(sounddevice, "play", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no device")))
    play_chime()


async def test_app_chimes_on_wake_and_ptt():
    from contracts import ListenerEvent
    app, _ = make_app()
    hits = []
    app.chime = lambda: hits.append(1)
    async with app.run_test():
        app._on_listener_event(ListenerEvent("wake"))
        app._on_listener_event(ListenerEvent("ptt_start"))
        app._on_listener_event(ListenerEvent("ptt_end"))
    assert len(hits) == 2


async def test_hud_only_does_not_chime():
    from main import hud_only_app
    from contracts import ListenerEvent
    app = hud_only_app()
    async with app.run_test():
        assert app.chime is None
        app._on_listener_event(ListenerEvent("wake"))
