import numpy as np

from contracts import AudioTelemetryFrame


def frame(rms, vad):
    return AudioTelemetryFrame(rms=rms, fft=np.full(64, 0.5, np.float32), vad=vad, ts=0.0)


def test_meter_quiet_room_is_excellent():
    from hud.voice_widget import VoiceMeter
    m = VoiceMeter()
    assert m.clarity()[0] == "WAITING"  # nothing heard yet
    for _ in range(40):
        m.push(frame(0.001, False))  # -60 dB room
    for _ in range(15):
        m.push(frame(0.05, True))    # -26 dB voice
    label, snr = m.clarity()
    assert abs(m.noise_db - -60) < 2 and label == "EXCELLENT" and snr > 30


def test_meter_noisy_room_is_poor_and_flagged():
    from hud.voice_widget import VoiceMeter
    m = VoiceMeter()
    for _ in range(40):
        m.push(frame(0.03, False))  # -30 dB: loud room
    for _ in range(15):
        m.push(frame(0.05, True))
    assert m.clarity()[0] == "POOR" and "NOISY ROOM" in m.warnings()


def test_meter_flags_too_loud():
    from hud.voice_widget import VoiceMeter
    m = VoiceMeter()
    m.push(frame(0.9, True))
    assert "TOO LOUD" in m.warnings()


def test_voice_panel_renders_readouts():
    from hud.voice_widget import VoiceWidget
    v = VoiceWidget()
    for _ in range(30):
        v.push(frame(0.001, False))
    for _ in range(10):
        v.push(frame(0.05, True), wake=0.4)
    v.heard("open safari", 1.4, 0.21)
    plain = v.render().plain
    for s in ("LEVEL", "NOISE", "CLARITY", "WAKE", "OPEN SAFARI", "1.4 S", "0.21 S"):
        assert s in plain, s


def test_voice_panel_mic_off_message():
    from hud.voice_widget import VoiceWidget
    v = VoiceWidget()
    v.mic_off("HUD-ONLY")
    assert "MIC OFF" in v.render().plain


def test_voice_panel_repaints_only_when_data_changed():
    from hud.voice_widget import VoiceWidget
    v = VoiceWidget()
    calls = []
    v.refresh = lambda *a, **k: calls.append(1)
    v._repaint_if_dirty()
    assert calls == []                       # nothing new: no repaint
    v.push(frame(0.05, True))
    v._repaint_if_dirty()
    v._repaint_if_dirty()
    assert len(calls) == 1                   # one repaint per batch of new data


def test_ticker_repaints_only_while_fading():
    import time
    from hud.log_widget import TickerWidget
    t = TickerWidget()
    calls = []
    t.refresh = lambda *a, **k: calls.append(1)
    t._repaint_if_fading()
    assert calls == []
    t.text, t._t0 = "hello", time.monotonic()
    t._repaint_if_fading()
    assert calls == [1]
