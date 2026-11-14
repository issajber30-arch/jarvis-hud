import numpy as np
from textual.app import App

import config
from contracts import AudioTelemetryFrame, HUDState, SystemTelemetryFrame
from hud.core_widget import CoreWidget, render_braille
from hud.log_widget import LogWidget, TickerWidget
from hud.telemetry_widget import TelemetryWidget


def test_render_braille_dims():
    pts = np.random.default_rng(0).uniform(-1, 1, (300, 3))
    rows = render_braille(pts, 30, 12)
    assert len(rows) == 12 and all(len(r) == 30 for r in rows)
    assert all(c == " " or 0x2800 <= ord(c) <= 0x28FF for r in rows for c in r)
    assert any(c not in " ⠀" for r in rows for c in r)


def test_render_braille_empty():
    assert render_braille(np.zeros((0, 3)), 5, 3) == [" " * 5] * 3


def test_collect_telemetry():
    f = TelemetryWidget().collect()
    assert isinstance(f, SystemTelemetryFrame)
    assert 0 <= f.cpu_pct <= 100 and 0 <= f.ram_pct <= 100
    assert f.battery_pct is None or 0 <= f.battery_pct <= 100
    assert f.net_up_kbps >= 0 and f.net_down_kbps >= 0


def test_log_truncates():
    w = LogWidget()
    w.log_line("x" * 500)
    assert len(w.lines[-1]) <= config.LOG_MAX_CHARS
    assert "▸" in w.lines[-1]
    for i in range(300):
        w.log_line(f"line {i}")
    assert len(w.lines) == 200
    assert w.lines[-1].endswith("LINE 299")


def test_core_inertia_and_decay():
    c = CoreWidget()
    base = c._omega
    c.push_audio(AudioTelemetryFrame(rms=1.0, fft=np.zeros(64), vad=True, ts=0))
    for _ in range(30):
        c.tick(1 / 60)
    assert c._omega > base * 2
    c.push_audio(AudioTelemetryFrame(rms=0.0, fft=np.zeros(64), vad=False, ts=0))
    peak = c._omega
    for _ in range(600):
        c.tick(1 / 60)
    assert c._omega < peak and c._omega < base * 1.5


class _App(App):
    def compose(self):
        yield CoreWidget(id="core")
        yield TelemetryWidget(id="tele")
        yield LogWidget(id="log")
        yield TickerWidget(id="ticker")


async def test_widgets_mount():
    app = _App()
    async with app.run_test(size=(120, 40)) as pilot:
        core = app.query_one(CoreWidget)
        for s in HUDState:
            core.state = s
            await pilot.pause(0.05)
        core.state = HUDState.SPEAKING
        core.push_audio(AudioTelemetryFrame(rms=0.5, fft=np.zeros(64), vad=True, ts=0))
        app.query_one(LogWidget).log_line("exec open_app safari")
        app.query_one(LogWidget).ticker("hello there")
        core.state = HUDState.CONFIRM
        await pilot.pause(0.2)
        core.state = HUDState.ERROR
        await pilot.pause(0.1)


def test_core_frame_is_cheap():
    # Regression: per-character spans made a 120x55 frame cost ~17 ms and pinned a core at 60 fps.
    # Real cost is ~3.5 ms; 8 ms catches that regression 2x over without flaking under parallel load
    # (the frame budget at 30 fps is 33 ms).
    import time
    from unittest.mock import PropertyMock, patch
    from rich.console import Console
    from textual.geometry import Size
    from hud.core_widget import CoreWidget
    w = CoreWidget()
    console = Console(width=120, file=open("/dev/null", "w"), color_system="truecolor")
    with patch.object(CoreWidget, "size", new_callable=PropertyMock, return_value=Size(120, 55)):
        list(console.render(w.render()))  # warm-up
        t = time.perf_counter()
        for _ in range(20):
            list(console.render(w.render()))
        ms = (time.perf_counter() - t) / 20 * 1000
    assert ms < 8, f"core frame took {ms:.1f} ms"


def _core_rows(w=120, h=55, state=None):
    from unittest.mock import PropertyMock, patch
    from textual.geometry import Size
    from hud.core_widget import CoreWidget
    c = CoreWidget()
    if state is not None:
        c.state = state
    with patch.object(CoreWidget, "size", new_callable=PropertyMock, return_value=Size(w, h)):
        return c.render()


def test_core_shows_jarvis_label_in_centre():
    text = _core_rows()
    rows = text.plain.splitlines()
    mid = rows[len(rows) // 2]
    assert "J A R V I S" in mid
    col = mid.index("J A R V I S") + len("J A R V I S") // 2
    assert abs(col - 60) <= 1  # horizontally centred


def test_core_globe_is_not_empty():
    # Baseline sphere lit only 16% of cells in its area ("looks empty"); globe + core must at least double it.
    rows = _core_rows().plain.splitlines()
    box = [r[30:90] for r in rows[8:47]]
    lit = sum(ch != " " for r in box for ch in r) / sum(len(r) for r in box)
    assert lit >= 0.32, f"only {lit:.0%} lit"


def test_core_label_fits_small_widget():
    text = _core_rows(w=8, h=3)  # narrower than the label: must not crash or overflow
    assert all(len(r) == 8 for r in text.plain.splitlines())



def test_core_arcs_are_continuous():
    # Lines read as dotted when neighbouring samples land >1 screen dot apart; check at a big full-screen size.
    import numpy as np
    from hud.core_widget import CoreWidget
    c = CoreWidget()
    s = min(2 * 240, 4 * 70) / 2 * 0.72  # projection scale used by _bits at 240x70
    assert len(c._arcs) >= 80, "data sphere needs many arc fragments"
    for arc in c._arcs:
        steps = np.linalg.norm(np.diff(arc, axis=0), axis=1) * s
        assert steps.max() <= 1.2


def test_core_uses_gold_palette():
    import config
    text = _core_rows()
    styles = {str(sp.style) for sp in text.spans}
    assert config.HUD_PRIMARY in styles and "#4FD1FF" not in " ".join(styles)


def test_core_confirm_state_stands_out():
    import config
    from contracts import HUDState
    styles = {str(sp.style) for sp in _core_rows(state=HUDState.CONFIRM).spans}
    assert config.HUD_CONFIRM in styles
