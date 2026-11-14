import asyncio
from datetime import datetime

from textual.app import App, ComposeResult


def test_spark_shape():
    from hud.telemetry_widget import _spark
    s = _spark([0, 50, 100, 25], 10)
    assert len(s) == 10 and set(s) <= set(" ▁▂▃▄▅▆▇█")
    assert _spark([], 5) == " " * 5


def test_left_rail_is_full():
    from hud.telemetry_widget import TelemetryWidget
    w = TelemetryWidget()
    w.update_frame()
    w.update_frame()
    plain = w.render().plain
    for section in ("CORES", "CPU 60S", "MEM", "DISK", "NET", "UPTIME", "TOP"):
        assert section in plain, section
    assert sum(1 for line in plain.splitlines() if line.strip()) >= 24


def test_status_panel():
    from hud.status_widget import StatusWidget
    s = StatusWidget()
    s.set("MIC", "ONLINE")
    s.set("BRAIN", "OFFLINE")
    plain = s.render().plain
    assert "MIC" in plain and "ONLINE" in plain and "OFFLINE" in plain
    assert "VOICE" in plain and "STANDBY" in plain  # untouched modules default to standby


def test_agenda_panel():
    from hud.status_widget import AgendaWidget
    a = AgendaWidget()
    assert "NO EVENTS" in a.render().plain
    a.set_items([{"kind": "event", "title": "Standup", "start": datetime(2026, 11, 14, 9, 0)},
                 {"kind": "reminder", "title": "Call mom", "start": datetime(2026, 11, 14, 17, 0)}])
    plain = a.render().plain
    assert "STANDUP" in plain and "09:00" in plain and "CALL MOM" in plain
    a.set_error("NO CALENDAR ACCESS")
    assert "NO CALENDAR ACCESS" in a.render().plain


async def test_rails_mount_together():
    from hud.status_widget import AgendaWidget, StatusWidget
    from hud.telemetry_widget import TelemetryWidget

    class T(App):
        def compose(self) -> ComposeResult:
            yield TelemetryWidget()
            yield StatusWidget()
            yield AgendaWidget()

    async with T().run_test() as pilot:
        await pilot.pause(0.2)


def test_multi_row_graph():
    from hud.telemetry_widget import _graph
    rows = _graph([0, 25, 50, 100], 6, 4, top=100)
    assert len(rows) == 4 and all(len(r) == 6 for r in rows)
    assert rows[0][-1] == "█" and rows[-1][-1] == "█"   # 100% fills the full height
    assert rows[0][-4] == " " and rows[-1][-4] == " "     # 0% leaves the column empty
    assert _graph([], 3, 2) == ["   ", "   "]


def test_status_mode_and_state_do_not_collide():
    from hud.status_widget import StatusWidget
    s = StatusWidget()
    s.mode, s.state = "HUD-ONLY", "SPEAKING"
    plain = s.render().plain
    assert "HUD-ONLYSTATE" not in plain and "MODE  HUD-ONLY" in plain and "STATE SPEAKING" in plain


def test_ticker_type_out_grows_then_completes():
    import time
    from hud.log_widget import TickerWidget
    t = TickerWidget()
    t.type_out("one two three four", 1.0)
    seen = []
    for age in (0.0, 0.3, 0.6, 0.9, 1.2):
        t._t0 = time.monotonic() - age
        seen.append(t.render().plain)
    assert all(s.startswith("JARVIS ▸ ") for s in seen)
    assert [len(s) for s in seen] == sorted(len(s) for s in seen)
    assert seen[-1] == "JARVIS ▸ one two three four" and seen[0] != seen[-1]
    t.say("hello")
    assert t.render().plain == "▸ hello"


async def test_say_types_caption_for_each_sentence():
    from main import JarvisApp
    from hud.log_widget import TickerWidget
    from tests.test_main import make_app
    app, _ = make_app()
    async with app.run_test() as pilot:
        orig = app.speaker.speak

        async def slow(sentences):
            await asyncio.sleep(0.1)
            return await orig(sentences)

        app.speaker.speak = slow
        await app._say(["Certainly sir."])
        assert app.query_one(TickerWidget).text == "Certainly sir."
