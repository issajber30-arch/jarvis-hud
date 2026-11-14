import asyncio

import pytest

from contracts import ActionResult, HUDState, FastRouteMatch
from main import JarvisApp


class FakeRouter:
    def __init__(self, table=None):
        self.table = table or {}

    def route(self, text):
        hit = self.table.get(text.lower())
        return FastRouteMatch(hit[0], hit[1], 0.99, text, 0.05) if hit else None


class FakeSpeaker:
    def __init__(self):
        self.said: list[str] = []
        self.is_speaking = False

    async def speak(self, sentences):
        if not isinstance(sentences, list):
            sentences = [s async for s in sentences]
        self.said += sentences
        return True

    def stop(self):
        pass


class FakeListener:
    def __init__(self):
        self.followups = []

    def start(self): pass
    def stop(self): pass
    def set_speaking(self, v): pass
    def open_followup(self, s): self.followups.append(s)
    def ptt_down(self): pass
    def ptt_up(self): pass


class FakeBrain:
    def __init__(self, sentences, actions=(), followup=False):
        self.sentences, self.actions, self.followup = sentences, actions, followup
        self.wants_followup = False
        self.calls = 0

    async def respond(self, text, context, execute):
        self.calls += 1
        for a in self.actions:
            await execute(a)
        for s in self.sentences:
            yield s
        self.wants_followup = self.followup


def make_app(router=None, brain=None, controller=None):
    calls = []

    async def default_controller(action, confirm):
        calls.append(action)
        return ActionResult(action.name, True, "Done")

    app = JarvisApp(router=router or FakeRouter(), brain=brain or FakeBrain(["Certainly."]),
                    speaker=FakeSpeaker(), listener=FakeListener(),
                    controller=controller or default_controller, agenda_fn=lambda: [], greet=False)
    return app, calls


async def test_fast_path():
    app, calls = make_app(router=FakeRouter({"open safari": ("open_app", {"app": "Safari"})}))
    async with app.run_test():
        r = await app.handle_utterance("open safari")
    assert r.via == "fast" and len(r.actions) == 1 and calls[0].args == {"app": "Safari"}
    assert app.brain.calls == 0


async def test_agenda_spoken():
    async def ctl(action, confirm):
        return ActionResult(action.name, True, "Here is today. Standup at 09:00. Dentist at 15:30.")
    app, _ = make_app(router=FakeRouter({"what's on the agenda today": ("agenda_today", {})}), controller=ctl)
    async with app.run_test():
        r = await app.handle_utterance("what's on the agenda today")
    assert "Standup" in r.spoken[0] and "Dentist" in r.spoken[0]


async def test_claude_path_followup():
    from contracts import Action
    brain = FakeBrain(["Your day is set."], actions=[Action("create_event", {}), Action("create_reminder", {})],
                      followup=True)
    app, calls = make_app(brain=brain)
    async with app.run_test():
        r = await app.handle_utterance("plan my day")
    assert r.via == "claude" and r.followup and len(r.actions) == 2
    assert app.listener.followups and app.state is HUDState.FOLLOWUP


async def test_confirm_rejected():
    handled = []

    async def ctl(action, confirm):
        if not await confirm("Delete report.pdf"):
            return ActionResult(action.name, False, "Cancelled")
        handled.append(action)
        return ActionResult(action.name, True, "Done")

    from contracts import Action
    app, _ = make_app(brain=FakeBrain(["Very well."], actions=[Action("delete_file", {"path": "~/report.pdf"})]),
                      controller=ctl)
    async with app.run_test():
        task = asyncio.create_task(app.handle_utterance("delete report.pdf"))
        for _ in range(50):
            await asyncio.sleep(0.01)
            if app.pending_confirm is not None:
                break
        assert app.state is HUDState.CONFIRM
        app.answer_confirm("no thanks")
        r = await task
    assert not handled and r.actions[0].ok is False


async def test_confirm_accepted_by_voice():
    from contracts import Action
    handled = []

    async def ctl(action, confirm):
        if await confirm("Delete report.pdf"):
            handled.append(action)
        return ActionResult(action.name, bool(handled), "Done")

    app, _ = make_app(brain=FakeBrain(["Done."], actions=[Action("delete_file", {})]), controller=ctl)
    async with app.run_test():
        task = asyncio.create_task(app.handle_utterance("delete it"))
        for _ in range(50):
            await asyncio.sleep(0.01)
            if app.pending_confirm is not None:
                break
        app.answer_confirm("Yes, please.")
        await task
    assert handled


async def test_empty_transcript():
    app, _ = make_app()
    async with app.run_test():
        r = await app.handle_utterance("   ")
    assert r.via == "none" and app.brain.calls == 0 and r.spoken


async def test_log_is_redacted():
    app, _ = make_app(brain=FakeBrain(["ok"]))
    async with app.run_test():
        await app.handle_utterance("my key is sk-ant-api03-abcdefghijklmnopqrstuvwxyz0123456789")
        logged = " ".join(app.query_one("LogWidget").lines)
    assert "abcdefghijklmnop" not in logged.lower()


@pytest.mark.parametrize("text,expected", [("yes", True), ("Yes, please.", True), ("confirm", True),
                                           ("do it", True), ("no", False), ("yesterday", False), ("", False)])
def test_is_yes(text, expected):
    from main import is_yes
    assert is_yes(text) is expected


async def test_app_boots_headless():
    app, _ = make_app()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.query_one("CoreWidget") is not None


def test_no_private_name_clashes_with_textual():
    # Overriding Textual's own _log/_context silently hung the app; keep our private names unique.
    from textual.app import App
    ours = [n for n, v in vars(JarvisApp).items() if n.startswith("_") and not n.startswith("__")
            and callable(v) and getattr(v, "__module__", "") == "main"]
    assert not [n for n in ours if hasattr(App, n)]


def test_prepare_runtime_before_tui():
    # Regression: router model download inside Textual crashed with "bad value(s) in fds_to_keep".
    import os
    from multiprocessing import resource_tracker
    from main import prepare_runtime
    prepare_runtime()
    assert os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] == "1"
    assert resource_tracker._resource_tracker._fd is not None


async def test_boot_warms_speech_recognition(monkeypatch):
    # Regression: Whisper loaded on the first command (~12 s wait); boot must warm it.
    import audio.stt
    calls = []
    monkeypatch.setattr(audio.stt, "transcribe", lambda a: calls.append(len(a)) or "")
    app = JarvisApp(router=FakeRouter(), brain=FakeBrain(["ok"]), speaker=FakeSpeaker(), listener=FakeListener(),
                    controller=lambda a, c: None, agenda_fn=lambda: [], greet=True)
    async with app.run_test() as pilot:
        for _ in range(100):
            await pilot.pause(0.02)
            if calls:
                break
    assert calls, "speech recognition was not warmed at boot"


async def test_hud_only_mode_boots_without_audio_or_models():
    from main import hud_only_app
    app = hud_only_app()
    async with app.run_test() as pilot:
        await pilot.pause(0.3)
        await pilot.press("space")  # talk key must be harmless with no mic
        assert app.query_one("CoreWidget") is not None
        assert "HUD-ONLY" in " ".join(app.query_one("LogWidget").lines)


async def test_hud_only_rails_show_standby_and_agenda():
    from main import hud_only_app
    app = hud_only_app()
    async with app.run_test(size=(180, 60)) as pilot:
        await pilot.pause(0.3)
        status = app.query_one("StatusWidget")
        assert status.mode == "HUD-ONLY" and set(status.modules.values()) == {"STANDBY"}
        assert "NO EVENTS" in app.query_one("AgendaWidget").render().plain


async def test_boot_lights_status_and_loads_agenda(monkeypatch):
    import audio.stt
    from datetime import datetime
    monkeypatch.setattr(audio.stt, "transcribe", lambda a: "")
    agenda = [{"kind": "event", "title": "Standup", "start": datetime.now()}]
    app = JarvisApp(router=FakeRouter(), brain=FakeBrain(["ok"]), speaker=FakeSpeaker(), listener=FakeListener(),
                    controller=lambda a, c: None, agenda_fn=lambda: agenda, greet=True)
    async with app.run_test(size=(180, 60)) as pilot:
        status = app.query_one("StatusWidget")
        for _ in range(100):
            await pilot.pause(0.02)
            if status.modules["MIC"] == "ONLINE" and app.query_one("AgendaWidget").items:
                break
        for m in ("ROUTER", "VOICE", "SPEECH", "MIC", "CALENDAR"):
            assert status.modules[m] == "ONLINE", m
        assert "STANDUP" in app.query_one("AgendaWidget").render().plain


async def test_voice_panel_hears_only_the_mic(monkeypatch):
    import numpy as np
    import audio.stt
    from contracts import AudioTelemetryFrame
    app, _ = make_app()
    monkeypatch.setattr(audio.stt, "transcribe", lambda a: "open safari")
    async with app.run_test(size=(180, 60)) as pilot:
        voice = app.query_one("VoiceWidget")
        f = AudioTelemetryFrame(rms=0.05, fft=np.zeros(64, np.float32), vad=True, ts=0.0)
        app._push_mic(f)
        app._push_audio(f)  # Jarvis's own voice: sphere only
        assert voice.meter.frames == 1
        await app._on_utterance_audio(np.zeros(16000, np.float32))
        await pilot.pause(0.1)
        assert "OPEN SAFARI" in voice.render().plain


async def test_boot_freezes_model_objects_out_of_gc(monkeypatch):
    # Regression: with voice/speech models loaded (~370k objects) each full GC pass froze the HUD and audio ~60 ms.
    import gc
    import audio.stt
    monkeypatch.setattr(audio.stt, "transcribe", lambda a: "")
    calls = []
    monkeypatch.setattr(gc, "freeze", lambda: calls.append("freeze"))
    app = JarvisApp(router=FakeRouter(), brain=FakeBrain(["ok"]), speaker=FakeSpeaker(), listener=FakeListener(),
                    controller=lambda a, c: None, agenda_fn=list, greet=True)
    async with app.run_test() as pilot:
        for _ in range(100):
            await pilot.pause(0.02)
            if calls:
                break
    assert calls == ["freeze"]


async def test_hud_log_is_mirrored_to_a_redacted_file(tmp_path, monkeypatch):
    import main
    monkeypatch.setattr(main, "LOG_FILE", tmp_path / "jarvis.log")
    app, _ = make_app()
    async with app.run_test():
        app._hud_log("ROUTE CLAUDE key sk-ant-api03-abcdefghijklmnopqrstuvwxyz0123456789")
    text = (tmp_path / "jarvis.log").read_text()
    assert "ROUTE CLAUDE" in text and "abcdefghijklmnop" not in text
