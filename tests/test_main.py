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
