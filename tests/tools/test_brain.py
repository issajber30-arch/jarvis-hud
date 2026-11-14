import json
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest

from contracts import Action, ActionResult
from tools import claude_brain as cb, lines


class FakeStream:
    def __init__(self, texts, content, stop="end_turn"):
        self.texts, self.final = texts, NS(content=content, stop_reason=stop)
    async def __aenter__(self): return self
    async def __aexit__(self, *a): return False
    @property
    def text_stream(self):
        async def gen():
            for t in self.texts: yield t
        return gen()
    async def get_final_message(self): return self.final


class FakeClient:
    def __init__(self, streams):
        self.streams, self.calls = list(streams), []
        self.messages = NS(stream=self._stream)
    def _stream(self, **kw):
        self.calls.append(json.loads(json.dumps(kw, default=str)))
        return self.streams.pop(0)


async def collect(gen): return [s async for s in gen]


@pytest.fixture(autouse=True)
def _no_dotenv(monkeypatch):
    monkeypatch.setattr(cb, "load_dotenv", lambda *a, **k: None)


async def test_brain_offline_yields_error_line(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    out = await collect(cb.ClaudeBrain().respond("hi", {}, AsyncMock()))
    bank = lines._bank()["offline"]
    assert len(out) == 1 and out[0] in bank


async def test_brain_api_error_yields_offline(monkeypatch):
    class Boom:
        messages = NS(stream=lambda **kw: (_ for _ in ()).throw(cb.anthropic.APIConnectionError(request=NS())))
    out = await collect(cb.ClaudeBrain(client=Boom()).respond("hi", {}, AsyncMock()))
    assert out[0] in lines._bank()["offline"]


async def test_brain_tool_loop():
    tool = NS(type="tool_use", id="t1", name="create_reminder", input={"title": "call mum"})
    client = FakeClient([
        FakeStream(["Setting that up now. "], [NS(type="text", text="Setting that up now. "), tool], "tool_use"),
        FakeStream(["Done. Anything else", " for today?"], [NS(type="text", text="Done. Anything else for today?")]),
    ])
    execute = AsyncMock(return_value=ActionResult("create_reminder", True, "Reminder created"))
    brain = cb.ClaudeBrain(client=client)
    out = await collect(brain.respond("remind me to call mum", {}, execute))
    assert out == ["Setting that up now.", "Done.", "Anything else for today?"]
    execute.assert_awaited_once_with(Action("create_reminder", {"title": "call mum"}))
    assert brain.wants_followup is False
    second = client.calls[1]["messages"][-1]
    assert second["role"] == "user" and second["content"][0]["tool_use_id"] == "t1"
    names = {t["name"] for t in client.calls[0]["tools"]}
    assert "create_reminder" in names and "run_shell" in names
    assert client.calls[0]["model"] == "claude-haiku-5-5"


async def test_brain_followup_after_two_actions():
    mk = lambda i: NS(type="tool_use", id=f"t{i}", name="create_reminder", input={"title": f"r{i}"})
    client = FakeStream, None
    c = FakeClient([FakeStream([], [mk(1), mk(2)], "tool_use"), FakeStream(["All set."], [NS(type="text", text="All set.")])])
    execute = AsyncMock(return_value=ActionResult("create_reminder", True, "ok"))
    brain = cb.ClaudeBrain(client=c)
    await collect(brain.respond("plan", {}, execute))
    assert execute.await_count == 2 and brain.wants_followup is True
    # tool_results go back in ONE user message
    assert len(c.calls[1]["messages"][-1]["content"]) == 2


async def test_brain_redacts_outbound():
    secret = "sk-ant-api03-SUPERSECRETVALUE1234567890"
    c = FakeClient([FakeStream(["Fine."], [NS(type="text", text="Fine.")])])
    await collect(cb.ClaudeBrain(client=c).respond(f"my key is {secret}", {"note": secret, "agenda": []}, AsyncMock()))
    assert "SUPERSECRET" not in json.dumps(c.calls)


async def test_brain_redacts_inbound_text():
    secret = "ghp_" + "a" * 36
    c = FakeClient([FakeStream([f"Your token is {secret}. "], [NS(type="text", text="x")])])
    out = await collect(cb.ClaudeBrain(client=c).respond("hi", {}, AsyncMock()))
    assert secret not in " ".join(out)


async def test_brain_history_capped(monkeypatch):
    monkeypatch.setattr(cb.config, "HISTORY_TURNS", 2)
    c = FakeClient([FakeStream(["ok."], [NS(type="text", text="ok.")]) for _ in range(4)])
    brain = cb.ClaudeBrain(client=c)
    for i in range(4):
        await collect(brain.respond(f"q{i}", {}, AsyncMock()))
    assert len(brain.history) == 4 and brain.history[0]["role"] == "user"
    assert [m["content"] for m in brain.history if m["role"] == "user"] == ["q2", "q3"]
