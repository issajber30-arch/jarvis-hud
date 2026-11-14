"""Claude brain: streams whole sentences, runs tool calls through the caller's executor."""
from __future__ import annotations

import json
import os
import re
from typing import AsyncIterator, Awaitable, Callable

import anthropic
from dotenv import load_dotenv

import config
from contracts import ACTIONS, Action, ActionResult, Tier
from tools import lines
from tools.redact import redact

_MAX_LOOPS = 8
_BOUNDARY = re.compile(r"[.!?][\"')\]]*\s")

_S, _I = {"type": "string"}, {"type": "integer", "minimum": 0, "maximum": 100}
_APP = {"app": {"type": "string", "description": "Application name, e.g. Safari"}}
_ISO = {"type": "string", "description": "ISO-8601 local time, e.g. 2026-11-14T17:00:00"}
_PROPS: dict[str, tuple[dict, list[str]]] = {
    "open_app": (_APP, ["app"]), "quit_app": (_APP, ["app"]),
    "set_volume": ({"level": _I}, ["level"]), "set_brightness": ({"level": _I}, ["level"]),
    "set_timer": ({"minutes": {"type": "number", "exclusiveMinimum": 0, "maximum": 1440}}, ["minutes"]),
    "create_event": ({"title": _S, "start": _ISO, "end": _ISO}, ["title", "start", "end"]),
    "create_reminder": ({"title": _S, "due": _ISO}, ["title"]),
    "delete_event": ({"id": _S}, ["id"]), "delete_reminder": ({"id": _S}, ["id"]),
    "reschedule_event": ({"id": _S, "start": _ISO, "end": _ISO}, ["id", "start", "end"]),
    "move_file": ({"src": _S, "dst": _S}, ["src", "dst"]), "delete_file": ({"path": _S}, ["path"]),
    "run_shell": ({"cmd": _S}, ["cmd"]), "send_message": ({"to": _S, "text": _S}, ["to", "text"]),
}
_DESC = {
    "agenda_today": "Read today's calendar events and reminders.", "system_status": "Report CPU, memory and battery.",
    "delete_event": "Delete a calendar event by id (asks the owner to confirm).",
    "run_shell": "Run a shell command (owner must confirm; dangerous commands are blocked).",
}


def _tools() -> list[dict]:
    out = []
    for name, tier in ACTIONS.items():
        if tier is Tier.BLOCKED:
            continue
        props, req = _PROPS.get(name, ({}, []))
        out.append({"name": name, "description": _DESC.get(name, name.replace("_", " ").capitalize() + "."),
                    "input_schema": {"type": "object", "properties": props, "required": req,
                                     "additionalProperties": False}})
    return out


class ClaudeBrain:
    def __init__(self, model: str = config.CLAUDE_MODEL, client=None):
        load_dotenv(config.ROOT / ".env")  # never logs or prints the key
        self.model = model
        self._client = client
        self.history: list[dict] = []
        self.wants_followup = False
        self._system = (config.ROOT / "prompts" / "jarvis_system.md").read_text(encoding="utf-8")

    def _get_client(self):
        if self._client is None:
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise anthropic.AnthropicError("no API key")
            self._client = anthropic.AsyncAnthropic(timeout=30.0, max_retries=1)
        return self._client

    def _system_prompt(self, context: dict) -> str:
        ctx = redact(json.dumps(context, default=str, ensure_ascii=False))
        return f"{self._system}\n\n<context>\n{ctx}\n</context>"

    async def respond(self, text: str, context: dict,
                      execute: Callable[[Action], Awaitable[ActionResult]]) -> AsyncIterator[str]:
        self.wants_followup = False
        executed = 0
        user = redact(text)
        messages = self.history + [{"role": "user", "content": user}]
        spoken: list[str] = []
        try:
            client = self._get_client()
            system = self._system_prompt(context)
            for _ in range(_MAX_LOOPS):
                buf = ""
                async with client.messages.stream(
                        model=self.model, max_tokens=config.CLAUDE_MAX_TOKENS, system=system,
                        tools=_tools(), messages=messages, output_config={"effort": "low"}) as stream:
                    async for chunk in stream.text_stream:
                        buf += chunk
                        while (m := _BOUNDARY.search(buf)):
                            sent, buf = buf[:m.end()].strip(), buf[m.end():]
                            if sent:
                                spoken.append(sent)
                                yield redact(sent)
                    final = await stream.get_final_message()
                if buf.strip():
                    spoken.append(buf.strip())
                    yield redact(buf.strip())
                if final.stop_reason == "refusal" and not spoken:
                    yield lines.pick("failed")
                if final.stop_reason != "tool_use":
                    break
                messages.append({"role": "assistant", "content": final.content})
                results = []
                for b in final.content:
                    if b.type != "tool_use":
                        continue
                    r = await execute(Action(b.name, dict(b.input)))
                    executed += 1
                    results.append({"type": "tool_result", "tool_use_id": b.id,
                                    "content": redact(r.message), "is_error": not r.ok})
                messages.append({"role": "user", "content": results})  # all results in one message
        except anthropic.AnthropicError:
            yield lines.pick("offline")
            return
        self.wants_followup = executed >= 2
        # keep only plain text between turns (tool/thinking blocks stay inside the turn)
        self.history += [{"role": "user", "content": user}, {"role": "assistant", "content": " ".join(spoken) or "Done."}]
        self.history = self.history[-2 * config.HISTORY_TURNS:]
