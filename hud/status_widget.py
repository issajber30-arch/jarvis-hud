"""Right rail panels: Jarvis module status lights and today's agenda."""
from __future__ import annotations

import time
from datetime import datetime

from rich.text import Text
from textual.widget import Widget

import config

MODULES = ("MIC", "WAKE", "SPEECH", "VOICE", "ROUTER", "BRAIN", "CALENDAR")
_STATE_STYLE = {
    "ONLINE": "HUD_PRIMARY",
    "LOADING": "HUD_GLOW",
    "STANDBY": "HUD_DIM",
    "OFFLINE": "HUD_ERROR",
}


class StatusWidget(Widget):
    """One light per Jarvis module, plus session mode, state and uptime."""

    DEFAULT_CSS = "StatusWidget { border: round #6B3F0E; padding: 0 1; height: 13; }"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.border_title = "STATUS"
        self.modules = {m: "STANDBY" for m in MODULES}
        self.mode, self.state = "LIVE", "IDLE"
        self._t0 = time.monotonic()

    def on_mount(self) -> None:
        self.set_interval(1, self.refresh)

    def set(self, module: str, state: str) -> None:
        self.modules[module] = state
        self.refresh()

    def render(self) -> Text:
        t = Text(no_wrap=True, overflow="crop")
        for name, state in self.modules.items():
            color = getattr(config, _STATE_STYLE.get(state, "HUD_DIM"))
            t.append("● " if state == "ONLINE" else "○ ", style=color)
            t.append(f"{name:<10}", style=config.HUD_DIM)
            t.append(f"{state}\n", style=color)
        up = int(time.monotonic() - self._t0)
        t.append("MODE  ", style=config.HUD_DIM)
        t.append(f"{self.mode}\n", style=config.HUD_PRIMARY)
        t.append("STATE ", style=config.HUD_DIM)
        t.append(f"{self.state}\n", style=config.HUD_PRIMARY)
        t.append(f"SESSION {up // 3600:02d}:{up // 60 % 60:02d}:{up % 60:02d}", style=config.HUD_DIM)
        return t


class AgendaWidget(Widget):
    """Today's events and reminders, refreshed by the app."""

    DEFAULT_CSS = "AgendaWidget { border: round #6B3F0E; padding: 0 1; height: 14; }"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.border_title = "AGENDA · TODAY"
        self.items: list[dict] = []
        self.error: str | None = None

    def set_items(self, items: list[dict]) -> None:
        self.items, self.error = list(items), None
        self.refresh()

    def set_error(self, message: str) -> None:
        self.error = message
        self.refresh()

    def render(self) -> Text:
        t = Text(no_wrap=True, overflow="crop")
        if self.error:
            return t.append(self.error, style=config.HUD_DIM)
        if not self.items:
            return t.append("NO EVENTS TODAY", style=config.HUD_DIM)
        now = datetime.now()
        for it in self.items:
            start = it.get("start")
            past = isinstance(start, datetime) and start < now and it.get("kind") == "event"
            when = start.strftime("%H:%M") if isinstance(start, datetime) else "--:--"
            t.append(f"{when} ", style=config.HUD_DIM if past else config.HUD_GLOW)
            t.append("◆ " if it.get("kind") == "event" else "◇ ", style=config.HUD_DIM)
            t.append(f"{str(it.get('title', '')).upper()}\n", style=config.HUD_DIM if past else config.HUD_PRIMARY)
        return t
