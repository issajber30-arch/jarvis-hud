"""Right-rail ACTIVE panel: live timers counting down and short-lived confirmation notes."""
from __future__ import annotations

import math
import time

from rich.text import Text
from textual.widget import Widget

import config
from tools.redact import redact


class CardsWidget(Widget):
    DEFAULT_CSS = "CardsWidget { border: round #6B3F0E; padding: 0 1; height: 6; }"
    MAX_ROWS = 4

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.border_title = "ACTIVE"
        self.timers: list[list] = []   # [label, ends_at]
        self.notes: list[list] = []    # [text, expires_at]
        self.clock = time.monotonic    # tests swap this
        self.on_done = None            # callable(label) when a timer reaches zero

    def on_mount(self) -> None:
        self.set_interval(1.0, self._tick)  # the only timer; it repaints only while a timer is active or an item expires

    def add_timer(self, label: str, seconds: float) -> None:
        self.timers.append([redact(label).upper(), self.clock() + seconds])
        self.refresh()

    def add_note(self, text: str, ttl: float = 20) -> None:
        self.notes.append([redact(text).upper(), self.clock() + ttl])
        self.refresh()

    def _tick(self) -> None:
        now = self.clock()
        done = [t for t in self.timers if t[1] <= now]
        gone = [n for n in self.notes if n[1] <= now]
        self.timers = [t for t in self.timers if t not in done]
        self.notes = [n for n in self.notes if n not in gone]
        for label, _ in done:
            if self.on_done:
                self.on_done(label)
        if self.timers or done or gone:
            self.refresh()

    def render(self) -> Text:
        now = self.clock()
        out = Text(no_wrap=True, overflow="ellipsis")
        rows = [("⏱ ", f"{label}  {s // 60:02d}:{s % 60:02d}", config.HUD_GLOW)
                for label, end in self.timers for s in [max(math.ceil(end - now), 0)]]
        rows += [("", text, config.HUD_PRIMARY) for text, _ in self.notes]
        if not rows:
            return out.append("NOTHING ACTIVE", style=config.HUD_DIM)
        for i, (icon, text, style) in enumerate(rows[: self.MAX_ROWS]):
            out.append(("\n" if i else "") + icon + text, style=style)
        return out
