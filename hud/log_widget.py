"""Right rail: aerospace-style event log, plus the bottom utterance ticker."""
from __future__ import annotations

import time
from collections import deque

from rich.text import Text
from textual.widget import Widget

import config

KEEP = 200
FADE_S = 4.0


class TickerWidget(Widget):
    """One-row ticker; text fades cyan -> dim over FADE_S seconds."""

    DEFAULT_CSS = "TickerWidget { height: 1; }"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.text = ""
        self._t0 = 0.0

    def on_mount(self) -> None:
        self.set_interval(0.1, self.refresh)

    def say(self, text: str) -> None:
        self.text, self._t0 = text[: config.LOG_MAX_CHARS], time.monotonic()
        self.refresh()

    def render(self) -> Text:
        age = time.monotonic() - self._t0
        if not self.text or age >= FADE_S:
            return Text("")
        style = config.STARK_CYAN if age < FADE_S / 2 else config.STARK_DIM
        return Text("▸ " + self.text, style=style, no_wrap=True, overflow="ellipsis")


class LogWidget(Widget):
    DEFAULT_CSS = "LogWidget { border: round #1B4F63; padding: 0 1; }"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.border_title = "LOG"
        self.lines: deque[str] = deque(maxlen=KEEP)

    def log_line(self, text: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} ▸ {text}".upper()
        self.lines.append(line[: config.LOG_MAX_CHARS])
        self.refresh()

    def ticker(self, text: str) -> None:
        # shortcut: forwards to the first TickerWidget in the app; no-op when none is mounted.
        if self.is_attached:
            for t in self.app.query(TickerWidget):
                t.say(text)

    def render(self) -> Text:
        h = max(self.size.height - 2, 1)
        w = max(self.size.width - 4, 1)
        out = Text(no_wrap=True, overflow="ellipsis")
        for i, line in enumerate(list(self.lines)[-h:]):
            if i:
                out.append("\n")
            out.append(line[:w], style=config.STARK_CYAN)
        return out
