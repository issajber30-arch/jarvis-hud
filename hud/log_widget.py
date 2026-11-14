"""Right rail: aerospace-style event log, plus the bottom utterance ticker."""
from __future__ import annotations

import math
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
        self._type_s = 0.0    # > 0 while Jarvis's caption types out
        self._words: list[str] = []

    def on_mount(self) -> None:
        self.set_interval(0.1, self._repaint_if_fading)

    def _repaint_if_fading(self) -> None:
        # Only repaint while text is on screen or just finished fading (one last clear).
        if self.text and time.monotonic() - self._t0 < self._type_s + FADE_S + 0.2:
            self.refresh()

    def say(self, text: str) -> None:
        self.text, self._t0, self._type_s = text[: config.LOG_MAX_CHARS], time.monotonic(), 0.0
        self.refresh()

    def type_out(self, text: str, seconds: float) -> None:
        """Reveal Jarvis's sentence word by word over `seconds`; the 0.1 s timer repaints, no per-word timers."""
        self.text, self._t0, self._type_s = text[: config.LOG_MAX_CHARS], time.monotonic(), max(seconds, 0.1)
        self._words = self.text.split()
        self.refresh()

    def render(self) -> Text:
        age = time.monotonic() - self._t0
        if not self.text or age >= self._type_s + FADE_S:
            return Text("")
        fade_age = age - self._type_s
        style = config.HUD_PRIMARY if fade_age < FADE_S / 2 else config.HUD_DIM
        if self._type_s:
            n = math.ceil(len(self._words) * min(age / self._type_s, 1.0))
            out = Text(no_wrap=True, overflow="ellipsis")
            out.append("JARVIS ▸ ", style=config.HUD_GLOW)
            return out.append(" ".join(self._words[:n]), style=style)
        return Text("▸ " + self.text, style=style, no_wrap=True, overflow="ellipsis")


class LogWidget(Widget):
    DEFAULT_CSS = "LogWidget { border: round #6B3F0E; padding: 0 1; }"

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
            out.append(line[:w], style=config.HUD_PRIMARY)
        return out
