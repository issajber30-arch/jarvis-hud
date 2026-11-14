"""Centred overlay shown over the sphere while a yes/no confirmation is pending."""
from __future__ import annotations

import time

from rich.text import Text
from textual.widget import Widget

import config
from tools.redact import redact


class ConfirmCard(Widget):
    DEFAULT_CSS = """
    ConfirmCard { layer: overlay; position: absolute; display: none; width: 46; height: 6;
                  border: round #9FE8FF; padding: 0 1; color: #9FE8FF; background: #070503; }
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.border_title = "CONFIRM"
        self.readback, self.total, self._ends = "", 1.0, 0.0
        self.clock = time.monotonic
        self._timer = None

    def on_mount(self) -> None:
        self._timer = self.set_interval(0.1, self.refresh, pause=True)  # repaints only while visible

    def show(self, readback: str, seconds: float, centre: tuple[int, int] = (0, 0)) -> None:
        self.readback, self.total = redact(readback), max(seconds, 0.1)
        self._ends = self.clock() + self.total
        self.styles.offset = (max(centre[0] - 23, 0), max(centre[1] - 3, 0))
        self.display = True
        if self._timer:
            self._timer.resume()

    def hide(self) -> None:
        self.display = False
        if self._timer:
            self._timer.pause()

    def remaining(self) -> float:
        return min(max(self._ends - self.clock(), 0.0), self.total)

    def render(self) -> Text:
        w = max(self.size.width - 4, 10)
        left = self.remaining()
        filled = round(w * left / self.total)
        t = Text(no_wrap=True, overflow="ellipsis")
        t.append(self.readback.upper()[:w] + "\n", style=f"bold {config.HUD_GLOW}")
        t.append('SAY "YES" · ENTER\n', style=config.HUD_CONFIRM)
        t.append("█" * filled, style=config.HUD_CONFIRM)
        t.append("░" * (w - filled), style=config.HUD_DIM)
        t.append(f"\n{left:4.1f} S", style=config.HUD_DIM)
        return t
