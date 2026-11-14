"""Hero widget: a rotating 3D point-cloud sphere + ring drawn on a Braille canvas."""
from __future__ import annotations

import numpy as np
from rich.text import Text
from textual.reactive import reactive
from textual.widget import Widget

import config
from contracts import AudioTelemetryFrame, HUDState

# Braille dot bit for (row, col) inside a 2x4 cell.
_BIT = np.array([[0x01, 0x08], [0x02, 0x10], [0x04, 0x20], [0x40, 0x80]], dtype=np.uint8)

_COLORS = {
    HUDState.CONFIRM: config.STARK_AMBER,
    HUDState.ERROR: config.STARK_RED,
}


def _cloud(n_sphere: int = 1100, n_ring: int = 260) -> np.ndarray:
    i = np.arange(n_sphere) + 0.5
    phi = np.arccos(1 - 2 * i / n_sphere)
    theta = np.pi * (1 + 5**0.5) * i  # fibonacci sphere
    sphere = np.stack([np.cos(theta) * np.sin(phi), np.cos(phi), np.sin(theta) * np.sin(phi)], 1)
    a = np.linspace(0, 2 * np.pi, n_ring, endpoint=False)
    ring = 1.35 * np.stack([np.cos(a), np.zeros_like(a), np.sin(a)], 1)
    return np.vstack([sphere, ring])


def _bits(points: np.ndarray, w: int, h: int, scale: float = 1.0) -> np.ndarray:
    """Project x/y of `points` (about -1..1) to an (h, w) array of Braille bit masks."""
    out = np.zeros((h, w), dtype=np.uint8)
    if len(points) == 0 or w < 1 or h < 1:
        return out
    px, py = 2 * w, 4 * h
    s = min(px, py) / 2 * 0.9 * scale
    x = np.round(px / 2 + points[:, 0] * s).astype(int)
    y = np.round(py / 2 - points[:, 1] * s).astype(int)
    ok = (x >= 0) & (x < px) & (y >= 0) & (y < py)
    x, y = x[ok], y[ok]
    np.bitwise_or.at(out, (y // 4, x // 2), _BIT[y % 4, x % 2])
    return out


def _chars(bits: np.ndarray) -> list[str]:
    return ["".join(" " if b == 0 else chr(0x2800 + int(b)) for b in row) for row in bits]


def render_braille(points: np.ndarray, w: int, h: int) -> list[str]:
    return _chars(_bits(points, w, h))


def _rot(ay: float, ax: float) -> np.ndarray:
    cy, sy, cx, sx = np.cos(ay), np.sin(ay), np.cos(ax), np.sin(ax)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    return rx @ ry


class CoreWidget(Widget):
    DEFAULT_CSS = "CoreWidget { background: #05080C; }"
    state: reactive[HUDState] = reactive(HUDState.IDLE)

    BASE_OMEGA = 0.35  # rad/s at rest

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._pts = _cloud()  # precomputed once
        self._angle = 0.0
        self._omega = self.BASE_OMEGA
        self._rms = 0.0
        self._pulse = 0.0

    def on_mount(self) -> None:
        self.set_interval(1 / config.HUD_FPS, lambda: self.tick(1 / config.HUD_FPS))

    def push_audio(self, frame: AudioTelemetryFrame) -> None:
        self._rms = float(np.clip(frame.rms, 0.0, 1.0))

    def watch_state(self, *_: object) -> None:
        self.refresh()

    def tick(self, dt: float) -> None:
        target = self.BASE_OMEGA + 4.0 * self._rms
        k = 1 - np.exp(-(8.0 if target > self._omega else 1.5) * dt)  # fast attack, slow decay
        self._omega += (target - self._omega) * k
        self._angle = (self._angle + self._omega * dt) % (2 * np.pi)
        amp = self._rms if self.state is HUDState.SPEAKING else 0.0
        self._pulse += (amp - self._pulse) * (1 - np.exp(-12.0 * dt))
        self._rms *= np.exp(-3.0 * dt)  # level decays if no new frames arrive
        self.refresh()

    def render(self) -> Text:
        w, h = max(self.size.width, 1), max(self.size.height, 1)
        p = self._pts @ _rot(self._angle, 0.45).T
        p[:, :2] *= 1 + 0.18 * self._pulse
        near = _bits(p[p[:, 2] > 0], w, h)
        far = _bits(p[p[:, 2] <= 0][::2], w, h)  # sparser, dimmer back half
        bright = _COLORS.get(self.state, config.STARK_CYAN)
        if self.state is HUDState.SPEAKING:
            bright = "#BFF0FF"
        dim = config.STARK_DIM
        out = Text(no_wrap=True)
        for ny, fy in zip(_chars(near), _chars(far | near)):
            for n, f in zip(ny, fy):
                out.append(f, style=bright if n != " " else dim)
            out.append("\n")
        return out
