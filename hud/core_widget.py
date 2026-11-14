"""Hero widget: a rotating wireframe globe with two orbit rings and a glowing core, drawn on a Braille canvas."""
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
_GLOW = "#E6FBFF"
LABEL = "J A R V I S"


def _circle(n: int) -> np.ndarray:
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([np.cos(a), np.zeros_like(a), np.sin(a)], 1)


def _globe(n_lat: int = 7, n_lon: int = 12, per_line: int = 220) -> np.ndarray:
    """Latitude circles + meridians on the unit sphere."""
    c = _circle(per_line)
    lats = [c * np.cos(t) + [0, np.sin(t), 0] for t in np.linspace(-np.pi / 2, np.pi / 2, n_lat + 2)[1:-1]]
    t = np.linspace(0, 2 * np.pi, per_line, endpoint=False)
    lons = [np.stack([np.cos(t) * np.cos(p), np.sin(t), np.cos(t) * np.sin(p)], 1)
            for p in np.linspace(0, np.pi, n_lon, endpoint=False)]
    return np.vstack(lats + lons)


def _orbit(radius: float, n: int = 420, dash: int = 14) -> np.ndarray:
    """A dashed ring in the xz plane; dashes make its spin visible."""
    pts = radius * _circle(n)
    return pts[(np.arange(n) // dash) % 3 != 2]


def _core(n: int = 900, radius: float = 0.24) -> np.ndarray:
    """A solid ball: fibonacci points on several nested shells."""
    i = np.arange(n) + 0.5
    phi, theta = np.arccos(1 - 2 * i / n), np.pi * (1 + 5**0.5) * i
    shell = np.stack([np.cos(theta) * np.sin(phi), np.cos(phi), np.sin(theta) * np.sin(phi)], 1)
    r = radius * np.sqrt((np.arange(n) % 4 + 1) / 4)[:, None]
    return shell * r


def _cloud() -> np.ndarray:
    """Kept for callers/tests that want a plain point cloud: the globe."""
    return _globe()


def _bits(points: np.ndarray, w: int, h: int, scale: float = 1.0) -> np.ndarray:
    """Project x/y of `points` (about -1..1) to an (h, w) array of Braille bit masks."""
    out = np.zeros((h, w), dtype=np.uint8)
    if len(points) == 0 or w < 1 or h < 1:
        return out
    px, py = 2 * w, 4 * h
    s = min(px, py) / 2 * 0.72 * scale  # leaves room for the orbit rings
    x = np.round(px / 2 + points[:, 0] * s).astype(int)
    y = np.round(py / 2 - points[:, 1] * s).astype(int)
    ok = (x >= 0) & (x < px) & (y >= 0) & (y < py)
    x, y = x[ok], y[ok]
    np.bitwise_or.at(out, (y // 4, x // 2), _BIT[y % 4, x % 2])
    return out


_GLYPH = np.array([" "] + [chr(0x2800 + i) for i in range(1, 256)])  # bit mask -> Braille cell


def _chars(bits: np.ndarray) -> list[str]:
    return ["".join(row) for row in _GLYPH[bits]]


def render_braille(points: np.ndarray, w: int, h: int) -> list[str]:
    return _chars(_bits(points, w, h))


def _rot(ay: float, ax: float, az: float = 0.0) -> np.ndarray:
    cy, sy, cx, sx, cz, sz = np.cos(ay), np.sin(ay), np.cos(ax), np.sin(ax), np.cos(az), np.sin(az)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ rx @ ry


# Cell layers, painted in increasing priority.
_DIM, _BRIGHT, _CORE, _TEXT = 1, 2, 3, 4


class CoreWidget(Widget):
    DEFAULT_CSS = "CoreWidget { background: #05080C; }"
    state: reactive[HUDState] = reactive(HUDState.IDLE)

    BASE_OMEGA = 0.35  # rad/s at rest

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._globe = _globe()  # precomputed once
        self._orbit_a, self._orbit_b = _orbit(1.32), _orbit(1.22)
        self._core = _core()
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

    def _layers(self, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
        """Braille bits and per-cell layer index for one frame."""
        a = self._angle
        globe = self._globe @ _rot(a, 0.35).T
        rings = np.vstack([self._orbit_a @ _rot(a * 1.7, 1.15, 0.30).T,
                           self._orbit_b @ _rot(-a * 1.2, 1.25, -0.55).T])
        front = np.vstack([globe[globe[:, 2] > 0], rings[rings[:, 2] > 0]])
        back = np.vstack([globe[globe[:, 2] <= 0][::2], rings[rings[:, 2] <= 0][::2]])  # sparser back half
        glow = 1 + 0.9 * self._pulse + 0.5 * self._rms  # core breathes with either voice
        core = self._core @ _rot(-a * 2.0, 0.6).T * glow
        b_back, b_front, b_core = _bits(back, w, h), _bits(front, w, h), _bits(core, w, h)
        layer = np.where(b_core > 0, _CORE, np.where(b_front > 0, _BRIGHT, np.where(b_back > 0, _DIM, 0)))
        return b_back | b_front | b_core, layer

    def render(self) -> Text:
        w, h = max(self.size.width, 1), max(self.size.height, 1)
        bits, layer = self._layers(w, h)
        rows = _chars(bits)
        label = f" {LABEL} "
        if len(label) <= w:  # stamp the name on the centre row, padded so it reads over the core
            mid, x0 = h // 2, (w - len(label)) // 2
            rows[mid] = rows[mid][:x0] + label + rows[mid][x0 + len(label):]
            layer[mid, x0:x0 + len(label)] = _TEXT

        bright = _COLORS.get(self.state, config.STARK_CYAN)
        if self.state is HUDState.SPEAKING:
            bright = "#BFF0FF"
        styles = {_DIM: config.STARK_DIM, _BRIGHT: bright, _CORE: _GLOW, _TEXT: f"bold {_GLOW}"}
        out = Text(no_wrap=True)
        for text_row, layer_row in zip(rows, layer.tolist()):
            # one span per style run; blank cells join the current run (they show no colour)
            run, style = [], None
            for ch, li in zip(text_row, layer_row):
                s = styles[li] if li else style
                if s != style and run:
                    out.append("".join(run), style=style)
                    run = []
                style = s
                run.append(ch)
            out.append("".join(run), style=style)
            out.append("\n")
        return out
