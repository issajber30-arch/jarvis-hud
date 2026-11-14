"""Hero widget: a holographic data sphere — nested shells of broken arcs, data shards and a swirling core —
drawn in shimmering binary digits. Original design in the holographic-HUD genre; no film assets."""
from __future__ import annotations

import time

import numpy as np
from rich.text import Text
from textual.reactive import reactive
from textual.widget import Widget

import config
from contracts import AudioTelemetryFrame, HUDState

# Braille dot bit for (row, col) inside a 2x4 cell.
_BIT = np.array([[0x01, 0x08], [0x02, 0x10], [0x04, 0x20], [0x40, 0x80]], dtype=np.uint8)

LABEL = "J A R V I S"
_DOTS_PER_UNIT = 130  # samples per unit length: keeps lines <=1 screen dot apart up to ~240x70 cells
_SHELL_RADII = np.linspace(0.50, 1.32, 8)  # outer shell uses the room the old orbit rings had
_SHELL_SPIN = np.array([0.6, 0.9, 0.45, 1.1, 0.7, 0.5, 0.8, 0.4])  # all drift right, each at its own pace
_SHELL_TILT = np.array([0.30, 0.85, -0.40, 1.10, 0.15, -0.95, 0.60, -0.20])


def _samples(length: float) -> int:
    return max(2, int(np.ceil(length * _DOTS_PER_UNIT)) + 1)


def _basis(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Two orthonormal vectors spanning a random great circle."""
    n = rng.normal(size=3)
    n /= np.linalg.norm(n)
    u = np.cross(n, [0.0, 1.0, 0.0] if abs(n[1]) < 0.9 else [1.0, 0.0, 0.0])
    u /= np.linalg.norm(u)
    return u, np.cross(n, u)


def _arc(rng: np.random.Generator, radius: float) -> np.ndarray:
    u, v = _basis(rng)
    start, span = rng.uniform(0, 2 * np.pi), rng.uniform(0.25, 1.5)
    t = np.linspace(start, start + span, _samples(span * radius))
    return radius * (np.outer(np.cos(t), u) + np.outer(np.sin(t), v))


def _shard(rng: np.random.Generator, radius: float) -> np.ndarray:
    """A short radial segment: reads as a data fragment between shells."""
    d = rng.normal(size=3)
    d /= np.linalg.norm(d)
    length = rng.uniform(0.05, 0.14)
    return np.outer(np.linspace(radius, radius + length, _samples(length)), d)


def _data_sphere(seed: int = 7) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Per-shell point arrays, plus every individual line (for continuity checks)."""
    rng = np.random.default_rng(seed)  # fixed layout: the same sphere every launch
    shells, lines = [], []
    for r in _SHELL_RADII:
        parts = [_arc(rng, r) for _ in range(int(8 + 14 * r))]
        parts += [_shard(rng, r) for _ in range(int(4 + 6 * r))]
        lines += parts
        shells.append(np.vstack(parts))
    return shells, lines


def _core(n: int = 700, radius: float = 0.2) -> np.ndarray:
    """Dense inner ball plus a two-armed spiral: the bright swirling heart."""
    i = np.arange(n) + 0.5
    phi, theta = np.arccos(1 - 2 * i / n), np.pi * (1 + 5**0.5) * i
    ball = np.stack([np.cos(theta) * np.sin(phi), np.cos(phi), np.sin(theta) * np.sin(phi)], 1)
    ball *= radius * 0.6 * np.sqrt((np.arange(n) % 4 + 1) / 4)[:, None]
    t = np.linspace(0, 3 * np.pi, _samples(3 * np.pi * radius))
    rr = radius * (0.25 + 0.75 * t / t[-1])
    arms = [np.stack([rr * np.cos(t + k), 0.15 * rr * np.sin(2 * t), rr * np.sin(t + k)], 1) for k in (0, np.pi)]
    return np.vstack([ball] + arms)


def _cloud() -> np.ndarray:
    """All shells as one point cloud."""
    return np.vstack(_data_sphere()[0])


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


_BPM = {HUDState.IDLE: 60, HUDState.LISTENING: 72, HUDState.FOLLOWUP: 72, HUDState.THINKING: 88,
        HUDState.SPEAKING: 66, HUDState.CONFIRM: 92, HUDState.ERROR: 110}
_BEAT_W = 0.07  # pulse width as a fraction of one beat; wide enough to look smooth at 30 fps


def heartbeat(phase) -> np.ndarray:
    """Lub-dub over one beat (phase 0..1): a strong pulse, a softer one, then rest."""
    p = np.asarray(phase, dtype=float)

    def pulse(centre: float, amp: float) -> np.ndarray:
        d = (p - centre + 0.5) % 1.0 - 0.5  # wrap so the beat is seamless across phase 1 -> 0
        return amp * np.exp(-(d / _BEAT_W) ** 2)

    return pulse(0.05, 1.0) + pulse(0.29, 0.6)


def _glide(cur: float, target: float, rate: float, dt: float) -> float:
    """Frame-rate independent easing: every parameter change is a glide, never a jump."""
    return cur + (target - cur) * (1 - np.exp(-rate * dt))


class CoreWidget(Widget):
    DEFAULT_CSS = "CoreWidget { background: #070503; }"
    state: reactive[HUDState] = reactive(HUDState.IDLE)

    BASE_OMEGA = 0.35  # rad/s at rest

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._shells, self._arcs = _data_sphere()  # precomputed once
        self._core = _core()
        self._angle = 0.0
        self._omega = self.BASE_OMEGA
        self._rms = 0.0
        self._pulse = 0.0
        self._digits = np.zeros((0, 0), dtype="<U1")  # per-cell 0/1, resized on demand
        self._rng = np.random.default_rng()
        self._beat = 0.0           # heartbeat phase, 0..1
        self._scale = 1.0          # eased shell scale (listening pulls the shells in)
        self._burst = 0.0          # error burst, decays to 0
        self._fft = np.zeros(64, np.float32)
        self._since = 0.0          # time since last drawn frame
        self._state_t0 = time.monotonic()
        self._last_state = HUDState.IDLE
        self._visible = len(self._shells)
        # Directions and steps for the listening voice ring (radial spectrum ticks), built once.
        a = np.linspace(0, 2 * np.pi, 128, endpoint=False)
        self._ring_dirs = np.stack([np.cos(a), np.zeros_like(a), np.sin(a)], 1)
        self._ring_t = np.linspace(0, 1, 34)
        self._ring_amp_index = np.abs(np.linspace(-1, 1, 128, endpoint=False)) * 63  # mirror 64 bins around

    # ---- public knobs -------------------------------------------------------------------------------
    @property
    def bpm(self) -> int:
        return _BPM.get(self.state, 60)

    @property
    def target_fps(self) -> int:
        # Always full rate: the heartbeat pulse visibly steps at 15 fps (owner chose smooth over battery).
        return config.HUD_FPS

    @property
    def shell_scale(self) -> float:
        return self._scale * (1 + 0.4 * self._burst)

    def set_boot_progress(self, fraction: float) -> None:
        self._visible = max(1, int(np.ceil(np.clip(fraction, 0, 1) * len(self._shells))))
        self.refresh()

    def on_mount(self) -> None:
        self.set_interval(1 / config.HUD_FPS, lambda: self.tick(1 / config.HUD_FPS))

    def push_audio(self, frame: AudioTelemetryFrame) -> None:
        self._rms = float(np.clip(frame.rms, 0.0, 1.0))
        self._fft = 0.5 * self._fft + 0.5 * np.asarray(frame.fft, np.float32)

    def watch_state(self, *_: object) -> None:
        self._on_state_change()
        self.refresh()

    def _on_state_change(self) -> None:
        if self.state is not self._last_state:
            self._last_state, self._state_t0 = self.state, time.monotonic()
            if self.state is HUDState.ERROR:
                self._burst = 1.0

    # ---- animation clock ----------------------------------------------------------------------------
    def tick(self, dt: float) -> None:
        self._on_state_change()
        target = self.BASE_OMEGA + 4.0 * self._rms
        k = 1 - np.exp(-(8.0 if target > self._omega else 1.5) * dt)  # fast attack, slow decay
        self._omega += (target - self._omega) * k
        self._angle = (self._angle + self._omega * dt) % (2 * np.pi)
        amp = self._rms if self.state is HUDState.SPEAKING else 0.0
        self._pulse += (amp - self._pulse) * (1 - np.exp(-12.0 * dt))
        self._rms *= np.exp(-3.0 * dt)  # level decays if no new frames arrive
        self._fft *= np.exp(-4.0 * dt)
        self._beat = (self._beat + dt * self.bpm / 60) % 1.0  # phase carries over when the rate changes
        listening = self.state in (HUDState.LISTENING, HUDState.FOLLOWUP)
        target = 0.9 if listening else 0.82 if self.state is HUDState.CONFIRM else 1.0  # make room for rings
        self._scale = _glide(self._scale, target, 3.0, dt)
        self._burst *= np.exp(-3.0 * dt)
        self._since += dt
        if self._since >= 1 / self.target_fps - 1e-6:
            self._since = 0.0
            self.refresh()

    # ---- geometry -----------------------------------------------------------------------------------
    def _voice_ring(self) -> np.ndarray:
        """Radial spectrum ticks on a ring round the sphere: your voice, live, while listening."""
        amp = np.interp(self._ring_amp_index, np.arange(64), self._fft)
        r = 1.22 + 0.22 * self._ring_t  # just outside the pulled-in shells, so it reads as its own ring
        pts = self._ring_dirs[:, None, :] * r[None, :, None]
        keep = self._ring_t[None, :] <= amp[:, None] * 1.0
        return pts[keep] @ _rot(self._angle * 0.5, 1.15).T

    def _confirm_ring(self) -> np.ndarray:
        """An arc that drains clockwise over the confirmation window."""
        frac = float(np.clip(1 - (time.monotonic() - self._state_t0) / config.CONFIRM_TIMEOUT_S, 0, 1))
        if frac <= 0:
            return np.zeros((0, 3))
        th = np.pi / 2 - np.linspace(0, 2 * np.pi * frac, _samples(2 * np.pi * 1.3 * frac))
        return np.stack([1.3 * np.cos(th), 1.3 * np.sin(th), np.full_like(th, 0.5)], 1)

    def _layers(self, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
        """Braille bits and per-cell layer index for one frame."""
        a = self._angle
        hb = float(heartbeat(self._beat))
        shells = np.vstack([pts @ _rot(a * spin, tilt, 0.2 * i).T
                            for i, (pts, spin, tilt) in enumerate(
                                zip(self._shells[:self._visible], _SHELL_SPIN, _SHELL_TILT))])
        shells = shells * (self.shell_scale * (1 + 0.03 * hb))
        front, back = shells[shells[:, 2] > 0], shells[shells[:, 2] <= 0][::2]  # sparser, dimmer back half
        glowing = [self._core @ _rot(-a * 2.0, 0.6).T * ((1 + 0.9 * self._pulse + 0.5 * self._rms) * (1 + 0.25 * hb))]
        extra = []
        if self.state is HUDState.THINKING:  # a scan band sweeps up and down through the sphere
            band = np.sin((time.monotonic() - self._state_t0) * 2.2) * 1.1
            glowing.append(shells[np.abs(shells[:, 1] - band) < 0.08])
        if self.state in (HUDState.LISTENING, HUDState.FOLLOWUP):
            glowing.append(self._voice_ring())  # rings glow so they stand apart from the shells
        if self.state is HUDState.CONFIRM:
            glowing.append(self._confirm_ring())
        if extra:
            front = np.vstack([front] + extra)
        b_back, b_front, b_core = _bits(back, w, h), _bits(front, w, h), _bits(np.vstack(glowing), w, h)
        layer = np.where(b_core > 0, _CORE, np.where(b_front > 0, _BRIGHT, np.where(b_back > 0, _DIM, 0)))
        return b_back | b_front | b_core, layer

    def _binary(self, bits: np.ndarray) -> list[str]:
        """Lit cells show a 0 or 1 that flip over time; faster while thinking, like data being crunched."""
        if self._digits.shape != bits.shape:
            self._digits = self._rng.choice(np.array(["0", "1"]), size=bits.shape)
        rate = 0.2 if self.state is HUDState.THINKING else 0.04
        flip = self._rng.random(bits.shape) < rate
        self._digits[flip] = np.where(self._digits[flip] == "0", "1", "0")
        return ["".join(row) for row in np.where(bits > 0, self._digits, " ")]

    def render(self) -> Text:
        w, h = max(self.size.width, 1), max(self.size.height, 1)
        bits, layer = self._layers(w, h)
        rows = self._binary(bits)
        label = f" {LABEL} "
        if len(label) <= w:  # stamp the name on the centre row, padded so it reads over the core
            mid, x0 = h // 2, (w - len(label)) // 2
            rows[mid] = rows[mid][:x0] + label + rows[mid][x0 + len(label):]
            layer[mid, x0:x0 + len(label)] = _TEXT

        bright = {HUDState.CONFIRM: config.HUD_CONFIRM, HUDState.ERROR: config.HUD_ERROR}.get(
            self.state, config.HUD_PRIMARY)
        if self.state is HUDState.SPEAKING:
            bright = config.HUD_SPEAKING
        styles = {_DIM: config.HUD_DIM, _BRIGHT: bright, _CORE: config.HUD_GLOW, _TEXT: f"bold {config.HUD_GLOW}"}
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
