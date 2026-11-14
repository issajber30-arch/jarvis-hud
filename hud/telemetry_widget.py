"""Left rail: live system telemetry — clock, per-core CPU, history graphs, memory, disk, power, net, top apps."""
from __future__ import annotations

import subprocess
import time
from collections import deque
from datetime import datetime

import psutil
from rich.text import Text
from textual.widget import Widget

import config
from contracts import SystemTelemetryFrame

_SPARK = " ▁▂▃▄▅▆▇█"
HISTORY = 60  # seconds of history in the graphs
TOP_EVERY_S = 3  # process scan is the costliest read; refresh it less often


def _bar(pct: float, width: int = 14) -> str:
    n = round(max(0.0, min(pct, 100.0)) / 100 * width)
    return "█" * n + "░" * (width - n)


def _spark(values, width: int, top: float | None = None) -> str:
    """Last `width` values as a block-character graph, scaled to `top` (or the max seen)."""
    vals = list(values)[-width:]
    if not vals:
        return " " * width
    peak = top or max(max(vals), 1e-9)
    s = "".join(_SPARK[min(8, round(max(v, 0) / peak * 8))] for v in vals)
    return s.rjust(width)


def _graph(values, width: int, rows: int, top: float | None = None) -> list[str]:
    """Last `width` values as a `rows`-tall block graph (top row first), scaled to `top` (or the max seen)."""
    vals = list(values)[-width:]
    if not vals:
        return [" " * width] * rows
    peak = top or max(max(vals), 1e-9)
    eighths = [round(min(max(v, 0) / peak, 1) * rows * 8) for v in vals]  # height in 1/8-cell steps
    out = []
    for r in range(rows - 1, -1, -1):  # r = cells below this row
        out.append("".join(_SPARK[min(8, max(0, e - r * 8))] for e in eighths).rjust(width))
    return out


def _dur(seconds: float) -> str:
    h, m = divmod(int(seconds) // 60, 60)
    return f"{h // 24}D {h % 24:02d}H" if h >= 24 else f"{h:02d}H {m:02d}M"


class TelemetryWidget(Widget):
    DEFAULT_CSS = "TelemetryWidget { border: round #6B3F0E; padding: 0 1; }"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.border_title = "SYS"
        self._net = psutil.net_io_counters()
        self._net_t = time.monotonic()
        self.frame: SystemTelemetryFrame | None = None
        self._cpu_hist: deque[float] = deque(maxlen=HISTORY)
        self._net_hist: deque[float] = deque(maxlen=HISTORY)
        self._cores: list[float] = []
        self._top: list[tuple[str, float]] = []
        self._top_t = 0.0
        psutil.cpu_percent(percpu=True)  # prime: the first per-core read is always 0
        self._top_apps()  # prime per-process CPU too, or the first TOP list shows all 0%

    def on_mount(self) -> None:
        self.update_frame()
        self.set_interval(1 / config.TELEMETRY_HZ, self.update_frame)

    def update_frame(self) -> None:
        self.frame = self.collect()
        self._cores = psutil.cpu_percent(percpu=True)
        self._cpu_hist.append(self.frame.cpu_pct)
        self._net_hist.append(self.frame.net_up_kbps + self.frame.net_down_kbps)
        if time.monotonic() - self._top_t >= TOP_EVERY_S:
            self._top, self._top_t = self._top_apps(), time.monotonic()
        self.refresh()

    @staticmethod
    def _top_apps(n: int = 5) -> list[tuple[str, float]]:
        procs = []
        for p in psutil.process_iter(["name", "cpu_percent"]):
            try:
                procs.append((p.info["name"] or "?", p.info["cpu_percent"] or 0.0))
            except (psutil.Error, KeyError):
                continue
        return sorted(procs, key=lambda x: -x[1])[:n]

    def collect(self) -> SystemTelemetryFrame:
        now = time.monotonic()
        net = psutil.net_io_counters()
        dt = max(now - self._net_t, 1e-3)
        up = max(net.bytes_sent - self._net.bytes_sent, 0) / dt / 1024
        down = max(net.bytes_recv - self._net.bytes_recv, 0) / dt / 1024
        self._net, self._net_t = net, now
        bat = psutil.sensors_battery() if hasattr(psutil, "sensors_battery") else None
        try:
            r = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True, text=True, timeout=1,
            )
            branch = r.stdout.strip() or None if r.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            branch = None
        return SystemTelemetryFrame(
            cpu_pct=psutil.cpu_percent(None),  # non-blocking; first call reads 0
            ram_pct=psutil.virtual_memory().percent,
            battery_pct=bat.percent if bat else None,
            charging=bool(bat and bat.power_plugged),
            net_up_kbps=up,
            net_down_kbps=down,
            git_branch=branch,
            ts=time.time(),
        )

    def render(self) -> Text:
        f = self.frame
        t = Text(no_wrap=True, overflow="crop")
        if f is None:
            return t
        hi, dim, warn = config.HUD_PRIMARY, config.HUD_DIM, config.HUD_ERROR
        width = max(self.size.width, 30)
        graph_w = max(width - 2, 10)

        def head(label: str) -> None:
            t.append(f"\n{label} ", style=dim)
            t.append("─" * max(width - len(label) - 2, 0) + "\n", style=dim)

        now = datetime.now()
        t.append(now.strftime("%H:%M:%S"), style=f"bold {config.HUD_GLOW}")
        t.append(now.strftime("  %a %d %b").upper() + "\n", style=dim)

        head("CORES")
        for i, pct in enumerate(self._cores):
            t.append(f"C{i:<2} ", style=dim)
            t.append(_bar(pct, graph_w - 9), style=warn if pct > 90 else hi)
            t.append(f"{pct:4.0f}\n", style=hi)

        head("CPU 60S")
        for row in _graph(self._cpu_hist, graph_w, 4, 100):
            t.append(row + "\n", style=hi)
        t.append(f"AVG {f.cpu_pct:3.0f}%", style=hi)
        load = psutil.getloadavg() if hasattr(psutil, "getloadavg") else (0, 0, 0)
        t.append(f"   LOAD {load[0]:.2f}\n", style=dim)

        vm, du = psutil.virtual_memory(), psutil.disk_usage("/")
        head("MEM")
        t.append(_bar(vm.percent, graph_w - 6), style=warn if vm.percent > 90 else hi)
        t.append(f" {vm.percent:3.0f}%\n", style=hi)
        t.append(f"{vm.used / 2**30:4.1f} / {vm.total / 2**30:.0f} GB\n", style=dim)
        head("DISK")
        t.append(_bar(du.percent, graph_w - 6), style=warn if du.percent > 90 else hi)
        t.append(f" {du.percent:3.0f}%\n", style=hi)
        t.append(f"{du.free / 2**30:4.0f} GB FREE\n", style=dim)

        head("PWR")
        if f.battery_pct is None:
            t.append("AC POWER\n", style=hi)
        else:
            low = f.battery_pct < 15 and not f.charging
            t.append(_bar(f.battery_pct, graph_w - 6), style=warn if low else hi)
            t.append(f" {f.battery_pct:3.0f}%\n", style=hi)
            bat = psutil.sensors_battery()
            left = bat.secsleft if bat else None
            status = "CHARGING" if f.charging else (
                _dur(left) + " LEFT" if isinstance(left, int) and left > 0 else "ON BATTERY")
            t.append(status + "\n", style=dim)

        head("NET")
        for row in _graph(self._net_hist, graph_w, 3):
            t.append(row + "\n", style=hi)
        t.append(f"▲ {f.net_up_kbps:6.0f} KB/S  ▼ {f.net_down_kbps:6.0f} KB/S\n", style=hi)

        head("UPTIME")
        t.append(_dur(time.time() - psutil.boot_time()), style=hi)
        t.append(f"   GIT {(f.git_branch or '--').upper()}\n", style=dim)

        head("TOP")
        for name, cpu in self._top:
            t.append(f"{name[: graph_w - 7].upper():<{graph_w - 6}}", style=hi)
            t.append(f"{cpu:5.0f}%\n", style=dim)
        return t
