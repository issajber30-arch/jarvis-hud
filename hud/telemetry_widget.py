"""Left rail: system telemetry (CPU, RAM, power, net, git branch)."""
from __future__ import annotations

import subprocess
import time

import psutil
from rich.text import Text
from textual.widget import Widget

import config
from contracts import SystemTelemetryFrame


def _bar(pct: float, width: int = 14) -> str:
    n = round(max(0.0, min(pct, 100.0)) / 100 * width)
    return "█" * n + "░" * (width - n)


class TelemetryWidget(Widget):
    DEFAULT_CSS = "TelemetryWidget { border: round #1B4F63; padding: 0 1; }"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.border_title = "SYS"
        self._net = psutil.net_io_counters()
        self._net_t = time.monotonic()
        self.frame: SystemTelemetryFrame | None = None

    def on_mount(self) -> None:
        self.update_frame()
        self.set_interval(1 / config.TELEMETRY_HZ, self.update_frame)

    def update_frame(self) -> None:
        self.frame = self.collect()
        self.refresh()

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
        t = Text(no_wrap=True)
        if f is None:
            return t
        cyan, dim = config.STARK_CYAN, config.STARK_DIM
        rows = [("SYS.CPU", f.cpu_pct), ("SYS.RAM", f.ram_pct)]
        if f.battery_pct is not None:
            rows.append(("PWR" + ("+" if f.charging else " "), f.battery_pct))
        for label, pct in rows:
            warn = pct > 90 or (label.startswith("PWR") and pct < 15 and not f.charging)
            t.append(f"{label:<8}", style=dim)
            t.append(_bar(pct), style=config.STARK_RED if warn else cyan)
            t.append(f" {pct:3.0f}%\n", style=cyan)
        if f.battery_pct is None:
            t.append("PWR     AC\n", style=dim)
        t.append("NET ▲▼  ", style=dim)
        t.append(f"{f.net_up_kbps:.0f}/{f.net_down_kbps:.0f} KB/S\n", style=cyan)
        t.append("GIT     ", style=dim)
        t.append((f.git_branch or "--").upper(), style=cyan)
        return t
