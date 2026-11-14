"""Shared types every part of Jarvis codes against. Only the main thread edits this file."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

import numpy as np


class HUDState(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    CONFIRM = "confirm"
    FOLLOWUP = "followup"
    ERROR = "error"


_S = HUDState
TRANSITIONS: dict[HUDState, set[HUDState]] = {
    _S.IDLE: {_S.LISTENING, _S.ERROR},
    _S.LISTENING: {_S.THINKING, _S.IDLE, _S.ERROR},
    _S.THINKING: {_S.SPEAKING, _S.CONFIRM, _S.IDLE, _S.ERROR},
    _S.SPEAKING: {_S.IDLE, _S.LISTENING, _S.FOLLOWUP, _S.CONFIRM, _S.ERROR},
    _S.CONFIRM: {_S.THINKING, _S.SPEAKING, _S.IDLE},
    _S.FOLLOWUP: {_S.LISTENING, _S.IDLE},
    _S.ERROR: {_S.IDLE},
}


def can_transition(a: HUDState, b: HUDState) -> bool:
    return b in TRANSITIONS[a]


class Tier(str, Enum):
    IMMEDIATE = "immediate"
    CONFIRM = "confirm"
    BLOCKED = "blocked"


# Action registry: name -> tier. Args are documented in comments; handlers live in tools/mac_controller.py.
ACTIONS: dict[str, Tier] = {
    "open_app": Tier.IMMEDIATE,          # {app}
    "quit_app": Tier.IMMEDIATE,          # {app}
    "set_volume": Tier.IMMEDIATE,        # {level: 0-100}
    "volume_up": Tier.IMMEDIATE,
    "volume_down": Tier.IMMEDIATE,
    "mute": Tier.IMMEDIATE,
    "set_brightness": Tier.IMMEDIATE,    # {level: 0-100}
    "music_play": Tier.IMMEDIATE,
    "music_pause": Tier.IMMEDIATE,
    "music_next": Tier.IMMEDIATE,
    "music_previous": Tier.IMMEDIATE,
    "set_timer": Tier.IMMEDIATE,         # {minutes}
    "system_status": Tier.IMMEDIATE,
    "agenda_today": Tier.IMMEDIATE,
    "create_event": Tier.IMMEDIATE,      # {title, start, end}  ISO-8601 local
    "create_reminder": Tier.IMMEDIATE,   # {title, due?}        ISO-8601 local
    "delete_event": Tier.CONFIRM,        # {id}
    "reschedule_event": Tier.CONFIRM,    # {id, start, end}
    "delete_reminder": Tier.CONFIRM,     # {id}
    "move_file": Tier.CONFIRM,           # {src, dst}
    "delete_file": Tier.CONFIRM,         # {path}
    "run_shell": Tier.CONFIRM,           # {cmd}
    "send_message": Tier.CONFIRM,        # {to, text}
}

FAST_ROUTE_ACTIONS: frozenset[str] = frozenset(
    k for k, v in ACTIONS.items() if v is Tier.IMMEDIATE
) - {"create_event", "create_reminder"}


@dataclass(frozen=True)
class AudioTelemetryFrame:
    rms: float
    fft: np.ndarray  # 64 bins, 0..1
    vad: bool
    ts: float


@dataclass(frozen=True)
class SystemTelemetryFrame:
    cpu_pct: float
    ram_pct: float
    battery_pct: float | None
    charging: bool
    net_up_kbps: float
    net_down_kbps: float
    git_branch: str | None
    ts: float


@dataclass(frozen=True)
class FastRouteMatch:
    action: str
    args: dict
    score: float
    phrase: str
    latency_ms: float


@dataclass(frozen=True)
class Action:
    name: str
    args: dict = field(default_factory=dict)

    @property
    def tier(self) -> Tier:
        return ACTIONS.get(self.name, Tier.BLOCKED)  # unknown names are never executed


@dataclass(frozen=True)
class ActionResult:
    action: str
    ok: bool
    message: str


@dataclass
class ListenerEvent:
    kind: Literal["wake", "ptt_start", "ptt_end", "utterance", "barge_in"]
    audio: np.ndarray | None = None


@dataclass
class JarvisResponse:
    transcript: str
    spoken: list[str]
    actions: list[ActionResult]
    via: Literal["fast", "claude", "none"]
    followup: bool
