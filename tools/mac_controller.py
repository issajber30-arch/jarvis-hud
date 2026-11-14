"""Tier-enforcing action executor.

Safety rules: tiers come from contracts.ACTIONS, never from the model. Every subprocess uses an argv list.
User/LLM text reaches AppleScript only as `osascript` argv items (`on run argv`), never inside script source.
The single shell=True-style call is run_shell, reachable only after check_shell() and an explicit confirmation.
"""
from __future__ import annotations

import asyncio
import os
import re
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Awaitable, Callable

from contracts import ACTIONS, Action, ActionResult, Tier
from tools import calendar as cal
from tools import lines
from tools.redact import redact

# ---------------------------------------------------------------- safety checks

_SECRET_DIRS = {".ssh", ".aws", ".gnupg", ".kube", ".docker"}
_SECRET_SUFFIX = {".pem", ".key", ".p12", ".pfx", ".keychain", ".keychain-db", ".kdbx"}
_SECRET_WORDS = ("key", "token", "secret", "password", "passwd", "credential", "id_rsa", "id_ed25519")


def is_secret_path(path: str) -> bool:
    p = Path(path.strip("\"'")).expanduser()
    try:
        p = p.resolve()  # follow symlinks so a link to ~/.ssh is still caught
    except OSError:
        pass
    parts = [x.lower() for x in p.parts]
    name = parts[-1] if parts else ""
    if _SECRET_DIRS & set(parts) or any(a == ".config" and b == "gh" for a, b in zip(parts, parts[1:])):
        return True
    if name.startswith(".env") or p.suffix.lower() in _SECRET_SUFFIX:
        return True
    return any(w in name for w in _SECRET_WORDS)


_BLOCKED_WORDS = {"sudo", "su", "doas", "security", "launchctl", "mkfs", "csrutil", "op", "bw", "bitwarden",
                  "1password", "eval", "osascript", "dscl", "passwd", "env", "printenv", "curl", "wget", "base64",
                  "openssl", "xargs"}
_BLOCKED_RE = [re.compile(p) for p in (
    r"keychain", r"1password", r"bitwarden", r"diskutil\s+(?:secure)?erase", r"\bmkfs\b",
    r"\brm\s+(?:-\w+\s+)*-\w*[rf]\w*\s+(?:-\w+\s+)*(?:/|~|\$home)(?:\*|\s|$)", r"base64\s+(?:-d|--decode|-D)",
    r"\bxxd\s+-r", r"\bdd\s+.*\bof=/dev/",
    r"\b(?:python[\d.]*|perl|ruby|node)\s+(?:-\w+\s+)*-\w*[ce]\b", r"\b(?:ba|z)?sh\s+(?:-\w+\s+)*-\w*c\b")]
_CHAIN = re.compile(r"\$\(|`|[|;<>]|&&")  # substitution, chaining, redirection: checked on the raw command
_MAX_CMD = 200


def _normalise(cmd: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[\"'\\`]", "", cmd.lower())).strip()


def check_shell(cmd: str) -> Tier:
    if len(cmd) > _MAX_CMD or _CHAIN.search(cmd):
        return Tier.BLOCKED
    n = _normalise(cmd)
    tokens = [t for t in re.split(r"[\s;|&()<>=,]+", n) if t]
    names = {t.rsplit("/", 1)[-1] for t in tokens} | {t.rsplit("/", 1)[-1].split(".")[0] for t in tokens}
    if names & _BLOCKED_WORDS or any(r.search(n) for r in _BLOCKED_RE):
        return Tier.BLOCKED
    if any(is_secret_path(t) for t in tokens if "/" in t or t.startswith(".") or "." in t or t.startswith("~")):
        return Tier.BLOCKED
    return Tier.CONFIRM


_BLOCKED_APP = re.compile(r"1?password|bitwarden|keychain|lastpass|dashlane|keeper|enpass|keepass", re.I)

# ---------------------------------------------------------------- subprocess helpers

async def _run(argv: list[str], timeout: float = 15.0) -> tuple[int, str]:
    """Run an argv list (no shell). Returns (returncode, combined output)."""
    try:
        proc = await asyncio.create_subprocess_exec(*argv, stdout=asyncio.subprocess.PIPE,
                                                    stderr=asyncio.subprocess.STDOUT)
        out, _ = await asyncio.wait_for(proc.communicate(), timeout)
        return proc.returncode or 0, out.decode(errors="replace").strip()
    except FileNotFoundError:
        return 127, f"{argv[0]} not found"
    except asyncio.TimeoutError:
        proc.kill()
        return 124, "timed out"


def _osa(script: str, *args: str) -> list[str]:
    """Build an osascript argv: script lines go in -e, user text only after `end run` as argv items."""
    argv = ["osascript", "-e", "on run argv"]
    for line in script.strip().splitlines():
        argv += ["-e", line.strip()]
    return argv + ["-e", "end run", *map(str, args)]


def _ok(a: Action, msg: str = "Done") -> ActionResult:
    return ActionResult(a.name, True, msg)


def _fail(a: Action, msg: str) -> ActionResult:
    return ActionResult(a.name, False, msg)


def _from_run(a: Action, rc: int, out: str, ok_msg: str = "Done") -> ActionResult:
    return _ok(a, ok_msg) if rc == 0 else _fail(a, redact(out)[:200] or "Command failed")


def _app(a: Action) -> str:
    name = str(a.args.get("app", "")).strip()
    if not name or len(name) > 64 or any(ord(c) < 32 for c in name) or "/" in name or ":" in name \
            or name.lower().endswith(".app"):
        raise ValueError("invalid app name")
    if _BLOCKED_APP.search(name):
        raise PermissionError("password managers are off limits")
    return name


def _level(a: Action) -> int:
    return max(0, min(100, int(a.args["level"])))


def _dt(v) -> datetime:
    return v if isinstance(v, datetime) else datetime.fromisoformat(str(v))


def _home() -> Path:
    return Path("~").expanduser().resolve()


_DOTFILES = (".zshrc", ".zprofile", ".bashrc", ".bash_profile", ".profile")
_TOP_DIRS = ("Documents", "Desktop", "Downloads", "Pictures", "Music", "Movies")


def _user_path(raw, top_ok: bool = True) -> Path:
    r = str(raw)
    if not r.startswith(("/", "~")):
        raise PermissionError("path must be absolute")
    p = Path(r).expanduser().resolve()
    h = _home()
    if is_secret_path(str(p)):
        raise PermissionError("secret path")
    if p == h or h not in p.parents:
        raise PermissionError("path must be inside the home folder")
    if p == h / "Library" or h / "Library" in p.parents or p in {h / d for d in _DOTFILES}:
        raise PermissionError("path is protected")
    if not top_ok and p in {h / d for d in _TOP_DIRS}:
        raise PermissionError("cannot touch a top-level home folder")
    return p

# ---------------------------------------------------------------- handlers

async def _open_app(a):
    rc, out = await _run(_osa("tell application (item 1 of argv) to activate", _app(a)))
    return _from_run(a, rc, out, "Opened")


async def _quit_app(a):
    rc, out = await _run(_osa("tell application (item 1 of argv) to quit", _app(a)))
    return _from_run(a, rc, out, "Quit")


async def _set_volume(a):
    lvl = _level(a)
    rc, out = await _run(_osa("set volume output volume ((item 1 of argv) as integer)", str(lvl)))
    return _from_run(a, rc, out, f"Volume {lvl}")


def _vol_step(delta: int):
    async def h(a):
        script = "set v to output volume of (get volume settings)\n" \
                 "set volume output volume (v + ((item 1 of argv) as integer))"
        rc, out = await _run(_osa(script, str(delta)))
        return _from_run(a, rc, out)
    return h


async def _mute(a):
    rc, out = await _run(_osa("set volume output muted (not (output muted of (get volume settings)))"))
    return _from_run(a, rc, out)


async def _set_brightness(a):
    lvl = _level(a)
    exe = shutil.which("brightness")  # optional CLI (brew install brightness); macOS has no scripting hook
    if not exe:
        return _fail(a, "Brightness control needs the 'brightness' tool: brew install brightness")
    rc, out = await _run([exe, f"{lvl / 100:.2f}"])
    return _from_run(a, rc, out, f"Brightness {lvl}")


def _music(cmd: str):
    async def h(a):
        rc, out = await _run(_osa(f'tell application "Music" to {cmd}'))
        return _from_run(a, rc, out)
    return h


_timers: set[asyncio.Task] = set()


async def _ring(minutes: float):
    await asyncio.sleep(minutes * 60)
    await _run(_osa("display notification (item 1 of argv) with title \"Jarvis\"", "Timer finished"))
    await _run(["afplay", "/System/Library/Sounds/Glass.aiff"])


async def _set_timer(a):
    m = float(a.args["minutes"])
    if not 0 < m <= 1440:
        return _fail(a, "Timer must be between 0 and 1440 minutes")
    t = asyncio.create_task(_ring(m))
    _timers.add(t)
    t.add_done_callback(_timers.discard)
    return _ok(a, f"Timer set for {m:g} minutes")


async def _system_status(a):
    import psutil
    cpu, ram = psutil.cpu_percent(interval=0.2), psutil.virtual_memory().percent
    b = psutil.sensors_battery()
    bat = f", battery {b.percent:.0f} percent{' charging' if b.power_plugged else ''}" if b else ""
    return _ok(a, f"CPU {cpu:.0f} percent, memory {ram:.0f} percent{bat}")


def _clock(d: datetime) -> str:
    return d.strftime("%-I:%M %p").lower().replace(":00 ", " ")


async def _agenda_today(a):
    try:
        items = await asyncio.to_thread(cal.agenda_today)
    except cal.CalendarPermissionError:
        return _fail(a, lines.pick("permission_calendar"))
    if not items:
        return _ok(a, lines.pick("agenda_empty"))
    parts = [f"{i['title']} at {_clock(i['start'])}" if i["kind"] == "event" or i["start"].hour or i["start"].minute
             else i["title"] for i in items]
    return _ok(a, lines.pick("agenda_intro") + " " + ". ".join(parts) + ".")


def _cal(fn, ok_msg):
    async def h(a):
        try:
            await asyncio.to_thread(fn, a)
        except cal.CalendarPermissionError:
            return _fail(a, lines.pick("permission_calendar"))
        return _ok(a, ok_msg)
    return h


def _w_create_event(a):
    t = str(a.args["title"]).strip()
    if not t:
        raise ValueError("title required")
    cal.create_event(t, _dt(a.args["start"]), _dt(a.args["end"]))


def _w_create_reminder(a):
    t = str(a.args["title"]).strip()
    if not t:
        raise ValueError("title required")
    due = a.args.get("due")
    cal.create_reminder(t, _dt(due) if due else None)


async def _move_file(a):
    src, dst = _user_path(a.args["src"], False), _user_path(a.args["dst"])
    if not src.exists():
        return _fail(a, "Source does not exist")
    if dst.is_dir():
        dst = dst / src.name
    if dst.exists():
        return _fail(a, "Destination already exists")
    shutil.move(str(src), str(dst))
    return _ok(a, f"Moved {src.name}")


async def _delete_file(a):
    p = _user_path(a.args["path"], False)
    if not p.exists():
        return _fail(a, "File does not exist")
    trash = _home() / ".Trash"
    trash.mkdir(exist_ok=True)
    shutil.move(str(p), str(trash / f"{p.name}.{int(time.time())}"))  # reversible: Trash, not rm
    return _ok(a, f"Moved {p.name} to the Trash")


_ENV_SECRET = ("KEY", "TOKEN", "SECRET", "PASSWORD")


async def _run_shell(a):
    cmd = str(a.args["cmd"])
    if check_shell(cmd) is Tier.BLOCKED:  # re-check at the point of execution
        return _fail(a, "Blocked command")
    try:
        env = {k: v for k, v in os.environ.items() if not any(w in k.upper() for w in _ENV_SECRET)}
        proc = await asyncio.create_subprocess_shell(cmd, stdout=asyncio.subprocess.PIPE,
                                                     stderr=asyncio.subprocess.STDOUT, env=env, cwd=Path.home())
        out, _ = await asyncio.wait_for(proc.communicate(), 30)
    except asyncio.TimeoutError:
        proc.kill()
        return _fail(a, "Command timed out")
    text = redact(out.decode(errors="replace").strip())[:300]
    return ActionResult(a.name, proc.returncode == 0, text or ("Done" if proc.returncode == 0 else "Failed"))


async def _send_message(a):
    to, text = str(a.args["to"]).strip(), str(a.args["text"])
    if not to or not text or len(text) > 1000:
        return _fail(a, "Invalid message")
    script = 'set svc to 1st account whose service type = iMessage\n' \
             'set b to participant (item 1 of argv) of svc\n' \
             'send (item 2 of argv) to b'
    rc, out = await _run(_osa(f'tell application "Messages"\n{script}\nend tell', to, redact(text)))
    return _from_run(a, rc, out, "Message sent")


HANDLERS: dict[str, Callable[[Action], Awaitable[ActionResult]]] = {
    "open_app": _open_app, "quit_app": _quit_app, "set_volume": _set_volume,
    "volume_up": _vol_step(10), "volume_down": _vol_step(-10), "mute": _mute, "set_brightness": _set_brightness,
    "music_play": _music("play"), "music_pause": _music("pause"), "music_next": _music("next track"),
    "music_previous": _music("previous track"), "set_timer": _set_timer, "system_status": _system_status,
    "agenda_today": _agenda_today,
    "create_event": _cal(_w_create_event, "Event created"),
    "create_reminder": _cal(_w_create_reminder, "Reminder created"),
    "delete_event": _cal(lambda a: cal.delete_event(str(a.args["id"])), "Event deleted"),
    "reschedule_event": _cal(lambda a: cal.reschedule_event(str(a.args["id"]), _dt(a.args["start"]),
                                                            _dt(a.args["end"])), "Event moved"),
    "delete_reminder": _cal(lambda a: cal.delete_reminder(str(a.args["id"])), "Reminder deleted"),
    "move_file": _move_file, "delete_file": _delete_file, "run_shell": _run_shell, "send_message": _send_message,
}


async def _readback(a: Action) -> str:
    g = a.args.get
    if a.name in ("delete_event", "reschedule_event", "delete_reminder"):
        try:
            title = await asyncio.to_thread(cal.title_of, str(g("id", "")))
        except Exception:  # no permission / no store: fall back to the generic wording
            title = None
        what = f"\"{title}\"" if title else "an unknown calendar item"
        verb = "Reschedule" if a.name == "reschedule_event" else "Delete"
        return redact(f"{verb} {what}")
    return redact({
        "delete_file": f"Move {Path(str(g('path', ''))).name} to the Trash",
        "move_file": f"Move {Path(str(g('src', ''))).name} to {g('dst', '')}",
        "run_shell": f"Run the command: {g('cmd', '')}",
        "send_message": f"Send a message to {g('to', '')}: {g('text', '')}",
    }.get(a.name, a.name.replace("_", " ")))


def _blocked_reason(a: Action) -> str | None:
    if a.tier is Tier.BLOCKED:
        return "That action is not permitted"
    if a.name == "run_shell" and check_shell(str(a.args.get("cmd", ""))) is Tier.BLOCKED:
        return "That command is blocked for safety"
    if a.name in ("move_file", "delete_file"):
        if any(is_secret_path(str(a.args.get(k, ""))) for k in ("src", "dst", "path") if k in a.args):
            return "That path is protected"
    return None


async def execute(action: Action, confirm: Callable[[str], Awaitable[bool]]) -> ActionResult:
    reason = _blocked_reason(action)
    if reason:
        return ActionResult(action.name, False, reason)
    if action.tier is Tier.CONFIRM:
        rb = await _readback(action)
        if len(rb) > 300:  # never confirm a truncated description
            return ActionResult(action.name, False, "Too long to confirm safely")
        if not await confirm(rb):
            return ActionResult(action.name, False, "Cancelled")
    handler = HANDLERS.get(action.name)
    if handler is None:
        return ActionResult(action.name, False, "Not implemented")
    try:
        return await handler(action)
    except PermissionError as e:
        return ActionResult(action.name, False, str(e) or "Not permitted")
    except (KeyError, ValueError, TypeError, LookupError, OSError, RuntimeError) as e:
        return ActionResult(action.name, False, redact(f"Could not do that: {type(e).__name__}"))
