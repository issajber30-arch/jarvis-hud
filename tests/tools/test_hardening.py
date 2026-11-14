import os
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from contracts import Action, Tier
from tools import calendar as cal
from tools import claude_brain as cb
from tools import mac_controller as mc
from tools.redact import redact


async def _yes(_): return True


@pytest.mark.parametrize("s,secret", [
    ("DB_PASSWORD=hunter2", "hunter2"), ("ANTHROPIC_API_KEY=abc123def", "abc123def"),
    ('{"password": "x y"}', "x y"), ('password="a b c"', "b c"), ("Authorization: Bearer abcdefghij0123456789", "abcdefghij0123456789"),
    ("AIza" + "A" * 35, "A" * 35), ("sk_live_abcdef123456", "abcdef123456"), ("sk_test_abcdef123456", "abcdef123456")])
def test_redact_more(s, secret):
    out = redact(s)
    assert secret not in out and "[REDACTED]" in out


def test_redact_keeps_prose():
    s = "the password policy is strict"
    assert redact(s) == s


@pytest.mark.parametrize("cmd", [
    "echo $(whoami)", "echo `id`", "ls | wc", "ls; ls", "ls && ls", "ls || ls", "ls > x", "cat < x",
    "env", "printenv", "python -c 'print(1)'", "python3 -c x", "perl -e x", "ruby -e x", "node -e x",
    "bash -c ls", "sh -c ls", "zsh -c ls", "curl x.com", "wget x.com", "base64 f", "openssl rand 4",
    "eval ls", "xargs ls", "ls " + "a" * 200])
def test_shell_more_blocked(cmd):
    assert mc.check_shell(cmd) is Tier.BLOCKED


def test_shell_plain_ok():
    assert mc.check_shell("ls -la ~/Desktop") is Tier.CONFIRM


async def test_shell_env_scrubbed(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "leakme")
    monkeypatch.setenv("MY_SECRET", "leakme")
    monkeypatch.setenv("KEEP_ME", "visible")
    r = await mc.execute(Action("run_shell", {"cmd": "echo [$ANTHROPIC_API_KEY$MY_SECRET] $KEEP_ME; pwd".replace(";", "&&")}), _yes)
    assert r.ok is False  # && is blocked outright
    r = await mc.execute(Action("run_shell", {"cmd": "echo [$ANTHROPIC_API_KEY$MY_SECRET] $KEEP_ME"}), _yes)
    assert r.ok and "leakme" not in r.message and "visible" in r.message
    r = await mc.execute(Action("run_shell", {"cmd": "pwd"}), _yes)
    assert Path(r.message).resolve() == tmp_path.resolve()


def test_brain_reads_key_from_dotenv_not_environ(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(cb, "dotenv_values", lambda *a, **k: {"ANTHROPIC_API_KEY": "sk-from-dotenv"})
    seen = {}
    monkeypatch.setattr(cb.anthropic, "AsyncAnthropic", lambda **kw: seen.update(kw) or object())
    b = cb.ClaudeBrain()
    b._get_client()
    assert seen["api_key"] == "sk-from-dotenv" and "ANTHROPIC_API_KEY" not in os.environ


def test_brain_no_key_raises(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(cb, "dotenv_values", lambda *a, **k: {})
    with pytest.raises(cb.anthropic.AnthropicError):
        cb.ClaudeBrain()._get_client()


@pytest.mark.parametrize("name", ["/Applications/Safari.app", "Safari.app", "a:b", "x/y"])
@pytest.mark.parametrize("act", ["open_app", "quit_app"])
async def test_app_name_rejected(monkeypatch, name, act):
    run = AsyncMock(return_value=(0, ""))
    monkeypatch.setattr(mc, "_run", run)
    assert not (await mc.execute(Action(act, {"app": name}), _yes)).ok
    run.assert_not_called()


async def test_readback_too_long_refused():
    confirm = AsyncMock(return_value=True)
    h = AsyncMock()
    mc.HANDLERS["send_message"], old = h, mc.HANDLERS["send_message"]
    try:
        r = await mc.execute(Action("send_message", {"to": "bob", "text": "x" * 400}), confirm)
    finally:
        mc.HANDLERS["send_message"] = old
    assert not r.ok and "Too long to confirm safely" in r.message
    confirm.assert_not_called(); h.assert_not_called()


async def test_readback_includes_title(monkeypatch):
    monkeypatch.setattr(cal, "title_of", lambda i: "Dentist" if i == "e1" else None)
    assert "Dentist" in await mc._readback(Action("delete_event", {"id": "e1"}))
    assert "Dentist" in await mc._readback(Action("reschedule_event", {"id": "e1"}))
    assert "Dentist" in await mc._readback(Action("delete_reminder", {"id": "e1"}))
    assert "an unknown calendar item" in await mc._readback(Action("delete_event", {"id": "zz"}))


def test_title_of_found_and_missing(monkeypatch):
    from types import SimpleNamespace as NS
    ev = NS(title=lambda: "Standup")
    store = NS(eventWithIdentifier_=lambda i: ev if i == "e1" else None, calendarItemWithIdentifier_=lambda i: None)
    monkeypatch.setattr(cal, "_get_store", lambda: store)
    monkeypatch.setattr(cal, "_grant", lambda *a: None)
    assert cal.title_of("e1") == "Standup" and cal.title_of("nope") is None


@pytest.fixture
def home(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    for d in ("Library", "Documents", "Desktop", ".Trash"):
        (tmp_path / d).mkdir()
    (tmp_path / "Documents" / "a.txt").write_text("x")
    (tmp_path / "Library" / "l.txt").write_text("x")
    (tmp_path / ".zshrc").write_text("x")
    return tmp_path


@pytest.mark.parametrize("act,args", [
    ("delete_file", {"path": "{h}/Library/l.txt"}), ("delete_file", {"path": "{h}/.zshrc"}),
    ("delete_file", {"path": "{h}/Documents"}), ("delete_file", {"path": "{h}/Desktop"}),
    ("delete_file", {"path": "Documents/a.txt"}),
    ("move_file", {"src": "{h}/Documents/a.txt", "dst": "{h}/Library/a.txt"}),
    ("move_file", {"src": "{h}/Documents/a.txt", "dst": "{h}/.zprofile"}),
    ("move_file", {"src": "{h}/Library/l.txt", "dst": "{h}/Desktop/l.txt"}),
    ("move_file", {"src": "{h}/Documents", "dst": "{h}/Desktop/D"}),
    ("move_file", {"src": "Documents/a.txt", "dst": "{h}/Desktop/a.txt"}),
    ("move_file", {"src": "{h}/Documents/a.txt", "dst": "Desktop/a.txt"})])
async def test_path_refusals(home, act, args):
    args = {k: v.format(h=home) for k, v in args.items()}
    assert not (await mc.execute(Action(act, args), _yes)).ok
    assert (home / "Documents" / "a.txt").exists() and (home / "Library" / "l.txt").exists() and (home / ".zshrc").exists()


async def test_path_inside_top_level_ok(home):
    r = await mc.execute(Action("move_file", {"src": f"{home}/Documents/a.txt", "dst": "~/Desktop/a.txt"}), _yes)
    assert r.ok and (home / "Desktop" / "a.txt").exists()
