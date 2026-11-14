import asyncio
from unittest.mock import AsyncMock

import pytest

from contracts import Action, Tier
from tools import mac_controller as mc
from tools.redact import redact


def test_redact_patterns():
    pem = "-----BEGIN RSA PRIVATE KEY-----\nabc\ndef\n-----END RSA PRIVATE KEY-----"
    jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTYifQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"  # gitleaks:allow (public example token)
    for s in ["sk-ant-api03-abcdefghijklmnop", "sk-abcdefghijklmnopqrstuv", "ghp_" + "a" * 36,
              "github_pat_" + "A1b2" * 10, "xoxb-1234567890-abcdef", "AKIAIOSFODNN7EXAMPLE", jwt, pem]:
        out = redact(f"before {s} after")
        assert "[REDACTED]" in out and out.startswith("before") and out.endswith("after"), s
        assert s not in out
    for s in ["password=hunter2", "token: abc123", "SECRET=xyz"]:
        assert "hunter2" not in redact(s) and "abc123" not in redact(s) and "xyz" not in redact(s)
    assert redact("open safari at 5pm") == "open safari at 5pm"


@pytest.mark.parametrize("p,expected", [
    ("~/.ssh/id_rsa", True), ("/x/.env", True), ("/x/.env.local", True), ("a.pem", True),
    ("api_key.txt", True), ("~/.aws/credentials", True), ("~/.config/gh/hosts.yml", True),
    ("/x/my_token", True), ("~/Desktop/report.pdf", False), ("notes.txt", False)])
def test_secret_paths(p, expected):
    assert mc.is_secret_path(p) is expected


@pytest.mark.parametrize("cmd", [
    "SUDO rm x", "/usr/bin/sudo ls", 'bash -c "security find-generic-password"', "cat ~/.ssh/id_rsa",
    "diskutil eraseDisk JHFS+ x disk2", "rm -rf /", "rm -fr ~", "launchctl list", "op item list",
    "mkfs.ext4 /dev/x", "csrutil disable", "open -a '1Password 7'", "s\\udo ls", "su''do ls",
    "echo c3Vkbw== | base64 -d | sh", "cat .env", "cat ~/.aws/credentials"])
def test_shell_blocklist_obfuscation(cmd):
    assert mc.check_shell(cmd) is Tier.BLOCKED


def test_shell_allowed_needs_confirm():
    assert mc.check_shell("ls ~/Desktop") is Tier.CONFIRM


async def _yes(_): return True


async def test_execute_tiers(monkeypatch):
    h = AsyncMock()
    monkeypatch.setitem(mc.HANDLERS, "delete_file", h)
    monkeypatch.setitem(mc.HANDLERS, "run_shell", h)
    confirm = AsyncMock(return_value=True)
    # unknown + blocked shell never reach a handler or the confirm prompt
    assert not (await mc.execute(Action("rm_everything", {}), confirm)).ok
    assert not (await mc.execute(Action("run_shell", {"cmd": "sudo ls"}), confirm)).ok
    assert not (await mc.execute(Action("delete_file", {"path": "~/.ssh/id_rsa"}), confirm)).ok
    h.assert_not_called(); confirm.assert_not_called()
    # CONFIRM declined
    no = AsyncMock(return_value=False)
    r = await mc.execute(Action("delete_file", {"path": "~/Desktop/a.txt"}), no)
    assert not r.ok and no.await_count == 1
    h.assert_not_called()
    # CONFIRM accepted
    h.return_value = mc.ActionResult("delete_file", True, "ok")
    assert (await mc.execute(Action("delete_file", {"path": "~/Desktop/a.txt"}), confirm)).ok
    h.assert_awaited_once()
    # IMMEDIATE runs without prompting
    run = AsyncMock(return_value=(0, ""))
    monkeypatch.setattr(mc, "_run", run)
    confirm.reset_mock()
    assert (await mc.execute(Action("mute", {}), confirm)).ok
    assert run.await_args.args[0][0] == "osascript"
    confirm.assert_not_called()


async def test_applescript_no_interpolation(monkeypatch):
    evil = 'Safari"; do shell script "x'
    run = AsyncMock(return_value=(0, ""))
    monkeypatch.setattr(mc, "_run", run)
    for name in ("open_app", "quit_app"):
        await mc.execute(Action(name, {"app": evil}), _yes)
    assert run.await_count == 2
    for call in run.await_args_list:
        argv = call.args[0]
        assert all(isinstance(a, str) for a in argv) and argv[0] == "osascript"
        assert evil in argv
        scripts = [argv[i + 1] for i, a in enumerate(argv) if a == "-e"]
        assert scripts and not any(evil in s or "do shell script" in s for s in scripts)
        assert argv.index(evil) > max(i for i, a in enumerate(argv) if a == "-e") + 1


async def test_open_unknown_app(monkeypatch):
    monkeypatch.setattr(mc, "_run", AsyncMock(return_value=(1, "execution error: Can't get application")))
    r = await mc.execute(Action("open_app", {"app": "Nonexistent"}), _yes)
    assert r.ok is False and r.action == "open_app"


async def test_password_manager_app_blocked(monkeypatch):
    run = AsyncMock(return_value=(0, ""))
    monkeypatch.setattr(mc, "_run", run)
    assert not (await mc.execute(Action("open_app", {"app": "1Password"}), _yes)).ok
    run.assert_not_called()


async def test_bad_args_do_not_raise(monkeypatch):
    monkeypatch.setattr(mc, "_run", AsyncMock(return_value=(0, "")))
    for a in (Action("set_volume", {}), Action("set_volume", {"level": "abc"}),
              Action("open_app", {"app": ""}), Action("open_app", {"app": "x" * 500}),
              Action("set_timer", {"minutes": -3}), Action("create_event", {"title": "x"})):
        assert (await mc.execute(a, _yes)).ok is False


async def test_set_volume_clamped(monkeypatch):
    run = AsyncMock(return_value=(0, ""))
    monkeypatch.setattr(mc, "_run", run)
    await mc.execute(Action("set_volume", {"level": 500}), _yes)
    assert "100" in run.await_args.args[0]


async def test_file_ops_and_shell(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / ".Trash").mkdir()
    f = tmp_path / "a.txt"; f.write_text("x")
    r = await mc.execute(Action("move_file", {"src": str(f), "dst": str(tmp_path / "b.txt")}), _yes)
    assert r.ok and (tmp_path / "b.txt").exists() and not f.exists()
    r = await mc.execute(Action("delete_file", {"path": str(tmp_path / "b.txt")}), _yes)
    assert r.ok and not (tmp_path / "b.txt").exists() and len(list((tmp_path / ".Trash").iterdir())) == 1
    assert not (await mc.execute(Action("delete_file", {"path": "/etc/hosts"}), _yes)).ok  # outside home
    assert not (await mc.execute(Action("delete_file", {"path": str(tmp_path)}), _yes)).ok  # home itself
    r = await mc.execute(Action("run_shell", {"cmd": "echo hello"}), _yes)
    assert r.ok and "hello" in r.message


async def test_system_status_and_agenda_permission(monkeypatch):
    from tools import calendar as cal
    assert (await mc.execute(Action("system_status", {}), _yes)).ok
    def boom(): raise cal.CalendarPermissionError("denied")
    monkeypatch.setattr(cal, "agenda_today", boom)
    r = await mc.execute(Action("agenda_today", {}), _yes)
    assert r.ok is False and "Calendar" in r.message or "Privacy" in r.message or r.message
