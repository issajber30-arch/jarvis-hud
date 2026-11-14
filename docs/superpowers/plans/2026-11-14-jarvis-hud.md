# Jarvis HUD Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A terminal-native voice assistant ("hey Jarvis") that controls the owner's Mac, calendar and reminders, with a Textual HUD, published as a public GitHub repo.

**Architecture:** Main thread writes `config.py` + `contracts.py`; four builders (data, tools, audio, hud) then work in parallel on their own `feat/*` branch in separate git worktrees with zero file overlap; integration wires `main.py` on `dev`. Simple commands resolve locally via a model2vec cosine router; everything else goes to Claude with tool use, executed through a tier-enforcing controller.

**Tech Stack (versions verified by Codex, 2026-11-14):** Python 3.12 (uv), textual 8.2.8, numpy, sounddevice, openwakeword 0.6.0 + ai-edge-litert 2.1.6 (TFLite — the ONNX path is broken on macOS arm64), silero-vad 6.2.3, mlx-whisper 0.4.3, mlx-audio 0.6.5 + misaki (Kokoro `bm_george`, `lang_code="b"`), model2vec 0.10.0, anthropic, pyobjc-framework-EventKit 13.0, psutil, pyyaml, python-dotenv, pytest, pytest-asyncio.

**Spec:** `docs/superpowers/specs/2026-11-14-jarvis-hud-design.md`

## Global Constraints

- Project root `~/Projects/jarvis-hud/`; Python **3.12** in `.venv` via `uv` (system 3.14 is not used).
- Hardware target Apple M4, 16 GB, macOS 26; permissions granted to Terminal.app.
- Builders run on **sonnet**; zero file overlap between builders; only the main thread edits `config.py`, `contracts.py`, `main.py`, `requirements.txt`.
- Router: p95 < 15 ms including query embedding; ≥ 90 % held-out accuracy; score < 0.80 → Claude.
- Barge-in: audio stops < 20 ms after Space/barge-in event.
- Never read, log, speak or send secrets; `redact()` on everything bound for Claude, logs, HUD, TTS. No file-content reading in v1.
- Tiers enforced in code (`contracts.ACTIONS`), never by prompt.
- All subprocesses use argv lists with no shell; user/LLM text is never interpolated into an AppleScript source string (pass it as `on run argv` arguments).
- Lines and persona are original wording — no film quotes, no Marvel assets, no bundled reference voice.
- Repo public; `.gitignore` is the first commit; `gitleaks` scan passes before every push.
- Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. Silence/noise transcribed as a Whisper hallucination ("Thank you.", "you", "") → ignored, no action, back to IDLE. Test in Task 4 (`test_stt_filters_hallucinations`).
2. No `ANTHROPIC_API_KEY` or network down → spoken "offline" line, HUD ERROR then IDLE, no crash. Test in Task 3 (`test_brain_offline_yields_error_line`).
3. Calendar/Reminders permission denied → spoken instruction to grant access, no crash. Test in Task 3 (`test_agenda_permission_denied`).
4. Obfuscated dangerous shell (`SUDO`, `/usr/bin/sudo`, `bash -c "security find-generic-password"`, `cat ~/.ssh/id_rsa`) → BLOCKED. Test in Task 3 (`test_shell_blocklist_obfuscation`).
5. Unknown app name → failure line, no exception; long text truncated in HUD log. Tests in Task 3 (`test_open_unknown_app`) and Task 5 (`test_log_truncates`).

---

### Task 0: Repo bootstrap (main thread)

**Files:** Create `.gitignore`, `.env.example`, `requirements.txt`, `pytest.ini`, `tests/__init__.py`

- [ ] **Step 1:** `git init -b main`; write `.gitignore` (`.env`, `.env.*`, `!.env.example`, `.venv/`, `__pycache__/`, `*.pyc`, `.pytest_cache/`, `data/reference_voice/*.wav`, `data/reference_voice/*.txt`, `logs/`, `*.log`, `.DS_Store`, `.cache/`, `models/`); commit it alone: `chore: add gitignore`.
- [ ] **Step 2:** `uv venv --python 3.12 .venv`; install the stack; pin exact versions in `requirements.txt`. `.env.example` = `ANTHROPIC_API_KEY=`. `pytest.ini`: `asyncio_mode = auto`, `testpaths = tests`.
- [ ] **Step 3:** Verify: `.venv/bin/python -c "import textual, mlx_whisper, openwakeword, model2vec, anthropic, EventKit, sounddevice, silero_vad, mlx_audio"` exits 0; download `hey_jarvis` TFLite model via `openwakeword.utils.download_models(["hey_jarvis"])`.
- [ ] **Step 4:** Commit `chore: python 3.12 env and pinned deps` + spec/plan docs; create `dev` branch.

### Task 1: Shared config and contracts (main thread)

**Files:** Create `config.py`, `contracts.py`, `tests/test_contracts.py`

**Interfaces — Produces (every builder codes against these):**

```python
# contracts.py
class HUDState(str, Enum): IDLE, LISTENING, THINKING, SPEAKING, CONFIRM, FOLLOWUP, ERROR
TRANSITIONS: dict[HUDState, set[HUDState]]
#  IDLE→{LISTENING,ERROR}; LISTENING→{THINKING,IDLE,ERROR}; THINKING→{SPEAKING,CONFIRM,IDLE,ERROR};
#  SPEAKING→{IDLE,LISTENING,FOLLOWUP,CONFIRM,ERROR}; CONFIRM→{THINKING,SPEAKING,IDLE};
#  FOLLOWUP→{LISTENING,IDLE}; ERROR→{IDLE}
def can_transition(a: HUDState, b: HUDState) -> bool

class Tier(str, Enum): IMMEDIATE, CONFIRM, BLOCKED
ACTIONS: dict[str, Tier]   # names only; handlers live in tools/mac_controller.py
#  IMMEDIATE: open_app{app}, quit_app{app}, set_volume{level}, volume_up, volume_down, mute,
#    set_brightness{level}, music_play, music_pause, music_next, music_previous,
#    set_timer{minutes}, system_status, agenda_today, create_event{title,start,end},
#    create_reminder{title,due?}
#  CONFIRM: delete_event{id}, reschedule_event{id,start,end}, delete_reminder{id},
#    move_file{src,dst}, delete_file{path}, run_shell{cmd}, send_message{to,text}
FAST_ROUTE_ACTIONS: frozenset[str]  # IMMEDIATE minus create_event, create_reminder

@dataclass(frozen=True) class AudioTelemetryFrame: rms: float; fft: np.ndarray; vad: bool; ts: float   # fft: 64 bins, 0..1
@dataclass(frozen=True) class SystemTelemetryFrame: cpu_pct: float; ram_pct: float; battery_pct: float | None; charging: bool; net_up_kbps: float; net_down_kbps: float; git_branch: str | None; ts: float
@dataclass(frozen=True) class FastRouteMatch: action: str; args: dict; score: float; phrase: str; latency_ms: float
@dataclass(frozen=True) class Action: name: str; args: dict    # property tier -> ACTIONS.get(name, Tier.BLOCKED)
@dataclass(frozen=True) class ActionResult: action: str; ok: bool; message: str
@dataclass class ListenerEvent: kind: Literal["wake","ptt_start","ptt_end","utterance","barge_in"]; audio: np.ndarray | None = None
@dataclass class JarvisResponse: transcript: str; spoken: list[str]; actions: list[ActionResult]; via: Literal["fast","claude","none"]; followup: bool
```

`config.py` constants: `ROOT=Path(__file__).parent`, `SAMPLE_RATE=16000`, `BLOCK_SIZE=1280` (80 ms, openWakeWord frame; VAD consumes it as 512-sample chunks), `WAKE_MODEL="hey_jarvis"`, `WAKE_FRAMEWORK="tflite"`, `WAKE_THRESHOLD=0.5`, `WAKE_THRESHOLD_SPEAKING=0.8`, `VAD_THRESHOLD=0.5`, `SILENCE_END_MS=700`, `MAX_UTTERANCE_S=20`, `FOLLOWUP_WINDOW_S=8`, `CONFIRM_TIMEOUT_S=10`, `WHISPER_MODEL="mlx-community/whisper-small.en-mlx"`, `EMBED_MODEL="minishlab/potion-base-8M"`, `ROUTER_THRESHOLD=0.80`, `VECTORS_PATH=ROOT/"data/command_vectors.npz"`, `CLAUDE_MODEL="claude-haiku-5-5"`, `CLAUDE_MAX_TOKENS=1024`, `HISTORY_TURNS=20`, `TTS_VOICE="bm_george"`, `TTS_LANG="b"`, `TTS_SPEED=1.0`, `HUD_FPS=60`, `TELEMETRY_HZ=1`, `LOG_MAX_CHARS=200`, colours `STARK_CYAN="#4FD1FF"`, `STARK_DIM="#1B4F63"`, `STARK_AMBER="#FFB547"`, `STARK_RED="#FF4D4D"`, `STARK_BG="#05080C"`.

- [ ] **Step 1: Failing tests** — `test_unknown_action_is_blocked` (`Action("rm_everything", {}).tier is Tier.BLOCKED`); `test_fast_route_subset` (`FAST_ROUTE_ACTIONS` ⊆ IMMEDIATE names and `"create_event" not in FAST_ROUTE_ACTIONS`); `test_transitions` (`can_transition(IDLE, LISTENING)`, `not can_transition(IDLE, SPEAKING)`, every state has an entry).
- [ ] **Step 2:** `pytest tests/test_contracts.py` → FAIL (import error).
- [ ] **Step 3:** Implement both files.
- [ ] **Step 4:** PASS. Commit on `dev`: `feat: shared config and contracts`. Create `feat/data`, `feat/tools`, `feat/audio`, `feat/hud` from `dev` as worktrees in `~/Projects/jarvis-hud-wt/<name>`.

---

**Tasks 2–5 run in parallel, one sonnet builder each, in its own worktree. Each builder may only create/modify the files listed in its task plus its own `tests/<area>/` folder.**

### Task 2: Data — synthetic commands and vectors (`feat/data`)

**Files:** Create `data/__init__.py`, `data/generate_synthetic.py`, `data/command_vectors.npz`, `data/reference_voice/README.md`, `tests/data/test_synthetic.py`

**Interfaces — Produces:** `command_vectors.npz` keys: `vectors` (float32 N×D, L2-normalised, embedded with `config.EMBED_MODEL`), `actions` (N str), `args` (N JSON strings, slots kept as `{app}`/`{level}`/`{minutes}` templates), `phrases` (N str), `split` (N, "train"/"test", ~15 % test). Functions `generate_pairs() -> list[tuple[str, str, dict]]`, `build(path: Path) -> None`.

- [ ] **Step 1: Failing tests** — `test_pairs_cover_fast_routes` (every `FAST_ROUTE_ACTIONS` name has ≥ 8 phrases; total ≥ 120); `test_no_non_fast_actions`; `test_npz_shape_and_norm` (build to tmp; row norms 1 ± 1e-4; equal lengths; both splits present); `test_agenda_phrases` (contains "what's on the agenda today" plus ≥ 7 variants).
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement; `reference_voice/README.md` explains dropping `jarvis_ref.wav` + `jarvis_ref.txt` and that it must be the owner's own or a licensed voice.
- [ ] **Step 4:** PASS; `python -m data.generate_synthetic` writes the npz. Commit `feat(data): synthetic command set and vectors`.

### Task 3: Tools — router, controller, calendar, brain, safety, persona (`feat/tools`)

**Files:** Create `tools/__init__.py`, `tools/fast_router.py`, `tools/mac_controller.py`, `tools/calendar.py`, `tools/claude_brain.py`, `tools/redact.py`, `tools/lines.py`, `prompts/jarvis_system.md`, `prompts/lines.yaml`, `tests/tools/*`. Builder loads the `claude-api` skill before writing `claude_brain.py`.

**Interfaces — Produces:**
- `redact.redact(text: str) -> str` — replaces `sk-ant-…`, `sk-…`, `ghp_…`/`github_pat_…`, `xox…`, `AKIA…`, JWTs, `password=`/`token=`/`secret=` values and PEM blocks with `[REDACTED]`.
- `mac_controller.is_secret_path(path: str) -> bool`; `check_shell(cmd: str) -> Tier` (BLOCKED on a normalised, case-insensitive match for `sudo`, `security`, `keychain`, `diskutil erase`, `rm -rf /`, `mkfs`, `csrutil`, `launchctl`, `1password`, `op `, `bitwarden`, or any secret path; otherwise CONFIRM); `async execute(action: Action, confirm: Callable[[str], Awaitable[bool]]) -> ActionResult` — BLOCKED → refused; CONFIRM → `await confirm(readback)` first; IMMEDIATE → run.
- `calendar.agenda_today() -> list[dict]` (`{"kind": "event"|"reminder", "title", "start", "end", "id"}`; birthday/holiday calendars excluded; local midnight-to-midnight); `create_event(title, start: datetime, end: datetime) -> str`; `create_reminder(title, due: datetime | None) -> str`; `delete_event(id)`, `reschedule_event(id, start, end)`, `delete_reminder(id)`; raises `CalendarPermissionError` if denied. Access requested via `requestFullAccessToEventsWithCompletion_` / `requestFullAccessToRemindersWithCompletion_`.
- `fast_router.FastRouter(path=config.VECTORS_PATH)`; `.route(text: str) -> FastRouteMatch | None` (None below `ROUTER_THRESHOLD`); slot extraction by per-action regex (`app` = words after open/launch/quit/close; `level`/`minutes` = first integer).
- `lines.pick(situation: str) -> str` (no consecutive repeats per situation). `lines.yaml` situations: `wake, done, failed, confirm, cancelled, not_heard, greeting_morning, greeting_afternoon, greeting_evening, agenda_intro, agenda_empty, low_battery, high_cpu, followup_invite, offline, permission_calendar, goodbye` — ≥ 8 original lines each.
- `claude_brain.ClaudeBrain(model=config.CLAUDE_MODEL)`; `async respond(text: str, context: dict, execute: Callable[[Action], Awaitable[ActionResult]]) -> AsyncIterator[str]` — streams whole sentences; tools = every non-BLOCKED `ACTIONS` entry with JSON schemas; tool loop until `end_turn`; outbound text passes `redact`; keeps last `HISTORY_TURNS`; `.wants_followup` True after a turn with ≥ 2 executed actions; missing key / API error → yields `lines.pick("offline")`.
- `jarvis_system.md`: British butler persona, spoken style (no markdown, ≤ 3 sentences unless planning), live context block (time, agenda, battery), rules: never ask for or reveal secrets, use tools for actions, after planning summarise briefly.

- [ ] **Step 1: Failing tests:** `test_redact_patterns`; `test_secret_paths` (`~/.ssh/id_rsa`, `/x/.env`, `a.pem`, `api_key.txt` → True; `~/Desktop/report.pdf` → False); `test_shell_blocklist_obfuscation` (Review Focus 4 → BLOCKED; `ls ~/Desktop` → CONFIRM); `test_execute_tiers` (BLOCKED never calls handler; CONFIRM with confirm→False → `ok=False`, handler not called; IMMEDIATE calls handler with osascript mocked); `test_applescript_no_interpolation` (app name `Safari"; do shell script "x` reaches osascript only as an argv item); `test_open_unknown_app`; `test_router_accuracy_latency` (test split accuracy ≥ 0.90; p95 < 15 ms over 200 warm calls; "write me a poem about rain" → None); `test_router_slots` ("open safari" → `{"app": "safari"}`, "volume to 40" → `{"level": 40}`); `test_agenda_permission_denied`; `test_agenda_today_filters`; `test_lines_no_repeat`; `test_brain_offline_yields_error_line`; `test_brain_tool_loop` (mocked client: tool_use `create_reminder` then text → `execute` called once, sentences yielded, `wants_followup` False); `test_brain_redacts_outbound` (`sk-ant-…` in context never reaches the mocked request).
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement. Router tests use the real npz if present, else `data.generate_synthetic.build` into tmp (import only).
- [ ] **Step 4:** PASS. Commit `feat(tools): router, safe controller, calendar, brain, persona`.

### Task 4: Audio — listener, STT, speaker (`feat/audio`)

**Files:** Create `audio/__init__.py`, `audio/listener.py`, `audio/stt.py`, `audio/speaker.py`, `tests/audio/*`

**Interfaces — Produces:**
- `listener.fft_frame(block: np.ndarray, vad: bool) -> AudioTelemetryFrame` — 64 log-spaced bins normalised 0..1.
- `Listener(on_event: Callable[[ListenerEvent], None], on_audio: Callable[[AudioTelemetryFrame], None], stream_factory=None)`; `.start()`, `.stop()`, `.set_speaking(bool)` (wake threshold → `WAKE_THRESHOLD_SPEAKING`), `.open_followup(seconds: float)`, `.ptt_down()`, `.ptt_up()`. Wake/PTT starts capture; Silero VAD ends it after `SILENCE_END_MS` or `MAX_UTTERANCE_S`; emits `utterance` with float32 mono 16 kHz audio. Wake/PTT while speaking emits `barge_in` first. openWakeWord with `inference_framework=config.WAKE_FRAMEWORK`.
- `stt.transcribe(audio: np.ndarray) -> str` — sync (app runs it in a thread); `language="en"`; returns `""` for < 0.3 s audio or hallucinations (`"thank you."`, `"you"`, `"thanks for watching"`, punctuation-only).
- `Speaker(on_audio: Callable[[AudioTelemetryFrame], None], stream_factory=None)`; `async speak(sentences: AsyncIterator[str] | list[str]) -> bool` (False if interrupted); `.stop()` silences within 20 ms (flag checked per ≤ 10 ms output block + stream abort); `.is_speaking`. Kokoro via mlx-audio (`voice=TTS_VOICE`, `lang_code=TTS_LANG`) yields per-segment audio, so synthesise sentence N+1 while N plays. `clone_voice_available() -> bool` true only if `data/reference_voice/jarvis_ref.wav` exists.

- [ ] **Step 1: Failing tests:** `test_fft_frame_shape`; `test_stt_filters_hallucinations` (mocked mlx_whisper "Thank you." → ""; 0.1 s audio → "" without model call); `test_speaker_stop_latency` (fake device; `stop()` mid 3 s sentence → loop exits < 20 ms; `speak` returns False); `test_listener_ptt_utterance` (ptt_down, tone, ptt_up → one `utterance`); `test_listener_barge_in_while_speaking` (`set_speaking(True)`, ptt_down → `barge_in` before `ptt_start`).
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** PASS. Commit `feat(audio): wake/PTT listener, local STT, streaming speaker with barge-in`.

### Task 5: HUD — widgets and stylesheet (`feat/hud`)

**Files:** Create `hud/__init__.py`, `hud/core_widget.py`, `hud/telemetry_widget.py`, `hud/log_widget.py`, `hud/jarvis.tcss`, `tests/hud/*`

**Interfaces — Produces:**
- `CoreWidget(Widget)`: reactive `state: HUDState`; `.push_audio(frame: AudioTelemetryFrame)`; rotating 3D point sphere/torus projected with numpy rotation matrices onto a Braille (U+2800) canvas, updated via `set_interval(1 / HUD_FPS)`; inertial spin speeds up with rms; colour by state (cyan idle/listening, bright pulse speaking, amber CONFIRM, red ERROR). Pure helper `render_braille(points: np.ndarray, w: int, h: int) -> list[str]`.
- `TelemetryWidget(Widget)`: `collect() -> SystemTelemetryFrame` (psutil; battery None if absent; git branch via argv `git rev-parse --abbrev-ref HEAD`, None on failure); refresh at `TELEMETRY_HZ`.
- `LogWidget(Widget)`: `.log_line(text: str)` → `HH:MM:SS ▸ TEXT` uppercase, truncated to `LOG_MAX_CHARS`, last 200 kept; `.ticker(text: str)` bottom utterance ticker fading over 4 s.
- `jarvis.tcss`: palette from `config`; left rail 28 cols, centre core, right rail 40 cols, bottom ticker 1 row.

- [ ] **Step 1: Failing tests:** `test_render_braille_dims` (`h` strings of length `w`, only U+2800–U+28FF/space); `test_collect_telemetry`; `test_log_truncates` (500-char line → ≤ 200; 300 lines → 200 kept); `test_widgets_mount` (`App.run_test()` mounts all three, sets `state=CONFIRM`, renders without exception).
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** PASS. Commit `feat(hud): braille core, telemetry rail, log rail and ticker`.

---

### Task 6: Integration (main thread, `dev`)

**Files:** Create `main.py`, `tests/test_main.py`; Modify `requirements.txt` if a builder needed an extra pin.

**Interfaces — Produces:** `JarvisApp(App)` and `async JarvisApp.handle_utterance(text: str) -> JarvisResponse` (testable without audio).

Flow: empty text → `not_heard`, `via="none"`. `FastRouter.route` match → `mac_controller.execute`; `agenda_today` is read aloud (`agenda_intro` + items, or `agenda_empty`); `via="fast"`. Else THINKING → `ClaudeBrain.respond` piped into `Speaker.speak`; context = time, `agenda_today()` (empty on permission error), latest `SystemTelemetryFrame`. CONFIRM: speak readback; accept Enter or transcript in {"yes", "confirm", "do it"} within `CONFIRM_TIMEOUT_S`, else `cancelled`. If `brain.wants_followup`: speak `followup_invite`, FOLLOWUP, `listener.open_followup(FOLLOWUP_WINDOW_S)`. All HUD log/ticker text passes `redact`. Space → `ptt_down/up`, plus `speaker.stop()` if speaking. `q`/Ctrl+C → `goodbye`, clean shutdown. On launch speak `greeting_morning|afternoon|evening` by local hour (<12, <18, else). Telemetry heads-up once per episode: battery < 15 % and not charging → `low_battery`; CPU > 90 % for 30 s → `high_cpu` (only while IDLE).

- [ ] **Step 1:** Merge `feat/data`, `feat/tools`, `feat/audio`, `feat/hud` into `dev` with `--no-ff`; full `pytest` → all PASS.
- [ ] **Step 2: Failing tests:** `test_fast_path`; `test_agenda_spoken` (2 mocked events → both titles spoken); `test_claude_path_followup`; `test_confirm_rejected`; `test_empty_transcript` (no brain call); `test_app_boots_headless` (`run_test()` with audio mocked mounts and quits).
- [ ] **Step 3:** Implement `main.py`. Run → PASS.
- [ ] **Step 4: Manual E2E** (owner present, Terminal.app): grant mic/Calendars/Reminders/Automation; "hey Jarvis, open Safari"; "what's on the agenda today"; "remind me at 5 to call mom"; "plan my day" → events + follow-up invite; "delete report.pdf on my desktop" → amber confirm, say "no"; Space mid-sentence stops speech; "read my ssh key" → refused. Record results; capture demo GIF.
- [ ] **Step 5:** Commit `feat: integrate HUD, audio, router and brain`.

### Task 7: Review, README, GitHub (main thread)

**Files:** Create `README.md`, `LICENSE` (MIT), `docs/demo.gif`

- [ ] **Step 1:** Run `security-review` and `code-review` skills on the branch; Codex review of `redact.py`, `mac_controller.py`, `claude_brain.py` (pasted, no secrets); fix findings; full `pytest` green.
- [ ] **Step 2:** README: pitch + demo GIF, features, quickstart (uv, `.env`, Terminal.app permissions), how it works (Mermaid flow), safety model (tiers table + redaction), voice lines (`lines.yaml`), project layout, roadmap, credits/licences (Kokoro, Whisper, openWakeWord), disclaimer (fan project, not affiliated with Marvel).
- [ ] **Step 3:** `gitleaks detect --source . --no-banner` → no leaks; `git ls-files | grep -E '\.env$|\.wav$'` → empty.
- [ ] **Step 4:** `gh repo create jarvis-hud --public --source . --description "Terminal-native voice assistant for macOS — local speech, Claude brain, Stark-style HUD"`; push `main`, `dev`, `feat/*`; PRs `feat/* → dev` (merge), then `dev → main` (merge); add topics `voice-assistant macos textual mlx claude`.
- [ ] **Step 5:** Open the repo page in the browser; confirm README + GIF render; report URL.

### Task 8: LinkedIn & portfolio (after Task 7; approval-gated)

- [ ] **Step 1:** Load `linkedin-profile-editor`; draft Featured entry (title, hook, repo link, demo GIF) and launch post; show before/after previews.
- [ ] **Step 2:** Publish via Composio only after the owner's explicit yes.
