# Jarvis HUD — Design

Date: 2026-11-14 · Status: draft for approval

## Intent

A terminal-native voice assistant for the owner's own Mac. Say "hey Jarvis" (or hold Space), speak, and it acts: controls the Mac, manages calendar and reminders, plans the day, and answers questions in a British voice that can be interrupted. A full-screen Textual HUD (3D Braille core, live system vitals, action log) shows what it is doing.

Why: daily-use assistant for the owner, and a public portfolio piece (GitHub + LinkedIn Featured) — the project before the owner's last big project. It must be clean enough to read and impressive enough to demo in a screen recording.

## Goals

- Wake word "hey Jarvis" + Space push-to-talk, fully local speech-to-text.
- Simple commands resolve locally in <15 ms via a cosine-similarity router; everything else goes to Claude.
- Spoken replies stream sentence-by-sentence; user can barge in (<20 ms to stop audio).
- macOS control, calendar, reminders, small tasks, "what's on the agenda today", day planning.
- Public GitHub repo with a readable README and a clean branch workflow.

## Non-goals

- No web/mobile UI, no multi-user, no cloud hosting.
- No always-on background daemon at login (run from terminal; may come later).
- No Google Calendar API — calendar access is through macOS only.

## Anti-goals (failure even if it works)

- Jarvis ever reads, prints, logs or sends a password, key or token.
- A misheard word deletes data without a confirmation.
- Shipping Marvel assets or a cloned voice of a real actor in the public repo.

## Constraints (verified on the machine)

- Apple M4, 16 GB RAM, macOS 26.6 (the original brief said M5; tune for M4).
- System Python is 3.14 — too new for mlx-whisper / openWakeWord / Silero wheels. Project pins **Python 3.12 via `uv`** in `.venv`.
- `gh` is logged in as `issajber30-arch`.
- Claude access via `ANTHROPIC_API_KEY` in `.env` (git-ignored; `.env.example` committed).

## Approaches considered (calendar/reminders access)

| Option | Trade-off | Decision |
|---|---|---|
| **EventKit via pyobjc** | Native, instant reads, covers every account synced into Calendar/Reminders; one macOS permission prompt | **Chosen by owner** |
| AppleScript to Calendar.app | Simplest, but 2–5 s per query and can pop the app open | Rejected: too slow for voice |
| Google Calendar via Composio | Works anywhere, but Google-only, needs internet, sends schedule through a third party | Rejected: against local-first |

The rest of the stack comes from the owner's brief (mlx-whisper, openWakeWord, Silero VAD, Kokoro `bm_george`, Textual, AsyncAnthropic).

## Architecture

Directory layout is exactly as in the owner's brief (`~/Projects/jarvis-hud/`: `main.py`, `config.py`, `contracts.py`, `prompts/`, `data/`, `audio/`, `hud/`, `tools/`), plus `tools/calendar.py` (EventKit) and `tests/`.

Flow: `listener` (wake/PTT + VAD) → `stt` (mlx-whisper) → `fast_router` → either a local action in `mac_controller` / `calendar`, or `claude_brain` → `JarvisResponse` (spoken text + action list) → `mac_controller` executes → `speaker` streams TTS. `main.py` owns the `HUDState` machine (IDLE, LISTENING, THINKING, SPEAKING, CONFIRM, ERROR) and pushes telemetry frames to the HUD widgets.

Build order: main thread writes `config.py` + `contracts.py` first; then 4 parallel builders with zero file overlap:

1. **audio** — `audio/*`
2. **hud** — `hud/*`
3. **tools** — `tools/*`, `prompts/jarvis_system.md`
4. **data** — `data/*` (synthetic pairs + `command_vectors.npz`)

Then integration: `main.py`, `requirements.txt`, `.venv`, vectors, headless tests.

## Permissions model (owner's decision)

- **Runs immediately:** open/close apps, volume, brightness, music, timers, system stats, create calendar events, create reminders/tasks, read today's agenda, plan the day.
- **Amber confirm (say "yes"/press Enter):** delete or move files, delete/reschedule events, delete reminders, free-form shell commands, sending messages.
- **Blocked always:** `sudo`, Keychain/`security` CLI, password-manager apps, disk erase, reading `.env*`, `~/.ssh`, `~/.aws`, `~/.config/gh`, `*.pem`, `*key*`, `*token*`. Every string sent to Claude or written to logs passes a secret-redaction filter (API-key/token/password patterns). The Anthropic key is never put in a prompt.
- New abilities = one new entry in the action registry + a few synthetic training phrases.

## Agenda feature

"What's on the agenda today" (and variants) is a fast-route intent: EventKit reads today's events + reminders due today, Jarvis speaks them directly with no cloud call. "Plan my day" goes to Claude with that same agenda injected as context. Owner's decision: Jarvis creates all the planned events/reminders immediately (no confirm), reads a short summary, then ends with "Tell me if I missed anything, or if there's something you'd like to add." The mic then stays open for ~8 s follow-up without the wake word, so additions flow straight in.

## Personality lines (owner's request: "many more lines, not just plan my day")

Jarvis speaks varied, in-character lines across every situation, never the same one twice in a row. A `prompts/lines.yaml` bank (≥ 8 lines per situation, original wording — no film quotes) covers: wake acknowledgement ("At your service.", "Listening, sir."), done, action failed, confirm read-back, cancelled, didn't catch that, time-of-day greetings, agenda intro/empty day, low battery/high CPU heads-up, goodbye. Fast-route actions pick from the bank locally (no cloud cost); Claude gets the same persona in `jarvis_system.md` for open replies.

Follow-up invite applies to **any multi-step task** (planning, several reminders, a batch of actions), not only "plan my day": Jarvis closes with a varied invite ("Anything I missed?", "Tell me if there's something you'd like to add.") and keeps the mic open ~8 s for a reply without the wake word. Adding new lines = editing the YAML, no code.

## Voice

Default voice Kokoro `bm_george`. The F5-TTS clone hook loads `data/reference_voice/jarvis_ref.wav` only if the owner supplies it; README states it must be the owner's voice or a licensed voice. No reference clip ships in the repo.

## GitHub & docs

- Repo `jarvis-hud`, **public** (owner's decision). `.gitignore` committed first, before any other file: `.env*` (except `.env.example`), `.venv/`, `__pycache__/`, model caches, `data/reference_voice/*.wav`, logs, `.DS_Store`. A secret scan runs before every push.
- Branches: `main` (stable, protected-by-convention), `dev` (integration), `feat/audio`, `feat/hud`, `feat/tools`, `feat/data` — one per builder, merged into `dev` by PR, then `dev` → `main` by PR once tests pass.
- README: one-line pitch, demo GIF, features, quickstart (uv + permissions), how it works diagram, safety model, project layout, roadmap, credits/licenses. Plain language.
- LinkedIn/portfolio (owner wants it polished): once a working demo exists — a demo GIF/short video of the HUD, a Featured project entry with a strong one-line hook, and a launch post. Drafts shown as before/after previews; nothing posts without the owner's explicit approval.

## Testing

Headless: router latency (<15 ms p95) and accuracy on held-out phrases; contract round-trips; secret-redaction and blocklist unit tests; permission tiers (immediate vs confirm vs blocked); EventKit read with mocks; HUD renders in Textual `run_test` without crashing. Manual E2E: wake word → command → action → spoken reply → barge-in.

## Calls made after the builder check (owner to review)

- **Brief = the owner's pasted prompt** (directory tree + execution protocol). The main thread defines all contracts in `contracts.py` before builders start: `HUDState` enum + allowed transitions, `AudioTelemetryFrame`, `SystemTelemetryFrame`, `FastRouteMatch`, `Action` (name, args, tier), `JarvisResponse` (spoken text, actions, state).
- **Tiers live in code, not the prompt.** Each action in the registry carries a fixed tier (immediate / confirm / blocked); `mac_controller` enforces it. Free-form shell is always *confirm*, and its command string is pattern-checked against the blocklist (sudo, security, rm -rf /, diskutil erase, cat of secret paths…) → *blocked*.
- **Confirm flow:** Jarvis reads the action back ("Delete report.pdf — confirm?"); accepted by saying "yes"/"confirm" or pressing Enter; anything else, Space, or 10 s silence cancels.
- **Claude uses tool use**, not hand-rolled JSON: spoken text streams as normal text (sentence-chunked to TTS), actions arrive as tool calls validated against the registry. `JarvisResponse` is the internal record of a turn. History: last 20 turns in memory, not saved to disk.
- **Router embeddings:** `model2vec` static embeddings (sub-millisecond on CPU). The 15 ms budget includes embedding the query. Below a 0.80 cosine score the request falls through to Claude. Pass = p95 < 15 ms and ≥ 90 % accuracy on held-out phrases.
- **Barge-in / echo:** Space always interrupts. While speaking, the wake word is still active but needs a higher threshold to avoid Jarvis triggering itself; README recommends headphones for voice barge-in.
- **No file reading in v1.** Jarvis can open/move/delete files (with confirm) but never reads contents, which keeps secrets out of reach. Redaction applies to Claude prompts, logs, the HUD log and spoken output.
- **Permissions target Terminal.app** (the owner's terminal): Microphone, Calendars + Reminders (full access), Automation (System Events, Music). README walks through granting them.
- **Calendar defaults:** new events go to the system default calendar, reminders to the default list; "today" = local timezone midnight-to-midnight; birthday/holiday calendars excluded from the agenda.
- **TTS:** Kokoro via `mlx-audio`; fallback `kokoro-onnx` if streaming is too slow.

## Left to the builder

Exact Whisper model size (fit M4/16 GB latency), Claude model ID (low-latency default, configurable in `config.py`), FFT/visual details, synthetic phrase wording, log formatting.
