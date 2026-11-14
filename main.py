"""J.A.R.V.I.S. — terminal HUD entry point and turn orchestrator."""
from __future__ import annotations

import asyncio
import os
import re
from datetime import datetime

from textual.app import App, ComposeResult
from textual.binding import Binding

import config
from contracts import Action, ActionResult, HUDState, JarvisResponse, ListenerEvent
from hud.core_widget import CoreWidget
from hud.log_widget import LogWidget, TickerWidget
from hud.telemetry_widget import TelemetryWidget
from tools import lines
from tools.redact import redact

_YES = {"yes", "yeah", "yep", "yes please", "confirm", "do it", "go ahead", "affirmative"}


def is_yes(text: str) -> bool:
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split()) in _YES


class JarvisApp(App):
    CSS_PATH = "hud/jarvis.tcss"
    TITLE = "J.A.R.V.I.S."
    BINDINGS = [
        Binding("space", "talk", "Talk / interrupt", priority=True),
        Binding("enter", "confirm_yes", "Confirm", priority=True),
        Binding("q", "quit_jarvis", "Quit"),
    ]

    def __init__(self, *, router=None, brain=None, speaker=None, listener=None, controller=None,
                 agenda_fn=None, greet: bool = True):
        super().__init__()
        self.router, self.brain, self.speaker, self.listener = router, brain, speaker, listener
        self.controller, self.agenda_fn, self.greet = controller, agenda_fn, greet
        self.state = HUDState.IDLE
        self.pending_confirm: asyncio.Future | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._warned: set[str] = set()
        self._cpu_hot_s = 0

    def compose(self) -> ComposeResult:
        yield TelemetryWidget()
        yield CoreWidget()
        yield LogWidget()
        yield TickerWidget()

    # ---------- wiring ----------
    def on_mount(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._hud_log("BOOT SEQUENCE")
        if self.controller is None:
            from tools import mac_controller
            self.controller = mac_controller.execute
        if self.agenda_fn is None:
            from tools import calendar
            self.agenda_fn = calendar.agenda_today
        if self.brain is None:
            from tools.claude_brain import ClaudeBrain
            self.brain = ClaudeBrain()
        if self.speaker is None:
            from audio.speaker import Speaker
            self.speaker = Speaker(on_audio=self._from_thread(self._push_audio))
        if self.listener is None:
            from audio.listener import Listener
            self.listener = Listener(on_event=self._from_thread(self._on_listener_event),
                                     on_audio=self._from_thread(self._push_audio))
        if self.greet:
            self.run_worker(self._boot(), group="boot")
        elif self.router is None:
            from tools.fast_router import FastRouter
            self.router = FastRouter()
        self.set_interval(5, self._heads_up)

    async def _boot(self) -> None:
        # Heavy models load off the UI thread; greet once the voice is warm.
        if self.router is None:
            from tools.fast_router import FastRouter
            self.router = await asyncio.to_thread(FastRouter)
        self._hud_log("ROUTER ONLINE")
        if hasattr(self.speaker, "_synth"):
            await asyncio.to_thread(self.speaker._synth, "Ready.")
        self._hud_log("VOICE ONLINE")
        try:
            await asyncio.to_thread(self.listener.start)
            self._hud_log("MIC ONLINE · SAY 'HEY JARVIS' OR TAP SPACE")
        except Exception as e:  # no mic / permission denied: HUD still usable for logs
            self._hud_log(f"MIC OFFLINE {type(e).__name__}")
        h = datetime.now().hour
        await self._say([lines.pick("greeting_morning" if h < 12 else "greeting_afternoon" if h < 18
                                    else "greeting_evening")])

    def _from_thread(self, fn):
        # Audio callbacks fire on sounddevice threads; hop to the event loop without blocking them.
        def call(arg):
            if self._loop is not None and not self._loop.is_closed():
                self._loop.call_soon_threadsafe(fn, arg)
        return call

    def _push_audio(self, frame) -> None:
        self.query_one(CoreWidget).push_audio(frame)

    def _set_state(self, state: HUDState) -> None:
        self.state = state
        self.query_one(CoreWidget).state = state

    def _hud_log(self, text: str) -> None:
        self.query_one(LogWidget).log_line(redact(text))

    def _ticker(self, text: str) -> None:
        self.query_one(LogWidget).ticker(redact(text))

    # ---------- listener events ----------
    def _on_listener_event(self, ev: ListenerEvent) -> None:
        if ev.kind == "barge_in":
            self.speaker.stop()
            self._hud_log("BARGE-IN")
        elif ev.kind in ("wake", "ptt_start"):
            self._set_state(HUDState.LISTENING)
            self._hud_log("WAKE DETECTED" if ev.kind == "wake" else "PTT OPEN")
        elif ev.kind == "utterance" and ev.audio is not None:
            self.run_worker(self._on_utterance_audio(ev.audio), group="stt")

    async def _on_utterance_audio(self, audio) -> None:
        from audio import stt
        text = await asyncio.to_thread(stt.transcribe, audio)
        if self.pending_confirm is not None:
            self.answer_confirm(text)
            return
        self.run_worker(self.handle_utterance(text), group="turn", exclusive=True)

    # ---------- confirmation ----------
    def answer_confirm(self, text: str) -> None:
        fut = self.pending_confirm
        if fut is not None and not fut.done():
            self._ticker(text)
            fut.set_result(is_yes(text))

    async def _confirm(self, readback: str) -> bool:
        prev = self.state
        self._set_state(HUDState.CONFIRM)
        self._hud_log(f"CONFIRM? {readback}")
        await self.speaker.speak([readback, lines.pick("confirm")])
        self.pending_confirm = asyncio.get_running_loop().create_future()
        self.listener.open_followup(config.CONFIRM_TIMEOUT_S)  # reply without the wake word
        try:
            ok = await asyncio.wait_for(self.pending_confirm, config.CONFIRM_TIMEOUT_S)
        except TimeoutError:
            ok = False
        finally:
            self.pending_confirm = None
        self._hud_log("CONFIRMED" if ok else "CANCELLED")
        if not ok:
            await self.speaker.speak([lines.pick("cancelled")])
        self._set_state(prev if prev is not HUDState.CONFIRM else HUDState.THINKING)
        return ok

    # ---------- a turn ----------
    async def _say(self, sentences) -> bool:
        self._set_state(HUDState.SPEAKING)
        self.listener.set_speaking(True)
        try:
            return await self.speaker.speak(sentences)
        finally:
            self.listener.set_speaking(False)

    async def _execute(self, action: Action, results: list[ActionResult]) -> ActionResult:
        self._hud_log(f"EXEC {action.name} {' '.join(str(v) for v in action.args.values())}")
        r = await self.controller(action, self._confirm)
        self._hud_log(f"{'OK' if r.ok else 'FAIL'} {action.name} {r.message}")
        results.append(r)
        return r

    def _brain_context(self, agenda: list) -> dict:
        frame = self.query_one(TelemetryWidget).frame
        return {"now": datetime.now().strftime("%A %d %B %Y, %H:%M"), "agenda": agenda,
                "battery_pct": frame.battery_pct if frame else None,
                "charging": frame.charging if frame else None}

    async def handle_utterance(self, text: str) -> JarvisResponse:
        text = text.strip()
        results: list[ActionResult] = []
        if not text:
            line = lines.pick("not_heard")
            await self._say([line])
            self._set_state(HUDState.IDLE)
            return JarvisResponse(text, [line], [], "none", False)

        self._ticker(text)
        match = self.router.route(text)
        if match:
            self._hud_log(f"ROUTE {match.action} {match.score:.2f} {match.latency_ms:.1f}MS")
            r = await self._execute(Action(match.action, match.args), results)
            line = r.message if r.message not in ("Done", "") else lines.pick("done" if r.ok else "failed")
            await self._say([line])
            self._set_state(HUDState.IDLE)
            return JarvisResponse(text, [line], results, "fast", False)

        self._set_state(HUDState.THINKING)
        self._hud_log("ROUTE CLAUDE")
        try:
            agenda = await asyncio.to_thread(self.agenda_fn)
        except Exception:  # permission denied or EventKit unavailable: plan without it
            agenda = []
        spoken: list[str] = []
        # shortcut: sentences are spoken one at a time so a confirm can't overlap speech (~0.2 s gap each);
        # upgrade to a shared sentence queue if gaps become noticeable.
        async for sentence in self.brain.respond(text, self._brain_context(agenda),
                                                 lambda a: self._execute(a, results)):
            spoken.append(sentence)
            self._hud_log(f"SAY {sentence}")
            if not await self._say([sentence]):
                break  # interrupted
        followup = bool(self.brain.wants_followup)
        if followup:
            invite = lines.pick("followup_invite")
            spoken.append(invite)
            await self._say([invite])
            self._set_state(HUDState.FOLLOWUP)
            self.listener.open_followup(config.FOLLOWUP_WINDOW_S)
        else:
            self._set_state(HUDState.IDLE)
        return JarvisResponse(text, spoken, results, "claude", followup)

    # ---------- heads-up lines ----------
    async def _heads_up(self) -> None:
        frame = self.query_one(TelemetryWidget).frame
        if frame is None or self.state is not HUDState.IDLE:
            return
        if frame.battery_pct is not None and frame.battery_pct < 15 and not frame.charging:
            if "battery" not in self._warned:
                self._warned.add("battery")
                await self._say([lines.pick("low_battery")])
                self._set_state(HUDState.IDLE)
        else:
            self._warned.discard("battery")
        self._cpu_hot_s = self._cpu_hot_s + 5 if frame.cpu_pct > 90 else 0
        if self._cpu_hot_s >= 30 and "cpu" not in self._warned:
            self._warned.add("cpu")
            await self._say([lines.pick("high_cpu")])
            self._set_state(HUDState.IDLE)
        elif self._cpu_hot_s == 0:
            self._warned.discard("cpu")

    # ---------- keys ----------
    def action_talk(self) -> None:
        if self.pending_confirm is not None:
            self.answer_confirm("no")  # Space cancels a pending confirmation
        elif self.speaker.is_speaking:
            self.speaker.stop()
            self.listener.ptt_down()
        elif self.state is HUDState.LISTENING:
            self.listener.ptt_up()
        else:
            self.listener.ptt_down()

    def action_confirm_yes(self) -> None:
        self.answer_confirm("yes")

    async def action_quit_jarvis(self) -> None:
        self.speaker.stop()
        try:
            await asyncio.wait_for(self.speaker.speak([lines.pick("goodbye")]), 6)
        except TimeoutError:
            pass
        self.listener.stop()
        self.exit()


def prepare_runtime() -> None:
    """Call before the TUI takes over the terminal."""
    # Model downloads draw tqdm bars whose multiprocessing lock spawns a helper that needs the real stderr;
    # inside Textual stderr has no fd and the spawn crashes. Silence bars and start the helper up front.
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TQDM_DISABLE"] = "1"
    from multiprocessing import resource_tracker
    resource_tracker.ensure_running()


if __name__ == "__main__":
    prepare_runtime()
    JarvisApp().run()
