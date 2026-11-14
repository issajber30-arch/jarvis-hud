"""Synthetic fast-route command set -> embedded vectors (run: python -m data.generate_synthetic)."""
from __future__ import annotations

import json
import re
import random
from pathlib import Path

import numpy as np

import config
from contracts import FAST_ROUTE_ACTIONS

APPS = ["safari", "spotify", "notes", "terminal", "slack", "calendar", "messages", "mail", "finder", "music"]
LEVELS = [10, 25, 30, 40, 50, 60, 75, 80, 100]
MINUTES = [1, 2, 5, 10, 15, 20, 30, 45]

# action -> (arg slot or None, templates). Slots are filled from the lists above.
TEMPLATES: dict[str, tuple[str | None, list[str]]] = {
    "open_app": ("app", [
        "open {app}", "launch {app}", "start {app}", "hey jarvis open {app}", "can you open {app}",
        "please open {app} for me", "fire up {app}", "bring up {app}", "open up {app}",
        "jarvis launch {app}", "i need {app} open", "could you launch {app}", "open the {app} app",
        "launch {app} for me", "start up {app}", "fire up the {app} app", "bring up the {app} app", "get {app} going", "open {app} up", "start {app} please", "pull up the {app} app", "open {app} now", "go ahead and open {app}", "get {app} open", "run {app}", "switch on {app}", "pull up {app}", "open {app} please",
    ]),
    "quit_app": ("app", [
        "quit {app}", "close {app}", "hey jarvis quit {app}", "shut down {app}", "exit {app}",
        "please close {app}", "can you quit {app}", "kill {app}", "close the {app} app",
        "jarvis close {app}", "i'm done with {app} so quit it", "could you shut {app} down",
        "please quit {app}", "quit {app} now", "kill the {app} app", "force quit {app}", "close out of {app}", "end {app}", "exit the {app} app", "shut down the {app} app", "terminate the {app} app", "close {app} please", "shut {app} down", "terminate {app}", "get rid of {app}", "close down {app}", "quit the {app} app",
    ]),
    "set_volume": ("level", [
        "set the volume to {level}", "volume to {level}", "hey jarvis set volume to {level} percent",
        "put the volume at {level}", "change the volume to {level}", "make the volume {level}",
        "set volume {level}", "turn the volume to {level} percent", "can you set the volume to {level}",
        "jarvis volume {level}", "i want the volume at {level}", "adjust the volume to {level}",
    ]),
    "volume_up": (None, [
        "turn it up", "volume up", "louder", "hey jarvis turn the volume up", "make it louder",
        "raise the volume", "increase the volume", "a bit louder please", "crank it up",
        "jarvis volume up a bit", "can you turn the sound up", "bump the volume up",
        "raise the sound", "turn the sound up", "can you turn the volume up", "make it a little louder", "volume higher", "up a bit", "turn up the volume", "turn it up a bit", "louder please", "more volume", "i can't hear it turn it up", "turn the volume up a little", "make the sound louder",
    ]),
    "volume_down": (None, [
        "turn it down", "volume down", "quieter", "hey jarvis turn the volume down", "make it quieter",
        "lower the volume", "decrease the volume", "a bit quieter please", "too loud turn it down",
        "jarvis volume down a bit", "can you turn the sound down", "reduce the volume",
        "lower the sound", "turn the sound down", "can you turn the volume down", "make it a little quieter", "volume lower", "down a bit", "turn down the volume", "turn it down a bit", "quieter please", "less volume", "softer", "not so loud", "turn the volume down a little", "it's too loud",
    ]),
    "mute": (None, [
        "mute", "mute the sound", "hey jarvis mute", "silence the audio", "mute the volume",
        "turn the sound off", "shut the sound off", "mute it", "jarvis mute please", "mute please", "mute the speakers", "shut the sound off", "turn off the sound", "no sound", "mute the audio", "silence",
        "can you mute the computer", "kill the sound", "turn off the audio",
    ]),
    "set_brightness": ("level", [
        "set the brightness to {level}", "brightness to {level}", "hey jarvis set brightness to {level} percent",
        "put the screen brightness at {level}", "change the brightness to {level}", "make the screen {level} percent bright",
        "set brightness {level}", "turn the brightness to {level} percent", "can you set the brightness to {level}",
        "jarvis brightness {level}", "adjust the screen brightness to {level}", "dim the screen to {level} percent",
    ]),
    "music_play": (None, [
        "play music", "play", "resume the music", "hey jarvis play some music", "start the music",
        "continue playing", "play the song", "unpause the music", "jarvis play music",
        "can you play some music", "put some music on", "start playing music",
        "play some tunes", "play the next song please", "start some music", "play me something", "play the music", "play a song", "play something", "hit play", "resume playing", "play my music",
    ]),
    "music_pause": (None, [
        "pause the music", "pause", "stop the music", "hey jarvis pause the music", "pause the song",
        "stop playing", "hold the music", "jarvis pause", "can you pause the music",
        "stop the song", "pause playback", "turn the music off",
        "hold on pause it", "pause the audio", "pause it", "can you stop the music", "pause the track", "stop playing music", "stop the audio",
    ]),
    "music_next": (None, [
        "next song", "skip", "next track", "hey jarvis skip this song", "skip this track",
        "play the next song", "skip to the next one", "jarvis next track", "can you skip this",
        "go to the next song", "i don't like this song skip it", "next one please",
        "next", "skip to the next song", "next track please", "move on to the next song", "skip ahead to the next track", "play the next track", "skip song",
    ]),
    "music_previous": (None, [
        "previous song", "go back", "previous track", "hey jarvis play the previous song", "last song",
        "play the last track", "go back a song", "jarvis previous track", "can you go to the previous song",
        "back to the last song", "play the song before this one", "rewind to the previous track",
        "go back one song", "previous", "go to the previous track", "play the last song", "back one track", "play the previous track", "last track",
    ]),
    "set_timer": ("minutes", [
        "set a timer for {minutes} minutes", "timer for {minutes} minutes", "hey jarvis set a {minutes} minute timer",
        "start a {minutes} minute timer", "remind me in {minutes} minutes with a timer", "set the timer to {minutes} minutes",
        "can you set a timer for {minutes} minutes", "jarvis timer {minutes} minutes", "start a timer for {minutes} minutes",
        "i need a {minutes} minute timer", "count down {minutes} minutes", "give me a {minutes} minute timer",
    ]),
    "system_status": (None, [
        "system status", "how's the computer doing", "hey jarvis system status", "how is my mac doing",
        "check the system", "check system status", "system check", "what's the cpu usage", "how much battery do i have", "give me a status report",
        "jarvis how's the system", "how's the battery", "how much memory am i using", "run a system check",
        "jarvis system status", "how's my system", "how are the system stats", "check the cpu and memory", "how is the computer running", "what's the system status", "give me the system status", "how is the system", "status report", "how hard is the cpu working",
    ]),
    "agenda_today": (None, [
        "what's on the agenda today", "what's on my calendar today", "what do i have today", "hey jarvis what's my agenda",
        "read me today's schedule", "what's on my schedule", "do i have any meetings today", "jarvis what's on today",
        "what's my day look like", "tell me my agenda for today", "what's on my plate today", "any events today",
        "read me today's schedule", "what's my schedule for today", "read my calendar", "do i have any meetings today", "any meetings today", "what meetings do i have", "what's on the calendar today", "what do i have on today", "what's scheduled for today", "do i have anything on today", "what's coming up", "what's happening today", "what's on my agenda", "what's coming up today", "hey jarvis what's on the agenda today",
    ]),
}


def _grouped() -> list[tuple[str, str, dict, str]]:
    """(phrase, action, args-template, group). Variants of one base phrase share a group (split unit)."""
    rng = random.Random(0)
    fills = {"app": APPS, "level": LEVELS, "minutes": MINUTES}
    out, seen = [], set()
    for action, (slot, tpls) in TEMPLATES.items():
        for t in tpls:
            vals = rng.sample(fills[slot], len(fills[slot]) if slot == "app" else 3) if slot else [None]
            for v in vals:
                base = t.format(**{slot: v}) if slot else t
                plain = re.sub(r"^(hey )?jarvis ", "", base)
                for ph in (base, f"hey jarvis {plain}", f"jarvis {plain}", f"{plain} please"):
                    if ph not in seen:
                        seen.add(ph)
                        out.append((ph, action, {slot: "{%s}" % slot} if slot else {}, f"{action}|{plain}"))
    return out


def generate_pairs() -> list[tuple[str, str, dict]]:
    """(phrase, action, args-template). Phrases carry concrete slot values; args keep {slot} templates."""
    return [(p, a, g) for p, a, g, _ in _grouped()]


def build(path: Path) -> None:
    from model2vec import StaticModel

    rows = _grouped()
    pairs = [(p, a, g) for p, a, g, _ in rows]
    phrases = [p for p, _, _ in pairs]
    rng = random.Random(0)
    split = ["train"] * len(pairs)
    for action in sorted(FAST_ROUTE_ACTIONS):  # stratified ~15 % test per action, split by base phrase
        groups = sorted({k for _, a, _, k in rows if a == action})
        test = set(rng.sample(groups, max(2, round(0.15 * len(groups)))))
        for i, r in enumerate(rows):
            if r[3] in test:
                split[i] = "test"
    vecs = StaticModel.from_pretrained(config.EMBED_MODEL).encode(phrases).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path, vectors=vecs, actions=np.array([a for _, a, _ in pairs]),
        args=np.array([json.dumps(g) for _, _, g in pairs]), phrases=np.array(phrases), split=np.array(split),
    )


if __name__ == "__main__":
    build(config.VECTORS_PATH)
