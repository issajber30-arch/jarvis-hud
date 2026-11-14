"""Fixture: a small command-vectors npz in the Task 2 format (the real data/ package lives on another branch)."""
import json

import numpy as np
import pytest

import config

_PHRASES = {
    "open_app": (["open Safari", "launch Spotify", "start up Slack", "open the Notes app", "please open Calendar",
                  "launch Terminal", "open Messages", "bring up Finder"], ["open Mail", "launch Photos"]),
    "quit_app": (["quit Safari", "close Spotify", "quit the Notes app", "exit Slack", "please quit Calendar",
                  "close Terminal", "shut down Messages", "quit Finder"], ["close Mail", "quit Photos"]),
    "set_volume": (["set the volume to 50", "volume to 30 percent", "turn the volume to 80", "set volume to 20",
                    "make the volume 70", "change the volume to 10", "volume level 60", "put the volume at 40"],
                   ["set volume to 90 percent", "set the volume to 25"]),
    "volume_up": (["turn it up", "volume up", "louder", "increase the volume", "raise the volume",
                   "turn the sound up", "make it louder", "a bit louder please"],
                  ["crank up the volume", "turn up the sound"]),
    "volume_down": (["turn it down", "volume down", "quieter", "decrease the volume", "lower the volume",
                     "turn the sound down", "make it quieter", "a bit quieter please"],
                    ["reduce the volume", "turn down the sound"]),
    "mute": (["mute", "mute the sound", "mute the volume", "silence the computer", "silence everything",
              "mute my mac", "kill the sound", "toggle mute"], ["please mute", "mute everything"]),
    "set_brightness": (["set brightness to 50", "brightness to 30 percent", "dim the screen to 20",
                        "set the screen brightness to 80", "make the screen brightness 70",
                        "change brightness to 40", "brightness level 60", "put the brightness at 90"],
                       ["set brightness to 10 percent", "set the screen brightness to 55"]),
    "music_play": (["play music", "play some music", "start the music", "resume the music", "press play",
                    "play my music", "start playing music", "play the song"],
                   ["play the music please", "resume playback"]),
    "music_pause": (["pause the music", "pause", "stop the music", "pause playback", "stop playing",
                     "pause my music", "hold the music", "pause the song"],
                    ["please pause the music", "stop the song"]),
    "music_next": (["next song", "skip this song", "next track", "skip the track", "play the next song",
                    "skip to the next song", "skip", "go to the next track"],
                   ["skip ahead to the next song", "play the next track"]),
    "music_previous": (["previous song", "go back a song", "previous track", "play the previous song",
                        "last song", "go to the previous track", "back to the last track",
                        "play the song before"], ["play the previous track", "go back to the last song"]),
    "set_timer": (["set a timer for 5 minutes", "timer for 10 minutes", "start a 15 minute timer",
                   "set a timer for 20 minutes", "timer for 3 minutes", "set a 25 minute timer",
                   "countdown 30 minutes", "start a timer for 45 minutes"],
                  ["set a timer for 8 minutes", "start a 12 minute timer"]),
    "system_status": (["how is my computer doing", "system status", "how is the mac doing",
                       "check system status", "what is the cpu usage", "how much battery do I have",
                       "how is my system", "give me a system report"],
                      ["how is the battery", "how is the cpu doing"]),
    "agenda_today": (["what's on the agenda today", "what's on my agenda", "what do I have today",
                      "what is on my schedule today", "read my agenda", "what's on my calendar today",
                      "tell me my agenda for today", "what's my schedule"],
                     ["what's on the agenda for today", "what do I have on today"]),
}
_ARGS = {"open_app": {"app": "{app}"}, "quit_app": {"app": "{app}"}, "set_volume": {"level": "{level}"},
         "set_brightness": {"level": "{level}"}, "set_timer": {"minutes": "{minutes}"}}


@pytest.fixture(scope="session")
def vectors_path(tmp_path_factory):
    from model2vec import StaticModel
    rows = [(a, p, "train") for a, (tr, _) in _PHRASES.items() for p in tr]
    rows += [(a, p, "test") for a, (_, te) in _PHRASES.items() for p in te]
    vec = StaticModel.from_pretrained(config.EMBED_MODEL).encode([r[1] for r in rows]).astype(np.float32)
    vec /= np.linalg.norm(vec, axis=1, keepdims=True)
    path = tmp_path_factory.mktemp("vec") / "command_vectors.npz"
    np.savez(path, vectors=vec, actions=np.array([r[0] for r in rows]),
             args=np.array([json.dumps(_ARGS.get(r[0], {})) for r in rows]),
             phrases=np.array([r[1] for r in rows]), split=np.array([r[2] for r in rows]))
    return path
