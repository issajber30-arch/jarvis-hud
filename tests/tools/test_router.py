import time

import numpy as np

from contracts import FAST_ROUTE_ACTIONS
from tools.fast_router import FastRouter


def test_fixture_covers_all_fast_actions(vectors_path):
    d = np.load(vectors_path)
    assert set(d["actions"]) == set(FAST_ROUTE_ACTIONS)


def test_router_accuracy_latency(real_vectors_path):
    # graded on the real 600+ held-out phrases; the tiny fixture is too thin to measure accuracy
    d = np.load(real_vectors_path)
    r = FastRouter(real_vectors_path, exclude_split="test")
    test = [(str(p), str(a)) for p, a, s in zip(d["phrases"], d["actions"], d["split"]) if s == "test"]
    hits = sum(1 for p, a in test if (m := r.route(p)) and m.action == a)
    acc = hits / len(test)
    print(f"router held-out accuracy {acc:.3f} on {len(test)} phrases")
    assert acc >= 0.90
    assert r.route("write me a poem about rain") is None
    for _ in range(20):
        r.route("set a timer for 5 minutes")
    times = []
    for i in range(200):
        t = time.perf_counter()
        r.route(test[i % len(test)][0])
        times.append((time.perf_counter() - t) * 1000)
    p95 = float(np.percentile(times, 95))
    print(f"router p95 latency {p95:.2f} ms")
    assert p95 < 15


def test_router_slots(vectors_path):
    r = FastRouter(vectors_path)
    assert r.route("open safari").args == {"app": "Safari"}
    assert r.route("volume to 40").args == {"level": 40}
    assert r.route("set a timer for 7 minutes").args == {"minutes": 7}
    assert r.route("close the Notes app please").args == {"app": "Notes"}


def test_router_missing_slot_falls_through(vectors_path):
    assert FastRouter(vectors_path).route("set the volume to loud") is None



APPS = {"safari": "Safari", "spotify": "Spotify", "photos": "Photos", "visual studio code": "Visual Studio Code", "music": "Music"}


def test_app_slot_found_anywhere():
    # Transcripts arrive as "Hey Jarvis, open Safari." and phrasing varies; the app is found by name, not position
    from tools.fast_router import find_app, canon_apps
    for text in ["hey jarvis open photos", "Hey Jarvis, fire up Photos.", "i need photos open", "can you quit photos please"]:
        assert find_app(text, APPS) == "Photos", text
    assert find_app("open visual studio code", APPS) == "Visual Studio Code"  # longest name wins
    assert find_app("close the window", APPS) is None  # not an installed app -> Claude handles it
    assert canon_apps("Hey Jarvis, can you quit Spotify?", APPS) == "hey jarvis can you quit app?"


def test_music_is_not_an_app_name():
    # "start some music" must play music, not open Music.app
    from tools.fast_router import installed_apps
    assert "music" not in installed_apps()
