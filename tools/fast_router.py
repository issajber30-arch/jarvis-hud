"""Cosine-similarity intent router over static model2vec embeddings."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np

import config
from contracts import FAST_ROUTE_ACTIONS, FastRouteMatch

_INT = re.compile(r"\d+")
# App routes need a command word: "what's the weather like" must reach Claude, not open Weather.app.
_APP_VERBS = re.compile(r"\b(open|launch|start|fire|bring|pull|run|load|kick|switch|get|need|quit|close|exit|"
                        r"shut|kill|stop|end|terminate|rid|done)\b", re.I)
# Names the synthetic phrases use; merged with installed apps so stored phrases and queries canonicalise alike.
COMMON_APPS = ["Safari", "Spotify", "Notes", "Terminal", "Slack", "Calendar", "Messages", "Mail", "Finder", "Photos"]
_APP_DIRS = [Path("/Applications"), Path("/System/Applications"), Path("/System/Applications/Utilities"),
             Path.home() / "Applications"]


def installed_apps() -> dict[str, str]:
    """lowercase name -> display name for every .app on this Mac, plus COMMON_APPS."""
    apps = {a.lower(): a for a in COMMON_APPS}
    for d in _APP_DIRS:
        if d.is_dir():
            apps.update({p.stem.lower(): p.stem for p in d.glob("*.app")})
    apps.pop("music", None)  # collides with music_play ("start some music"); "open the Music app" goes to Claude
    return apps


def _app_pattern(apps: dict[str, str]) -> re.Pattern:
    names = sorted(apps, key=len, reverse=True)  # longest first: "visual studio code" before "code"
    return re.compile(r"\b(" + "|".join(map(re.escape, names)) + r")\b")


def find_app(text: str, apps: dict[str, str], pattern: re.Pattern | None = None) -> str | None:
    m = (pattern or _app_pattern(apps)).search(text.lower())
    return apps[m.group(1)] if m else None


def canon_apps(text: str, apps: dict[str, str], pattern: re.Pattern | None = None) -> str:
    # App names are unbounded vocabulary: replace them with "app" so the verb decides open vs quit.
    t = text.lower().replace(",", "")
    return " ".join((pattern or _app_pattern(apps)).sub("app", t).split())


def _slot(name: str, text: str, apps: dict[str, str], pattern: re.Pattern | None = None):
    if name == "app":
        return find_app(text, apps, pattern)
    m = _INT.search(text)
    if not m:
        return None
    n = int(m.group())
    return n * 60 if name == "minutes" and re.search(r"\bhours?\b", text, re.I) else n


class FastRouter:
    def __init__(self, path=config.VECTORS_PATH, exclude_split: str | None = None, apps: dict[str, str] | None = None):
        self._apps = apps if apps is not None else installed_apps()
        self._pat = _app_pattern(self._apps)
        from model2vec import StaticModel
        d = np.load(path, allow_pickle=False)
        keep = np.ones(len(d["actions"]), bool) if exclude_split is None else d["split"] != exclude_split
        self._vec = d["vectors"][keep].astype(np.float32)
        self._actions = [str(a) for a in d["actions"][keep]]
        self._args = [json.loads(str(a)) for a in d["args"][keep]]
        self._phrases = [str(p) for p in d["phrases"][keep]]
        self._model = StaticModel.from_pretrained(config.EMBED_MODEL)
        for i, p in enumerate(self._phrases):
            c = canon_apps(p, self._apps, self._pat)
            if c != p.lower():
                v = self._model.encode([c])[0].astype(np.float32)
                self._vec[i] = v / np.linalg.norm(v)

    def route(self, text: str) -> FastRouteMatch | None:
        t0 = time.perf_counter()
        text = text.strip()
        if not text:
            return None
        q = self._model.encode([canon_apps(text, self._apps, self._pat)])[0].astype(np.float32)
        norm = float(np.linalg.norm(q))
        if norm == 0:
            return None
        sims = self._vec @ (q / norm)
        i = int(np.argmax(sims))
        score = float(sims[i])
        action = self._actions[i]
        if score < config.ROUTER_THRESHOLD or action not in FAST_ROUTE_ACTIONS:
            return None
        if action in ("open_app", "quit_app") and not _APP_VERBS.search(text):
            return None
        args = {}
        for key, tmpl in self._args[i].items():
            slot = re.fullmatch(r"\{(\w+)\}", str(tmpl))
            val = _slot(slot.group(1), text, self._apps, self._pat) if slot else tmpl
            if val is None:
                return None  # cannot fill a slot: let Claude handle it
            args[key] = val
        return FastRouteMatch(action, args, score, self._phrases[i], (time.perf_counter() - t0) * 1000)
