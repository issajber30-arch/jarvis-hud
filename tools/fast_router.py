"""Cosine-similarity intent router over static model2vec embeddings."""
from __future__ import annotations

import json
import re
import time

import numpy as np

import config
from contracts import FAST_ROUTE_ACTIONS, FastRouteMatch

_APP = re.compile(r"^(?:please\s+)?(?:open|launch|start|quit|close|exit|shut\s+down)\s+(?:up\s+)?(?:the\s+)?(.+?)"
                  r"(?:\s+app(?:lication)?)?(?:\s+(?:please|for me))?\W*$", re.I)
# App names are unbounded vocabulary: collapse "open Safari" / "quit the Notes app" to "open app" / "quit app"
# in both the stored phrases and the query so the verb decides the intent, not the name.
_CANON = re.compile(r"^(?:please\s+)?(open|launch|quit|close|exit|shut\s+down)\s+(?:up\s+)?(?:the\s+)?(\S+(?:\s+\S+){0,2}?)"
                    r"(?:\s+app(?:lication)?)?(?:\s+(?:please|for me))?\W*$", re.I)
_VERB = {"open": "open", "launch": "open", "quit": "quit", "close": "quit", "exit": "quit", "shut": "quit"}
_INT = re.compile(r"\d+")


def _canon(text: str) -> str:
    m = _CANON.match(text.strip())
    return f"{_VERB[m.group(1).split()[0].lower()]} app" if m else text


def _slot(name: str, text: str):
    if name == "app":
        m = _APP.match(text.strip())
        return m.group(1).strip() if m and m.group(1).strip() else None
    m = _INT.search(text)
    if not m:
        return None
    n = int(m.group())
    return n * 60 if name == "minutes" and re.search(r"\bhours?\b", text, re.I) else n


class FastRouter:
    def __init__(self, path=config.VECTORS_PATH, exclude_split: str | None = None):
        from model2vec import StaticModel
        d = np.load(path, allow_pickle=False)
        keep = np.ones(len(d["actions"]), bool) if exclude_split is None else d["split"] != exclude_split
        self._vec = d["vectors"][keep].astype(np.float32)
        self._actions = [str(a) for a in d["actions"][keep]]
        self._args = [json.loads(str(a)) for a in d["args"][keep]]
        self._phrases = [str(p) for p in d["phrases"][keep]]
        self._model = StaticModel.from_pretrained(config.EMBED_MODEL)
        for i, p in enumerate(self._phrases):
            c = _canon(p)
            if c != p:
                v = self._model.encode([c])[0].astype(np.float32)
                self._vec[i] = v / np.linalg.norm(v)

    def route(self, text: str) -> FastRouteMatch | None:
        t0 = time.perf_counter()
        text = text.strip()
        if not text:
            return None
        q = self._model.encode([_canon(text)])[0].astype(np.float32)
        norm = float(np.linalg.norm(q))
        if norm == 0:
            return None
        sims = self._vec @ (q / norm)
        i = int(np.argmax(sims))
        score = float(sims[i])
        action = self._actions[i]
        if score < config.ROUTER_THRESHOLD or action not in FAST_ROUTE_ACTIONS:
            return None
        args = {}
        for key, tmpl in self._args[i].items():
            slot = re.fullmatch(r"\{(\w+)\}", str(tmpl))
            val = _slot(slot.group(1), text) if slot else tmpl
            if val is None:
                return None  # cannot fill a slot: let Claude handle it
            args[key] = val
        return FastRouteMatch(action, args, score, self._phrases[i], (time.perf_counter() - t0) * 1000)
