"""Spoken-line bank: random, never the same line twice in a row per situation."""
import random
from functools import lru_cache

import yaml

import config

_last: dict[str, str] = {}


@lru_cache(maxsize=1)
def _bank() -> dict[str, list[str]]:
    return yaml.safe_load((config.ROOT / "prompts" / "lines.yaml").read_text(encoding="utf-8"))


def pick(situation: str) -> str:
    options = [x for x in _bank()[situation] if x != _last.get(situation)]
    _last[situation] = random.choice(options)
    return _last[situation]
