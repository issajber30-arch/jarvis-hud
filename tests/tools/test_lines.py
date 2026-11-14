import yaml

import config
from tools import lines

SITUATIONS = "wake done failed confirm cancelled not_heard greeting_morning greeting_afternoon greeting_evening agenda_intro agenda_empty low_battery high_cpu followup_invite offline permission_calendar goodbye".split()


def test_bank_complete():
    bank = yaml.safe_load((config.ROOT / "prompts" / "lines.yaml").read_text())
    assert set(bank) == set(SITUATIONS)
    for s in SITUATIONS:
        assert len(set(bank[s])) >= 8, s
        assert all(isinstance(x, str) and x.strip() and "*" not in x and "#" not in x for x in bank[s])
    assert any("missed anything" in x and "add" in x for x in bank["followup_invite"])


def test_lines_no_repeat():
    for s in SITUATIONS:
        seq = [lines.pick(s) for _ in range(60)]
        assert all(a != b for a, b in zip(seq, seq[1:])), s
