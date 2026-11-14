import numpy as np

from contracts import FAST_ROUTE_ACTIONS
from data.generate_synthetic import build, generate_pairs


def test_pairs_cover_fast_routes():
    pairs = generate_pairs()
    assert len(pairs) >= 120
    for a in FAST_ROUTE_ACTIONS:
        assert sum(1 for _, act, _ in pairs if act == a) >= 8, a


def test_no_non_fast_actions():
    assert {a for _, a, _ in generate_pairs()} <= FAST_ROUTE_ACTIONS


def test_npz_shape_and_norm(tmp_path):
    p = tmp_path / "v.npz"
    build(p)
    d = np.load(p)
    n = len(d["vectors"])
    assert d["vectors"].dtype == np.float32
    assert all(len(d[k]) == n for k in ("actions", "args", "phrases", "split"))
    assert np.allclose(np.linalg.norm(d["vectors"], axis=1), 1, atol=1e-4)
    assert {"train", "test"} <= set(d["split"].tolist())


def test_agenda_phrases():
    ph = [p for p, a, _ in generate_pairs() if a == "agenda_today"]
    assert "what's on the agenda today" in ph
    assert len(ph) >= 8
