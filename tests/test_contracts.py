from contracts import (
    ACTIONS, FAST_ROUTE_ACTIONS, TRANSITIONS, Action, HUDState, Tier, can_transition,
)


def test_unknown_action_is_blocked():
    assert Action("rm_everything", {}).tier is Tier.BLOCKED


def test_known_tiers():
    assert Action("open_app", {"app": "Safari"}).tier is Tier.IMMEDIATE
    assert Action("delete_file", {"path": "x"}).tier is Tier.CONFIRM


def test_fast_route_subset():
    immediate = {k for k, v in ACTIONS.items() if v is Tier.IMMEDIATE}
    assert FAST_ROUTE_ACTIONS <= immediate
    assert "create_event" not in FAST_ROUTE_ACTIONS
    assert "agenda_today" in FAST_ROUTE_ACTIONS


def test_transitions():
    assert can_transition(HUDState.IDLE, HUDState.LISTENING)
    assert not can_transition(HUDState.IDLE, HUDState.SPEAKING)
    assert set(TRANSITIONS) == set(HUDState)
