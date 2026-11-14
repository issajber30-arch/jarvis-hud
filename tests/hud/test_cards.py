import asyncio

from contracts import Action, ActionResult
from hud.cards_widget import CardsWidget
from tests.test_main import FakeBrain, make_app


async def test_timer_counts_down_and_expires():
    app, _ = make_app()
    async with app.run_test() as pilot:
        c = app.query_one(CardsWidget)
        now = [100.0]
        c.clock = lambda: now[0]
        done = []
        c.on_done = done.append
        c.add_timer("timer", 90)
        assert "TIMER  01:30" in c.render().plain
        now[0] = 130.0
        c._tick()
        assert "TIMER  01:00" in c.render().plain
        now[0] = 190.0
        c._tick()
        assert not c.timers and done == ["TIMER"] and "NOTHING ACTIVE" in c.render().plain


async def test_note_expires_after_ttl():
    app, _ = make_app()
    async with app.run_test():
        c = app.query_one(CardsWidget)
        now = [0.0]
        c.clock = lambda: now[0]
        c.add_note("✓ reminder 17:00 call mom", ttl=20)
        assert "✓ REMINDER 17:00 CALL MOM" in c.render().plain
        now[0] = 19.0
        c._tick()
        assert c.notes
        now[0] = 21.0
        c._tick()
        assert not c.notes


async def test_actions_feed_cards():
    async def ctl(action, confirm):
        return ActionResult(action.name, True, "Done")

    acts = [Action("set_timer", {"minutes": 2}), Action("create_reminder", {"title": "call mom"}),
            Action("create_event", {"title": "lunch", "start": "x", "end": "y"})]
    app, _ = make_app(brain=FakeBrain(["ok"], actions=acts), controller=ctl)
    async with app.run_test():
        await app.handle_utterance("set things")
        c = app.query_one(CardsWidget)
        assert len(c.timers) == 1 and 100 < c.timers[0][1] - c.clock() <= 120
        assert [n[0] for n in c.notes] == ["✓ REMINDER CALL MOM", "✓ EVENT LUNCH"]


async def test_failed_action_adds_no_card():
    async def ctl(action, confirm):
        return ActionResult(action.name, False, "nope")

    app, _ = make_app(brain=FakeBrain(["ok"], actions=[Action("set_timer", {"minutes": 2})]), controller=ctl)
    async with app.run_test():
        await app.handle_utterance("timer")
        assert not app.query_one(CardsWidget).timers
