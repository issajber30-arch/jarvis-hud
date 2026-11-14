import asyncio

from contracts import Action, ActionResult
from hud.confirm_card import ConfirmCard
from tests.test_main import FakeBrain, make_app


async def test_card_visible_only_while_confirm_pending():
    async def ctl(action, confirm):
        ok = await confirm("Delete report.pdf")
        return ActionResult(action.name, ok, "Done")

    app, _ = make_app(brain=FakeBrain(["ok"], actions=[Action("delete_file", {})]), controller=ctl)
    async with app.run_test() as pilot:
        card = app.query_one(ConfirmCard)
        assert not card.display
        task = asyncio.create_task(app.handle_utterance("delete it"))
        for _ in range(50):
            await asyncio.sleep(0.01)
            if app.pending_confirm is not None:
                break
        await pilot.pause(0.05)
        assert card.display and "DELETE REPORT.PDF" in card.render().plain
        assert 'SAY "YES" · ENTER' in card.render().plain
        assert card.region.width == 46
        app.answer_confirm("yes")
        await task
        await pilot.pause(0.05)
        assert not card.display and card._timer._active.is_set() is False


async def test_bar_drains():
    app, _ = make_app()
    async with app.run_test():
        card = app.query_one(ConfirmCard)
        now = [0.0]
        card.clock = lambda: now[0]
        card.show("x", 10)
        full = card.render().plain.count("█")
        now[0] = 5.0
        assert card.render().plain.count("█") < full and card.remaining() == 5.0


async def test_card_is_centred_over_sphere_and_grid_intact():
    app, _ = make_app()
    async with app.run_test(size=(180, 50)) as pilot:
        core = app.query_one("CoreWidget").region
        card = app.query_one(ConfirmCard)
        card.show("x", 10, (core.x + core.width // 2, core.y + core.height // 2))
        await pilot.pause(0.05)
        assert core.contains_region(card.region)
        assert abs((card.region.x + 23) - (core.x + core.width // 2)) <= 1
        assert app.query_one("TickerWidget").region.y == 49
