from datetime import datetime
from types import SimpleNamespace as NS

import pytest
from Foundation import NSDate, NSDateComponents

from tools import calendar as cal


def _date(h, m=0):
    d = datetime.now().replace(hour=h, minute=m, second=0, microsecond=0)
    return NSDate.dateWithTimeIntervalSince1970_(d.timestamp())


def _comps(h, m=0):
    n = datetime.now()
    c = NSDateComponents.alloc().init()
    c.setYear_(n.year); c.setMonth_(n.month); c.setDay_(n.day); c.setHour_(h); c.setMinute_(m)
    return c


class FakeStore:
    def __init__(self, granted=True):
        self.granted = granted
        self.cals = [NS(title=lambda: "Work", type=lambda: 1), NS(title=lambda: "Birthdays", type=lambda: 4),
                     NS(title=lambda: "UK Holidays", type=lambda: 3)]
        self.events = [NS(title=lambda: "Standup", startDate=lambda: _date(9), endDate=lambda: _date(10),
                          eventIdentifier=lambda: "e1", calendar=lambda: self.cals[0]),
                       NS(title=lambda: "Sam's birthday", startDate=lambda: _date(0), endDate=lambda: _date(1),
                          eventIdentifier=lambda: "e2", calendar=lambda: self.cals[1]),
                       NS(title=lambda: "Bank holiday", startDate=lambda: _date(0), endDate=lambda: _date(1),
                          eventIdentifier=lambda: "e3", calendar=lambda: self.cals[2])]
        self.reminders = [NS(title=lambda: "Call mum", dueDateComponents=lambda: _comps(17),
                             calendarItemIdentifier=lambda: "r1")]

    def requestFullAccessToEventsWithCompletion_(self, cb): cb(self.granted, None)
    def requestFullAccessToRemindersWithCompletion_(self, cb): cb(self.granted, None)
    def calendarsForEntityType_(self, t): return self.cals
    def predicateForEventsWithStartDate_endDate_calendars_(self, s, e, cals): return ("ev", cals)
    def eventsMatchingPredicate_(self, p): return [e for e in self.events if e.calendar() in p[1]]
    def predicateForIncompleteRemindersWithDueDateStarting_ending_calendars_(self, s, e, c): return ("rem", c)
    def fetchRemindersMatchingPredicate_completion_(self, p, cb):
        # PyObjC aborts the process if a void completion block returns a value — enforce it like macOS does
        assert cb(self.reminders) is None, "EventKit completion must return None"


def test_agenda_today_filters(monkeypatch):
    monkeypatch.setattr(cal, "_get_store", lambda: FakeStore())
    items = cal.agenda_today()
    assert [(i["kind"], i["title"]) for i in items] == [("event", "Standup"), ("reminder", "Call mum")]
    assert items[0]["id"] == "e1" and items[1]["id"] == "r1"
    assert items[0]["start"].hour == 9


def test_agenda_permission_denied(monkeypatch):
    monkeypatch.setattr(cal, "_get_store", lambda: FakeStore(granted=False))
    with pytest.raises(cal.CalendarPermissionError):
        cal.agenda_today()
