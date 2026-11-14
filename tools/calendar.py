"""Calendar and Reminders through EventKit (pyobjc). Blocking; callers run it in a thread."""
from __future__ import annotations

import threading
from datetime import datetime, timedelta

_TIMEOUT = 10.0
_EVENTS, _REMINDERS = 0, 1          # EKEntityType
_BIRTHDAY = 4                       # EKCalendarTypeBirthday


class CalendarPermissionError(Exception):
    pass


def _get_store():
    from EventKit import EKEventStore
    return EKEventStore.alloc().init()


def _grant(store, kind: int) -> None:
    """Request full access once per call; raise CalendarPermissionError if denied or unanswered."""
    done, result = threading.Event(), []

    def cb(granted, error):
        result.append(bool(granted))
        done.set()

    req = store.requestFullAccessToEventsWithCompletion_ if kind == _EVENTS \
        else store.requestFullAccessToRemindersWithCompletion_
    req(cb)
    if not done.wait(_TIMEOUT) or not result[0]:
        raise CalendarPermissionError("Calendar/Reminders access denied")


def _ns(dt: datetime):
    from Foundation import NSDate
    return NSDate.dateWithTimeIntervalSince1970_(dt.timestamp())


def _py(nsdate) -> datetime:
    return datetime.fromtimestamp(nsdate.timeIntervalSince1970())


def _day() -> tuple[datetime, datetime]:
    start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    return start, start + timedelta(days=1)


def _wanted(cal) -> bool:
    return cal.type() != _BIRTHDAY and "holiday" not in str(cal.title()).lower()


def _due(comps) -> datetime | None:
    if comps is None:
        return None
    from Foundation import NSCalendar
    d = NSCalendar.currentCalendar().dateFromComponents_(comps)
    return _py(d) if d else None


def agenda_today() -> list[dict]:
    start, end = _day()
    store = _get_store()
    _grant(store, _EVENTS)
    cals = [c for c in store.calendarsForEntityType_(_EVENTS) if _wanted(c)]
    items = []
    if cals:
        pred = store.predicateForEventsWithStartDate_endDate_calendars_(_ns(start), _ns(end), cals)
        for e in store.eventsMatchingPredicate_(pred) or []:
            items.append({"kind": "event", "title": str(e.title()), "start": _py(e.startDate()),
                          "end": _py(e.endDate()), "id": str(e.eventIdentifier())})
    _grant(store, _REMINDERS)
    rcals = [c for c in store.calendarsForEntityType_(_REMINDERS) if _wanted(c)]
    got, done = [], threading.Event()
    pred = store.predicateForIncompleteRemindersWithDueDateStarting_ending_calendars_(_ns(start), _ns(end), rcals)

    def on_reminders(r) -> None:  # must return None: PyObjC aborts the process on a non-void block return
        got.extend(r or [])
        done.set()

    store.fetchRemindersMatchingPredicate_completion_(pred, on_reminders)
    done.wait(_TIMEOUT)
    for r in got:
        due = _due(r.dueDateComponents())
        if due and start <= due < end:
            items.append({"kind": "reminder", "title": str(r.title()), "start": due, "end": due,
                          "id": str(r.calendarItemIdentifier())})
    return sorted(items, key=lambda i: i["start"])


def _save(ok_err) -> None:
    ok, err = ok_err
    if not ok:
        raise RuntimeError(f"EventKit save failed: {err}")


def create_event(title: str, start: datetime, end: datetime) -> str:
    from EventKit import EKEvent
    store = _get_store()
    _grant(store, _EVENTS)
    ev = EKEvent.eventWithEventStore_(store)
    ev.setTitle_(title); ev.setStartDate_(_ns(start)); ev.setEndDate_(_ns(end))
    ev.setCalendar_(store.defaultCalendarForNewEvents())
    _save(store.saveEvent_span_error_(ev, 0, None))
    return str(ev.eventIdentifier())


def create_reminder(title: str, due: datetime | None = None) -> str:
    from EventKit import EKReminder
    from Foundation import NSCalendar, NSCalendarUnitDay, NSCalendarUnitHour, NSCalendarUnitMinute
    from Foundation import NSCalendarUnitMonth, NSCalendarUnitYear
    store = _get_store()
    _grant(store, _REMINDERS)
    rem = EKReminder.reminderWithEventStore_(store)
    rem.setTitle_(title)
    rem.setCalendar_(store.defaultCalendarForNewReminders())
    if due:
        units = NSCalendarUnitYear | NSCalendarUnitMonth | NSCalendarUnitDay | NSCalendarUnitHour | NSCalendarUnitMinute
        rem.setDueDateComponents_(NSCalendar.currentCalendar().components_fromDate_(units, _ns(due)))
    _save(store.saveReminder_commit_error_(rem, True, None))
    return str(rem.calendarItemIdentifier())


def delete_event(id: str) -> None:
    store = _get_store()
    _grant(store, _EVENTS)
    ev = store.eventWithIdentifier_(id)
    if ev is None:
        raise LookupError("event not found")
    _save(store.removeEvent_span_error_(ev, 0, None))


def reschedule_event(id: str, start: datetime, end: datetime) -> None:
    store = _get_store()
    _grant(store, _EVENTS)
    ev = store.eventWithIdentifier_(id)
    if ev is None:
        raise LookupError("event not found")
    ev.setStartDate_(_ns(start)); ev.setEndDate_(_ns(end))
    _save(store.saveEvent_span_error_(ev, 0, None))


def delete_reminder(id: str) -> None:
    store = _get_store()
    _grant(store, _REMINDERS)
    rem = store.calendarItemWithIdentifier_(id)
    if rem is None:
        raise LookupError("reminder not found")
    _save(store.removeReminder_commit_error_(rem, True, None))


def title_of(id: str) -> str | None:
    store = _get_store()
    _grant(store, _EVENTS)
    item = store.eventWithIdentifier_(id)
    if item is None:
        _grant(store, _REMINDERS)
        item = store.calendarItemWithIdentifier_(id)
    return str(item.title()) if item is not None else None
