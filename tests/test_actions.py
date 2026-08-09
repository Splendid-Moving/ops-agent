"""
The four side effects: idempotency, partial failure, and precise retry.

These are the highest-value tests in the project. Everything else produces a
wrong answer when it breaks; these produce a customer charged twice, or a truck
that was never booked for a job the agent said was confirmed.
"""

import pytest

from agent.nodes import actions
from agent.state import (
    ACTION_CALENDAR,
    ACTION_CONTACT,
    ACTION_EMAIL,
    ACTION_INVOICE,
    ALL_ACTIONS,
    failed_actions,
    merge_ledger,
    new_ledger,
    succeeded,
)
from services import calendar, ghl


@pytest.fixture
def intake():
    from datetime import datetime, timedelta

    from services.calendar import LA_TZ

    return {
        "full_name": "Sarah Chen",
        "email": "sarah@example.com",
        "phone": "+1(818)555-0142",
        "pickup_address": "412 N Maple St, Burbank CA 91505",
        "dropoff_address": "1830 Pine St, Glendale CA 91206",
        "move_date": (datetime.now(LA_TZ) + timedelta(days=10)).strftime("%m/%d/%Y"),
        "arrival_time": "8-9am",
        "movers": "3",
        "job_notes": "Third floor walkup",
    }


@pytest.fixture
def spy(monkeypatch):
    """Count every outbound call so double-fires are visible."""
    calls: dict[str, int] = {}

    def counted(name, result):
        def fn(*args, **kwargs):
            calls[name] = calls.get(name, 0) + 1
            return result
        return fn

    monkeypatch.setattr(ghl, "upsert_contact", counted("contact", {"contact_id": "c1", "created": True}))
    monkeypatch.setattr(ghl, "create_invoice", counted("invoice", {"invoice_id": "i1"}))
    monkeypatch.setattr(ghl, "send_invoice", counted("send_invoice", {"sent": True}))
    monkeypatch.setattr(ghl, "send_email", counted("email", {"sent": True}))
    monkeypatch.setattr(calendar, "create_event", counted("event", {"event_id": "e1", "html_link": "x"}))
    monkeypatch.setattr(actions.maps, "get_distance", lambda *a, **k: "8.2 miles")
    return calls


# ── Idempotency ────────────────────────────────────────────────────────────────

def test_a_succeeded_action_never_runs_twice(intake, spy):
    """
    The single most important guarantee. Resuming an interrupt re-runs the node,
    and a retry re-enters the whole execute stage — neither may re-charge the
    customer.
    """
    state = {"intake": intake, "ledger": new_ledger()}

    first = actions.act_upsert_contact(state)
    state["ledger"] = merge_ledger(state["ledger"], first["ledger"])
    assert spy["contact"] == 1

    # Run it again, exactly as a retry or a resume would.
    second = actions.act_upsert_contact(state)
    assert spy["contact"] == 1, "contact was created twice"
    assert second == {}, "a completed action must report no change"


def test_every_action_is_individually_idempotent(intake, spy):
    state = {"intake": intake, "ledger": new_ledger()}

    for node in (actions.act_upsert_contact, actions.act_calendar_event,
                 actions.act_deposit_invoice, actions.act_confirmation_email):
        update = node(state)
        state["ledger"] = merge_ledger(state["ledger"], update["ledger"])

    counts = dict(spy)

    # Second pass over all four — nothing should move.
    for node in (actions.act_upsert_contact, actions.act_calendar_event,
                 actions.act_deposit_invoice, actions.act_confirmation_email):
        assert node(state) == {}

    assert dict(spy) == counts, "an action re-fired on the second pass"


def test_invoice_is_created_and_sent_exactly_once(intake, spy):
    """The one that costs money if it repeats."""
    state = {"intake": intake, "ledger": merge_ledger(
        new_ledger(), {ACTION_CONTACT: {"status": "success", "result": {"contact_id": "c1"}}}
    )}
    update = state["ledger"]
    for _ in range(3):
        result = actions.act_deposit_invoice({"intake": intake, "ledger": update})
        if result:
            update = merge_ledger(update, result["ledger"])

    assert spy["invoice"] == 1
    assert spy["send_invoice"] == 1


# ── Partial failure ────────────────────────────────────────────────────────────

def test_one_failure_does_not_stop_the_others(intake, spy, monkeypatch):
    """
    A booking where the calendar write fails must still produce an invoice and
    an email — and an accurate record of which step failed.
    """
    monkeypatch.setattr(
        calendar, "create_event",
        lambda **kw: (_ for _ in ()).throw(RuntimeError("calendar is down")),
    )

    ledger = new_ledger()
    for node in (actions.act_upsert_contact, actions.act_calendar_event,
                 actions.act_deposit_invoice, actions.act_confirmation_email):
        ledger = merge_ledger(ledger, node({"intake": intake, "ledger": ledger})["ledger"])

    assert succeeded(ledger, ACTION_CONTACT)
    assert succeeded(ledger, ACTION_INVOICE)
    assert succeeded(ledger, ACTION_EMAIL)
    assert failed_actions(ledger) == [ACTION_CALENDAR]
    assert "calendar is down" in ledger[ACTION_CALENDAR]["error"]


def test_retry_reruns_only_what_failed(intake, spy, monkeypatch):
    """The reason the ledger exists."""
    monkeypatch.setattr(
        calendar, "create_event",
        lambda **kw: (_ for _ in ()).throw(RuntimeError("transient")),
    )

    ledger = new_ledger()
    for node in (actions.act_upsert_contact, actions.act_calendar_event,
                 actions.act_deposit_invoice, actions.act_confirmation_email):
        ledger = merge_ledger(ledger, node({"intake": intake, "ledger": ledger})["ledger"])

    before = dict(spy)
    assert failed_actions(ledger) == [ACTION_CALENDAR]

    # Calendar recovers; retry the whole stage.
    monkeypatch.setattr(calendar, "create_event",
                        lambda **kw: {"event_id": "e2", "html_link": "link"})
    for node in (actions.act_upsert_contact, actions.act_calendar_event,
                 actions.act_deposit_invoice, actions.act_confirmation_email):
        if update := node({"intake": intake, "ledger": ledger}):
            ledger = merge_ledger(ledger, update["ledger"])

    assert not failed_actions(ledger)
    assert ledger[ACTION_CALENDAR]["result"]["event_id"] == "e2"
    # Nothing that already worked was touched again.
    assert spy["contact"] == before["contact"]
    assert spy["invoice"] == before["invoice"]
    assert spy["email"] == before["email"]


def test_an_action_never_raises_into_the_graph(intake, monkeypatch):
    """
    A raise would abort the remaining branches and leave no record of what
    happened — strictly worse than a recorded failure.
    """
    monkeypatch.setattr(ghl, "upsert_contact",
                        lambda **kw: (_ for _ in ()).throw(ValueError("boom")))
    update = actions.act_upsert_contact({"intake": intake, "ledger": new_ledger()})
    assert update["ledger"][ACTION_CONTACT]["status"] == "failed"
    assert "boom" in update["ledger"][ACTION_CONTACT]["error"]


def test_attempts_are_counted(intake, monkeypatch):
    monkeypatch.setattr(ghl, "upsert_contact",
                        lambda **kw: (_ for _ in ()).throw(ValueError("boom")))
    ledger = new_ledger()
    for expected in (1, 2, 3):
        ledger = merge_ledger(
            ledger, actions.act_upsert_contact({"intake": intake, "ledger": ledger})["ledger"]
        )
        assert ledger[ACTION_CONTACT]["attempts"] == expected


# ── The contact gate ───────────────────────────────────────────────────────────

def test_no_contact_means_skip_straight_to_report():
    """Invoice and email have nothing to attach to without a contact id."""
    ledger = merge_ledger(new_ledger(), {ACTION_CONTACT: {"status": "failed", "error": "x"}})
    assert actions.contact_gate({"ledger": ledger}) == "report"


def test_contact_success_fans_out_to_all_three():
    ledger = merge_ledger(
        new_ledger(), {ACTION_CONTACT: {"status": "success", "result": {"contact_id": "c1"}}}
    )
    assert actions.contact_gate({"ledger": ledger}) == actions.FAN_OUT
    assert len(actions.FAN_OUT) == 3


def test_invoice_refuses_without_a_contact_id(intake):
    update = actions.act_deposit_invoice({"intake": intake, "ledger": new_ledger()})
    assert update["ledger"][ACTION_INVOICE]["status"] == "failed"
    assert "contact" in update["ledger"][ACTION_INVOICE]["error"].lower()


# ── Calendar payload ───────────────────────────────────────────────────────────

def test_calendar_deposit_says_unpaid_not_the_amount(intake, spy, monkeypatch):
    """
    The Deposit line tracks payment, not billing. A separate automation rewrites
    it once the money lands.
    """
    captured = {}
    monkeypatch.setattr(calendar, "create_event",
                        lambda **kw: captured.update(kw) or {"event_id": "e1"})
    actions.act_calendar_event({"intake": intake, "ledger": new_ledger()})
    assert "Deposit: ???" in captured["description"]
    assert "Deposit: 50" not in captured["description"]


def test_labor_job_gets_the_suffix(intake, spy, monkeypatch):
    captured = {}
    monkeypatch.setattr(calendar, "create_event",
                        lambda **kw: captured.update(kw) or {"event_id": "e1"})
    actions.act_calendar_event({"intake": {**intake, "is_labor": True}, "ledger": new_ledger()})
    assert captured["title"].endswith("(labor)")


def test_event_carries_a_fingerprint_for_duplicate_detection(intake, spy, monkeypatch):
    captured = {}
    monkeypatch.setattr(calendar, "create_event",
                        lambda **kw: captured.update(kw) or {"event_id": "e1"})
    actions.act_calendar_event({"intake": intake, "ledger": new_ledger()})
    assert captured["fingerprint"] == actions.job_fingerprint(intake)


def test_fingerprint_is_stable_across_phone_formatting(intake):
    """Same customer, same date, differently typed phone — one fingerprint."""
    a = actions.job_fingerprint({**intake, "phone": "(818) 555-0142"})
    b = actions.job_fingerprint({**intake, "phone": "+18185550142"})
    assert a == b


def test_fingerprint_differs_by_date(intake):
    a = actions.job_fingerprint(intake)
    b = actions.job_fingerprint({**intake, "move_date": "12/25/2026"})
    assert a != b


def test_unparseable_date_fails_rather_than_guessing(intake):
    """Booking a move for 'today' because a date failed to parse is unacceptable."""
    update = actions.act_calendar_event(
        {"intake": {**intake, "move_date": "whenever"}, "ledger": new_ledger()}
    )
    assert update["ledger"][ACTION_CALENDAR]["status"] == "failed"


# ── Reporting ──────────────────────────────────────────────────────────────────

def test_report_names_the_failed_step(intake):
    ledger = new_ledger()
    ledger = merge_ledger(ledger, {
        ACTION_CONTACT: {"status": "success", "result": {"contact_id": "c1"}},
        ACTION_CALENDAR: {"status": "failed", "error": "RuntimeError: calendar is down"},
        ACTION_INVOICE: {"status": "success", "result": {"invoice_id": "i1"}},
        ACTION_EMAIL: {"status": "success", "result": {}},
    })
    text = actions.report({"intake": intake, "ledger": ledger})["messages"][0].content
    assert "Partly done" in text
    assert "calendar is down" in text
    assert "retry" in text.lower()


def test_report_is_unambiguous_about_full_success(intake):
    ledger = {a: {"status": "success", "result": {}} for a in ALL_ACTIONS}
    text = actions.report({"intake": intake, "ledger": ledger})["messages"][0].content
    assert "Done" in text
    assert "✗" not in text
