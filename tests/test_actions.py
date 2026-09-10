"""
The five side effects: idempotency, partial failure, and precise retry.

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
    ACTION_SMS,
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
    monkeypatch.setattr(ghl, "send_sms", counted("sms", {"sent": True}))
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

    every = (actions.act_upsert_contact, actions.act_calendar_event,
             actions.act_deposit_invoice, actions.act_confirmation_email,
             actions.act_customer_sms)

    for node in every:
        update = node(state)
        state["ledger"] = merge_ledger(state["ledger"], update["ledger"])

    counts = dict(spy)

    # Second pass over all five — nothing should move.
    for node in every:
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


# ── The text to the customer ───────────────────────────────────────────────────
# It says two specific things have happened. Sending it when one of them has not
# is telling a customer something untrue about their own booking, so the rule it
# has to keep is "both, or neither".

def _ledger_with(**statuses):
    """A ledger where each named action has the given status."""
    ledger = new_ledger()
    updates = {
        ACTION_CONTACT: {"status": "success", "result": {"contact_id": "c1"},
                         "error": None, "attempts": 1},
    }
    for action_name, status in statuses.items():
        updates[action_name] = {
            "status": status,
            "result": {"ok": True} if status == "success" else {},
            "error": None if status == "success" else "boom",
            "attempts": 1,
        }
    return merge_ledger(ledger, updates)


def test_the_customer_is_texted_once_both_have_gone_out(intake, spy):
    state = {"intake": intake,
             "ledger": _ledger_with(**{ACTION_EMAIL: "success", ACTION_INVOICE: "success"})}

    update = actions.act_customer_sms(state)

    assert spy["sms"] == 1
    entry = update["ledger"][ACTION_SMS]
    assert entry["status"] == "success"
    assert entry["result"]["message"] == (
        "Hi Sarah, the confirmation email & deposit link have been sent!"
    )


@pytest.mark.parametrize(
    "email,invoice",
    [("failed", "success"), ("success", "failed"), ("failed", "failed"),
     ("pending", "success"), ("success", "pending")],
)
def test_nothing_is_texted_unless_both_went_out(intake, spy, email, invoice):
    state = {"intake": intake,
             "ledger": _ledger_with(**{ACTION_EMAIL: email, ACTION_INVOICE: invoice})}

    update = actions.act_customer_sms(state)

    assert "sms" not in spy, "texted the customer that something happened when it hadn't"
    assert update["ledger"][ACTION_SMS]["status"] == "skipped"


def test_a_skipped_text_is_not_reported_as_a_failure(intake, spy):
    state = {"intake": intake,
             "ledger": _ledger_with(**{ACTION_EMAIL: "failed", ACTION_INVOICE: "success"})}
    ledger = merge_ledger(state["ledger"], actions.act_customer_sms(state)["ledger"])

    assert ACTION_SMS not in failed_actions(ledger)
    assert "Confirmation email" in ledger[ACTION_SMS]["error"], "should say what it waited on"


def test_a_skip_is_reconsidered_on_a_retry(intake, spy):
    """
    The bug this guards against would be silent and permanent: if a held-back
    text were recorded as done, the customer would never be told — not even
    after the retry that finally sent the email.
    """
    state = {"intake": intake,
             "ledger": _ledger_with(**{ACTION_EMAIL: "failed", ACTION_INVOICE: "success"})}
    ledger = merge_ledger(state["ledger"], actions.act_customer_sms(state)["ledger"])
    assert "sms" not in spy

    # The operator says "retry", and the email goes out this time.
    ledger = merge_ledger(ledger, {ACTION_EMAIL: {"status": "success", "result": {},
                                                 "error": None, "attempts": 2}})
    update = actions.act_customer_sms({"intake": intake, "ledger": ledger})

    assert spy["sms"] == 1, "the text was never sent, even once it was true"
    assert update["ledger"][ACTION_SMS]["status"] == "success"


def test_a_sent_text_is_never_sent_twice(intake, spy):
    state = {"intake": intake,
             "ledger": _ledger_with(**{ACTION_EMAIL: "success", ACTION_INVOICE: "success"})}
    state["ledger"] = merge_ledger(state["ledger"], actions.act_customer_sms(state)["ledger"])

    assert actions.act_customer_sms(state) == {}
    assert spy["sms"] == 1


@pytest.mark.parametrize(
    "full_name,expected",
    [("Sarah Chen", "Sarah"), ("Nik", "Nik"), ("  Jordan  Lee ", "Jordan"),
     ("", "there"), (None, "there")],
)
def test_the_greeting_uses_their_first_name(full_name, expected):
    """
    Substituted here, not left for GoHighLevel: this goes out through
    /conversations/messages, which sends the characters it is given. A customer
    reading "Hi {first_name}," is worse than no text at all.
    """
    assert actions.first_name({"full_name": full_name}) == expected
    assert "{" not in actions.CUSTOMER_SMS.format(first_name=expected)


def test_the_report_counts_all_five_steps_and_explains_a_skip(intake, spy):
    ledger = _ledger_with(**{ACTION_CALENDAR: "success", ACTION_EMAIL: "failed",
                             ACTION_INVOICE: "success"})
    ledger = merge_ledger(ledger, actions.act_customer_sms(
        {"intake": intake, "ledger": ledger})["ledger"])

    text = actions.report({"intake": intake, "ledger": ledger})["messages"][0].content

    assert f"of {len(ALL_ACTIONS)} steps failed" in text
    assert "1 of 5" in text, "a skipped text is not a failed step"
    assert "Text to the customer" in text and "not sent" in text
