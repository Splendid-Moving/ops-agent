"""
Invoice dates, and why a booking silently lost its payment link.

act_deposit_invoice passed the MOVE date as the invoice's issue date, while
create_invoice derived the due date from `now`. Every move further out than
tomorrow therefore asked GHL to issue an invoice after its own due date, which
it refuses with "Issue date cannot be after due date".

Nothing caught it. The checklist enforces a 2-day minimum lead time, so the
failing case was the NORMAL case and the passing case was the rare one. The
contact, the calendar event and the confirmation email all succeeded, so the
customer got a booking confirmation promising a payment link that never arrived.
"""

from datetime import datetime, timedelta

import pytest

from agent.nodes import actions
from services import config, ghl
from services.calendar import LA_TZ


@pytest.fixture
def captured(monkeypatch):
    """Capture the invoice payload instead of sending it."""
    sent: dict = {}

    class FakeResponse:
        ok, status_code = True, 201
        @staticmethod
        def json():
            return {"invoice": {"_id": "inv_test"}}

    def fake_post(url, headers=None, json=None, timeout=None):
        if "/invoices/" in url and not url.rstrip("/").endswith("send"):
            sent.update(json)
        return FakeResponse()

    monkeypatch.setattr(config, "dry_run", lambda: False)
    monkeypatch.setattr(ghl, "get_contact", lambda _: {"contactName": "Sarah Chen", "phone": "+18185550142", "email": "s@example.com"})
    monkeypatch.setattr(ghl, "get_business_details", lambda: {"name": "Splendid Moving"})
    monkeypatch.setattr(ghl.requests, "post", fake_post)
    return sent


# ── The rule GHL enforces ──────────────────────────────────────────────────────

@pytest.mark.parametrize("days_out", [0, 1, 2, 7, 30, 365])
def test_due_date_is_never_before_the_issue_date(captured, days_out):
    """The one invariant. Violating it is a hard 400 from GHL, not a warning."""
    issue = (datetime.now(LA_TZ) + timedelta(days=days_out)).strftime("%Y-%m-%d")
    ghl.create_invoice(contact_id="c1", amount=50.0, item_name="Deposit", issue_date=issue)
    assert captured["dueDate"] >= captured["issueDate"]


def test_due_date_follows_the_issue_date_not_today(captured):
    """
    Deriving it from `now` is what made a future issue date unrepresentable
    without a 400. Keeping both in one expression removes the whole class.
    """
    issue = (datetime.now(LA_TZ) + timedelta(days=10)).strftime("%Y-%m-%d")
    ghl.create_invoice(contact_id="c1", amount=50.0, item_name="Deposit", issue_date=issue)
    expected = (datetime.strptime(issue, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    assert captured["dueDate"] == expected


# ── What the node asks for ─────────────────────────────────────────────────────

def test_the_deposit_is_issued_today_not_on_the_move_date(captured, monkeypatch):
    """
    The deposit holds the slot, so it is due now. Issuing it on the move date
    would mean the customer is not asked to pay until the day they move — which
    is not a deposit at all, quite apart from the API rejecting it.
    """
    monkeypatch.setattr(actions.ghl, "send_invoice", lambda *a, **k: {"sent": True})
    move_date = (datetime.now(LA_TZ) + timedelta(days=21)).strftime("%m/%d/%Y")

    actions.act_deposit_invoice.__wrapped__({
        "intake": {"move_date": move_date, "phone": "(818) 555-0142"},
        "ledger": {actions.ACTION_CONTACT: {"status": "success", "result": {"contact_id": "c1"}}},
    })

    assert captured["issueDate"] == datetime.now(LA_TZ).strftime("%Y-%m-%d")


def test_the_move_date_still_reaches_the_customer(captured, monkeypatch):
    """It belongs in the description — just not in the dates."""
    monkeypatch.setattr(actions.ghl, "send_invoice", lambda *a, **k: {"sent": True})
    move_date = (datetime.now(LA_TZ) + timedelta(days=21)).strftime("%m/%d/%Y")

    actions.act_deposit_invoice.__wrapped__({
        "intake": {"move_date": move_date, "phone": "(818) 555-0142"},
        "ledger": {actions.ACTION_CONTACT: {"status": "success", "result": {"contact_id": "c1"}}},
    })

    printed = captured["name"] + str(captured["items"])
    assert move_date in printed


# ── Errors have to be readable ─────────────────────────────────────────────────

def test_ghl_errors_carry_the_reason_in_the_message():
    """
    The ledger records str(exc) and the log prints the traceback. With the body
    only on an attribute, both showed "Invoice creation failed" and the real
    explanation was never seen by anyone.
    """
    exc = ghl.GHLError("Invoice creation failed", 400,
                       '{"message":"Issue date cannot be after due date"}')
    assert "Issue date cannot be after due date" in str(exc)
    assert "400" in str(exc)


def test_a_bodyless_error_still_reads_cleanly():
    assert str(ghl.GHLError("Something broke")) == "Something broke"
