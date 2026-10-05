"""
Showing, before anything is written, which contact a booking will update.

Two things this buys. A returning customer is worth knowing about. And a
mistyped phone number that happens to match somebody else's record is otherwise
invisible: the booking data is perfectly valid, every check passes, and the
agent quietly overwrites a different customer's details.
"""

from datetime import datetime, timedelta

import pytest

from agent.nodes import confirm as confirm_module
from services import ghl
from services.calendar import LA_TZ


def _intake(**overrides) -> dict:
    base = {
        "full_name": "Katherine Caneba",
        "email": "katherineleenc@gmail.com",
        "phone": "(906) 369-4644",
        "pickup_address": "412 N Maple Ave, Burbank CA 91505",
        "dropoff_address": "1830 Pine St, Glendale CA 91206",
        "move_date": (datetime.now(LA_TZ) + timedelta(days=10)).strftime("%m/%d/%Y"),
        "arrival_time": "8-9am",
        "movers": "3",
        "job_notes": "none",
        "notes_asked": True,
    }
    base.update(overrides)
    return base


RETURNING = {"id": "c1", "name": "Katherine Caneba",
             "last_move": "10/03/2026", "since": "2025-01-14"}


# ── The returning-customer line ────────────────────────────────────────────────

def test_a_returning_customer_is_marked_on_the_summary():
    text = confirm_module._summary(_intake(), {}, None, RETURNING)
    assert "returning customer" in text
    assert "10/03/2026" in text


def test_a_new_customer_gets_no_such_line():
    text = confirm_module._summary(_intake(), {}, None, None)
    assert "returning customer" not in text


def test_a_contact_with_no_recorded_move_falls_back_to_the_join_date():
    existing = {"id": "c1", "name": "Katherine Caneba", "last_move": "", "since": "2025-01-14"}
    text = confirm_module._summary(_intake(), {}, None, existing)
    assert "on file since 2025-01-14" in text


# ── The dangerous case ─────────────────────────────────────────────────────────

def test_a_phone_matching_a_different_name_is_mentioned_in_one_line():
    """
    Worth knowing which record gets updated, but usually harmless — a
    nickname, a spouse, a contact saved under an email. One FYI line, no alarm.
    """
    existing = dict(RETURNING, name="bosshouse777@gmail.com")
    text = confirm_module._summary(_intake(), {}, None, existing)
    line = "FYI: this updates an existing GoHighLevel contact (bosshouse777@gmail.com)."
    assert line in text
    assert "⚠️  That phone" not in text


@pytest.mark.parametrize("on_file", [
    "katherine caneba",       # GHL lower-cases some records
    "  Katherine   Caneba ",  # and pads others
    "KATHERINE CANEBA",
])
def test_casing_and_spacing_are_not_treated_as_a_different_person(on_file):
    """A false alarm on every returning customer would train people to ignore it."""
    text = confirm_module._summary(_intake(), {}, None, dict(RETURNING, name=on_file))
    assert "FYI: this updates" not in text


# ── The lookup itself ──────────────────────────────────────────────────────────

def test_lookup_failure_never_blocks_a_booking(monkeypatch):
    def explode(*a, **k):
        raise ghl.GHLError("search is down", 503, "")

    monkeypatch.setattr(confirm_module.ghl, "find_existing_contact", explode)
    assert confirm_module._find_existing_contact(_intake()) is None


def test_email_is_tried_when_the_phone_finds_nothing(monkeypatch):
    """
    The location deduplicates on phone AND email, so a customer who changed
    number still resolves to the same record — the summary must agree.
    """
    monkeypatch.setattr(ghl, "find_contact_by_phone", lambda _: None)
    monkeypatch.setattr(ghl, "find_contact_by_email",
                        lambda _: {"id": "c1", "firstName": "Katherine", "lastName": "Caneba"})
    found = ghl.find_existing_contact("(906) 369-4644", "katherineleenc@gmail.com")
    assert found["id"] == "c1"
    assert found["name"] == "Katherine Caneba"


def test_nobody_found_is_not_an_error(monkeypatch):
    monkeypatch.setattr(ghl, "find_contact_by_phone", lambda _: None)
    monkeypatch.setattr(ghl, "find_contact_by_email", lambda _: None)
    assert ghl.find_existing_contact("(818) 555-0000", "nobody@example.com") is None


# ── Date shapes GHL actually returns ───────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ("1790985600000", "10/03/2026"),   # epoch ms — from the SEARCH endpoint
    ("2026-10-03", "10/03/2026"),      # ISO — from a direct contact fetch
    ("", ""),
    ("not a date", ""),
])
def test_both_date_shapes_render(raw, expected):
    """
    The same field comes back two different ways depending on which endpoint
    returned the contact. Showing a dispatcher "1790985600000" would be worse
    than showing nothing.
    """
    assert ghl._as_us_date(raw) == expected


# ── The lookup must stay read-only ─────────────────────────────────────────────

def test_the_lookup_runs_above_the_interrupt_and_only_reads():
    """
    It sits before interrupt(), so it re-runs on every resume. A GET is fine
    there; anything else would fire repeatedly on a single approval.
    """
    import inspect
    from pathlib import Path

    source = Path(inspect.getfile(ghl)).read_text()
    body = source.split("def find_existing_contact")[1].split("\n@traceable")[0]
    for write in ("requests.post", "requests.put", "requests.delete"):
        assert write not in body


# ── A past move on file makes the lead source Previous Customer ───────────────

PAST = {"id": "c1", "name": "Katherine Caneba", "last_move": "03/14/2025", "since": "2025-01-14"}


def test_a_past_move_on_file_sets_previous_customer():
    intake = confirm_module._with_returning_source(_intake(), PAST)
    assert intake["source"] == "Previous Customer"


def test_the_summary_shows_the_source():
    intake = confirm_module._with_returning_source(_intake(), PAST)
    text = confirm_module._summary(intake, {}, None, PAST)
    assert "Source      Previous Customer" in text


def test_an_explicit_source_is_not_overwritten_by_the_lookup():
    intake = confirm_module._with_returning_source(_intake(source="Referral"), PAST)
    assert intake["source"] == "Referral"


@pytest.mark.parametrize("existing", [
    None,
    dict(PAST, last_move=""),                    # a lead that never booked
    dict(PAST, last_move="12/31/2099"),          # the move hasn't happened yet
    dict(PAST, name="Sarah Chen"),               # phone matched somebody else
])
def test_no_previous_move_means_no_source(existing):
    intake = confirm_module._with_returning_source(_intake(), existing)
    assert "source" not in intake


def test_approval_carries_the_filled_source_to_the_actions(monkeypatch):
    """The filled source must reach the calendar's Source: line, not just the summary."""
    monkeypatch.setattr(confirm_module, "_find_duplicate", lambda _: None)
    monkeypatch.setattr(confirm_module, "_find_existing_contact", lambda _: PAST)
    monkeypatch.setattr(confirm_module, "interrupt", lambda _: "yes")
    command = confirm_module.confirm({"intake": _intake()})
    assert command.update["intake"]["source"] == "Previous Customer"
