"""
The confirmation email is the customer-facing record of the booking terms.

Tests here guard the two things that would embarrass the company: an
unresolved merge tag reaching a customer, and a term silently going missing.
"""

import pytest

from schemas import email_template as t

FULL_MOVE = {
    "full_name": "Aemond Targareyan",
    "phone": "+1(818)505-4576",
    "move_date": "08/14/2026",
    "arrival_time": "9-10am",
    "movers": "3",
    "pickup_address": "527 E Cypress Ave, Burbank CA 91501",
    "dropoff_address": "412 N Maple Ave, Burbank CA 91505",
    "extra_stop": "",
    "is_labor": False,
}
LABOR_ONLY = {**FULL_MOVE, "is_labor": True, "dropoff_address": ""}


def test_no_unresolved_merge_tags_reach_the_customer():
    """
    The GHL snippet uses {{contact.first_name}}, which GHL substitutes only
    when GHL sends it. This agent posts raw html, so a leftover tag would be
    delivered literally.
    """
    for intake in (FULL_MOVE, LABOR_ONLY):
        body = t.html(intake)
        assert "{{" not in body
        assert "}}" not in body


def test_the_booking_details_are_actually_present():
    body = t.html(FULL_MOVE)

    assert "Aemond Targareyan" in body
    assert "08/14/2026" in body
    assert "9-10am" in body
    assert "527 E Cypress Ave" in body
    assert "412 N Maple Ave" in body
    assert "145 cash" in body          # rate derived from crew size


def test_every_term_survives():
    """
    These are the clauses customers dispute later, and this email is the
    record. Losing one silently is the failure that matters.
    """
    body = t.html(FULL_MOVE)

    assert body.count("<li") == len(t.TERMS)
    for phrase in (
        "72 hours in advance",
        "minimum labor charge of 3 hours",
        "Double Drive Time",
        "parking ticket",
        "does not assume responsibility for the safe transportation of plants",
        "DO NOT accept checks, Zelle, Venmo",
        "California Public Utility Commission",
        "$0.60 cents per pound",
    ):
        assert phrase in body, f"missing term: {phrase}"


def test_labor_only_has_no_dropoff_row():
    """A labor job has no destination — a blank 'To:' looks like an omission."""
    body = t.html(LABOR_ONLY)

    assert "412 N Maple" not in body
    assert "Labor only" in body


def test_absent_extra_stop_is_omitted_not_blank():
    assert "Extra stop" not in t.html(FULL_MOVE)


def test_extra_stop_appears_when_there_is_one():
    body = t.html({**FULL_MOVE, "extra_stop": "900 S Brand Blvd, Glendale CA 91204"})

    assert "Extra stop" in body
    assert "900 S Brand Blvd" in body


def test_deposit_comes_from_config_not_hardcoded(monkeypatch):
    monkeypatch.setenv("DEPOSIT_AMOUNT", "75")
    body = t.html(FULL_MOVE)

    assert "$75 deposit" in body
    assert "$50 deposit" not in body


def test_subject_names_the_company_and_date():
    subject = t.subject(FULL_MOVE)

    assert "Splendid Moving" in subject
    assert "08/14/2026" in subject


@pytest.mark.parametrize("missing", ["full_name", "extra_stop", "dropoff_address"])
def test_missing_optional_fields_do_not_break_rendering(missing):
    """A half-filled intake must still produce a sendable email."""
    intake = {k: v for k, v in FULL_MOVE.items() if k != missing}
    body = t.html(intake)

    assert "BOOKING CONFIRMATION" in body
    assert "None" not in body
