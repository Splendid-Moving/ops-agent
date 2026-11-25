"""
Crews bigger than the standard 2-4: the dispatcher types the rate (cash / card)
and, above 4 movers, the number of trucks. Two or more trucks puts
"(N trucks)" in the calendar title and a Trucks row in the customer email.
"""

from datetime import datetime, timedelta

import pytest

from agent.nodes import actions, confirm
from agent.state import new_ledger
from schemas import checklist as cl
from schemas import email_template
from services import calendar, ghl, rates
from services.calendar import LA_TZ
from services.ghl import CustomField


@pytest.fixture
def intake():
    return {
        "full_name": "Sarah Chen",
        "email": "sarah@example.com",
        "phone": "(818) 555-0142",
        "pickup_address": "412 N Maple St, Burbank CA 91505",
        "dropoff_address": "1830 Pine St, Glendale CA 91206",
        "move_date": (datetime.now(LA_TZ) + timedelta(days=10)).strftime("%m/%d/%Y"),
        "arrival_time": "8-9am",
        "movers": "6",
        "rate": "$250 cash /$270 card",
        "trucks": "2",
        "job_notes": "- Third floor walkup",
        "notes_asked": True,
    }


# ── Checklist ──────────────────────────────────────────────────────────────────

def _missing(intake: dict) -> set[str]:
    return {spec.name for spec in cl.evaluate(intake).missing}


def test_standard_crew_asks_for_neither_rate_nor_trucks(intake):
    intake.update(movers="3")
    del intake["rate"], intake["trucks"]
    assert cl.evaluate(intake).is_complete


def test_five_movers_asks_for_rate_and_trucks(intake):
    intake.update(movers="5")
    del intake["rate"], intake["trucks"]
    assert _missing(intake) == {"rate", "trucks"}
    questions = " ".join(cl.evaluate(intake).all_questions())
    assert "cash / card" in questions and "trucks" in questions


def test_big_labor_job_asks_for_rate_but_not_trucks(intake):
    intake.update(movers="8", is_labor=True)
    del intake["rate"], intake["trucks"], intake["dropoff_address"]
    assert _missing(intake) == {"rate"}


def test_unreadable_rate_is_rejected(intake):
    intake["rate"] = "two fifty"
    assert "rate" in cl.evaluate(intake).invalid


def test_complete_big_job_passes(intake):
    assert cl.evaluate(intake).is_complete


@pytest.mark.parametrize("typed", ["250/270", "250 / 270", "$250/$270", "250, 270", "$250 cash /$270 card"])
def test_rate_is_normalised_to_the_ghl_style(typed):
    assert rates.format_custom_rate(typed) == "$250 cash /$270 card"


# ── Calendar ───────────────────────────────────────────────────────────────────

@pytest.fixture
def captured(monkeypatch):
    seen = {}
    monkeypatch.setattr(calendar, "create_event",
                        lambda **kw: seen.update(kw) or {"event_id": "e1"})
    monkeypatch.setattr(actions.maps, "get_distance", lambda *a, **k: "8.2 miles")
    return seen


def test_two_trucks_go_in_the_title(intake, captured):
    actions.act_calendar_event({"intake": intake, "ledger": new_ledger()})
    assert captured["title"] == "Sarah Chen (2 trucks)"
    assert "Rate: $250 cash /$270 card" in captured["description"]
    assert "Movers: 6" in captured["description"]
    assert "Notes:\n- 2 trucks\n- Third floor walkup" in captured["description"]


def test_one_truck_keeps_the_plain_title(intake, captured):
    intake["trucks"] = "1"
    actions.act_calendar_event({"intake": intake, "ledger": new_ledger()})
    assert captured["title"] == "Sarah Chen"
    assert "trucks" not in captured["description"]


def test_no_new_description_field_is_added(intake, captured):
    """The description is a wire format: a 'Trucks:' line would leak downstream."""
    actions.act_calendar_event({"intake": intake, "ledger": new_ledger()})
    assert "Trucks:" not in captured["description"]


# ── GHL ────────────────────────────────────────────────────────────────────────

@pytest.fixture
def contact_fields(monkeypatch):
    seen = {}
    def fake(**kw):
        ghl.validate_custom_fields(kw["custom_fields"])  # what the real call enforces
        seen.update(kw["custom_fields"])
        return {"contact_id": "c1", "created": True}
    monkeypatch.setattr(ghl, "upsert_contact", fake)
    return seen


def test_ghl_gets_movers_and_trucks_but_a_blank_custom_rate(intake, contact_fields):
    actions.act_upsert_contact({"intake": intake, "ledger": new_ledger()})
    assert contact_fields[CustomField.MOVERS] == "6"
    assert contact_fields[CustomField.TRUCKS] == "2"
    assert contact_fields[CustomField.RATE] == ""


def test_crew_bigger_than_the_ghl_dropdown_still_books(intake, contact_fields):
    intake.update(movers="10", trucks="4")
    update = actions.act_upsert_contact({"intake": intake, "ledger": new_ledger()})
    assert update["ledger"][actions.ACTION_CONTACT]["status"] == "success"
    assert contact_fields[CustomField.MOVERS] == ""
    assert contact_fields[CustomField.TRUCKS] == ""


# ── Customer email and confirm summary ─────────────────────────────────────────

def test_email_shows_rate_movers_and_trucks(intake):
    body = email_template.html(intake)
    assert "$250 cash /$270 card" in body
    assert "Trucks" in body
    assert "<strong>6</strong>" in body and "<strong>2</strong>" in body


def test_email_has_no_trucks_row_for_one_truck(intake):
    intake["trucks"] = "1"
    assert "Trucks" not in email_template.html(intake)


def test_confirm_summary_shows_everything(intake):
    text = confirm._summary(intake, {}, None)
    assert "6 movers — $250 cash /$270 card (custom)" in text
    assert "Trucks      2" in text
    assert "Sarah Chen (2 trucks)" in text
    assert "Rate dropdown has no option" in text
