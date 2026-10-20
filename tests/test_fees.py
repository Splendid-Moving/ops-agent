"""
Fees read from the job notes for the crew app's bill.

The failure that matters is a WRONG number on a customer's bill, so most of
these pin what must NOT become a fee.
"""

from datetime import datetime, timedelta

import pytest

from agent.nodes import actions, confirm
from services.calendar import LA_TZ
from services.fees import parse_fees


def test_reads_each_kind_from_its_own_line():
    f = parse_fees("- Gas fee $50\n- Piano $150\n- Discount $20")
    assert (f.gas_fee, f.heavy_item_fee, f.discount) == (50, 150, 20)
    assert f.unclear == []


def test_adds_up_several_heavy_items():
    assert parse_fees("- piano $150, safe $75").heavy_item_fee == 225


def test_cents_and_thousands():
    assert parse_fees("- Gas fee: $1,200.50").gas_fee == 1200.5


def test_dollar_sign_after_the_number():
    assert parse_fees("- gas 50$").gas_fee == 50


def test_amount_off_is_a_discount():
    assert parse_fees("- $20 off, returning customer").discount == 20


@pytest.mark.parametrize("notes", [
    "- 3rd floor walkup",
    "- 2 trucks needed",
    "- gas fee 50",                 # no $ — never guessed
    "- drop off keys at the office",
    "- heavy sofa, 3 flights",
    "",
])
def test_nothing_without_a_dollar_amount(notes):
    f = parse_fees(notes)
    assert not f.any() and f.unclear == []


def test_two_kinds_on_one_line_are_reported_not_guessed():
    f = parse_fees("- gas $50 and piano $150")
    assert not f.any()
    assert f.unclear == ["gas $50 and piano $150"]


def test_percent_discount_is_reported_not_converted():
    f = parse_fees("- 10% discount")
    assert f.discount is None
    assert f.unclear == ["10% discount"]


@pytest.fixture
def intake():
    return {
        "full_name": "Sarah Chen",
        "phone": "+1(818)555-0142",
        "email": "sarah@example.com",
        "pickup_address": "412 N Maple St, Burbank CA 91505",
        "dropoff_address": "1830 Pine St, Glendale CA 91206",
        "move_date": (datetime.now(LA_TZ) + timedelta(days=10)).strftime("%m/%d/%Y"),
        "arrival_time": "8-9am",
        "movers": "3",
        "job_notes": "- Gas fee $50\n- Piano $150\n- Third floor",
    }


def test_payload_carries_the_fees_and_keeps_the_notes(intake):
    payload = actions.app_job_payload(intake)
    assert payload["gas_fee"] == 50
    assert payload["heavy_item_fee"] == 150
    assert payload["discount"] is None
    assert "Gas fee $50" in payload["job_notes"]


def test_payload_without_fees_sends_none(intake):
    intake["job_notes"] = "- Third floor"
    payload = actions.app_job_payload(intake)
    assert payload["gas_fee"] is None and payload["heavy_item_fee"] is None


def test_summary_shows_fees_when_the_app_is_connected(intake, monkeypatch):
    monkeypatch.setenv("JOBS_APP_WEBHOOK_URL", "https://script.google.com/macros/s/x/exec")
    monkeypatch.setenv("JOBS_APP_WEBHOOK_SECRET", "s3cret")
    text = confirm._summary(intake, {}, None)
    assert "App fees    Gas $50 · Heavy item $150" in text


def test_summary_flags_a_fee_it_could_not_read(intake, monkeypatch):
    monkeypatch.setenv("JOBS_APP_WEBHOOK_URL", "https://script.google.com/macros/s/x/exec")
    monkeypatch.setenv("JOBS_APP_WEBHOOK_SECRET", "s3cret")
    intake["job_notes"] = "- gas $50 and piano $150"
    text = confirm._summary(intake, {}, None)
    assert "Fee not read" in text


def test_summary_quiet_when_the_app_is_not_connected(intake, monkeypatch):
    monkeypatch.delenv("JOBS_APP_WEBHOOK_URL", raising=False)
    monkeypatch.delenv("JOBS_APP_WEBHOOK_SECRET", raising=False)
    assert "App fees" not in confirm._summary(intake, {}, None)
