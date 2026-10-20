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
    f = parse_fees("- Gas fee $50\n- Piano $150\n- Discount 10%")
    assert (f.gas_fee, f.heavy_item_fee, f.discount_percent) == (50, 150, 10)
    assert f.unclear == []


def test_adds_up_several_heavy_items():
    assert parse_fees("- piano $150, safe $75").heavy_item_fee == 225


def test_cents_and_thousands():
    assert parse_fees("- Gas fee: $1,200.50").gas_fee == 1200.5


def test_dollar_sign_after_the_number():
    assert parse_fees("- gas 50$").gas_fee == 50


def test_percent_off_is_a_discount():
    assert parse_fees("- 15% off, returning customer").discount_percent == 15


def test_dollar_discount_is_reported_not_converted():
    # The app keeps the discount as a percent; $20 can't be turned into one.
    f = parse_fees("- $20 off")
    assert f.discount_percent is None
    assert f.unclear == ["$20 off"]


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


def test_two_fees_joined_by_and_are_both_read():
    # The agent often rewrites the notes into one sentence like this.
    f = parse_fees("- $150 heavy fee and $70 gas fee mentioned by staff for this job.")
    assert (f.heavy_item_fee, f.gas_fee) == (150, 70)
    assert f.unclear == []


def test_gas_plus_piano():
    f = parse_fees("gas $50 plus piano $150 & 10% discount")
    assert (f.gas_fee, f.heavy_item_fee, f.discount_percent) == (50, 150, 10)


def test_one_amount_for_two_kinds_is_reported_not_guessed():
    f = parse_fees("- gas and piano $200")
    assert not f.any()
    assert f.unclear == ["gas and piano $200"]


def test_percent_discount():
    f = parse_fees("- 10% discount")
    assert f.discount_percent == 10
    assert f.unclear == []


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
    assert payload["discount_percent"] is None
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
    intake["job_notes"] = "- gas and piano $200"
    text = confirm._summary(intake, {}, None)
    assert "Fee not read" in text


def test_summary_quiet_when_the_app_is_not_connected(intake, monkeypatch):
    monkeypatch.delenv("JOBS_APP_WEBHOOK_URL", raising=False)
    monkeypatch.delenv("JOBS_APP_WEBHOOK_SECRET", raising=False)
    assert "App fees" not in confirm._summary(intake, {}, None)


def test_payload_sends_discount_as_percent_number(intake):
    intake["job_notes"] = "- 10% discount"
    assert actions.app_job_payload(intake)["discount_percent"] == 10


def test_summary_shows_percent(intake, monkeypatch):
    monkeypatch.setenv("JOBS_APP_WEBHOOK_URL", "https://script.google.com/macros/s/x/exec")
    monkeypatch.setenv("JOBS_APP_WEBHOOK_SECRET", "s3cret")
    intake["job_notes"] = "- Gas fee $50\n- 10% off"
    assert "App fees    Gas $50 · Discount 10%" in confirm._summary(intake, {}, None)
