"""
Guards on GHL writes.

Both failures these prevent are silent at the API level — GHL returns 200 and
then does the wrong thing — so they have to be caught before the request.
"""

import pytest

from services import rates
from services.ghl import PICKLIST_VALUES, CustomField, validate_custom_fields


# ── The event-trigger checkbox ─────────────────────────────────────────────────

def test_refuses_to_check_create_google_event():
    """
    Checking this fires the GHL workflow that calls ghl_calendar_sync. Since the
    agent creates its own calendar event, allowing this would double-book every
    job.
    """
    with pytest.raises(ValueError, match="Create Google Event"):
        validate_custom_fields({CustomField.CREATE_GOOGLE_EVENT: "Click to create"})


def test_refuses_even_a_falsy_value_on_the_trigger():
    """The field is never safe to write, whatever the value looks like."""
    with pytest.raises(ValueError):
        validate_custom_fields({CustomField.CREATE_GOOGLE_EVENT: ""})


# ── Dropdown values ────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "near_miss",
    [
        "$115 cash / $125 card",   # stray space — was the real GHL value until 2026-08-07
        "$115 cash/$125 card",     # no space at all
        "$115 Cash /$125 Card",    # wrong case
        "115 cash /125 card",      # no dollar signs
    ],
)
def test_rejects_rate_strings_that_are_not_dropdown_options(near_miss):
    """
    The dangerous class of bug: right numbers, wrong punctuation. GHL returns
    200 and leaves the field empty, so the job books with no rate on it.
    """
    with pytest.raises(ValueError, match="not a valid option"):
        validate_custom_fields({CustomField.RATE: near_miss})


def test_accepts_the_exact_dropdown_options():
    for movers in rates.SUPPORTED_MOVER_COUNTS:
        validate_custom_fields({CustomField.RATE: rates.format_rate(movers)})


@pytest.mark.parametrize("movers", ["2", "3", "4", "5", "6"])
def test_accepts_valid_crew_sizes(movers):
    validate_custom_fields({CustomField.MOVERS: movers})


@pytest.mark.parametrize("movers", ["1", "7", "8", "two"])
def test_rejects_crew_sizes_ghl_does_not_offer(movers):
    with pytest.raises(ValueError):
        validate_custom_fields({CustomField.MOVERS: movers})


@pytest.mark.parametrize(
    "source", ["Yelp", "Google My Business", "Thumbtack", "Previous Customer",
               "Local Service Ads", "Referral"]
)
def test_accepts_real_lead_sources(source):
    validate_custom_fields({CustomField.ORIGIN: source})


def test_rejects_an_invented_lead_source():
    """'Google' is not an option — 'Google My Business' is."""
    with pytest.raises(ValueError):
        validate_custom_fields({CustomField.ORIGIN: "Google"})


# ── Pass-through ───────────────────────────────────────────────────────────────

def test_free_text_fields_are_unconstrained():
    validate_custom_fields({
        CustomField.MOVING_FROM: "412 N Maple Ave, Burbank CA 91505",
        CustomField.JOB_NOTES: "anything at all",
        CustomField.ARRIVAL_TIME: "8-9am",
    })


def test_empty_values_are_allowed_on_dropdowns():
    """Leaving a field blank is legitimate — only wrong values are rejected."""
    validate_custom_fields({CustomField.RATE: "", CustomField.ORIGIN: None})


def test_empty_payload_is_fine():
    validate_custom_fields({})


def test_every_picklist_id_is_a_real_field():
    """Catches a typo'd id in PICKLIST_VALUES that would disable a guard."""
    known = {v for k, v in vars(CustomField).items() if not k.startswith("_") and isinstance(v, str)}
    assert set(PICKLIST_VALUES).issubset(known)
