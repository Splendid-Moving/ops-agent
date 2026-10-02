"""
The lead source is a six-option dropdown, and the model reaches for others.

"SMS" and "Text" turn up whenever the screenshot is a messaging app — the model
reports how the customer is talking to us rather than where they came from. GHL
accepts an invalid option and silently discards it, and the value meanwhile goes
on the calendar's Source: line, which four other repos read.
"""

import pytest

from agent.nodes.extract_screenshot import _to_intake
from schemas.intake import ScreenshotExtraction
from services.ghl import PICKLIST_VALUES, CustomField


def _with_source(value: str) -> dict:
    extraction = ScreenshotExtraction(source={"value": value, "confidence": 0.95})
    intake, _ = _to_intake(extraction)
    return intake


@pytest.mark.parametrize("invented", ["SMS", "Text message", "Email", "Website",
                                      "WhatsApp", "Phone call", "Facebook"])
def test_an_invented_source_is_dropped(invented):
    """Blank beats wrong: an empty Source: line is honest, a made-up one is not."""
    assert "source" not in _with_source(invented)


@pytest.mark.parametrize("valid", sorted(PICKLIST_VALUES[CustomField.ORIGIN]))
def test_every_real_dropdown_option_survives(valid):
    assert _with_source(valid)["source"] == valid


@pytest.mark.parametrize("alias,expected", [
    ("yelp", "Yelp"),
    ("LSA", "Local Service Ads"),
    ("gmb", "Google My Business"),
    ("previous costumer", "Previous Customer"),   # the live misspelling
])
def test_known_aliases_still_resolve(alias, expected):
    assert _with_source(alias)["source"] == expected


def test_the_prompt_names_the_six_options():
    """
    The code guard alone would silently drop a source on every messaging-app
    screenshot. Telling the model the allowed set is what makes it right.
    """
    from agent.nodes import extract_screenshot as node
    for option in PICKLIST_VALUES[CustomField.ORIGIN]:
        assert option in node.SYSTEM_PROMPT_BODY


# ── Previous Customer: a source, never a note ─────────────────────────────────

@pytest.mark.parametrize("line", [
    "Previous customer",
    "returning client",
    "Repeat customer - moved them in 2024",
    "They used us before",
    "we moved them before",
    "previous costumer",
])
def test_a_previous_customer_note_becomes_the_source(line):
    extraction = ScreenshotExtraction(
        notes={"value": f"- {line}\n- Piano on 2nd floor", "confidence": 0.9})
    intake, confidence = _to_intake(extraction)
    assert intake["source"] == "Previous Customer"
    assert confidence["source"] == 1.0
    assert intake["job_notes"] == "- Piano on 2nd floor"


def test_previous_customer_overrides_the_app_the_message_came_through():
    """A past customer messaging on Yelp is not a Yelp lead."""
    extraction = ScreenshotExtraction(
        source={"value": "Yelp", "confidence": 0.9},
        notes={"value": "- Returning customer", "confidence": 0.9},
    )
    intake, _ = _to_intake(extraction)
    assert intake["source"] == "Previous Customer"
    assert "job_notes" not in intake


def test_a_note_that_was_only_previous_customer_leaves_notes_unset():
    extraction = ScreenshotExtraction(notes={"value": "previous customer", "confidence": 0.9})
    intake, _ = _to_intake(extraction)
    assert "job_notes" not in intake
    assert intake["source"] == "Previous Customer"


@pytest.mark.parametrize("typed", [
    "previous customer, Friday 8-9am",
    "Returning client — 3 guys",
    "she used us before",
])
def test_the_dispatcher_typing_it_sets_the_source(typed):
    from agent.nodes.extract_screenshot import _apply_typed_previous_customer
    intake, confidence = {"source": "Yelp"}, {}
    _apply_typed_previous_customer(intake, confidence, typed)
    assert intake["source"] == "Previous Customer"


def test_ordinary_text_does_not_set_the_source():
    from agent.nodes.extract_screenshot import _apply_typed_previous_customer
    intake, confidence = {}, {}
    _apply_typed_previous_customer(intake, confidence, "Friday 8-9am, 3 guys, $50 gas fee")
    assert "source" not in intake


def test_the_prompt_says_previous_customer_is_never_a_note():
    from agent.nodes import extract_screenshot as node
    assert "It is a SOURCE, never a note" in node.SYSTEM_PROMPT_BODY


# ── …and from a reply to a question ───────────────────────────────────────────

def _apply_reply(intake: dict, **fields) -> tuple[dict, dict]:
    from agent.nodes.ask_missing import ParsedReply, _apply
    updates = _apply(intake, ParsedReply(**fields))
    return intake, updates


def test_a_reply_naming_the_source_sets_it():
    intake, _ = _apply_reply({}, source="previous customer")
    assert intake["source"] == "Previous Customer"


def test_a_reply_note_saying_previous_customer_moves_to_the_source():
    intake, _ = _apply_reply({}, job_notes="- Previous customer\n- Gate code 1234")
    assert intake["source"] == "Previous Customer"
    assert intake["job_notes"] == "- Gate code 1234"
    assert intake["notes_asked"] is True


def test_a_reply_with_an_invented_source_leaves_the_existing_one():
    intake, updates = _apply_reply({"source": "Yelp"}, source="Text message")
    assert intake["source"] == "Yelp"
    assert "source" not in updates
