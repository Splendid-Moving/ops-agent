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
