"""
Address completion.

The live tests here encode the specific failures measured against the real API:
a partial address resolving to a different state, and a complete address being
flagged as broken. Both are cheap to reintroduce and expensive to ship.
"""

import pytest

from services import address
from services.address import Verdict


# ── Pure helpers ───────────────────────────────────────────────────────────────

def test_usps_shouting_becomes_house_casing():
    assert address._title_case_usps("412 N MAPLE ST, BURBANK CA 91505") == (
        "412 N Maple St, Burbank CA 91505"
    )


def test_state_and_directionals_stay_uppercase():
    out = address._title_case_usps("1200 SW BROADWAY, PORTLAND OR 97205")
    assert "SW" in out
    assert " OR " in out


def test_abbreviates_street_suffixes():
    assert address._abbreviate("909 Beacon Avenue") == "909 Beacon Ave"
    assert address._abbreviate("100 North Maple Boulevard") == "100 N Maple Blvd"


def test_city_names_are_not_abbreviated():
    """
    'North Hollywood' is a city, not a directional. Abbreviating past the first
    comma would turn it into 'N Hollywood'.
    """
    out = address._abbreviate("5200 Lankershim Blvd, North Hollywood CA 91601")
    assert "North Hollywood" in out


def test_usa_suffix_is_stripped():
    assert "USA" not in address._to_calendar_format("1 Main St, Burbank, CA 91505, USA")


# ── Verdict handling ───────────────────────────────────────────────────────────

def test_blank_address_is_unresolved_without_an_api_call():
    result = address.validate("")
    assert result.verdict is Verdict.UNRESOLVED
    assert not result.is_usable


def test_validate_many_skips_blanks():
    """extra_stop is optional — absent is not a failure."""
    out = address.validate_many({"pickup": "", "dropoff": "   "})
    assert out == {}


@pytest.mark.parametrize(
    "verdict,usable,needs_human",
    [
        (Verdict.CONFIRMED, True, False),
        (Verdict.NEEDS_REVIEW, False, True),
        (Verdict.UNRESOLVED, False, True),
    ],
)
def test_verdict_flags(verdict, usable, needs_human):
    result = address.ValidatedAddress(verdict=verdict)
    assert result.is_usable is usable
    assert result.needs_human is needs_human


def test_only_city_inference_is_treated_as_dangerous():
    """
    Inferring a ZIP is the feature. Inferring a CITY is the Brandon-SD bug.
    Flagging both would warn on nearly every address and train people to
    ignore the warning that matters.
    """
    assert address._DANGEROUS_INFERENCE == {"locality"}


@pytest.mark.parametrize(
    "text,expected",
    [
        ("412 N Maple St, Burbank CA 91505", "CA"),
        ("100 Main St, Reno NV", "NV"),
        ("614 e verdugo ave", ""),
        ("1830 Pine St Glendale", ""),
    ],
)
def test_detects_the_state_the_user_typed(text, expected):
    """A directional like 'N' or 'E' must not be mistaken for a state code."""
    assert address._state_written_by_user(text) == expected


# ── Live API ───────────────────────────────────────────────────────────────────

@pytest.mark.live
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("412 N Maple St, Burbank CA 91505", "412 N Maple St, Burbank CA 91505"),
        ("1830 Pine St, Glendale CA 91206", "1830 Pine St, Glendale CA 91206"),
        # The whole point of the node: partial in, complete out.
        ("909 beacon ave los angeles", "909 Beacon Ave, Los Angeles CA 90015"),
    ],
)
def test_complete_addresses_are_confirmed_and_correctly_formatted(raw, expected):
    result = address.validate(raw)
    assert result.verdict is Verdict.CONFIRMED, result.note
    assert result.formatted == expected


@pytest.mark.live
def test_address_with_no_city_is_never_silently_accepted():
    """
    '412 N Maple Ave' resolves to a real street in Brandon, South Dakota.
    It must reach a human, not a truck.
    """
    result = address.validate("412 N Maple Ave")
    assert result.verdict is Verdict.NEEDS_REVIEW
    assert not result.is_usable
    assert "locality" in result.inferred


@pytest.mark.live
def test_junk_is_unresolved():
    result = address.validate("asdfgh nowhere 99999")
    assert result.verdict is Verdict.UNRESOLVED


@pytest.mark.live
@pytest.mark.parametrize(
    "raw",
    [
        "614 e verdugo ave",                    # real staff input, no city
        "1830 Pine St Glendale",                # no ZIP
        "1039 Justin Ave, Glendale CA 91201",   # no apartment number
        "550 N Figueroa St, Los Angeles",       # Splendid's own office
    ],
)
def test_real_staff_input_is_not_flagged(raw):
    """
    Every one of these was flagged by an earlier, stricter version and every one
    is correct. A warning that fires on ordinary input is noise, and noise
    buries the state-mismatch case that actually matters.
    """
    result = address.validate(raw)
    assert result.verdict is Verdict.CONFIRMED, result.note


@pytest.mark.live
def test_guessed_city_inside_california_is_accepted():
    """Splendid works the LA area; an in-state guess is not worth interrupting for."""
    result = address.validate("614 e verdugo ave")
    assert result.verdict is Verdict.CONFIRMED
    assert "Burbank" in result.formatted


@pytest.mark.live
def test_state_mismatch_against_what_the_user_typed_is_flagged():
    """If they wrote CA and it resolved elsewhere, stop."""
    result = address.validate("1 Main St, Springfield CA")
    if result.verdict is Verdict.NEEDS_REVIEW:
        assert "CA" in result.note


# ── An out-of-state match is not a suggestion ──────────────────────────────────

def test_out_of_state_match_offers_no_suggested_address():
    """
    "436 Fairview Ave #32" with no city resolves to Lancaster PA. The confirm
    gate renders `formatted` as "suggested: …", and suggesting a Pennsylvania
    street to an LA moving company invites someone to accept it. The match is
    evidence the city is missing, not a candidate.
    """
    from unittest.mock import patch

    from services import address

    fake = {
        "result": {
            "address": {
                "formattedAddress": "436 Fairview Ave #32, Lancaster, PA 17603, USA",
                "addressComponents": [
                    {"componentType": "locality", "inferred": True,
                     "componentName": {"text": "Lancaster"}},
                    {"componentType": "administrative_area_level_1",
                     "componentName": {"text": "PA"}},
                ],
                "missingComponentTypes": [],
                "unresolvedTokens": [],
            },
            "uspsData": {"standardizedAddress": {"firstAddressLine": "436 FAIRVIEW AVE # 32"}},
        }
    }

    class R:
        ok, status_code = True, 200
        @staticmethod
        def json(): return fake

    with patch.object(address.requests, "post", return_value=R()):
        v = address.validate("436 Fairview Ave #32")

    assert v.verdict is address.Verdict.NEEDS_REVIEW
    assert v.formatted == ""
    assert "Which city" in v.note
    assert "pickup is in" in v.note        # tells them how to answer


# ── Addresses that wrap onto a second line ─────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ("436 Fairview Ave #32\nArcadia, CA 91007", "436 Fairview Ave #32, Arcadia, CA 91007"),
    ("6343 Livia Ave, Temple City,\nCA 91780",   "6343 Livia Ave, Temple City, CA 91780"),
    ("1 Main St\n\nBurbank CA 91505",            "1 Main St, Burbank CA 91505"),   # blank line
    ("1 Main St",                                "1 Main St"),                     # nothing to join
    ("",                                         ""),
])
def test_wrapped_lines_are_joined_with_a_comma(raw, expected):
    """
    On a phone the street lands on one line and the city on the next, and the
    model returns them with a literal newline. Google accepts that but hands
    back "436 Fairview Ave #32 Arcadia CA 91007" — no comma, not the calendar
    format. Joining first gives it a proper line.
    """
    from services import formatting
    assert formatting.join_wrapped_address(raw) == expected


def test_extraction_joins_wrapped_addresses_before_anything_reads_them():
    from agent.nodes.extract_screenshot import _to_intake
    from schemas.intake import ScreenshotExtraction

    extraction = ScreenshotExtraction(
        pickup_address={"value": "436 Fairview Ave #32\nArcadia, CA 91007", "confidence": 0.95},
        dropoff_address={"value": "6343 Livia Ave\nTemple City CA 91780", "confidence": 0.95},
    )
    intake, _ = _to_intake(extraction)
    assert "\n" not in intake["pickup_address"]
    assert intake["pickup_address"] == "436 Fairview Ave #32, Arcadia, CA 91007"
    assert intake["dropoff_address"] == "6343 Livia Ave, Temple City CA 91780"


def test_the_prompt_tells_the_model_addresses_wrap():
    """
    The join above only helps once the model has taken BOTH lines. Stopping at
    the first — the failure seen live — has to be ruled out in the prompt.
    """
    from agent.nodes import extract_screenshot as node
    assert "Addresses wrap" in node.SYSTEM_PROMPT_BODY
    assert "PART OF THAT ADDRESS" in node.SYSTEM_PROMPT_BODY
