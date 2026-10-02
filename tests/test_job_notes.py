"""
Job notes must carry what the other fields cannot, and nothing else.

The notes print on the calendar event directly beneath the date, the addresses,
the crew size and the rate. A note restating those is noise a dispatcher reads
past on every job — and the agent was producing exactly that, twice over,
because two near-identical schema fields were both filled and then joined.
"""

import pytest

from schemas.intake import ScreenshotExtraction
from agent.nodes.extract_screenshot import _to_intake


def _extraction(**fields) -> ScreenshotExtraction:
    return ScreenshotExtraction(**fields)


def test_notes_come_from_one_field_only():
    """
    The second field is gone. While both existed the model filled both and the
    join produced "$60 gas fee. | Service listed as local_move. $60 gas fee
    noted. Move scheduled for..." — the fee twice, then a prose restatement.
    """
    assert "overall_notes" not in ScreenshotExtraction.model_fields


def test_a_real_note_survives_as_a_bullet():
    extraction = _extraction(notes={"value": "$60 gas fee", "confidence": 0.95})
    intake, _ = _to_intake(extraction)
    assert intake["job_notes"] == "- $60 gas fee"


def test_no_separator_can_appear_in_a_single_note():
    """The " | " join is what stitched the duplicate together."""
    extraction = _extraction(notes={"value": "$60 gas fee", "confidence": 0.95})
    intake, _ = _to_intake(extraction)
    assert "|" not in intake["job_notes"]


def test_an_absent_note_leaves_the_field_unset():
    """
    Unset matters: the checklist still asks about notes, and an empty string
    would read as "asked and answered none".
    """
    intake, _ = _to_intake(_extraction())
    assert "job_notes" not in intake


def test_a_low_confidence_note_is_dropped():
    """Below the threshold it is treated as missing, like every other field."""
    extraction = _extraction(notes={"value": "maybe stairs?", "confidence": 0.2})
    intake, _ = _to_intake(extraction)
    assert "job_notes" not in intake


def test_whitespace_only_notes_do_not_count():
    extraction = _extraction(notes={"value": "   ", "confidence": 0.95})
    intake, _ = _to_intake(extraction)
    assert "job_notes" not in intake


def test_the_prompt_forbids_restating_other_fields():
    """
    The rule is the fix — the schema change alone only stops the duplication,
    not the prose summary. Pinned because a prompt edit could quietly drop it.
    """
    from agent.nodes import extract_screenshot as node
    assert "NEVER restate a value that already has its own field" in node.SYSTEM_PROMPT_BODY


# ── The bullet format ──────────────────────────────────────────────────────────

def test_several_facts_each_get_their_own_line():
    extraction = _extraction(
        notes={"value": "- $60 gas fee\n- Third floor walk-up", "confidence": 0.95}
    )
    intake, _ = _to_intake(extraction)
    assert intake["job_notes"] == "- $60 gas fee\n- Third floor walk-up"


def test_bullets_stop_notes_leaking_into_the_calendar_as_fields():
    """
    The real reason for the format. Notes sit on the lines after "Notes:" in
    the event description, which four other repos parse with ^([A-Za-z ]+):.
    A bare "Parking: tight" becomes a phantom `parking` field in their
    pipelines; a leading "- " cannot match.
    """
    from services import calendar, formatting

    raw = "Parking: tight\nStairs: 3 flights"
    leaky = calendar.build_description(
        customer="X", phone="p", date="d", from_address="a", to_address="b", notes=raw)
    safe = calendar.build_description(
        customer="X", phone="p", date="d", from_address="a", to_address="b",
        notes=formatting.format_notes(raw))

    assert "parking" in calendar.parse_description(leaky)
    assert "parking" not in calendar.parse_description(safe)


@pytest.mark.parametrize("raw,expected", [
    ("$60 gas fee", "- $60 gas fee"),
    ("- already bulleted", "- already bulleted"),          # not doubled up
    ("• unicode bullet", "- unicode bullet"),
    ("* asterisk", "- asterisk"),
    ("a | b", "- a\n- b"),                                  # the old join format
    ("a\n\n\nb", "- a\n- b"),                               # blank lines dropped
    ("", ""),
    ("   ", ""),
])
def test_note_shapes_normalise(raw, expected):
    from services import formatting
    assert formatting.format_notes(raw) == expected


def test_an_address_in_a_note_is_not_split_on_its_comma():
    """Splitting on commas would cut "1561 W 223rd St, Torrance" in half."""
    from services import formatting
    assert formatting.format_notes("drop at 1561 W 223rd St, Torrance") == \
        "- drop at 1561 W 223rd St, Torrance"


# ── Notes may come from the screenshot, but only what helps the move ─────────

def test_screenshot_notes_are_kept_when_nothing_was_typed():
    """
    The customer conversation is where stairs, parking and special requests
    turn up. Keeping them saves the dispatcher retyping what is on screen.
    """
    extraction = _extraction(notes={"value": "- 3rd floor walk-up\n- Street parking only",
                                    "confidence": 0.9})
    intake, _ = _to_intake(extraction)
    assert intake["job_notes"] == "- 3rd floor walk-up\n- Street parking only"


def test_the_prompt_allows_screenshot_notes_and_lists_what_qualifies():
    from agent.nodes import extract_screenshot as node
    body = node.SYSTEM_PROMPT_BODY
    assert "ONLY what the staff member typed" not in body
    for kind in ("stairs", "parking", "special requests"):
        assert kind in body


def test_the_prompt_keeps_the_rate_out_of_notes():
    from agent.nodes import extract_screenshot as node
    assert "the hourly rate" in node.SYSTEM_PROMPT_BODY


@pytest.mark.parametrize("rate_line", [
    "$145 cash / $155 card",
    "Rate $125/hr",
    "Quoted 145 per hour",
    "$115 an hour",
    "hourly rate agreed",
])
def test_rate_lines_are_stripped_in_code(rate_line):
    """The prompt says no rates; this makes it true when the model disagrees."""
    extraction = _extraction(notes={"value": f"- {rate_line}\n- Piano", "confidence": 0.9})
    intake, _ = _to_intake(extraction)
    assert intake["job_notes"] == "- Piano"


@pytest.mark.parametrize("fee", ["$50 gas fee", "$75 stair fee", "$40 for packing materials"])
def test_fees_are_not_mistaken_for_rates(fee):
    from services import formatting
    assert formatting.clean_job_notes(fee) == (f"- {fee}", None)


def test_notes_that_were_only_a_rate_leave_the_field_unset():
    """Unset, not empty: the dispatcher is still asked about notes."""
    intake, _ = _to_intake(_extraction(notes={"value": "$145 cash /$155 card", "confidence": 0.9}))
    assert "job_notes" not in intake
