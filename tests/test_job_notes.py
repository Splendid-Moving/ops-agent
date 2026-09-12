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


# ── The standing deposit terms ─────────────────────────────────────────────────
#
# Every job takes the same deposit: it is config.deposit_amount(), it is step
# three of every booking, and the calendar event has its own `Deposit:` line.
# The quote boilerplate staff send every customer sits in the screenshot
# looking exactly like an agreed extra charge, so it was being captured
# "generously" and printed on the calendar of every job booked from a thread
# that contained it.

@pytest.mark.parametrize(
    "boilerplate",
    [
        "$50 deposit required; subtracted from total at end of move",
        "- $50 deposit required, subtracted from the total at the end of the move",
        "A $50 deposit is required to book",
        "Deposit of $50 to reserve the slot, applied to the final balance",
        "$50 non-refundable deposit",
    ],
)
def test_the_standing_deposit_terms_never_become_a_note(boilerplate):
    extraction = _extraction(notes={"value": boilerplate, "confidence": 0.95})
    intake, _ = _to_intake(extraction)
    assert "job_notes" not in intake


def test_a_deposit_already_paid_is_still_worth_a_line():
    """
    The one case that is news rather than terms — the dispatcher does need to
    know this one, so the filter has to be narrower than "any line saying
    deposit".
    """
    extraction = _extraction(
        notes={"value": "Deposit already paid in cash", "confidence": 0.95}
    )
    intake, _ = _to_intake(extraction)
    assert intake["job_notes"] == "- Deposit already paid in cash"


def test_a_real_note_survives_alongside_the_terms():
    """The terms usually arrive joined to something that is worth keeping."""
    extraction = _extraction(
        notes={
            "value": "- $50 deposit required, subtracted from total\n- Third floor walk-up",
            "confidence": 0.95,
        }
    )
    intake, _ = _to_intake(extraction)
    assert intake["job_notes"] == "- Third floor walk-up"


def test_dropping_the_terms_leaves_the_field_unset_not_blank():
    """
    Same distinction the rest of this file turns on: unset means the checklist
    still asks, and a dispatcher who has a note can give it.
    """
    extraction = _extraction(
        notes={"value": "$50 deposit required", "confidence": 0.95}
    )
    intake, _ = _to_intake(extraction)
    assert "job_notes" not in intake


def test_the_prompt_rules_the_deposit_out_too():
    """Pinned for the same reason as the restatement rule above."""
    from agent.nodes import extract_screenshot as node
    assert "NOT the deposit" in node.SYSTEM_PROMPT_BODY


def test_a_dispatcher_who_types_a_deposit_note_is_taken_at_their_word():
    """
    The filter is for what the model reads out of a screenshot. A typed note is
    a deliberate instruction, and nothing here second-guesses it.
    """
    from schemas import checklist as cl
    from services import formatting

    spec = cl.BY_NAME["job_notes"]
    assert spec.normalizer("$50 deposit collected on site") == (
        "- $50 deposit collected on site"
    )
    assert formatting.format_notes("$50 deposit required") == "- $50 deposit required"


# ── "No notes" means no notes ──────────────────────────────────────────────────

@pytest.mark.parametrize(
    "answer",
    ["none", "None.", "no", "No notes", "nothing", "n/a", "nope", "  none  "],
)
def test_a_refusal_is_an_empty_field_not_a_note(answer):
    """
    The checklist invites "none is fine", so the answer arrives constantly. It
    answers the question — it is not a fact about the job — and printing
    "- none" beneath the addresses on the calendar is exactly the noise this
    whole file exists to keep out.
    """
    from services import formatting
    assert formatting.format_notes(answer) == ""


def test_a_note_that_merely_starts_with_no_is_kept():
    """The check is on the whole answer. "No parking on the street" is a note."""
    from services import formatting
    assert formatting.format_notes("No parking on the street") == (
        "- No parking on the street"
    )
    assert formatting.format_notes("No elevator, 3rd floor") == (
        "- No elevator, 3rd floor"
    )
