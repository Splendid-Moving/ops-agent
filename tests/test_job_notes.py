"""
Job notes must carry what the other fields cannot, and nothing else.

The notes print on the calendar event directly beneath the date, the addresses,
the crew size and the rate. A note restating those is noise a dispatcher reads
past on every job — and the agent was producing exactly that, twice over,
because two near-identical schema fields were both filled and then joined.
"""

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


def test_a_real_note_survives():
    extraction = _extraction(notes={"value": "$60 gas fee", "confidence": 0.95})
    intake, _ = _to_intake(extraction)
    assert intake["job_notes"] == "$60 gas fee"


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
