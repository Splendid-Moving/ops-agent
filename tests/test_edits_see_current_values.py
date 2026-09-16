"""
The reply parser must see what is on file, or edits are impossible.

"Remove the gas fee line from notes" is an instruction relative to a value the
dispatcher can see and the model could not. It returned null — nothing to
remove from — and the edit vanished without a word. Twice in one conversation,
which is what made the agent feel like a regex bot.

In the LangGraph design vocabulary: every LLM step has static context (the
prompt) and dynamic context (from state). This step had the first and lacked
the second.
"""

import pytest

from agent.nodes import ask_missing as am


NOTES = "- $145/hour cash price\n- Gas fee $50 waived\n- 5% off rate for previous customer"


def _prompt(**intake) -> str:
    return am._parse_prompt(["Arrival window?"], editing=False, current=intake)


# ── The current values reach the model ─────────────────────────────────────────

def test_the_prompt_shows_what_is_on_file():
    text = _prompt(full_name="Tiffany Pan", job_notes=NOTES, movers="3")
    assert "Tiffany Pan" in text
    assert "Gas fee $50 waived" in text
    assert "Crew size" in text and "3" in text


def test_internal_bookkeeping_is_not_shown():
    text = _prompt(full_name="X", _ask_rounds=4, _pending_edit="y", notes_asked=True)
    assert "_ask_rounds" not in text
    assert "_pending_edit" not in text
    assert "notes_asked" not in text


def test_an_empty_booking_says_so():
    assert "(nothing yet)" in _prompt()


def test_labor_flag_is_rendered_in_words():
    assert "full move" in _prompt(is_labor=False)
    assert "labor only" in _prompt(is_labor=True)


# ── The rules the model is given ───────────────────────────────────────────────

def test_the_prompt_demands_complete_values_not_instructions():
    """The failure mode is returning the instruction, or a fragment."""
    text = _prompt(job_notes=NOTES)
    assert "COMPLETE new value" in text
    assert "never the instruction itself" in text


def test_removals_are_told_to_leave_other_lines_untouched():
    """
    The first cut of this over-deleted: asked to drop one of four lines, the
    model dropped two, having pattern-matched the worked example instead of
    editing the real value.
    """
    text = _prompt(job_notes=NOTES)
    assert "touches ONLY the line named" in text
    assert "one fewer, never two" in text


def test_the_worked_example_uses_fake_data():
    """
    An example built from the real notes was copied back verbatim. The
    illustration must be data the model cannot mistake for the booking.
    """
    text = _prompt(job_notes=NOTES)
    assert "Piano, ground floor" in text          # the illustration
    assert "Work from the values ABOVE" in text


# ── Both entrances pass the booking through ────────────────────────────────────

def test_the_edit_path_passes_the_current_intake(monkeypatch):
    seen = {}

    def spy(reply, questions, *, editing, current=None):
        seen.update(current or {})
        return am.ParsedReply(movers="4")

    monkeypatch.setattr(am, "_parse", spy)
    am.ask_missing({"intake": {"full_name": "Tiffany Pan", "movers": "3",
                               "_pending_edit": "make it 4 movers"}})
    assert seen["full_name"] == "Tiffany Pan"


def test_the_question_path_passes_the_current_intake(monkeypatch):
    seen = {}

    def spy(reply, questions, *, editing, current=None):
        seen.update(current or {})
        return am.ParsedReply(arrival_time="8-9am")

    monkeypatch.setattr(am, "interrupt", lambda _: "8-9am")
    monkeypatch.setattr(am, "_parse", spy)
    am.ask_missing({"intake": {"full_name": "Tiffany Pan", "job_notes": NOTES}})
    assert seen["job_notes"] == NOTES
