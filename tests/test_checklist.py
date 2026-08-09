"""
Checklist logic. Pure, no network, no model.

This is the gate that decides whether a job is safe to create, so it gets
tested harder than anything else in the project.
"""

from datetime import datetime, timedelta

import pytest

from schemas import checklist as cl
from services.calendar import LA_TZ


def _future(days: int = 10) -> str:
    return (datetime.now(LA_TZ) + timedelta(days=days)).strftime("%m/%d/%Y")


COMPLETE_MOVE = {
    "full_name": "Sarah Chen",
    "email": "sarah@example.com",
    "phone": "(818) 555-0142",
    "pickup_address": "412 N Maple Ave, Burbank CA 91505",
    "dropoff_address": "1830 Pine St, Glendale CA 91206",
    "move_date": _future(),
    "arrival_time": "8-9am",
    "movers": "3",
    "job_notes": "Third floor walkup",
}


# ── The happy path ─────────────────────────────────────────────────────────────

def test_complete_job_passes():
    result = cl.evaluate(COMPLETE_MOVE)
    assert result.is_complete
    assert result.all_questions() == []


# ── Labor inference ────────────────────────────────────────────────────────────

def test_two_addresses_means_not_labor_without_asking():
    """The common case must not cost a question."""
    assert cl.infer_is_labor("412 N Maple Ave", "1830 Pine St", None) is False
    assert not cl.evaluate(COMPLETE_MOVE).needs_labor_answer


def test_one_address_is_ambiguous_and_must_be_asked():
    """
    A labor job and a full move with a missed drop-off look identical.
    Guessing labor here would skip a required address.
    """
    assert cl.infer_is_labor("412 N Maple Ave", "", None) is None

    result = cl.evaluate({**COMPLETE_MOVE, "dropoff_address": ""})
    assert result.needs_labor_answer
    assert not result.is_complete
    assert any("labor-only" in q for q in result.all_questions())


def test_explicit_labor_flag_wins_over_inference():
    assert cl.infer_is_labor("a", "b", True) is True
    assert cl.infer_is_labor("a", "", False) is False


def test_labor_job_does_not_require_dropoff():
    job = {**COMPLETE_MOVE, "dropoff_address": "", "is_labor": True}
    result = cl.evaluate(job)
    assert result.is_complete
    assert "dropoff_address" not in {s.name for s in result.missing}


def test_non_labor_job_does_require_dropoff():
    job = {**COMPLETE_MOVE, "dropoff_address": "", "is_labor": False}
    result = cl.evaluate(job)
    assert not result.is_complete
    assert "dropoff_address" in {s.name for s in result.missing}


# ── Job notes: always asked ────────────────────────────────────────────────────

def test_notes_are_asked_when_absent():
    """Extra charges and gas fees live here — never skip the question."""
    result = cl.evaluate({**COMPLETE_MOVE, "job_notes": ""})
    assert not result.is_complete
    assert "job_notes" in {s.name for s in result.missing}


def test_notes_stop_being_asked_once_answered_as_none():
    """An explicit 'no notes' is a complete answer, not a gap to re-ask."""
    job = {**COMPLETE_MOVE, "job_notes": "", "notes_asked": True}
    assert cl.evaluate(job).is_complete


# ── Extra stop: never asked ────────────────────────────────────────────────────

def test_extra_stop_is_never_requested():
    result = cl.evaluate({**COMPLETE_MOVE, "extra_stop": ""})
    assert result.is_complete
    assert "extra_stop" not in {s.name for s in result.missing}
    assert cl.BY_NAME["extra_stop"].never_ask


# ── Required fields ────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "field", ["full_name", "email", "phone", "pickup_address",
              "move_date", "arrival_time", "movers"]
)
def test_each_required_field_blocks_when_missing(field):
    result = cl.evaluate({**COMPLETE_MOVE, field: ""})
    assert not result.is_complete
    assert field in {s.name for s in result.missing}


def test_empty_intake_asks_for_everything_at_once():
    """No screenshot at all: the agent must still be able to book by asking."""
    result = cl.evaluate({})
    names = {s.name for s in result.missing}
    assert {"full_name", "email", "phone", "pickup_address",
            "move_date", "arrival_time", "movers", "job_notes"} <= names
    assert result.needs_labor_answer
    # One batched ask, not a drip of questions.
    assert len(result.all_questions()) >= 8


# ── Validators ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("bad", ["not-an-email", "sarah@", "@example.com", "sarah example.com"])
def test_bad_emails_rejected(bad):
    result = cl.evaluate({**COMPLETE_MOVE, "email": bad})
    assert "email" in result.invalid


@pytest.mark.parametrize("good", ["sarah@example.com", "a.b+tag@sub.domain.co"])
def test_good_emails_accepted(good):
    assert cl.valid_email(good) is None


@pytest.mark.parametrize("bad", ["555", "12345", "abcdefghij"])
def test_bad_phones_rejected(bad):
    result = cl.evaluate({**COMPLETE_MOVE, "phone": bad})
    assert "phone" in result.invalid


@pytest.mark.parametrize("good", ["(818) 555-0142", "8185550142", "+1 818-555-0142"])
def test_good_phones_accepted(good):
    assert cl.valid_phone(good) is None


def test_past_move_date_rejected():
    past = (datetime.now(LA_TZ) - timedelta(days=3)).strftime("%m/%d/%Y")
    result = cl.evaluate({**COMPLETE_MOVE, "move_date": past})
    assert "move_date" in result.invalid
    assert "past" in result.invalid["move_date"]


def test_unparseable_date_rejected():
    result = cl.evaluate({**COMPLETE_MOVE, "move_date": "sometime next week"})
    assert "move_date" in result.invalid


def test_short_lead_time_warns_but_does_not_block():
    """Same-day jobs happen. The human decides at the confirm gate."""
    tomorrow = (datetime.now(LA_TZ) + timedelta(days=1)).strftime("%m/%d/%Y")
    result = cl.evaluate({**COMPLETE_MOVE, "move_date": tomorrow})
    assert "move_date" in result.warnings
    assert "move_date" not in result.invalid
    assert result.is_complete


@pytest.mark.parametrize("bad", ["morning", "sometime", "9"])
def test_bad_arrival_windows_rejected(bad):
    result = cl.evaluate({**COMPLETE_MOVE, "arrival_time": bad})
    assert "arrival_time" in result.invalid


@pytest.mark.parametrize("good", ["8-9am", "2-4pm", "11am-1pm", "9am"])
def test_good_arrival_windows_accepted(good):
    assert cl.valid_arrival_time(good) is None


@pytest.mark.parametrize("size", ["2", "3", "4"])
def test_supported_crew_sizes_accepted(size):
    assert cl.valid_movers(size) is None


@pytest.mark.parametrize("size", ["5", "6"])
def test_out_of_scope_crew_sizes_explain_themselves(size):
    """
    5 and 6 are real sizes GHL accepts but the agent doesn't automate. The
    message must say 'do it by hand', not 'invalid number'.
    """
    error = cl.valid_movers(size)
    assert error is not None
    assert "by hand" in error or "aren't automated" in error


@pytest.mark.parametrize("size", ["1", "7", "12", "many"])
def test_nonsense_crew_sizes_rejected_plainly(size):
    error = cl.valid_movers(size)
    assert error is not None
    assert "must be one of" in error


@pytest.mark.parametrize("bad", ["", "abc", "Maple Ave"])
def test_addresses_without_a_number_rejected(bad):
    assert cl.valid_address(bad) is not None


# ── Batched asking ─────────────────────────────────────────────────────────────

def test_questions_are_batched_not_dripped():
    """
    Validation knows the full gap list before asking, so the user answers once
    rather than over four round trips.
    """
    result = cl.evaluate({
        "full_name": "Sarah Chen",
        "email": "sarah@example.com",
        "phone": "(818) 555-0142",
        "pickup_address": "412 N Maple Ave, Burbank CA 91505",
        "dropoff_address": "1830 Pine St, Glendale CA 91206",
    })
    questions = result.all_questions()
    joined = " ".join(questions).lower()
    assert "date" in joined
    assert "arrival" in joined
    assert "movers" in joined
    assert "notes" in joined
    assert len(questions) >= 4
