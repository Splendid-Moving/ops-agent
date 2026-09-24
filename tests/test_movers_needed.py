"""
Per-day crew totals, computed in Python.

Asked how many movers Saturday needed, the agent had no tool that answered it.
It pulled a three-day block, tried to split it by day in its head, and produced
23 — then, challenged, withdrew the number entirely, then produced 19. Only the
last was right.

Multiplying crew size by job count and summing is arithmetic. A model should
never be doing it on a number the yard staffs against.
"""

import pytest

from agent.nodes import analytics


def _job(date: str, movers: str, customer: str = "X", labor: bool = False) -> dict:
    return {"calendar_date": date, "movers": movers, "customer": customer,
            "is_labor": labor, "start_time": "", "end_time": ""}


@pytest.fixture
def calendar_jobs(monkeypatch):
    """Serve a fixed job list to the tool, bypassing Google."""
    def serve(jobs):
        monkeypatch.setattr(analytics.calendar, "list_jobs", lambda s, e: jobs)
    return serve


def _run(start="2026-09-26", end="2026-09-26") -> str:
    return analytics.movers_needed.invoke({"start_date": start, "end_date": end})


# ── The arithmetic ─────────────────────────────────────────────────────────────

def test_movers_are_summed_across_the_days_jobs(calendar_jobs):
    """The live Saturday: 3x2 + 3x3 + 1x4 = 19."""
    calendar_jobs([
        *[_job("09/26/2026", "2") for _ in range(3)],
        *[_job("09/26/2026", "3") for _ in range(3)],
        _job("09/26/2026", "4"),
    ])
    out = _run()
    assert "19 movers" in out
    assert "7 jobs" in out


def test_the_breakdown_shows_how_the_total_was_reached(calendar_jobs):
    """So a human can check it at a glance instead of trusting the number."""
    calendar_jobs([_job("09/26/2026", "2"), _job("09/26/2026", "2"), _job("09/26/2026", "3")])
    out = _run()
    assert "2 jobs x 2 movers = 4" in out
    assert "1 job x 3 movers = 3" in out
    assert "7 movers" in out


def test_each_day_is_counted_separately(calendar_jobs):
    """
    The original failure: one combined three-day figure, split by day in the
    model's head. Every day gets its own line.
    """
    calendar_jobs([
        _job("09/25/2026", "2"), _job("09/25/2026", "4"),
        _job("09/26/2026", "3"),
    ])
    out = _run("2026-09-25", "2026-09-26")
    assert "09/25/2026" in out and "6 movers" in out
    assert "09/26/2026" in out and "3 movers" in out


def test_days_are_named_so_saturday_can_be_asked_for(calendar_jobs):
    calendar_jobs([_job("09/26/2026", "3")])
    assert "Saturday" in _run()


def test_days_appear_in_date_order(calendar_jobs):
    calendar_jobs([_job("09/26/2026", "2"), _job("09/24/2026", "2"), _job("09/25/2026", "2")])
    out = _run("2026-09-24", "2026-09-26")
    assert out.index("09/24/2026") < out.index("09/25/2026") < out.index("09/26/2026")


def test_a_multi_day_range_gets_a_total(calendar_jobs):
    calendar_jobs([_job("09/25/2026", "2"), _job("09/26/2026", "3")])
    out = _run("2026-09-25", "2026-09-26")
    assert "5 movers" in out


def test_a_single_day_gets_no_redundant_total(calendar_jobs):
    calendar_jobs([_job("09/26/2026", "3")])
    assert "Range total" not in _run()


# ── Missing crew sizes are surfaced, never counted as zero ─────────────────────

@pytest.mark.parametrize("blank", ["", "   ", "?", "unspecified"])
def test_a_job_with_no_crew_size_is_flagged_not_silently_skipped(calendar_jobs, blank):
    """
    Treating a blank as zero understates staffing for a real job. The yard
    would be short a crew and nobody would know why.
    """
    calendar_jobs([_job("09/26/2026", "3"), _job("09/26/2026", blank, customer="Ann")])
    out = _run()
    assert "3 movers" in out
    assert "no crew size" in out.lower()
    assert "Ann" in out


def test_labor_jobs_count_toward_the_total(calendar_jobs):
    """A labor-only job still needs bodies."""
    calendar_jobs([_job("09/26/2026", "2", labor=True), _job("09/26/2026", "3")])
    assert "5 movers" in _run()


def test_an_empty_day_says_so(calendar_jobs):
    calendar_jobs([])
    assert "No jobs" in _run()


# ── The assumption is stated, so the model stops inventing nuance ─────────────

def test_the_output_states_what_the_number_means(calendar_jobs):
    """
    Left to itself the model editorialised: "21 mover-slots across standard
    crews plus 6 on the big crew; 23 unique movers if everything overlaps."
    None of that came from the calendar. The tool says what it counted.
    """
    calendar_jobs([_job("09/26/2026", "3")])
    out = _run()
    assert "assumes no crew works more than one job" in out.lower()


# ── Range guards match the other tools ─────────────────────────────────────────

def test_a_backwards_range_is_rejected(calendar_jobs):
    calendar_jobs([])
    with pytest.raises(ValueError, match="before"):
        _run("2026-09-26", "2026-09-25")


# ── The prompt rules that cover how it failed ─────────────────────────────────

def test_the_prompt_forbids_promising_a_follow_up():
    """
    "I'll check the calendar and tell you in a minute" — then nothing, because
    nothing runs after the turn ends. The user waited for a message that could
    never arrive.
    """
    text = analytics._system_prompt()
    assert "one turn" in text.lower()
    assert "give me a minute" in text


def test_the_prompt_says_to_recheck_rather_than_retract():
    """
    Challenged on a wrong number, it withdrew entirely — "I can't confirm that
    figure, don't rely on it" — when the calendar had the answer all along.
    """
    text = analytics._system_prompt()
    assert "Re-check, don't retract" in text


def test_the_prompt_routes_crew_questions_to_the_tool():
    assert "movers_needed" in analytics._system_prompt()


def test_the_prompt_forbids_hand_splitting_a_multi_day_result():
    """Exactly how the wrong Saturday figure was produced."""
    assert "split a multi-day tool result into single days by hand" in analytics._system_prompt()
