"""
Intake flow: interrupts, resume behaviour, and the routing decisions around
them.

The idempotency tests here are the most important in the project. A graph
resumes by RE-RUNNING the paused node from the top, so anything above an
interrupt() executes again on every resume. If a side effect ever drifts above
one of those lines, a customer gets billed twice — and nothing else in the test
suite would notice.
"""

import ast
import inspect
from pathlib import Path

import pytest

from agent.nodes import ask_missing as ask_module
from agent.nodes import confirm as confirm_module
from agent.nodes import validate_checklist as validate_module


# ── Structural: no side effects above interrupt() ──────────────────────────────

SIDE_EFFECT_MARKERS = (
    "upsert_contact", "create_invoice", "send_invoice", "send_sms", "send_email",
    "create_event", "requests.post", "requests.put", "requests.delete",
)


def _source(module) -> str:
    return Path(inspect.getfile(module)).read_text()


@pytest.mark.parametrize("module", [ask_module, confirm_module])
def test_interrupt_nodes_perform_no_writes(module):
    """
    Neither interrupt node may contain a write of any kind — not above the
    interrupt, not below it. All writes belong in the act_* nodes downstream of
    the confirm gate, where they run exactly once per approval.
    """
    source = _source(module)
    # Strip the docstrings, which legitimately name these functions in prose.
    tree = ast.parse(source)
    code_only = "\n".join(
        ast.unparse(node)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Call, ast.Assign, ast.Expr))
        and not isinstance(getattr(node, "value", None), ast.Constant)
    )
    for marker in SIDE_EFFECT_MARKERS:
        assert marker not in code_only, (
            f"{module.__name__} contains {marker!r}. On resume this node re-runs "
            "from the top, so a write here fires again every time."
        )


def test_confirm_only_reads_before_interrupting():
    """
    confirm() does one lookup before pausing — the duplicate-job check. That is
    a calendar READ, which is safe to repeat. Assert it stays a read.
    """
    source = _source(confirm_module)
    assert "find_duplicate_job" in source
    assert "create_event" not in source.replace('"""', "")


# ── Confirm routing ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("yes", ["yes", "y", "ok", "go ahead", "confirm", "do it", "  YES  "])
def test_affirmative_replies_execute(yes, monkeypatch):
    monkeypatch.setattr(confirm_module, "interrupt", lambda _: yes)
    monkeypatch.setattr(confirm_module, "_find_duplicate", lambda _: None)
    command = confirm_module.confirm({"intake": {"full_name": "X"}})
    assert command.goto == "execute"
    assert command.update["approved"] is True


@pytest.mark.parametrize("no", ["no", "n", "cancel", "stop", "never mind", "ABORT"])
def test_negative_replies_end_without_executing(no, monkeypatch):
    monkeypatch.setattr(confirm_module, "interrupt", lambda _: no)
    monkeypatch.setattr(confirm_module, "_find_duplicate", lambda _: None)
    command = confirm_module.confirm({"intake": {"full_name": "X"}})
    assert command.goto == "__end__"
    assert command.update["approved"] is False


@pytest.mark.parametrize("edit", ["make it 4 movers", "arrival 10-11am", "change the date to 9/1"])
def test_anything_else_is_treated_as_an_edit(edit, monkeypatch):
    """
    An edit must never be mistaken for approval. Routing back through
    ask_missing means it gets re-parsed and re-validated, so an edit cannot
    bypass the checklist.
    """
    monkeypatch.setattr(confirm_module, "interrupt", lambda _: edit)
    monkeypatch.setattr(confirm_module, "_find_duplicate", lambda _: None)
    command = confirm_module.confirm({"intake": {"full_name": "X"}})
    assert command.goto == "ask_missing"
    assert command.update.get("approved") is not True


def test_ambiguous_reply_does_not_approve(monkeypatch):
    """'yes please but change the time' must NOT be read as plain approval."""
    monkeypatch.setattr(confirm_module, "interrupt", lambda _: "yes but make it 4 movers")
    monkeypatch.setattr(confirm_module, "_find_duplicate", lambda _: None)
    command = confirm_module.confirm({"intake": {"full_name": "X"}})
    assert command.goto == "ask_missing"


# ── Loop guard ─────────────────────────────────────────────────────────────────

def test_loop_gives_up_after_max_rounds():
    """A validator that can never be satisfied must not re-ask forever."""
    intake = {"_ask_rounds": validate_module.MAX_ASK_ROUNDS}
    assert validate_module.next_step({"intake": intake}) == "give_up"


def test_loop_continues_below_the_limit():
    assert validate_module.next_step({"intake": {"_ask_rounds": 1}}) == "ask_missing"


def test_complete_intake_goes_to_confirm():
    from datetime import datetime, timedelta

    from services.calendar import LA_TZ

    intake = {
        "full_name": "Sarah Chen",
        "email": "sarah@example.com",
        "phone": "(818) 555-0142",
        "pickup_address": "412 N Maple Ave, Burbank CA 91505",
        "dropoff_address": "1830 Pine St, Glendale CA 91206",
        "move_date": (datetime.now(LA_TZ) + timedelta(days=10)).strftime("%m/%d/%Y"),
        "arrival_time": "8-9am",
        "movers": "3",
        "job_notes": "none",
        "notes_asked": True,
    }
    assert validate_module.next_step({"intake": intake}) == "confirm"


# ── Summary rendering ──────────────────────────────────────────────────────────

def _complete_intake(**overrides):
    from datetime import datetime, timedelta

    from services.calendar import LA_TZ

    base = {
        "full_name": "Sarah Chen",
        "email": "sarah@example.com",
        "phone": "+1(818)555-0142",
        "pickup_address": "412 N Maple Ave, Burbank CA 91505",
        "dropoff_address": "1830 Pine St, Glendale CA 91206",
        "move_date": (datetime.now(LA_TZ) + timedelta(days=10)).strftime("%m/%d/%Y"),
        "arrival_time": "8-9am",
        "movers": "3",
        "job_notes": "Third floor walkup",
        "notes_asked": True,
    }
    base.update(overrides)
    return base


def test_summary_states_nothing_has_happened_yet():
    text = confirm_module._summary(_complete_intake(), {}, None)
    assert "Nothing has happened yet" in text


def test_summary_lists_all_four_actions():
    """The user is approving four irreversible things; all four must be shown."""
    text = confirm_module._summary(_complete_intake(), {}, None).lower()
    assert "gohighlevel" in text
    assert "calendar" in text
    assert "deposit invoice" in text
    assert "confirmation email" in text


def test_summary_shows_derived_rate_and_deposit():
    text = confirm_module._summary(_complete_intake(), {}, None)
    assert "$145 cash /$155 card" in text
    assert "$50" in text


def test_labor_job_hides_dropoff_and_marks_the_title():
    intake = _complete_intake(is_labor=True, dropoff_address="")
    text = confirm_module._summary(intake, {}, None)
    assert "labor only" in text
    assert "Drop-off" not in text


def test_duplicate_job_is_surfaced_prominently():
    duplicate = {"title": "Sarah Chen", "move_date": "08/14/2026", "movers": "3"}
    text = confirm_module._summary(_complete_intake(), {}, duplicate)
    assert "already a job" in text
    assert "double it up" in text


def test_unverified_addresses_are_surfaced():
    """The Montebello / Wyoming failure mode must reach the human."""
    intake = _complete_intake(
        address_status={
            "pickup_address": "needs_review: filled in locality. Worth a check.",
            "dropoff_address": "confirmed",
        }
    )
    text = confirm_module._summary(intake, {}, None)
    assert "couldn't fully verify" in text.lower()
    assert "pickup_address" in text
    # A confirmed address should not be flagged.
    assert text.count("dropoff_address") == 0


def test_extra_stop_appears_only_when_present():
    assert "Extra stop" not in confirm_module._summary(_complete_intake(), {}, None)
    with_stop = confirm_module._summary(
        _complete_intake(extra_stop="500 S Brand Blvd, Glendale CA 91204"), {}, None
    )
    assert "Extra stop" in with_stop


# ── Relative dates in extraction ───────────────────────────────────────────────

def test_extraction_prompt_carries_todays_date():
    """
    Without a date anchor the model cannot resolve "tomorrow", so it reported
    such dates at low confidence, they fell below the threshold, and the agent
    asked for a date the screenshot had already supplied.
    """
    from datetime import datetime

    from agent.nodes.extract_screenshot import _system_prompt
    from services.calendar import LA_TZ

    prompt = _system_prompt()
    today = datetime.now(LA_TZ)
    assert f"{today:%m/%d/%Y}" in prompt
    assert "{today}" not in prompt and "{{" not in prompt, "template left unfilled"


@pytest.mark.live
@pytest.mark.parametrize(
    "phrase,offset_days",
    [("we need movers tomorrow", 1), ("moving the day after tomorrow", 2)],
)
def test_relative_dates_resolve_to_usable_values(phrase, offset_days):
    from datetime import datetime, timedelta

    from langchain_core.messages import HumanMessage, SystemMessage

    from agent.models import get_model
    from agent.nodes.extract_screenshot import _system_prompt
    from schemas.intake import ScreenshotExtraction
    from services.calendar import LA_TZ

    model = get_model("extract_screenshot").with_structured_output(ScreenshotExtraction)
    result = model.invoke([
        SystemMessage(content=_system_prompt()),
        HumanMessage(content=f"There is no screenshot. Extract from this message:\n\n{phrase}"),
    ])

    expected = (datetime.now(LA_TZ) + timedelta(days=offset_days)).strftime("%m/%d/%Y")
    assert result.move_date.value == expected
    assert result.move_date.is_usable, "resolvable date must not be re-asked"


@pytest.mark.live
def test_genuinely_vague_dates_are_still_left_blank():
    """Resolving 'tomorrow' must not turn into guessing at 'sometime in spring'."""
    from langchain_core.messages import HumanMessage, SystemMessage

    from agent.models import get_model
    from agent.nodes.extract_screenshot import _system_prompt
    from schemas.intake import ScreenshotExtraction

    model = get_model("extract_screenshot").with_structured_output(ScreenshotExtraction)
    result = model.invoke([
        SystemMessage(content=_system_prompt()),
        HumanMessage(content="There is no screenshot. Extract from this message:\n\n"
                             "we're thinking of moving sometime in spring maybe"),
    ])
    assert not result.move_date.is_usable
