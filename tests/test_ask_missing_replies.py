"""
What ask_missing does with a reply that isn't an answer.

While the graph is paused the router is bypassed — the channel hands every
message straight to this node as Command(resume=...). So a dispatcher who
wanders off mid-booking ("how many jobs last month?") has their question parsed
as if it were a phone number. Getting nothing out of it is correct. Treating it
as a failed attempt is not: six detours would hit the loop guard and abandon a
booking the user never actually got wrong.
"""

from datetime import datetime, timedelta

import pytest

from agent.nodes import ask_missing as am
from agent.nodes import validate_checklist as validate_module
from services.calendar import LA_TZ


def _reply(**kwargs) -> am.ParsedReply:
    return am.ParsedReply(**kwargs)


def _incomplete() -> dict:
    return {"full_name": "Sarah Chen", "pickup_address": "412 N Maple Ave, Burbank CA 91505"}


def _complete(**overrides) -> dict:
    base = {
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
    base.update(overrides)
    return base


def _run(monkeypatch, intake: dict, reply_text: str, parsed: am.ParsedReply | None) -> dict:
    """Drive the node once with a canned model result."""
    monkeypatch.setattr(am, "interrupt", lambda _: reply_text)
    monkeypatch.setattr(am, "_parse", lambda *a, **k: parsed)
    return am.ask_missing({"intake": intake})["intake"]


# ── Off-topic replies ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("kind", ["off_topic", "later"])
def test_a_reply_that_answers_nothing_does_not_burn_a_round(monkeypatch, kind):
    """
    The loop guard exists to stop the bot re-asking a question that can never be
    satisfied. A dispatcher changing the subject is not that, and counting it
    would abandon a perfectly fine booking after six detours.
    """
    out = _run(monkeypatch, _incomplete(), "how many jobs last month?", _reply(reply_kind=kind))
    assert out.get("_ask_rounds", 0) == 0


@pytest.mark.parametrize("kind", ["off_topic", "later"])
def test_the_detour_is_acknowledged_in_the_next_question(monkeypatch, kind):
    """Silence reads as the bot ignoring them, which is what feels broken."""
    out = _run(monkeypatch, _incomplete(), "is Adilet in today?", _reply(reply_kind=kind))
    assert out["_aside"] == am.ASIDE[kind]


def test_the_acknowledgement_is_shown_once_then_dropped(monkeypatch):
    """Otherwise it repeats above every question for the rest of the booking."""
    intake = _incomplete() | {"_aside": am.ASIDE["off_topic"]}
    out = _run(monkeypatch, intake, "(818) 555-0142", _reply(phone="(818) 555-0142"))
    assert "_aside" not in out


def test_an_answer_still_counts_even_if_the_model_calls_it_off_topic(monkeypatch):
    """
    An extracted field is hard evidence and overrules the classification — the
    same model-proposes/Python-disposes rule as the rest of the checklist.
    Otherwise a misclassifying model could stop the counter forever.
    """
    out = _run(
        monkeypatch, _incomplete(), "818-555-0142 btw how was your weekend",
        _reply(phone="(818) 555-0142", reply_kind="off_topic"),
    )
    assert out["_ask_rounds"] == 1
    assert "_aside" not in out


def test_an_unparseable_reply_still_counts_as_an_attempt(monkeypatch):
    """A model failure is not evidence the user was off topic."""
    out = _run(monkeypatch, _incomplete(), "asdfgh", None)
    assert out["_ask_rounds"] == 1


# ── The backstop counter ───────────────────────────────────────────────────────

def test_every_exchange_counts_toward_the_absolute_limit(monkeypatch):
    """
    _ask_turns is the guard on the guard: if the model ever classified every
    reply as off-topic, _ask_rounds would never move and give_up could never
    fire.
    """
    out = _run(monkeypatch, _incomplete(), "unrelated", _reply(reply_kind="off_topic"))
    assert out["_ask_turns"] == 1


def test_give_up_fires_on_turns_even_when_no_round_was_ever_counted():
    intake = {"_ask_rounds": 0, "_ask_turns": validate_module.MAX_ASK_TURNS}
    assert validate_module.next_step({"intake": intake}) == "give_up"


def test_the_turn_limit_is_looser_than_the_round_limit():
    """Detours should cost patience, not the booking."""
    assert validate_module.MAX_ASK_TURNS > validate_module.MAX_ASK_ROUNDS


# ── Edits handed over by the confirm gate ──────────────────────────────────────

def test_an_edit_from_confirm_is_applied(monkeypatch):
    """
    confirm routes "make it 4 movers" here. The intake is already complete, so
    the old code returned {} and the edit vanished — the summary re-rendered
    unchanged and the user's correction was silently lost.
    """
    monkeypatch.setattr(am, "_parse", lambda *a, **k: _reply(movers="4"))
    intake = _complete(_pending_edit="make it 4 movers")
    out = am.ask_missing({"intake": intake})["intake"]
    assert out["movers"] == "4"


def test_an_edit_never_pauses_to_ask_again(monkeypatch):
    """The user typed it a second ago; asking them to retype it is the bug."""
    def explode(_):
        raise AssertionError("interrupt() must not be reached on the edit path")

    monkeypatch.setattr(am, "interrupt", explode)
    monkeypatch.setattr(am, "_parse", lambda *a, **k: _reply(arrival_time="10-11am"))
    am.ask_missing({"intake": _complete(_pending_edit="arrival 10-11am")})


def test_the_edit_marker_is_consumed(monkeypatch):
    """Left behind, it would re-apply on every later pass through this node."""
    monkeypatch.setattr(am, "_parse", lambda *a, **k: _reply(movers="4"))
    out = am.ask_missing({"intake": _complete(_pending_edit="make it 4 movers")})["intake"]
    assert "_pending_edit" not in out


def test_an_edit_that_breaks_the_checklist_is_caught(monkeypatch):
    """
    An edit must not be able to bypass validation — it re-enters the normal
    loop, so "make it 5 movers" is rejected like any other bad value.
    """
    from schemas import checklist as cl

    monkeypatch.setattr(am, "_parse", lambda *a, **k: _reply(movers="5"))
    out = am.ask_missing({"intake": _complete(_pending_edit="make it 5 movers")})["intake"]
    assert not cl.evaluate(out).is_complete


# ── Internal bookkeeping must not leak into the question ───────────────────────

def test_the_question_never_shows_internal_counters():
    """These rendered as "Ask Rounds: 2" in the middle of a real question."""
    intake = _incomplete() | {"_ask_rounds": 2, "_ask_turns": 3, "_pending_edit": "x"}
    known = {
        k: v for k, v in intake.items()
        if v not in (None, "", {}) and not k.startswith("_") and k not in am.HIDDEN_FIELDS
    }
    text = am._format_question(known, ["What date is the move?"])
    assert "Ask Rounds" not in text
    assert "Ask Turns" not in text
    assert "Pending Edit" not in text
    assert "Sarah Chen" in text


def test_the_aside_leads_the_question_when_present():
    text = am._format_question({"full_name": "Sarah Chen"}, ["What date?"], am.ASIDE["off_topic"])
    assert text.startswith(am.ASIDE["off_topic"])
