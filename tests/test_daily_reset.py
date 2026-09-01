"""
Conversations are discarded once a day.

Implemented as a check on the way in rather than a job that fires at 00:00.
A timer would have to survive Railway restarting on every deploy, could race
the webhook mid-turn, and would do work on days nobody messages. Checking when
a message arrives is equivalent from the dispatcher's side — they can never
reach yesterday's state — with nothing to drift or crash.

The subtle part is the boundary. It is the LOS ANGELES calendar date, so
"midnight" means midnight to the person typing. Checkpoint timestamps are UTC,
where the same instant can belong to a different day.
"""

from datetime import date, datetime, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from channels import google_chat as gc
from services import config

LA = ZoneInfo("America/Los_Angeles")
UTC = ZoneInfo("UTC")


@pytest.fixture
def thread(monkeypatch):
    """A fake graph whose single thread has a settable last-activity time."""
    state = {"created_at": None, "values": {}, "next": (), "deleted": []}

    class FakeCheckpointer:
        @staticmethod
        def delete_thread(tid):
            state["deleted"].append(tid)

    class FakeGraph:
        checkpointer = FakeCheckpointer()

        @staticmethod
        def get_state(_cfg):
            return SimpleNamespace(
                created_at=state["created_at"], values=state["values"],
                next=state["next"], tasks=(),
            )

    monkeypatch.setattr(gc, "_graph", FakeGraph())
    monkeypatch.setattr(config, "daily_reset", lambda: True)
    return state


def _at(dt: datetime) -> str:
    """A checkpoint timestamp, stored the way LangGraph stores it — UTC ISO."""
    return dt.astimezone(UTC).isoformat()


# ── The rule ───────────────────────────────────────────────────────────────────

def test_a_conversation_from_yesterday_is_discarded(thread):
    thread["created_at"] = _at(datetime.now(LA) - timedelta(days=1))
    assert gc.expire_stale_thread("t1") is not None
    assert thread["deleted"] == ["t1"]


def test_a_conversation_from_today_is_kept(thread):
    thread["created_at"] = _at(datetime.now(LA) - timedelta(minutes=5))
    assert gc.expire_stale_thread("t1") is None
    assert thread["deleted"] == []


def test_no_conversation_means_nothing_to_discard(thread):
    thread["created_at"] = None
    assert gc.expire_stale_thread("t1") is None


def test_a_long_running_conversation_within_one_day_survives(thread):
    """Someone booking since this morning must not be wiped at lunchtime."""
    now = datetime.now(LA)
    thread["created_at"] = _at(now.replace(hour=0, minute=1)) if now.hour > 0 else _at(now)
    assert gc.expire_stale_thread("t1") is None


# ── The boundary is Los Angeles, not UTC ───────────────────────────────────────

@pytest.mark.parametrize("checkpoint_la,today,expired", [
    # 23:58 on the 30th, first message at 00:01 on the 31st. Three minutes
    # apart on the clock, one calendar day apart in LA — it goes.
    (datetime(2026, 8, 30, 23, 58, tzinfo=LA), date(2026, 8, 31), True),
    # 00:01 and 23:58 the SAME day — nearly 24 hours, same date, it stays.
    (datetime(2026, 8, 31, 0, 1, tzinfo=LA), date(2026, 8, 31), False),
    # 16:00 LA on the 30th is 23:00 UTC on the 30th — same day either way.
    (datetime(2026, 8, 30, 16, 0, tzinfo=LA), date(2026, 8, 30), False),
    # 22:00 LA on the 30th is 05:00 UTC on the 31st. Comparing the raw UTC date
    # against the LA date would call this "today" and keep a stale thread.
    (datetime(2026, 8, 30, 22, 0, tzinfo=LA), date(2026, 8, 31), True),
    # 00:30 LA on the 31st is 07:30 UTC on the 31st — both agree, it stays.
    (datetime(2026, 8, 31, 0, 30, tzinfo=LA), date(2026, 8, 31), False),
])
def test_the_day_boundary_is_los_angeles_not_utc(thread, monkeypatch,
                                                 checkpoint_la, today, expired):
    """
    Checkpoints are stored in UTC, which is 7-8 hours ahead of LA. An evening
    conversation therefore lands on the NEXT UTC date, so comparing UTC dates
    would expire threads that are only a couple of hours old.
    """
    monkeypatch.setattr(gc, "_today_la", lambda: today)
    thread["created_at"] = _at(checkpoint_la)
    assert (gc.expire_stale_thread("t1") is not None) is expired


# ── The switch ─────────────────────────────────────────────────────────────────

def test_it_can_be_turned_off(thread, monkeypatch):
    monkeypatch.setattr(config, "daily_reset", lambda: False)
    thread["created_at"] = _at(datetime.now(LA) - timedelta(days=30))
    assert gc.expire_stale_thread("t1") is None
    assert thread["deleted"] == []


def test_the_default_is_on(monkeypatch):
    monkeypatch.delenv("DAILY_RESET", raising=False)
    assert config.daily_reset() is True


@pytest.mark.parametrize("value", ["false", "FALSE", "0", "no", " No "])
def test_falsey_values_turn_it_off(monkeypatch, value):
    monkeypatch.setenv("DAILY_RESET", value)
    assert config.daily_reset() is False


# ── Losing a booking has to be visible ─────────────────────────────────────────

def test_an_unfinished_booking_is_announced(thread):
    """
    Silence here would look exactly like the agent forgetting mid-conversation,
    which is the failure this whole codebase keeps having to design against.
    """
    thread["created_at"] = _at(datetime.now(LA) - timedelta(days=1))
    thread["next"] = ("ask_missing",)
    thread["values"] = {"intake": {"full_name": "Sarah Chen"}}

    notice = gc._expiry_notice(gc.expire_stale_thread("t1"))
    assert "Sarah Chen" in notice
    assert "Nothing was created" in notice


def test_a_finished_conversation_is_cleared_silently(thread):
    """Housekeeping nobody lost anything to. Announcing it is noise."""
    thread["created_at"] = _at(datetime.now(LA) - timedelta(days=1))
    thread["next"] = ()                       # not paused — it ran to the end
    thread["values"] = {"intake": {"full_name": "Sarah Chen"}}

    assert gc._expiry_notice(gc.expire_stale_thread("t1")) == ""


def test_an_idle_chat_with_no_booking_is_cleared_silently(thread):
    thread["created_at"] = _at(datetime.now(LA) - timedelta(days=1))
    thread["next"] = ("ask_missing",)
    thread["values"] = {"intake": {}}
    assert gc._expiry_notice(gc.expire_stale_thread("t1")) == ""


def test_no_expiry_means_no_notice():
    assert gc._expiry_notice(None) == ""


# ── Against a real checkpointer ────────────────────────────────────────────────

def test_the_thread_is_really_gone_afterwards(monkeypatch, tmp_path):
    """The fakes above prove the decision; this proves the deletion."""
    import sqlite3

    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.graph import END, START, StateGraph
    from typing_extensions import TypedDict

    class S(TypedDict):
        n: int

    conn = sqlite3.connect(str(tmp_path / "t.sqlite"), check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    graph = (
        StateGraph(S).add_node("a", lambda s: {"n": s.get("n", 0) + 1})
        .add_edge(START, "a").add_edge("a", END).compile(checkpointer=saver)
    )
    cfg = {"configurable": {"thread_id": "real"}}
    graph.invoke({"n": 0}, cfg)
    assert graph.get_state(cfg).values == {"n": 1}

    monkeypatch.setattr(gc, "_graph", graph)
    assert gc.reset_thread("real") is True
    assert graph.get_state(cfg).values == {}
