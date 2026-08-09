"""
Routing + state tests.

The routing cases hit a real model, so they are marked `live` and excluded from
the default run:
    pytest                 # fast, no network
    pytest -m live         # includes model calls
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agent import state as st
from agent.nodes.router import _has_image, route


# ── Ledger reducer (pure) ──────────────────────────────────────────────────────

def test_new_ledger_starts_all_pending():
    ledger = st.new_ledger()
    assert set(ledger) == set(st.ALL_ACTIONS)
    assert all(v["status"] == "pending" for v in ledger.values())
    assert not st.all_done(ledger)


def test_parallel_branches_do_not_clobber_each_other():
    """
    The three post-contact actions run in parallel, each returning only its own
    key. Without the reducer, two of the three results would be lost.
    """
    base = st.new_ledger()
    calendar = {st.ACTION_CALENDAR: {"status": "success", "result": {"event_id": "e1"}}}
    invoice = {st.ACTION_INVOICE: {"status": "success", "result": {"invoice_id": "i1"}}}
    email = {st.ACTION_EMAIL: {"status": "failed", "error": "boom"}}

    merged = base
    for update in (calendar, invoice, email):
        merged = st.merge_ledger(merged, update)

    assert merged[st.ACTION_CALENDAR]["result"]["event_id"] == "e1"
    assert merged[st.ACTION_INVOICE]["result"]["invoice_id"] == "i1"
    assert merged[st.ACTION_EMAIL]["status"] == "failed"
    assert merged[st.ACTION_CONTACT]["status"] == "pending"


def test_retry_replaces_a_failed_entry():
    ledger = st.merge_ledger(
        st.new_ledger(), {st.ACTION_EMAIL: {"status": "failed", "error": "timeout", "attempts": 1}}
    )
    assert st.failed_actions(ledger) == [st.ACTION_EMAIL]

    ledger = st.merge_ledger(
        ledger, {st.ACTION_EMAIL: {"status": "success", "result": {"sent": True}, "attempts": 2}}
    )
    assert st.failed_actions(ledger) == []
    assert ledger[st.ACTION_EMAIL]["attempts"] == 2


def test_succeeded_guard_is_what_prevents_double_firing():
    """Each act_* node checks this before doing anything. It is the idempotency latch."""
    ledger = st.merge_ledger(
        st.new_ledger(), {st.ACTION_CONTACT: {"status": "success", "result": {"contact_id": "c1"}}}
    )
    assert st.succeeded(ledger, st.ACTION_CONTACT)
    assert not st.succeeded(ledger, st.ACTION_CALENDAR)


def test_all_done_accepts_skipped():
    ledger = {name: {"status": "success"} for name in st.ALL_ACTIONS}
    assert st.all_done(ledger)
    ledger[st.ACTION_EMAIL] = {"status": "skipped"}
    assert st.all_done(ledger)
    ledger[st.ACTION_EMAIL] = {"status": "failed"}
    assert not st.all_done(ledger)


def test_merge_handles_empty_sides():
    assert st.merge_ledger(None, None) == {}
    assert st.merge_ledger(None, {"a": {"status": "success"}}) == {"a": {"status": "success"}}
    assert st.merge_ledger({"a": {"status": "success"}}, None) == {"a": {"status": "success"}}


# ── Image detection (pure) ─────────────────────────────────────────────────────

def test_detects_image_content_block():
    msg = HumanMessage(
        content=[
            {"type": "text", "text": "book this"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,xxx"}},
        ]
    )
    assert _has_image(msg)


def test_plain_text_is_not_an_image():
    assert not _has_image(HumanMessage(content="how many jobs last month?"))
    assert not _has_image(AIMessage(content="hello"))


# ── Routing (live model) ───────────────────────────────────────────────────────

ROUTING_CASES = [
    ("how many jobs did we have last month?", "analytics"),
    ("what's on the calendar Friday?", "analytics"),
    ("how many Yelp jobs in July?", "analytics"),
    ("who are we moving tomorrow?", "analytics"),
    ("how many labor jobs this year", "analytics"),
    ("new job for Sarah Chen moving next Friday", "intake"),
    ("book this one", "intake"),
    ("hey", "chat"),
    ("what can you do?", "chat"),
    ("never mind", "chat"),
    ("thanks!", "chat"),
    # Contentless booking requests belong in chat, not intake. There is nothing
    # to extract, so the right move is to ask for the screenshot rather than
    # open a booking with no data in it. In real use these arrive WITH an image,
    # and _has_image short-circuits them to intake before the model is consulted.
    ("can you set this customer up", "chat"),
    ("add a new job", "chat"),
]


@pytest.mark.live
@pytest.mark.parametrize("message,expected", ROUTING_CASES)
def test_router_picks_the_right_lane(message, expected):
    result = route({"messages": [HumanMessage(content=message)]})
    assert result["intent"] == expected, f"{message!r} -> {result['intent']}, wanted {expected}"


@pytest.mark.live
def test_image_routes_to_intake_without_a_model_call():
    """An attachment is decisive on its own — no need to spend a model call."""
    msg = HumanMessage(
        content=[
            {"type": "text", "text": ""},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,xxx"}},
        ]
    )
    assert route({"messages": [msg]})["intent"] == "intake"


def test_empty_conversation_routes_to_chat():
    assert route({"messages": []})["intent"] == "chat"
