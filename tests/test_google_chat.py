"""
Tests for the Google Chat channel adapter.

Weighted towards the two failure modes that are silent and expensive:

  1. An unstable thread_id, which abandons a half-finished booking without any
     error — the user just sees the agent forget everything they said.
  2. Sending a new message to a paused graph instead of a resume value, which
     does the same thing for a different reason.

Neither shows up as an exception, so neither would be caught without a test.
"""

import base64
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langgraph.types import Command

from channels import google_chat as gc


# ── Fixtures ───────────────────────────────────────────────────────────────────


def message_event(text="hello", *, space="spaces/AAA", threaded=False,
                  thread="spaces/AAA/threads/TTT", attachments=None):
    return {
        "type": "MESSAGE",
        "space": {
            "name": space,
            "spaceThreadingState": "THREADED_MESSAGES" if threaded else "UNTHREADED_MESSAGES",
        },
        "message": {
            "argumentText": text,
            "text": f"@Ops Agent {text}",
            "thread": {"name": thread},
            "attachment": attachments or [],
        },
    }


@pytest.fixture(autouse=True)
def _no_verification(monkeypatch):
    """Most tests are about routing, not auth. Auth has its own tests."""
    monkeypatch.setenv("CHAT_VERIFY_REQUESTS", "false")


# ── thread_id: the booking-continuity rule ────────────────────────────────────


def test_unthreaded_space_keys_on_the_space_not_the_message_thread():
    """
    Direct messages are unthreaded, and Chat gives each message its own thread
    name there. Keying on it would start a fresh conversation for every reply,
    so a multi-turn booking could never complete.
    """
    first = message_event("book this", threaded=False, thread="spaces/AAA/threads/ONE")
    second = message_event("3 movers", threaded=False, thread="spaces/AAA/threads/TWO")

    assert gc.thread_id_for(first) == gc.thread_id_for(second)
    assert gc.thread_id_for(first) == "gchat:spaces/AAA"


def test_threaded_space_keys_on_the_thread_so_bookings_stay_separate():
    """In a real threaded space, two threads are two different conversations."""
    one = message_event(threaded=True, thread="spaces/AAA/threads/ONE")
    two = message_event(threaded=True, thread="spaces/AAA/threads/TWO")

    assert gc.thread_id_for(one) != gc.thread_id_for(two)
    assert gc.thread_id_for(one) == "gchat:spaces/AAA/threads/ONE"


def test_different_spaces_never_share_a_thread():
    a = message_event(space="spaces/AAA")
    b = message_event(space="spaces/BBB")
    assert gc.thread_id_for(a) != gc.thread_id_for(b)


# ── resume vs new message ─────────────────────────────────────────────────────


def test_paused_graph_gets_a_resume_command_not_a_new_message():
    """Sending a message to a paused graph discards the booking in progress."""
    result = gc.build_graph_input(message_event("8-9am, 3 movers"), is_paused=True)

    assert isinstance(result, Command)
    assert result.resume == "8-9am, 3 movers"


def test_fresh_turn_gets_a_message():
    result = gc.build_graph_input(message_event("how many jobs last month"), is_paused=False)

    assert isinstance(result, dict)
    assert result["messages"][0].content == "how many jobs last month"


def test_image_while_paused_is_ignored_rather_than_resumed_with():
    """
    An image sent mid-booking is a new job, not an answer to the pending
    question. Resuming with it would feed a base64 blob into the reply parser.
    """
    event = message_event(
        "here", attachments=[{"contentType": "image/png",
                              "attachmentDataRef": {"resourceName": "x"}}]
    )
    result = gc.build_graph_input(event, is_paused=True)

    assert isinstance(result, Command)
    assert result.resume == "here"


def test_image_on_a_fresh_turn_becomes_a_multimodal_message(monkeypatch):
    monkeypatch.setattr(gc, "download_attachment", lambda a: (b"\x89PNG-bytes", "image/png"))

    event = message_event(
        "book this", attachments=[{"contentType": "image/png",
                                   "attachmentDataRef": {"resourceName": "x"}}]
    )
    result = gc.build_graph_input(event, is_paused=False)

    content = result["messages"][0].content
    assert content[0]["text"] == "book this"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert base64.b64encode(b"\x89PNG-bytes").decode() in content[1]["image_url"]["url"]


def test_non_image_attachments_are_skipped(monkeypatch):
    monkeypatch.setattr(gc, "download_attachment",
                        lambda a: pytest.fail("should not download a PDF"))

    event = message_event("see this", attachments=[{"contentType": "application/pdf"}])
    result = gc.build_graph_input(event, is_paused=False)

    assert result["messages"][0].content == "see this"


def test_bot_mention_is_stripped():
    """The raw text includes '@Ops Agent', which would pollute the model context."""
    assert gc.strip_mention(message_event("book this job")) == "book this job"


# ── the confirm card ──────────────────────────────────────────────────────────


def test_confirm_card_buttons_send_the_strings_the_graph_already_accepts():
    """
    confirm.py accepts plain 'yes'/'no'. Keeping the buttons on that contract is
    what lets the graph stay ignorant that Chat exists.
    """
    card = gc.confirm_card("Here's what I'll create.")
    buttons = card["card"]["sections"][1]["widgets"][0]["buttonList"]["buttons"]

    values = {
        b["onClick"]["action"]["parameters"][0]["value"] for b in buttons
    }
    assert values == {"yes", "no"}
    assert all(b["onClick"]["action"]["function"] == "confirm_decision" for b in buttons)


def test_confirm_card_preserves_line_breaks():
    """Cards render HTML — raw newlines collapse and the summary becomes a wall."""
    card = gc.confirm_card("line one\nline two")
    text = card["card"]["sections"][0]["widgets"][0]["textParagraph"]["text"]
    assert "<br>" in text
    assert "\n" not in text


def test_confirm_card_escapes_html_so_customer_data_cannot_break_it():
    card = gc.confirm_card("Notes: <b>fragile</b> & heavy")
    text = card["card"]["sections"][0]["widgets"][0]["textParagraph"]["text"]
    assert "&lt;b&gt;" in text
    assert "&amp;" in text


# ── authentication ────────────────────────────────────────────────────────────


def test_missing_bearer_token_is_rejected(monkeypatch):
    monkeypatch.setenv("CHAT_VERIFY_REQUESTS", "true")
    monkeypatch.setenv("GOOGLE_CHAT_AUDIENCE", "1234567890")
    assert gc.verify_request(None) is False
    assert gc.verify_request("not-a-bearer") is False


def test_verification_fails_closed_when_audience_is_unset(monkeypatch):
    """
    A missing audience must reject, never wave requests through — this endpoint
    is public and everything behind it writes to a real CRM.
    """
    monkeypatch.setenv("CHAT_VERIFY_REQUESTS", "true")
    monkeypatch.setenv("GOOGLE_CHAT_AUDIENCE", "")
    assert gc.verify_request("Bearer anything") is False


def test_a_forged_token_is_rejected(monkeypatch):
    monkeypatch.setenv("CHAT_VERIFY_REQUESTS", "true")
    monkeypatch.setenv("GOOGLE_CHAT_AUDIENCE", "1234567890")
    assert gc.verify_request("Bearer totally.made.up") is False


def test_workspace_addon_issuer_is_accepted(monkeypatch):
    """
    Chat apps built as Workspace add-ons sign as
    service-<project>@gcp-sa-gsuiteaddons.iam.gserviceaccount.com, not
    chat@system. Accepting only chat@system rejected every real request while
    the audience matched perfectly — which is indistinguishable from a
    correctly-configured app being mysteriously ignored.
    """
    monkeypatch.delenv("GOOGLE_CHAT_ISSUER", raising=False)
    from services import config

    assert config.issuer_is_google(
        "service-326684148481@gcp-sa-gsuiteaddons.iam.gserviceaccount.com"
    )
    assert config.issuer_is_google("chat@system.gserviceaccount.com")


def test_non_google_issuers_are_rejected(monkeypatch):
    monkeypatch.delenv("GOOGLE_CHAT_ISSUER", raising=False)
    from services import config

    assert not config.issuer_is_google("attacker@evil.com")
    assert not config.issuer_is_google("")
    # Lookalike domain must not pass on a substring match.
    assert not config.issuer_is_google("x@gcp-sa-gsuiteaddons.iam.gserviceaccount.com.evil.com")


def test_pinned_issuer_overrides_the_default_pair(monkeypatch):
    monkeypatch.setenv("GOOGLE_CHAT_ISSUER", "service-123@gcp-sa-gsuiteaddons.iam.gserviceaccount.com")
    from services import config

    assert config.issuer_is_google("service-123@gcp-sa-gsuiteaddons.iam.gserviceaccount.com")
    # Pinning means pinning — chat@system is no longer implicitly trusted.
    assert not config.issuer_is_google("chat@system.gserviceaccount.com")


def test_an_unexpected_issuer_is_recorded_never_silent(monkeypatch):
    """
    The original bug: a verified token from an unexpected signer returned False
    with no log and no record, so the 401 had no discoverable cause.
    """
    monkeypatch.setenv("CHAT_VERIFY_REQUESTS", "true")
    monkeypatch.setenv("GOOGLE_CHAT_AUDIENCE", "https://example.com/google-chat")
    monkeypatch.delenv("GOOGLE_CHAT_ISSUER", raising=False)

    monkeypatch.setattr(
        gc.google_id_token, "verify_oauth2_token",
        lambda *a, **kw: {"email": "someone-else@example.com", "aud": "https://example.com/google-chat"},
    )

    assert gc.verify_request("Bearer whatever") is False

    recorded = gc.last_rejection()
    assert recorded["reason"] == "unexpected_issuer"
    assert recorded["token_issuer"] == "someone-else@example.com"


def test_webhook_returns_401_when_unverified(monkeypatch):
    monkeypatch.setenv("CHAT_VERIFY_REQUESTS", "true")
    monkeypatch.setenv("GOOGLE_CHAT_AUDIENCE", "1234567890")

    app = FastAPI()
    app.include_router(gc.router)
    client = TestClient(app)

    response = client.post("/google-chat", json=message_event())
    assert response.status_code == 401


# ── webhook dispatch ──────────────────────────────────────────────────────────


@pytest.fixture
def client(monkeypatch):
    """Run background work inline so tests can assert on it."""
    calls = []
    monkeypatch.setattr(gc, "_spawn", lambda fn, *a, **kw: calls.append((a, kw)))

    app = FastAPI()
    app.include_router(gc.router)
    c = TestClient(app)
    c.spawned = calls
    return c


def test_message_event_is_processed_in_the_background(client):
    """
    The webhook must return immediately. Chat abandons the request at 30s, and
    a screenshot intake routinely takes longer than that.
    """
    response = client.post("/google-chat", json=message_event("hi"))

    assert response.status_code == 200
    assert len(client.spawned) == 1


def test_added_to_space_replies_synchronously_without_running_the_graph(client):
    event = {"type": "ADDED_TO_SPACE", "space": {"name": "spaces/AAA"}}
    response = client.post("/google-chat", json=event)

    assert "Splendid Moving" in response.json()["text"]
    assert client.spawned == []


def test_button_click_resumes_with_the_button_value(client):
    event = {
        **message_event(),
        "type": "CARD_CLICKED",
        "common": {"invokedFunction": "confirm_decision"},
        "action": {
            "actionMethodName": "confirm_decision",
            "parameters": [{"key": "decision", "value": "yes"}],
        },
    }
    response = client.post("/google-chat", json=event)

    assert response.status_code == 200
    (args, _), = client.spawned
    assert args[1] == "yes"


def test_unknown_event_types_are_ignored(client):
    response = client.post("/google-chat", json={"type": "SOMETHING_NEW"})
    assert response.status_code == 200
    assert client.spawned == []
