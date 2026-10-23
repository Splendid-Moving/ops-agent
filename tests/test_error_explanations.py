"""
Failures must reach staff as plain English: what went wrong, what didn't
happen, and whether trying again will help.

The live case: OpenAI ran out of credits, the screenshot read failed silently,
and the agent asked for every field as if the image were blank.
"""

import ssl

from langchain_core.messages import HumanMessage

from agent.nodes import extract_screenshot as node
from channels import google_chat
from services import errors


def _error(module: str, message: str, status: int | None = None) -> Exception:
    """An exception that looks like it came from `module`, the way real SDK errors do."""
    cls = type("FakeError", (Exception,), {"__module__": module})
    exc = cls(message)
    if status is not None:
        exc.status_code = status
    return exc


OUT_OF_CREDITS = _error(
    "openai", "Error code: 429 - {'error': {'message': 'You have no credits remaining.', "
    "'code': 'credit_balance_exhausted'}}", status=429,
)


# ── the translator ────────────────────────────────────────────────────────────

def test_out_of_credits_says_to_top_up():
    why = errors.explain(OUT_OF_CREDITS)
    assert why.problem == "OpenAI is out of credits."
    assert "top up" in why.fix


def test_a_plain_rate_limit_says_wait_and_retry():
    why = errors.explain(_error("openai", "Rate limit reached", status=429))
    assert "busy" in why.problem
    assert "Try again" in why.fix


def test_a_dropped_connection_says_try_again():
    why = errors.explain(ssl.SSLEOFError("EOF occurred in violation of protocol"))
    assert "Couldn't reach" in why.problem
    assert "Try again" in why.fix


def test_a_bad_login_names_the_service():
    why = errors.explain(_error("services.ghl", "Contact search failed [401] Invalid JWT"))
    assert why.problem == "GoHighLevel rejected our login."


def test_an_unknown_error_keeps_the_useful_part_of_the_message():
    """GHL explains itself in the message body — that must not be thrown away."""
    why = errors.explain(_error("services.ghl", "Invoice failed [422] Issue date cannot be after due date"))
    assert "Issue date cannot be after due date" in why.problem


# ── the screenshot reader ─────────────────────────────────────────────────────

def _screenshot_state(**extra) -> dict:
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
    return {"messages": [HumanMessage(content=[{"type": "text", "text": "Book this job."}, image])],
            **extra}


def _failing_model(monkeypatch, exc: Exception):
    class Model:
        def with_structured_output(self, _):
            return self

        def invoke(self, _):
            raise exc

    monkeypatch.setattr(node, "get_model", lambda _: Model())
    monkeypatch.setattr(node, "_ocr_text", lambda _: "")


def test_a_failed_screenshot_read_explains_itself(monkeypatch):
    _failing_model(monkeypatch, OUT_OF_CREDITS)
    out = node.extract_screenshot(_screenshot_state())
    aside = out["intake"]["_aside"]
    assert "Couldn't read the screenshot" in aside
    assert "out of credits" in aside


def test_a_failed_read_does_not_bring_back_the_previous_customer(monkeypatch):
    """A new screenshot is a new job, even when reading it fails."""
    _failing_model(monkeypatch, OUT_OF_CREDITS)
    out = node.extract_screenshot(_screenshot_state(intake={"full_name": "Previous Customer"}))
    assert "full_name" not in out["intake"]


# ── a turn that fails outright ────────────────────────────────────────────────

def test_a_crash_says_what_did_not_happen():
    text = google_chat.crash_text(OUT_OF_CREDITS)
    assert "nothing was booked or sent" in text
    assert "Error code" not in text
