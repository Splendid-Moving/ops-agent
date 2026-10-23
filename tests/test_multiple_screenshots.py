"""
Two screenshots attached to ONE Chat message.

The real case: the customer's details are split across two screens, so staff
grab two photos and send them together. Both must reach the model in a SINGLE
call — not two separate extractions merged afterwards.

That distinction is the whole design. Given both images at once the model sees
one job and decides for itself which address is the pickup and which the
drop-off. Extracted separately they would be two half-jobs and the code would
have to referee a conflict it has no context to settle. Where the direction is
genuinely unclear the prompt already requires both confidences to drop, which
sends it to the checklist to ask a human.
"""

import base64

import pytest

from langchain_core.messages import HumanMessage

from agent.nodes.extract_screenshot import _image_parts, _text_of
from channels import google_chat


def _attachment(name: str) -> dict:
    return {"contentType": "image/png", "attachmentDataRef": {"resourceName": name}}


def _event(*names: str, text: str = "") -> dict:
    return {"message": {"text": text, "attachment": [_attachment(n) for n in names]}}


def _fake_download(monkeypatch, payload: bytes = b"png-bytes"):
    monkeypatch.setattr(google_chat, "download_attachment", lambda _: (payload, "image/png"))


# ── The channel must pass every image through ──────────────────────────────────

def test_both_images_reach_the_graph(monkeypatch):
    _fake_download(monkeypatch)
    result = google_chat.build_graph_input(_event("a.png", "b.png"), is_paused=False)
    assert len(_image_parts(result["messages"][0])) == 2


def test_they_arrive_as_one_message_not_two(monkeypatch):
    """
    One message means one extraction call with both images in it. Two messages
    would mean two independent reads and a conflict to referee afterwards.
    """
    _fake_download(monkeypatch)
    result = google_chat.build_graph_input(_event("a.png", "b.png"), is_paused=False)
    assert len(result["messages"]) == 1


def test_a_caption_survives_alongside_the_images(monkeypatch):
    """Staff often add "labor only" or "she wants 3 guys" with the screenshots."""
    _fake_download(monkeypatch)
    result = google_chat.build_graph_input(
        _event("a.png", "b.png", text="labor only"), is_paused=False
    )
    assert _text_of(result["messages"][0]) == "labor only"


def test_non_images_are_left_out(monkeypatch):
    """A PDF among the attachments must not be handed to a vision model."""
    _fake_download(monkeypatch)
    event = _event("a.png")
    event["message"]["attachment"].append(
        {"contentType": "application/pdf", "attachmentDataRef": {"resourceName": "x.pdf"}}
    )
    result = google_chat.build_graph_input(event, is_paused=False)
    assert len(_image_parts(result["messages"][0])) == 1


def test_an_attachment_that_fails_to_download_does_not_lose_the_other(monkeypatch):
    """One bad download must not cost the whole booking."""
    calls = {"n": 0}

    def flaky(_):
        calls["n"] += 1
        return None if calls["n"] == 1 else (b"png-bytes", "image/png")

    monkeypatch.setattr(google_chat, "download_attachment", flaky)
    result = google_chat.build_graph_input(_event("bad.png", "good.png"), is_paused=False)
    assert len(_image_parts(result["messages"][0])) == 1


# ── The extraction node must use all of them ───────────────────────────────────

def test_the_node_reads_every_attached_image(monkeypatch):
    """Not just the first — a second screenshot dropped here is invisible."""
    _fake_download(monkeypatch)
    result = google_chat.build_graph_input(_event("a.png", "b.png"), is_paused=False)
    assert len(_image_parts(result["messages"][0])) == 2


def test_ocr_runs_over_each_image(monkeypatch):
    """
    OCR is what pins down emails and phone numbers. Running it on only the
    first image would silently drop that protection for the second.
    """
    from agent.nodes import extract_screenshot as node

    seen = []
    monkeypatch.setattr(node.ocr, "extract_text", lambda payload: seen.append(payload) or "text")
    monkeypatch.setattr(node.progress, "done", lambda *a, **k: None)

    b64 = base64.b64encode(b"png-bytes").decode()
    images = [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
    ]
    node._ocr_text(images)
    assert len(seen) == 2


# ── A screenshot is never dropped silently ─────────────────────────────────────

def test_a_failed_download_is_retried(monkeypatch):
    """Google's connection drops now and then (SSLEOFError, seen live); retry it."""
    calls = {"n": 0}

    class FlakyDownloader:
        def __init__(self, buffer, request):
            self.buffer = buffer

        def next_chunk(self):
            calls["n"] += 1
            if calls["n"] == 1:
                raise ConnectionError("EOF occurred in violation of protocol")
            self.buffer.write(b"png-bytes")
            return None, True

    class FakeService:
        def media(self):
            return self

        def download_media(self, resourceName):
            return object()

    monkeypatch.setattr(google_chat, "chat_service", lambda: FakeService())
    monkeypatch.setattr(google_chat, "MediaIoBaseDownload", FlakyDownloader)
    monkeypatch.setattr(google_chat, "_sleep", lambda _: None)

    assert google_chat.download_attachment(_attachment("a.png")) == (b"png-bytes", "image/png")
    assert calls["n"] == 2


def test_the_user_is_told_when_a_screenshot_could_not_be_opened(monkeypatch):
    """The live failure: contact screenshot lost, agent asked for name/email/phone with no explanation."""
    calls = {"n": 0}

    def flaky(_):
        calls["n"] += 1
        return None if calls["n"] == 1 else (b"png-bytes", "image/png")

    monkeypatch.setattr(google_chat, "download_attachment", flaky)
    graph_input, notice = google_chat.prepare_input(_event("bad.png", "good.png"), is_paused=False)

    assert len(_image_parts(graph_input["messages"][0])) == 1
    assert "1 of your 2 screenshots" in notice


def test_no_notice_when_every_screenshot_arrives(monkeypatch):
    _fake_download(monkeypatch)
    _, notice = google_chat.prepare_input(_event("a.png", "b.png"), is_paused=False)
    assert notice == ""


def test_a_screenshot_sent_mid_booking_gets_an_explanation_not_silence(monkeypatch):
    """
    The second live failure: re-sending the lost screenshot while the agent was
    waiting on a question did nothing, three times over.
    """
    monkeypatch.setattr(google_chat, "download_attachment",
                        lambda _: pytest.fail("a paused booking cannot take an image"))
    graph_input, notice = google_chat.prepare_input(_event("a.png"), is_paused=True)

    assert graph_input is None
    assert "already in progress" in notice


def test_text_sent_with_a_mid_booking_screenshot_is_still_used(monkeypatch):
    graph_input, notice = google_chat.prepare_input(
        _event("a.png", text="3 movers"), is_paused=True
    )
    assert graph_input.resume == "3 movers"
    assert "already in progress" in notice
