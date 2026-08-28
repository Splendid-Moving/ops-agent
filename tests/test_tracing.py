"""
Tracing config.

The thing actually being protected here is NOT the tracing. It is the
`thread_id`. `tracing.run_config()` replaced the hand-written
`{"configurable": {"thread_id": ...}}` at all five graph call sites, and that
dict is what lets a paused booking be resumed. If tracing labels ever displace
it, every reply silently starts a new conversation and discards a half-finished
job — the worst failure this app has, and one that looks like forgetfulness
rather than an error.

So: the first test is the load-bearing one. The rest guard the labels that make
a trace readable, with `dry_run` the one that matters most — a dry-run trace and
a live trace look identical, and only one of them charged a real customer.
"""

import pytest

from services import tracing


# ── The part that must never break ─────────────────────────────────────────────

def test_thread_id_survives_in_configurable():
    cfg = tracing.run_config("thread-abc", channel=tracing.CHANNEL_WEB)
    assert cfg["configurable"]["thread_id"] == "thread-abc"


def test_resume_keeps_the_same_thread_id():
    """A relabel must never move the thread — that would orphan the booking."""
    cfg = tracing.run_config("thread-abc", channel=tracing.CHANNEL_GOOGLE_CHAT)
    assert tracing.as_resume(cfg)["configurable"]["thread_id"] == "thread-abc"


# ── Labels ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("dry, expected_tag", [(True, "dry_run"), (False, "live")])
def test_dry_run_is_on_every_run(monkeypatch, dry, expected_tag):
    """
    Never read a trace from this agent without checking this. The two modes are
    indistinguishable in the trace itself.
    """
    monkeypatch.setenv("DRY_RUN", "true" if dry else "false")
    cfg = tracing.run_config("t", channel=tracing.CHANNEL_WEB)
    assert cfg["metadata"]["dry_run"] is dry
    assert expected_tag in cfg["tags"]


def test_thread_id_is_duplicated_into_metadata():
    """LangSmith's Threads view groups on a metadata key of this exact name."""
    cfg = tracing.run_config("t-9", channel=tracing.CHANNEL_CLI)
    assert cfg["metadata"]["thread_id"] == "t-9"


def test_run_name_identifies_channel_and_turn():
    cfg = tracing.run_config("t", channel=tracing.CHANNEL_GOOGLE_CHAT,
                             turn=tracing.TURN_BUTTON)
    assert cfg["run_name"] == "ops-agent:google_chat:button"


def test_caller_metadata_is_merged():
    cfg = tracing.run_config("t", channel=tracing.CHANNEL_WEB, has_image=True)
    assert cfg["metadata"]["has_image"] is True


def test_as_resume_relabels_without_losing_caller_metadata():
    """
    The channel is only known at the original call site, and extra fields like
    the Chat space are only known there too. Losing either on resume would make
    the second half of a booking untraceable back to where it came from.
    """
    cfg = tracing.run_config(
        "t", channel=tracing.CHANNEL_GOOGLE_CHAT, chat_space="spaces/AAA"
    )
    resumed = tracing.as_resume(cfg)

    assert resumed["metadata"]["turn"] == tracing.TURN_RESUME
    assert resumed["metadata"]["channel"] == tracing.CHANNEL_GOOGLE_CHAT
    assert resumed["metadata"]["chat_space"] == "spaces/AAA"
    assert "turn:resume" in resumed["tags"]
    assert cfg["metadata"]["turn"] == tracing.TURN_MESSAGE, "original was mutated"


# ── Startup check ──────────────────────────────────────────────────────────────

def test_off_by_default(monkeypatch):
    """Tracing uploads real customer data, so a missing value must mean off."""
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    assert tracing.configure().startswith("LangSmith tracing: OFF")


def test_warns_when_enabled_without_a_key(monkeypatch):
    """
    The one failure mode worth a test: the SDK uploads from a background thread
    and swallows its own errors, so a missing key is otherwise completely
    silent — the app runs fine and the project just stays empty.
    """
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "")
    assert "LANGSMITH_API_KEY is empty" in tracing.configure()


def test_status_never_leaks_the_key(monkeypatch):
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_pt_secret")
    assert "lsv2_pt_secret" not in str(tracing.status())
    assert tracing.status()["api_key_present"] is True


# ── Redaction ──────────────────────────────────────────────────────────────────

def test_screenshot_is_not_uploaded():
    """
    The image is a picture of a customer's private message thread, and a
    multi-megabyte base64 string is useless in the UI regardless. Only its size
    should ever leave the machine.
    """
    from services import ocr

    redacted = ocr._redact_image({"image_b64": "A" * 4096})
    assert "AAAA" not in redacted["image_b64"]
    assert "4096" in redacted["image_b64"]
