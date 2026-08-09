"""
OCR reconciliation.

The case this exists for, measured on a real booking: GPT read a customer's
email as nikitatitrev354@gmail.com when the screenshot said
nikitatitarev354@gmail.com. One dropped character, a still-valid address, and
no downstream check anywhere that could notice.
"""

import pytest

from services import ocr

# What OCR would return for that screenshot.
OCR_TEXT = """Matt Daemon
nikitatitarev354@gmail.com
(818) 505-4576
From: 614 E Verdugo Ave
To: 1039 Justin Ave, Glendale
Moving next Saturday
"""


# ── The real bug ───────────────────────────────────────────────────────────────

def test_the_live_email_bug_is_corrected():
    corrected, note = ocr.correct_email("nikitatitrev354@gmail.com", OCR_TEXT)
    assert corrected == "nikitatitarev354@gmail.com"
    assert note and "corrected" in note


def test_a_correct_email_is_left_alone_and_reported_as_unchanged():
    corrected, note = ocr.correct_email("nikitatitarev354@gmail.com", OCR_TEXT)
    assert corrected == "nikitatitarev354@gmail.com"
    assert note is None


# ── Not over-correcting ────────────────────────────────────────────────────────

def test_an_unrelated_email_is_not_swapped_in():
    """
    A screenshot often contains support@, no-reply@, or a second person on the
    thread. Replacing a correctly-read address with one of those would be far
    worse than the bug being fixed.
    """
    text = "support@yelp.com\nno-reply@example.com"
    corrected, note = ocr.correct_email("sarah.chen@gmail.com", text)
    assert corrected == "sarah.chen@gmail.com"
    assert note is None


def test_ocr_supplies_an_email_gpt_missed_only_when_unambiguous():
    corrected, note = ocr.correct_email("", "contact: solo@example.com")
    assert corrected == "solo@example.com"
    assert note is not None


def test_ocr_does_not_guess_when_several_emails_are_present():
    """With two candidates there is no basis for choosing the customer's."""
    corrected, note = ocr.correct_email("", "a@example.com and b@example.com")
    assert corrected == ""
    assert note is None


def test_no_ocr_text_changes_nothing():
    corrected, note = ocr.correct_email("sarah@example.com", "")
    assert corrected == "sarah@example.com"
    assert note is None


# ── Phone ──────────────────────────────────────────────────────────────────────

def test_matching_phone_keeps_gpt_formatting():
    """Same digits in a different shape is agreement, not an error."""
    corrected, note = ocr.correct_phone("+1(818)505-4576", OCR_TEXT)
    assert corrected == "+1(818)505-4576"
    assert note is None


def test_phone_digit_slip_is_corrected():
    corrected, note = ocr.correct_phone("+1(818)505-4576", "call me on (818) 505-4567")
    assert "8185054567" in corrected.replace("-", "").replace("(", "").replace(")", "")
    assert note is not None


# ── Finders ────────────────────────────────────────────────────────────────────

def test_finds_emails_in_noisy_text():
    assert ocr.emails_in(OCR_TEXT) == ["nikitatitarev354@gmail.com"]


@pytest.mark.parametrize(
    "text",
    ["(818) 505-4576", "818-505-4576", "+1 818 505 4576", "8185054576"],
)
def test_finds_phones_in_common_shapes(text):
    assert ocr.phones_in(text)


def test_best_match_respects_the_threshold():
    assert ocr.best_match("abc@x.com", ["completely@different.org"]) is None
    assert ocr.best_match("abc@x.com", ["abd@x.com"]) == "abd@x.com"


def test_best_match_handles_empty_input():
    assert ocr.best_match("", ["a@b.com"]) is None
    assert ocr.best_match("a@b.com", []) is None


# ── Live ───────────────────────────────────────────────────────────────────────

@pytest.mark.live
def test_cloud_vision_is_reachable():
    """
    Skips cleanly while the API is still being enabled, so the suite stays green
    either way.
    """
    import base64

    png = base64.b64encode(bytes.fromhex(
        "89504e470d0a1a0a0000000d494844520000000100000001080600000"
        "01f15c4890000000a49444154789c63000100000500010d0a2db4"
        "0000000049454e44ae426082"
    )).decode()
    try:
        ocr.extract_text(png)
    except ocr.OCRUnavailable as exc:
        pytest.skip(f"Cloud Vision not enabled yet: {exc}")
