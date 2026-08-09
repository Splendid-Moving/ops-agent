"""
Character-accurate text extraction, via Google Cloud Vision.

WHY THIS EXISTS ALONGSIDE THE VISION MODEL
------------------------------------------
GPT's vision does not transcribe, it reconstructs. That is exactly why it
handles messy screenshots, odd layouts and handwriting well — and exactly why
it occasionally smooths an unusual string toward a more plausible one. Measured
live, it read a real customer's address as:

    nikitatitarev354@gmail.com   ->   nikitatitrev354@gmail.com

One dropped character. Every other field in this agent has redundancy that
catches such a slip: a bad ZIP fails address validation, a bad street fails to
geocode, a bad city trips the state check. An email has none. Every character is
load-bearing and the result is still a perfectly well-formed address, so the
regex validator passes it and the confidence score never flags it — the model
was not hesitating between readings, it simply read it wrong and felt fine.

OCR has the opposite profile: it reports the characters actually present, and
understands nothing. So the division of labour is:

    OCR   -> which characters are on screen        (exact)
    GPT   -> which string belongs in which field   (understanding)

This module supplies the first half. The correction logic in
extract_screenshot.py applies it only to the high-entropy fields, where OCR is
strictly better and GPT's context adds nothing.
"""

import base64
import json
import logging
import re
import threading

import google.auth.transport.requests as google_requests
import requests
from google.oauth2 import service_account

from services import config

logger = logging.getLogger(__name__)

_ENDPOINT = "https://vision.googleapis.com/v1/images:annotate"
_TIMEOUT = 20
_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]

# Authenticated with the SERVICE ACCOUNT, not the Maps API key.
#
# The Maps key is restricted to a specific list of APIs (Google reports
# API_KEY_SERVICE_BLOCKED), and widening that list means editing a credential
# several other production repos depend on. The service account already
# authenticates Calendar, needs no console changes, and is the right mechanism
# for server-to-server calls anyway — API keys are for clients that cannot keep
# a secret, which is not this.
_creds = None
_creds_lock = threading.Lock()


def _token() -> str:
    """Cached OAuth token, refreshed as needed. Thread-safe for the web server."""
    global _creds
    with _creds_lock:
        if _creds is None:
            raw = config.google_credentials_b64()
            if not raw:
                raise OCRUnavailable("GOOGLE_CREDENTIALS_B64 is not set")
            info = json.loads(base64.b64decode(raw).decode())
            _creds = service_account.Credentials.from_service_account_info(
                info, scopes=_SCOPES
            )
        if not _creds.valid:
            _creds.refresh(google_requests.Request())
        return _creds.token

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
#: US phone numbers in any of the shapes people actually type.
PHONE_RE = re.compile(r"(?:\+?1[\s.\-]?)?\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}")


class OCRUnavailable(RuntimeError):
    """Cloud Vision could not be reached. An operator problem, not a data one."""


def extract_text(image_b64: str) -> str:
    """
    All text in an image, as one string. `image_b64` is raw base64 — no data
    URL prefix.

    Raises OCRUnavailable so callers can degrade to GPT-only extraction rather
    than failing a booking.
    """
    try:
        resp = requests.post(
            _ENDPOINT,
            headers={"Authorization": f"Bearer {_token()}"},
            json={
                "requests": [
                    {
                        "image": {"content": image_b64},
                        # DOCUMENT_TEXT_DETECTION is tuned for dense text like
                        # message threads and forms, which is what these are.
                        "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
                    }
                ]
            },
            timeout=_TIMEOUT,
        )
    except OCRUnavailable:
        raise
    except Exception as exc:
        raise OCRUnavailable(f"Cloud Vision unreachable: {exc}") from exc

    if resp.status_code in (401, 403):
        raise OCRUnavailable(
            "Cloud Vision rejected the service account. Check that the Cloud "
            "Vision API is enabled on the service account's project "
            f"and that {'the account'} has the Cloud Vision API User role. "
            f"Response: {resp.text[:200]}"
        )
    if not resp.ok:
        raise OCRUnavailable(f"Cloud Vision returned HTTP {resp.status_code}: {resp.text[:200]}")

    responses = resp.json().get("responses") or [{}]
    if error := responses[0].get("error"):
        raise OCRUnavailable(f"Cloud Vision error: {error.get('message', error)}")

    return responses[0].get("fullTextAnnotation", {}).get("text", "")


def emails_in(text: str) -> list[str]:
    return EMAIL_RE.findall(text or "")


def phones_in(text: str) -> list[str]:
    return PHONE_RE.findall(text or "")


def _similarity(a: str, b: str) -> float:
    """Character-level similarity, 0-1. Cheap stdlib ratio, no extra dependency."""
    from difflib import SequenceMatcher

    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def best_match(candidate: str, options: list[str], *, threshold: float = 0.7) -> str | None:
    """
    The option most similar to `candidate`, or None if nothing is close.

    The threshold guards against replacing a correctly-read email with an
    unrelated one that happened to be in the same screenshot — support
    addresses, "no-reply@", a second person on the thread.
    """
    if not candidate or not options:
        return None
    scored = sorted(((_similarity(candidate, o), o) for o in options), reverse=True)
    score, winner = scored[0]
    return winner if score >= threshold else None


def correct_email(gpt_value: str, ocr_text: str) -> tuple[str, str | None]:
    """
    Reconcile GPT's email against what OCR found in the pixels.

    Returns (value_to_use, note). The note is non-None only when a correction
    was made, so it can be logged and surfaced.
    """
    if not gpt_value:
        # GPT missed it entirely but OCR may have caught it. Only safe when the
        # screenshot contains exactly one email — otherwise there is no basis
        # for choosing which belongs to the customer.
        found = emails_in(ocr_text)
        if len(found) == 1:
            return found[0], f"OCR supplied an email GPT missed: {found[0]}"
        return "", None

    match = best_match(gpt_value, emails_in(ocr_text))
    if match and match != gpt_value:
        return match, f"OCR corrected {gpt_value!r} -> {match!r}"
    return gpt_value, None


def correct_phone(gpt_value: str, ocr_text: str) -> tuple[str, str | None]:
    """
    Same idea for phone numbers, compared on digits only so formatting
    differences are not mistaken for transcription errors.
    """
    if not gpt_value:
        return "", None

    def digits(s: str) -> str:
        d = re.sub(r"\D", "", s)
        return d[-10:] if len(d) >= 10 else d

    target = digits(gpt_value)
    if not target:
        return gpt_value, None

    for found in phones_in(ocr_text):
        if digits(found) == target:
            return gpt_value, None  # agreement, keep GPT's formatting

    close = best_match(target, [digits(p) for p in phones_in(ocr_text)], threshold=0.8)
    if close and close != target:
        return close, f"OCR corrected phone {target} -> {close}"
    return gpt_value, None
