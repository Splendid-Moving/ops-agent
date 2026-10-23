"""
Turns a technical error into a short plain-English explanation for staff.

Every message answers two questions: what went wrong, and whether waiting or
trying again will fix it. Callers add the third — what did NOT happen because
of it — since only they know. The raw error still goes to the logs.
"""

import re
import ssl
from dataclasses import dataclass


@dataclass(frozen=True)
class Explanation:
    problem: str   # "OpenAI (the AI that reads screenshots) is out of credits."
    fix: str       # "Waiting won't help — someone needs to add credits…"

    def __str__(self) -> str:
        return f"{self.problem} {self.fix}"


def _service(exc: BaseException, text: str) -> str:
    """Best guess at which outside service failed, from where the error came from."""
    origin = f"{type(exc).__module__} {text}".lower()
    if "openai" in origin or "openrouter" in origin:
        return "The AI service (OpenAI)"
    if "ghl" in origin or "leadconnector" in origin or "gohighlevel" in origin:
        return "GoHighLevel"
    if "jobsapp" in type(exc).__name__.lower() or "ops app" in origin:
        return "The crew app"
    if "google" in origin or "httplib2" in origin:
        return "Google"
    if "vercel" in origin:
        return "Vercel"
    return ""


def explain(exc: BaseException) -> Explanation:
    text = str(exc)
    low = text.lower()
    name = type(exc).__name__
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if not isinstance(status, int):
        # GHL and OpenAI also write the HTTP code into the message: "[401]", "Error code: 429".
        found = re.search(r"\[(\d{3})\]|error code:?\s*(\d{3})", low)
        status = int(found.group(1) or found.group(2)) if found else None
    service = _service(exc, text)
    who = service or "An outside service"

    if "insufficient_quota" in low or "no credits" in low or "credit_balance_exhausted" in low:
        return Explanation(
            f"{service or 'The AI service'} is out of credits.",
            "Waiting won't help — someone needs to add credits to the account, then try again.",
        )

    if status in (401, 403) or "unauthorized" in low or "invalid_api_key" in low \
            or "invalid_grant" in low:
        return Explanation(
            f"{who} rejected our login.",
            "Waiting won't help — the API key or credentials need updating.",
        )

    if status == 429 or "ratelimit" in name.lower() or "rate limit" in low:
        return Explanation(
            f"{who} is getting too many requests right now.",
            "Wait a minute and try again.",
        )

    if isinstance(exc, (TimeoutError, ConnectionError, ssl.SSLError)) \
            or "timeout" in name.lower() or "connection" in name.lower() \
            or "timed out" in low or (isinstance(status, int) and status >= 500):
        return Explanation(
            f"Couldn't get through to {service or 'an outside service'} — a network hiccup or a brief outage on their side.",
            "Usually fixed by trying again in a minute.",
        )

    # Unknown: keep a short slice of the real message. Services like GHL put the
    # useful part there ("Issue date cannot be after due date").
    detail = text.strip().splitlines()[0][:200] if text.strip() else name
    return Explanation(
        f"Something unexpected went wrong ({detail}).",
        "Try once more; if it happens again, it needs someone to look at it.",
    )
