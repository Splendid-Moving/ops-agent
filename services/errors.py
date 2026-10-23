"""
Turns a technical error into a short plain-English explanation for staff.

Every message answers two questions, briefly: what went wrong, and what fixes it. Callers add the third — what did NOT happen because
of it — since only they know. The raw error still goes to the logs.
"""

import re
import ssl
from dataclasses import dataclass


@dataclass(frozen=True)
class Explanation:
    problem: str   # "OpenAI is out of credits."
    fix: str       # "Someone needs to top up the account."

    def __str__(self) -> str:
        return f"{self.problem} {self.fix}"


def _service(exc: BaseException, text: str) -> str:
    """Best guess at which outside service failed, from where the error came from."""
    origin = f"{type(exc).__module__} {text}".lower()
    if "openai" in origin or "openrouter" in origin:
        return "OpenAI"
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
            "Someone needs to top up the account.",
        )

    if status in (401, 403) or "unauthorized" in low or "invalid_api_key" in low \
            or "invalid_grant" in low:
        return Explanation(
            f"{who} rejected our login.",
            "The API key needs updating.",
        )

    if status == 429 or "ratelimit" in name.lower() or "rate limit" in low:
        return Explanation(
            f"{who} is busy.",
            "Try again in a minute.",
        )

    if isinstance(exc, (TimeoutError, ConnectionError, ssl.SSLError)) \
            or "timeout" in name.lower() or "connection" in name.lower() \
            or "timed out" in low or (isinstance(status, int) and status >= 500):
        return Explanation(
            f"Couldn't reach {service or 'an outside service'}.",
            "Try again in a minute.",
        )

    # Unknown: keep a short slice of the real message. Services like GHL put the
    # useful part there ("Issue date cannot be after due date").
    detail = text.strip().splitlines()[0][:200] if text.strip() else name
    return Explanation(
        f"Something went wrong ({detail}).",
        "Try again; if it repeats, someone needs to look at it.",
    )
