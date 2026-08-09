"""
Central config. Every env var the agent reads is declared here and nowhere else.

Values are read lazily via functions rather than captured at import time, so that
`langgraph dev` hot-reload and tests can change the environment without a restart.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root regardless of where python was invoked from.
_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")


# ── GoHighLevel ────────────────────────────────────────────────────────────────

GHL_BASE_URL = "https://services.leadconnectorhq.com"
GHL_API_VERSION = "2021-07-28"


def ghl_token() -> str:
    return os.getenv("GHL_ACCESS_TOKEN", "")


def ghl_location_id() -> str:
    return os.getenv("GHL_LOCATION_ID", "")


def ghl_user_id() -> str:
    return os.getenv("GHL_USER_ID", "")


# ── Google ─────────────────────────────────────────────────────────────────────

def google_credentials_b64() -> str:
    return os.getenv("GOOGLE_CREDENTIALS_B64", "")


def calendar_id() -> str:
    return os.getenv("GOOGLE_CALENDAR_ID", "primary")


def maps_api_key() -> str:
    return os.getenv("GOOGLE_MAPS_API_KEY", "")


# ── Google Chat ────────────────────────────────────────────────────────────────

#: Classic Chat apps sign webhook requests with this service account.
CHAT_ISSUER = "chat@system.gserviceaccount.com"

#: Chat apps built as Google Workspace add-ons sign with a per-project account
#: instead: service-<PROJECT_NUMBER>@gcp-sa-gsuiteaddons.iam.gserviceaccount.com
#: Checking only for CHAT_ISSUER rejects every add-on request — and because the
#: audience still matches, it looks exactly like a valid token being refused.
CHAT_ADDON_ISSUER_SUFFIX = "@gcp-sa-gsuiteaddons.iam.gserviceaccount.com"


def chat_allowed_issuers() -> set[str]:
    """
    Exact issuer allow-list, if pinned via GOOGLE_CHAT_ISSUER.

    Empty means "accept either known Google signer" — see `issuer_is_google`.
    Pinning is tighter but needs updating if the app is rebuilt in a new
    project, so it stays optional.
    """
    raw = os.getenv("GOOGLE_CHAT_ISSUER", "").strip()
    return {e.strip() for e in raw.split(",") if e.strip()}


def issuer_is_google(email: str) -> bool:
    """
    Whether this token was signed by a Google service account we accept.

    Safe without pinning the exact address, because the audience check has
    already bound the token to this specific endpoint URL — and Google only
    mints those for this app.
    """
    if not email:
        return False
    if allowed := chat_allowed_issuers():
        return email in allowed
    return email == CHAT_ISSUER or email.endswith(CHAT_ADDON_ISSUER_SUFFIX)

CHAT_SCOPES = ["https://www.googleapis.com/auth/chat.bot"]


def chat_credentials_b64() -> str:
    """
    Service account for posting messages back to Chat.

    Falls back to the Calendar service account, which is correct when the same
    account is configured as the Chat app's identity. Kept separable because
    the Chat app may well end up on a different Google Cloud project.
    """
    return os.getenv("GOOGLE_CHAT_CREDENTIALS_B64", "") or google_credentials_b64()


def chat_audience() -> str:
    """
    Expected `aud` claim on incoming webhook JWTs. Must match what's set in the
    Chat API console:
      - "App URL" audience    -> the full https URL of the webhook endpoint
      - "Project Number"      -> the Google Cloud project number
    """
    return os.getenv("GOOGLE_CHAT_AUDIENCE", "").strip()


def chat_audience_is_project_number() -> bool:
    """Project numbers are all digits; endpoint URLs are not."""
    return chat_audience().isdigit()


def chat_verify_requests() -> bool:
    """
    Verify the Google-issued JWT on every webhook call.

    Defaults to True, and a missing or malformed value must never mean "skip
    verification" — an unverified webhook lets anyone on the internet book real
    jobs. Only set this false for local testing against a fake payload.
    """
    return os.getenv("CHAT_VERIFY_REQUESTS", "true").strip().lower() not in ("false", "0", "no")


# ── Business ───────────────────────────────────────────────────────────────────

TIMEZONE = "America/Los_Angeles"

#: Depot address — used for the >30mi early-slot rule.
DEPOT_ADDRESS = "909 Beacon Ave, Los Angeles, CA"

#: Minimum days between "today" and an acceptable move date.
MIN_LEAD_DAYS = 2


def deposit_amount() -> float:
    """Dollar amount of the deposit INVOICE. Not what goes on the calendar."""
    return float(os.getenv("DEPOSIT_AMOUNT", "50"))


#: What the calendar event's `Deposit:` line says at booking time.
#:
#: Deliberately NOT the deposit amount. The line tracks whether the deposit has
#: been *paid*, not what was billed — a separate existing automation rewrites it
#: once payment lands. Writing "50" here would claim a payment that hasn't
#: happened. "???" is what ghl_calendar_sync already writes, so downstream
#: tooling and the crew both already understand it.
CALENDAR_DEPOSIT_PLACEHOLDER = "???"


# ── Safety ─────────────────────────────────────────────────────────────────────

def dry_run() -> bool:
    """
    True  -> every write is logged and skipped. Reads still hit the live API.
    False -> LIVE. Real contacts, real invoices, real emails to real people.

    Defaults to True. A missing or malformed value must never mean "go live".
    """
    return os.getenv("DRY_RUN", "true").strip().lower() not in ("false", "0", "no")


def web_ui_token() -> str:
    """
    Shared secret gating the browser UI when it is exposed publicly.

    The local dev server (`python server.py`) binds to localhost and needs no
    token. The deployed app (app.py) refuses to serve the UI at all unless this
    is set, because an unauthenticated page on a public URL would let anyone
    who finds it book real jobs and text real customers.
    """
    return os.getenv("WEB_UI_TOKEN", "").strip()


# ── Models ─────────────────────────────────────────────────────────────────────

def model_backend() -> str:
    return os.getenv("MODEL_BACKEND", "openai").strip().lower()
