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


# ── Models ─────────────────────────────────────────────────────────────────────

def model_backend() -> str:
    return os.getenv("MODEL_BACKEND", "openai").strip().lower()
