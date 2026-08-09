"""
Pure formatting + parsing helpers. No network, no state — everything here is
directly unit-testable.

Ported from ghl_calendar_sync/app.py, which is the source of truth for the
formats the rest of the Splendid Moving stack expects.
"""

import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from services import config

LA_TZ = ZoneInfo(config.TIMEZONE)


# ── Dates ──────────────────────────────────────────────────────────────────────

_DATE_FORMATS = ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%Y/%m/%d", "%d-%m-%Y")


def parse_date(date_str: str) -> datetime | None:
    """
    Parse a date in any of the formats seen across the stack.

    Returns None when unparseable. The original in ghl_calendar_sync fell back to
    "today" — acceptable for a webhook that must not drop a lead, but wrong here:
    silently booking a move for today because a date failed to parse is far worse
    than asking the user to repeat it.
    """
    if not date_str:
        return None
    s = str(date_str).strip()

    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass

    if m := re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", s):
        return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    if m := re.search(r"(\d{1,2})/(\d{1,2})/(\d{4})", s):
        return datetime(int(m.group(3)), int(m.group(1)), int(m.group(2)))
    return None


def format_date(dt: datetime) -> str:
    """mm/dd/yyyy — the form the calendar `Date:` line uses."""
    return dt.strftime("%m/%d/%Y")


# ── Arrival windows ────────────────────────────────────────────────────────────

def _apply_period(hour: int, minute: int, period: str) -> tuple[int, int]:
    if period == "pm" and hour != 12:
        hour += 12
    elif period == "am" and hour == 12:
        hour = 0
    return hour, minute


def parse_arrival_time(time_str: str, base_date: datetime) -> tuple[str, str] | None:
    """
    Parse an arrival window into (start_iso, end_iso) in LA time.

    Handles: '8am-9am', '11am-1pm', '8-9am', '2-4pm', '11-1pm', '9am', '9:30am'.

    The '11-1pm' case is why this can't be a simple regex: with only a trailing
    period, a start hour greater than the end hour means the start is am.

    Returns None when unparseable — the caller asks rather than guessing.
    """
    if not time_str:
        return None

    def make(h: int, m: int) -> str:
        return base_date.replace(hour=h, minute=m, second=0, microsecond=0, tzinfo=LA_TZ).isoformat()

    s = str(time_str).strip().lower().replace(" ", "")

    # Both sides carry a period: "8am-9am", "11am-1pm"
    if m := re.match(r"^(\d+)(?::(\d+))?(am|pm)[-–to]+(\d+)(?::(\d+))?(am|pm)$", s):
        sh, sm = _apply_period(int(m.group(1)), int(m.group(2) or 0), m.group(3))
        eh, em = _apply_period(int(m.group(4)), int(m.group(5) or 0), m.group(6))
        return make(sh, sm), make(eh, em)

    # Period only at the end: "8-9am", "2-4pm", "11-1pm"
    if m := re.match(r"^(\d+)(?::(\d+))?[-–](\d+)(?::(\d+))?(am|pm)$", s):
        sh_raw, sm_raw = int(m.group(1)), int(m.group(2) or 0)
        eh_raw, em_raw, end_period = int(m.group(3)), int(m.group(4) or 0), m.group(5)
        if end_period == "am":
            start_period = "am"
        elif sh_raw > eh_raw:
            start_period = "am"  # "11-1pm" -> 11am to 1pm
        else:
            start_period = "pm"  # "2-4pm"  -> both pm
        sh, sm = _apply_period(sh_raw, sm_raw, start_period)
        eh, em = _apply_period(eh_raw, em_raw, end_period)
        return make(sh, sm), make(eh, em)

    # Single time: "9am", "9:30am" -> a 15-minute window ending at that time
    if m := re.match(r"^(\d+)(?::(\d+))?(am|pm)$", s):
        eh, em = _apply_period(int(m.group(1)), int(m.group(2) or 0), m.group(3))
        end = base_date.replace(hour=eh, minute=em, second=0, microsecond=0, tzinfo=LA_TZ)
        return (end - timedelta(minutes=15)).isoformat(), end.isoformat()

    return None


def normalize_arrival_label(time_str: str) -> str:
    """Compact label for the GHL Arrival Time field, e.g. '8-9am'."""
    return str(time_str or "").strip().lower().replace(" ", "")


# ── Phone / address ────────────────────────────────────────────────────────────

def format_phone(raw: str) -> str:
    """Normalize to +1(NXX)NXX-XXXX. Returns the input unchanged if it isn't 10 digits."""
    if not raw:
        return ""
    digits = re.sub(r"\D", "", str(raw))
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    if len(digits) == 10:
        return f"+1({digits[:3]}){digits[3:6]}-{digits[6:]}"
    return str(raw).strip()


#: Labels a screenshot may carry into the extracted value. Seen live: an
#: extraction returned "From: 614 E Verdugo Ave", and the calendar builder then
#: prefixed its own "From: ", producing "From: From: 614 E Verdugo Ave".
_ADDRESS_LABEL = re.compile(
    r"^\s*(from|to|pickup|pick\s*up|drop\s*-?\s*off|dropoff|destination|origin|"
    r"address|moving\s+from|moving\s+to|current\s+address|new\s+address|"
    r"extra\s+stop|stop)\s*[:\-–]\s*",
    re.IGNORECASE,
)


def strip_address_label(raw: str) -> str:
    """
    Remove a leading field label. Applied repeatedly, since a screenshot line
    like "Pickup: From: 123 Main St" carries two.
    """
    text = str(raw or "").strip()
    while True:
        stripped = _ADDRESS_LABEL.sub("", text, count=1).strip()
        if stripped == text:
            return text
        text = stripped


def format_address(raw: str) -> str:
    """
    Normalize to: 544 E Amazing St, Burbank CA 91501

    One comma after the street, none before the state abbreviation, and a
    5-digit ZIP. Google returns ZIP+4 ("91501-2385"); the extra four digits are
    correct but nobody at Splendid writes them, and the crew reads these off a
    phone — so they are dropped to match the 1,598 events already on the
    calendar.
    """
    if not raw:
        return ""
    addr = strip_address_label(raw)
    addr = re.sub(r"\s+", " ", addr.strip())
    # ZIP+4 -> ZIP5
    addr = re.sub(r"\b(\d{5})-\d{4}\b", r"\1", addr)
    addr = re.sub(r",\s*([A-Z]{2})\s+(\d{5})", r" \1 \2", addr)
    addr = re.sub(r",\s*,", ",", addr)
    return addr.strip().strip(",").strip()


def format_deposit(amount: float | str) -> str:
    """
    Deposit as a bare number, e.g. '50'.

    Matches the dominant existing convention: of 1,598 real jobs in the last
    year, 1,459 store the deposit as a bare integer. Writing '$50' instead would
    make this agent's events the odd ones out for anything reading that field.
    """
    try:
        value = float(str(amount).replace("$", "").strip())
    except (ValueError, TypeError):
        return str(amount or "").strip()
    return f"{value:g}"


def split_name(full_name: str) -> tuple[str, str]:
    """Split a full name into (first, last). Everything after the first token is last."""
    parts = str(full_name or "").strip().split()
    if not parts:
        return "", ""
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], " ".join(parts[1:])


# ── Lead source -> event colour ────────────────────────────────────────────────

#: Real spelling and casing variants found on live calendar events. Without
#: this, "Previous Customer", "previous customer" and "Previous costumer" count
#: as three separate lead sources and every breakdown is subtly wrong.
_SOURCE_ALIASES = {
    "previous customer": "Previous Customer",
    "previous costumer": "Previous Customer",
    "previous client": "Previous Customer",
    "yelp": "Yelp",
    "local service ads": "Local Service Ads",
    "lsa": "Local Service Ads",
    "google local services": "Local Service Ads",
    "google my business": "Google My Business",
    "gmb": "Google My Business",
    "google": "Google My Business",
    "referral": "Referral",
    "thumbtack": "Thumbtack",
}


def normalize_source(raw: str) -> str:
    """
    Canonical lead source for aggregation.

    Returns 'unspecified' for blanks — 58% of live events have no source, so
    that bucket is real signal, not an error to hide.
    """
    cleaned = str(raw or "").strip()
    if not cleaned:
        return "unspecified"
    return _SOURCE_ALIASES.get(cleaned.lower(), cleaned)


def get_color_id(source: str) -> str | None:
    """Yelp -> 6 (tangerine), Google LSA -> 2 (sage), anything else -> default blue."""
    if not source:
        return None
    s = source.lower().strip()
    if "yelp" in s:
        return "6"
    if "local service" in s or "lsa" in s:
        return "2"
    return None
