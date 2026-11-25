"""
Rate table, keyed by crew size.

⚠️  The strings here are not cosmetic. GHL's `Rate` field is a SINGLE_OPTIONS
dropdown, so a value that does not match one of its options byte-for-byte is
rejected and the field silently stays empty — the API still returns 200.

These are read directly off the live GHL location. They are ALSO editable in
the GHL UI, and have changed once already (a stray space after the slash on the
2-mover option was cleaned up on 2026-08-07). Whenever bookings come back with
an empty Rate field, re-query the live picklist first:

    GET /locations/{id}/customFields  ->  field "Rate" -> picklistOptions

`test_rate_strings_match_the_live_ghl_dropdown_exactly` pins these against
PICKLIST_VALUES, but nothing can detect a UI-side edit except re-querying.

Scope: 2, 3 and 4 movers have a standard rate (below). Any other crew size is a
special job: the dispatcher types the rate as "cash / card" (e.g. "250/270") and
it is carried on the intake as `rate`. `effective_rate` is the one place that
decides which applies. A custom rate has no GHL `Rate` dropdown option, so the
GHL field is left blank for those jobs — the calendar, email and crew app still
show it.
"""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Rate:
    movers: int
    cash: int
    card: int
    #: Exact GHL dropdown option. MUST match the live picklist byte-for-byte.
    ghl_option: str

    def format(self) -> str:
        """The string written to both the GHL Rate field and the calendar event."""
        return self.ghl_option


RATES: dict[int, Rate] = {
    2: Rate(movers=2, cash=115, card=125, ghl_option="$115 cash /$125 card"),
    3: Rate(movers=3, cash=145, card=155, ghl_option="$145 cash /$155 card"),
    4: Rate(movers=4, cash=180, card=190, ghl_option="$180 cash /$190 card"),
}

#: Crew sizes the agent can book end to end.
SUPPORTED_MOVER_COUNTS = tuple(sorted(RATES))

#: Above this many movers the dispatcher is also asked how many trucks.
MAX_SINGLE_TRUCK_MOVERS = 4


def _as_int(movers: int | str) -> int | None:
    try:
        return int(str(movers).strip())
    except (ValueError, TypeError):
        return None


def is_supported_crew_size(movers: int | str) -> bool:
    """True if the agent can price and book this crew size on its own."""
    return _as_int(movers) in SUPPORTED_MOVER_COUNTS


def is_valid_crew_size(movers: int | str) -> bool:
    """Any whole number of movers, one or more."""
    count = _as_int(movers)
    return count is not None and count >= 1


def needs_custom_rate(movers: int | str) -> bool:
    """A valid crew size with no standard rate — the dispatcher must give one."""
    return is_valid_crew_size(movers) and not is_supported_crew_size(movers)


def asks_for_trucks(movers: int | str) -> bool:
    """More than 4 movers can mean more than one truck, so it is asked."""
    count = _as_int(movers)
    return count is not None and count > MAX_SINGLE_TRUCK_MOVERS


_MONEY = r"\$?\s*(\d+(?:\.\d{1,2})?)"
_CUSTOM_RATE = re.compile(
    rf"^\s*{_MONEY}\s*(?:cash)?\s*(?:/|,|\band\b|\|)\s*{_MONEY}\s*(?:card)?\s*(?:/\s*h(?:ou)?r)?\s*$",
    re.IGNORECASE,
)


def parse_custom_rate(text: str) -> tuple[int, int] | None:
    """
    "250/270", "$250 / $270", "$250 cash /$270 card" -> (250, 270).
    Cash first, card second — the order the dispatcher types them in.
    """
    m = _CUSTOM_RATE.match(str(text or ""))
    if not m:
        return None
    cash, card = (round(float(v)) for v in m.groups())
    if cash <= 0 or card <= 0:
        return None
    return cash, card


def format_custom_rate(text: str) -> str:
    """Canonical form, styled like the GHL options: "$250 cash /$270 card"."""
    parsed = parse_custom_rate(text)
    if parsed is None:
        return str(text or "").strip()
    cash, card = parsed
    return f"${cash} cash /${card} card"


def effective_rate(intake: dict) -> Rate | None:
    """
    The rate this job is billed at: the standard one for 2-4 movers, otherwise
    the one the dispatcher typed. None when neither exists yet.
    """
    movers = intake.get("movers", "")
    if standard := rate_for(movers):
        return standard
    count = _as_int(movers)
    parsed = parse_custom_rate(intake.get("rate", ""))
    if count is None or parsed is None:
        return None
    cash, card = parsed
    return Rate(movers=count, cash=cash, card=card,
                ghl_option=f"${cash} cash /${card} card")


def format_effective_rate(intake: dict) -> str:
    """Rate string for the calendar, the email and the confirm summary."""
    rate = effective_rate(intake)
    return rate.format() if rate else ""


def ghl_rate_option(intake: dict) -> str:
    """Only a standard rate can go in the GHL dropdown; a custom one goes blank."""
    return format_rate(intake.get("movers", ""))


def truck_count(intake: dict) -> int:
    """How many trucks. 1 unless the dispatcher said otherwise; 0 for labor."""
    if intake.get("is_labor"):
        return 0
    count = _as_int(intake.get("trucks", ""))
    return count if count and count >= 1 else 1


def rate_for(movers: int | str) -> Rate | None:
    """Standard hourly Rate, or None if the size is unsupported."""
    count = _as_int(movers)
    return RATES.get(count) if count is not None else None


def format_rate(movers: int | str) -> str:
    """GHL-safe rate string, or '' when the size is unsupported."""
    rate = rate_for(movers)
    return rate.format() if rate else ""
