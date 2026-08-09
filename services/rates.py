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

Scope: the agent handles 2-, 3- and 4-mover jobs. GHL's `Movers` dropdown also
allows 5 and 6, but those have no `Rate` option and are priced by hand — per the
user, the agent will not be asked to book them, so they are explicitly out of
scope rather than half-supported. `is_supported_crew_size` is the gate.
"""

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

#: Sizes GHL accepts but the agent does not handle — priced by hand.
OUT_OF_SCOPE_MOVER_COUNTS = (5, 6)


def _as_int(movers: int | str) -> int | None:
    try:
        return int(str(movers).strip())
    except (ValueError, TypeError):
        return None


def is_supported_crew_size(movers: int | str) -> bool:
    """True if the agent can price and book this crew size on its own."""
    return _as_int(movers) in SUPPORTED_MOVER_COUNTS


def is_out_of_scope_crew_size(movers: int | str) -> bool:
    """
    True for 5 and 6 — real crew sizes GHL accepts, but priced by hand and
    deliberately not automated. Distinguished from junk input so the agent can
    say "that one's manual" instead of "that isn't a valid number".
    """
    return _as_int(movers) in OUT_OF_SCOPE_MOVER_COUNTS


def rate_for(movers: int | str) -> Rate | None:
    """Standard hourly Rate, or None if the size is unsupported."""
    count = _as_int(movers)
    return RATES.get(count) if count is not None else None


def format_rate(movers: int | str) -> str:
    """GHL-safe rate string, or '' when the size is unsupported."""
    rate = rate_for(movers)
    return rate.format() if rate else ""
