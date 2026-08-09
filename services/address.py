"""
Address completion and verification, via Google's Address Validation API.

WHY THIS IS ITS OWN SERVICE, AND WHY IT IS STRICT
--------------------------------------------------
Customers send partial addresses — often just "412 N Maple Ave". The calendar
needs the full, correctly-cased form: "412 N Maple Ave, Burbank CA 91505".

Completing that is not a formatting problem, it is a guessing problem, and
guessing wrong sends a truck to the wrong building. Measured on the live Maps
key, taking the first Places Autocomplete prediction produced:

    "412 N Maple Ave"        -> 412 N Maple Ave, Montebello CA 90640   (invented a city)
    "1830 pine st glendale"  -> Pine St, Glendo WY 82213               (wrong state)

Both look plausible. Neither is right. So this module does not return a string —
it returns a verdict, and the caller is expected to put anything that isn't
CONFIRMED in front of a human.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum

import requests

from services import config
from services.formatting import format_address, strip_address_label

logger = logging.getLogger(__name__)

_ENDPOINT = "https://addressvalidation.googleapis.com/v1:validateAddress"
_TIMEOUT = 15


class Verdict(str, Enum):
    #: Every component confirmed by Google. Safe to use.
    CONFIRMED = "confirmed"
    #: Resolved, but Google inferred or changed something. Show the user.
    NEEDS_REVIEW = "needs_review"
    #: Could not be resolved at all. Ask the user to re-enter.
    UNRESOLVED = "unresolved"
    #: The API itself is unavailable (not enabled, quota, network).
    UNAVAILABLE = "unavailable"


class AddressValidationUnavailable(RuntimeError):
    """Raised when the Address Validation API cannot be reached at all."""


@dataclass
class ValidatedAddress:
    verdict: Verdict
    #: Calendar-format address, e.g. "412 N Maple Ave, Burbank CA 91505".
    formatted: str = ""
    #: Exactly what the user typed.
    original: str = ""
    #: Components Google added that the user never supplied (e.g. a city).
    inferred: list[str] = field(default_factory=list)
    #: Components Google could not confirm.
    unconfirmed: list[str] = field(default_factory=list)
    #: Human-readable explanation for the confirm gate.
    note: str = ""

    @property
    def is_usable(self) -> bool:
        return self.verdict is Verdict.CONFIRMED

    @property
    def needs_human(self) -> bool:
        return self.verdict in (Verdict.NEEDS_REVIEW, Verdict.UNRESOLVED)


#: Tokens that must stay fully capitalised when title-casing a USPS line.
_KEEP_UPPER = {
    "N", "S", "E", "W", "NE", "NW", "SE", "SW", "NBR",
    # every US state abbreviation
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL",
    "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT",
    "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI",
    "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "DC",
}


def _title_case_usps(line: str) -> str:
    """
    USPS returns SHOUTING ("412 N MAPLE ST, BURBANK CA 91505"). Convert to the
    house casing while preserving directionals and the state code.

    Ambiguity worth knowing: "LA" is both Louisiana and an abbreviation people
    write for Los Angeles. Keeping it upper is right for the state field, which
    is where it appears in a USPS line.
    """
    def fix(token: str) -> str:
        core = token.strip(",")
        trailing = token[len(core):]
        if core.upper() in _KEEP_UPPER:
            return core.upper() + trailing
        if any(ch.isdigit() for ch in core):
            return core.upper() + trailing  # 91505, 101ST
        return core.capitalize() + trailing

    return " ".join(fix(t) for t in line.split())


#: USPS suffix abbreviations, applied to Google's long-form output so every
#: address the agent writes looks like the 1,598 already on the calendar
#: ("412 N Maple Ave", not "412 North Maple Avenue").
_ABBREVIATIONS = {
    "Avenue": "Ave", "Street": "St", "Boulevard": "Blvd", "Drive": "Dr",
    "Road": "Rd", "Lane": "Ln", "Court": "Ct", "Place": "Pl", "Terrace": "Ter",
    "Circle": "Cir", "Parkway": "Pkwy", "Highway": "Hwy", "Square": "Sq",
    "Trail": "Trl", "Way": "Way",
    "North": "N", "South": "S", "East": "E", "West": "W",
    "Northeast": "NE", "Northwest": "NW", "Southeast": "SE", "Southwest": "SW",
    "Apartment": "Apt", "Suite": "Ste", "Unit": "Unit", "Building": "Bldg",
}


def _abbreviate(addr: str) -> str:
    """Shorten street suffixes and directionals, leaving city names alone."""
    def fix(token: str) -> str:
        core = token.strip(",")
        trailing = token[len(core):]
        return _ABBREVIATIONS.get(core, core) + trailing

    # Only touch the street portion — "North Hollywood" is a city, not a
    # directional, and must survive intact.
    head, sep, tail = addr.partition(",")
    return " ".join(fix(t) for t in head.split()) + sep + tail


def _to_calendar_format(google_formatted: str) -> str:
    """
    Convert Google's "412 North Maple Street, Burbank, CA 91505, USA" into the
    house format "412 N Maple Ave, Burbank CA 91505".

    Prefer the USPS standardized line where available — it is already
    abbreviated the way the calendar and every downstream parser expect.
    """
    addr = re.sub(r",?\s*USA$", "", google_formatted.strip(), flags=re.IGNORECASE)
    return format_address(_abbreviate(addr))


#: Home state. Splendid Moving is a Los Angeles company: pickups are in
#: California essentially always, and a guessed city outside it is wrong.
HOME_STATE = "CA"

#: Inferring a CITY means Google chose WHERE in the country this is. That is
#: only dangerous when it lands somewhere implausible — measured against real
#: staff input, "614 e verdugo ave" -> Burbank CA is correct and useful, while
#: "412 N Maple Ave" -> Brandon SD is the failure. The discriminator is not
#: whether Google guessed, but whether the guess left the service area.
#:
#: Inferring a STATE or ZIP is the opposite: it is the whole point of this node.
_DANGEROUS_INFERENCE = {"locality"}

_STATE_CODES = {c for c in _KEEP_UPPER if len(c) == 2} - {
    "N", "S", "E", "W", "NE", "NW", "SE", "SW",
}


def _resolved_state(address_block: dict) -> str:
    for component in address_block.get("addressComponents", []):
        if component.get("componentType") == "administrative_area_level_1":
            return component.get("componentName", {}).get("text", "").upper()
    return ""


def _state_written_by_user(text: str) -> str:
    """The state the user actually typed, if any. Sits near the end."""
    for token in reversed(re.findall(r"\b([A-Za-z]{2})\b", text.upper())):
        if token in _STATE_CODES:
            return token
    return ""


def validate(address: str, *, region: str = "US") -> ValidatedAddress:
    """
    Complete and verify a single address.

    Never raises for a bad address — an unusable input comes back as a verdict
    the caller can show the user. Only raises if the API itself is unreachable,
    which is an operator problem rather than a data problem.
    """
    # Strip any field label the extractor carried over ("From: 614 E Verdugo
    # Ave"). Done before the lookup, not just on the way out: a labelled string
    # degrades Google's match as well as the stored value.
    original = strip_address_label(address or "")
    if not original:
        return ValidatedAddress(
            verdict=Verdict.UNRESOLVED, original=original, note="No address given."
        )

    try:
        resp = requests.post(
            _ENDPOINT,
            params={"key": config.maps_api_key()},
            json={
                "address": {"regionCode": region, "addressLines": [original]},
                # Returns the USPS-standardized line, which is abbreviated the
                # way the calendar expects ("412 N MAPLE ST" not "412 North
                # Maple Street"). US addresses only, which is all we handle.
                "enableUspsCass": True,
            },
            timeout=_TIMEOUT,
        )
    except Exception as exc:
        raise AddressValidationUnavailable(f"Address Validation unreachable: {exc}") from exc

    if resp.status_code == 403:
        raise AddressValidationUnavailable(
            "Address Validation API is not enabled on this Google Cloud project. "
            "Enable it: console.cloud.google.com -> APIs & Services -> "
            "Enable APIs -> 'Address Validation API'."
        )
    if not resp.ok:
        raise AddressValidationUnavailable(
            f"Address Validation returned HTTP {resp.status_code}: {resp.text[:200]}"
        )

    result = resp.json().get("result", {})
    address_block = result.get("address", {})

    # Prefer the USPS line — already abbreviated the way the calendar expects.
    # But USPS only returns a COMPLETE line when it could fully standardize;
    # otherwise it echoes a fragment with no state or ZIP, which is worse than
    # Google's own formatting. Require a state + ZIP before trusting it.
    usps_line = (
        result.get("uspsData", {}).get("standardizedAddress", {}).get("firstAddressLine", "")
    )
    usps_is_complete = bool(re.search(r"\b[A-Z]{2}\s+\d{5}", usps_line.upper()))

    formatted = (
        format_address(_title_case_usps(usps_line))
        if usps_is_complete
        else _to_calendar_format(address_block.get("formattedAddress", ""))
    )

    # Only LOCATION components matter for the dangerous case. Google routinely
    # reports street_number as UNCONFIRMED_BUT_PLAUSIBLE for perfectly good
    # addresses — treating that as a problem would flag almost everything and
    # train the user to click through warnings.
    inferred_location: list[str] = []
    for component in address_block.get("addressComponents", []):
        name = component.get("componentType", "")
        if component.get("inferred") and name in _DANGEROUS_INFERENCE:
            inferred_location.append(name)

    missing = [m for m in (address_block.get("missingComponentTypes") or [])]
    unresolved_tokens = address_block.get("unresolvedTokens") or []
    # addressComplete is read but deliberately not gated on — see the note
    # further down about why it fires on perfectly usable addresses.

    # Fatal only when the STREET itself couldn't be identified. `addressComplete`
    # is deliberately not fatal on its own: Google sets it false for a missing
    # subpremise (apartment/suite), and plenty of real houses have none — that
    # rejected Splendid's own office address at 550 N Figueroa St.
    street_unknown = bool({"route", "street_number"} & set(missing))
    if not formatted or street_unknown:
        return ValidatedAddress(
            verdict=Verdict.UNRESOLVED,
            formatted=formatted,
            original=original,
            inferred=inferred_location,
            note=(
                f"Couldn't resolve {original!r}."
                + (f" Missing: {', '.join(missing)}." if missing else "")
                + (f" Didn't understand: {', '.join(unresolved_tokens)}." if unresolved_tokens else "")
            ),
        )

    resolved_state = _resolved_state(address_block)
    user_state = _state_written_by_user(original)

    # The user named a state and Google resolved to a different one. Always
    # wrong, and always worth stopping for.
    if user_state and resolved_state and user_state != resolved_state:
        return ValidatedAddress(
            verdict=Verdict.NEEDS_REVIEW,
            formatted=formatted,
            original=original,
            inferred=inferred_location,
            note=(
                f"You wrote {user_state}, but this resolved to {resolved_state}: "
                f"{formatted!r}. Check before booking."
            ),
        )

    # Google picked the city itself AND landed outside California. That is the
    # Brandon-SD failure — a real street in a state nobody mentioned.
    #
    # A guessed city INSIDE California is left alone on purpose. Splendid works
    # the LA area, so "614 e verdugo ave" -> Burbank CA is not a guess worth
    # interrupting for; flagging it taught staff to click through warnings,
    # which is how the one that matters gets missed.
    if inferred_location and resolved_state and resolved_state != HOME_STATE:
        return ValidatedAddress(
            verdict=Verdict.NEEDS_REVIEW,
            formatted=formatted,
            original=original,
            inferred=inferred_location,
            note=(
                f"{original!r} has no city, and Google placed it in "
                f"{resolved_state}: {formatted!r}. That's outside the service "
                "area — confirm the city before booking."
            ),
        )

    if unresolved_tokens:
        return ValidatedAddress(
            verdict=Verdict.NEEDS_REVIEW,
            formatted=formatted,
            original=original,
            note=f"{original!r} -> {formatted!r}, ignoring {', '.join(unresolved_tokens)}.",
        )

    # Deliberately NOT flagged, after checking against real staff input:
    #
    #   missing subpremise  — Google reports this for most single-family homes.
    #                         Flagging it warned on nearly every house.
    #   addressComplete=False — set for a missing ZIP or unit even when the
    #                         address is perfectly usable ("1830 Pine St
    #                         Glendale" is fine).
    #
    # Both fired constantly on correct addresses. A warning that appears on
    # everything is not a warning, and it buries the state-mismatch case above,
    # which is the one that actually sends a truck to the wrong place.

    # Google's own possibleNextAction is deliberately NOT used as a gate. It
    # returns "FIX" for addresses that are complete and correct (it cannot
    # confirm exact house numbers on many residential streets), so gating on it
    # flagged even "1830 Pine St, Glendale CA 91206". The specific checks above
    # carry the real signal.
    return ValidatedAddress(
        verdict=Verdict.CONFIRMED,
        formatted=formatted,
        original=original,
        note=f"{formatted} (confirmed)",
    )


def validate_many(addresses: dict[str, str]) -> dict[str, ValidatedAddress]:
    """
    Validate several labelled addresses at once, e.g.
    {"pickup": "...", "dropoff": "...", "extra_stop": "..."}.

    Blank entries are skipped rather than reported as failures — extra_stop is
    genuinely optional.
    """
    return {
        label: validate(value)
        for label, value in addresses.items()
        if (value or "").strip()
    }
