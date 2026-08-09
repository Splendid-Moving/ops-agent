"""
NODE: Resolve addresses
PURPOSE: Complete every partial address into the full house format, and flag
         anything Google had to guess at.
INPUT:   state.intake (pickup_address, dropoff_address, extra_stop)
OUTPUT:  {"intake": {...addresses replaced..., "address_status": {...}}}

MANDATORY. This runs on every intake — it is not something the agent chooses to
invoke. Customers routinely send "412 N Maple Ave" with no city, and the
calendar needs "412 N Maple Ave, Burbank CA 91505".

Only CONFIRMED results are written back silently. Anything Google inferred or
could not confirm keeps the ORIGINAL text and is surfaced at the confirm gate,
because the failure mode is not a formatting error — measured on the live key,
naive completion turned "1830 pine st glendale" into an address in Wyoming.
"""

import logging

from agent import progress
from agent.state import OpsAgentState
from schemas.intake import ADDRESS_FIELDS
from services import address as address_service
from services.address import AddressValidationUnavailable, Verdict

logger = logging.getLogger(__name__)


def resolve_addresses(state: OpsAgentState) -> dict:
    intake = dict(state.get("intake") or {})

    candidates = {
        name: intake.get(name, "")
        for name in ADDRESS_FIELDS
        if str(intake.get(name, "") or "").strip()
    }
    if not candidates:
        return {}

    progress.working(f"Verifying {len(candidates)} address"
                     f"{'es' if len(candidates) != 1 else ''}\u2026")
    try:
        results = address_service.validate_many(candidates)
    except AddressValidationUnavailable as exc:
        # An operator problem, not a data problem. Degrade rather than block:
        # the addresses stay as typed and the confirm gate says they're
        # unverified, so a human still sees them before anything is booked.
        logger.error("Address validation unavailable: %s", exc)
        progress.warn("Could not verify addresses", "using them exactly as written")
        intake["address_status"] = {
            name: f"unverified — {exc}" for name in candidates
        }
        return {"intake": intake}

    status: dict[str, str] = {}
    for name, result in results.items():
        if result.verdict is Verdict.CONFIRMED:
            intake[name] = result.formatted
            status[name] = "confirmed"
        else:
            # Keep what the user gave us. Overwriting with a guess is how a
            # truck ends up in the wrong city.
            status[name] = f"{result.verdict.value}: {result.note}"
            if result.formatted:
                status[name] += f" | suggested: {result.formatted}"
            logger.warning("Address %s needs review: %s", name, result.note)

    confirmed = sum(1 for v in status.values() if v == "confirmed")
    if confirmed:
        progress.done(f"Confirmed {confirmed} address"
                      f"{'es' if confirmed != 1 else ''}")
    if flagged := len(status) - confirmed:
        progress.warn(f"{flagged} address needs a look")
    intake["address_status"] = status
    return {"intake": intake}


def unresolved_addresses(intake: dict) -> dict[str, str]:
    """Addresses that still need a human decision, for the confirm gate."""
    return {
        name: note
        for name, note in (intake.get("address_status") or {}).items()
        if not note.startswith("confirmed")
    }
