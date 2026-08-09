"""
NODE: Confirm
PURPOSE: Show exactly what is about to be created, and wait for an explicit OK.
INPUT:   state.intake
OUTPUT:  {"approved": bool, "intake": {...}} then routes to execute or END

⚠️  CONTAINS interrupt(). This node deliberately contains NOTHING but summary
formatting, the interrupt, and the routing decision. No API calls, no writes,
nothing that could fire twice — because on every resume this node re-runs from
the top.

This is the last checkpoint before four irreversible things happen: a CRM
contact, a calendar booking, an invoice text to the customer, and a confirmation
email. Everything shown here should be everything that will happen.
"""

import logging
from typing import Literal

from langgraph.types import Command, interrupt

from agent.nodes.resolve_addresses import unresolved_addresses
from agent.state import OpsAgentState
from schemas import checklist as cl
from schemas.intake import EDITABLE_FIELDS
from services import calendar, config, rates

logger = logging.getLogger(__name__)


def _summary(intake: dict, warnings: dict, duplicate: dict | None) -> str:
    name = intake.get("full_name", "?")
    labor = intake.get("is_labor")
    movers = intake.get("movers", "?")
    rate = rates.format_rate(movers) or "(no standard rate)"
    deposit = config.deposit_amount()

    lines = [
        "Here's what I'll create. Nothing has happened yet.",
        "",
        f"  Customer    {name}{' (labor only)' if labor else ''}",
        f"  Phone       {intake.get('phone', '?')}",
        f"  Email       {intake.get('email', '?')}",
        f"  Date        {intake.get('move_date', '?')} at {intake.get('arrival_time', '?')}",
        f"  Crew        {movers} movers — {rate}",
        f"  Pickup      {intake.get('pickup_address', '?')}",
    ]
    if intake.get("extra_stop"):
        lines.append(f"  Extra stop  {intake['extra_stop']}")
    if not labor:
        lines.append(f"  Drop-off    {intake.get('dropoff_address', '?')}")
    if intake.get("job_notes"):
        lines.append(f"  Notes       {intake['job_notes']}")

    lines += [
        "",
        "That means:",
        "  1. Create or update the contact in GoHighLevel",
        "  2. Book the calendar event",
        f"  3. Text them a ${deposit:.0f} deposit invoice",
        "  4. Send the confirmation email",
    ]

    # Anything Google couldn't confirm outright — the wrong-city failure mode.
    if unresolved := unresolved_addresses(intake):
        lines += ["", "⚠️  Addresses I couldn't fully verify:"]
        lines += [f"  {name}: {note}" for name, note in unresolved.items()]

    if warnings:
        lines += ["", "⚠️  " + "; ".join(warnings.values())]

    if duplicate:
        lines += [
            "",
            "⚠️  There's already a job on the calendar for this customer and date:",
            f"     {duplicate.get('title')} on {duplicate.get('move_date')} "
            f"({duplicate.get('movers')} movers)",
            "     Booking again would double it up.",
        ]

    lines += [
        "",
        "Reply 'yes' to go ahead, 'no' to cancel, or tell me what to change "
        "(e.g. 'arrival 10-11am', 'make it 4 movers').",
    ]
    return "\n".join(lines)


def _find_duplicate(intake: dict) -> dict | None:
    """
    Check for an existing job for the same customer and date, whoever created
    it. Read-only, so it is safe to run before the interrupt — but it must stay
    read-only for exactly that reason.
    """
    phone, date = intake.get("phone", ""), intake.get("move_date", "")
    if not phone or not date:
        return None
    try:
        return calendar.find_duplicate_job(phone, date)
    except Exception:
        logger.exception("Duplicate check failed; continuing without it")
        return None


def confirm(state: OpsAgentState) -> Command[Literal["execute", "ask_missing", "__end__"]]:
    intake = dict(state.get("intake") or {})
    result = cl.evaluate(intake)
    duplicate = _find_duplicate(intake)

    # ── Above the interrupt: pure formatting + a read-only lookup. ──
    answer = interrupt(
        {
            "type": "confirm",
            "message": _summary(intake, result.warnings, duplicate),
            "intake": intake,
            "duplicate": duplicate,
        }
    )
    # ── Below: runs once per resume. ──

    decision = str(answer or "").strip().lower()

    if decision in ("yes", "y", "ok", "okay", "go", "go ahead", "confirm", "do it", "send it"):
        return Command(update={"approved": True}, goto="execute")

    if decision in ("no", "n", "cancel", "stop", "nope", "abort", "never mind"):
        return Command(update={"approved": False}, goto="__end__")

    # Anything else is treated as an edit instruction. Re-parsing through
    # ask_missing keeps one path for interpreting free text, and re-validates,
    # so an edit can never bypass the checklist.
    logger.info("Confirm reply treated as an edit: %r", decision[:120])
    return Command(
        update={"intake": {**intake, "_pending_edit": str(answer)}},
        goto="ask_missing",
    )


def apply_edit(intake: dict, edit: str) -> dict:
    """Placeholder hook for structured edits; ask_missing does the parsing today."""
    return intake


__all__ = ["confirm", "EDITABLE_FIELDS"]
