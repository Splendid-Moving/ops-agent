"""
NODES: the four side effects, plus reconcile and report.

    act_upsert_contact   -> GHL contact          (gate: the other three need its id)
    act_calendar_event   -> Google Calendar event
    act_deposit_invoice  -> GHL invoice, texted to the customer
    act_confirmation_email -> GHL email

These live in one module because they share the idempotency guard below. Four
copies of the same latch is four chances to forget it, and forgetting it means
charging a customer twice.

THE RULE EVERY NODE HERE FOLLOWS
--------------------------------
Check the ledger first. If this action already succeeded, do nothing and return.
That is what makes a retry safe: re-running the whole execute stage after a
partial failure re-fires only the parts that failed.

All four run strictly AFTER the confirm interrupt, so they are reached exactly
once per approval. Nothing here may ever move above an interrupt().
"""

import hashlib
import logging
from collections.abc import Callable
from datetime import datetime
from functools import wraps

from langchain_core.messages import AIMessage

from agent import progress
from agent.state import (
    ACTION_CALENDAR,
    ACTION_CONTACT,
    ACTION_EMAIL,
    ACTION_INVOICE,
    ALL_ACTIONS,
    OpsAgentState,
    failed_actions,
    succeeded,
)
from schemas import email_template
from services import calendar, config, formatting, ghl, maps, rates
from services.ghl import CustomField

logger = logging.getLogger(__name__)


def job_fingerprint(intake: dict) -> str:
    """
    Stable id for "this customer, this date". Used to spot a job that is
    already on the calendar, and stored on the event so a re-run can find it.
    """
    key = f"{ghl.normalize_phone(intake.get('phone', ''))}|{intake.get('move_date', '')}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def action(name: str) -> Callable:
    """
    Wrap a side effect with the ledger: skip if already done, record success or
    failure, never raise into the graph.

    Catching everything is deliberate. One failed action must not abort the
    other three — a job with a contact and a calendar event but no invoice is
    recoverable; a job that stopped halfway with no record of what happened is
    not.
    """

    def decorator(fn: Callable[[OpsAgentState], dict]) -> Callable[[OpsAgentState], dict]:
        @wraps(fn)
        def wrapper(state: OpsAgentState) -> dict:
            ledger = state.get("ledger") or {}

            if succeeded(ledger, name):
                logger.info("%s already succeeded — skipping", name)
                return {}

            attempts = ledger.get(name, {}).get("attempts", 0) + 1
            try:
                result = fn(state)
                logger.info("%s ok: %s", name, result)
                progress.done(_DONE_TEXT.get(name, f"{name} done"))
                return {
                    "ledger": {
                        name: {
                            "status": "success",
                            "result": result,
                            "error": None,
                            "attempts": attempts,
                        }
                    }
                }
            except Exception as exc:
                logger.exception("%s failed", name)
                progress.error(f"{_LABELS_SHORT.get(name, name)} failed",
                               f"{type(exc).__name__}: {exc}")
                return {
                    "ledger": {
                        name: {
                            "status": "failed",
                            "result": {},
                            "error": f"{type(exc).__name__}: {exc}",
                            "attempts": attempts,
                        }
                    }
                }

        return wrapper

    return decorator


_DONE_TEXT = {
    ACTION_CONTACT: "Contact saved in GoHighLevel",
    ACTION_CALENDAR: "Job added to the calendar",
    ACTION_INVOICE: "Deposit payment link texted",
    ACTION_EMAIL: "Confirmation email sent",
}

_LABELS_SHORT = {
    ACTION_CONTACT: "Contact",
    ACTION_CALENDAR: "Calendar event",
    ACTION_INVOICE: "Deposit link",
    ACTION_EMAIL: "Confirmation email",
}


def _contact_id(state: OpsAgentState) -> str:
    return (state.get("ledger") or {}).get(ACTION_CONTACT, {}).get("result", {}).get("contact_id", "")


# ── 1. Contact (gate) ──────────────────────────────────────────────────────────

@action(ACTION_CONTACT)
def act_upsert_contact(state: OpsAgentState) -> dict:
    progress.working("Creating the contact in GoHighLevel…")
    intake = state.get("intake") or {}
    first, last = formatting.split_name(intake.get("full_name", ""))

    custom_fields = {
        CustomField.MOVING_FROM: intake.get("pickup_address", ""),
        CustomField.MOVING_TO: intake.get("dropoff_address", ""),
        CustomField.EXTRA_STOP: intake.get("extra_stop", ""),
        CustomField.MOVING_DATE: intake.get("move_date", ""),
        CustomField.ARRIVAL_TIME: intake.get("arrival_time", ""),
        CustomField.MOVERS: str(intake.get("movers", "")),
        CustomField.RATE: rates.format_rate(intake.get("movers", "")),
        CustomField.JOB_NOTES: intake.get("job_notes", ""),
    }
    if intake.get("is_labor"):
        custom_fields[CustomField.LABOR] = "Labor"

    # NOTE: CustomField.CREATE_GOOGLE_EVENT is deliberately absent and is
    # rejected by validate_custom_fields if it ever appears. Setting it fires
    # the GHL workflow that builds a calendar event, and this agent builds its
    # own — that would double-book every job.
    return ghl.upsert_contact(
        first_name=first,
        last_name=last,
        phone=intake.get("phone", ""),
        email=intake.get("email", ""),
        custom_fields=custom_fields,
        tags=["ops-agent"],
    )


# ── 2. Calendar event ──────────────────────────────────────────────────────────

@action(ACTION_CALENDAR)
def act_calendar_event(state: OpsAgentState) -> dict:
    progress.working("Booking the calendar event…")
    intake = state.get("intake") or {}

    move_date = formatting.parse_date(intake.get("move_date", ""))
    if move_date is None:
        raise ValueError(f"Unparseable move date {intake.get('move_date')!r}")

    window = formatting.parse_arrival_time(intake.get("arrival_time", ""), move_date)
    if window is None:
        raise ValueError(f"Unparseable arrival window {intake.get('arrival_time')!r}")
    start_iso, end_iso = window

    pickup = intake.get("pickup_address", "")
    dropoff = intake.get("dropoff_address", "")
    extra_stop = intake.get("extra_stop", "")

    # Distance is informational; a failure here must not block the booking.
    distance = maps.get_distance(pickup, dropoff, extra_stop or None) if dropoff else ""

    description = calendar.build_description(
        customer=intake.get("full_name", ""),
        phone=intake.get("phone", ""),
        date=formatting.format_date(move_date),
        from_address=pickup,
        to_address=dropoff,
        extra_stop=extra_stop,
        distance=distance,
        rate=rates.format_rate(intake.get("movers", "")),
        movers=str(intake.get("movers", "")),
        # Tracks whether the deposit has been PAID, not what was billed. A
        # separate existing automation rewrites this once payment lands.
        deposit=config.CALENDAR_DEPOSIT_PLACEHOLDER,
        source=intake.get("source", ""),
        notes=intake.get("job_notes", ""),
    )

    return calendar.create_event(
        title=intake.get("full_name", "") + (" (labor)" if intake.get("is_labor") else ""),
        description=description,
        start_iso=start_iso,
        end_iso=end_iso,
        color_id=formatting.get_color_id(intake.get("source", "")),
        fingerprint=job_fingerprint(intake),
    )


# ── 3. Deposit invoice, texted ─────────────────────────────────────────────────

@action(ACTION_INVOICE)
def act_deposit_invoice(state: OpsAgentState) -> dict:
    progress.working("Texting the deposit payment link…")
    intake = state.get("intake") or {}
    contact_id = _contact_id(state)
    if not contact_id:
        raise RuntimeError("No contact id — cannot invoice without a contact.")

    move_date = formatting.parse_date(intake.get("move_date", ""))
    issue_date = (move_date or datetime.now(calendar.LA_TZ)).strftime("%Y-%m-%d")
    amount = config.deposit_amount()

    invoice = ghl.create_invoice(
        contact_id=contact_id,
        amount=amount,
        item_name=f"Moving deposit — {intake.get('move_date', '')}",
        issue_date=issue_date,
        description=(
            f"Deposit to reserve your move on {intake.get('move_date', '')}. "
            "Applied to your final balance."
        ),
    )
    invoice_id = invoice.get("invoice_id")
    if not invoice_id:
        raise RuntimeError("Invoice created but no id returned")

    # Texted, per the agreed flow — not emailed.
    sent = ghl.send_invoice(invoice_id, action="sms")
    return {"invoice_id": invoice_id, "amount": amount, **sent}


# ── 4. Confirmation email ──────────────────────────────────────────────────────

@action(ACTION_EMAIL)
def act_confirmation_email(state: OpsAgentState) -> dict:
    progress.working("Sending the confirmation email…")
    intake = state.get("intake") or {}
    contact_id = _contact_id(state)
    if not contact_id:
        raise RuntimeError("No contact id — cannot email without a contact.")

    return ghl.send_email(
        contact_id,
        subject=email_template.subject(intake),
        html=email_template.html(intake),
        email_to=intake.get("email") or None,
    )


# ── Reconcile + report ─────────────────────────────────────────────────────────

_LABELS = {
    ACTION_CONTACT: "GHL contact",
    ACTION_CALENDAR: "Calendar event",
    ACTION_INVOICE: "Deposit invoice (texted)",
    ACTION_EMAIL: "Confirmation email",
}


#: Run in parallel once the contact exists. Independent of each other, so one
#: failing does not stop the other two.
FAN_OUT = ["act_calendar", "act_invoice", "act_email"]


def contact_gate(state: OpsAgentState) -> list[str] | str:
    """
    Fan out to the three remaining actions, or skip to the report.

    Returning a LIST is how LangGraph fans out to several nodes at once. The
    other three all need a contact id, so without one there is nothing to
    attach an invoice or email to — firing them anyway would just produce three
    guaranteed failures and a confusing report.
    """
    if not succeeded(state.get("ledger") or {}, ACTION_CONTACT):
        return "report"
    return FAN_OUT


def report(state: OpsAgentState) -> dict:
    """Plain-language summary of what actually happened, including what didn't."""
    ledger = state.get("ledger") or {}
    intake = state.get("intake") or {}
    name = intake.get("full_name", "the job")
    dry = config.dry_run()

    failed = failed_actions(ledger)
    done = [a for a in ALL_ACTIONS if succeeded(ledger, a)]

    lines: list[str] = []

    if dry:
        lines.append("DRY RUN — nothing below actually happened.\n")

    if not failed:
        lines.append(f"Done. {name} is booked.")
    else:
        lines.append(f"Partly done — {len(failed)} of 4 steps failed.")

    for act in ALL_ACTIONS:
        entry = ledger.get(act, {})
        status = entry.get("status", "pending")
        label = _LABELS[act]
        if status == "success":
            result = entry.get("result", {})
            detail = (
                result.get("event_id")
                or result.get("contact_id")
                or result.get("invoice_id")
                or ""
            )
            lines.append(f"  ✓ {label}" + (f"  ({detail})" if detail and not dry else ""))
        elif status == "failed":
            lines.append(f"  ✗ {label} — {entry.get('error', 'unknown error')}")
        else:
            lines.append(f"  — {label} not attempted")

    if link := ledger.get(ACTION_CALENDAR, {}).get("result", {}).get("html_link"):
        lines.append(f"\n{link}")

    if failed:
        lines.append(
            "\nSay 'retry' and I'll re-run only the failed steps — "
            f"the {len(done)} that worked won't be repeated."
        )

    return {"messages": [AIMessage(content="\n".join(lines))]}
