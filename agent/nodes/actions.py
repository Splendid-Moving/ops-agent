"""
NODES: the six side effects, plus reconcile and report.

    act_upsert_contact     -> GHL contact        (gate: everything after needs its id)
    act_calendar_event     -> Google Calendar event
    act_app_job            -> job in the crew's AppSheet app (skipped if not connected)
    act_deposit_invoice    -> GHL invoice, texted to the customer
    act_confirmation_email -> GHL email
    act_customer_sms       -> GHL text, ONLY if the two above both went out

These live in one module because they share the idempotency guard below. Six
copies of the same latch is five chances to forget it, and forgetting it means
charging a customer twice.

THE RULE EVERY NODE HERE FOLLOWS
--------------------------------
Check the ledger first. If this action already succeeded, do nothing and return.
That is what makes a retry safe: re-running the whole execute stage after a
partial failure re-fires only the parts that failed.

All six run strictly AFTER the confirm interrupt, so they are reached exactly
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
    ACTION_APP_JOB,
    ACTION_CALENDAR,
    ACTION_CONTACT,
    ACTION_EMAIL,
    ACTION_INVOICE,
    ACTION_SMS,
    ALL_ACTIONS,
    OpsAgentState,
    failed_actions,
    succeeded,
)
from schemas import checklist as cl
from schemas import email_template
from services import calendar, config, errors, fees, formatting, ghl, jobs_app, maps, rates
from services.ghl import CustomField

logger = logging.getLogger(__name__)


class ActionSkipped(Exception):
    """
    Raised by an action that does not apply on this run. NOT a failure.

    The difference matters twice over. A skip is not reported as something that
    went wrong, and — because it is recorded as "skipped" rather than "success"
    — it is re-evaluated on a retry. Recording a skip as success would be the
    worse bug of the two: the text that was held back because the email failed
    would then never be sent, even after the email went out on the retry.
    """


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
            except ActionSkipped as why:
                # Not an attempt — nothing was tried, so the count does not move
                # and a retry will look at it again.
                logger.info("%s skipped: %s", name, why)
                return {
                    "ledger": {
                        name: {
                            "status": "skipped",
                            "result": {},
                            "error": str(why),
                            "attempts": attempts - 1,
                        }
                    }
                }
            except Exception as exc:
                logger.exception("%s failed", name)
                # Staff read this in the report, so plain English; the raw
                # error is in the log line above.
                why = str(errors.explain(exc))
                progress.error(f"{_LABELS_SHORT.get(name, name)} failed", why)
                return {
                    "ledger": {
                        name: {
                            "status": "failed",
                            "result": {},
                            "error": why,
                            "attempts": attempts,
                        }
                    }
                }

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

        return wrapper

    return decorator


_DONE_TEXT = {
    ACTION_CONTACT: "Contact saved in GoHighLevel",
    ACTION_CALENDAR: "Job added to the calendar",
    ACTION_APP_JOB: "Job added to the crew app",
    ACTION_INVOICE: "Deposit payment link texted",
    ACTION_EMAIL: "Confirmation email sent",
    ACTION_SMS: "Customer texted that both went out",
}

_LABELS_SHORT = {
    ACTION_CONTACT: "Contact",
    ACTION_CALENDAR: "Calendar event",
    ACTION_APP_JOB: "Crew app job",
    ACTION_INVOICE: "Deposit link",
    ACTION_EMAIL: "Confirmation email",
    ACTION_SMS: "Customer text",
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
    # A TBD address has nothing to measure to — "TBD" is not a place.
    can_measure = pickup and dropoff and not cl.is_tbd(pickup) and not cl.is_tbd(dropoff)
    distance = maps.get_distance(pickup, dropoff, extra_stop or None) if can_measure else ""

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


# ── 2b. Job in the crew app ────────────────────────────────────────────────────

def _address_or_blank(value: str) -> str:
    """
    The crew app treats every address as a real place: it geocodes it for drive
    times and turns it into a map link. "TBD" is not a place, so it goes in as
    blank and the job notes say what is still unknown.
    """
    value = (value or "").strip()
    return "" if not value or cl.is_tbd(value) else value


def app_job_payload(intake: dict, contact_id: str = "") -> dict:
    """
    The job as the crew app stores it. Pure — no I/O — so it is tested directly.

    Shapes that matter on the other side:
      - move_date is ISO (YYYY-MM-DD); the database column is a DATE.
      - scheduled_time is the START of the arrival window as HH:MM, in LA time.
        The app has a single start time, so the full window goes into the notes.
      - move_type is one of the app's three options: Labor, Local, Long Distance.
        The agent only books local moves and labor-only jobs.
      - rates are the plain numbers for the crew size, not the GHL string.
    """
    move_date = formatting.parse_date(intake.get("move_date", ""))
    if move_date is None:
        raise ValueError(f"Unparseable move date {intake.get('move_date')!r}")

    window = formatting.parse_arrival_time(intake.get("arrival_time", ""), move_date)
    if window is None:
        raise ValueError(f"Unparseable arrival window {intake.get('arrival_time')!r}")
    start = datetime.fromisoformat(window[0])

    is_labor = bool(intake.get("is_labor"))
    pickup = _address_or_blank(intake.get("pickup_address", ""))
    dropoff = "" if is_labor else _address_or_blank(intake.get("dropoff_address", ""))
    extra_stop = _address_or_blank(intake.get("extra_stop", ""))

    rate = rates.rate_for(intake.get("movers", ""))
    office_fees = fees.parse_fees(intake.get("job_notes", ""))

    notes = [f"Arrival window: {intake.get('arrival_time', '').strip()}"]
    if is_labor:
        notes.append("LABOR ONLY — no transport.")
    if not pickup:
        notes.append("Pickup address TBD.")
    if not is_labor and not dropoff:
        notes.append("Drop-off address TBD.")
    if intake.get("job_notes"):
        notes.append(str(intake["job_notes"]).strip())

    return {
        "fingerprint": job_fingerprint(intake),
        "ghl_contact_id": contact_id,
        "full_name": (intake.get("full_name") or "").strip(),
        "phone": ghl.normalize_phone(intake.get("phone", "")),
        "email": ghl.normalize_email(intake.get("email", "")),
        "move_date": move_date.strftime("%Y-%m-%d"),
        "scheduled_time": start.strftime("%H:%M"),
        "move_type": "Labor" if is_labor else "Local",
        "move_size": (intake.get("move_size") or "").strip(),
        "origin_address": pickup,
        "destination_address": dropoff,
        "extra_stop": extra_stop,
        "movers_required": rate.movers if rate else None,
        "cash_rate_per_hour": rate.cash if rate else None,
        "card_rate_per_hour": rate.card if rate else None,
        "lead_source": (intake.get("source") or "").strip(),
        "job_notes": "\n".join(notes),
        # Read from the notes so the bill includes them; the notes keep the text.
        "gas_fee": office_fees.gas_fee,
        "heavy_item_fee": office_fees.heavy_item_fee,
        "discount_percent": office_fees.discount_percent,   # 10 means 10%
    }


@action(ACTION_APP_JOB)
def act_app_job(state: OpsAgentState) -> dict:
    """
    Put the job into the crew's app, so the office does not re-type it.

    Skipped rather than failed when the link is not configured: a deployment
    that has not been connected yet books exactly as it did before.

    Safe to retry. The webhook keys the job on the same fingerprint the calendar
    uses (phone + date), so a request whose response was lost comes back with
    the job it already created rather than making a second one.
    """
    if not config.jobs_app_enabled():
        raise ActionSkipped("not connected (JOBS_APP_WEBHOOK_URL / JOBS_APP_WEBHOOK_SECRET not set)")

    progress.working("Adding the job to the crew app…")
    intake = state.get("intake") or {}
    return jobs_app.create_job(app_job_payload(intake, _contact_id(state)))


# ── 3. Deposit invoice, texted ─────────────────────────────────────────────────

@action(ACTION_INVOICE)
def act_deposit_invoice(state: OpsAgentState) -> dict:
    progress.working("Texting the deposit payment link…")
    intake = state.get("intake") or {}
    contact_id = _contact_id(state)
    if not contact_id:
        raise RuntimeError("No contact id — cannot invoice without a contact.")

    # Issued today, because that is when we want it paid — the deposit is what
    # holds the slot. Dating it to the move date made the invoice due three days
    # before it was issued, which GHL refuses with "Issue date cannot be after
    # due date", so every booking further out than tomorrow silently lost its
    # payment link. The move date belongs in the description, not in the dates.
    issue_date = datetime.now(calendar.LA_TZ).strftime("%Y-%m-%d")
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


# ── 5. Tell the customer both went out ─────────────────────────────────────────

#: What the customer receives. One line, because it is a receipt rather than a
#: message that wants a reply.
CUSTOMER_SMS = "Hi {first_name}, the confirmation email & deposit link have been sent!"


def first_name(intake: dict) -> str:
    """
    What to call them in the text.

    Substituted here rather than left as a merge field for GoHighLevel to fill
    in: merge fields are resolved for workflows and templates, and this goes out
    through /conversations/messages, which sends exactly the characters it is
    given. A customer receiving "Hi {first_name}," is worse than no text at all.
    """
    first, _ = formatting.split_name(intake.get("full_name", ""))
    return first or "there"


@action(ACTION_SMS)
def act_customer_sms(state: OpsAgentState) -> dict:
    """
    Only once BOTH the confirmation email and the deposit link have gone out.

    The text says two specific things happened, so sending it when one of them
    did not is telling the customer something untrue — and the one thing worse
    than a booking half-completed is a customer who has been told it wasn't.
    """
    ledger = state.get("ledger") or {}

    missing = [
        _LABELS_SHORT[name]
        for name in (ACTION_EMAIL, ACTION_INVOICE)
        if not succeeded(ledger, name)
    ]
    if missing:
        raise ActionSkipped(f"{' and '.join(missing)} didn't go out")

    contact_id = _contact_id(state)
    if not contact_id:
        raise RuntimeError("No contact id — cannot text without a contact.")

    progress.working("Texting the customer to say both went out…")
    intake = state.get("intake") or {}
    message = CUSTOMER_SMS.format(first_name=first_name(intake))
    return {"message": message, **ghl.send_sms(contact_id, message)}


# ── Reconcile + report ─────────────────────────────────────────────────────────

_LABELS = {
    ACTION_CONTACT: "GHL contact",
    ACTION_CALENDAR: "Calendar event",
    ACTION_APP_JOB: "Job in the crew app (AppSheet)",
    ACTION_INVOICE: "Deposit invoice (texted)",
    ACTION_EMAIL: "Confirmation email",
    ACTION_SMS: "Text to the customer",
}


#: Run in parallel once the contact exists. Independent of each other, so one
#: failing does not stop the others.
#:
#: act_sms is NOT among them. It reports on two of these, so it has to wait for
#: them — it runs after all of them converge, and decides for itself whether
#: there is anything true to say.
FAN_OUT = ["act_calendar", "act_app_job", "act_invoice", "act_email"]


def contact_gate(state: OpsAgentState) -> list[str] | str:
    """
    Fan out to the remaining actions, or skip to the report.

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
        lines.append(f"Partly done — {len(failed)} of {len(ALL_ACTIONS)} steps failed.")
        if ACTION_CONTACT in failed:
            # Every other step hangs off the contact, so none of them ran.
            lines.append("So nothing else was done — no calendar event, deposit link, "
                         "email or text.")

    for act in ALL_ACTIONS:
        entry = ledger.get(act, {})
        status = entry.get("status", "pending")
        label = _LABELS[act]
        if status == "success":
            result = entry.get("result", {})
            detail = (
                result.get("event_id")
                or result.get("job_id")
                or result.get("contact_id")
                or result.get("invoice_id")
                or ""
            )
            lines.append(f"  ✓ {label}" + (f"  ({detail})" if detail and not dry else ""))
        elif status == "failed":
            lines.append(f"  ✗ {label} — {entry.get('error', 'unknown error')}")
        elif status == "skipped":
            # Deliberately held back, not broken. Says why, because "not
            # attempted" on its own reads like something went wrong.
            lines.append(f"  — {label} not sent: {entry.get('error', 'did not apply')}")
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
