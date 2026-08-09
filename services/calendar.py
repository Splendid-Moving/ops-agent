"""
Google Calendar client.

The calendar IS the job database for this business. Four other repos regex-parse
the event description, so `build_description` is effectively a wire format —
see the warning on that function before changing anything about it.
"""

import base64
import json
import logging
import re
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from google.oauth2 import service_account
from googleapiclient.discovery import build

from services import config

logger = logging.getLogger(__name__)

LA_TZ = ZoneInfo(config.TIMEZONE)

SCOPES = ["https://www.googleapis.com/auth/calendar"]

#: Namespace for agent metadata on events. Stored in extendedProperties.private,
#: NEVER in the description — see build_description.
FINGERPRINT_KEY = "splendid_fingerprint"


_service = None


def get_service():
    """Authenticated Calendar service, memoized per process."""
    global _service
    if _service is not None:
        return _service

    creds_b64 = config.google_credentials_b64()
    if not creds_b64:
        raise RuntimeError("GOOGLE_CREDENTIALS_B64 is not set")

    info = json.loads(base64.b64decode(creds_b64).decode("utf-8"))
    creds = service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
    _service = build("calendar", "v3", credentials=creds, cache_discovery=False)
    return _service


# ── Description format ─────────────────────────────────────────────────────────

def build_description(
    *,
    customer: str,
    phone: str,
    date: str,
    from_address: str,
    to_address: str,
    distance: str = "",
    rate: str = "",
    movers: str = "",
    deposit: str = config.CALENDAR_DEPOSIT_PLACEHOLDER,
    source: str = "",
    notes: str = "",
    extra_stop: str = "",
) -> str:
    """
    Build the canonical job description.

    ⚠️  THIS IS A WIRE FORMAT. move_reminders, job_form_automation,
    invoice_automation and ghl_calendar_sync all parse it with
    `^([A-Za-z ]+):\\s*(.*)` — meaning ANY line shaped like "Word:" becomes a
    field in their output. Adding a line here silently injects a field into four
    other production pipelines.

    Agent metadata belongs in extendedProperties.private (see create_event).
    tests/test_formatting.py pins this output so a regression is caught locally.
    """
    lines = [
        f"Customer: {customer}",
        f"Phone: {phone}",
        f"Date: {date}",
        f"From: {from_address}",
    ]
    if extra_stop:
        lines.append(f"Extra stop: {extra_stop}")
    lines += [
        f"To: {to_address}",
        f"Distance: {distance}",
        f"Rate: {rate}",
        f"Movers: {movers}",
        f"Deposit: {deposit}",
        f"Source: {source}",
        f"Notes:\n{notes}",
    ]
    return "\n".join(lines)


def parse_description(description: str) -> dict[str, str]:
    """
    Parse a job description back into fields. Mirrors the regex used by
    move_reminders/nodes/fetch_calendar_jobs.py so this agent sees exactly what
    the other pipelines see. Multi-line values (Notes) are joined.
    """
    fields: dict[str, str] = {}
    if not description:
        return fields

    current_key: str | None = None
    buffer: list[str] = []

    for line in description.strip().splitlines():
        if m := re.match(r"^([A-Za-z ]+):\s*(.*)", line):
            if current_key:
                fields[current_key] = "\n".join(buffer).strip()
            current_key = m.group(1).strip().lower().replace(" ", "_")
            buffer = [m.group(2).strip()]
        elif current_key:
            buffer.append(line.strip())

    if current_key:
        fields[current_key] = "\n".join(buffer).strip()
    return fields


def is_job_event(fields: dict[str, str]) -> bool:
    """
    A calendar event is a job only if it carries customer, phone, and a date.
    Filters out crew meetings, blocks, PTO and anything else on the calendar.
    """
    return bool(
        fields.get("customer", "").strip()
        and fields.get("phone", "").strip()
        and (fields.get("date", "").strip() or fields.get("move_date", "").strip())
    )


# ── Reads ──────────────────────────────────────────────────────────────────────

def list_events(start: datetime, end: datetime) -> list[dict[str, Any]]:
    """Raw events in a datetime range. Recurring events are expanded."""
    result = (
        get_service()
        .events()
        .list(
            calendarId=config.calendar_id(),
            timeMin=start.astimezone(LA_TZ).isoformat(),
            timeMax=end.astimezone(LA_TZ).isoformat(),
            singleEvents=True,
            orderBy="startTime",
            maxResults=2500,
        )
        .execute()
    )
    return result.get("items", [])


def list_jobs(start: datetime, end: datetime) -> list[dict[str, Any]]:
    """
    Job events in a range, parsed into flat dicts. Non-job events are dropped.

    This is the primitive the analytics node aggregates over.
    """
    jobs = []
    for event in list_events(start, end):
        fields = parse_description(event.get("description", ""))
        if not is_job_event(fields):
            continue

        start_block = event.get("start", {})
        end_block = event.get("end", {})
        title = (event.get("summary") or "").strip()

        # Where the event ACTUALLY sits on the calendar, as mm/dd/yyyy.
        #
        # Distinct from `move_date`, which is the `Date:` line typed into the
        # description. They normally agree, but not always — a real July 2026
        # event sat on Jul 2 while its description read 07/03/2026. The crew
        # works from the calendar, so anything about *when* a job happens must
        # use this field, not the typed one.
        raw_start = start_block.get("dateTime") or start_block.get("date", "")
        calendar_date = ""
        if raw_start:
            try:
                calendar_date = datetime.fromisoformat(raw_start).astimezone(LA_TZ).strftime("%m/%d/%Y")
            except ValueError:
                calendar_date = ""

        jobs.append(
            {
                "event_id": event.get("id"),
                "title": title,
                "is_labor": "(labor)" in title.lower(),
                "customer": fields.get("customer", ""),
                "phone": fields.get("phone", ""),
                "calendar_date": calendar_date,
                "move_date": fields.get("date", "") or fields.get("move_date", ""),
                #: True when the typed date disagrees with where the event sits.
                "date_mismatch": bool(
                    calendar_date
                    and fields.get("date", "")
                    and calendar_date != fields.get("date", "")
                ),
                "start_time": start_block.get("dateTime") or start_block.get("date", ""),
                "end_time": end_block.get("dateTime") or end_block.get("date", ""),
                "from_address": fields.get("from", ""),
                "to_address": fields.get("to", ""),
                "extra_stop": fields.get("extra_stop", ""),
                "distance": fields.get("distance", ""),
                "rate": fields.get("rate", ""),
                "movers": fields.get("movers", ""),
                "deposit": fields.get("deposit", ""),
                "source": fields.get("source", ""),
                "notes": fields.get("notes", ""),
                "color_id": event.get("colorId"),
                "html_link": event.get("htmlLink"),
            }
        )
    return jobs


def find_by_fingerprint(fingerprint: str) -> list[dict[str, Any]]:
    """
    Look up events this agent created, by fingerprint. Used for duplicate
    detection before booking — cheap because it's an indexed private property
    query rather than a scan.
    """
    result = (
        get_service()
        .events()
        .list(
            calendarId=config.calendar_id(),
            privateExtendedProperty=f"{FINGERPRINT_KEY}={fingerprint}",
            singleEvents=True,
            maxResults=10,
        )
        .execute()
    )
    return result.get("items", [])


def find_duplicate_job(phone: str, move_date: str) -> dict[str, Any] | None:
    """
    Catch a job already on the calendar for this customer and date, regardless of
    who created it — the agent, the GHL webhook, or a human. Matches on
    normalized phone so formatting differences don't cause a false negative.
    """
    from services.ghl import normalize_phone

    from services.formatting import parse_date

    parsed = parse_date(move_date)
    if not parsed:
        return None

    day_start = parsed.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=LA_TZ)
    day_end = day_start + timedelta(days=1)

    target = normalize_phone(phone)
    for job in list_jobs(day_start, day_end):
        if target and normalize_phone(job["phone"]) == target:
            return job
    return None


# ── Writes ─────────────────────────────────────────────────────────────────────

def create_event(
    *,
    title: str,
    description: str,
    start_iso: str,
    end_iso: str,
    color_id: str | None = None,
    fingerprint: str | None = None,
) -> dict[str, Any]:
    """
    Create a job event.

    `fingerprint` goes into extendedProperties.private, which is invisible to
    humans and to every regex parser in the stack, and is queryable via
    find_by_fingerprint.

    Returns {"event_id": str, "html_link": str, "dry_run": bool}.
    """
    body: dict[str, Any] = {
        "summary": title,
        "description": description,
        "start": {"dateTime": start_iso, "timeZone": config.TIMEZONE},
        "end": {"dateTime": end_iso, "timeZone": config.TIMEZONE},
    }
    if color_id:
        body["colorId"] = color_id
    if fingerprint:
        body["extendedProperties"] = {"private": {FINGERPRINT_KEY: fingerprint}}

    if config.dry_run():
        logger.info("[DRY RUN] create_event %r %s\n%s", title, start_iso, description)
        return {"event_id": "dry-run-event-id", "html_link": "", "dry_run": True}

    event = (
        get_service()
        .events()
        .insert(calendarId=config.calendar_id(), body=body)
        .execute()
    )
    return {
        "event_id": event.get("id"),
        "html_link": event.get("htmlLink"),
        "dry_run": False,
    }
