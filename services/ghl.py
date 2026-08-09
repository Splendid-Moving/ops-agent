"""
GoHighLevel API client.

Consolidates GHL logic that currently lives copy-pasted across four repos:
  - invoice_automation/services/ghl.py   (invoices, contact lookup)
  - calendar/pages/api/book.js           (contact upsert + duplicate handling)
  - move_reminders/nodes/send_reminders.py (conversations/messages)
  - main_website/api/submit-quote.js     (custom field IDs)

Every write respects DRY_RUN. When it is on, the payload is logged and a
synthetic result is returned so callers can keep running without side effects.
"""

import logging
import re
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import requests

from services import config

logger = logging.getLogger(__name__)

_TIMEOUT = 15


# ── Custom field IDs ───────────────────────────────────────────────────────────
# Single source of truth. Previously duplicated in three separate repos.

class CustomField:
    """
    Live field IDs, read from the GHL location. Several are constrained
    dropdowns — see PICKLIST_VALUES below.
    """

    MOVE_SIZE = "yS4Bj6LtQ3lLCuju7vl0"          # MULTIPLE_OPTIONS
    MOVING_FROM = "KyE8Eopo3MXg4aXjGnqS"        # TEXT
    MOVING_TO = "DjfpJEtJnBnDBP6nvJ1l"          # TEXT
    EXTRA_STOP = "R7tFJR0wWLkRkd4l0Gd6"         # TEXT
    MOVING_DATE = "VuatzebiX5qPrzGjl4d4"        # DATE
    ARRIVAL_TIME = "BZMRDjwmqFl957qHlTO6"       # TEXT
    MOVERS = "bB6TkyeEhBbrX1ao9eEr"             # SINGLE_OPTIONS: 2,3,4,5,6
    RATE = "VxtRrePqXpDGyAzs0TsI"               # SINGLE_OPTIONS: 3 exact strings
    FLAT_RATE = "GKTYHiNkcjb5nFxoDnda"          # TEXTBOX_LIST: Cash / Card
    LABOR = "5s3A7imsbRSiWbvSox0N"              # CHECKBOX
    JOB_NOTES = "OKkA8Uw1nVKbzAXSQuyd"          # LARGE_TEXT — read by ghl_calendar_sync
    ADDITIONAL_DETAILS = "HZgxySrqsR4IICCBWZr5"  # LARGE_TEXT
    ORIGIN = "i71w1J9MFtRcyAQYqElg"             # SINGLE_OPTIONS
    TRUCKS = "EqKPS3CMJ0jcmMo4PrVJ"             # SINGLE_OPTIONS: 1,2,3
    HEARD_ABOUT_US = "gQGxbIQ0LwMsLM6enp0h"     # MULTIPLE_OPTIONS
    BAD_MOVE = "cf9E3HWw8Qnoh6Xze7ph"           # RADIO

    #: ⚠️  DO NOT SET THIS FIELD.
    #: Checking it fires the GHL workflow that calls ghl_calendar_sync, which
    #: creates a calendar event. This agent creates its own event directly, so
    #: setting this would produce two events for one job. Listed here so the
    #: id is documented and the hazard is impossible to miss.
    CREATE_GOOGLE_EVENT = "F1qpj7jFGQCu03rqgnP8"  # CHECKBOX: "Click to create"


#: Fields the agent is never allowed to write, whatever it is asked to do.
FORBIDDEN_FIELDS = frozenset({CustomField.CREATE_GOOGLE_EVENT})


#: Exact accepted values for constrained dropdowns. A value outside these sets
#: is rejected by GHL and the field silently stays empty.
PICKLIST_VALUES: dict[str, frozenset[str]] = {
    CustomField.MOVERS: frozenset({"2", "3", "4", "5", "6"}),
    # Editable in the GHL UI — re-query the live picklist if Rate ever comes
    # back empty on a booking. Last synced 2026-08-08.
    CustomField.RATE: frozenset({
        "$115 cash /$125 card",
        "$145 cash /$155 card",
        "$180 cash /$190 card",
    }),
    CustomField.ORIGIN: frozenset({
        "Yelp", "Google My Business", "Thumbtack",
        "Previous Customer", "Local Service Ads", "Referral",
    }),
    CustomField.TRUCKS: frozenset({"1", "2", "3"}),
    CustomField.MOVE_SIZE: frozenset({
        "Studio", "1 Bedroom", "2 Bedroom", "3 Bedroom",
        "4 Bedroom", "5 Bedroom", "Other",
    }),
}


class GHLError(RuntimeError):
    """A GHL API call failed. Carries the response body, which GHL uses for detail."""

    def __init__(self, message: str, status: int | None = None, body: str = ""):
        super().__init__(message)
        self.status = status
        self.body = body


def validate_custom_fields(custom_fields: dict[str, str]) -> None:
    """
    Guard every write to a GHL custom field.

    Two failure modes this exists to prevent, both silent otherwise:

    1. A value outside a dropdown's option set. GHL accepts the request and
       leaves the field empty, so the job looks booked but has no rate on it.
    2. Setting `Create Google Event`, which fires the workflow that creates a
       calendar event — the agent already creates its own, so that means two
       events for one job.

    Raises ValueError rather than warning: both cases are bugs, and failing
    before the API call is far cheaper than reconciling afterwards.
    """
    for field_id, value in custom_fields.items():
        if field_id in FORBIDDEN_FIELDS:
            raise ValueError(
                f"Refusing to write custom field {field_id} (Create Google Event). "
                "Checking it triggers ghl_calendar_sync, which would create a "
                "second calendar event for this job."
            )

        allowed = PICKLIST_VALUES.get(field_id)
        if allowed and value not in (None, "") and str(value) not in allowed:
            raise ValueError(
                f"{str(value)!r} is not a valid option for GHL field {field_id}. "
                f"Allowed: {sorted(allowed)}. GHL would silently discard this value."
            )


def _headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {config.ghl_token()}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Version": config.GHL_API_VERSION,
    }


def _url(path: str) -> str:
    return f"{config.GHL_BASE_URL}/{path.lstrip('/')}"


def normalize_phone(phone: str) -> str:
    """Last 10 digits — the only form that reliably matches across GHL and Calendar."""
    digits = re.sub(r"\D", "", str(phone or ""))
    return digits[-10:] if len(digits) >= 10 else digits


# ── Reads ──────────────────────────────────────────────────────────────────────

def find_contact_by_phone(phone: str) -> dict[str, Any] | None:
    """
    Search contacts by phone. GHL's search is fuzzy, so results are re-checked
    against the normalized number rather than trusted directly.
    """
    target = normalize_phone(phone)
    if not target:
        return None

    resp = requests.get(
        _url("/contacts/"),
        headers=_headers(),
        params={"locationId": config.ghl_location_id(), "query": str(phone).strip()},
        timeout=_TIMEOUT,
    )
    if not resp.ok:
        raise GHLError("Contact search failed", resp.status_code, resp.text)

    for contact in resp.json().get("contacts", []):
        if normalize_phone(contact.get("phone", "")) == target:
            return contact
    return None


def get_contact(contact_id: str) -> dict[str, Any]:
    resp = requests.get(_url(f"/contacts/{contact_id}"), headers=_headers(), timeout=_TIMEOUT)
    if not resp.ok:
        raise GHLError(f"Could not fetch contact {contact_id}", resp.status_code, resp.text)
    data = resp.json()
    return data.get("contact", data)


def get_business_details() -> dict[str, Any]:
    """Business block from the location record — required on the invoice payload."""
    resp = requests.get(
        _url(f"/locations/{config.ghl_location_id()}"), headers=_headers(), timeout=_TIMEOUT
    )
    if not resp.ok:
        raise GHLError("Could not fetch location", resp.status_code, resp.text)
    return resp.json().get("location", {}).get("business", {})


# ── Contact upsert ─────────────────────────────────────────────────────────────

def upsert_contact(
    *,
    first_name: str,
    last_name: str,
    phone: str,
    email: str,
    custom_fields: dict[str, str] | None = None,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """
    Create a contact, or update it if GHL says it already exists.

    GHL signals a duplicate with a 400 whose body contains "duplicated contacts"
    and the existing id at meta.contactId. The update then MUST omit locationId —
    it is required on POST and rejected on PUT. That asymmetry is undocumented and
    was the cause of a production bug fixed in calendar/pages/api/book.js (dd17884).

    Returns {"contact_id": str, "created": bool, "dry_run": bool}.
    """
    validate_custom_fields(custom_fields or {})

    payload: dict[str, Any] = {
        "firstName": first_name,
        "lastName": last_name,
        "name": f"{first_name} {last_name}".strip(),
        "phone": phone,
        "email": email,
        "locationId": config.ghl_location_id(),
        "tags": tags or ["ops-agent"],
        "customFields": [
            {"id": fid, "value": val}
            for fid, val in (custom_fields or {}).items()
            if val not in (None, "")
        ],
    }

    if config.dry_run():
        logger.info("[DRY RUN] upsert_contact %s %s <%s>", first_name, last_name, email)
        return {"contact_id": "dry-run-contact-id", "created": True, "dry_run": True}

    resp = requests.post(_url("/contacts/"), headers=_headers(), json=payload, timeout=_TIMEOUT)

    if resp.ok:
        data = resp.json()
        contact_id = (data.get("contact") or {}).get("id") or data.get("id")
        return {"contact_id": contact_id, "created": True, "dry_run": False}

    # Duplicate -> update the existing record instead of failing.
    body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
    if "duplicated contacts" in str(body.get("message", "")):
        existing_id = (body.get("meta") or {}).get("contactId")
        if not existing_id:
            raise GHLError("Duplicate contact reported without an id", resp.status_code, resp.text)

        update_payload = {k: v for k, v in payload.items() if k != "locationId"}
        update = requests.put(
            _url(f"/contacts/{existing_id}"),
            headers=_headers(),
            json=update_payload,
            timeout=_TIMEOUT,
        )
        if not update.ok:
            raise GHLError("Could not update existing contact", update.status_code, update.text)
        return {"contact_id": existing_id, "created": False, "dry_run": False}

    raise GHLError("Contact creation failed", resp.status_code, resp.text)


# ── Invoices ───────────────────────────────────────────────────────────────────

def create_invoice(
    *,
    contact_id: str,
    amount: float,
    item_name: str,
    issue_date: str,
    description: str = "",
) -> dict[str, Any]:
    """
    Create an invoice. `issue_date` is YYYY-MM-DD. Due date is issue + 1 day,
    matching the existing invoice_automation behaviour.

    Returns {"invoice_id": str, "dry_run": bool}.
    """
    if config.dry_run():
        logger.info("[DRY RUN] create_invoice $%s for contact %s", amount, contact_id)
        return {"invoice_id": "dry-run-invoice-id", "dry_run": True}

    contact = get_contact(contact_id)
    business = get_business_details()
    tz = ZoneInfo(config.TIMEZONE)

    name = (
        contact.get("contactName")
        or f"{contact.get('firstName', '')} {contact.get('lastName', '')}".strip()
        or "Customer"
    )

    payload = {
        "altId": config.ghl_location_id(),
        "altType": "location",
        "name": item_name,
        "currency": "USD",
        "issueDate": issue_date,
        "dueDate": (datetime.now(tz) + timedelta(days=1)).strftime("%Y-%m-%d"),
        "businessDetails": {
            "name": business.get("name", "Splendid Moving"),
            "city": business.get("city", ""),
            "state": business.get("state", ""),
            "country": business.get("country", "US"),
            "postalCode": business.get("postalCode", ""),
            "website": business.get("website", ""),
            "logoUrl": business.get("logoUrl", ""),
        },
        "contactDetails": {
            "id": contact_id,
            "name": name,
            "phoneNo": contact.get("phone", ""),
            "email": contact.get("email", ""),
        },
        "items": [
            {
                "name": item_name,
                "description": description or item_name,
                "currency": "USD",
                "amount": float(amount),
                "qty": 1,
                "unitPrice": float(amount),
            }
        ],
    }

    resp = requests.post(_url("/invoices/"), headers=_headers(), json=payload, timeout=_TIMEOUT)
    if not resp.ok:
        raise GHLError("Invoice creation failed", resp.status_code, resp.text)

    data = resp.json()
    invoice = data.get("invoice", data)
    return {"invoice_id": invoice.get("_id") or invoice.get("id"), "dry_run": False}


def send_invoice(invoice_id: str, action: str = "sms") -> dict[str, Any]:
    """
    Deliver an invoice to the contact. action is 'sms' or 'email'.

    liveMode=True means GHL actually delivers it. This is the call that puts a
    real payment link in a real customer's hands.
    """
    if action not in ("sms", "email"):
        action = "sms"

    if config.dry_run():
        logger.info("[DRY RUN] send_invoice %s via %s", invoice_id, action)
        return {"sent": True, "action": action, "dry_run": True}

    resp = requests.post(
        _url(f"/invoices/{invoice_id}/send"),
        headers=_headers(),
        json={
            "altId": config.ghl_location_id(),
            "altType": "location",
            "action": action,
            "liveMode": True,
            "userId": config.ghl_user_id(),
        },
        timeout=_TIMEOUT,
    )
    if not resp.ok:
        raise GHLError(f"Invoice {invoice_id} send failed", resp.status_code, resp.text)
    return {"sent": True, "action": action, "dry_run": False}


# ── Conversations ──────────────────────────────────────────────────────────────

def send_sms(contact_id: str, message: str) -> dict[str, Any]:
    if config.dry_run():
        logger.info("[DRY RUN] send_sms to %s:\n%s", contact_id, message)
        return {"sent": True, "dry_run": True}

    resp = requests.post(
        _url("/conversations/messages"),
        headers=_headers(),
        json={"contactId": contact_id, "type": "SMS", "message": message},
        timeout=_TIMEOUT,
    )
    if not resp.ok:
        raise GHLError("SMS send failed", resp.status_code, resp.text)
    return {"sent": True, "dry_run": False, "response": resp.json()}


def send_email(
    contact_id: str,
    subject: str,
    html: str,
    *,
    email_to: str | None = None,
) -> dict[str, Any]:
    """
    Send an email through GHL so it lands on the contact's conversation record.

    NOTE: the exact field set for type=Email is not rendered in GHL's public docs.
    This payload is the documented shape; verify_services.py exercises it against a
    test contact before Phase 4 depends on it.
    """
    if config.dry_run():
        logger.info("[DRY RUN] send_email to %s | %s\n%s", contact_id, subject, html)
        return {"sent": True, "dry_run": True}

    payload: dict[str, Any] = {
        "contactId": contact_id,
        "type": "Email",
        "subject": subject,
        "html": html,
    }
    if email_to:
        payload["emailTo"] = email_to

    resp = requests.post(
        _url("/conversations/messages"), headers=_headers(), json=payload, timeout=_TIMEOUT
    )
    if not resp.ok:
        raise GHLError("Email send failed", resp.status_code, resp.text)
    return {"sent": True, "dry_run": False, "response": resp.json()}
