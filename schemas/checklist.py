"""
The booking checklist — what must be true before a job can be created.

Deliberately data, not code. Adding a requirement means adding a FieldSpec, not
editing graph logic. Every check here is plain Python: the LLM proposes values,
this module decides whether they are acceptable. No model ever gets to declare
the checklist satisfied.

RULES ENCODED HERE
------------------
  always required    full name, email, phone, pickup address
  required unless    drop-off address — not needed for labor-only jobs
    labor
  always asked       move date, arrival window, crew size, job notes
  never asked        extra stop (captured only if volunteered)
  derived            rate (from crew size), deposit (fixed)
"""

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from services import config, formatting, rates
from services.calendar import LA_TZ

# ── Validators ─────────────────────────────────────────────────────────────────

_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")


def valid_email(value: str) -> str | None:
    """Returns an error message, or None when acceptable."""
    if not _EMAIL.match(value.strip()):
        return f"{value!r} doesn't look like an email address."
    return None


def valid_phone(value: str) -> str | None:
    digits = re.sub(r"\D", "", value)
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    if len(digits) != 10:
        return f"{value!r} isn't a 10-digit US phone number."
    return None


def valid_name(value: str) -> str | None:
    if len(value.strip()) < 2:
        return "That name looks too short."
    if not re.search(r"[A-Za-z]", value):
        return f"{value!r} doesn't contain a name."
    return None


def valid_move_date(value: str) -> str | None:
    """
    Parseable, not in the past, and inside the 2-day minimum lead time.

    Lead time is a warning rather than a hard failure — same-day jobs do happen
    and the confirm gate is where a human decides.
    """
    parsed = formatting.parse_date(value)
    if parsed is None:
        return f"Couldn't read {value!r} as a date. Try 03/14/2026."

    move_day = parsed.date()
    today = datetime.now(LA_TZ).date()
    if move_day < today:
        return f"{value} is in the past."
    if move_day < today + timedelta(days=config.MIN_LEAD_DAYS):
        return (
            f"NOTE: {value} is inside the usual {config.MIN_LEAD_DAYS}-day lead time. "
            "Fine if intended."
        )
    return None


def valid_arrival_time(value: str) -> str | None:
    if formatting.parse_arrival_time(value, datetime(2026, 1, 1)) is None:
        return f"Couldn't read {value!r} as an arrival window. Try '8-9am' or '2-4pm'."
    return None


def valid_movers(value: str) -> str | None:
    if rates.is_supported_crew_size(value):
        return None
    if rates.is_out_of_scope_crew_size(value):
        return (
            f"{value}-mover jobs are priced by hand and aren't automated — "
            "book that one directly in GHL."
        )
    return f"Crew size must be one of {', '.join(map(str, rates.SUPPORTED_MOVER_COUNTS))}."


def valid_address(value: str) -> str | None:
    """
    Shallow check only. Real completion and verification is the address node's
    job — this just rejects obvious non-addresses before spending an API call.
    """
    if len(value.strip()) < 5:
        return f"{value!r} is too short to be an address."
    if not re.search(r"\d", value):
        return f"{value!r} has no street number."
    return None


# ── Field specs ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FieldSpec:
    name: str
    label: str
    #: Which side effects break without it — drives partial-failure messaging.
    required_for: tuple[str, ...]
    #: Question shown when it's missing.
    ask: str
    validator: Callable[[str], str | None] | None = None
    normalizer: Callable[[str], str] | None = None
    #: Not required when the job is labor-only.
    skip_if_labor: bool = False
    #: Must be asked even though a blank answer is acceptable.
    always_ask: bool = False
    #: Never proactively asked; captured only if volunteered.
    never_ask: bool = False
    examples: tuple[str, ...] = field(default=())


CHECKLIST: tuple[FieldSpec, ...] = (
    FieldSpec(
        name="full_name",
        label="Full name",
        required_for=("contact", "calendar", "invoice", "email"),
        ask="What's the customer's full name?",
        validator=valid_name,
        examples=("Sarah Chen", "Maria De La Cruz"),
    ),
    FieldSpec(
        name="email",
        label="Email",
        required_for=("contact", "email"),
        ask="What's their email address?",
        validator=valid_email,
        normalizer=lambda v: v.strip().lower(),
        examples=("sarah@example.com",),
    ),
    FieldSpec(
        name="phone",
        label="Phone",
        required_for=("contact", "calendar", "invoice"),
        ask="What's their phone number?",
        validator=valid_phone,
        normalizer=formatting.format_phone,
        examples=("(818) 555-0142",),
    ),
    FieldSpec(
        name="pickup_address",
        label="Pickup address",
        required_for=("contact", "calendar"),
        ask="What's the pickup address?",
        validator=valid_address,
        examples=("412 N Maple Ave, Burbank CA 91505",),
    ),
    FieldSpec(
        name="dropoff_address",
        label="Drop-off address",
        required_for=("contact", "calendar"),
        ask="What's the drop-off address?",
        validator=valid_address,
        skip_if_labor=True,
        examples=("1830 Pine St, Glendale CA 91206",),
    ),
    FieldSpec(
        name="extra_stop",
        label="Extra stop",
        required_for=(),
        ask="",  # never asked
        validator=valid_address,
        never_ask=True,
    ),
    FieldSpec(
        name="move_date",
        label="Move date",
        required_for=("calendar", "contact"),
        ask="What date is the move?",
        validator=valid_move_date,
        examples=("03/14/2026", "next Friday"),
    ),
    FieldSpec(
        name="arrival_time",
        label="Arrival window",
        required_for=("calendar", "contact"),
        ask="What's the arrival window?",
        validator=valid_arrival_time,
        normalizer=formatting.normalize_arrival_label,
        examples=("8-9am", "2-4pm", "11am-1pm"),
    ),
    FieldSpec(
        name="movers",
        label="Crew size",
        required_for=("calendar", "contact"),
        ask="How many movers?",
        validator=valid_movers,
        normalizer=lambda v: re.sub(r"\D", "", v) or v.strip(),
        examples=("2", "3", "4"),
    ),
    FieldSpec(
        name="job_notes",
        label="Job notes",
        required_for=(),
        # Always asked: additional charges, gas fees and similar live here, and
        # they are commercially important. A blank answer is fine; not asking
        # is not.
        ask="Any notes for this job? (extra charges, gas fee, stairs, parking — 'none' is fine)",
        always_ask=True,
    ),
)

BY_NAME: dict[str, FieldSpec] = {spec.name: spec for spec in CHECKLIST}


# ── Labor inference ────────────────────────────────────────────────────────────


def infer_is_labor(pickup: str, dropoff: str, explicit: bool | None) -> bool | None:
    """
    Decide labor-only status without asking, where possible.

    Two distinct addresses means it is a move, not labor — that inference is
    safe and saves a question on the common case. One address is NOT sufficient
    evidence of labor: a full move whose drop-off simply wasn't extracted looks
    identical. Those return None and get asked.
    """
    if explicit is not None:
        return explicit

    has_pickup = bool((pickup or "").strip())
    has_dropoff = bool((dropoff or "").strip())

    if has_pickup and has_dropoff:
        return False  # two addresses -> definitely a move
    return None       # ambiguous -> ask


# ── Evaluation ─────────────────────────────────────────────────────────────────


@dataclass
class ChecklistResult:
    missing: list[FieldSpec]
    invalid: dict[str, str]      # field name -> error message
    warnings: dict[str, str]     # field name -> non-blocking note
    needs_labor_answer: bool

    @property
    def is_complete(self) -> bool:
        return not self.missing and not self.invalid and not self.needs_labor_answer

    def all_questions(self) -> list[str]:
        """Every outstanding question, for one batched ask."""
        questions = []
        if self.needs_labor_answer:
            questions.append(
                "Is this a labor-only job (loading/unloading help, no transport), "
                "or a full move?"
            )
        questions.extend(spec.ask for spec in self.missing if spec.ask)
        questions.extend(f"{BY_NAME[name].label}: {err}" for name, err in self.invalid.items())
        return questions


def evaluate(intake: dict) -> ChecklistResult:
    """
    Check an intake record against the checklist.

    Pure and deterministic — same input, same result, no model involved.
    """
    is_labor = infer_is_labor(
        intake.get("pickup_address", ""),
        intake.get("dropoff_address", ""),
        intake.get("is_labor"),
    )

    missing: list[FieldSpec] = []
    invalid: dict[str, str] = {}
    warnings: dict[str, str] = {}

    for spec in CHECKLIST:
        if spec.never_ask:
            continue
        if spec.skip_if_labor and is_labor is True:
            continue

        value = str(intake.get(spec.name, "") or "").strip()

        if not value:
            # Notes are always asked, but only once — a recorded "none" counts.
            if spec.always_ask and intake.get("notes_asked"):
                continue
            missing.append(spec)
            continue

        if spec.validator and (error := spec.validator(value)):
            # A "NOTE:" prefix marks advisory output rather than a hard failure.
            if error.startswith("NOTE:"):
                warnings[spec.name] = error.removeprefix("NOTE:").strip()
            else:
                invalid[spec.name] = error

    return ChecklistResult(
        missing=missing,
        invalid=invalid,
        warnings=warnings,
        # Only block on labor when it actually changes what's required.
        needs_labor_answer=is_labor is None,
    )
