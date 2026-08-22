"""
NODE: Extract screenshot
PURPOSE: Read customer details out of an attached image into a strict schema.
INPUT:   state.messages (last human message, possibly with image parts)
OUTPUT:  {"intake": {...}, "field_confidence": {...}}

Highest-stakes model call in the agent: its output becomes a real contact, a
real calendar event, and a real invoice. The prompt is written to make the model
comfortable saying "I don't know" — a blank field costs one question, a
confidently misread phone number costs a customer.

If there is no image (the user is booking by describing the job), this node
returns an empty record and the checklist asks for everything.
"""

import logging
from datetime import datetime, timedelta

from langchain_core.messages import HumanMessage, SystemMessage

from agent import progress
from agent.models import get_model
from agent.state import OpsAgentState, booking_has_run, new_ledger
from schemas import business_context
from schemas.intake import MIN_CONFIDENCE, ScreenshotExtraction
from services import formatting, ocr
from services.calendar import LA_TZ

logger = logging.getLogger(__name__)


def _system_prompt() -> str:
    """
    Built per call so TODAY is always current.

    This is not cosmetic. Without a date anchor the model cannot resolve
    "tomorrow" or "next Friday", so it reported them verbatim at low confidence,
    they fell below the threshold, and the agent asked for a date the screenshot
    had already given it.
    """
    now = datetime.now(LA_TZ)
    tomorrow = now + timedelta(days=1)
    return SYSTEM_PROMPT_BODY.format(
        today=f"{now:%A, %B %-d, %Y}",
        today_iso=f"{now:%m/%d/%Y}",
        tomorrow=f"{tomorrow:%A}",
        tomorrow_date=f"{tomorrow:%m/%d/%Y}",
        context=business_context.for_extraction(),
    )


SYSTEM_PROMPT_BODY = f"""You extract moving-job details from screenshots for \
Splendid Moving, a Los Angeles moving company. Staff paste in screenshots of \
customer enquiries — Yelp message threads, SMS conversations, emails, web form \
submissions, handwritten notes — and you turn them into structured data.

{{context}}

# Today
Today is {{today}} — {{today_iso}} — in Los Angeles.
Tomorrow is {{tomorrow}}, {{tomorrow_date}}.

# Your one job
Report what is *actually visible*. You are not filling in a form; you are \
transcribing evidence.

# The cost of being wrong
Your output creates a real CRM contact, books a real truck, and sends a real \
customer an invoice. A field you leave blank costs one follow-up question. A \
field you guess wrong sends a crew to the wrong address on the wrong day. \
These are not close in cost. When torn, leave it blank.

# Confidence, honestly
Give every field a confidence from 0.0 to 1.0:
  1.0       printed clearly, unambiguous, you can point at it
  0.8-0.9   clearly legible, minor ambiguity (e.g. it's obviously the customer's \
name but not labelled as such)
  {MIN_CONFIDENCE}       the threshold — anything below this is treated as missing and \
double-checked with a human
  0.4-0.7   readable but uncertain: cut-off text, ambiguous role, plausible \
misreading of a digit
  0.0-0.3   guessing

Do not inflate confidence. A 0.6 that gets checked is a success. A 0.9 that is \
wrong is a failure.

# source_text
For every field with a value, copy the literal text you read it from, verbatim, \
including surrounding words if that's what disambiguates it. This lets a human \
audit your reading without reopening the image.

# Field-specific rules

**full_name** — the CUSTOMER's name. Screenshots often contain other names: \
the business, a Yelp rep, the staff member. If several names appear and it is \
unclear which is the customer, lower your confidence and say why in source_text.

**phone** — digits exactly as shown. Do not reformat, do not add or drop digits. \
If partially obscured, leave blank rather than reconstructing.

**email** — exactly as shown. Watch for line-wrapped addresses split across two \
lines; join them only when it is unambiguous.

**pickup_address / dropoff_address** — copy exactly what is written, even if \
incomplete. "412 N Maple Ave" with no city is a perfectly good extraction — \
write it as-is. NEVER add a city, state or ZIP that is not written down. A \
separate verification step completes addresses properly; inventing one here \
corrupts that step's input.
  - Determine direction from context: "moving from X to Y", "pickup"/"dropoff", \
"current address"/"new address".
  - If two addresses appear but the direction is unclear, put the first in \
pickup, and lower BOTH confidences.
  - **Return the address ONLY — never the label in front of it.** The label tells \
you which field it belongs in; it is not part of the value.
      screenshot shows "From: 614 E Verdugo Ave"
        pickup_address = "614 E Verdugo Ave"       correct
        pickup_address = "From: 614 E Verdugo Ave" wrong — the calendar adds
                                                   its own "From:" and you get
                                                   "From: From: 614 E Verdugo Ave"

**extra_stop** — only if a third address is clearly part of the same job. Never \
infer one.

**is_labor** — set true ONLY on explicit evidence: "labor only", "just need \
help loading", "loading help", "no truck needed". Set false ONLY on explicit \
evidence of transport between two places. Otherwise null. Do NOT reason from \
how many addresses you found — that inference is handled elsewhere, with more \
context than you have.

**move_date** — the date of the MOVE, not the date of the message. Screenshots \
often show both.

RESOLVE RELATIVE DATES YOURSELF, against today's date above, and output \
**mm/dd/yyyy**. You know what day it is, so "tomorrow", "next Friday" and \
"the 14th" are answerable — treat them as high confidence (0.9+) when the \
reference is unambiguous, exactly as you would a printed date. Put the original \
wording in source_text.

  "moving tomorrow"      -> {{tomorrow_date}}   confidence 0.95
  "next Friday"          -> the Friday of next week
  "the 14th"             -> the next 14th still in the future
  "Saturday"             -> the coming Saturday
  "early next month"     -> too vague. Leave blank.
  "sometime in spring"   -> too vague. Leave blank.

Only drop to low confidence when the reference is genuinely ambiguous (e.g. \
"next weekend" close to a weekend, or two candidate dates on screen). Reporting \
a resolvable date as uncertain makes the agent ask a question the screenshot \
already answered.

**arrival_time** — a time or window, e.g. "8-9am", "morning", "2pm".

**movers** — crew size, only if stated. Do not infer it from home size.

**move_size** — studio / 1 bedroom / 2 bedroom / 3 bedroom / 4 bedroom / \
5 bedroom / other, only if stated.

**source** — where the lead came from, if visible. The Yelp UI in the screenshot \
is itself good evidence of "Yelp".

**notes / overall_notes** — anything a dispatcher would want: stairs, elevator, \
walk-up floor number, heavy or unusual items, parking or permit constraints, \
timing constraints, mentions of extra fees. Capture generously; this is cheap \
to include and expensive to lose.

# When the image has nothing useful
Return the empty schema with all confidences at 0.0. That is a valid, useful \
answer — do not manufacture plausible-looking data to fill it."""


def _raw_base64(part: dict) -> str:
    """Pull the bare base64 payload out of an image content block."""
    if part.get("type") == "image_url":
        url = (part.get("image_url") or {}).get("url", "")
        return url.split(",", 1)[1] if "," in url else ""
    source = part.get("source") or {}
    return source.get("data", "")


def _ocr_text(images: list[dict]) -> str:
    """
    Character-accurate text for the attached images.

    Degrades to "" rather than failing the booking — GPT-only extraction is how
    this worked before OCR existed, and it is better than refusing to read a
    screenshot because a secondary API is down.
    """
    chunks = []
    for part in images:
        payload = _raw_base64(part)
        if not payload:
            continue
        try:
            chunks.append(ocr.extract_text(payload))
            progress.done("Checked exact spelling")
        except ocr.OCRUnavailable as exc:
            logger.warning("OCR unavailable, continuing without it: %s", exc)
            return ""
    return "\n".join(c for c in chunks if c)


def _apply_ocr_corrections(intake: dict, ocr_text: str) -> list[str]:
    """
    Overwrite the high-entropy fields with what OCR actually saw.

    Only email and phone. Everything else benefits more from GPT's understanding
    of layout than from exact characters, and every other field has a downstream
    check that catches a slip. These two do not.
    """
    if not ocr_text:
        return []

    notes: list[str] = []

    corrected, note = ocr.correct_email(intake.get("email", ""), ocr_text)
    if note:
        notes.append(note)
        logger.info("OCR: %s", note)
    if corrected:
        intake["email"] = corrected

    corrected, note = ocr.correct_phone(intake.get("phone", ""), ocr_text)
    if note:
        notes.append(note)
        logger.info("OCR: %s", note)
        intake["phone"] = formatting.format_phone(corrected)

    return notes


def _image_parts(message) -> list[dict]:
    content = getattr(message, "content", None)
    if not isinstance(content, list):
        return []
    return [
        part for part in content
        if isinstance(part, dict) and part.get("type") in ("image", "image_url")
    ]


def _text_of(message) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    return " ".join(
        part.get("text", "")
        for part in content
        if isinstance(part, dict) and part.get("type") == "text"
    ).strip()


def _to_intake(extraction: ScreenshotExtraction) -> tuple[dict, dict[str, float]]:
    """
    Flatten an extraction into the working record, dropping anything below the
    confidence threshold so the checklist asks about it instead.
    """
    intake: dict = {}
    confidence: dict[str, float] = {}

    simple = (
        "full_name", "email", "phone",
        "pickup_address", "dropoff_address", "extra_stop",
        "move_date", "arrival_time", "movers", "move_size", "source",
    )
    for name in simple:
        field = getattr(extraction, name)
        confidence[name] = field.confidence
        if field.is_usable:
            intake[name] = field.value.strip()

    # Normalise the two fields with a canonical form, so downstream comparison
    # and duplicate detection work on consistent values.
    if intake.get("phone"):
        intake["phone"] = formatting.format_phone(intake["phone"])
    if intake.get("source"):
        intake["source"] = formatting.normalize_source(intake["source"])

    if extraction.is_labor is not None:
        intake["is_labor"] = extraction.is_labor

    # Extraction notes seed job_notes but do NOT count as having asked — the
    # user is still prompted, because extra charges are agreed with staff and
    # will not be in a customer's screenshot.
    notes = [
        part for part in (
            extraction.notes.value if extraction.notes.is_usable else None,
            extraction.overall_notes,
        ) if part
    ]
    if notes:
        intake["job_notes"] = " | ".join(notes)

    return intake, confidence


#: Phrases that mean "run the failed steps again", NOT "start a new job".
#:
#: A retry deliberately reuses the existing intake and ledger — that is how the
#: ledger skips the actions that already succeeded. Clearing state on a retry
#: would re-run all four and double-book the customer.
_RETRY_WORDS = {"retry", "try again", "run it again", "re-run", "rerun", "resend"}


def _is_new_job(state: OpsAgentState, has_image: bool, text: str) -> bool:
    """
    Whether this turn starts a fresh booking rather than continuing one.

    A completed booking leaves its customer in `intake` and its successes in
    `ledger`. Without this check the next job in the same conversation inherits
    both: the summary shows the previous customer's details, and every action
    node sees "already succeeded" and skips — so the agent reports a booking it
    never made.

    A retry is the one case that must NOT reset, since reusing the ledger is
    exactly what makes it re-run only the failed steps.
    """
    if text.strip().lower().strip(".!") in _RETRY_WORDS:
        return False

    # A screenshot is unambiguous. Nobody pastes an image to continue a booking
    # that is already underway — that arrives as a resume, not through here.
    if has_image:
        return True

    return booking_has_run(state.get("ledger"))


def _fresh_start() -> dict:
    """State reset for a new booking. Everything a previous job could leak."""
    return {
        "ledger": new_ledger(),
        "missing_fields": [],
        "approved": False,
        "duplicate_warning": None,
        "job_fingerprint": "",
        # NOT field_confidence: every return below sets it explicitly, and
        # since `reset` is splatted last it would overwrite the real value.
    }


def extract_screenshot(state: OpsAgentState) -> dict:
    messages = state.get("messages", [])
    if not messages:
        return {"intake": {}, "field_confidence": {}}

    last = messages[-1]
    images = _image_parts(last)
    accompanying = _text_of(last)

    # Decided once, before any extraction: is this a new job or a continuation?
    # `carried` is what survives from the previous turn — nothing, if new.
    if _is_new_job(state, bool(images), accompanying):
        carried: dict = {}
        reset = _fresh_start()
        if state.get("intake"):
            logger.info("New job — discarding the previous booking's intake.")
    else:
        carried = dict(state.get("intake") or {})
        reset = {}

    # No image, but the message may still describe the job in prose — staff
    # often type the details instead of pasting a screenshot. Extracting from
    # that text is the same problem with the same schema, so it uses the same
    # prompt; only the evidence differs.
    if not images:
        if not accompanying.strip():
            return {"intake": carried, "field_confidence": {}, **reset}

        progress.working("Reading the job details\u2026")
        logger.info("Intake with no image — extracting from message text.")
        model = get_model("extract_screenshot").with_structured_output(ScreenshotExtraction)
        try:
            extraction = model.invoke([
                SystemMessage(content=_system_prompt()),
                HumanMessage(content=(
                    "There is no screenshot. Extract the job details from this message "
                    "written by a staff member. Treat it exactly as you would text read "
                    "out of an image — report only what is stated, leave the rest blank.\n\n"
                    f"{accompanying}"
                )),
            ])
        except Exception:
            logger.exception("Text extraction failed")
            return {"intake": carried, "field_confidence": {}, **reset}

        intake, confidence = _to_intake(extraction)
        merged = {**intake, **carried}
        progress.done(f"Picked up {len(intake)} details")
        logger.info("Extracted %d usable fields from text", len(intake))
        return {"intake": merged, "field_confidence": confidence, **reset}

    model = get_model("extract_screenshot").with_structured_output(ScreenshotExtraction)

    # Character-accurate text, given to the model as evidence alongside the
    # image. Two independent reads of the same pixels: one that understands
    # layout, one that gets the characters right.
    progress.working("Reading the screenshot\u2026")
    ocr_text = _ocr_text(images)

    instructions = "Extract the moving-job details from this screenshot."
    if ocr_text:
        instructions += (
            "\n\nBelow is the EXACT text an OCR engine read from this same image. "
            "It has no idea what any of it means, so you decide which string "
            "belongs in which field — but where it contains a string you are also "
            "reading, TRUST ITS SPELLING OVER YOURS. It transcribes characters; "
            "you reconstruct them, which is why unusual names and email addresses "
            "come out subtly wrong.\n\n"
            f"--- OCR TEXT ---\n{ocr_text}\n--- END OCR TEXT ---"
        )
    if accompanying:
        instructions += (
            f"\n\nThe staff member also wrote: {accompanying!r} — treat this as "
            "additional evidence, and prefer it over the image where they conflict."
        )

    content: list[dict] = [{"type": "text", "text": instructions}, *images]

    try:
        extraction = model.invoke([SystemMessage(content=_system_prompt()), HumanMessage(content=content)])
    except Exception:
        logger.exception("Screenshot extraction failed")
        # Not fatal: fall through with nothing and let the checklist ask.
        return {"intake": dict(state.get("intake") or {}), "field_confidence": {}}

    intake, confidence = _to_intake(extraction)

    # Deterministic backstop. The prompt above asks the model to defer to OCR;
    # this enforces it for the two fields where a single wrong character is
    # invisible to every other check in the system.
    for note in _apply_ocr_corrections(intake, ocr_text):
        progress.warn("Fixed a misread", note)
    progress.done(f"Read {len(intake)} details from the screenshot")

    # Anything already confirmed by the user outranks a fresh extraction.
    merged = {**intake, **carried}

    logger.info(
        "Extracted %d usable fields (below threshold: %s)%s",
        len(intake),
        [k for k, v in confidence.items() if 0 < v < MIN_CONFIDENCE] or "none",
        "" if ocr_text else " [no OCR]",
    )
    return {"intake": merged, "field_confidence": confidence, **reset}
