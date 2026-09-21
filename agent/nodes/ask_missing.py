"""
NODE: Ask missing
PURPOSE: Ask the user for everything the checklist still needs, in one message,
         then parse their free-text reply back into fields.
INPUT:   state.intake, state.missing_fields
OUTPUT:  {"intake": {...updated...}}

⚠️  CONTAINS interrupt(). When the graph resumes, THIS NODE RE-RUNS FROM THE
TOP. Every line above the interrupt() executes again on every resume. So this
node performs no side effects whatsoever — it only formats a question, waits,
and parses the answer. All real API calls live in the act_* nodes, downstream
of the confirm gate.

── Two entrances ──────────────────────────────────────────────────────────────

Normally the checklist is short of something and this node asks for it. But
`confirm` also routes here when the user replies to the summary with an edit
("make it 4 movers"). In that case the text is ALREADY in hand, so the node
parses it and returns without pausing. Interrupting there would ask the user to
retype what they just typed.

── Not every reply is an answer ───────────────────────────────────────────────

While the graph is paused the router is bypassed entirely — the channel sends
every message straight here as Command(resume=...). So "how many jobs did we
have last month?" lands in this node as an answer to "what's their phone
number?". It parses to nothing, which is correct, but it must not be treated
like a failed attempt: it is not evidence that the question is unanswerable.
See _record_round().
"""

import logging
from datetime import datetime
from typing import Literal

from langchain_core.messages import SystemMessage
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from agent.models import get_model
from agent.state import OpsAgentState
from schemas import checklist as cl
from services.calendar import LA_TZ

logger = logging.getLogger(__name__)


#: Said back when the dispatcher writes something that isn't an answer.
#:
#: Without this the node silently re-asks the identical question, which reads
#: as the bot ignoring them — the single most common complaint about a stuck
#: form. Saying "later" costs one line and keeps the booking obviously alive.
ASIDE = {
    "off_topic": "I'll come back to that once this booking is done.",
    "later": "No rush — this booking keeps until you're ready. Nothing is lost.",
}

#: Intake keys that are bookkeeping, not booking data. They must never be shown
#: back to the user: the underscore ones render as "Ask Rounds: 2" and
#: "Pending Edit: make it 4 movers" in the middle of a question.
HIDDEN_FIELDS = {"address_status", "notes_asked", "field_confidence", "duplicate_warning"}


class ParsedReply(BaseModel):
    """Fields recovered from a free-text answer. Everything optional."""

    full_name: str | None = None
    email: str | None = None
    phone: str | None = None
    pickup_address: str | None = Field(
        default=None,
        description="The address as written, or the literal string 'TBD' if "
                    "they say they don't have it yet.",
    )
    dropoff_address: str | None = Field(
        default=None,
        description="The address as written, or the literal string 'TBD' if "
                    "they say they don't have it yet.",
    )
    extra_stop: str | None = Field(
        default=None, description="Only if the user volunteers a third address."
    )
    move_date: str | None = Field(
        default=None, description="Resolved to mm/dd/yyyy. Never leave it relative."
    )
    arrival_time: str | None = Field(
        default=None, description="Compact window: '8-9am', '2-4pm', '11am-1pm'."
    )
    movers: str | None = Field(default=None, description="Digits only: '2', '3', '4'.")
    is_labor: bool | None = Field(
        default=None,
        description="True for labor-only, False for a full move, null if not addressed.",
    )
    job_notes: str | None = Field(
        default=None,
        description=(
            "Notes text, one fact per line, each line starting with '- '. "
            "If the user says 'none'/'no'/'nothing', return the empty string — "
            "that is an answer, distinct from not addressing it at all (null)."
        ),
    )
    unclear: list[str] = Field(
        default_factory=list,
        description="Fields the user seemed to address but ambiguously.",
    )
    reply_kind: Literal["answer", "off_topic", "later"] = Field(
        default="answer",
        description=(
            "What the dispatcher is doing, ignoring any fields above. "
            "'answer' — engaging with the questions asked. "
            "'off_topic' — asking about or raising something unrelated to this "
            "booking, e.g. 'how many jobs last month?' or 'is Adilet in today?'. "
            "'later' — saying they will come back to it, e.g. 'I'm in a meeting, "
            "one sec'. When in doubt use 'answer'."
        ),
    )


def _current_values(intake: dict) -> str:
    """The booking as it stands, for the parser to edit against."""
    lines = []
    for name, value in intake.items():
        if value in (None, "", {}) or name.startswith("_") or name in HIDDEN_FIELDS:
            continue
        label = cl.BY_NAME[name].label if name in cl.BY_NAME else name
        if name == "is_labor":
            value = "labor only" if value else "full move"
        shown = str(value).replace("\n", "\n" + " " * 18)
        lines.append(f"  {label + ':':<16}{shown}")
    return "\n".join(lines) if lines else "  (nothing yet)"


def _parse_prompt(outstanding: list[str], editing: bool, current: dict) -> str:
    now = datetime.now(LA_TZ)

    if editing:
        context = """They are looking at a finished booking summary and have \
just told you what to CHANGE about it. Extract only the fields they are \
actually changing — everything they did not mention stays as it is, so a null \
is how you leave a field alone."""
    else:
        context = f"""They were asked:
{chr(10).join(f'  - {q}' for q in outstanding)}"""

    return f"""You are parsing a dispatcher's reply into structured booking fields \
for a Los Angeles moving company.

Today is {now:%A, %B %-d, %Y} ({now:%Y-%m-%d}), America/Los_Angeles.

# What is on file right now
{_current_values(current)}

{context}

Extract only what they actually answered. Leave everything else null — a null is \
harmless, an invented value books the wrong job.

# Edits are instructions, not values
A dispatcher often describes a CHANGE to something above rather than stating \
the new value outright: "remove the gas fee line", "drop the second note", \
"add that it's a walk-up", "pickup is in Arcadia", "change the date to the \
14th". Every field you return must be the COMPLETE new value after applying \
what they said — never the instruction itself, and never a fragment.

Work from the values ABOVE, not from these illustrations:

  on file   Job notes:  - Piano, ground floor
                        - Dog in the yard
                        - Narrow driveway, small truck
                        - Customer prefers text
  they say  "drop the dog line"
  return    job_notes = "- Piano, ground floor\\n- Narrow driveway, small truck\\n- Customer prefers text"

  on file   Pickup address:  8 Elm Ct #4
  they say  "pickup is in Whittier"
  return    pickup_address = "8 Elm Ct #4, Whittier"

  they say  "remove the piano line"   but no such line is on file
  return    job_notes = null   and list it in `unclear`

For notes specifically: a removal touches ONLY the line named. Every other \
line comes back exactly as it is on file — same words, same order. Count the \
lines before and after; a removal of one line leaves one fewer, never two.

Rules:
- **Dates**: resolve relative references against today and output mm/dd/yyyy. \
"next Friday" -> the Friday of next week. "the 14th" -> the next 14th that is \
in the future. If a date is genuinely ambiguous, leave it null and list it in \
`unclear`.
- **Arrival windows**: normalise to compact form. "eight to nine in the morning" \
-> "8-9am". "2 to 4" in an afternoon context -> "2-4pm".
- **Crew size**: digits only. "3 guys", "three movers", "3" all -> "3".
- **Labor**: "labor only", "just loading help", "no truck" -> is_labor true. \
"full move", "moving them from X to Y" -> false. Not mentioned -> null.
- **Notes**: "none", "no", "nothing", "n/a" -> empty string, NOT null. The \
difference matters: empty means asked and answered, null means never addressed.
- **Addresses**: copy exactly as written, even if incomplete. Do NOT add a city, \
state or ZIP. A separate step completes them; a guess here corrupts it.
- **An address they don't have yet**: "drop-off later", "customer will send \
it", "TBD", "don't have the address yet", "she'll text it" -> the literal \
string "TBD" for that address. This is a real answer — the dispatcher has \
decided the job can be booked without it. Do NOT put a city or a ZIP in the \
address field as a stand-in; "Yorba Linda" is not an address.
- **reply_kind**: judge the message as a whole. A reply can both answer a \
question AND wander off — if they answered anything at all, it is an "answer".

One reply often answers several questions at once — e.g. "next Friday 8-9am, \
3 guys, no notes" answers four."""


def _parse(reply: str, questions: list[str], *, editing: bool,
           current: dict | None = None) -> ParsedReply | None:
    """
    Model call. Returns None when it fails, so the caller can re-ask.

    `current` is what makes edits work. Without it the model can extract a
    value the dispatcher states ("8-9am") but has nothing to apply "remove the
    gas fee line" to, so it returns null and the edit silently vanishes.
    """
    model = get_model("parse_reply").with_structured_output(ParsedReply)
    try:
        return model.invoke(
            [SystemMessage(content=_parse_prompt(questions, editing, current or {})),
             ("human", reply)]
        )
    except Exception:
        logger.exception("Could not parse reply %r", reply[:200])
        return None


def _apply(intake: dict, parsed: ParsedReply) -> dict:
    """
    Write parsed values into the intake, normalised. Mutates `intake` and
    returns just the fields that landed, which is what tells the caller whether
    this reply was a real answer.
    """
    updates = parsed.model_dump(exclude={"unclear", "reply_kind"}, exclude_none=True)

    # An explicit "no notes" must be recorded, otherwise the checklist asks again
    # forever. This is the only field where empty string is a real answer.
    if "job_notes" in updates:
        intake["notes_asked"] = True

    for name, value in updates.items():
        if spec := cl.BY_NAME.get(name):
            if spec.normalizer and isinstance(value, str) and value:
                value = spec.normalizer(value)
        intake[name] = value

    if parsed.unclear:
        logger.info("Ambiguous in reply: %s", parsed.unclear)

    return updates


def _record_round(intake: dict, updates: dict, kind: str) -> None:
    """
    Decide whether this exchange counted as an attempt at the question.

    Two counters, because they guard different things:

      _ask_rounds  attempts to ANSWER. Its limit is what stops the bot asking
                   for a phone number nine times when a validator can never be
                   satisfied.
      _ask_turns   every exchange, answer or not. A pure backstop: if the model
                   ever misclassified every reply as off-topic, _ask_rounds
                   would never move and give_up could never fire.

    Note who decides. An extracted field is hard evidence and overrules the
    model's own classification — same rule as everywhere else here: the model
    proposes, Python disposes. `kind` only gets a say when nothing was
    extracted, where there is nothing else to go on.
    """
    intake["_ask_turns"] = int(intake.get("_ask_turns", 0)) + 1

    if not updates and kind in ASIDE:
        intake["_aside"] = ASIDE[kind]
        logger.info("Reply was %r — acknowledging, not counting a round.", kind)
        return

    intake["_ask_rounds"] = int(intake.get("_ask_rounds", 0)) + 1


def _acknowledge_tbd(intake: dict, updates: dict) -> None:
    """
    Say out loud that "later" was heard. Otherwise the next message just shows
    "Drop-off address: TBD" in the list and the dispatcher is left wondering
    whether that was understood or is about to be asked again.
    """
    deferred = [
        cl.BY_NAME[name].label.lower()
        for name in ("pickup_address", "dropoff_address")
        if name in updates and cl.is_tbd(updates[name])
    ]
    if deferred:
        intake["_aside"] = (
            f"Got it — {' and '.join(deferred)} marked as to be confirmed. "
            "Remember to add it before the move."
        )


def ask_missing(state: OpsAgentState) -> dict:
    intake = dict(state.get("intake") or {})

    # ── Entrance 2: an edit handed over by the confirm gate ────────────────────
    # The text is already here, so parse it and hand back. No interrupt: the
    # user typed this a second ago and must not be asked to repeat it.
    if edit := intake.pop("_pending_edit", None):
        logger.info("Applying edit from the confirm gate: %r", str(edit)[:120])
        if parsed := _parse(str(edit), [], editing=True, current=intake):
            _apply(intake, parsed)
        return {"intake": intake}

    # ── Entrance 1: the checklist is short of something ───────────────────────
    result = cl.evaluate(intake)
    if result.is_complete:
        return {}

    questions = result.all_questions()
    known = {
        k: v for k, v in intake.items()
        if v not in (None, "", {}) and not k.startswith("_") and k not in HIDDEN_FIELDS
    }

    # ── Everything above this line re-runs on every resume. Keep it pure. ──
    reply = interrupt(
        {
            "type": "missing_fields",
            "message": _format_question(known, questions, intake.get("_aside")),
            "questions": questions,
            "known": known,
        }
    )
    # ── Everything below runs once per resume, with the user's answer. ──

    intake.pop("_aside", None)  # shown once; re-set below if still off topic

    if not isinstance(reply, str) or not reply.strip():
        _record_round(intake, {}, "answer")
        return {"intake": intake}

    parsed = _parse(reply, questions, editing=False, current=intake)
    if parsed is None:
        _record_round(intake, {}, "answer")
        return {"intake": intake}

    updates = _apply(intake, parsed)
    _record_round(intake, updates, parsed.reply_kind)
    _acknowledge_tbd(intake, updates)
    return {"intake": intake}


def _format_question(known: dict, questions: list[str], aside: str | None = None) -> str:
    """
    Show what's already captured before asking. Context makes the answer better
    and lets the user correct a misread field in the same breath.
    """
    lines = []
    if aside:
        lines += [aside, ""]

    if known:
        lines.append("Here's what I have so far:")
        for name, value in known.items():
            label = cl.BY_NAME[name].label if name in cl.BY_NAME else name.replace("_", " ").title()
            if name == "is_labor":
                value = "labor only" if value else "full move"
            lines.append(f"  {label}: {value}")
        lines.append("")

    lines.append("I still need:" if len(questions) > 1 else "One thing:")
    lines.extend(f"  • {q}" for q in questions)
    lines.append("")
    lines.append("Answer however you like — one message is fine.")
    return "\n".join(lines)
