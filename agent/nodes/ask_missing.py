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
"""

import logging
from datetime import datetime

from langchain_core.messages import SystemMessage
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from agent.models import get_model
from agent.state import OpsAgentState
from schemas import checklist as cl
from services.calendar import LA_TZ

logger = logging.getLogger(__name__)


class ParsedReply(BaseModel):
    """Fields recovered from a free-text answer. Everything optional."""

    full_name: str | None = None
    email: str | None = None
    phone: str | None = None
    pickup_address: str | None = None
    dropoff_address: str | None = None
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
            "Notes text. If the user says 'none'/'no'/'nothing', return the empty "
            "string — that is an answer, distinct from not addressing it at all (null)."
        ),
    )
    unclear: list[str] = Field(
        default_factory=list,
        description="Fields the user seemed to address but ambiguously.",
    )


def _parse_prompt(outstanding: list[str]) -> str:
    now = datetime.now(LA_TZ)
    return f"""You are parsing a dispatcher's reply into structured booking fields \
for a Los Angeles moving company.

Today is {now:%A, %B %-d, %Y} ({now:%Y-%m-%d}), America/Los_Angeles.

They were asked:
{chr(10).join(f'  - {q}' for q in outstanding)}

Extract only what they actually answered. Leave everything else null — a null is \
harmless, an invented value books the wrong job.

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

One reply often answers several questions at once — e.g. "next Friday 8-9am, \
3 guys, no notes" answers four."""


def ask_missing(state: OpsAgentState) -> dict:
    intake = dict(state.get("intake") or {})
    result = cl.evaluate(intake)

    if result.is_complete:
        return {}

    questions = result.all_questions()

    known = {
        k: v for k, v in intake.items()
        if v not in (None, "", {}) and k not in ("address_status", "notes_asked")
    }

    # ── Everything above this line re-runs on every resume. Keep it pure. ──
    reply = interrupt(
        {
            "type": "missing_fields",
            "message": _format_question(known, questions),
            "questions": questions,
            "known": known,
        }
    )
    # ── Everything below runs once per resume, with the user's answer. ──

    # Count every completed round. Without this the loop guard in
    # validate_checklist can never fire, and a field whose validator can never
    # be satisfied would re-ask forever.
    intake["_ask_rounds"] = int(intake.get("_ask_rounds", 0)) + 1

    if not isinstance(reply, str) or not reply.strip():
        return {"intake": intake}

    model = get_model("parse_reply").with_structured_output(ParsedReply)
    try:
        parsed = model.invoke(
            [SystemMessage(content=_parse_prompt(questions)), ("human", reply)]
        )
    except Exception:
        logger.exception("Could not parse reply %r", reply[:200])
        return {"intake": intake}

    updates = parsed.model_dump(exclude={"unclear"}, exclude_none=True)

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

    return {"intake": intake}


def _format_question(known: dict, questions: list[str]) -> str:
    """
    Show what's already captured before asking. Context makes the answer better
    and lets the user correct a misread field in the same breath.
    """
    lines = []
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
