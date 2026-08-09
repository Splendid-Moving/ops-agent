"""
NODE: Analytics
PURPOSE: Answer questions about jobs that already exist, by reading and
         aggregating Google Calendar events.
INPUT:   state.messages
OUTPUT:  {"messages": [AIMessage]}

Read-only. Nothing in this lane writes anywhere.

Aggregation happens in Python, not in the model. The model chooses a date range
and calls a tool; the counting is done by code. Asking a model to tally 150
events is slow, expensive, and wrong often enough to matter — and "how many jobs
last month" is a number that gets used.
"""

import logging
from collections import Counter
from datetime import datetime, timedelta

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.config import get_stream_writer

from agent import progress
from agent.models import get_model
from agent.state import OpsAgentState
from schemas import business_context
from services import calendar, formatting

logger = logging.getLogger(__name__)

#: Hard ceiling on a single query, to stop "how many jobs ever" from pulling
#: years of events into context.
MAX_RANGE_DAYS = 400


def _parse_range(start_date: str, end_date: str) -> tuple[datetime, datetime]:
    """Parse YYYY-MM-DD bounds into inclusive LA-time datetimes."""
    start = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=calendar.LA_TZ)
    end = datetime.strptime(end_date, "%Y-%m-%d").replace(
        hour=23, minute=59, second=59, tzinfo=calendar.LA_TZ
    )
    if end < start:
        raise ValueError(f"end_date {end_date} is before start_date {start_date}")
    if (end - start).days > MAX_RANGE_DAYS:
        raise ValueError(
            f"Range is {(end - start).days} days; the maximum is {MAX_RANGE_DAYS}. "
            "Ask about a shorter period."
        )
    return start, end


def _arrival_window(job: dict) -> str:
    """
    Render an event's span as the arrival window it actually represents,
    e.g. "8-9am".

    Formatting this as a window rather than a start time is the fix for a real
    misreading: given "8:00 AM" the model reported "the job starts at 8:00 AM".
    The prompt says otherwise too, but data that cannot be misread beats an
    instruction not to misread it.
    """
    start, end = job.get("start_time", ""), job.get("end_time", "")
    if not start or "T" not in start:
        return "time not set"

    def label(iso: str, keep_period: bool) -> str:
        dt = datetime.fromisoformat(iso)
        text = dt.strftime("%-I:%M") if dt.minute else dt.strftime("%-I")
        return text + (dt.strftime("%p").lower() if keep_period else "")

    if not end or "T" not in end:
        return label(start, True)

    same_half = datetime.fromisoformat(start).strftime("%p") == datetime.fromisoformat(end).strftime("%p")
    return f"{label(start, not same_half)}-{label(end, True)}"


@tool
def count_jobs(start_date: str, end_date: str) -> str:
    """Count jobs in a date range, with a breakdown by lead source and crew size.

    Use this for "how many" questions. Dates are YYYY-MM-DD and both ends are
    inclusive.

    Args:
        start_date: First day of the range, YYYY-MM-DD.
        end_date: Last day of the range, YYYY-MM-DD.
    """
    progress.working(f"Checking the calendar, {start_date} to {end_date}\u2026")
    start, end = _parse_range(start_date, end_date)
    jobs = calendar.list_jobs(start, end)

    if not jobs:
        progress.done("No jobs in that range")
        return f"No jobs between {start_date} and {end_date}."

    progress.done(f"Found {len(jobs)} jobs")
    labor = sum(1 for j in jobs if j["is_labor"])
    sources = Counter(formatting.normalize_source(j["source"]) for j in jobs)
    crews = Counter((j["movers"] or "unspecified").strip() for j in jobs)

    lines = [
        f"{len(jobs)} jobs between {start_date} and {end_date}.",
        f"  full moves: {len(jobs) - labor}",
        f"  labor only: {labor}",
        "  by source: " + ", ".join(f"{s}={n}" for s, n in sources.most_common()),
        "  by crew size: " + ", ".join(f"{c} movers={n}" for c, n in sorted(crews.items())),
    ]

    # Source is blank on a large share of events. Say so explicitly, otherwise
    # the model reports "15 Yelp jobs" as though the rest were attributed.
    unspecified = sources.get("unspecified", 0)
    if unspecified:
        pct = round(100 * unspecified / len(jobs))
        lines.append(
            f"  NOTE: {unspecified} of {len(jobs)} jobs ({pct}%) have no lead source "
            "recorded, so the source breakdown covers only the rest. Say this when "
            "reporting source numbers."
        )
    return "\n".join(lines)


@tool
def list_jobs(start_date: str, end_date: str) -> str:
    """List individual jobs in a date range with customer, time, addresses and crew.

    Use this for "what's on the calendar" or "who are we moving" questions,
    not for counting. Dates are YYYY-MM-DD and both ends are inclusive.

    Args:
        start_date: First day of the range, YYYY-MM-DD.
        end_date: Last day of the range, YYYY-MM-DD.
    """
    progress.working("Reading the schedule\u2026")
    start, end = _parse_range(start_date, end_date)
    jobs = calendar.list_jobs(start, end)

    if not jobs:
        progress.done("Nothing scheduled in that range")
        return f"No jobs between {start_date} and {end_date}."

    progress.done(f"Read {len(jobs)} jobs")
    # Keep the payload bounded — a month can hold 150+ jobs.
    LIMIT = 40
    lines = [
        "Times below are ARRIVAL WINDOWS, not start times or durations.",
        "",
    ]
    for job in jobs[:LIMIT]:
        labor = " (labor)" if job["is_labor"] else ""
        # Calendar position, not the typed Date: line — see busiest_days.
        mismatch = f" [description says {job['move_date']}]" if job.get("date_mismatch") else ""
        lines.append(
            f"{job['calendar_date']} arrives {_arrival_window(job)} — "
            f"{job['customer']}{labor} | {job['movers'] or '?'} movers | "
            f"{job['from_address'] or '?'} -> {job['to_address'] or '?'}{mismatch}"
        )

    if len(jobs) > LIMIT:
        lines.append(f"... and {len(jobs) - LIMIT} more (showing the first {LIMIT} of {len(jobs)})")
    return "\n".join(lines)


@tool
def busiest_days(start_date: str, end_date: str, top_n: int = 5) -> str:
    """Find the days with the most jobs in a range. Use for capacity questions.

    Args:
        start_date: First day of the range, YYYY-MM-DD.
        end_date: Last day of the range, YYYY-MM-DD.
        top_n: How many days to return.
    """
    progress.working("Working out the busiest days\u2026")
    start, end = _parse_range(start_date, end_date)
    jobs = calendar.list_jobs(start, end)
    if not jobs:
        return f"No jobs between {start_date} and {end_date}."

    # Group by where the event SITS, not by the Date: line typed into its
    # description. The two normally agree, but a real July event sat on Jul 2
    # with a description reading 07/03/2026 — counting by the typed date
    # reported 11 jobs on a day that had 10. The crew works from the calendar.
    per_day = Counter(j["calendar_date"] for j in jobs if j["calendar_date"])

    lines = [f"Busiest days between {start_date} and {end_date} (capacity is 9/day):"]
    for day, count in per_day.most_common(max(1, min(top_n, 20))):
        flag = "  <- at or over capacity" if count >= 9 else ""
        lines.append(f"  {day}: {count} jobs{flag}")

    # A typed date that disagrees with the event's position is a data-entry
    # error someone should fix, so say so rather than quietly papering over it.
    if mismatched := [j for j in jobs if j.get("date_mismatch")]:
        lines.append("")
        lines.append(
            f"NOTE: {len(mismatched)} event(s) have a 'Date:' in the description that "
            "disagrees with where the event sits on the calendar. Counted by calendar "
            "position. Worth mentioning to the user, with the customer name(s):"
        )
        for job in mismatched[:5]:
            lines.append(
                f"  {job['customer']}: on the calendar {job['calendar_date']}, "
                f"description says {job['move_date']}"
            )
    return "\n".join(lines)


TOOLS = [count_jobs, list_jobs, busiest_days]


def _system_prompt() -> str:
    """Built per-call so 'today' is always current."""
    now = datetime.now(calendar.LA_TZ)
    yesterday = now - timedelta(days=1)
    first_this_month = now.replace(day=1)
    last_month_end = first_this_month - timedelta(days=1)

    return f"""You answer questions about Splendid Moving's jobs by querying the \
Google Calendar, where every job is one event.

{business_context.for_calendar_questions()}

# Answering questions

Today is {now:%A, %B %-d, %Y} ({now:%Y-%m-%d}) in Los Angeles.
For reference: yesterday was {yesterday:%Y-%m-%d}; last month was \
{last_month_end:%B %Y}, running {last_month_end:%Y-%m}-01 to {last_month_end:%Y-%m-%d}.

Resolve relative dates yourself, then call a tool with explicit YYYY-MM-DD \
bounds. Both ends are inclusive.

Tool choice:
- "how many" -> count_jobs
- "who / what's scheduled / show me" -> list_jobs
- "busiest / capacity / how full" -> busiest_days

Rules:
- NEVER invent a number. Every figure you give must come from a tool result.
- The tools already exclude non-job events (crew meetings, blocks), so their \
counts are the real job counts. Do not adjust them.
- Report what the tool returned. If a breakdown is interesting, mention it \
briefly; do not dump the whole thing unless asked.
- If a question is ambiguous about the period, pick the most likely reading, \
state which range you used, and answer.

Be brief and concrete, like a colleague reading off a dashboard."""


def analytics(state: OpsAgentState) -> dict:
    agent = create_agent(model=get_model("analytics"), tools=TOOLS)
    agent_input = {"messages": [SystemMessage(content=_system_prompt()), *state["messages"]]}

    try:
        # STREAM the sub-agent rather than invoke it.
        #
        # LangGraph's stream writer does not cross into a nested agent's own
        # run, so progress emitted inside these tools is invisible to the
        # caller if we just .invoke(). Streaming the sub-agent and forwarding
        # its custom events keeps "Checking the calendar…" arriving live
        # instead of after the answer is already written.
        writer = get_stream_writer()
        final: dict = {}
        for mode, chunk in agent.stream(agent_input, stream_mode=["custom", "values"]):
            if mode == "custom":
                if writer is not None:
                    writer(chunk)
            else:
                final = chunk

        messages = final.get("messages") or []
        if not messages:
            raise RuntimeError("analytics agent produced no messages")
        return {"messages": [messages[-1]]}
    except Exception as exc:
        logger.exception("Analytics failed")
        return {
            "messages": [
                AIMessage(
                    content=(
                        f"I couldn't read the calendar just now — {type(exc).__name__}. "
                        "Worth retrying; if it keeps happening the calendar credentials "
                        "may need a look."
                    )
                )
            ]
        }
