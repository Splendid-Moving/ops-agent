"""
NODE: Site traffic
PURPOSE: Answer questions about splendidmoving.com traffic from Vercel.
INPUT:   state.messages
OUTPUT:  {"messages": [AIMessage]}

Read-only. Two tools, one per shape rather than one per question: a breakdown
tool with a `by` parameter, and a comparison tool that exists only because its
arithmetic must happen in Python.
"""

import logging
from datetime import date, datetime, timedelta

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.config import get_stream_writer

from agent import progress
from agent.models import get_model
from agent.state import OpsAgentState
from services import vercel
from services.calendar import LA_TZ

logger = logging.getLogger(__name__)

SITE = "splendidmoving.com"

#: Friendly names for the tool's `by` values, and the Vercel dimension each maps to.
BREAKDOWNS = {"total": None, **{k: v for k, v in vercel.DIMENSIONS.items()}}

#: Blank dimension values are real answers, not gaps. Most of the site's
#: traffic arrives with no referrer at all.
BLANK_LABELS = {
    "referrerHostname": "direct (no referrer)",
    "deviceType": "unknown device",
    "country": "unknown country",
    "browserName": "unknown browser",
}


def _parse_day(value: str, field: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        raise ValueError(f"{field} must be YYYY-MM-DD, got {value!r}") from None


def _too_old(start: date) -> bool:
    """Whether a breakdown starting here reaches past the plan's window."""
    return (date.today() - start).days > vercel.AGGREGATE_WINDOW_DAYS


def _window_refusal(start: date) -> str:
    return (
        f"I can't break traffic down that far back. The Hobby plan only keeps "
        f"{vercel.AGGREGATE_WINDOW_DAYS} days of day-by-day detail, and "
        f"{start.isoformat()} is {(date.today() - start).days} days ago.\n"
        "Plain totals have no such limit — ask for the total over that period "
        "instead, or pick a range inside the last "
        f"{vercel.AGGREGATE_WINDOW_DAYS} days."
    )


def _label(row: dict, dimension: str) -> str:
    if dimension == "day":
        return row.get("timestamp", "")[:10]
    value = (row.get(dimension) or "").strip()
    return value or BLANK_LABELS.get(dimension, "(not recorded)")


def _all_time(note: str) -> str:
    """
    Lifetime totals. Only reachable by sending NO date range — with explicit
    dates the count endpoint is capped at 31 days like everything else.
    """
    progress.working("Reading all-time site traffic…")
    try:
        totals = vercel.count_visits()
    except vercel.VercelError as exc:
        return (f"I couldn't reach Vercel just now — {exc}. "
                "This is a lookup failure, not a reading of zero traffic.")
    progress.done(f"{totals['visitors']} visitors all time")
    prefix = f"{note}. " if note else ""
    return (f"{prefix}{SITE}, all time: {totals['pageviews']:,} pageviews from "
            f"{totals['visitors']:,} visitors.")


@tool
def site_traffic(start_date: str | None = None, end_date: str | None = None,
                 by: str = "total") -> str:
    """Website traffic for splendidmoving.com.

    Omit BOTH dates to get the all-time total — that is the only way Vercel
    returns lifetime numbers on this plan. With dates, everything is capped at
    the last 31 days.

    Args:
        start_date: First day, YYYY-MM-DD (inclusive). Omit for all time.
        end_date: Last day, YYYY-MM-DD (inclusive). Omit for all time.
        by: total | day | page | referrer | country | device | browser.
    """
    if by not in BREAKDOWNS:
        return (f"{by!r} is not a breakdown I have. Use one of: "
                f"{', '.join(BREAKDOWNS)}.")
    dimension = BREAKDOWNS[by]

    if not start_date and not end_date:
        if dimension is not None:
            return (f"A {by} breakdown needs a date range — only plain totals "
                    "work without one. Give me a start and end date inside the "
                    f"last {vercel.AGGREGATE_WINDOW_DAYS} days.")
        return _all_time(note="")

    if not start_date or not end_date:
        return "Give me both a start date and an end date, or neither."

    start, end = _parse_day(start_date, "start_date"), _parse_day(end_date, "end_date")
    if end < start:
        return f"{end_date} is before {start_date}."

    if _too_old(start):
        if dimension is not None:
            return _window_refusal(start)
        # A total reaching past the window. The all-time figure is the closest
        # honest answer, but it answers a DIFFERENT question, so say so.
        return _all_time(note=(
            f"{start_date} is past the {vercel.AGGREGATE_WINDOW_DAYS} days Vercel "
            "keeps for this plan, so I can't give you exactly that range. "
            "Here is the all-time total instead"))

    progress.working(f"Reading site traffic, {start_date} to {end_date}…")
    try:
        if dimension is None:
            totals = vercel.count_visits(since=start_date, until=end_date)
            progress.done(f"{totals['visitors']} visitors")
            return (f"{SITE}, {start_date} to {end_date}: "
                    f"{totals['pageviews']:,} pageviews from "
                    f"{totals['visitors']:,} visitors.")

        rows = vercel.aggregate_visits(since=start_date, until=end_date, by=dimension, limit=10)
    except vercel.VercelError as exc:
        logger.warning("Vercel lookup failed: %s", exc)
        return (f"I couldn't reach Vercel just now — {exc}. "
                "This is a lookup failure, not a reading of zero traffic.")

    if not rows:
        progress.done("No visits recorded")
        return f"No visits recorded on {SITE} between {start_date} and {end_date}."

    progress.done(f"Read {len(rows)} rows")
    total = sum(r.get("pageviews", 0) for r in rows)
    header = f"{SITE}, {start_date} to {end_date} — by {by} ({total:,} pageviews total)"
    if by == "day":
        header += ". Days are UTC, which is how Vercel buckets them."

    lines = [header, ""]
    for row in rows:
        lines.append(f"  {_label(row, dimension):<28} {row.get('pageviews', 0):>7,} views"
                     f"   {row.get('visitors', 0):>7,} visitors")
    return "\n".join(lines)


@tool
def compare_traffic(period_a_start: str, period_a_end: str,
                    period_b_start: str, period_b_end: str) -> str:
    """Compare traffic between two periods. Period A is the recent one.

    The change is computed here, not by you. Report the figures as returned.

    Args:
        period_a_start: First day of the recent period, YYYY-MM-DD.
        period_a_end: Last day of the recent period, YYYY-MM-DD.
        period_b_start: First day of the earlier period, YYYY-MM-DD.
        period_b_end: Last day of the earlier period, YYYY-MM-DD.
    """
    progress.working("Comparing two periods…")
    try:
        a = vercel.count_visits(since=period_a_start, until=period_a_end)
        b = vercel.count_visits(since=period_b_start, until=period_b_end)
    except vercel.VercelError as exc:
        return (f"I couldn't reach Vercel just now — {exc}. "
                "This is a lookup failure, not a reading of zero traffic.")

    def change(now: int, before: int) -> str:
        delta = now - before
        if before == 0:
            return f"{delta:+,} (no comparison — the earlier period was n/a, nothing recorded)"
        return f"{delta:+,} ({delta / before * 100:+.1f}%)"

    progress.done("Compared")
    return "\n".join([
        f"{SITE} traffic, two periods:",
        f"  {period_a_start} to {period_a_end}: "
        f"{a['pageviews']:,} pageviews, {a['visitors']:,} visitors",
        f"  {period_b_start} to {period_b_end}: "
        f"{b['pageviews']:,} pageviews, {b['visitors']:,} visitors",
        "",
        f"  pageviews {change(a['pageviews'], b['pageviews'])}",
        f"  visitors  {change(a['visitors'], b['visitors'])}",
    ])


TOOLS = [site_traffic, compare_traffic]


def _system_prompt() -> str:
    now = datetime.now(LA_TZ)
    return f"""You answer questions about traffic to {SITE}, the company's \
website, by querying Vercel Web Analytics.

Today is {now:%A, %B %-d, %Y} ({now:%Y-%m-%d}) in Los Angeles. Resolve relative \
dates against that — never against UTC, which is already tomorrow here after \
5pm — then call a tool with explicit YYYY-MM-DD bounds.

Tool choice:
- "how many visitors/pageviews" over a range -> site_traffic with by="total"
- "ever", "all time", "since launch", or any range older than a month -> \
site_traffic with by="total" and NO dates at all. Sending dates caps it at 31 \
days; omitting them is the only way to get lifetime numbers.
- which pages, where visitors come from, countries, devices -> site_traffic \
with the matching `by`
- "up or down", "compared to", "versus last week" -> compare_traffic

Rules:
- NEVER invent a number. Every figure comes from a tool result.
- Never compute a difference or a percentage yourself. compare_traffic returns \
them already worked out; report what it says.
- You have one turn. Nothing runs after you stop typing, so there is no "let \
me check and get back to you". Call the tool now.
- Challenged? Call the tool again and report what it says. Do not retract a \
figure you have not re-checked.
- Day-by-day detail only goes back {vercel.AGGREGATE_WINDOW_DAYS} days on this \
plan. If the tool says so, pass that on plainly — do not substitute zeros or \
guess at the shape of older traffic.
- Most visits arrive with no referrer; that is direct traffic, not missing \
data. Say "direct" rather than implying something failed.

This is the website, NOT the job calendar. Questions about jobs, moves, crews \
or customers are somebody else's lane.

Be brief and concrete, like a colleague reading off a dashboard."""


def site_traffic_agent(state: OpsAgentState) -> dict:
    agent = create_agent(model=get_model("site_traffic"), tools=TOOLS)
    agent_input = {"messages": [SystemMessage(content=_system_prompt()), *state["messages"]]}

    try:
        try:
            writer = get_stream_writer()
        except RuntimeError:
            writer = None

        final: dict = {}
        for mode, chunk in agent.stream(agent_input, stream_mode=["custom", "values"]):
            if mode == "custom" and writer is not None:
                writer(chunk)
            elif mode == "values":
                final = chunk

        messages = final.get("messages") or []
        if not messages:
            raise RuntimeError("site traffic agent produced no messages")
        return {"messages": [messages[-1]]}
    except Exception as exc:
        logger.exception("Site traffic lane failed")
        return {"messages": [AIMessage(content=(
            f"I couldn't read the site analytics just now — {type(exc).__name__}. "
            "Worth retrying; if it keeps happening the Vercel token may need a look."
        ))]}
