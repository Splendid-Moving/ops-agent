"""
LangSmith tracing.

WHAT TRACING GIVES YOU
----------------------
A LangSmith trace is a recording of one turn of the agent: which node ran,
what the model was asked, what it answered, how long each step took, what it
cost, and — with the `@traceable` decorators added in this package — every GHL,
Calendar, Maps and Vision call the run made along the way.

That matters here because the interesting failures in this agent are invisible
in the logs. "The invoice didn't send" could be a bad extraction, a rejected
custom field, or a GHL 400 three layers down. A trace shows which, in one view.

HOW MUCH OF THIS IS AUTOMATIC
-----------------------------
Most of it. LangGraph and LangChain instrument themselves: with the env vars
set, every node and every model call already appears in LangSmith with no code
changes. This module adds the two things that are NOT automatic:

  1. `run_config()` — the per-run labels (thread id, channel, dry-run, backend)
     that make a list of hundreds of runs searchable instead of a wall of
     identical rows.
  2. `configure()` — a startup check, because a mistyped API key otherwise
     fails silently and you find out by staring at an empty project.

The `@traceable` decorators on the service layer live next to the functions
they trace, in services/ghl.py, calendar.py, maps.py, address.py and ocr.py.

⚠️  PRIVACY
-----------
When tracing is on, run contents are uploaded to LangSmith. For this agent that
means real customer names, phone numbers, email addresses, home addresses and
the screenshots of their conversations. That is exactly what makes a trace
useful for debugging and exactly why `LANGSMITH_TRACING` defaults to false. See
the note on `hide_io_note()` for the blunt-instrument opt-out.
"""

import logging
import os

from services import config

logger = logging.getLogger(__name__)


# ── Startup ────────────────────────────────────────────────────────────────────

def configure() -> str:
    """
    Check the tracing setup once at boot and say plainly what it will do.

    Returns the same one-line summary it logs, so entrypoints can print it.

    This exists because every LangSmith misconfiguration fails the same silent
    way: the app runs perfectly, and the project stays empty. The SDK uploads
    runs from a background thread and deliberately swallows its own errors, so
    a wrong API key never raises anywhere you would look. Checking at boot turns
    a half-hour of confusion into one log line.
    """
    if not config.langsmith_tracing():
        summary = "LangSmith tracing: OFF (set LANGSMITH_TRACING=true to enable)"
        logger.info(summary)
        return summary

    if not config.langsmith_api_key():
        summary = (
            "LangSmith tracing: ON but LANGSMITH_API_KEY is empty — no runs will "
            "be uploaded. The SDK fails silently, so this is your only warning."
        )
        logger.warning(summary)
        return summary

    # The SDK reads LANGSMITH_PROJECT itself, but only if it is actually set;
    # config.langsmith_project() has a default that the SDK cannot see. Writing
    # it back keeps "what config.py says" and "where runs land" the same thing.
    os.environ.setdefault("LANGSMITH_PROJECT", config.langsmith_project())

    summary = f"LangSmith tracing: ON -> project {config.langsmith_project()!r}"
    logger.info(summary)

    if not config.dry_run():
        logger.warning(
            "LangSmith tracing is ON in LIVE mode — real customer names, phone "
            "numbers, addresses and screenshots will be uploaded to LangSmith."
        )
    return summary


def status() -> dict:
    """Tracing state for /api/status. Never reports the key itself."""
    return {
        "enabled": config.langsmith_tracing(),
        "project": config.langsmith_project(),
        "api_key_present": bool(config.langsmith_api_key()),
        "hide_inputs": _env_true("LANGSMITH_HIDE_INPUTS"),
        "hide_outputs": _env_true("LANGSMITH_HIDE_OUTPUTS"),
    }


def _env_true(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("true", "1", "yes")


def hide_io_note() -> str:
    """
    How to keep tracing's timing and structure without uploading the contents.

    Set LANGSMITH_HIDE_INPUTS=true and LANGSMITH_HIDE_OUTPUTS=true. You still
    get the shape of every run — which nodes ran, in what order, how long each
    took, where it failed — with the payloads stripped. It is a blunt tool: it
    also removes the extraction output, which is usually the thing you opened
    the trace to read. Useful for watching performance on live traffic, not for
    debugging a specific booking.
    """
    return "LANGSMITH_HIDE_INPUTS=true LANGSMITH_HIDE_OUTPUTS=true"


# ── Per-run labelling ──────────────────────────────────────────────────────────

#: Where a run came in from. Used as a tag, so keep these short and stable —
#: renaming one orphans every historical trace filtered on the old name.
CHANNEL_GOOGLE_CHAT = "google_chat"
CHANNEL_WEB = "web"
CHANNEL_CLI = "cli"

#: What kind of input started the run. The distinction matters more than it
#: looks: a "resume" is the second half of a booking that paused at an
#: interrupt, so its trace is short and starts mid-graph. Without this label
#: those look like broken runs.
TURN_MESSAGE = "message"
TURN_RESUME = "resume"
TURN_BUTTON = "button"


def run_config(
    thread_id: str,
    *,
    channel: str,
    turn: str = TURN_MESSAGE,
    **metadata,
) -> dict:
    """
    The config dict passed to `graph.invoke()` / `graph.stream()`.

    Replaces the bare `{"configurable": {"thread_id": thread_id}}` that used to
    be written out at each of the five call sites. It still carries the
    thread_id — that part is load-bearing, it is how a paused booking is
    resumed — and adds the labels LangSmith needs.

    Why each piece:

    `configurable.thread_id`
        Unchanged, and the reason this function must be used everywhere rather
        than only where tracing is convenient.

    `metadata.thread_id`
        Duplicated on purpose. LangSmith's Threads view groups runs by a
        metadata key named `thread_id`, `session_id` or `conversation_id`.
        With it, all six turns of one booking read as one conversation; without
        it they are six unrelated rows and you have to reassemble the story by
        timestamp.

    `metadata.dry_run`
        The single most valuable label here. A dry-run trace and a live trace
        look identical — same nodes, same "invoice created" message — and only
        one of them charged a real customer. Never read a trace from this agent
        without checking this field.

    `tags`
        The same facts again as flat strings, because LangSmith's run list
        filters on tags with one click, where metadata takes a query.

    `run_name`
        What the run is called in the list. The default would be "LangGraph"
        for every single run.

    Extra keyword arguments are merged into metadata, for anything a caller
    knows that this function cannot — the Chat space, a button's value.

    Costs nothing when tracing is off: LangGraph ignores keys it doesn't need,
    and nothing here is uploaded unless LANGSMITH_TRACING is true.
    """
    backend = config.model_backend()
    mode = "dry_run" if config.dry_run() else "live"

    return {
        "configurable": {"thread_id": thread_id},
        "run_name": f"ops-agent:{channel}:{turn}",
        "tags": [f"channel:{channel}", f"turn:{turn}", f"backend:{backend}", mode],
        "metadata": {
            "thread_id": thread_id,
            "channel": channel,
            "turn": turn,
            "dry_run": config.dry_run(),
            "model_backend": backend,
            **metadata,
        },
    }


#: Metadata keys `run_config` sets itself. Anything else in there came from a
#: caller's **metadata and must survive a relabel.
_OWN_METADATA_KEYS = ("thread_id", "channel", "turn", "dry_run", "model_backend")


def as_resume(cfg: dict) -> dict:
    """
    Relabel a config as a resume, keeping everything else about it.

    Whether a turn is a resume is only known after reading the graph's state,
    which happens after the config has already been built (the state read needs
    the thread_id the config carries). Rather than build the config twice at
    every call site, build it once and relabel here.

    Returns a new dict; the original is untouched.
    """
    meta = cfg.get("metadata") or {}
    extra = {k: v for k, v in meta.items() if k not in _OWN_METADATA_KEYS}
    return run_config(
        cfg["configurable"]["thread_id"],
        channel=meta.get("channel", "unknown"),
        turn=TURN_RESUME,
        **extra,
    )
