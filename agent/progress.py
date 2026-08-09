"""
Live progress events, surfaced to the UI while the graph runs.

Written for a dispatcher, not an engineer. "Checking the calendar…" and
"Found 153 jobs", never "invoking tool count_jobs with args {...}". The point is
that someone watching can tell what the agent is doing to their business, and
notice when it is doing something they did not expect.

Events travel on LangGraph's `custom` stream, so they arrive in real time rather
than at the end of the turn.
"""

import logging
from typing import Literal

from langgraph.config import get_stream_writer

logger = logging.getLogger(__name__)

Status = Literal["working", "done", "warn", "error"]


def emit(text: str, status: Status = "working", detail: str = "") -> None:
    """
    Push one progress line to the UI.

    Never raises. Progress is decoration — a failure to report must not take
    down the booking it was reporting on. Outside a graph run (tests, scripts)
    there is no writer and this quietly does nothing.
    """
    try:
        writer = get_stream_writer()
    except Exception:
        return
    if writer is None:
        return
    try:
        writer({"type": "progress", "status": status, "text": text, "detail": detail})
    except Exception:
        logger.debug("progress emit failed", exc_info=True)


def working(text: str, detail: str = "") -> None:
    emit(text, "working", detail)


def done(text: str, detail: str = "") -> None:
    emit(text, "done", detail)


def warn(text: str, detail: str = "") -> None:
    emit(text, "warn", detail)


def error(text: str, detail: str = "") -> None:
    emit(text, "error", detail)
