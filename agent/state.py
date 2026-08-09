"""
Graph state.

State is shared memory across every node. The important design decision here is
`ledger`: a per-action record of what actually happened, which is what lets the
agent recover from "contact created but calendar booking failed" without
redoing the half that worked.
"""

from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

# ── Action names ───────────────────────────────────────────────────────────────
# The four side effects, in dependency order. upsert_contact runs first and
# alone because the other three need the contact id it returns.

ACTION_CONTACT = "upsert_contact"
ACTION_CALENDAR = "calendar_event"
ACTION_INVOICE = "deposit_invoice"
ACTION_EMAIL = "confirmation_email"

ALL_ACTIONS = (ACTION_CONTACT, ACTION_CALENDAR, ACTION_INVOICE, ACTION_EMAIL)

ActionStatus = Literal["pending", "success", "failed", "skipped"]


class ActionResult(TypedDict, total=False):
    """Outcome of one side effect."""

    status: ActionStatus
    result: dict[str, Any]  # {"contact_id": ...} / {"event_id": ...} / {"invoice_id": ...}
    error: str | None
    attempts: int


def merge_ledger(
    current: dict[str, ActionResult] | None,
    update: dict[str, ActionResult] | None,
) -> dict[str, ActionResult]:
    """
    Reducer for the action ledger.

    The three post-contact actions run as parallel branches, each returning a
    dict holding only its own key. Without a reducer, LangGraph's default
    last-write-wins would discard two of the three results and the agent would
    report success for actions it never recorded.

    Key-level merge, not a deep merge: each action owns its entry outright, so
    a retry cleanly replaces the previous attempt.
    """
    if not current:
        return dict(update or {})
    if not update:
        return dict(current)
    return {**current, **update}


def new_ledger() -> dict[str, ActionResult]:
    """Fresh ledger with every action pending."""
    return {
        name: {"status": "pending", "result": {}, "error": None, "attempts": 0}
        for name in ALL_ACTIONS
    }


def succeeded(ledger: dict[str, ActionResult], action: str) -> bool:
    return (ledger or {}).get(action, {}).get("status") == "success"


def failed_actions(ledger: dict[str, ActionResult]) -> list[str]:
    return [
        name for name in ALL_ACTIONS if (ledger or {}).get(name, {}).get("status") == "failed"
    ]


def all_done(ledger: dict[str, ActionResult]) -> bool:
    return all(
        (ledger or {}).get(name, {}).get("status") in ("success", "skipped")
        for name in ALL_ACTIONS
    )


# ── Intent ─────────────────────────────────────────────────────────────────────

Intent = Literal["analytics", "intake", "chat"]


# ── Graph state ────────────────────────────────────────────────────────────────


class OpsAgentState(TypedDict, total=False):
    """
    Shared state for the whole graph.

    `total=False` because most fields only exist once the relevant branch has
    run — an analytics turn never populates the intake fields.
    """

    #: Conversation history. add_messages appends and de-duplicates by id.
    messages: Annotated[list[AnyMessage], add_messages]

    #: Which lane the router picked for the current turn.
    intent: Intent

    #: Extracted + user-supplied job fields. Shape defined in schemas/intake.py.
    intake: dict[str, Any]

    #: Per-field confidence from the vision extraction, 0.0–1.0.
    #: Low confidence is treated as missing rather than trusted.
    field_confidence: dict[str, float]

    #: What the checklist still needs. Empty means ready to confirm.
    missing_fields: list[str]

    #: True once the user has approved the summary at the confirm gate.
    approved: bool

    #: sha256(normalized_phone + move_date) — duplicate-booking detection.
    job_fingerprint: str

    #: An existing calendar job matching the fingerprint, surfaced at confirm.
    duplicate_warning: dict[str, Any] | None

    #: Per-action outcomes. See merge_ledger.
    ledger: Annotated[dict[str, ActionResult], merge_ledger]
