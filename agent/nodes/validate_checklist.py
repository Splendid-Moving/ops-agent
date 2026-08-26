"""
NODE: Validate checklist
PURPOSE: Decide whether the job is ready to book.
INPUT:   state.intake
OUTPUT:  {"missing_fields": [...]} and the routing decision

Pure Python. No model call. The LLM proposes values; this decides whether they
are acceptable. Keeping the gate deterministic means the same intake always
produces the same verdict, and no prompt change can accidentally loosen it.
"""

import logging

from agent.state import OpsAgentState
from schemas import checklist as cl

logger = logging.getLogger(__name__)

#: Stop asking after this many ATTEMPTS AT AN ANSWER and hand back to the user.
#: Guards against a field that can never satisfy its validator looping forever.
MAX_ASK_ROUNDS = 6

#: Absolute cap on exchanges, whether or not they were answers.
#:
#: Deliberately looser than MAX_ASK_ROUNDS, because a dispatcher wandering off
#: mid-booking should cost patience, not the booking. This is the guard on the
#: guard: ask_missing only counts a round when a field actually lands, so a
#: model that misclassified every reply as off-topic would freeze _ask_rounds at
#: zero and give_up could never fire.
MAX_ASK_TURNS = 14


def validate_checklist(state: OpsAgentState) -> dict:
    intake = dict(state.get("intake") or {})
    result = cl.evaluate(intake)

    outstanding = [spec.name for spec in result.missing] + list(result.invalid)

    logger.info(
        "Checklist: %s",
        "complete" if result.is_complete else f"missing {outstanding}"
        + (" +labor" if result.needs_labor_answer else ""),
    )
    return {"missing_fields": outstanding}


def next_step(state: OpsAgentState) -> str:
    """Conditional edge: ask for more, or move to the confirm gate."""
    intake = dict(state.get("intake") or {})
    result = cl.evaluate(intake)

    if result.is_complete:
        return "confirm"

    rounds = int(intake.get("_ask_rounds", 0))
    turns = int(intake.get("_ask_turns", 0))
    if rounds >= MAX_ASK_ROUNDS or turns >= MAX_ASK_TURNS:
        logger.warning(
            "Hit the ask limit after %d answer attempts / %d exchanges; stopping.",
            rounds, turns,
        )
        return "give_up"

    return "ask_missing"
