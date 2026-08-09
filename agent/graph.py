"""
Graph wiring. The whole topology lives here, readable top to bottom.

    START -> router -> analytics                                      -> END
                    -> chat                                           -> END
                    -> extract_screenshot -> resolve_addresses -> validate
                         validate --incomplete--> ask_missing -> resolve_addresses -> validate
                         validate --complete----> confirm
                             confirm --yes--> execute -> END
                             confirm --no---> END
                             confirm --edit-> ask_missing

The two interrupt points (ask_missing, confirm) are what make this a graph
rather than a script: the run pauses mid-flow, possibly for a long time, and
resumes exactly where it stopped.
"""

import logging

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph

from agent.nodes import actions
from agent.nodes import analytics as analytics_node
from agent.nodes import ask_missing as ask_missing_node
from agent.nodes import chat as chat_node
from agent.nodes import confirm as confirm_node
from agent.nodes import extract_screenshot as extract_node
from agent.nodes import resolve_addresses as address_node
from agent.nodes import router as router_node
from agent.nodes import validate_checklist as validate_node
from agent.state import OpsAgentState, new_ledger

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def give_up(state: OpsAgentState) -> dict:
    """Reached when the checklist can't be satisfied after several rounds."""
    missing = state.get("missing_fields") or []
    return {
        "messages": [
            AIMessage(
                content=(
                    "I'm going in circles on "
                    + (", ".join(missing) if missing else "this booking")
                    + ". Worth setting this one up directly in GHL, or start over "
                    "with a clearer screenshot."
                )
            )
        ]
    }


# ── Build ──────────────────────────────────────────────────────────────────────

def build_graph(checkpointer=None):
    """
    Compile the graph.

    A checkpointer is REQUIRED for ask_missing and confirm to work — without
    one, the graph cannot pause and resume. It also provides conversation
    memory, keyed by thread_id.

    Passing None lets LangGraph Server inject its own managed (Postgres-backed)
    checkpointer, which is what `langgraph dev` does.
    """
    builder = StateGraph(OpsAgentState)

    builder.add_node("router", router_node.route)
    builder.add_node("analytics", analytics_node.analytics)
    builder.add_node("chat", chat_node.chat)

    builder.add_node("extract_screenshot", extract_node.extract_screenshot)
    builder.add_node("resolve_addresses", address_node.resolve_addresses)
    builder.add_node("validate", validate_node.validate_checklist)
    builder.add_node("ask_missing", ask_missing_node.ask_missing)
    builder.add_node("confirm", confirm_node.confirm)
    builder.add_node("give_up", give_up)

    # Execution: contact first (the other three need its id), then fan out.
    builder.add_node("execute", actions.act_upsert_contact)
    builder.add_node("act_calendar", actions.act_calendar_event)
    builder.add_node("act_invoice", actions.act_deposit_invoice)
    builder.add_node("act_email", actions.act_confirmation_email)
    builder.add_node("report", actions.report)

    builder.add_edge(START, "router")
    builder.add_conditional_edges(
        "router",
        router_node.pick_lane,
        {"analytics": "analytics", "intake": "extract_screenshot", "chat": "chat"},
    )
    builder.add_edge("analytics", END)
    builder.add_edge("chat", END)

    # Intake: extract, complete the addresses, then check.
    builder.add_edge("extract_screenshot", "resolve_addresses")
    builder.add_edge("resolve_addresses", "validate")

    builder.add_conditional_edges(
        "validate",
        validate_node.next_step,
        {"ask_missing": "ask_missing", "confirm": "confirm", "give_up": "give_up"},
    )

    # Answers may contain new addresses, so they go back through resolution
    # before being re-checked. This is the loop.
    builder.add_edge("ask_missing", "resolve_addresses")

    # confirm returns a Command and routes itself to execute / ask_missing / END.

    # The contact must exist before anything can be attached to it. If it fails,
    # skip the other three rather than firing three guaranteed failures.
    builder.add_conditional_edges(
        "execute",
        actions.contact_gate,
        [*actions.FAN_OUT, "report"],
    )

    # These three are independent and run in parallel. The ledger's merge
    # reducer is what keeps their three writes from overwriting each other.
    for node in actions.FAN_OUT:
        builder.add_edge(node, "report")

    builder.add_edge("report", END)
    builder.add_edge("give_up", END)

    return builder.compile(checkpointer=checkpointer)


#: Entry point referenced by langgraph.json.
graph = build_graph()
