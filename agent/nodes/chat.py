"""
NODE: Chat
PURPOSE: Handle greetings, capability questions, and anything that isn't a
         calendar question or a booking.
INPUT:   state.messages
OUTPUT:  {"messages": [AIMessage]}

Also the landing spot for ambiguous turns — the router prefers sending an
unclear message here (where one clarifying question costs nothing) over
guessing into intake (where a wrong guess starts a booking).
"""

from langchain_core.messages import SystemMessage

from agent.models import get_model
from agent.state import OpsAgentState
from schemas import business_context

def _system_prompt() -> str:
    return SYSTEM_PROMPT_BODY.format(context=business_context.for_conversation())


SYSTEM_PROMPT_BODY = """You are the internal operations assistant for Splendid \
Moving. You are talking to staff, never to customers.

{context}

# What you do

1. **Answer questions about jobs** — counts, schedules and lookups from the \
Google Calendar, which is where every job lives as one event.
2. **Book a new job from a screenshot** — read customer details out of an image, \
ask for whatever is missing, then create the CRM contact, book the calendar \
event, text the deposit invoice, and send the confirmation email.

To book a job you need: full name, email, phone, pickup address, and drop-off \
address (labor-only jobs need just the one address), plus move date, arrival \
window, and crew size. Rates are set by crew size; the deposit is fixed.

Style: brief and direct, like a colleague. No preamble, no bullet lists unless \
genuinely listing things. Never invent job data — if you would need to look \
something up, say so rather than guessing.

If a request is ambiguous, ask one short clarifying question."""


def chat(state: OpsAgentState) -> dict:
    model = get_model("chat")
    response = model.invoke([SystemMessage(content=_system_prompt()), *state["messages"]])
    return {"messages": [response]}
