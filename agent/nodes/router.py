"""
NODE: Router
PURPOSE: Classify the current turn into one of three lanes.
INPUT:   state.messages
OUTPUT:  {"intent": "analytics" | "intake" | "chat"}

Note on scope: this node does NOT need to detect "the user is answering a
missing-field question". When the graph is paused at an interrupt(), a
Command(resume=...) re-enters the paused node directly and the router never
runs at all. The router only sees genuinely new turns.
"""

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agent.models import get_model
from agent.state import OpsAgentState

logger = logging.getLogger(__name__)


class Route(BaseModel):
    """Structured routing decision."""

    intent: str = Field(
        description=(
            "One of: 'analytics' for questions about existing jobs/calendar data; "
            "'intake' for booking a new job or processing a customer screenshot; "
            "'chat' for anything else."
        )
    )
    reasoning: str = Field(description="One short sentence explaining the choice.")


SYSTEM_PROMPT = """You route messages for the internal operations assistant at \
Splendid Moving, a Los Angeles moving company. Staff talk to you; customers never do.

Pick exactly one lane:

**analytics** — questions about jobs that already exist. Counting, looking up, \
summarising, checking the schedule.
  "how many jobs did we have last month?"
  "what's on the calendar Friday?"
  "how many Yelp jobs in July?"
  "who are we moving tomorrow?"

**intake** — booking a NEW job. Almost always accompanied by a screenshot of \
customer details, but not always.
  "book this one" (with image)
  [image with no text]
  "new job for Sarah Chen, moving next Friday"
  "can you set this customer up"

**chat** — anything else: greetings, questions about your capabilities, \
clarifications, thanks, or requests you cannot serve.
  "hey"
  "what can you do?"
  "never mind"

Rules:
- An image attachment almost always means **intake**. The only exception is an \
explicit question about existing data.
- If a message asks about the PAST or about EXISTING jobs, it is analytics, \
even if it names a customer.
- When genuinely torn between analytics and intake, choose **chat** — asking \
one clarifying question is much cheaper than starting the wrong workflow."""


def _has_image(message) -> bool:
    """True if the message carries an image part (multimodal content block)."""
    content = getattr(message, "content", None)
    if not isinstance(content, list):
        return False
    for part in content:
        if isinstance(part, dict) and part.get("type") in ("image", "image_url"):
            return True
    return False


def route(state: OpsAgentState) -> dict:
    messages = state.get("messages", [])
    if not messages:
        return {"intent": "chat"}

    last = messages[-1]

    # An image is a strong enough signal to skip the model call entirely.
    # Cheaper, and removes a class of misroute on image-only turns.
    if _has_image(last):
        logger.info("Router: image attachment -> intake")
        return {"intent": "intake"}

    model = get_model("router").with_structured_output(Route)

    # Only the last few turns — routing is about the current message, and a long
    # history makes the classifier drift toward whatever dominated earlier.
    recent = messages[-4:]
    decision = model.invoke([SystemMessage(content=SYSTEM_PROMPT), *recent])

    intent = decision.intent if decision.intent in ("analytics", "intake", "chat") else "chat"
    logger.info("Router: %s — %s", intent, decision.reasoning)
    return {"intent": intent}


def pick_lane(state: OpsAgentState) -> str:
    """Conditional-edge function. Maps state.intent to a node name."""
    return state.get("intent", "chat")
