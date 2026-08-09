"""
Per-node model registry.

This is the ONLY module that knows a model provider exists. Nodes call
`get_model("router")` and never import a chat class, so switching the whole
agent from OpenAI to OpenRouter is one env var, not a refactor.

Different nodes want genuinely different things — reading a blurry screenshot
of someone's address is not the same job as deciding whether a message is a
question or a booking — so each gets its own entry.
"""

import os
from dataclasses import dataclass

from langchain.chat_models import init_chat_model

from services import config


@dataclass(frozen=True)
class ModelSpec:
    """One node's model on each backend, plus why it was chosen."""

    openai: str
    openrouter: str
    rationale: str


REGISTRY: dict[str, ModelSpec] = {
    # Reads the customer screenshot. Highest stakes in the whole agent: a
    # misread digit in a phone number or a wrong street number reaches a real
    # customer. Worth the most capable model available.
    "extract_screenshot": ModelSpec(
        openai="gpt-5.1",
        openrouter="anthropic/claude-sonnet-4.6",
        rationale="vision accuracy; errors here are customer-visible",
    ),
    # Parses free-text answers ("next Friday 8-9am, 3 guys") into fields.
    # Needs solid date reasoning but the output is always re-validated in Python.
    "parse_reply": ModelSpec(
        openai="gpt-4.1",
        openrouter="anthropic/claude-sonnet-4.6",
        rationale="relative-date reasoning, output is Python-validated",
    ),
    # Answers calendar questions. Multi-step tool calling plus date math.
    "analytics": ModelSpec(
        openai="gpt-5.1",
        openrouter="anthropic/claude-sonnet-4.6",
        rationale="multi-step tool calling over calendar data",
    ),
    # Three-way intent classification. Runs on every single turn, so it should
    # be the cheapest thing that can do the job reliably.
    "router": ModelSpec(
        openai="gpt-4.1-mini",
        openrouter="anthropic/claude-haiku-4.5",
        rationale="cheap 3-way classification on every turn",
    ),
    # Small talk and formatting summaries.
    "chat": ModelSpec(
        openai="gpt-4.1-mini",
        openrouter="anthropic/claude-haiku-4.5",
        rationale="conversational filler, low stakes",
    ),
}


def _model_name(node: str) -> str:
    """Resolve a node's model, honouring a per-node env override."""
    if node not in REGISTRY:
        raise KeyError(f"No model registered for node {node!r}. Known: {sorted(REGISTRY)}")

    # e.g. OPS_MODEL_ROUTER=gpt-4o-mini
    if override := os.getenv(f"OPS_MODEL_{node.upper()}"):
        return override

    spec = REGISTRY[node]
    return spec.openrouter if config.model_backend() == "openrouter" else spec.openai


def _supports_temperature(model_name: str) -> bool:
    """
    GPT-5 family rejects any temperature other than the default — passing
    temperature=0 raises a 400. Everything else accepts it.
    """
    return not model_name.startswith(("gpt-5", "o1", "o3", "o4"))


def get_model(node: str, **overrides):
    """
    Chat model for a node.

    Deterministic (temperature=0) wherever the model allows it — this agent
    extracts data and classifies intent, neither of which benefits from
    sampling variety.
    """
    name = _model_name(node)
    backend = config.model_backend()

    kwargs: dict = {"timeout": 90, "max_retries": 3}
    if _supports_temperature(name):
        kwargs["temperature"] = 0
    kwargs.update(overrides)

    if backend == "openrouter":
        # ChatOpenRouter, not ChatOpenAI+base_url: the latter targets the
        # official OpenAI spec and drops OpenRouter's routing/reasoning fields.
        return init_chat_model(
            name,
            model_provider="openrouter",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            **kwargs,
        )

    return init_chat_model(name, model_provider="openai", **kwargs)


def describe() -> str:
    """Human-readable dump of what each node will actually call."""
    backend = config.model_backend()
    lines = [f"backend: {backend}"]
    for node in REGISTRY:
        lines.append(f"  {node:20} {_model_name(node):24} — {REGISTRY[node].rationale}")
    return "\n".join(lines)
