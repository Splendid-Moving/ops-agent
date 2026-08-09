#!/usr/bin/env python
"""
Terminal chat with the agent. Local dev only — the real frontend will talk to
LangGraph Server over HTTP.

    python chat_cli.py
    python chat_cli.py --image screenshot.png "book this one"

Commands: /new (fresh thread), /state (dump state), /quit
"""

import argparse
import base64
import mimetypes
import sys
import uuid
from pathlib import Path

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from agent.graph import build_graph
from services import config

DIM = "\033[2m"
BOLD = "\033[1m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
RESET = "\033[0m"


def image_message(path: str, text: str) -> HumanMessage:
    """Build a multimodal message. Images are inlined as data URLs."""
    data = Path(path).read_bytes()
    mime = mimetypes.guess_type(path)[0] or "image/png"
    b64 = base64.b64encode(data).decode()
    return HumanMessage(
        content=[
            {"type": "text", "text": text or "Book this job."},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
        ]
    )


def send(graph, thread_id: str, payload):
    """
    Send input to the graph, handling the resume protocol.

    THIS IS THE PART THAT IS EASY TO GET WRONG. When the graph is paused at an
    interrupt(), it must be resumed with Command(resume=...). Sending a plain
    message dict instead restarts the graph from the top and silently discards
    the in-progress booking.

    Any frontend built against this agent needs this exact check.
    """
    config_ = {"configurable": {"thread_id": thread_id}}
    snapshot = graph.get_state(config_)

    if snapshot.next:  # paused mid-graph, waiting on a human
        result = graph.invoke(Command(resume=payload), config_)
    else:
        message = payload if not isinstance(payload, str) else HumanMessage(content=payload)
        result = graph.invoke({"messages": [message]}, config_)

    # Surface a new pause, if the graph stopped at one.
    if interrupts := result.get("__interrupt__"):
        return None, interrupts[0].value
    return result, None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("message", nargs="?", help="send one message and exit")
    parser.add_argument("--image", help="attach an image")
    args = parser.parse_args()

    graph = build_graph(checkpointer=InMemorySaver())
    thread_id = str(uuid.uuid4())

    mode = "DRY RUN" if config.dry_run() else "LIVE — writes are real"
    print(f"{BOLD}Splendid Moving ops agent{RESET} {DIM}({mode}, backend={config.model_backend()}){RESET}")

    def show(result, pause):
        if pause is not None:
            print(f"\n{YELLOW}⏸  waiting on you{RESET}")
            print(pause if isinstance(pause, str) else pause.get("message", pause))
            return
        msg = result["messages"][-1]
        print(f"\n{CYAN}agent{RESET} {msg.content}")

    # One-shot mode
    if args.message or args.image:
        payload = (
            image_message(args.image, args.message or "")
            if args.image
            else HumanMessage(content=args.message)
        )
        show(*send(graph, thread_id, payload))
        return 0

    # Interactive
    print(f"{DIM}/new for a fresh thread, /state to inspect, /quit to exit{RESET}")
    while True:
        try:
            text = input(f"\n{BOLD}you{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not text:
            continue
        if text in ("/quit", "/exit"):
            return 0
        if text == "/new":
            thread_id = str(uuid.uuid4())
            print(f"{DIM}new thread{RESET}")
            continue
        if text == "/state":
            snap = graph.get_state({"configurable": {"thread_id": thread_id}})
            print(f"{DIM}intent={snap.values.get('intent')} "
                  f"messages={len(snap.values.get('messages', []))} "
                  f"next={snap.next}{RESET}")
            continue

        try:
            show(*send(graph, thread_id, text))
        except Exception as exc:
            print(f"\n\033[31merror\033[0m {type(exc).__name__}: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
