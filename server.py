#!/usr/bin/env python
"""
Local web server for testing the agent.

    python server.py         ->  http://localhost:8080

Deliberately thin. Its only real job is the resume protocol: when the graph is
paused at an interrupt, input must arrive as Command(resume=...) rather than as
a new message. Sending the wrong one silently restarts the graph and discards
the in-progress booking, which is the single easiest way to break this agent —
so it is handled in one place, here, and the browser never has to know.

This is a dev tool. It has no auth and binds to localhost only.
"""

import base64
import json
import logging
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from agent.graph import build_graph
from services import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

STATIC = Path(__file__).parent / "static"
DB_PATH = Path(__file__).parent / ".agent_threads.sqlite"

app = FastAPI(title="Splendid Moving ops agent")

# SQLite rather than in-memory so a server restart doesn't lose a half-finished
# booking that's paused waiting on an answer.
_cm = SqliteSaver.from_conn_string(str(DB_PATH))
checkpointer = _cm.__enter__()
graph = build_graph(checkpointer=checkpointer)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/status")
def status():
    return {
        "dry_run": config.dry_run(),
        "backend": config.model_backend(),
        "deposit": config.deposit_amount(),
    }


@app.post("/api/chat")
async def chat(
    message: str = Form(""),
    thread_id: str = Form(""),
    image: UploadFile | None = File(None),
):
    thread_id = thread_id or str(uuid.uuid4())
    cfg = {"configurable": {"thread_id": thread_id}}

    payload: object = message
    if image is not None:
        raw = await image.read()
        b64 = base64.b64encode(raw).decode()
        mime = image.content_type or "image/png"
        payload = HumanMessage(
            content=[
                {"type": "text", "text": message or "Book this job."},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            ]
        )

    try:
        snapshot = graph.get_state(cfg)

        if snapshot.next:
            # Paused at an interrupt. Resume values are plain text — an image
            # here would be a new job, not an answer, so it starts a new turn.
            resume_value = message if not isinstance(payload, HumanMessage) else message
            result = graph.invoke(Command(resume=resume_value), cfg)
        else:
            msg = payload if isinstance(payload, HumanMessage) else HumanMessage(content=message)
            result = graph.invoke({"messages": [msg]}, cfg)

    except Exception as exc:
        logger.exception("Graph invocation failed")
        return JSONResponse(
            {"thread_id": thread_id, "paused": False,
             "reply": f"Something broke: {type(exc).__name__}: {exc}"},
            status_code=200,
        )

    if interrupts := result.get("__interrupt__"):
        value = interrupts[0].value
        return {
            "thread_id": thread_id,
            "paused": True,
            "kind": value.get("type", "question"),
            "reply": value.get("message", str(value)),
        }

    messages = result.get("messages") or []
    reply = messages[-1].content if messages else "(no response)"
    return {
        "thread_id": thread_id,
        "paused": False,
        "kind": result.get("intent", ""),
        "reply": reply,
    }


@app.post("/api/chat/stream")
async def chat_stream(
    message: str = Form(""),
    thread_id: str = Form(""),
    image: UploadFile | None = File(None),
):
    """
    Same as /api/chat, but streams progress events as they happen.

    Nodes emit plain-language lines via agent.progress; LangGraph carries them
    on the `custom` stream. The browser gets to watch the agent work instead of
    staring at a spinner while four API calls happen invisibly.
    """
    thread_id = thread_id or str(uuid.uuid4())
    cfg = {"configurable": {"thread_id": thread_id}}

    payload: object = message
    if image is not None:
        raw = await image.read()
        b64 = base64.b64encode(raw).decode()
        mime = image.content_type or "image/png"
        payload = HumanMessage(
            content=[
                {"type": "text", "text": message or "Book this job."},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            ]
        )

    def event(kind: str, data: dict) -> str:
        return f"data: {json.dumps({'event': kind, **data})}\n\n"

    def run():
        yield event("start", {"thread_id": thread_id})
        try:
            snapshot = graph.get_state(cfg)
            if snapshot.next:
                graph_input = Command(resume=message)
            else:
                msg = payload if isinstance(payload, HumanMessage) else HumanMessage(content=message)
                graph_input = {"messages": [msg]}

            final: dict = {}
            for mode, chunk in graph.stream(
                graph_input, cfg, stream_mode=["custom", "values"]
            ):
                if mode == "custom" and isinstance(chunk, dict):
                    if chunk.get("type") == "progress":
                        yield event("progress", chunk)
                elif mode == "values":
                    final = chunk

            state = graph.get_state(cfg)
            if state.tasks and (interrupts := [t for t in state.tasks if t.interrupts]):
                value = interrupts[0].interrupts[0].value
                yield event("done", {
                    "paused": True,
                    "kind": value.get("type", "question"),
                    "reply": value.get("message", str(value)),
                })
            else:
                messages = final.get("messages") or []
                yield event("done", {
                    "paused": False,
                    "kind": final.get("intent", ""),
                    "reply": messages[-1].content if messages else "(no response)",
                })

        except Exception as exc:
            logger.exception("Stream failed")
            yield event("done", {
                "paused": False,
                "reply": f"Something broke: {type(exc).__name__}: {exc}",
            })

    return StreamingResponse(
        run(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/reset")
def reset():
    """Start a fresh thread. The old one stays in the DB."""
    return {"thread_id": str(uuid.uuid4())}


if __name__ == "__main__":
    import uvicorn

    mode = "DRY RUN — nothing will be created" if config.dry_run() else "*** LIVE — writes are real ***"
    print(f"\n  Splendid Moving ops agent")
    print(f"  {mode}")
    print(f"  http://localhost:8080\n")
    uvicorn.run(app, host="127.0.0.1", port=8080, log_level="warning")
