#!/usr/bin/env python
"""
Production entrypoint. This is what Railway runs.

    uvicorn app:app --host 0.0.0.0 --port $PORT

Serves two things:

  /google-chat   the Google Chat webhook — the real interface for the team
  /              the browser UI, ONLY if WEB_UI_TOKEN is set

The browser UI is kept because it is far easier to debug against than Chat, but
it is off by default in production. An unauthenticated page on a public URL
would let anyone who found it create GoHighLevel contacts, book trucks and text
real customers. `server.py` stays as-is for local work, where binding to
localhost is the protection.
"""

import logging
import os

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from agent.graph import build_graph
from channels import google_chat
from services import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _checkpointer():
    """
    Interrupts cannot work without a checkpointer, and a half-finished booking
    must survive a restart — Railway restarts on every deploy.

    SQLite on the container's local disk is the pragmatic choice for one
    instance at this volume. It does NOT survive a redeploy unless a Railway
    volume is mounted at the path, and it does not work across multiple
    instances. Both are noted in the README as the trigger for moving to
    Postgres.
    """
    from langgraph.checkpoint.sqlite import SqliteSaver

    path = os.getenv("CHECKPOINT_DB", "/data/agent_threads.sqlite")
    directory = os.path.dirname(path)
    if directory:
        try:
            os.makedirs(directory, exist_ok=True)
        except OSError:
            path = "agent_threads.sqlite"
            logger.warning("Could not use CHECKPOINT_DB directory; falling back to %s", path)

    logger.info("Checkpoints: %s", path)
    manager = SqliteSaver.from_conn_string(path)
    return manager.__enter__()


app = FastAPI(title="Splendid Moving ops agent")

graph = build_graph(checkpointer=_checkpointer())
google_chat.attach_graph(graph)
app.include_router(google_chat.router)


@app.get("/health")
def health():
    """Railway's health check. Deliberately reveals no configuration."""
    return {"status": "ok"}


@app.get("/api/status")
def status():
    return {
        "dry_run": config.dry_run(),
        "backend": config.model_backend(),
        "deposit": config.deposit_amount(),
        "web_ui": bool(config.web_ui_token()),
    }


# ── Optional browser UI ────────────────────────────────────────────────────────

if config.web_ui_token():
    from pathlib import Path

    from fastapi.responses import FileResponse

    import server as local_server

    # Share the one graph, so a booking started in the browser can be finished
    # from Chat and vice versa. Without this the two channels get separate
    # checkpointers and each holds half the conversation.
    local_server.set_graph(graph)

    STATIC = Path(__file__).parent / "static"

    @app.middleware("http")
    async def require_token(request: Request, call_next):
        """Gate only the UI routes; the Chat webhook has its own JWT check."""
        path = request.url.path
        if path.startswith(("/health", "/google-chat")):
            return await call_next(request)

        supplied = (
            request.query_params.get("token")
            or request.headers.get("x-ui-token", "")
        )
        if supplied != config.web_ui_token():
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        return await call_next(request)

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    # Reuse the local server's handlers rather than reimplementing the resume
    # protocol — getting that wrong in a second place is exactly the bug the
    # single-implementation rule exists to prevent.
    app.post("/api/chat")(local_server.chat)
    app.post("/api/chat/stream")(local_server.chat_stream)
    app.post("/api/reset")(local_server.reset)

    logger.info("Browser UI enabled at / (token required)")
else:
    logger.info("Browser UI disabled — set WEB_UI_TOKEN to enable it")


if __name__ == "__main__":
    import uvicorn

    mode = "DRY RUN" if config.dry_run() else "*** LIVE — writes are real ***"
    port = int(os.getenv("PORT", "8080"))
    logger.info("Splendid Moving ops agent — %s — port %s", mode, port)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
