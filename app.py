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


#: Module-level so the connection is never garbage collected. See below.
_checkpoint_conn = None


def _checkpointer():
    """
    Interrupts cannot work without a checkpointer, and a half-finished booking
    must survive a restart — Railway restarts on every deploy.

    The connection is opened directly and held in a module global rather than
    via `SqliteSaver.from_conn_string(...).__enter__()`. That helper returns a
    context manager, and if the only reference to it is a local variable, it is
    garbage collected as soon as this function returns — which closes the
    database underneath a perfectly live checkpointer. The failure surfaces
    much later, on the first message, as "Cannot operate on a closed database".

    `check_same_thread=False` is required because requests are served from a
    thread pool, so the connection is used from whichever thread handles the
    call. `timeout` makes concurrent writes wait rather than immediately
    raising "database is locked".

    SQLite on a mounted volume is the pragmatic choice for one instance at this
    volume of work. It does not work across multiple instances — see the
    numReplicas note in railway.json.
    """
    global _checkpoint_conn
    import sqlite3

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
    _checkpoint_conn = sqlite3.connect(path, check_same_thread=False, timeout=30)
    saver = SqliteSaver(_checkpoint_conn)
    saver.setup()
    return saver


app = FastAPI(title="Splendid Moving ops agent")

graph = build_graph(checkpointer=_checkpointer())
google_chat.attach_graph(graph)
app.include_router(google_chat.router)


@app.get("/health")
def health():
    """Railway's health check. Deliberately reveals no configuration."""
    return {"status": "ok"}


@app.get("/email-logo.png")
def email_logo():
    """
    The truck logo at the top of every confirmation email.

    Email clients cannot read a local file, so the image needs a public URL.
    Serving it from this app rather than GoHighLevel's media library keeps it
    tied to the repo: it deploys with the code, and nobody can break every
    future confirmation email by tidying up a media library.

    Cached for a year — it changes about never, and customers should not wait
    on a round trip to see the header.
    """
    from pathlib import Path

    from fastapi.responses import FileResponse

    return FileResponse(
        Path(__file__).parent / "static" / "email-logo.png",
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


@app.get("/api/status")
def status():
    """
    Health plus enough Chat config to diagnose a 401 without reading logs.

    Nothing here is secret: the audience is either a public webhook URL or a
    project number, and credentials are reported only as present/absent. A
    silent 401 is otherwise indistinguishable from a forged request, and
    Railway's interleaved logs make it slow to pin down.
    """
    return {
        "dry_run": config.dry_run(),
        "backend": config.model_backend(),
        "deposit": config.deposit_amount(),
        "web_ui": bool(config.web_ui_token()),
        "chat": {
            "verify_requests": config.chat_verify_requests(),
            "audience": config.chat_audience() or None,
            "audience_type": (
                "project_number" if config.chat_audience_is_project_number()
                else "endpoint_url" if config.chat_audience()
                else "NOT SET"
            ),
            "credentials_present": bool(config.chat_credentials_b64()),
            "traffic": google_chat.traffic(),
            "last_rejection": google_chat.last_rejection(),
            "last_unknown_event": google_chat.last_unknown_event(),
            "last_error": google_chat.last_error(),
        },
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
        # /email-logo.png MUST stay public: it is fetched by the customer's
        # email client, which has no token. Gating it would break the header
        # image in every confirmation email, and only for customers — it would
        # still look fine to anyone testing while logged in.
        if path.startswith(("/health", "/google-chat", "/email-logo.png")):
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
