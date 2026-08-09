"""
CHANNEL: Google Chat
PURPOSE: Let the team talk to the agent from Google Chat instead of the browser.
INPUT:   Google Chat interaction events (MESSAGE, CARD_CLICKED, ADDED_TO_SPACE)
OUTPUT:  Messages posted back into the same Chat space
DEPENDENCIES: fastapi, google-auth, google-api-python-client

This file contains no booking logic. It does four things:

  1. Proves the request really came from Google.
  2. Turns a Chat event into either a new message or a resume value.
  3. Runs the graph in the background, because Chat's 30-second webhook
     deadline is far shorter than a booking takes.
  4. Renders the result back — as a card with buttons when the agent is asking
     for approval, as plain text otherwise.

── The 30-second problem ──────────────────────────────────────────────────────

Google Chat abandons a webhook that takes longer than 30 seconds. A screenshot
intake does vision extraction, address resolution, several model calls, Maps and
GoHighLevel — routinely more than that.

So the webhook returns 200 immediately and does the real work in a background
task, posting results through the Chat REST API. The user sees a placeholder
message appear at once, which is then edited in place as the agent works and
finally becomes the answer. One message that evolves, rather than a wall of
fragments.

── Why threads matter here ────────────────────────────────────────────────────

`thread_id` is what lets a paused booking be resumed. Get it wrong and every
reply starts a brand-new conversation, silently discarding a half-finished job.
See `thread_id_for()` — the rule is subtler than it first appears.
"""

import base64
import io
import logging
import threading
import time
from typing import Any, Callable

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from google.auth.transport import requests as google_requests
from google.oauth2 import service_account
from google.oauth2 import id_token as google_id_token
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from services import config

logger = logging.getLogger(__name__)

router = APIRouter()

#: How often the placeholder message may be rewritten while the graph runs.
#: Chat rate-limits message edits; progress is decoration and must never be the
#: reason a booking fails.
_PROGRESS_PATCH_INTERVAL = 2.0

#: Set by app.py at startup. Kept as a module global so the webhook handler
#: stays a plain function and the graph is built exactly once.
_graph = None

#: Why the most recent webhook call was rejected, readable over HTTP.
#:
#: Railway interleaves stdout and stderr unpredictably, and a rejection that
#: logs nothing findable is indistinguishable from one that never happened.
#: This holds the answer in memory so it can be fetched directly instead of
#: hunted for. Never contains the token itself — only its public claims.
_last_rejection: dict | None = None


#: Every request to the webhook, counted before any authentication runs.
#:
#: Distinguishes "Google never called us" from "Google called and was rejected"
#: — the two have identical symptoms in Chat ("not responding") but completely
#: different causes. Counting before the auth check means no path can hide.
_requests_seen = 0
_last_request_at: str | None = None
_started_at: str | None = None


def traffic() -> dict:
    return {
        "requests_seen": _requests_seen,
        "last_request_at": _last_request_at,
        "process_started_at": _started_at,
    }


def last_rejection() -> dict | None:
    return _last_rejection


def _record_rejection(reason: str, **detail) -> None:
    global _last_rejection
    from datetime import datetime, timezone

    _last_rejection = {
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reason": reason,
        **detail,
    }
    logger.error("Chat request rejected: %s %s", reason, detail)


def attach_graph(graph) -> None:
    """Called once at startup. Also stamps process start, so a restart that
    resets the in-memory counters is visible rather than silently misleading."""
    global _graph, _started_at
    from datetime import datetime, timezone

    _graph = graph
    _started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")


# ── Authentication ─────────────────────────────────────────────────────────────


def verify_request(authorization: str | None) -> bool:
    """
    Confirm the bearer token was minted by Google Chat for *this* app.

    Returns False on any failure. There is deliberately no "allow on error"
    path: this endpoint is public, and everything behind it writes to a real
    CRM and texts real customers.
    """
    if not config.chat_verify_requests():
        logger.warning("Chat request verification is DISABLED — local testing only")
        return True

    if not authorization or not authorization.startswith("Bearer "):
        _record_rejection(
            "no_bearer_token",
            authorization_header=("absent" if not authorization
                                  else f"present but starts with {authorization[:12]!r}"),
        )
        return False

    audience = config.chat_audience()
    if not audience:
        _record_rejection("audience_not_configured")
        return False

    token = authorization.removeprefix("Bearer ").strip()
    request = google_requests.Request()

    try:
        if config.chat_audience_is_project_number():
            # Project-number audience: a self-signed JWT, so the signing certs
            # must be fetched from the issuer's own x509 endpoint.
            certs_url = (
                "https://www.googleapis.com/service_accounts/v1/metadata/x509/"
                + config.CHAT_ISSUER
            )
            claims = google_id_token.verify_token(
                token, request, audience, certs_url=certs_url
            )
            signer = claims.get("iss", "")
        else:
            # Endpoint-URL audience: a standard OpenID Connect ID token. The
            # signer is in `email`; `iss` is accounts.google.com for both
            # classic apps and Workspace add-ons, so it can't distinguish them.
            claims = google_id_token.verify_oauth2_token(token, request, audience)
            signer = claims.get("email", "")

        if config.issuer_is_google(signer):
            return True

        # A verified token from an unexpected signer. This MUST be recorded:
        # returning False silently here produced a 401 with no log line and no
        # recorded reason, which is what made this bug so hard to find.
        _record_rejection(
            "unexpected_issuer",
            token_issuer=signer or "(absent)",
            accepted=(sorted(config.chat_allowed_issuers())
                      or [config.CHAT_ISSUER, f"*{config.CHAT_ADDON_ISSUER_SUFFIX}"]),
            note="Workspace add-ons sign as service-<project>@gcp-sa-gsuiteaddons…",
        )
        return False

    except Exception as exc:
        _log_why_verification_failed(token, audience, exc)
        return False


def _log_why_verification_failed(token: str, expected_audience: str, exc: Exception) -> None:
    """
    Say precisely why a token was rejected.

    A bare 401 is close to useless here: the audience is configured in two
    places that must agree exactly (the Chat API console and this app's env),
    and a mismatch looks identical to a forged token. Decoding the claims
    without verifying grants nothing — the request is already rejected — but it
    turns "not responding" into a one-line diagnosis.
    """
    try:
        from google.auth import jwt

        claims = jwt.decode(token, verify=False)
    except Exception as decode_exc:
        _record_rejection(
            "token_not_a_readable_jwt",
            original_error=f"{type(exc).__name__}: {exc}",
            decode_error=f"{type(decode_exc).__name__}: {decode_exc}",
        )
        return

    actual = claims.get("aud")
    issuer = claims.get("iss") or claims.get("email")

    if actual != expected_audience:
        _record_rejection(
            "audience_mismatch",
            token_audience=actual,
            expected_audience=expected_audience,
            issuer=issuer,
            fix=f"Set GOOGLE_CHAT_AUDIENCE to {actual!r}",
        )
    else:
        _record_rejection(
            "signature_or_expiry",
            audience=actual,
            issuer=issuer,
            error=f"{type(exc).__name__}: {exc}",
        )


# ── Chat API client ────────────────────────────────────────────────────────────

_service = None
_service_lock = threading.Lock()


def chat_service():
    """Authenticated Chat API client, built once and reused."""
    global _service
    with _service_lock:
        if _service is None:
            creds_b64 = config.chat_credentials_b64()
            if not creds_b64:
                raise RuntimeError(
                    "No Chat service account. Set GOOGLE_CHAT_CREDENTIALS_B64 "
                    "(or GOOGLE_CREDENTIALS_B64) to a base64-encoded key."
                )
            import json

            info = json.loads(base64.b64decode(creds_b64).decode())
            creds = service_account.Credentials.from_service_account_info(
                info, scopes=config.CHAT_SCOPES
            )
            _service = build("chat", "v1", credentials=creds, cache_discovery=False)
        return _service


def post_message(space: str, text: str = "", card: dict | None = None,
                 thread: str | None = None) -> str:
    """Post into a space. Returns the new message's resource name, or ""."""
    body: dict[str, Any] = {}
    if text:
        body["text"] = text
    if card:
        body["cardsV2"] = [card]
    if thread:
        body["thread"] = {"name": thread}

    kwargs = {"parent": space, "body": body}
    if thread:
        # Without this, replying to a thread that Chat considers closed creates
        # a stray top-level message instead.
        kwargs["messageReplyOption"] = "REPLY_MESSAGE_FALLBACK_TO_NEW_THREAD"

    try:
        created = chat_service().spaces().messages().create(**kwargs).execute()
        return created.get("name", "")
    except Exception:
        logger.exception("Failed to post message to %s", space)
        return ""


def update_message(name: str, text: str = "", card: dict | None = None) -> None:
    """Rewrite a message we posted earlier. Best-effort."""
    if not name:
        return
    body: dict[str, Any] = {"text": text or ""}
    fields = ["text"]
    if card is not None:
        body["cardsV2"] = [card]
        fields.append("cardsV2")
    try:
        chat_service().spaces().messages().patch(
            name=name, updateMask=",".join(fields), body=body
        ).execute()
    except Exception:
        logger.debug("Could not update message %s", name, exc_info=True)


def download_attachment(attachment: dict) -> tuple[bytes, str] | None:
    """
    Fetch an uploaded image's bytes.

    Chat hands over a reference, not the file. Drive-hosted attachments use a
    different resource entirely and are skipped — asking the user to paste the
    image directly is clearer than half-supporting a path that needs separate
    Drive permissions.
    """
    ref = (attachment.get("attachmentDataRef") or {}).get("resourceName")
    if not ref:
        logger.info("Attachment has no uploaded-file reference; skipping")
        return None

    mime = attachment.get("contentType", "image/png")
    try:
        request = chat_service().media().download_media(resourceName=ref)
        buffer = io.BytesIO()
        downloader = MediaIoBaseDownload(buffer, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        return buffer.getvalue(), mime
    except Exception:
        logger.exception("Failed to download attachment %s", ref)
        return None


# ── Workspace add-on translation ──────────────────────────────────────────────
#
# This app is configured as a Google Workspace add-on, which speaks a different
# dialect from a classic Chat app in BOTH directions:
#
#   receives  {"chat": {"messagePayload": {...}}}   — no top-level "type"
#   expects   {"hostAppDataAction": {...}}          — not a bare message
#
# Rather than teach the whole file two dialects, translate at the edges: an
# add-on payload becomes the classic shape on the way in, and replies get
# wrapped on the way out. Everything between is unchanged — as are its tests.

#: Add-on payload key -> the classic event type it corresponds to.
_ADDON_PAYLOADS = {
    "messagePayload": "MESSAGE",
    "addedToSpacePayload": "ADDED_TO_SPACE",
    "removedFromSpacePayload": "REMOVED_FROM_SPACE",
    "buttonClickedPayload": "CARD_CLICKED",
    "appCommandPayload": "MESSAGE",
}

#: Shape of the last unrecognised payload, for diagnosis over HTTP. Keys only —
#: the values contain customer details.
_last_unknown_event: dict | None = None


def last_unknown_event() -> dict | None:
    return _last_unknown_event


def is_addon_event(raw: dict) -> bool:
    return "chat" in raw and isinstance(raw.get("chat"), dict)


def normalize_event(raw: dict) -> dict:
    """
    Return a classic-shaped Chat event, whichever dialect arrived.

    Classic payloads pass through untouched, so this stays correct if the app
    is ever rebuilt as a non-add-on Chat app.
    """
    global _last_unknown_event

    if not is_addon_event(raw):
        return raw

    chat = raw["chat"]

    for key, event_type in _ADDON_PAYLOADS.items():
        if payload := chat.get(key):
            event = {"type": event_type, **payload}
            # Button parameters live in commonEventObject for add-ons, not on
            # the payload itself.
            if event_type == "CARD_CLICKED":
                common = raw.get("commonEventObject") or {}
                event["_addon_parameters"] = common.get("parameters") or {}
                event["_addon_invoked_function"] = (
                    common.get("invokedFunction")
                    or payload.get("invokedFunction", "")
                )
            return event

    _last_unknown_event = {
        "chat_keys": sorted(chat.keys()),
        "top_level_keys": sorted(raw.keys()),
    }
    logger.error("Unrecognised add-on payload: %s", _last_unknown_event)
    return {"type": None}


def addon_reply(text: str = "", card: dict | None = None) -> dict:
    """Wrap a reply in the DataActions envelope an add-on must return."""
    message: dict[str, Any] = {}
    if text:
        message["text"] = text
    if card:
        message["cardsV2"] = [card]
    return {
        "hostAppDataAction": {
            "chatDataAction": {"createMessageAction": {"message": message}}
        }
    }


# ── Event -> graph translation ────────────────────────────────────────────────


def thread_id_for(event: dict) -> str:
    """
    Pick the LangGraph thread key for this event.

    This is the single most consequential line in the file. `thread_id` is how a
    paused booking is found again, so an unstable key means every answer to
    "what's the move date?" silently starts a fresh conversation and abandons
    the job in progress.

    Chat's own thread name is only stable in spaces that actually thread. In an
    unthreaded space — which is what direct messages usually are — each message
    gets its own thread name, so keying on it would break every multi-turn
    booking. There, the space itself is the conversation.
    """
    space = event.get("space") or {}
    space_name = space.get("name", "")

    threaded = space.get("spaceThreadingState") == "THREADED_MESSAGES"
    if threaded:
        thread = ((event.get("message") or {}).get("thread") or {}).get("name")
        if thread:
            return f"gchat:{thread}"

    return f"gchat:{space_name}"


def reply_thread_for(event: dict) -> str | None:
    """Thread to reply into, so answers land under the question."""
    space = event.get("space") or {}
    if space.get("spaceThreadingState") != "THREADED_MESSAGES":
        return None
    return ((event.get("message") or {}).get("thread") or {}).get("name")


def strip_mention(event: dict) -> str:
    """
    Remove the "@Ops Agent" prefix from the text.

    Chat includes the mention in the raw text; left in place it becomes noise in
    the model's context and, worse, can read as part of a customer's name.
    """
    message = event.get("message") or {}
    text = message.get("argumentText")
    if text is not None:
        return text.strip()
    return (message.get("text") or "").strip()


def build_graph_input(event: dict, is_paused: bool) -> object:
    """
    Turn a Chat event into graph input.

    A paused graph must receive `Command(resume=...)`; a fresh turn must receive
    a message. Sending the wrong one restarts the graph and discards the booking
    in progress — the same trap `server.py` exists to centralise, handled once
    here for the same reason.
    """
    text = strip_mention(event)

    if is_paused:
        # Resume values are plain strings for both interrupt types
        # (ask_missing and confirm). An image at this point would be a new job,
        # not an answer to the pending question, so it is ignored.
        return Command(resume=text)

    attachments = (event.get("message") or {}).get("attachment") or []
    images = []
    for attachment in attachments:
        if not attachment.get("contentType", "").startswith("image/"):
            continue
        if downloaded := download_attachment(attachment):
            raw, mime = downloaded
            b64 = base64.b64encode(raw).decode()
            images.append(
                {"type": "image_url",
                 "image_url": {"url": f"data:{mime};base64,{b64}"}}
            )

    if images:
        content = [{"type": "text", "text": text or "Book this job."}, *images]
        return {"messages": [HumanMessage(content=content)]}

    return {"messages": [HumanMessage(content=text)]}


# ── Rendering ──────────────────────────────────────────────────────────────────


def confirm_card(message: str) -> dict:
    """
    The approval gate, as a card with buttons.

    Buttons rather than typed text is a real safety gain, not decoration: this
    is the last moment before a contact, a booking, an invoice text and an email
    all fire. "yeah go ahead" is ambiguous to a parser; a button press is not.

    The values sent back are the same plain strings the graph already accepts,
    so `confirm.py` needs no knowledge that Chat exists.
    """
    return {
        "cardId": "confirm",
        "card": {
            "sections": [
                {"widgets": [{"textParagraph": {"text": _to_chat_markup(message)}}]},
                {
                    "widgets": [
                        {
                            "buttonList": {
                                "buttons": [
                                    {
                                        "text": "Book it",
                                        "onClick": {
                                            "action": {
                                                "function": "confirm_decision",
                                                "parameters": [
                                                    {"key": "decision", "value": "yes"}
                                                ],
                                            }
                                        },
                                    },
                                    {
                                        "text": "Cancel",
                                        "onClick": {
                                            "action": {
                                                "function": "confirm_decision",
                                                "parameters": [
                                                    {"key": "decision", "value": "no"}
                                                ],
                                            }
                                        },
                                    },
                                ]
                            }
                        }
                    ]
                },
                {
                    "widgets": [
                        {
                            "textParagraph": {
                                "text": (
                                    "<i>To change something, just reply — "
                                    "e.g. “arrival 10-11am” or “make it 4 movers”.</i>"
                                )
                            }
                        }
                    ]
                },
            ]
        },
    }


def _to_chat_markup(text: str) -> str:
    """
    Chat cards render a small HTML subset, not plain text — newlines collapse
    and the confirmation summary's alignment is lost. Convert explicitly.
    """
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")


# ── Running the graph ─────────────────────────────────────────────────────────


def _is_paused(thread_id: str) -> bool:
    cfg = {"configurable": {"thread_id": thread_id}}
    try:
        snapshot = _graph.get_state(cfg)
        return bool(snapshot.next)
    except Exception:
        logger.exception("Could not read graph state for %s", thread_id)
        return False


def _pending_interrupt(thread_id: str) -> dict | None:
    cfg = {"configurable": {"thread_id": thread_id}}
    try:
        state = _graph.get_state(cfg)
        for task in state.tasks or ():
            if task.interrupts:
                return task.interrupts[0].value
    except Exception:
        logger.exception("Could not read interrupts for %s", thread_id)
    return None


def run_graph(event: dict, decision: str | None = None) -> tuple[str, dict | None]:
    """
    Run the graph for one event and return (text, card).

    Knows nothing about how the answer gets delivered — the add-on path returns
    it inline, the classic path posts it via the API. Never raises: a failure
    becomes a message the user can actually read.
    """
    thread_id = thread_id_for(event)
    cfg = {"configurable": {"thread_id": thread_id}}

    try:
        paused = _is_paused(thread_id)

        if decision is not None:
            # A button press. Only meaningful against a paused graph; if the
            # booking already resolved (clicked twice, or answered in text
            # first) this must not start a new run.
            if not paused:
                return "That booking is already resolved.", None
            graph_input: object = Command(resume=decision)
        else:
            graph_input = build_graph_input(event, paused)

        _graph.invoke(graph_input, cfg)

        if interrupt_value := _pending_interrupt(thread_id):
            text = interrupt_value.get("message", "")
            if interrupt_value.get("type") == "confirm":
                return "", confirm_card(text)
            return text, None

        final = _graph.get_state(cfg).values
        messages = final.get("messages") or []
        return (str(messages[-1].content) if messages else "(no response)"), None

    except Exception as exc:
        logger.exception("Graph run failed for %s", thread_id)
        return (
            "Something broke and the booking did *not* go through.\n\n"
            f"`{type(exc).__name__}: {exc}`"
        ), None


def process_event(event: dict, decision: str | None = None) -> None:
    """
    Run the graph for one Chat event and post the result. Runs in a background
    thread — the webhook has already returned by now.

    Never raises: an uncaught error here would leave the user staring at a
    placeholder that never resolves, with no clue why.
    """
    space = (event.get("space") or {}).get("name", "")
    thread_id = thread_id_for(event)
    reply_thread = reply_thread_for(event)
    cfg = {"configurable": {"thread_id": thread_id}}

    placeholder = post_message(space, "_Working on it…_", thread=reply_thread)

    try:
        paused = _is_paused(thread_id)

        if decision is not None:
            # A button press. Only meaningful against a paused graph; if the
            # booking already resolved (someone clicked twice, or replied in
            # text first) this must not start a new run.
            if not paused:
                update_message(placeholder, "That booking is already resolved.")
                return
            graph_input: object = Command(resume=decision)
        else:
            graph_input = build_graph_input(event, paused)

        last_patch = 0.0
        for mode, chunk in _graph.stream(
            graph_input, cfg, stream_mode=["custom", "values"]
        ):
            if mode != "custom" or not isinstance(chunk, dict):
                continue
            if chunk.get("type") != "progress":
                continue
            now = time.monotonic()
            if now - last_patch < _PROGRESS_PATCH_INTERVAL:
                continue
            last_patch = now
            update_message(placeholder, f"_{chunk.get('text', 'Working…')}_")

        if interrupt_value := _pending_interrupt(thread_id):
            text = interrupt_value.get("message", "")
            if interrupt_value.get("type") == "confirm":
                update_message(placeholder, "", card=confirm_card(text))
            else:
                update_message(placeholder, text)
            return

        final = _graph.get_state(cfg).values
        messages = final.get("messages") or []
        reply = messages[-1].content if messages else "(no response)"
        update_message(placeholder, str(reply))

    except Exception as exc:
        logger.exception("Graph run failed for %s", thread_id)
        update_message(
            placeholder,
            f"Something broke and the booking did *not* go through.\n\n"
            f"`{type(exc).__name__}: {exc}`",
        )


# ── Webhook ────────────────────────────────────────────────────────────────────

#: Overridable so tests can run the handler synchronously.
_spawn: Callable[..., None] = lambda fn, *a, **kw: threading.Thread(
    target=fn, args=a, kwargs=kw, daemon=True
).start()


@router.post("/google-chat")
async def google_chat_webhook(request: Request):
    global _requests_seen, _last_request_at
    from datetime import datetime, timezone

    # Count first, before anything can reject or raise. This is the one number
    # that separates "Google never called" from "Google called and failed".
    _requests_seen += 1
    _last_request_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    if not verify_request(request.headers.get("authorization")):
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    raw = await request.json()
    addon = is_addon_event(raw)
    event = normalize_event(raw)
    event_type = event.get("type")

    def reply(text: str = "", card: dict | None = None) -> dict:
        """Shape a synchronous reply for whichever dialect is in use."""
        if addon:
            return addon_reply(text, card)
        body: dict[str, Any] = {}
        if text:
            body["text"] = text
        if card:
            body["cardsV2"] = [card]
        return body

    if event_type == "ADDED_TO_SPACE":
        return reply(
            "Hi — I'm the Splendid Moving ops agent.\n\n"
            "Ask me about the calendar (*how many jobs did we have last "
            "month?*), or send me a screenshot of a customer enquiry and "
            "I'll book it. I'll always show you exactly what I'm about to "
            "do before anything happens."
        )

    if event_type == "REMOVED_FROM_SPACE":
        return {}

    if event_type in ("MESSAGE", "CARD_CLICKED"):
        decision = None
        if event_type == "CARD_CLICKED":
            decision = _button_decision(event)
            if decision is None:
                return {}

        if addon:
            # Add-ons must answer inline: the app's identity belongs to
            # Google's add-on service account, not to the service account we
            # hold, so posting asynchronously through the Chat API is not
            # available to us here. Chat's 30-second deadline therefore applies
            # to the whole run — see the note in the module docstring.
            text, card = run_graph(event, decision)
            return reply(text, card)

        _spawn(process_event, event, decision)
        return {}

    logger.info("Ignoring Chat event type %r", event_type)
    return {}


def _button_decision(event: dict) -> str | None:
    """Pull the clicked button's value out of either dialect."""
    addon_params = event.get("_addon_parameters")
    if addon_params is not None:
        function = event.get("_addon_invoked_function", "")
        if function != "confirm_decision":
            return None
        return addon_params.get("decision", "no")

    function = (event.get("common", {}) or {}).get("invokedFunction") or (
        event.get("action") or {}
    ).get("actionMethodName")
    if function != "confirm_decision":
        return None
    params = {
        p.get("key"): p.get("value")
        for p in (event.get("action") or {}).get("parameters") or []
    }
    return params.get("decision", "no")
