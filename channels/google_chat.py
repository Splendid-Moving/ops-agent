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

── Two dialects ───────────────────────────────────────────────────────────────

This app is registered as a Google Workspace ADD-ON, which speaks differently
from a classic Chat app in both directions:

  receives  {"chat": {"messagePayload": ...}}   — no top-level "type"
  replies   {"hostAppDataAction": ...}          — not a bare message

`normalize_event` and `addon_reply` translate at the edges, so everything in
between — and its tests — is written against one shape. Classic payloads pass
through untouched, so the file still works if the app is ever re-registered as
a classic Chat app.

── The 30-second problem ──────────────────────────────────────────────────────

Google Chat abandons a webhook that takes longer than 30 seconds.

A classic Chat app escapes this by returning 200 immediately and posting the
answer later through the Chat REST API — that is what `process_event`,
`post_message` and `update_message` below are for, and they are still used on
the classic path.

An add-on cannot: its identity is Google's own gcp-sa-gsuiteaddons account, not
the service account we hold, so we cannot post on its behalf. The add-on path
therefore answers INLINE, inside the HTTP response, and the whole run has to
fit inside 30 seconds. Runs are timed (see `_record_run`) because exceeding the
deadline looks identical to a crash from the user's side while meaning the
opposite: the booking completed.

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


#: How long recent runs took, and anything that blew up.
#:
#: Chat abandons a webhook at 30 seconds and shows "unable to process your
#: request" — the SAME message it shows for a crash. Timing is the only way to
#: tell them apart, and it matters enormously: a crash did nothing, while a
#: timeout means the booking went through and the user was told it failed.
_recent_runs: list[dict] = []
_last_error: dict | None = None

#: Chat's hard deadline. A run that gets near it has already lost the race.
CHAT_DEADLINE_SECONDS = 30


def _record_run(kind: str, seconds: float) -> None:
    _recent_runs.append({
        "kind": kind,
        "seconds": round(seconds, 1),
        "over_deadline": seconds >= CHAT_DEADLINE_SECONDS,
    })
    del _recent_runs[:-10]
    if seconds >= CHAT_DEADLINE_SECONDS:
        logger.error(
            "Run took %.1fs — past Chat's %ss deadline. The user saw an error, "
            "but this work COMPLETED.", seconds, CHAT_DEADLINE_SECONDS,
        )


def _record_error(exc: Exception) -> None:
    global _last_error
    import traceback
    from datetime import datetime, timezone

    _last_error = {
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "error": f"{type(exc).__name__}: {exc}",
        "where": traceback.format_exc().strip().splitlines()[-3:],
    }
    logger.exception("Webhook failed")


def last_error() -> dict | None:
    return _last_error


def traffic() -> dict:
    return {
        "requests_seen": _requests_seen,
        "last_request_at": _last_request_at,
        "process_started_at": _started_at,
        "recent_runs": _recent_runs,
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

    Chat's thread name is not stable across a conversation. Sending a fresh
    message — rather than explicitly replying inside an existing thread —
    starts a NEW thread with a new name, even in a one-to-one chat. Keying on
    it means the second message never finds the booking the first one started.

    That failure is silent and looks like the agent "forgetting": it answers,
    but with only the fields from the latest message, having discarded
    everything read from the screenshot.

    So a direct message is always ONE conversation, keyed on the space,
    regardless of what threading state Chat reports for it. Only genuine
    multi-person spaces key on the thread, where separate threads really are
    separate conversations.
    """
    space = event.get("space") or {}
    space_name = space.get("name", "")

    space_type = space.get("spaceType") or space.get("type") or ""
    is_dm = space_type.upper() in ("DIRECT_MESSAGE", "DM")

    if not is_dm and space.get("spaceThreadingState") == "THREADED_MESSAGES":
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


def confirm_card(message: str, with_buttons: bool = True) -> dict:
    """
    The approval gate, as a card with buttons.

    Buttons rather than typed text is a real safety gain, not decoration: this
    is the last moment before a contact, a booking, an invoice text and an email
    all fire. "yeah go ahead" is ambiguous to a parser; a button press is not.

    The values sent back are the same plain strings the graph already accepts,
    so `confirm.py` needs no knowledge that Chat exists.
    """
    sections: list[dict] = [
        {"widgets": [{"textParagraph": {"text": _to_chat_markup(message)}}]}
    ]

    if with_buttons:
        sections.append(
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
                }
        )

    # No instruction line here: confirm.py already ends the summary with
    # "Reply 'yes' to go ahead, 'no' to cancel, or tell me what to change".
    # Adding it again printed the same sentence twice on every booking.

    return {"cardId": "confirm", "card": {"sections": sections}}


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


#: Typed phrases that wipe the conversation and any half-finished booking.
#:
#: Needed because the conversation is keyed to the Chat *space*, so deleting
#: the chat in Google Chat does not clear it — the next message would resume a
#: booking the user thought was long gone.
_RESET_WORDS = {
    "reset", "/reset", "start over", "start again", "clear",
    "clear chat", "new booking", "cancel booking", "forget it", "nevermind",
}


#: Slash commands, matched on the command word alone.
#:
#: These work whether or not they are registered in the Chat API console: a
#: registered command arrives with the text stripped out, and an unregistered
#: one simply arrives as a message beginning with "/". Both are handled, so the
#: console registration only buys the autocomplete menu.
_SLASH_RESET = {"/clear", "/reset", "/new", "/restart", "/cancel"}
_SLASH_HELP = {"/help", "/commands", "/?"}

HELP_TEXT = """*Splendid Moving ops agent*

*Ask about jobs*
  _how many jobs did we have last month?_
  _what's on the calendar Friday?_

*Book a job*
  Send a screenshot of the customer's details. I'll read it, ask for
  anything missing, then show you exactly what I'll create and wait for
  you to reply *yes*.

*Commands*
  `/clear`  — forget this conversation and start fresh
  `/help`   — this message

Nothing is created until you approve it."""


def slash_command(event: dict) -> str:
    """The leading /command of this message, lowercased, or ''."""
    message = event.get("message") or {}
    # argumentText has the command removed when it is registered, so the raw
    # text is the reliable place to look.
    for candidate in (message.get("text"), message.get("argumentText")):
        text = (candidate or "").strip()
        # Strip a leading @mention, which Chat includes in `text`.
        if text.startswith("@"):
            parts = text.split(None, 1)
            text = parts[1].strip() if len(parts) > 1 else ""
        if text.startswith("/"):
            return text.split()[0].lower()
    return ""


def is_reset_request(text: str) -> bool:
    return text.strip().lower().strip(".!") in _RESET_WORDS


def reset_thread(thread_id: str) -> bool:
    """
    Wipe a conversation, including any booking paused mid-question.

    Nothing already created in GoHighLevel or the calendar is touched — this
    only clears what the agent remembers.
    """
    checkpointer = getattr(_graph, "checkpointer", None)
    if checkpointer is None:
        return False
    try:
        checkpointer.delete_thread(thread_id)
        logger.info("Reset conversation %s", thread_id)
        return True
    except Exception:
        logger.exception("Could not reset %s", thread_id)
        return False


def run_graph(event: dict, decision: str | None = None,
              buttons: bool = True) -> tuple[str, dict | None]:
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
        # Logged every turn: a thread_id that changes between messages is the
        # one failure that looks like the agent forgetting rather than erroring.
        logger.info("thread=%s paused=%s", thread_id, paused)

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
                return "", confirm_card(text, with_buttons=buttons)
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

    try:
        raw = await request.json()
    except Exception:
        logger.exception("Chat webhook received unparseable JSON")
        return {}

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

    if event_type == "MESSAGE" and (command := slash_command(event)) in _SLASH_HELP and command:
        return reply(HELP_TEXT)

    if event_type == "MESSAGE" and (
        slash_command(event) in _SLASH_RESET or is_reset_request(strip_mention(event))
    ):
        # Handled before the graph runs: the whole point is to escape a
        # conversation that may be stuck mid-booking.
        thread_id = thread_id_for(event)
        if reset_thread(thread_id):
            return reply(
                "Cleared — we're starting fresh. Nothing in GoHighLevel or the "
                "calendar was changed; I've only forgotten the conversation."
            )
        return reply("I couldn't clear that. Try again in a moment.")

    if event_type in ("MESSAGE", "CARD_CLICKED"):
        decision = None
        if event_type == "CARD_CLICKED":
            decision = _button_decision(event)
            if decision is None:
                # Must not return a bare {} here: Chat renders that as "Ops
                # Agent is unable to process your request", which reads like a
                # crash to someone who just pressed "Book it" on a real job.
                return reply(
                    "I couldn't read that button press. Reply *yes* to go "
                    "ahead or *no* to cancel — nothing has been created."
                )

        if addon:
            # Add-ons must answer inline, so Chat's 30-second deadline applies
            # to the whole run. Timing is recorded because exceeding it looks
            # identical to a crash from the user's side, while meaning the
            # opposite: the work completed.
            started = time.monotonic()
            try:
                text, card = run_graph(event, decision, buttons=False)
            except Exception as exc:
                _record_error(exc)
                return reply(
                    "Something went wrong and I could not finish.\n\n"
                    f"`{type(exc).__name__}: {exc}`\n\n"
                    "Check GoHighLevel and the calendar before retrying — part "
                    "of this may already have gone through."
                )
            finally:
                _record_run("approval" if decision else "message",
                            time.monotonic() - started)
            return reply(text, card)

        _spawn(process_event, event, decision)
        return {}

    logger.info("Ignoring Chat event type %r", event_type)
    return {}


def _button_decision(event: dict) -> str | None:
    """
    Pull the clicked button's value out of whichever dialect arrived.

    Deliberately forgiving about *where* the value sits. Google moved both the
    function name and the parameters between the classic and add-on formats,
    and a button that silently does nothing is worse than one that guesses:
    the booking is already summarised and the user has pressed "Book it".

    Returns None only when this genuinely isn't our button — and records the
    payload shape when it looks like ours but can't be read, so the format can
    be corrected rather than guessed at again.
    """
    global _last_unknown_event

    # Function name — add-on puts it in commonEventObject, classic in
    # common.invokedFunction or action.actionMethodName.
    function = (
        event.get("_addon_invoked_function")
        or (event.get("common") or {}).get("invokedFunction")
        or (event.get("action") or {}).get("actionMethodName")
        or ""
    )

    # Parameters — a plain map for add-ons, a list of key/value pairs classically.
    params: dict = {}
    if isinstance(event.get("_addon_parameters"), dict):
        params = dict(event["_addon_parameters"])
    if not params:
        for pair in (event.get("action") or {}).get("parameters") or []:
            if isinstance(pair, dict) and "key" in pair:
                params[pair["key"]] = pair.get("value")
    if not params and isinstance((event.get("common") or {}).get("parameters"), dict):
        params = dict(event["common"]["parameters"])

    decision = params.get("decision")

    if decision in ("yes", "no"):
        return decision

    if function and function != "confirm_decision":
        return None      # somebody else's button

    _last_unknown_event = {
        "what": "card_click_without_a_readable_decision",
        "invoked_function": function or "(absent)",
        "parameters_found": params or "(none)",
        "event_keys": sorted(event.keys()),
    }
    logger.error("Unreadable card click: %s", _last_unknown_event)
    return None
