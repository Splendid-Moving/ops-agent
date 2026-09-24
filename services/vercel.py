"""
Vercel Web Analytics API client. Read-only.

Two endpoints: `count` for plain totals and `aggregate` for breakdowns. The
split matters because only `aggregate` is capped by the plan's reporting
window — on Hobby, the latest 31 days.
"""

import logging
from typing import Any

import requests
from langsmith import traceable

from services import config

logger = logging.getLogger(__name__)

BASE_URL = "https://api.vercel.com/v1/query/web-analytics"
_TIMEOUT = 20

#: How far back a breakdown can reach on the Hobby plan. Verified against the
#: live account: past this the API answers
#: "the hobby plan only grants access to the latest 31 days of data".
#: `count` has no such limit, so lifetime totals are still available.
AGGREGATE_WINDOW_DAYS = 31

#: Dimensions this client accepts, mapped to Vercel's names.
DIMENSIONS = {
    "day": "day",
    "page": "requestPath",
    "referrer": "referrerHostname",
    "country": "country",
    "device": "deviceType",
    "browser": "browserName",
}


class VercelError(RuntimeError):
    """A Vercel API call failed. Carries the status and body, never the token."""

    def __init__(self, message: str, status: int | None = None, body: str = ""):
        detail = f" [{status}] {body[:300]}" if (status or body) else ""
        super().__init__(f"{message}{detail}")
        self.status = status
        self.body = body


def _headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {config.vercel_token()}"}


def _params(**extra: Any) -> dict[str, Any]:
    params = {"projectId": config.website_project_id()}
    if team := config.vercel_team_id():
        params["teamId"] = team
    params.update({k: v for k, v in extra.items() if v is not None})
    return params


def _get(path: str, **extra: Any) -> dict[str, Any]:
    if not config.vercel_token():
        raise VercelError("VERCEL_TOKEN is not set.")
    if not config.website_project_id():
        raise VercelError("WEBSITE_PROJECT_ID is not set.")

    try:
        resp = requests.get(
            f"{BASE_URL}/{path}", headers=_headers(), params=_params(**extra), timeout=_TIMEOUT
        )
    except Exception as exc:
        # The token is in the request headers, so the exception can carry it.
        # Re-raise with our own message rather than the original.
        raise VercelError(f"Vercel unreachable: {type(exc).__name__}") from None

    if not resp.ok:
        raise VercelError(f"Vercel returned {resp.status_code}", resp.status_code, resp.text)
    return resp.json()


@traceable(run_type="tool", name="vercel.count_visits")
def count_visits(since: str | None = None, until: str | None = None) -> dict[str, int]:
    """Total pageviews and visitors. No reporting-window limit."""
    data = _get("visits/count", since=since, until=until).get("data", {})
    return {"pageviews": data.get("pageviews", 0), "visitors": data.get("visitors", 0)}


@traceable(run_type="tool", name="vercel.aggregate_visits")
def aggregate_visits(since: str, until: str, by: str, limit: int | None = None) -> list[dict]:
    """Rows grouped by one dimension. Capped at AGGREGATE_WINDOW_DAYS on Hobby."""
    return _get("visits/aggregate", since=since, until=until, by=by, limit=limit).get("data", [])
