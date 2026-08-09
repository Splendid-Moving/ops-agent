"""
Google Maps Distance Matrix. Feeds the `Distance:` line on the calendar event.

Ported from ghl_calendar_sync/app.py. Distance is informational for the crew,
so every failure path returns "" rather than raising — a missing distance must
never block a booking.
"""

import logging

import requests

from services import config

logger = logging.getLogger(__name__)

_ENDPOINT = "https://maps.googleapis.com/maps/api/distancematrix/json"
_METERS_PER_MILE = 1609.344


def _leg_miles(origin: str, destination: str) -> float | None:
    try:
        resp = requests.get(
            _ENDPOINT,
            params={
                "origins": origin,
                "destinations": destination,
                "mode": "driving",
                "units": "imperial",
                "key": config.maps_api_key(),
            },
            timeout=10,
        )
        data = resp.json()
        rows = data.get("rows") or []
        if not rows:
            logger.warning("Distance Matrix returned no rows: %s", data.get("status"))
            return None
        element = (rows[0].get("elements") or [{}])[0]
        if element.get("status") != "OK":
            logger.warning("Distance Matrix element status: %s", element.get("status"))
            return None
        return element["distance"]["value"] / _METERS_PER_MILE
    except Exception as exc:
        logger.error("Distance Matrix error: %s", exc)
        return None


def get_distance(from_address: str, to_address: str, extra_stop: str | None = None) -> str:
    """
    Formatted trip distance, e.g. '12.4 miles'. Returns '' on any failure.
    With an extra stop, sums both legs.
    """
    if not config.maps_api_key() or not from_address or not to_address:
        return ""

    if extra_stop:
        leg1 = _leg_miles(from_address, extra_stop)
        leg2 = _leg_miles(extra_stop, to_address)
        if leg1 is None or leg2 is None:
            return ""
        return f"{leg1 + leg2:.1f} miles"

    miles = _leg_miles(from_address, to_address)
    return f"{miles:.1f} miles" if miles is not None else ""
