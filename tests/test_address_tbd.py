"""
"The customer will send the drop-off later."

A job is sometimes booked before an address exists. Without a way to say so,
the checklist could only re-ask — it did, four times, to a dispatcher who had
already answered — and when they finally typed a city and a ZIP to make it stop,
"yorba Linda 92886" passed the validator (it has digits) and reached the confirm
gate as a drop-off.

The dispatcher makes the call that an address can wait. TBD is how the agent
records that call. Addresses only: everything else on the checklist is needed
for a side effect to fire.
"""

from datetime import datetime, timedelta

import pytest

from agent.nodes import ask_missing as am
from agent.nodes import confirm as confirm_module
from agent.nodes import resolve_addresses as ra
from schemas import checklist as cl
from schemas import email_template
from services.calendar import LA_TZ


def _complete(**overrides) -> dict:
    base = {
        "full_name": "Daman Singh", "email": "daman@example.com", "phone": "(909) 538-0959",
        "pickup_address": "11847 Kiowa Ave, Los Angeles CA 90049",
        "dropoff_address": "TBD",
        "move_date": (datetime.now(LA_TZ) + timedelta(days=10)).strftime("%m/%d/%Y"),
        "arrival_time": "8pm", "movers": "2", "job_notes": "- $60 gas fee", "notes_asked": True,
    }
    base.update(overrides)
    return base


# ── The checklist accepts a deferred address ───────────────────────────────────

@pytest.mark.parametrize("spelling", ["TBD", "tbd", " Tbd "])
def test_tbd_satisfies_the_address_requirement(spelling):
    assert cl.valid_address(spelling) is None


def test_a_booking_with_a_tbd_dropoff_is_complete():
    assert cl.evaluate(_complete()).is_complete


def test_a_tbd_dropoff_still_means_it_is_a_move_not_labor():
    """Two addresses, one of them pending, is still a move between two places."""
    assert cl.infer_is_labor("11847 Kiowa Ave", "TBD", None) is False


def test_tbd_is_canonicalised_by_the_normalizer():
    assert cl.normalize_address("tbd") == "TBD"
    assert cl.normalize_address("11847 Kiowa Ave\nLos Angeles CA 90049") == \
        "11847 Kiowa Ave, Los Angeles CA 90049"


# ── The validator no longer accepts a city and a ZIP ──────────────────────────

@pytest.mark.parametrize("not_an_address", [
    "yorba Linda",
    "yorba Linda 92886",          # the live case — digits, but only the ZIP
    "Los Angeles CA",
    "somewhere in Pasadena",
])
def test_a_place_name_is_not_an_address(not_an_address):
    err = cl.valid_address(not_an_address)
    assert err is not None
    assert "street number" in err
    assert "TBD" in err            # tells them the way out


@pytest.mark.parametrize("real", [
    "11847 Kiowa Ave, Los Angeles CA 90049",
    "436 Fairview Ave #32",
    "12B Main St",
    "1 Wilshire Blvd",
])
def test_real_addresses_still_pass(real):
    assert cl.valid_address(real) is None


# ── Nothing downstream treats TBD as a place ───────────────────────────────────

def test_resolve_addresses_does_not_send_tbd_to_google(monkeypatch):
    sent = {}
    monkeypatch.setattr(ra.address_service, "validate_many", lambda c: sent.update(c) or {})
    out = ra.resolve_addresses({"intake": _complete()})["intake"]
    assert "dropoff_address" not in sent
    assert out["address_status"]["dropoff_address"] == "tbd"


def test_resolve_addresses_handles_all_addresses_tbd(monkeypatch):
    called = []
    monkeypatch.setattr(ra.address_service, "validate_many", lambda c: called.append(c) or {})
    out = ra.resolve_addresses({"intake": _complete(pickup_address="TBD")})["intake"]
    assert called == []
    assert out["address_status"] == {"pickup_address": "tbd", "dropoff_address": "tbd"}


def test_the_summary_shows_it_as_a_decision():
    text = confirm_module._summary(_complete(), {}, None)
    assert "Drop-off    TBD — customer to confirm" in text


def test_the_customer_email_never_shows_the_word_tbd():
    html = email_template.html(_complete())
    assert "TBD" not in html
    assert "To be confirmed" in html


def test_the_calendar_skips_distance_for_a_tbd_address(monkeypatch):
    from agent.nodes import actions

    measured = []
    monkeypatch.setattr(actions.maps, "get_distance", lambda *a, **k: measured.append(a) or "1 mile")
    monkeypatch.setattr(actions.calendar, "create_event", lambda **k: {"event_id": "e", "description": k["description"]})
    result = actions.act_calendar_event.__wrapped__({"intake": _complete()})
    assert measured == []
    assert "To: TBD" in result["description"]
    assert "Distance: " not in result["description"] or "Distance: \n" in result["description"]


# ── The dispatcher hears that "later" was understood ───────────────────────────

def test_saying_later_is_acknowledged(monkeypatch):
    monkeypatch.setattr(am, "interrupt", lambda _: "drop off later")
    monkeypatch.setattr(am, "_parse", lambda *a, **k: am.ParsedReply(dropoff_address="TBD"))
    out = am.ask_missing({"intake": {"full_name": "Daman Singh"}})["intake"]
    assert out["dropoff_address"] == "TBD"
    assert "to be confirmed" in out["_aside"]
    assert "before the move" in out["_aside"]


def test_a_real_address_is_not_acknowledged_as_deferred(monkeypatch):
    monkeypatch.setattr(am, "interrupt", lambda _: "11847 Kiowa Ave")
    monkeypatch.setattr(am, "_parse", lambda *a, **k: am.ParsedReply(dropoff_address="11847 Kiowa Ave"))
    out = am.ask_missing({"intake": {"full_name": "Daman Singh"}})["intake"]
    assert "_aside" not in out


def test_the_parser_is_told_a_city_is_not_a_stand_in():
    text = am._parse_prompt(["What's the drop-off address?"], editing=False, current={})
    assert '"Yorba Linda" is not an address' in text
    assert 'literal string "TBD"' in text


def test_a_tbd_address_is_not_listed_as_unverified():
    """The Drop-off line already says TBD; a warning underneath about the same
    thing reads as the agent objecting to a decision it was just given."""
    intake = {"address_status": {"pickup_address": "confirmed", "dropoff_address": "tbd"}}
    assert ra.unresolved_addresses(intake) == {}
