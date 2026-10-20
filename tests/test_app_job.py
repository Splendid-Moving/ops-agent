"""
The job in the crew's AppSheet app.

What can go wrong here is quiet: a job that lands with the wrong date format
never shows on the crew's list, a "TBD" address gets geocoded to somewhere in
Kansas, and a retried request creates the job twice. These pin each of those.
"""

from datetime import datetime, timedelta

import pytest

from agent.nodes import actions
from agent.state import ACTION_APP_JOB, ACTION_CONTACT, merge_ledger, new_ledger
from services import config, jobs_app
from services.calendar import LA_TZ


@pytest.fixture
def intake():
    return {
        "full_name": "Sarah Chen",
        "email": " Sarah@Example.com ",
        "phone": "+1(818)555-0142",
        "pickup_address": "412 N Maple St, Burbank CA 91505",
        "dropoff_address": "1830 Pine St, Glendale CA 91206",
        "move_date": (datetime.now(LA_TZ) + timedelta(days=10)).strftime("%m/%d/%Y"),
        "arrival_time": "8-9am",
        "movers": "3",
        "move_size": "2 Bedroom",
        "source": "Yelp",
        "job_notes": "Third floor walkup",
    }


@pytest.fixture
def connected(monkeypatch):
    monkeypatch.setenv("JOBS_APP_WEBHOOK_URL", "https://script.google.com/macros/s/x/exec")
    monkeypatch.setenv("JOBS_APP_WEBHOOK_SECRET", "s3cret")


@pytest.fixture
def contact_done():
    return merge_ledger(
        new_ledger(), {ACTION_CONTACT: {"status": "success", "result": {"contact_id": "c1"}}}
    )


# ── The payload ────────────────────────────────────────────────────────────────

def test_payload_uses_the_shapes_the_database_expects(intake):
    p = actions.app_job_payload(intake, "c1")

    assert p["move_date"] == datetime.strptime(intake["move_date"], "%m/%d/%Y").strftime("%Y-%m-%d")
    assert p["scheduled_time"] == "08:00", "start of the window, 24-hour"
    assert p["move_type"] == "Local"
    assert p["phone"] == "8185550142", "last 10 digits, as matched on the other side"
    assert p["email"] == "sarah@example.com"
    assert p["ghl_contact_id"] == "c1"
    assert (p["movers_required"], p["cash_rate_per_hour"], p["card_rate_per_hour"]) == (3, 145, 155)
    assert p["lead_source"] == "Yelp"


def test_the_full_window_survives_in_the_notes(intake):
    """The app stores one start time; the crew still needs to know it was a window."""
    notes = actions.app_job_payload(intake)["job_notes"]
    assert "Arrival window: 8-9am" in notes
    assert "Third floor walkup" in notes


def test_afternoon_window_converts_to_24_hour(intake):
    intake["arrival_time"] = "2-4pm"
    assert actions.app_job_payload(intake)["scheduled_time"] == "14:00"


def test_fingerprint_matches_the_calendar_one(intake):
    """One identity for the job everywhere — that is what makes a retry safe."""
    assert actions.app_job_payload(intake)["fingerprint"] == actions.job_fingerprint(intake)


def test_labor_only_has_no_dropoff_and_says_so(intake):
    intake["is_labor"] = True
    p = actions.app_job_payload(intake)
    assert p["move_type"] == "Labor"
    assert p["destination_address"] == ""
    assert "LABOR ONLY" in p["job_notes"]
    assert "Drop-off address TBD" not in p["job_notes"], "labor has no drop-off to be TBD"


def test_tbd_address_goes_in_blank_and_is_flagged(intake):
    """The app geocodes every address. 'TBD' is not a place."""
    intake["dropoff_address"] = "TBD"
    p = actions.app_job_payload(intake)
    assert p["destination_address"] == ""
    assert "Drop-off address TBD" in p["job_notes"]


def test_unsupported_crew_size_sends_no_rates(intake):
    intake["movers"] = "5"
    p = actions.app_job_payload(intake)
    assert p["movers_required"] is None and p["cash_rate_per_hour"] is None


def test_unparseable_date_refuses_rather_than_guessing(intake):
    intake["move_date"] = "sometime next week"
    with pytest.raises(ValueError):
        actions.app_job_payload(intake)


# ── The action ─────────────────────────────────────────────────────────────────

def test_not_connected_is_a_skip_not_a_failure(intake, contact_done, monkeypatch):
    """An unconnected deployment must book exactly as it did before."""
    monkeypatch.delenv("JOBS_APP_WEBHOOK_URL", raising=False)
    monkeypatch.delenv("JOBS_APP_WEBHOOK_SECRET", raising=False)

    entry = actions.act_app_job({"intake": intake, "ledger": contact_done})["ledger"][ACTION_APP_JOB]
    assert entry["status"] == "skipped"
    assert "not connected" in entry["error"]


def test_connected_sends_the_payload_once(intake, contact_done, connected, monkeypatch):
    calls = []
    monkeypatch.setattr(jobs_app, "create_job",
                        lambda payload: calls.append(payload) or {"job_id": "j1", "created": True})

    state = {"intake": intake, "ledger": contact_done}
    first = actions.act_app_job(state)
    assert first["ledger"][ACTION_APP_JOB]["status"] == "success"
    assert calls[0]["ghl_contact_id"] == "c1"

    state["ledger"] = merge_ledger(state["ledger"], first["ledger"])
    assert actions.act_app_job(state) == {}, "already succeeded — must not fire again"
    assert len(calls) == 1


def test_a_refusal_is_recorded_as_a_failure_with_the_reason(intake, contact_done, connected, monkeypatch):
    def refuse(payload):
        raise jobs_app.JobsAppError("Ops app refused the job: Unauthorized")

    monkeypatch.setattr(jobs_app, "create_job", refuse)
    entry = actions.act_app_job({"intake": intake, "ledger": contact_done})["ledger"][ACTION_APP_JOB]
    assert entry["status"] == "failed"
    assert "Unauthorized" in entry["error"]


def test_report_shows_the_job_id(intake, monkeypatch):
    monkeypatch.setenv("DRY_RUN", "false")
    ledger = merge_ledger(new_ledger(), {
        ACTION_APP_JOB: {"status": "success", "result": {"job_id": "a1b2c3d4"}},
    })
    text = actions.report({"intake": intake, "ledger": ledger})["messages"][0].content
    assert "Job in the crew app" in text and "a1b2c3d4" in text


# ── The HTTP client ────────────────────────────────────────────────────────────

class _Resp:
    def __init__(self, status=200, json_body=None, text=""):
        self.status_code = status
        self.ok = 200 <= status < 300
        self._json = json_body
        self.text = text or str(json_body)

    def json(self):
        if self._json is None:
            raise ValueError("not json")
        return self._json


def test_client_sends_secret_and_event_type(connected, monkeypatch):
    monkeypatch.setenv("DRY_RUN", "false")
    sent = {}

    def fake_post(url, json, timeout):
        sent.update(url=url, body=json)
        return _Resp(json_body={"status": "success", "job_id": "j9", "customer_id": "cu1", "created": True})

    monkeypatch.setattr(jobs_app.requests, "post", fake_post)
    result = jobs_app.create_job({"full_name": "Sarah Chen", "secret": "spoofed"})

    assert sent["url"] == config.jobs_app_webhook_url()
    assert sent["body"]["secret"] == "s3cret", "the configured secret wins over anything in the payload"
    assert sent["body"]["event_type"] == "CREATE_JOB"
    assert result == {"job_id": "j9", "customer_id": "cu1", "created": True, "dry_run": False}


def test_client_raises_on_a_refusal_even_though_http_was_200(connected, monkeypatch):
    """Apps Script always answers 200; the refusal is in the body."""
    monkeypatch.setenv("DRY_RUN", "false")
    monkeypatch.setattr(jobs_app.requests, "post",
                        lambda *a, **k: _Resp(json_body={"status": "error", "message": "Unauthorized"}))
    with pytest.raises(jobs_app.JobsAppError, match="Unauthorized"):
        jobs_app.create_job({})


def test_client_explains_an_html_sign_in_page(connected, monkeypatch):
    monkeypatch.setenv("DRY_RUN", "false")
    monkeypatch.setattr(jobs_app.requests, "post",
                        lambda *a, **k: _Resp(json_body=None, text="<html>Sign in</html>"))
    with pytest.raises(jobs_app.JobsAppError, match="/exec URL"):
        jobs_app.create_job({})


def test_dry_run_never_calls_out(connected, monkeypatch):
    monkeypatch.setenv("DRY_RUN", "true")

    def boom(*a, **k):
        raise AssertionError("dry run made a network call")

    monkeypatch.setattr(jobs_app.requests, "post", boom)
    assert jobs_app.create_job({"full_name": "x"})["dry_run"] is True
