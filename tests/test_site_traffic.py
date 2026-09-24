"""
Website traffic lane — tests written before the tools exist.

Each test maps to a line in SPEC.md's contract. The ones that matter most are
the window refusal (Hobby caps breakdowns at 31 days) and the comparison
arithmetic: the calendar lane already proved that a model asked to add three
numbers gets it wrong, so every delta here is computed in Python.
"""

from datetime import date, datetime, timedelta

import pytest

from agent.nodes import site_traffic as st
from services import vercel


def _days_ago(n: int) -> str:
    return (date.today() - timedelta(days=n)).isoformat()


@pytest.fixture
def api(monkeypatch):
    """Serve canned Vercel responses; nothing in this file touches the network."""
    calls = []

    def fake_count(**kwargs):
        calls.append(("count", kwargs))
        return {"pageviews": 6225, "visitors": 4701}

    def fake_aggregate(**kwargs):
        calls.append(("aggregate", kwargs))
        return _rows_for(kwargs.get("by"))

    monkeypatch.setattr(vercel, "count_visits", fake_count)
    monkeypatch.setattr(vercel, "aggregate_visits", fake_aggregate)
    return calls


def _rows_for(by: str) -> list[dict]:
    if by == "referrerHostname":
        return [{"referrerHostname": "", "pageviews": 159, "visitors": 120},
                {"referrerHostname": "google.com", "pageviews": 80, "visitors": 61},
                {"referrerHostname": "yelp.com", "pageviews": 10, "visitors": 8}]
    if by == "requestPath":
        return [{"requestPath": "/", "pageviews": 209, "visitors": 150},
                {"requestPath": "/local-moving", "pageviews": 29, "visitors": 22}]
    if by == "country":
        return [{"country": "US", "pageviews": 185, "visitors": 140},
                {"country": "Others", "pageviews": 23, "visitors": 20}]
    if by == "deviceType":
        return [{"deviceType": "desktop", "pageviews": 125, "visitors": 90},
                {"deviceType": "", "pageviews": 6, "visitors": 5}]
    return [{"timestamp": "2026-09-17T00:00:00.000Z", "pageviews": 46, "visitors": 30},
            {"timestamp": "2026-09-18T00:00:00.000Z", "pageviews": 38, "visitors": 25}]


def _traffic(**kwargs) -> str:
    kwargs.setdefault("start_date", _days_ago(7))
    kwargs.setdefault("end_date", _days_ago(0))
    return st.site_traffic.invoke(kwargs)


# ── GOAL: the four questions ───────────────────────────────────────────────────

def test_visitor_totals_are_reported(api):
    out = _traffic(by="total")
    assert "6,225" in out and "4,701" in out


def test_top_pages(api):
    out = _traffic(by="page")
    assert "/local-moving" in out and "29" in out


def test_referrers(api):
    out = _traffic(by="referrer")
    assert "google.com" in out and "80" in out


def test_daily_breakdown(api):
    out = _traffic(by="day")
    assert "2026-09-17" in out and "46" in out


# ── FAILURE: the 31-day window must be stated, never answered with zeros ──────

def test_a_breakdown_past_the_window_is_refused_in_plain_words(api):
    out = _traffic(start_date=_days_ago(90), end_date=_days_ago(60), by="day")
    assert "31 days" in out
    # The point: a refusal must not read like a reading. No traffic figures.
    assert "pageviews" not in out and "visitors" not in out
    assert api == []                              # and no pointless API call


def test_the_refusal_says_what_can_be_asked_instead(api):
    out = _traffic(start_date=_days_ago(90), end_date=_days_ago(60), by="day")
    assert "total" in out.lower()                 # counts have no window


def test_totals_are_not_capped_by_the_window(api):
    """`count` has no reporting window — lifetime numbers are legitimate."""
    out = _traffic(start_date="2026-02-11", end_date=_days_ago(0), by="total")
    assert "6,225" in out
    assert api and api[0][0] == "count"


def test_a_range_straddling_the_window_edge_is_refused_not_silently_trimmed(api):
    """Trimming would answer a different question than the one asked."""
    out = _traffic(start_date=_days_ago(40), end_date=_days_ago(1), by="day")
    assert "31 days" in out


# ── FAILURE: deltas are Python's job ──────────────────────────────────────────

def test_comparison_reports_both_periods_and_the_change(api, monkeypatch):
    totals = iter([{"pageviews": 249, "visitors": 180},
                   {"pageviews": 300, "visitors": 210}])
    monkeypatch.setattr(vercel, "count_visits", lambda **k: next(totals))
    out = st.compare_traffic.invoke({
        "period_a_start": _days_ago(7), "period_a_end": _days_ago(0),
        "period_b_start": _days_ago(14), "period_b_end": _days_ago(7),
    })
    assert "249" in out and "300" in out
    assert "-51" in out
    assert "17" in out                    # -17.0%


def test_growth_from_zero_does_not_divide_by_zero(api, monkeypatch):
    totals = iter([{"pageviews": 50, "visitors": 40}, {"pageviews": 0, "visitors": 0}])
    monkeypatch.setattr(vercel, "count_visits", lambda **k: next(totals))
    out = st.compare_traffic.invoke({
        "period_a_start": _days_ago(7), "period_a_end": _days_ago(0),
        "period_b_start": _days_ago(14), "period_b_end": _days_ago(7),
    })
    assert "%" not in out or "n/a" in out.lower()
    assert "50" in out


# ── FAILURE: blank dimension values are real values ───────────────────────────

def test_a_blank_referrer_reads_as_direct_traffic(api):
    """159 of 249 visits last week had no referrer. An empty cell hides that."""
    out = _traffic(by="referrer")
    assert "direct" in out.lower()
    assert "159" in out


def test_a_blank_device_reads_as_unknown(api):
    out = _traffic(by="device")
    assert "unknown" in out.lower()


def test_the_others_bucket_is_passed_through_as_itself(api):
    """Vercel rolls the tail into `Others`. It is not a country."""
    out = _traffic(by="country")
    assert "Others" in out


# ── FAILURE: no traffic must not look like a broken lookup ────────────────────

def test_an_empty_period_says_no_visits_recorded(monkeypatch):
    monkeypatch.setattr(vercel, "aggregate_visits", lambda **k: [])
    out = _traffic(by="day")
    assert "no visits" in out.lower()
    assert "error" not in out.lower()


def test_an_api_failure_reads_as_a_failure_not_as_zero(monkeypatch):
    def boom(**k):
        raise vercel.VercelError("Vercel returned 503")
    monkeypatch.setattr(vercel, "aggregate_visits", boom)
    out = _traffic(by="day")
    assert "couldn't" in out.lower() or "could not" in out.lower()
    assert "no visits" not in out.lower()


def test_a_rate_limit_is_named_so_it_can_be_waited_out(monkeypatch):
    def boom(**k):
        raise vercel.VercelError("Vercel returned 429: rate limited")
    monkeypatch.setattr(vercel, "aggregate_visits", boom)
    assert "429" in _traffic(by="day") or "rate" in _traffic(by="day").lower()


# ── FAILURE: the token must never leave the process ───────────────────────────

def test_the_token_is_never_in_an_error_message(monkeypatch):
    monkeypatch.setattr(vercel.config, "vercel_token", lambda: "vcp_SECRET_VALUE")

    class Resp:
        ok, status_code, text = False, 401, '{"error":{"message":"invalid token"}}'
        headers = {"content-type": "application/json"}

    monkeypatch.setattr(vercel.requests, "get", lambda *a, **k: Resp())
    with pytest.raises(vercel.VercelError) as exc:
        vercel.count_visits(since="2026-09-01", until="2026-09-07")
    assert "vcp_SECRET_VALUE" not in str(exc.value)


# ── FAILURE: Los Angeles days, not UTC ────────────────────────────────────────

def test_ranges_are_sent_as_la_dates(api):
    """
    Vercel timestamps are UTC and LA is 7-8 hours behind, so "yesterday" in UTC
    starts mid-afternoon the day before in Los Angeles.
    """
    _traffic(start_date="2026-09-17", end_date="2026-09-24", by="day")
    _, kwargs = api[0]
    assert kwargs["since"].startswith("2026-09-17")
    assert kwargs["until"].startswith("2026-09-24")


def test_daily_buckets_are_labelled_as_utc_rather_than_silently_shifted(api):
    """
    Vercel buckets by UTC day and we cannot re-bucket without the raw hits.
    Relabelling 2026-09-17T00:00Z as "the 16th" would be a lie of the same
    size as ignoring the offset. So the output says which day it means.
    """
    out = _traffic(by="day")
    assert "2026-09-17" in out
    assert "UTC" in out


def test_today_and_yesterday_are_resolved_in_los_angeles(api):
    """
    The half we DO control. After 5pm in Los Angeles it is already tomorrow in
    UTC, so a range built from UTC would ask for a day that has not happened.
    """
    text = st._system_prompt()
    assert "Los Angeles" in text
    assert datetime.now(st.LA_TZ).strftime("%Y-%m-%d") in text


# ── The tool surface stays small ──────────────────────────────────────────────

def test_the_lane_exposes_exactly_two_tools():
    """One tool per shape, not per question — SPEC constraint."""
    assert len(st.TOOLS) == 2


def test_every_breakdown_the_spec_promises_is_accepted(api):
    for by in ("total", "day", "page", "referrer", "country", "device"):
        assert "not a valid" not in _traffic(by=by).lower()


def test_an_unknown_breakdown_is_rejected_clearly(api):
    out = _traffic(by="wormholes")
    assert "wormholes" in out


# ── FAILURE: the two "how many last month" lanes must not cross ───────────────

@pytest.mark.live
@pytest.mark.parametrize("message,expected", [
    # The dangerous pairs — same shape, different subject.
    ("how many jobs did we have last week?", "analytics"),
    ("how many people visited the site last week?", "site"),
    ("how many moves last month?", "analytics"),
    ("how many visitors last month?", "site"),
    # Website-only vocabulary
    ("what are people finding us through?", "site"),
    ("which pages get the most traffic?", "site"),
    ("is the website traffic up or down?", "site"),
    ("how many pageviews yesterday", "site"),
    # Calendar-only vocabulary
    ("who are we moving tomorrow?", "analytics"),
    ("how many movers do I need Saturday?", "analytics"),
    ("what's on the calendar Friday?", "analytics"),
])
def test_router_separates_jobs_from_website_traffic(message, expected):
    """
    Both lanes answer "how many ... last month". The only signal is WHAT is
    being counted, so this is the misroute most likely to happen and the one a
    user would never notice — a confident answer from the wrong dataset.
    """
    from langchain_core.messages import HumanMessage

    from agent.nodes import router

    assert router.route({"messages": [HumanMessage(message)]})["intent"] == expected


# ── All-time totals: the API gives them ONLY when no dates are sent ───────────

def test_omitting_both_dates_gives_the_all_time_total(api):
    """
    Verified against the live API: `count` with explicit dates is capped at 31
    days like everything else. Lifetime numbers come back only when no range
    is sent at all.
    """
    out = st.site_traffic.invoke({"by": "total"})
    assert "6,225" in out
    assert "all time" in out.lower()
    _, kwargs = api[0]
    assert kwargs.get("since") is None and kwargs.get("until") is None


def test_a_total_reaching_past_the_window_says_what_it_substituted(api):
    """
    Answering a different question than the one asked is fine only if you say
    so. Silently returning all-time for "since February" would be a lie.
    """
    out = st.site_traffic.invoke(
        {"start_date": _days_ago(200), "end_date": _days_ago(0), "by": "total"})
    assert "6,225" in out
    assert "instead" in out.lower()
    assert "31" in out


def test_a_breakdown_with_no_dates_is_refused(api):
    """Only totals work without a range; a breakdown needs one."""
    out = st.site_traffic.invoke({"by": "page"})
    assert "date" in out.lower()
    assert api == []
