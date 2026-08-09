"""
Pure-function tests. No network.

The description round-trip tests are the important ones: they pin a format that
four other production repos parse.
"""

from datetime import datetime

import pytest

from services import calendar, formatting, rates


# ── Rates ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "movers,expected",
    [
        (2, "$115 cash /$125 card"),
        (3, "$145 cash /$155 card"),
        (4, "$180 cash /$190 card"),
        ("3", "$145 cash /$155 card"),
    ],
)
def test_rate_table(movers, expected):
    assert rates.format_rate(movers) == expected


def test_rate_strings_match_the_live_ghl_dropdown_exactly():
    """
    GHL's Rate field is a dropdown. A value that isn't byte-identical to one of
    its options is accepted by the API and then silently discarded, leaving the
    job with no rate. This test is the guard against that.
    """
    from services.ghl import PICKLIST_VALUES, CustomField

    allowed = PICKLIST_VALUES[CustomField.RATE]
    for movers in rates.SUPPORTED_MOVER_COUNTS:
        assert rates.format_rate(movers) in allowed


@pytest.mark.parametrize("size", [2, 3, 4])
def test_supported_crew_sizes(size):
    assert rates.is_supported_crew_size(size)
    assert not rates.is_out_of_scope_crew_size(size)
    assert rates.format_rate(size)


@pytest.mark.parametrize("size", [5, 6])
def test_five_and_six_are_out_of_scope_not_invalid(size):
    """
    Real crew sizes GHL accepts, priced by hand, deliberately not automated.
    Kept distinct from junk input so the agent can explain rather than reject.
    """
    assert not rates.is_supported_crew_size(size)
    assert rates.is_out_of_scope_crew_size(size)
    assert rates.format_rate(size) == ""


@pytest.mark.parametrize("bad", [1, 7, 8, 0, -1, "many", "", None])
def test_junk_crew_sizes_are_neither_supported_nor_out_of_scope(bad):
    assert not rates.is_supported_crew_size(bad)
    assert not rates.is_out_of_scope_crew_size(bad)
    assert rates.format_rate(bad) == ""


def test_deposit_is_a_bare_number():
    """1,459 of 1,598 real jobs store '50', not '$50'."""
    assert formatting.format_deposit(50) == "50"
    assert formatting.format_deposit("$50") == "50"
    assert formatting.format_deposit(50.0) == "50"
    assert formatting.format_deposit(200) == "200"


def test_deposit_passes_through_non_numeric():
    """'NO' means no deposit and appears on 41 real jobs."""
    assert formatting.format_deposit("NO") == "NO"


def test_calendar_deposit_defaults_to_unpaid_placeholder():
    """
    The calendar Deposit line tracks whether the deposit was PAID, not what was
    billed. A separate automation rewrites it on payment. Writing the amount
    here would claim a payment that hasn't happened.
    """
    desc = calendar.build_description(
        customer="X", phone="p", date="d", from_address="a", to_address="b",
    )
    assert "Deposit: ???" in desc
    assert calendar.parse_description(desc)["deposit"] == "???"


# ── Phone ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("8185550142", "+1(818)555-0142"),
        ("18185550142", "+1(818)555-0142"),
        ("(818) 555-0142", "+1(818)555-0142"),
        ("+1 818-555-0142", "+1(818)555-0142"),
    ],
)
def test_format_phone(raw, expected):
    assert formatting.format_phone(raw) == expected


def test_format_phone_passes_through_unparseable():
    assert formatting.format_phone("call the office") == "call the office"


# ── Addresses ──────────────────────────────────────────────────────────────────

def test_format_address_drops_comma_before_state():
    assert (
        formatting.format_address("544 E Amazing St, Burbank, CA 91501")
        == "544 E Amazing St, Burbank CA 91501"
    )


def test_format_address_collapses_whitespace():
    assert formatting.format_address("  412   N Maple Ave,  Burbank CA 91505 ") == (
        "412 N Maple Ave, Burbank CA 91505"
    )


# ── Names ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "full,expected",
    [
        ("Sarah Chen", ("Sarah", "Chen")),
        ("Maria De La Cruz", ("Maria", "De La Cruz")),
        ("Cher", ("Cher", "")),
        ("", ("", "")),
    ],
)
def test_split_name(full, expected):
    assert formatting.split_name(full) == expected


# ── Dates ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw", ["2026-03-14", "03/14/2026", "2026/03/14"])
def test_parse_date_formats(raw):
    assert formatting.parse_date(raw).date() == datetime(2026, 3, 14).date()


def test_parse_date_returns_none_when_unparseable():
    """Must NOT silently fall back to today — that would book a move for today."""
    assert formatting.parse_date("sometime next week") is None
    assert formatting.parse_date("") is None


# ── Arrival windows ────────────────────────────────────────────────────────────

BASE = datetime(2026, 3, 14)


@pytest.mark.parametrize(
    "label,start_hour,end_hour",
    [
        ("8-9am", 8, 9),
        ("2-4pm", 14, 16),
        ("11-1pm", 11, 13),      # start > end with a trailing pm means start is am
        ("8am-9am", 8, 9),
        ("11am-1pm", 11, 13),
        ("10-11am", 10, 11),
    ],
)
def test_parse_arrival_window(label, start_hour, end_hour):
    start, end = formatting.parse_arrival_time(label, BASE)
    assert datetime.fromisoformat(start).hour == start_hour
    assert datetime.fromisoformat(end).hour == end_hour


def test_single_time_becomes_15_minute_window():
    start, end = formatting.parse_arrival_time("9am", BASE)
    assert datetime.fromisoformat(start).hour == 8
    assert datetime.fromisoformat(start).minute == 45
    assert datetime.fromisoformat(end).hour == 9


def test_unparseable_arrival_returns_none():
    assert formatting.parse_arrival_time("morning-ish", BASE) is None
    assert formatting.parse_arrival_time("", BASE) is None


# ── Lead source colours ────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "source,expected",
    [
        ("Yelp", "6"),
        ("yelp ad", "6"),
        ("Google Local Services", "2"),
        ("LSA", "2"),
        ("Referral", None),
        ("", None),
    ],
)
def test_color_id(source, expected):
    assert formatting.get_color_id(source) == expected


# ── The wire format ────────────────────────────────────────────────────────────

GOLDEN = """Customer: Sarah Chen
Phone: +1(818)555-0142
Date: 03/14/2026
From: 412 N Maple Ave, Burbank CA 91505
To: 1830 Pine St, Glendale CA 91206
Distance: 8.2 miles
Rate: $145 cash /$155 card
Movers: 3
Deposit: 50
Source: Yelp
Notes:
Third floor walkup"""


def test_description_matches_golden():
    """
    Pins the exact bytes four other repos parse. If this fails, a downstream
    pipeline is about to break — fix the code, do not update the golden string
    without checking move_reminders, job_form_automation and invoice_automation.
    """
    assert (
        calendar.build_description(
            customer="Sarah Chen",
            phone="+1(818)555-0142",
            date="03/14/2026",
            from_address="412 N Maple Ave, Burbank CA 91505",
            to_address="1830 Pine St, Glendale CA 91206",
            distance="8.2 miles",
            rate="$145 cash /$155 card",
            movers="3",
            deposit="50",
            source="Yelp",
            notes="Third floor walkup",
        )
        == GOLDEN
    )


def test_description_round_trips():
    fields = calendar.parse_description(GOLDEN)
    assert fields["customer"] == "Sarah Chen"
    assert fields["phone"] == "+1(818)555-0142"
    assert fields["date"] == "03/14/2026"
    assert fields["from"] == "412 N Maple Ave, Burbank CA 91505"
    assert fields["to"] == "1830 Pine St, Glendale CA 91206"
    assert fields["movers"] == "3"
    assert fields["deposit"] == "50"
    assert fields["notes"] == "Third floor walkup"
    assert calendar.is_job_event(fields)


def test_extra_stop_line_only_appears_when_present():
    with_stop = calendar.build_description(
        customer="X", phone="p", date="d", from_address="a",
        to_address="b", extra_stop="middle",
    )
    assert "Extra stop: middle" in with_stop
    assert calendar.parse_description(with_stop)["extra_stop"] == "middle"

    without = calendar.build_description(
        customer="X", phone="p", date="d", from_address="a", to_address="b",
    )
    assert "Extra stop" not in without


def test_non_job_events_are_rejected():
    """Crew meetings, blocks and PTO must not count toward job totals."""
    assert not calendar.is_job_event(calendar.parse_description("Team meeting 9am"))
    assert not calendar.is_job_event(calendar.parse_description(""))
    assert not calendar.is_job_event(
        calendar.parse_description("Customer: Bob\nPhone: 555")  # no date
    )


def test_multiline_notes_survive_round_trip():
    desc = calendar.build_description(
        customer="X", phone="p", date="d", from_address="a", to_address="b",
        notes="Line one\nLine two\nLine three",
    )
    assert calendar.parse_description(desc)["notes"] == "Line one\nLine two\nLine three"


# ── Lead source normalization ──────────────────────────────────────────────────

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Previous Customer", "Previous Customer"),
        ("previous customer", "Previous Customer"),
        ("Previous costumer", "Previous Customer"),   # real typo on live events
        ("yelp", "Yelp"),
        ("LSA", "Local Service Ads"),
        ("google local services", "Local Service Ads"),
        ("", "unspecified"),
        ("   ", "unspecified"),
        (None, "unspecified"),
    ],
)
def test_normalize_source(raw, expected):
    """
    Without this, three spellings of 'Previous Customer' count as three
    different lead sources and every breakdown is quietly wrong.
    """
    assert formatting.normalize_source(raw) == expected


def test_unknown_source_passes_through_unchanged():
    assert formatting.normalize_source("Nextdoor") == "Nextdoor"


# ── Address label stripping + ZIP ──────────────────────────────────────────────

@pytest.mark.parametrize(
    "raw,expected",
    [
        # Seen live: extraction kept the label, build_description added its own,
        # and the calendar event read "From: From: 614 E Verdugo Ave".
        ("From: 614 E Verdugo Ave, Burbank CA 91501", "614 E Verdugo Ave, Burbank CA 91501"),
        ("To: 1039 Justin Ave, Glendale CA 91201", "1039 Justin Ave, Glendale CA 91201"),
        ("Pickup: 412 N Maple St, Burbank CA 91505", "412 N Maple St, Burbank CA 91505"),
        ("Drop-off: 1830 Pine St, Glendale CA 91206", "1830 Pine St, Glendale CA 91206"),
        ("Moving from: 909 Beacon Ave, Los Angeles CA 90015", "909 Beacon Ave, Los Angeles CA 90015"),
        ("Extra stop: 500 S Brand Blvd, Glendale CA 91204", "500 S Brand Blvd, Glendale CA 91204"),
        # Two labels on one line.
        ("Pickup: From: 123 Main St, Burbank CA 91501", "123 Main St, Burbank CA 91501"),
    ],
)
def test_field_labels_are_stripped_from_addresses(raw, expected):
    assert formatting.format_address(raw) == expected


@pytest.mark.parametrize(
    "street",
    ["Fromage Way, Los Angeles CA 90001", "Toluca Lake Dr, Burbank CA 91505"],
)
def test_street_names_beginning_with_a_label_word_survive(street):
    """'Fromage' starts with 'From'. Only a label FOLLOWED BY a separator counts."""
    assert formatting.format_address(street) == street


def test_already_clean_addresses_are_unchanged():
    clean = "614 E Verdugo Ave, Burbank CA 91501"
    assert formatting.format_address(clean) == clean


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("614 E Verdugo Ave, Burbank CA 91501-2385", "614 E Verdugo Ave, Burbank CA 91501"),
        ("1039 Justin Ave, Glendale CA 91201-3612", "1039 Justin Ave, Glendale CA 91201"),
    ],
)
def test_zip_plus_four_is_truncated_to_five(raw, expected):
    """
    Google returns ZIP+4. It is correct but nobody at Splendid writes it, and
    the crew reads these off a phone.
    """
    assert formatting.format_address(raw) == expected


def test_strip_label_leaves_bare_text_alone():
    assert formatting.strip_address_label("614 E Verdugo Ave") == "614 E Verdugo Ave"
    assert formatting.strip_address_label("") == ""


# ── Calendar date vs typed date ────────────────────────────────────────────────

def test_job_records_both_the_calendar_date_and_the_typed_date():
    """
    A real July 2026 event sat on Jul 2 while its description read 07/03/2026.
    Counting by the typed date reported 11 jobs on a day that had 10, so both
    dates are kept and anything time-based uses the calendar one.
    """
    import services.calendar as cal

    event = {
        "id": "e1",
        "summary": "Mallory Craig",
        "description": calendar.build_description(
            customer="Mallory Craig", phone="+1(818)555-0142",
            date="07/03/2026", from_address="a", to_address="b",
        ),
        "start": {"dateTime": "2026-07-02T08:00:00-07:00"},
        "end": {"dateTime": "2026-07-02T09:00:00-07:00"},
    }
    original = cal.list_events
    try:
        cal.list_events = lambda *a, **k: [event]
        job = cal.list_jobs(datetime(2026, 7, 1), datetime(2026, 7, 31))[0]
    finally:
        cal.list_events = original

    assert job["calendar_date"] == "07/02/2026"   # where it actually is
    assert job["move_date"] == "07/03/2026"       # what someone typed
    assert job["date_mismatch"] is True


def test_no_mismatch_flag_when_the_dates_agree():
    import services.calendar as cal

    event = {
        "id": "e2",
        "summary": "Sarah Chen",
        "description": calendar.build_description(
            customer="Sarah Chen", phone="+1(818)555-0142",
            date="07/03/2026", from_address="a", to_address="b",
        ),
        "start": {"dateTime": "2026-07-03T08:00:00-07:00"},
        "end": {"dateTime": "2026-07-03T09:00:00-07:00"},
    }
    original = cal.list_events
    try:
        cal.list_events = lambda *a, **k: [event]
        job = cal.list_jobs(datetime(2026, 7, 1), datetime(2026, 7, 31))[0]
    finally:
        cal.list_events = original

    assert job["calendar_date"] == job["move_date"] == "07/03/2026"
    assert job["date_mismatch"] is False
