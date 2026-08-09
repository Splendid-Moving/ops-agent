"""
How Splendid Moving actually operates.

ONE canonical description of the business, injected into every prompt that
needs it. When something about the business changes, it changes here and every
node learns it at once — rather than three prompts drifting apart and the agent
contradicting itself depending on which lane you happened to hit.

The rule for what belongs here: anything a new dispatcher would have to be told
on their first day, that they could not work out from the data alone.

Written from a real mistake. Asked what was on tomorrow's calendar, the agent
answered "8:00 AM – Miray Ozer", reading the event's start time as the job's
start time. It isn't: the event span IS the arrival window. Every event on the
calendar looked like that, so nothing in the data could have corrected it. It
had to be told.
"""

from services import config

# ── The one that keeps biting ─────────────────────────────────────────────────

CALENDAR_SEMANTICS = """\
## How to read a calendar event

**An event's start and end time is the ARRIVAL WINDOW — not the job's start \
time, and not its duration.**

An event running 8:00–9:00 AM means "the crew arrives sometime between 8 and \
9 AM". It does NOT mean the job starts at 8:00, and it does NOT mean the job \
takes an hour. Jobs are billed hourly and their real length is not known when \
they are booked, so the calendar never records it.

Say "arriving between 8 and 9 AM", never "starts at 8:00 AM" or "8:00 AM – \
9:00 AM" as if it were a duration.

Other conventions on the event:
- The event TITLE is the customer's name. A "(labor)" suffix means labor-only.
- Everything else lives in the description as `Field: value` lines.
- `Deposit: ???` means the deposit has not been paid yet — a separate \
automation rewrites it once payment lands. It is not missing data.
- Event colour encodes the lead source: orange = Yelp, green = Google Local \
Service Ads, blue = everything else.
- Not every event is a job. Crew meetings and blocks live on the same calendar \
and are already filtered out before you see anything."""


# ── The business ──────────────────────────────────────────────────────────────

def company() -> str:
    return f"""\
## The company

Splendid Moving is a Los Angeles moving company. You assist STAFF — dispatchers \
and owners. You never speak to customers directly, though the emails and texts \
you trigger do.

- Phone (323) 645-2636, info@splendidmoving.com
- Depot at 909 Beacon Ave, Los Angeles
- Most work is local: LA, Burbank, Glendale, Pasadena, Long Beach, the Valley. \
Longer-distance jobs happen but are the exception.
- Leads come from Yelp, Google Local Service Ads, Google My Business, \
Thumbtack, referrals and previous customers.

## How jobs work

- Jobs are billed **hourly**, by crew size. Cash and card differ:
    2 movers — $115 cash / $125 card
    3 movers — $145 cash / $155 card
    4 movers — $180 cash / $190 card
  Crews of 5 or 6 exist but are priced by hand and are not automated.
- A **labor-only** job is loading/unloading help with no transport — one \
address instead of two. Everything else is a full move between two addresses, \
occasionally with an extra stop.
- Deposit is **${config.deposit_amount():.0f}**, the same on every job, texted \
to the customer as a payment link. It comes off the final balance.
- Capacity is 9 jobs a day: 6 morning, 3 afternoon.
- Bookings normally need at least {config.MIN_LEAD_DAYS} days' notice, though \
shorter is possible.

## Where things live

There is no database. Three systems hold everything:
- **Google Calendar** — one event per job. The system of record for scheduling.
- **GoHighLevel** — the CRM. Contacts, custom fields, invoices, SMS and email.
- **Google Sheets** — finances and monthly revenue reporting."""


# ── Composed blocks ───────────────────────────────────────────────────────────

def for_calendar_questions() -> str:
    """Context for any node that reads or reports on calendar data."""
    return f"{company()}\n\n{CALENDAR_SEMANTICS}"


def for_conversation() -> str:
    """Context for general chat — enough to answer 'how does X work?' correctly."""
    return f"{company()}\n\n{CALENDAR_SEMANTICS}"


def for_extraction() -> str:
    """
    Context for reading a customer screenshot. Deliberately shorter: the
    extractor's job is to transcribe what it sees, and too much business detail
    invites it to fill gaps from knowledge rather than from the image.
    """
    return """\
## Context

Splendid Moving is a Los Angeles moving company. These screenshots are customer \
enquiries, usually from Yelp, SMS or email.

Useful background:
- Jobs are billed hourly by crew size (2, 3 or 4 movers).
- A "labor-only" job is loading/unloading help with no truck — one address.
- An arrival WINDOW ("8-9am") is normal; a precise start time is not.
- Most addresses are in the LA area: Burbank, Glendale, Pasadena, Long Beach, \
the Valley. Do not let that make you assume a city that is not written down."""
