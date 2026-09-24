# Website traffic lane (Vercel Web Analytics)

## What we're building
A fourth lane in the existing ops-agent graph: the dispatcher asks about
**splendidmoving.com traffic** in Google Chat and gets an answer read from
Vercel's Web Analytics API. Read-only. Same pattern as the calendar lane —
Python does the counting, the model only picks the query and phrases the reply.

## Contract

GOAL: In Google Chat, these four questions get correct answers with no
follow-up needed, each in under 10 seconds:
- "how many people visited the site last week"
- "what are people finding us through"
- "which pages get the most traffic"
- "is traffic up or down vs the week before"

CONSTRAINTS:
- Vercel **Hobby** plan, verified against the live account:
  - `visits/aggregate` (any breakdown — by day, page, referrer, country,
    device) is capped at the **latest 31 days**. Past that the API returns a
    400 saying so. This is the hard limit on "compare periods".
  - `visits/count` is capped the same way **when dates are sent**. Lifetime
    numbers come back only when NO range is sent at all: 6,225 pageviews /
    4,701 visitors since 2026-02-11.
  - No custom events, no UTM parameters on this plan.
- Read-only. This lane never writes anything anywhere.
- One project: splendidmoving.com. Confirmed live — `_vercel/insights` is
  served by the site.
- Credentials in `ops-agent/.env` (gitignored): `VERCEL_TOKEN`,
  `WEBSITE_PROJECT_ID`, `VERCEL_TEAM_ID`. The same values also live in the
  workspace `.env` one level up; ops-agent keeps its own copy because Railway
  deploys it alone, which is how `GOOGLE_MAPS_API_KEY` is already handled.
- Must not grow the tool count carelessly: one tool per *shape*, not per
  question.

FORMAT:
- `services/vercel.py` — the API client. Knows nothing about the agent.
- `agent/nodes/site_traffic.py` — the lane: tools + agent, mirroring
  `analytics.py`.
- Router gains a fourth intent, `site`.
- Two tools only:
  - `site_traffic(start_date, end_date, by)` — `by` is total | day | page |
    referrer | country | device.
  - `compare_traffic(...)` — two periods, both totals, and the change computed
    **in Python**.

FAILURE (any of these = not done):
- A question about a period older than the plan's window returns zeros, or a
  number, instead of saying plainly that the data does not go back that far.
- Any percentage or delta is computed by the model rather than by Python.
- A calendar question ("how many jobs last week") is routed to this lane, or a
  traffic question is routed to the calendar lane.
- "Yesterday" is computed in UTC rather than Los Angeles time, so the day
  boundary is off by 7-8 hours.
- A Vercel outage, an expired token, or a 429 produces a stack trace or a
  silent wrong answer instead of a readable message.
- The site genuinely has no traffic in a period and the reply cannot be
  distinguished from a failed lookup.
- `VERCEL_TOKEN` appears in git, in a log line, or in a Chat message.
- Tests pass but nobody ran the four GOAL questions against the live API.

## Built on top of
- **Vercel Web Analytics REST API** (`/v1/query/web-analytics/visits/{count,aggregate}`)
  — public and GA as of 2026; same aggregated data as the dashboard, so numbers
  match what you see there. Saves building any tracking of our own.
- `requests` + the repo's existing `@traceable` tracing and `progress` helpers.
  No new dependency.
- Deliberately NOT using the Vercel MCP server or CLI: this runs inside a
  FastAPI process on Railway, where a plain HTTP call is simpler and testable.

## Gotchas we're handling
- **The 31-day window** on breakdowns. Checked in Python before the call, so
  the refusal is instant and identical every time; the API's own 400 is caught
  as a backstop in case Vercel moves the limit.
- **All-time totals need no dates.** "How many people have ever visited"
  calls `count` with the range omitted. A dated total that reaches past the
  window falls back to the all-time figure and says explicitly that it
  substituted — answering a different question is only acceptable out loud.
- **Timezones.** Vercel returns UTC timestamps; the dispatcher means Los
  Angeles days. Ranges are converted, same as the calendar lane already does.
- **Hobby collection pauses** after 50,000 events in a month. Traffic can stop
  being recorded silently. A day with zero pageviews is reported as zero *and*
  flagged, not smoothed over.
- **Router ambiguity.** "How many last month" fits both lanes. The router
  prompt gets explicit examples of each, and a test pins them.
- **Token scope.** A Vercel token is account-wide; there is no read-only scope.
  It goes in `.env` only, and the client never logs the header.
- **No traffic vs no answer.** An empty result says "no visits recorded",
  which reads differently from "couldn't reach Vercel".
- **Blank dimension values are real.** An empty `referrerHostname` means
  direct traffic (159 of 249 visits last week) and an empty `deviceType` means
  unknown. Both get readable labels rather than an empty cell.
- **The `Others` bucket.** When a breakdown has more distinct values than the
  requested limit, Vercel rolls the rest into `Others`. It is passed through
  as-is, never mistaken for a country or a page.

## Build sequence
1. ~~`services/vercel.py` + live smoke test.~~ DONE — see Status.
2. Tests for both tools, written against the GOAL/FAILURE lines, run once and
   failing.
3. The two tools, over fixed fake API responses.
4. The lane node + router intent + router tests.
5. Live run of the four GOAL questions through Google Chat.

## How to run and test
- Tests: `.venv/bin/python -m pytest -q` (from `ops-agent/`)
- Live check: `.venv/bin/python verify_services.py`
- Setup (one-time): create a Vercel access token at
  vercel.com → Settings → Tokens; copy the project id from the project's
  Settings → General. Put both in `.env` as `VERCEL_TOKEN` and
  `VERCEL_PROJECT_ID`; add `VERCEL_TEAM_ID` if the project belongs to a team
  rather than a personal account. Same variables go into Railway.

## Status
_Updated: 2026-09-24_
- Built: **all five phases; the lane is live.** `services/vercel.py`,
  `agent/nodes/site_traffic.py`, the `site` router intent and the graph edge.
  37 tests (26 offline + 11 live routing). All four GOAL questions answered
  correctly through the real API in 2.8-4.1s, against a 10s budget.
  Project is
  `main-website` (`prj_OMDGo1…`) on team `team_4cQPMDWrnPos…`, Web Analytics
  enabled and returning data. All six breakdowns confirmed working: day,
  requestPath, referrerHostname, country, deviceType, browserName.
- Changed from plan: (1) the window is 31 days and applies to `count` too
  whenever dates are sent — lifetime totals require omitting the range, found
  only by running the live question, not by the fakes. (2) `site_traffic`
  takes optional dates as a result. (3) env var is `WEBSITE_PROJECT_ID`.
- Next: add the three Vercel vars to Railway, then rotate the token — it was
  visible in the editor during setup.
- Watch out for: the token is a `vcp_`-prefixed scoped token. It works for
  web-analytics and project reads but returns 404 on `/v2/user` and 403 on
  `/v2/teams`, so don't use those to sanity-check it.
