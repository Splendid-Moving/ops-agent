# Splendid Moving — Ops Agent

An internal assistant the team talks to. It answers questions about jobs, and it
books new ones from a customer screenshot.

```
python server.py     →  http://localhost:8080
```

---

## What it does

**Answers questions about jobs.** Reads the Google Calendar and aggregates.

> *How many jobs did we have last month?*
> 153 jobs — 144 full moves, 9 labor-only. 2 movers: 73 · 3 movers: 51 · 4 movers: 26

**Books a job from a screenshot.** Drop in a customer enquiry — Yelp, SMS, email
— and it reads the details, asks for whatever is missing, shows you exactly what
it's about to do, and only then:

1. Creates or updates the **GoHighLevel contact**
2. Books the **calendar event**
3. Texts the customer a **$50 deposit payment link**
4. Sends the **confirmation email**

Nothing happens until you type `yes`.

---

## The one thing to understand

The hard part was never the four API calls. It's **never half-completing them**.

A misread screenshot or a mid-flight failure must not leave a customer holding a
payment link for a truck that was never booked. Almost every design decision
below exists for that reason.

---

## How it works

```
START
  │
  ▼
router ─────────► analytics ──────────────────────► END      (read-only)
  │                Google Calendar → counts, schedules
  │
  ├──────────────► chat ────────────────────────────► END
  │
  ▼ intake
extract_screenshot     GPT reads the image · OCR checks the exact characters
  │
  ▼
resolve_addresses      partial address → full, verified address
  │
  ▼
validate ◄──────────────────────────┐   ← PURE PYTHON: is this bookable?
  │                                 │
  ├── incomplete ──► ask_missing ────┘   ⏸ PAUSES, waits for you
  ├── stuck ───────► give_up ──────────► END
  │
  ▼ complete
confirm                                 ⏸ PAUSES, shows you everything
  ├── "no" ────────────────────────────► END
  ├── an edit ─────► ask_missing
  └── "yes"
        │
        ▼
   upsert_contact          ← must succeed; the other three need its id
        │
   ┌────┴──────┬───────────┐
   ▼           ▼           ▼            (parallel)
 calendar   invoice     email
   └────┬──────┴───────────┘
        ▼
      report                            ← what happened, including what didn't
```

The two ⏸ points are why this is a graph and not a script. The run genuinely
stops — state is saved to disk, the process can restart — and picks up at the
exact line when you reply.

---

## Design decisions worth knowing

### The LLM proposes; Python decides

The models do two things only: read a screenshot, and turn your reply into
candidate values. **Whether a job is complete enough to book is decided by plain
Python** ([`schemas/checklist.py`](schemas/checklist.py)) — a loop over a list of
rules, no model involved. Same input, same verdict, every time.

### Nothing that writes may sit above a pause

When a graph resumes from a pause, **the paused node re-runs from the top**. If
`send_invoice()` sat above that line, every resume would bill the customer again.

So `ask_missing.py` and `confirm.py` contain nothing but formatting and a pause.
Every write lives in its own node downstream of the confirm gate. A test parses
the source of both files and fails if an API call ever appears in either —
because no ordinary test would catch that drift.

### The action ledger

Each of the four side effects records its own outcome:

```python
{"status": "success", "result": {"contact_id": "..."}, "error": None, "attempts": 1}
```

Every action checks the ledger before doing anything and returns immediately if
it already succeeded. So a retry after a partial failure re-fires **only** what
failed — the contact isn't recreated, the customer isn't billed twice.

The three parallel actions each write one key, merged by a custom reducer.
Without it, LangGraph's default last-write-wins would silently discard two of
the three results.

### DRY_RUN fails safe

`banana`, empty, and unset all resolve to dry run. Only the literal string
`false` goes live.

---

## Bugs found in production, and what they taught

Each of these is now a test.

| What happened | Why | Fix |
|---|---|---|
| Agent asked for a move date the screenshot clearly showed | The extractor was never told today's date, so it couldn't resolve "tomorrow" and reported it at low confidence | Prompt carries the current date, with worked examples |
| Calendar read `From: From: 614 E Verdugo Ave` | Extraction kept the field label; the calendar builder added its own | Label stripped in code, not just asked for in the prompt |
| Customer email came out `nikitatitrev354@` instead of `nikitatitarev354@` | Vision models *reconstruct* text rather than transcribe it, and smooth unusual strings. Email has zero redundancy — every other field has a downstream check that catches a slip | Cloud Vision OCR reads the exact characters; GPT decides which string is the email |
| `412 N Maple Ave` resolved to Brandon, South Dakota | With no city, Google picks one | Flagged only when the guess **leaves California** — flagging every guess trained people to click through warnings |
| "8:00 AM – Miray Ozer" | Event times are **arrival windows**, not start times. Every event looks that way, so nothing in the data could correct it | Business rules written down in [`schemas/business_context.py`](schemas/business_context.py), and the tool output now says `arrives 8-9am` |
| July 3 reported 11 jobs; it had 10 | Counting grouped by the `Date:` typed into the description, not where the event sits. One event was on Jul 2 with a description reading 07/03 | Counts use calendar position; mismatches are surfaced rather than hidden |

That last one found **5 events in July** whose typed date disagrees with their
calendar position. The agent now reports those by name when you ask about a
busy day.

---

## Layout

```
ops-agent/
├── server.py              FastAPI + SSE. Owns the pause/resume protocol.
├── chat_cli.py            Terminal client, same protocol
├── static/index.html      The UI
│
├── agent/
│   ├── graph.py           The whole topology, readable top to bottom
│   ├── state.py           Shared state + the action ledger
│   ├── models.py          Per-node model registry (the only file naming a provider)
│   ├── progress.py        Plain-language progress events for the UI
│   └── nodes/             One file per step
│
├── schemas/
│   ├── checklist.py       ← booking rules, as data
│   ├── business_context.py ← how the company operates
│   ├── email_template.py  ← customer-facing copy
│   └── intake.py          Extraction + working-record shapes
│
├── services/              Thin API clients, all dry-run aware
│   ├── ghl.py             Contacts, invoices, SMS, email
│   ├── calendar.py        Events, the description wire format
│   ├── address.py         Address completion + verification
│   ├── ocr.py             Cloud Vision, exact characters
│   ├── maps.py            Distance
│   ├── rates.py           The rate table
│   └── formatting.py      Pure helpers, heavily tested
│
└── tests/                 247 fast + 29 live
```

**The three files to edit for business changes** are in `schemas/` and marked
above. They're data, not logic — no graph code needs touching.

---

## Things that will break if you're not careful

**The calendar description is a wire format.** Four other repos parse it with
`^([A-Za-z ]+):\s*(.*)`, meaning *any* line shaped like `Word:` becomes a field
in their output. Adding a line here silently injects a field into
`move_reminders`, `job_form_automation` and `invoice_automation`. Agent metadata
goes in `extendedProperties.private` instead. A golden-file test pins the format.

**GHL dropdowns must match byte-for-byte.** `Rate` is a picklist. A value that
isn't identical to one of its options is accepted with a 200 and then silently
discarded, leaving the job with no rate. The exact strings are in
`services/rates.py`, and they're editable in the GHL UI — if bookings start
coming back with an empty Rate, re-query the live picklist first.

**Never write the `Create Google Event` field.** Checking it fires the GHL
workflow that calls `ghl_calendar_sync`, which creates a calendar event — and
this agent creates its own. That would double-book every job. `services/ghl.py`
raises rather than trusting itself not to.

---

## Setup

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
cp .env.example .env      # then fill it in
python server.py
```

`.env` needs: `OPENAI_API_KEY`, `GHL_ACCESS_TOKEN`, `GHL_LOCATION_ID`,
`GHL_USER_ID`, `GOOGLE_CREDENTIALS_B64`, `GOOGLE_CALENDAR_ID`,
`GOOGLE_MAPS_API_KEY`. The GHL and Google values are shared with the sibling
repos under `splendid_moving/`.

Google APIs used: Calendar, Distance Matrix, Address Validation, Cloud Vision.
**Cloud Vision authenticates with the service account, not the Maps key** — the
key is restricted to a specific API list, and widening it would mean editing a
credential other repos depend on.

```bash
python verify_services.py     # check every integration, read-only
pytest                        # 247 fast tests
pytest -m live                # also hits real APIs
```

### Models

One entry per node in `agent/models.py`, each overridable by env var:

| Node | Model | Why |
|---|---|---|
| `extract_screenshot` | gpt-5.1 | errors reach customers |
| `analytics` | gpt-5.1 | multi-step tool calling |
| `parse_reply` | gpt-4.1 | date reasoning, Python-validated after |
| `router`, `chat` | gpt-4.1-mini | runs every turn, wants to be cheap |

`MODEL_BACKEND=openrouter` switches the whole agent over. No node names a
provider, so it's a one-line change.

---

## Not done yet

- **Deployment** — runs on a laptop today. Needs a host and `PostgresSaver`
  instead of SQLite.
- **Extraction tuning** — built and prompted carefully, but tested against few
  real screenshots.
- **Confirmation email copy** — placeholder text in `schemas/email_template.py`.
- **Crews of 5–6** — GHL accepts them, they have no rate option, deliberately
  out of scope.
- **Lead source and move size** — captured if a screenshot shows them, never
  asked for.
