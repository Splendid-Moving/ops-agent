# Splendid Moving — Ops Agent

An internal assistant the team talks to in **Google Chat**. It answers questions
about jobs, and it books new ones from a screenshot of a customer enquiry.

New to the codebase? Read **[LEARN.md](LEARN.md)** first — it explains how the
whole thing works and why it is shaped this way.

---

## What it does

**Answers questions about jobs.** Reads the Google Calendar and aggregates.

> *How many jobs did we have last month?*
> 153 jobs — 144 full moves, 9 labor-only. 2 movers: 73 · 3 movers: 51 · 4 movers: 26

**Books a job from a screenshot.** Drop in a customer enquiry — Yelp, SMS,
email — and it reads the details, asks for whatever is missing, shows you
exactly what it is about to do, and only then:

1. Creates or updates the **GoHighLevel contact**
2. Books the **calendar event**
3. Texts the customer a **$50 deposit invoice**
4. Sends the **confirmation email**

Nothing happens until you reply `yes`.

**Commands:** `/clear` forgets the conversation and starts fresh, `/help` lists
what it can do. Both also work typed as plain words (`reset`, `start over`).

---

## The one thing to understand

The hard part was never the four API calls. It is **never half-completing
them**.

A misread screenshot or a mid-flight failure must not leave a customer holding
a payment link for a truck that was never booked. Almost every design decision
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
extract_screenshot     a vision model reads the image
  │
  ▼
resolve_addresses      partial address → full, verified address
  │
  ▼
validate ◄──────────────────────────┐   ← PURE PYTHON: is this bookable?
  │                                 │
  │ something missing               │
  ▼                                 │
ask_missing ──── ⏸ PAUSES ──────────┘   waits for a human answer
  │
  │ complete
  ▼
confirm ──────── ⏸ PAUSES               shows the summary, waits for "yes"
  │
  ▼ approved
act_contact ──┬──► act_calendar ──┐
              ├──► act_invoice ───┤     each records its own result
              └──► act_email ─────┤
                                  ▼
                              report ──► END
```

Those two ⏸ pauses are the whole reason this is a LangGraph app rather than a
script. The graph genuinely stops — for a second or an hour — and resumes at
the same spot when the answer arrives.

---

## Layout

| Path | What lives there |
|---|---|
| `agent/` | The graph. `state.py` is the shared memory, `graph.py` wires the nodes together, `nodes/` is one file per step. |
| `services/` | Talking to the outside world — GoHighLevel, Google Calendar, Maps, OCR. Nothing here knows the agent exists. |
| `schemas/` | The business rules, as data. The booking checklist, the confirmation email, the rate table. **Change these to change behaviour.** |
| `channels/` | How humans reach the agent. Currently Google Chat. |
| `tests/` | 427 checks that run in under a second. |
| `static/` | The browser UI and the email logo. |

Entry points:

| File | Purpose |
|---|---|
| `app.py` | **Production.** Serves the Chat webhook. This is what Railway runs. |
| `server.py` | Local browser UI at `localhost:8080`. Dev only. |
| `chat_cli.py` | Local terminal chat. Dev only. |
| `verify_services.py` | Checks credentials and connectivity without touching the agent. |

---

## Running it locally

```bash
cp .env.example .env          # then fill it in
pip install -r requirements.txt

python verify_services.py     # confirm credentials work
python server.py              # browser UI  -> http://localhost:8080
python chat_cli.py            # or terminal
```

**`DRY_RUN=true` is the default and means every write is logged instead of
sent.** Reads still hit the live API, so you can develop against real calendar
data without risk of texting a customer.

```bash
pytest                        # 427 fast tests, no network
pytest -m live                # hits real APIs — needs credentials
```

---

## Deployment

Deployed on Railway, wired to Google Chat. Full setup — including the Google
Cloud side — is in **[DEPLOY.md](DEPLOY.md)**.

Two operational notes:

- **One instance only.** `railway.json` pins `numReplicas: 1`. Conversations are
  stored in SQLite on a mounted volume; two instances would each hold half of
  them. Scaling means moving to Postgres first.
- **The `/data` volume is required.** Without it, a booking paused waiting on
  someone's answer is wiped on every deploy.

---

## Observability

Every turn of the agent can be recorded as a **trace**: which node ran, what
each model was asked and answered, how long it took, what it cost, and every
GoHighLevel, Calendar, Maps and Vision call underneath. Traces are read at
[smith.langchain.com](https://smith.langchain.com).

This is worth having because the failures that actually happen here are
invisible in the logs. "The invoice didn't send" could be a misread screenshot,
a rejected dropdown value, or a GHL 400 three layers down — and the log line
looks the same for all three. A trace shows which, in one view.

### Turning it on

```bash
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_pt_...          # smith.langchain.com -> Settings -> API Keys
LANGSMITH_PROJECT=splendid-ops-agent-dev
```

The app prints what it decided at startup, e.g.
`LangSmith tracing: ON -> project 'splendid-ops-agent-dev'`, and `/api/status`
reports the same. **Check one of those before hunting for a missing trace** —
LangSmith uploads from a background thread and swallows its own errors, so a
mistyped key produces no error anywhere, just an empty project.

Use a different `LANGSMITH_PROJECT` locally than on Railway. Otherwise test runs
and real bookings land in one list and neither is readable.

### ⚠️ It uploads real customer data

Tracing sends run contents to LangSmith. For this agent that means real names,
phone numbers, email addresses, home addresses, and the screenshots of
customers' message threads. That is exactly what makes a trace useful and
exactly why `LANGSMITH_TRACING` defaults to `false`.

Two things reduce it:

- Screenshots are **never** uploaded to the OCR step — `services/ocr.py` swaps
  the image for its byte count. (The vision model call still carries the image;
  reading the screenshot is that step's whole job.)
- `LANGSMITH_HIDE_INPUTS=true` and `LANGSMITH_HIDE_OUTPUTS=true` keep the timing
  and shape of every run while dropping all payloads. Good for watching live
  traffic; useless for debugging one booking, since it also hides the
  extraction output.

### Reading a trace

Every run is labelled by `services/tracing.py`, so the run list is searchable
rather than 500 identical rows named "LangGraph":

| Label | Why it's there |
|---|---|
| `dry_run` / `live` | **Check this first, every time.** The two look identical in a trace and only one of them charged a real customer. |
| `channel:google_chat` / `web` / `cli` | Where the turn came in from. |
| `turn:message` / `resume` / `button` | A `resume` is the second half of a booking that paused at an interrupt, so its trace is short and starts mid-graph. Without the label those look like broken runs. |
| `backend:openai` / `openrouter` | Which model set ran. |

The **Threads** tab groups every turn of one booking into a single
conversation — that works because `thread_id` is written into the run metadata,
not just into `configurable`.

### What is traced, and by what

| Layer | How |
|---|---|
| Graph nodes, model calls, token counts | Automatic. LangGraph and LangChain instrument themselves. |
| GHL, Calendar, Maps, Vision calls | `@traceable(run_type="tool")` on the functions in `services/`. LangGraph cannot see inside a node, so without these an API failure is just "the node raised". |
| Run names, tags, metadata | `services/tracing.run_config()`, used at all five `graph.invoke()` / `.stream()` call sites. |

No new dependency — `langsmith` was already installed as a LangChain
dependency.

---

## Things worth knowing before you change anything

**The calendar event description format is load-bearing.** Four other repos
parse it with regex — `move_reminders`, `invoice_automation`,
`job_form_automation`, `ghl_calendar_sync`. Any line matching `Word:` becomes a
field to them, so agent metadata goes in `extendedProperties`, never in the
description. A golden-file test guards this.

**GoHighLevel's Rate field is a dropdown.** A value that is not byte-identical
to one of its options is accepted by the API and then silently discarded,
leaving the job with no rate. `services/rates.py` holds the exact strings and a
test pins them.

**Crews of 5 and 6 are deliberately out of scope.** GoHighLevel accepts them,
but they are priced by hand.

**The confirmation email renders its own values.** GoHighLevel only substitutes
`{{contact.*}}` when GHL itself sends a template; this agent posts raw HTML, so
a merge tag would reach the customer literally.

**Conversations expire daily.** A thread whose last activity was on an earlier
Los Angeles date is discarded on the next message, before anything reads it.
This is a check on the way in rather than a job scheduled at 00:00 — a timer
would have to survive Railway restarting on every deploy and could race the
webhook mid-turn, while the lazy version has nothing to drift. If an unfinished
booking is what gets discarded, the reply says so by name; a finished or idle
conversation is cleared silently. `DAILY_RESET=false` turns it off.

**While the graph is paused, the router does not run.** The channel sends every
message straight to the waiting node as `Command(resume=...)` — sending a plain
message instead would restart the graph and lose the booking. So a question
asked mid-booking ("how many jobs last month?") arrives at `ask_missing` as an
answer to whatever it asked. It parses to nothing, gets acknowledged, and does
not count against the loop guard. `/clear` is the way out, which is why it is
checked before the graph runs at all.

**GoHighLevel writes the deposit text, not this repo.** We hand it an invoice
id; it composes the SMS from the account's *Invoice Received* template. No API
field can change that wording — the template lives in Payments → Invoices &
Estimates → Settings → Notifications. The one part we do control is the
signature, via `sentFrom.fromName` in `services/ghl.py`, which is set to the
company so customers are not signed off to by whichever staff account holds the
API token.

**State persists until something clears it.** The graph's state survives every
turn — that is what lets a booking pause for an hour and resume. The cost is
that a finished booking leaves its customer in `intake` and its results in
`ledger`, and the next job in the same conversation would inherit both.
`extract_screenshot` decides at the top of the intake lane whether the turn
starts a new job and wipes those if so. `retry` is the deliberate exception:
reusing the old ledger is exactly how it re-runs only the failed steps.
