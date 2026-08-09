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
| `tests/` | 319 checks that run in under a second. |
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
pytest                        # 319 fast tests, no network
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
