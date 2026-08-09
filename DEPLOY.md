# Deploying to Railway + Google Chat

Everything here is done once. Steps marked **[you]** need a browser and your
Google/Railway logins — they can't be done from code.

The order matters: Railway gives you a URL, and Google Chat needs that URL.

---

## Overview

```
  Google Chat  ──webhook──►  Railway (app.py)  ──►  the agent
       ▲                                              │
       └──────── posts replies via Chat API ──────────┘
```

Two separate credentials are involved, and mixing them up is the most common
way this gets stuck:

| Credential | What it's for | Direction |
|---|---|---|
| `GOOGLE_CHAT_AUDIENCE` | proving requests really came from Google | inbound |
| `GOOGLE_CHAT_CREDENTIALS_B64` | letting the app post messages back | outbound |

---

## 1. Push the code **[you]**

Already done if you're reading this in GitHub. Otherwise:

```bash
git push -u origin main
```

---

## 2. Create the Railway service **[you]**

1. Railway → **New Project** → **Deploy from GitHub repo** → `Splendid-Moving/ops-agent`
2. It will detect Python and use `railway.json` automatically.
3. **Settings → Networking → Generate Domain.** Copy the URL —
   something like `https://ops-agent-production.up.railway.app`.

Your webhook URL is that domain with `/google-chat` on the end.

### Add a volume (do this now, not later)

**Settings → Volumes → New Volume**, mount path `/data`.

Without it, `agent_threads.sqlite` lives on the container's temporary disk and
is wiped on every deploy. A booking that's paused waiting on someone's answer
would be silently lost mid-conversation.

### Environment variables

Copy everything from your local `.env`, plus:

| Variable | Value |
|---|---|
| `CHECKPOINT_DB` | `/data/agent_threads.sqlite` |
| `GOOGLE_CHAT_AUDIENCE` | *(filled in at step 3)* |
| `CHAT_VERIFY_REQUESTS` | `true` |
| `DRY_RUN` | **`true` for now** — see step 5 |
| `WEB_UI_TOKEN` | leave blank unless you want the browser UI public |

> **Start with `DRY_RUN=true`.** The first deploy should not be able to text
> customers. You'll flip it once you've watched it work end to end.

---

## 3. Create the Google Chat app **[you]**

In [Google Cloud Console](https://console.cloud.google.com), using the same
project as your Calendar service account:

1. **APIs & Services → Enable APIs** → enable **Google Chat API**.
2. **Google Chat API → Configuration**, then fill in:

| Field | Value |
|---|---|
| App name | `Ops Agent` |
| Avatar URL | any image URL |
| Description | `Books jobs and answers calendar questions` |
| Functionality | ✅ Receive 1:1 messages ✅ Join spaces and group conversations |
| Connection settings | **HTTP endpoint URL** |
| HTTP endpoint URL | `https://<your-railway-domain>/google-chat` |
| Authentication Audience | **HTTP endpoint URL** |
| Visibility | your Workspace domain, or specific people to start |

3. Set `GOOGLE_CHAT_AUDIENCE` in Railway to the **exact** endpoint URL you
   entered — `https://<your-railway-domain>/google-chat`, no trailing slash.

> If you pick **Project Number** as the audience type instead, set
> `GOOGLE_CHAT_AUDIENCE` to the numeric project number. The code handles both,
> but it must match what you chose here or every request returns 401.

### The service account for posting replies

The app needs to post messages *back* into Chat, which needs a service account
with the `chat.bot` scope.

If your existing Calendar service account is on this same project, it will work
— leave `GOOGLE_CHAT_CREDENTIALS_B64` blank and it reuses
`GOOGLE_CREDENTIALS_B64`. Otherwise create one and:

```bash
base64 -i new-service-account.json | tr -d '\n'
```

…and paste that into `GOOGLE_CHAT_CREDENTIALS_B64`.

---

## 4. Test it

In Google Chat, search for **Ops Agent** and start a direct message.

| Try this | What should happen |
|---|---|
| `hello` | It introduces itself |
| `how many jobs did we have last month?` | A real count from the calendar |
| *drop in a screenshot* | It reads it, asks for anything missing, then shows a confirmation card with **Book it** / **Cancel** |

With `DRY_RUN=true` the confirmation card still appears and the buttons still
work — nothing is actually created. That's the point: you can walk the whole
flow safely.

### If nothing happens at all

Check Railway's logs first. The usual causes, in order:

- **401 in the logs** → `GOOGLE_CHAT_AUDIENCE` doesn't match the Chat config exactly.
- **No request arrives** → the endpoint URL in the Chat config is wrong, or missing `/google-chat`.
- **"No Chat service account"** → `GOOGLE_CHAT_CREDENTIALS_B64` / `GOOGLE_CREDENTIALS_B64` isn't set.
- **Placeholder appears, never updates** → the service account lacks the `chat.bot` scope.

---

## 5. Going live

Only after you've watched a full booking work in dry run:

1. Set `DRY_RUN=false` in Railway.
2. Do **one** real booking using your own phone and email as the customer.
3. Check all four things landed: GoHighLevel contact, calendar event, deposit
   text, confirmation email.
4. Delete the test artifacts.

---

## Things to know

**One instance only.** `railway.json` pins `numReplicas: 1`. The SQLite
checkpointer can't be shared across instances — two replicas would each hold
half the conversations. Scaling up means moving to Postgres first.

**The browser UI is off by default.** Set `WEB_UI_TOKEN` to a long random
string to enable it at `https://<domain>/?token=<that string>`. It's genuinely
easier to debug against than Chat, but it's a page that can book real jobs, so
it stays off unless you're actively using it.

**Both channels share one brain.** A booking started in the browser can be
finished in Chat, because they use the same graph and checkpointer.
