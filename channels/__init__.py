"""
Channels — the ways a human reaches the agent.

Each channel is an adapter: it translates one messaging surface into the two
things the graph understands (a new message, or a resume value) and translates
the graph's output back. No channel contains booking logic. Adding or removing
one must never require touching `agent/`.

  server.py                 browser UI, local dev
  channels/google_chat.py   Google Chat app, production
"""
