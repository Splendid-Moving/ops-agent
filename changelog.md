# Changelog

## 2026-11-25
- Any crew size is now accepted. 2-4 movers keep their standard rates; for any other size the bot asks for the rate, typed as `cash / card` (e.g. `250/270`).
- Above 4 movers the bot also asks how many trucks. With 2+ trucks the calendar event is titled `Name (N trucks)`, the calendar Notes start with `- N trucks`, the confirmation email gets a Trucks row, and GHL's Trucks field is set.
- GHL's Rate dropdown has no option for a custom rate, so it's left blank there (also Movers above 6, Trucks above 3); the confirm summary says so.

## 2026-11-16
- Added `draw_graph.py`, which saves the real LangGraph wiring as a Mermaid diagram in `docs/graph.md` (GitHub renders it). README links to it.
