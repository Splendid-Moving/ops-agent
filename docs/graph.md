# Ops agent graph

Generated from `agent/graph.py` by `draw_graph.py`. Do not edit by hand.

- **Solid arrow**: always goes to the next step.
- **Dotted arrow**: a decision. The step picks one route (or several, for the
  parallel booking actions after `execute`).
- `ask_missing` and `confirm` pause and wait for a human reply before continuing.

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	router(router)
	analytics(analytics)
	site(site)
	chat(chat)
	extract_screenshot(extract_screenshot)
	resolve_addresses(resolve_addresses)
	validate(validate)
	ask_missing(ask_missing)
	confirm(confirm)
	give_up(give_up)
	execute(execute)
	act_calendar(act_calendar)
	act_app_job(act_app_job)
	act_invoice(act_invoice)
	act_email(act_email)
	act_sms(act_sms)
	report(report)
	__end__([<p>__end__</p>]):::last
	__start__ --> router;
	act_app_job --> act_sms;
	act_calendar --> act_sms;
	act_email --> act_sms;
	act_invoice --> act_sms;
	act_sms --> report;
	ask_missing --> resolve_addresses;
	confirm -.-> __end__;
	confirm -.-> ask_missing;
	confirm -.-> execute;
	execute -.-> act_app_job;
	execute -.-> act_calendar;
	execute -.-> act_email;
	execute -.-> act_invoice;
	execute -.-> report;
	extract_screenshot --> resolve_addresses;
	resolve_addresses --> validate;
	router -.-> analytics;
	router -.-> chat;
	router -. &nbsp;intake&nbsp; .-> extract_screenshot;
	router -.-> site;
	validate -.-> ask_missing;
	validate -.-> confirm;
	validate -.-> give_up;
	analytics --> __end__;
	chat --> __end__;
	give_up --> __end__;
	report --> __end__;
	site --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```
