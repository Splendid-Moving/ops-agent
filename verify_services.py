#!/usr/bin/env python
"""
Phase 0 smoke test — proves every service can reach its API before any graph
code depends on it.

Read-only by default. Writes are attempted only with --write, and refuse to run
unless DRY_RUN=false is explicitly set, so there is no single flag that can
surprise a real customer.

    python verify_services.py            # read-only checks
    python verify_services.py --write    # also exercise writes (respects DRY_RUN)
"""

import argparse
import sys
from datetime import datetime, timedelta

from services import calendar, config, formatting, ghl, maps, rates

OK = "\033[32m✓\033[0m"
BAD = "\033[31m✗\033[0m"
WARN = "\033[33m!\033[0m"

failures: list[str] = []


def check(label: str, fn):
    try:
        detail = fn()
        print(f"  {OK} {label}" + (f" — {detail}" if detail else ""))
        return True
    except Exception as exc:
        print(f"  {BAD} {label} — {type(exc).__name__}: {exc}")
        failures.append(label)
        return False


def section(title: str):
    print(f"\n\033[1m{title}\033[0m")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="also exercise write paths")
    args = parser.parse_args()

    print("\033[1mSplendid Moving ops-agent — service verification\033[0m")

    # ── Environment ────────────────────────────────────────────────────────────
    section("Environment")
    required = {
        "GHL_ACCESS_TOKEN": config.ghl_token(),
        "GHL_LOCATION_ID": config.ghl_location_id(),
        "GHL_USER_ID": config.ghl_user_id(),
        "GOOGLE_CREDENTIALS_B64": config.google_credentials_b64(),
        "GOOGLE_CALENDAR_ID": config.calendar_id(),
        "GOOGLE_MAPS_API_KEY": config.maps_api_key(),
    }
    for name, value in required.items():
        if value:
            print(f"  {OK} {name} — {len(value)} chars")
        else:
            print(f"  {BAD} {name} — NOT SET")
            failures.append(name)

    mode = "DRY RUN (writes are logged, not sent)" if config.dry_run() else "LIVE — WRITES ARE REAL"
    marker = OK if config.dry_run() else WARN
    print(f"  {marker} DRY_RUN={config.dry_run()} → {mode}")
    print(f"  {OK} deposit = ${config.deposit_amount():.0f}")
    print(f"  {OK} model backend = {config.model_backend()}")

    # ── Pure logic ─────────────────────────────────────────────────────────────
    section("Rate table")
    for movers in rates.VALID_MOVER_COUNTS:
        print(f"  {OK} {movers} movers — {rates.format_rate(movers)}")

    # ── Google Calendar ────────────────────────────────────────────────────────
    section("Google Calendar (read-only)")
    now = datetime.now(calendar.LA_TZ)

    def calendar_auth():
        calendar.get_service()
        return "authenticated"

    if check("service account auth", calendar_auth):
        first_of_this_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        last_month_end = first_of_this_month - timedelta(seconds=1)
        last_month_start = last_month_end.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        def count_last_month():
            events = calendar.list_events(last_month_start, last_month_end)
            jobs = calendar.list_jobs(last_month_start, last_month_end)
            labor = sum(1 for j in jobs if j["is_labor"])
            return (
                f"{last_month_start:%B %Y}: {len(jobs)} jobs "
                f"({labor} labor) out of {len(events)} total events"
            )

        check("list + parse last month", count_last_month)

        def sample_job():
            jobs = calendar.list_jobs(last_month_start, last_month_end)
            if not jobs:
                return "no jobs found in range — nothing to sample"
            j = jobs[0]
            return f"{j['customer']!r} on {j['move_date']} | {j['movers']} movers | rate {j['rate']!r}"

        check("parse a real job event", sample_job)

    # ── GoHighLevel ────────────────────────────────────────────────────────────
    section("GoHighLevel (read-only)")

    def business():
        b = ghl.get_business_details()
        return b.get("name") or "(location has no business name set)"

    check("fetch location / business details", business)

    # ── Google Maps ────────────────────────────────────────────────────────────
    section("Google Maps")

    def distance():
        d = maps.get_distance(
            "412 N Maple Ave, Burbank CA 91505", "1830 Pine St, Glendale CA 91206"
        )
        if not d:
            raise RuntimeError("returned empty — check API key and billing")
        return d

    check("distance matrix", distance)

    # ── Writes ─────────────────────────────────────────────────────────────────
    if args.write:
        section("Writes")
        if not config.dry_run():
            print(f"  {WARN} DRY_RUN is false. Refusing to auto-run live writes.")
            print("      Exercise these deliberately, with a known test contact.")
        else:
            check(
                "upsert_contact (dry run)",
                lambda: ghl.upsert_contact(
                    first_name="Test", last_name="Contact",
                    phone="+1(555)555-0100", email="test@example.com",
                )["contact_id"],
            )
            check(
                "create_event (dry run)",
                lambda: calendar.create_event(
                    title="Test Contact",
                    description=calendar.build_description(
                        customer="Test Contact", phone="+1(555)555-0100",
                        date="01/01/2027", from_address="A", to_address="B",
                        rate=rates.format_rate(3), movers="3",
                        deposit=f"${config.deposit_amount():.0f}",
                    ),
                    start_iso="2027-01-01T08:00:00-08:00",
                    end_iso="2027-01-01T09:00:00-08:00",
                )["event_id"],
            )

    # ── Summary ────────────────────────────────────────────────────────────────
    print()
    if failures:
        print(f"\033[31m{len(failures)} check(s) failed:\033[0m " + ", ".join(failures))
        return 1
    print("\033[32mAll checks passed.\033[0m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
