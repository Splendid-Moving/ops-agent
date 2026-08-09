"""
Confirmation email sent to the customer once a job is booked.

PLACEHOLDER COPY. Written to be inoffensive and factually correct so the flow
can be tested end to end; the wording is expected to be replaced. Everything
customer-facing is in this one file for exactly that reason — rewriting the
email should never mean touching graph or service code.

Facts here must match BRAND_INFO.md: (323) 645-2636, info@splendidmoving.com.
"""

from services import config, rates

PHONE = "(323) 645-2636"
EMAIL = "info@splendidmoving.com"


def subject(intake: dict) -> str:
    return f"Your move is confirmed — {intake.get('move_date', '')}"


def _row(label: str, value: str) -> str:
    return (
        f'<tr>'
        f'<td style="padding:6px 16px 6px 0;color:#666;white-space:nowrap;vertical-align:top">{label}</td>'
        f'<td style="padding:6px 0;color:#111"><strong>{value}</strong></td>'
        f"</tr>"
    )


def html(intake: dict) -> str:
    """Confirmation email body. Deposit amount comes from config, never hardcoded."""
    first_name = (intake.get("full_name", "") or "there").split()[0]
    is_labor = bool(intake.get("is_labor"))
    deposit = config.deposit_amount()
    rate = rates.format_rate(intake.get("movers", ""))

    rows = [
        _row("Date", intake.get("move_date", "")),
        _row("Arrival window", intake.get("arrival_time", "")),
        _row("Crew", f"{intake.get('movers', '')} movers"),
    ]
    if rate:
        rows.append(_row("Rate", rate))
    rows.append(_row("Pickup" if not is_labor else "Address", intake.get("pickup_address", "")))
    if intake.get("extra_stop"):
        rows.append(_row("Extra stop", intake["extra_stop"]))
    if not is_labor and intake.get("dropoff_address"):
        rows.append(_row("Drop-off", intake["dropoff_address"]))

    job_word = "labor job" if is_labor else "move"

    return f"""\
<div style="font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
            font-size:15px;line-height:1.55;color:#111;max-width:560px">
  <p>Hi {first_name},</p>

  <p>Your {job_word} with Splendid Moving is confirmed. Here are the details:</p>

  <table style="border-collapse:collapse;margin:18px 0">
    {"".join(rows)}
  </table>

  <p>
    To hold your spot we ask for a <strong>${deposit:.0f} deposit</strong>. We've sent a
    payment link by text — it takes about a minute. The deposit comes off your
    final balance.
  </p>

  <p>
    Our crew will call when they're on the way. If anything changes, or you think
    of something we should know about — stairs, elevator reservations, parking,
    anything unusually heavy — just reply to this email or give us a call.
  </p>

  <p style="margin-top:24px">
    Thanks for choosing us,<br>
    <strong>Splendid Moving</strong><br>
    <a href="tel:+13236452636" style="color:#111">{PHONE}</a> ·
    <a href="mailto:{EMAIL}" style="color:#111">{EMAIL}</a>
  </p>
</div>"""
