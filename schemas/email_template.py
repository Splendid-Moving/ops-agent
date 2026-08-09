"""
Confirmation email sent to the customer once a job is booked.

Content matches the company's existing GoHighLevel booking-confirmation
snippet. Everything customer-facing lives in this one file, so rewording the
email never means touching graph or service code.

── Why the merge fields are resolved here, not by GoHighLevel ─────────────────

The GHL snippet uses {{contact.first_name}} and friends, which GHL substitutes
when *it* sends the template from a workflow. This agent sends through
POST /conversations/messages with an html body, which is delivered as-is — a
{{contact.*}} tag would reach the customer literally. So the values are
rendered here from the intake we already hold, which is also the only version
that can omit a row rather than print a blank one.

That difference matters for real bookings: labor-only jobs have no drop-off
address, and most jobs have no extra stop. The GHL template prints those as
empty lines; this one leaves them out.

Facts here must match BRAND_INFO.md: (323) 645-2636, info@splendidmoving.com.
"""

import os

from services import config, rates

PHONE = "(323) 645-2636"
PHONE_E164 = "+13236452636"
EMAIL = "info@splendidmoving.com"

#: The truck logo, matching the GoHighLevel confirmation template.
#:
#: Served by this app (see /email-logo.png in app.py) rather than linked from
#: GHL's media library, so it ships with the code and cannot be broken by
#: someone tidying up media. Email clients cannot read a local file, so it has
#: to be a public URL either way.
#:
#: Many clients block images by default, so the email must still read correctly
#: without it — which is why nothing but branding lives up there.
PUBLIC_BASE_URL = os.getenv(
    "PUBLIC_BASE_URL", "https://ops-agent-production-5261.up.railway.app"
).rstrip("/")

LOGO_URL = os.getenv("EMAIL_LOGO_URL", f"{PUBLIC_BASE_URL}/email-logo.png")

NAVY = "#0d2149"

#: The terms, verbatim from the existing GHL snippet. Bold marks the operative
#: clause in each — these are the ones customers dispute later, and the email is
#: the record.
TERMS: list[str] = [
    "We request that you give us at least <strong>72 hours in advance if you "
    "intend to cancel your reservation</strong> for any reason or change your "
    "move date.",

    "You are <strong>billed from the moment the movers arrive at your pick-up "
    "location</strong> until the completion of the move-in process at your "
    "final destination.",

    "There is a <strong>minimum labor charge of 3 hours.</strong> After that, "
    "we bill in 15-minute increments at the same rate.",

    "<strong>Additional charges may apply to items considered excessively "
    "heavy.</strong> This includes, but is not limited to, any item weighing "
    "150 lbs or more that needs to be carried up or down stairs, as well as any "
    "item weighing over 250 lbs, regardless of the presence of stairs. Examples "
    "of such items include double-door refrigerators, pianos, safes, "
    "treadmills, etc. Any additional charges must be coordinated with the "
    "office.",

    "We will move as fast and as safely as possible.",

    "Transportation is charged as <strong>Double Drive Time (DDT)</strong> if "
    "the distance between your pick-up and delivery address is more than 15 "
    "miles. The additional time goes on top of the 3-hour minimum.",

    "If during loading or unloading we receive a <strong>parking "
    "ticket</strong>, The Customer is obligated to reimburse us for the entire "
    "amount stated on the ticket.",

    "Due to varying quantities of goods, <strong>we cannot guarantee that all "
    "items will fit in the truck in one trip.</strong> If needed, we can "
    "provide an additional truck, subject to availability. The arrival time and "
    "size of the extra truck depend on the company's schedule, and it will be "
    "provided at an additional charge.",

    "Please note that our company <strong>does not assume responsibility for "
    "the safe transportation of plants.</strong> Due to the fact that our "
    "moving trucks can become quite shaky during transit, there is no secure "
    "method to ensure the safety of plants inside the vehicle.",

    "If your move involves loading into a <strong>POD, rental truck, or any "
    "container not operated by Splendid Moving</strong>, please note that "
    "cardboard boxes or moving blankets must be available on-site for our team "
    "to properly wrap and protect your furniture prior to loading. You are "
    "welcome to provide your own materials or purchase them from us in advance. "
    "If protective materials are not available and you choose to proceed "
    "without them, Splendid Moving assumes no responsibility for any damage to "
    "items once they have been placed inside the container. Our liability "
    "covers your belongings only while they are in our care — from the point of "
    "pickup through placement into the container.",

    "We only accept <strong>physical cash or debit/credit cards</strong> as "
    "forms of payment; we DO NOT accept checks, Zelle, Venmo, or any other "
    "payment methods.",

    "According to the rules of California Public Utility Commission that "
    "regulates the rights and obligations of the moving companies, the Customer "
    "has to pay the full amount for the moving services after the movers have "
    "completed the job. If the Customer does not pay the full amount, the "
    "moving company reserves the right not to honor any Customer's claim(s).",

    "<strong>Valuation</strong> — all items that are located inside or outside "
    "of all the facilities, and areas where moving takes place will be "
    "automatically covered for $0.60 cents per pound per article at no "
    "additional cost.",
]


def subject(intake: dict) -> str:
    return f"Booking Confirmation — Splendid Moving, {intake.get('move_date', '')}"


def _row(label: str, value: str) -> str:
    return (
        '<tr>'
        '<td style="padding:5px 16px 5px 0;color:#666;white-space:nowrap;'
        'vertical-align:top;font-size:15px">'
        f"{label}</td>"
        f'<td style="padding:5px 0;color:#111;font-size:15px"><strong>{value}</strong></td>'
        "</tr>"
    )


def _detail_rows(intake: dict) -> str:
    """
    Booking details, skipping anything absent.

    A blank "Extra stop:" line reads as though something was forgotten, and on
    a labor-only job there is no drop-off address at all.
    """
    is_labor = bool(intake.get("is_labor"))
    name = intake.get("full_name", "")
    rate = rates.format_rate(intake.get("movers", ""))

    rows = [_row("Customer", name)] if name else []
    rows += [
        _row("Move date", intake.get("move_date", "")),
        _row("Arrival time", intake.get("arrival_time", "")),
        _row("Phone", intake.get("phone", "")),
        _row("From", intake.get("pickup_address", "")),
    ]
    if intake.get("extra_stop"):
        rows.append(_row("Extra stop", intake["extra_stop"]))
    if not is_labor and intake.get("dropoff_address"):
        rows.append(_row("To", intake["dropoff_address"]))
    if rate:
        rows.append(_row("Rate", rate))
    if movers := intake.get("movers"):
        rows.append(_row("Movers", str(movers)))
    if is_labor:
        rows.append(_row("Service", "Labor only (no transport)"))

    return "".join(rows)


def html(intake: dict) -> str:
    """Confirmation email body. Deposit amount comes from config, never hardcoded."""
    deposit = config.deposit_amount()
    terms = "".join(
        f'<li style="margin:0 0 12px 0">{t}</li>' for t in TERMS
    )

    return f"""\
<div style="font-family:Inter,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
            font-size:15px;line-height:1.55;color:#111;max-width:620px">

  <img src="{LOGO_URL}" alt="Splendid Moving" width="220"
       style="display:block;max-width:220px;height:auto;margin:0 0 28px 0;border:0">

  <h1 style="font-size:22px;font-weight:700;color:{NAVY};margin:0 0 20px 0;
             letter-spacing:.02em">BOOKING CONFIRMATION</h1>

  <table style="border-collapse:collapse;margin:0 0 28px 0">
    {_detail_rows(intake)}
  </table>

  <p style="margin:0 0 24px 0">
    To hold your spot we ask for a <strong>${deposit:.0f} deposit</strong>. We've
    sent a payment link by text — it takes about a minute, and the deposit comes
    off your final balance.
  </p>

  <ul style="padding-left:20px;margin:0 0 28px 0">
    {terms}
  </ul>

  <p style="margin:0 0 24px 0">
    We look forward to becoming your chosen moving company and providing moving
    services to you, your family, and your friends for many years to come!
  </p>

  <hr style="border:0;border-top:1px solid #e3e3e3;margin:0 0 20px 0">

  <p style="margin:0;color:#444">
    Best regards,<br>
    <strong style="color:{NAVY}">Splendid Moving</strong><br>
    <a href="tel:{PHONE_E164}" style="color:#444;text-decoration:none">{PHONE}</a><br>
    <a href="mailto:{EMAIL}" style="color:#444;text-decoration:none">{EMAIL}</a>
  </p>
</div>"""
