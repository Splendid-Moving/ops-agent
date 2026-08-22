"""
A second booking in the same conversation must not inherit the first.

The reported bug: after "Aemond Targareyan is booked", asking for a new job in
the same Chat thread produced Aemond's details again — his name, phone, email
and pickup address — with only the genuinely new fields differing.

Two separate leaks caused it, and the second is worse than the one that was
visible:

  intake  — the previous customer's fields survived and outranked the new
            extraction, so the summary showed the wrong person.

  ledger  — every action still read "success" from the previous booking, so
            act_upsert_contact and friends would skip. The agent would report
            four ticks having done nothing at all.

The exception is a retry, which reuses both on purpose: that is precisely how
it re-runs only the steps that failed.
"""

import pytest

from agent.nodes.extract_screenshot import _fresh_start, _is_new_job
from agent.state import ACTION_CONTACT, ALL_ACTIONS, booking_has_run, new_ledger


def completed_ledger():
    return {name: {"status": "success", "result": {}, "error": None, "attempts": 1}
            for name in ALL_ACTIONS}


def booked_state():
    """State exactly as a finished booking leaves it."""
    return {
        "intake": {"full_name": "Aemond Targareyan", "phone": "+1(818)505-4576",
                   "pickup_address": "527 E Cypress Ave, Burbank CA 91501"},
        "ledger": completed_ledger(),
    }


# ── has anything actually run? ────────────────────────────────────────────────


def test_a_fresh_conversation_has_not_booked_anything():
    assert not booking_has_run(None)
    assert not booking_has_run({})
    assert not booking_has_run(new_ledger())      # all pending


def test_a_completed_booking_is_recognised():
    assert booking_has_run(completed_ledger())


def test_a_failed_action_still_counts_as_having_run():
    """A half-done booking must not be inherited either."""
    ledger = {**new_ledger(), ACTION_CONTACT: {"status": "failed", "error": "boom"}}
    assert booking_has_run(ledger)


# ── new job vs continuation ───────────────────────────────────────────────────


def test_a_new_screenshot_always_starts_a_new_job():
    """
    Screenshot 3 of the report: a new customer's screenshot returned the
    previous customer. An image is unambiguous — a booking already underway
    resumes through ask_missing and never reaches this node.
    """
    assert _is_new_job(booked_state(), has_image=True, text="book this")


def test_asking_for_a_job_after_one_completed_starts_fresh():
    """Screenshot 2: 'Book a job for tomorrow' returned the previous customer."""
    assert _is_new_job(booked_state(), has_image=False, text="Book a job for tomorrow")


def test_saying_it_is_a_new_job_starts_fresh():
    """Screenshot 4: 'I am giving a new job' returned the previous customer."""
    assert _is_new_job(booked_state(), has_image=False, text="I'm am giving a new job")


def test_continuing_an_unbooked_conversation_keeps_what_was_gathered():
    """Mid-booking detail added by text must not wipe the screenshot's fields."""
    partial = {"intake": {"full_name": "Jordan Bray"}, "ledger": new_ledger()}
    assert not _is_new_job(partial, has_image=False, text="3 movers")


@pytest.mark.parametrize("text", ["retry", "Retry", "try again", "re-run", "resend."])
def test_retry_never_resets(text):
    """
    The one case that must keep both. Resetting here would clear the ledger and
    re-run all four actions — texting a second invoice and double-booking the
    truck.
    """
    assert not _is_new_job(booked_state(), has_image=False, text=text)


def test_retry_still_resets_nothing_even_with_an_image():
    """An image outranks nothing: 'retry' is explicit and wins."""
    assert not _is_new_job(booked_state(), has_image=True, text="retry")


# ── what the reset actually clears ────────────────────────────────────────────


def test_reset_returns_every_action_to_pending():
    """The dangerous leak: a stale 'success' makes an action skip silently."""
    ledger = _fresh_start()["ledger"]

    assert not booking_has_run(ledger)
    for name in ALL_ACTIONS:
        assert ledger[name]["status"] == "pending"


def test_reset_clears_approval_so_the_new_job_is_confirmed_again():
    """Inheriting approval would fire four actions with no human check."""
    assert _fresh_start()["approved"] is False


def test_reset_clears_the_duplicate_warning_and_fingerprint():
    reset = _fresh_start()
    assert reset["duplicate_warning"] is None
    assert reset["job_fingerprint"] == ""


def test_reset_does_not_touch_field_confidence():
    """
    It is splatted last into the return dict, so a key here would overwrite the
    freshly computed confidence with an empty one.
    """
    assert "field_confidence" not in _fresh_start()
