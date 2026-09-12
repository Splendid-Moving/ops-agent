"""
Who the confirmation email is actually addressed to.

GHL will not deliver a /conversations/messages email to an address that is not
already on the contact. It answers anything else with

    400 CONVERSATIONS_MSG_INVALID_EMAILTO
    "Cannot send message as emailTo is not under contact's primary or
     additional emails"

and the comparison is verbatim, against the lowercased address GHL stored. So a
lead screenshot reading "Sarah@Gmail.com" — a phone auto-capitalises the first
letter of anything typed into a text thread — books the contact fine, texts the
deposit link fine, and then cannot email the person it just created.

Nothing upstream catches it: the address is valid, it is the right address, and
verify_services.py exercises the send with a test contact whose address is
already lowercase.
"""

import pytest

from services import config, ghl


@pytest.fixture
def ghl_calls(monkeypatch):
    """Record what goes out, and answer with a contact GHL would have stored."""
    calls: dict = {"contact": {"id": "c1", "email": "sarah@gmail.com"}, "posted": None}

    class FakeResponse:
        def __init__(self, ok=True, status_code=201, body=None, text=""):
            self.ok = ok
            self.status_code = status_code
            self._body = body or {}
            self.text = text
            self.headers = {"content-type": "application/json"}

        def json(self):
            return self._body

    def fake_get(url, headers=None, params=None, timeout=None):
        return FakeResponse(body={"contact": calls["contact"]})

    def fake_post(url, headers=None, json=None, timeout=None):
        calls["posted"] = json
        return calls.get("response") or FakeResponse()

    monkeypatch.setattr(config, "dry_run", lambda: False)
    monkeypatch.setattr(ghl.requests, "get", fake_get)
    monkeypatch.setattr(ghl.requests, "post", fake_post)
    calls["FakeResponse"] = FakeResponse
    return calls


# ── Normalisation ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "as_written",
    [
        "Sarah@Gmail.com",    # phone keyboard capitalised the first letter
        "SARAH@GMAIL.COM",    # pasted from a form that shouts
        " sarah@gmail.com ",  # line-wrapped in the screenshot
        "sarah@Gmail.com",
    ],
)
def test_an_address_is_stored_the_way_ghl_stores_it(as_written, ghl_calls):
    """
    The contact is written with the normalised address, so it is the same
    string the email is later checked against.
    """
    ghl.upsert_contact(
        first_name="Sarah", last_name="Chen",
        phone="+1(818)555-0142", email=as_written,
    )
    assert ghl_calls["posted"]["email"] == "sarah@gmail.com"


@pytest.mark.parametrize(
    "as_written",
    ["Sarah@Gmail.com", "SARAH@GMAIL.COM", " sarah@gmail.com ", "sarah@gmail.com"],
)
def test_the_primary_address_is_never_put_in_emailTo(as_written, ghl_calls):
    """
    The case that kept failing. However the address was written, it is the
    contact's primary — so the field GHL validates is left off altogether and
    the mail goes to the primary, which is where it was going anyway.
    """
    ghl.send_email("c1", subject="Your move", html="<p>hi</p>", email_to=as_written)
    assert "emailTo" not in ghl_calls["posted"]


# ── Additional addresses ───────────────────────────────────────────────────────

def test_an_additional_address_is_sent_normalised(ghl_calls):
    """A second address on the contact is deliverable, and is named explicitly."""
    ghl_calls["contact"] = {
        "id": "c1",
        "email": "sarah@gmail.com",
        "additionalEmails": [{"email": "sarah.chen@work.com"}],
    }
    ghl.send_email("c1", subject="Your move", html="<p>hi</p>",
                   email_to="Sarah.Chen@Work.com")
    assert ghl_calls["posted"]["emailTo"] == "sarah.chen@work.com"


def test_an_additional_address_is_named_when_there_is_no_primary(ghl_calls):
    """
    Leaving emailTo off means "send to the primary", so a contact that has no
    primary has to be told which address to use.
    """
    ghl_calls["contact"] = {
        "id": "c1", "email": "", "additionalEmails": ["sarah@gmail.com"],
    }
    ghl.send_email("c1", subject="Your move", html="<p>hi</p>",
                   email_to="Sarah@Gmail.com")
    assert ghl_calls["posted"]["emailTo"] == "sarah@gmail.com"


def test_additional_emails_are_read_in_both_shapes_ghl_returns():
    """Bare strings from one endpoint, {"email": ...} objects from another."""
    contact = {
        "email": "Sarah@Gmail.com",
        "additionalEmails": ["Work@Chen.com", {"email": "alt@chen.com"}],
    }
    assert ghl.contact_emails(contact) == [
        "sarah@gmail.com", "work@chen.com", "alt@chen.com",
    ]


def test_a_contact_with_no_email_lists_nothing():
    assert ghl.contact_emails({"id": "c1", "email": None}) == []


# ── When it is genuinely the wrong address ─────────────────────────────────────

def test_the_rejection_names_the_addresses_ghl_would_have_taken(ghl_calls):
    """
    An address that really is not on the contact still fails — GHL decides
    that, not us. What changes is that the dispatcher is told which addresses
    the contact does have, instead of a canonicalCode and a traceId.
    """
    FakeResponse = ghl_calls["FakeResponse"]
    ghl_calls["response"] = FakeResponse(
        ok=False,
        status_code=400,
        text='{"statusCode":400,"message":"Cannot send message as emailTo is not '
             'under contact\'s primary or additional emails","canonicalCode":'
             '"CONVERSATIONS_MSG_INVALID_EMAILTO"}',
    )

    with pytest.raises(ghl.GHLError) as exc:
        ghl.send_email("c1", subject="Your move", html="<p>hi</p>",
                       email_to="typo@gmial.com")

    message = str(exc.value)
    assert "typo@gmial.com" in message
    assert "sarah@gmail.com" in message


def test_the_contact_is_not_fetched_when_no_address_was_asked_for(monkeypatch, ghl_calls):
    """No emailTo, nothing to check — GHL routes it to the primary itself."""
    def no_reads(*args, **kwargs):
        raise AssertionError("send_email should not read the contact here")

    monkeypatch.setattr(ghl.requests, "get", no_reads)
    ghl.send_email("c1", subject="Your move", html="<p>hi</p>")
    assert "emailTo" not in ghl_calls["posted"]
