"""
Shared test setup.

TRACING IS FORCED OFF FOR THE WHOLE SUITE.

`services/config.py` calls `load_dotenv()` at import, so the real `.env` is
loaded during tests like anywhere else. With `LANGSMITH_TRACING=true` in it —
which is the normal state once you are using tracing — every `@traceable`
service function in the suite uploaded a run to the live LangSmith project.

That is bad in three separate ways:

  1. It pollutes the real project. 366 tests fire hundreds of orphan runs with
     no parent trace, burying the handful of real bookings you wanted to read.
  2. It is slow and it is network. The suite's whole selling point is that it
     runs in under a second offline.
  3. Test fixtures contain fake customers, so the project fills with data that
     looks real and is not.

`autouse=True` means this runs for every test without any test asking for it.
It sets the environment before any test body executes, and `monkeypatch` at
session scope restores it afterwards.

To trace a test run on purpose — debugging one specific failure — comment this
out temporarily, or point LANGSMITH_PROJECT at a scratch project first.
"""

import pytest


@pytest.fixture(autouse=True, scope="session")
def _no_tracing_during_tests():
    from _pytest.monkeypatch import MonkeyPatch

    mp = MonkeyPatch()
    # Both names: the SDK has read LANGCHAIN_* historically and LANGSMITH_* now,
    # and an old name left set in a shell would silently re-enable uploads.
    mp.setenv("LANGSMITH_TRACING", "false")
    mp.setenv("LANGCHAIN_TRACING_V2", "false")
    yield
    mp.undo()
