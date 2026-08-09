"""
The checkpointer must outlive the function that created it.

`SqliteSaver.from_conn_string(path)` returns a *context manager*. Calling
`.__enter__()` on it and keeping only the saver works right up until the
context manager is garbage collected — which closes the database underneath a
checkpointer that is still very much in use.

Nothing failed at startup. The app booted, reported healthy, and served
/health for minutes. The first actual message then died with
"Cannot operate on a closed database", far from the cause.

These tests force a collection and then use the database, which is the only
way this shows up short of deploying it.
"""

import gc
import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver


def _write_and_read(checkpointer) -> bool:
    """Exercise a real query, which is what actually touches the connection."""
    cfg = {"configurable": {"thread_id": "lifetime-test", "checkpoint_ns": ""}}
    checkpointer.get_tuple(cfg)
    return True


def test_context_manager_pattern_dies_after_collection(tmp_path):
    """Documents the bug, so the reason for the current code is not lost."""
    db = tmp_path / "cm.sqlite"

    def build():
        # The pattern that shipped: the manager is a local and dies here.
        return SqliteSaver.from_conn_string(str(db)).__enter__()

    checkpointer = build()
    gc.collect()

    try:
        _write_and_read(checkpointer)
    except sqlite3.ProgrammingError as exc:
        assert "closed database" in str(exc)
        return
    # If a future langgraph keeps the connection alive on its own, the current
    # explicit-connection approach is still correct — just no longer load-bearing.


def test_explicit_connection_survives_collection(tmp_path):
    """The pattern app.py and server.py now use."""
    db = tmp_path / "explicit.sqlite"

    holder = {}

    def build():
        holder["conn"] = sqlite3.connect(str(db), check_same_thread=False, timeout=30)
        saver = SqliteSaver(holder["conn"])
        saver.setup()
        return saver

    checkpointer = build()
    gc.collect()

    assert _write_and_read(checkpointer)


def test_connection_is_usable_from_another_thread(tmp_path):
    """
    Requests are served from a thread pool, so the connection is used from
    whichever thread handles the call. Without check_same_thread=False this
    raises only under concurrency — never in a single-threaded test run.
    """
    import threading

    db = tmp_path / "threaded.sqlite"
    conn = sqlite3.connect(str(db), check_same_thread=False, timeout=30)
    checkpointer = SqliteSaver(conn)
    checkpointer.setup()

    errors = []

    def use_it():
        try:
            _write_and_read(checkpointer)
        except Exception as exc:  # noqa: BLE001 - recorded and asserted below
            errors.append(exc)

    thread = threading.Thread(target=use_it)
    thread.start()
    thread.join()

    assert errors == [], f"checkpointer unusable off-thread: {errors}"
