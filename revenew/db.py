"""Connection management. The one place a file path becomes a `sqlite3.Connection`.

Runtime code imports `connect()` from here and nothing else touches
`sqlite3.connect` directly -- that discipline is what keeps the isolation
promise in db/schema.sql true in practice: grep for `sqlite3.connect` and every
result is either this function or a harness module that says explicitly why it
needs a second file attached.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "db" / "schema.sql"

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "revenew.db"


def connect(db_path: str | Path = DEFAULT_DB_PATH, *, row_factory: bool = True) -> sqlite3.Connection:
    """Open the runtime database with WAL mode and foreign keys on.

    WAL is what lets the FastAPI process and a concurrent replay/CLI process
    both hold the file open without one blocking the other on every write --
    the non-functional requirement is 10k decisions/night on a single writer,
    not concurrent writers, but the webhook receiver and the dashboard reader
    are two connections even in that single-writer world.
    """
    # check_same_thread=False: FastAPI's async endpoints run on the event-loop
    # thread while a sync generator dependency's setup/teardown runs via
    # run_in_threadpool -- so a connection created for one request can be
    # created and torn down in different threads even though it is never
    # used concurrently by more than one. sqlite3's default same-thread check
    # exists to catch genuine concurrent misuse from multiple threads at
    # once, which this is not: each connection here is request-scoped, used
    # sequentially, and closed at the end of that one request.
    conn = sqlite3.connect(str(db_path), timeout=30.0, check_same_thread=False)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 30000")
    # NORMAL is the standard pairing with WAL: durability against a crashed
    # process is unchanged (the WAL survives), the only risk is losing the
    # last few commits on a full power loss, and in return every commit stops
    # forcing an fsync. Left at the SQLite default (FULL) this workload -- a
    # few thousand small commits per replayed day -- was the dominant cost in
    # a 30-day run overrunning its ~40s budget by 5x; see harness/run_replay.py.
    conn.execute("PRAGMA synchronous = NORMAL")
    if row_factory:
        conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str | Path = DEFAULT_DB_PATH, *, reset: bool = False) -> None:
    """Create the schema. `reset=True` deletes any existing file first.

    Idempotent by construction otherwise: schema.sql uses bare CREATE TABLE, so
    running init_db against an already-initialized file raises rather than
    silently doing nothing -- that's deliberate, since a silent no-op here
    would hide a schema drift between what's on disk and what schema.sql says
    the shape should be.
    """
    path = Path(db_path)
    if reset and path.exists():
        path.unlink()
        for suffix in ("-wal", "-shm"):
            sidecar = path.with_name(path.name + suffix)
            if sidecar.exists():
                sidecar.unlink()

    conn = connect(path, row_factory=False)
    try:
        conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        conn.commit()
    finally:
        conn.close()
