"""Connection management for harness.db. Mirrors revenew/db.py deliberately --
same WAL/foreign-key setup -- but is a separate module so that nothing in
revenew/ can import a harness connection helper by accident."""

from __future__ import annotations

import sqlite3
from pathlib import Path

HARNESS_SCHEMA_PATH = Path(__file__).resolve().parent.parent / "db" / "harness_schema.sql"
DEFAULT_HARNESS_DB_PATH = Path(__file__).resolve().parent.parent / "harness.db"


def connect(db_path: str | Path = DEFAULT_HARNESS_DB_PATH, *, row_factory: bool = True) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), timeout=30.0)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    if row_factory:
        conn.row_factory = sqlite3.Row
    return conn


def init_harness_db(db_path: str | Path = DEFAULT_HARNESS_DB_PATH, *, reset: bool = False) -> None:
    path = Path(db_path)
    if reset and path.exists():
        path.unlink()
        for suffix in ("-wal", "-shm"):
            sidecar = path.with_name(path.name + suffix)
            if sidecar.exists():
                sidecar.unlink()

    conn = connect(path, row_factory=False)
    try:
        conn.executescript(HARNESS_SCHEMA_PATH.read_text(encoding="utf-8"))
        conn.commit()
    finally:
        conn.close()
