"""Shared fixtures: a seeded runtime DB and its matching harness DB, small
enough to build in a test run (hundreds of customers, not thousands)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from harness.db import connect as hconnect
from harness.db import init_harness_db
from harness.fixture import generate_population, seed_runtime_db, write_ground_truth
from revenew.db import connect as rconnect
from revenew.db import init_db

NOW = datetime(2026, 1, 1, tzinfo=UTC)
SEED = 20260101


@pytest.fixture
def seeded(tmp_path: Path):
    """(connection, SeedData) for a revenew.db already loaded with synthetic
    customers/orders/products. Bundled together because sqlite3.Connection
    does not support arbitrary attributes, and several tests need both the
    live connection and the intended-segment labels the generator produced."""
    db_path = tmp_path / "revenew.db"
    init_db(db_path, reset=True)
    conn = rconnect(db_path)
    data = generate_population(seed=SEED, n_customers=600, now=NOW)
    seed_runtime_db(conn, data)
    yield conn, data
    conn.close()


@pytest.fixture
def seeded_conn(seeded):
    return seeded[0]


@pytest.fixture
def harness_conn(tmp_path: Path):
    db_path = tmp_path / "harness.db"
    init_harness_db(db_path, reset=True)
    conn = hconnect(db_path)
    write_ground_truth(conn, seed=SEED)
    yield conn
    conn.close()
