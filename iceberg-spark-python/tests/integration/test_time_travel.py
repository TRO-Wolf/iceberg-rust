"""Integration tests for time travel: VERSION AS OF and TIMESTAMP AS OF.

Requires PyIceberg + SQLite catalog.  Skipped automatically when unavailable.

Run with:
    uv run pytest tests/integration/test_time_travel.py -v
"""

from __future__ import annotations

import os
import tempfile
import time
from datetime import datetime, timedelta

import pytest

try:
    import pyiceberg.catalog.sql  # noqa: F401
    from iceberg_spark import IcebergSession
    HAS_SQL_CATALOG = True
except ImportError:
    HAS_SQL_CATALOG = False

pytestmark = pytest.mark.skipif(
    not HAS_SQL_CATALOG,
    reason="pyiceberg[sql-sqlite] / sqlalchemy not installed — skipping integration tests",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def warehouse():
    with tempfile.TemporaryDirectory(prefix="iceberg_tt_test_") as d:
        yield d


@pytest.fixture(scope="module")
def session(warehouse):
    db_path = os.path.join(warehouse, "catalog.db")
    sess = (
        IcebergSession.builder()
        .catalog(
            "sql",
            name="tt_catalog",
            uri=f"sqlite:///{db_path}",
            warehouse=f"file://{warehouse}",
        )
        .build()
    )
    yield sess
    sess.stop()


@pytest.fixture(scope="module", autouse=True)
def setup_namespace(session):
    try:
        session.sql("CREATE DATABASE IF NOT EXISTS tt")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTimeTravel:
    """Test VERSION AS OF and TIMESTAMP AS OF queries."""

    @pytest.fixture(scope="class")
    def time_travel_table(self, session):
        """Create a table with two distinct snapshots separated by time.

        Returns a dict with snapshot info for use in tests.
        """
        tbl = "tt.tt_travel"
        # Clean up if exists
        try:
            session.sql(f"DROP TABLE IF EXISTS {tbl}")
        except Exception:
            pass

        session.sql(
            f"CREATE TABLE {tbl} (id INT, name STRING) USING iceberg"
        )

        # Snapshot 1: insert first batch
        session.sql(f"INSERT INTO {tbl} VALUES (1, 'Alice'), (2, 'Bob')")

        # Sleep to ensure distinct timestamps between snapshots
        time.sleep(1.5)
        # Record a timestamp between the two snapshots
        ts_between = datetime.now()

        # Small extra sleep to ensure timestamp ordering
        time.sleep(0.5)

        # Snapshot 2: insert second batch
        session.sql(f"INSERT INTO {tbl} VALUES (3, 'Carol'), (4, 'Dave')")

        # Query snapshots metadata to get snapshot IDs
        snap_df = session.sql(f"SELECT * FROM tt.tt_travel.snapshots")
        snap_rows = snap_df.collect()

        # Sort by committed_at to identify snapshot order
        snap_rows.sort(key=lambda r: r["committed_at"])
        assert len(snap_rows) >= 2, (
            f"Expected at least 2 snapshots, got {len(snap_rows)}"
        )

        snapshot_1_id = snap_rows[0]["snapshot_id"]
        snapshot_2_id = snap_rows[-1]["snapshot_id"]

        return {
            "table": tbl,
            "snapshot_1_id": snapshot_1_id,
            "snapshot_2_id": snapshot_2_id,
            "ts_between": ts_between,
        }

    def test_version_as_of_snapshot_1(self, session, time_travel_table):
        """VERSION AS OF <snapshot_1_id> returns only the first batch of data."""
        tbl = time_travel_table["table"]
        snap_id = time_travel_table["snapshot_1_id"]

        df = session.sql(
            f"SELECT * FROM {tbl} VERSION AS OF {snap_id} ORDER BY id"
        )
        rows = df.collect()
        ids = [r["id"] for r in rows]
        assert ids == [1, 2], f"Expected [1, 2] but got {ids}"

    def test_version_as_of_snapshot_2(self, session, time_travel_table):
        """VERSION AS OF <snapshot_2_id> returns all data."""
        tbl = time_travel_table["table"]
        snap_id = time_travel_table["snapshot_2_id"]

        df = session.sql(
            f"SELECT * FROM {tbl} VERSION AS OF {snap_id} ORDER BY id"
        )
        rows = df.collect()
        ids = [r["id"] for r in rows]
        assert ids == [1, 2, 3, 4], f"Expected [1, 2, 3, 4] but got {ids}"

    def test_timestamp_as_of_between_snapshots(self, session, time_travel_table):
        """TIMESTAMP AS OF with a time between snapshots returns snapshot 1 data."""
        tbl = time_travel_table["table"]
        ts = time_travel_table["ts_between"]
        ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")

        df = session.sql(
            f"SELECT * FROM {tbl} TIMESTAMP AS OF '{ts_str}' ORDER BY id"
        )
        rows = df.collect()
        ids = [r["id"] for r in rows]
        assert ids == [1, 2], (
            f"Expected [1, 2] for timestamp between snapshots, got {ids}"
        )

    def test_current_select_returns_all(self, session, time_travel_table):
        """Normal SELECT * returns all data from both inserts."""
        tbl = time_travel_table["table"]

        df = session.sql(f"SELECT * FROM {tbl} ORDER BY id")
        rows = df.collect()
        ids = [r["id"] for r in rows]
        assert ids == [1, 2, 3, 4], (
            f"Expected [1, 2, 3, 4] for current table, got {ids}"
        )
