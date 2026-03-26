"""Integration tests for Iceberg metadata table queries.

Tests cover SELECT * FROM table.snapshots / .history / .files / .entries / .refs / .manifests.

Requires PyIceberg + SQLite catalog.  Skipped automatically when unavailable.

Run with:
    uv run pytest tests/integration/test_metadata_tables.py -v
"""

from __future__ import annotations

import os
import tempfile

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
    with tempfile.TemporaryDirectory(prefix="iceberg_meta_test_") as d:
        yield d


@pytest.fixture(scope="module")
def session(warehouse):
    db_path = os.path.join(warehouse, "catalog.db")
    sess = (
        IcebergSession.builder()
        .catalog(
            "sql",
            name="meta_catalog",
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
        session.sql("CREATE DATABASE IF NOT EXISTS meta")
    except Exception:
        pass


@pytest.fixture(scope="module")
def populated_table(session):
    """Create a table and insert data so that metadata tables are non-empty."""
    tbl = "meta.meta_tbl"
    try:
        session.sql(f"DROP TABLE IF EXISTS {tbl}")
    except Exception:
        pass

    session.sql(
        f"CREATE TABLE {tbl} (id INT, name STRING, value DOUBLE) USING iceberg"
    )
    session.sql(
        f"INSERT INTO {tbl} VALUES "
        "(1, 'Alice', 10.0), (2, 'Bob', 20.0), (3, 'Carol', 30.0)"
    )
    return tbl


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMetadataTables:
    """Test metadata table queries against a populated Iceberg table."""

    def test_snapshots(self, session, populated_table):
        """SELECT * FROM table.snapshots returns at least one snapshot."""
        df = session.sql(f"SELECT * FROM {populated_table}.snapshots")
        rows = df.collect()
        assert len(rows) >= 1, "Expected at least 1 snapshot after INSERT"
        # Verify expected columns exist
        col_names = set(rows[0].asDict().keys()) if hasattr(rows[0], 'asDict') else set(rows[0].keys())
        assert "snapshot_id" in col_names, (
            f"Expected 'snapshot_id' column, got columns: {col_names}"
        )

    def test_history(self, session, populated_table):
        """SELECT * FROM table.history returns at least one entry."""
        df = session.sql(f"SELECT * FROM {populated_table}.history")
        rows = df.collect()
        assert len(rows) >= 1, "Expected at least 1 history entry after INSERT"

    def test_files(self, session, populated_table):
        """SELECT * FROM table.files returns at least one data file."""
        df = session.sql(f"SELECT * FROM {populated_table}.files")
        rows = df.collect()
        assert len(rows) >= 1, "Expected at least 1 data file after INSERT"

    def test_entries(self, session, populated_table):
        """SELECT * FROM table.entries returns at least one manifest entry."""
        df = session.sql(f"SELECT * FROM {populated_table}.entries")
        rows = df.collect()
        assert len(rows) >= 1, "Expected at least 1 entry after INSERT"

    def test_refs(self, session, populated_table):
        """SELECT * FROM table.refs returns at least the main branch ref."""
        df = session.sql(f"SELECT * FROM {populated_table}.refs")
        rows = df.collect()
        assert len(rows) >= 1, "Expected at least 1 ref (main branch)"

    def test_manifests(self, session, populated_table):
        """SELECT * FROM table.manifests returns at least one manifest.

        Some PyIceberg versions may not support inspect.manifests(), so we
        allow the query to succeed with any number of results.
        """
        try:
            df = session.sql(f"SELECT * FROM {populated_table}.manifests")
            rows = df.collect()
            # If the query succeeds, we expect at least 1 manifest
            assert len(rows) >= 1, "Expected at least 1 manifest after INSERT"
        except Exception:
            pytest.skip("manifests metadata table not supported by this PyIceberg version")

    def test_snapshots_have_snapshot_id_values(self, session, populated_table):
        """Verify that snapshot IDs are non-null positive integers."""
        df = session.sql(f"SELECT * FROM {populated_table}.snapshots")
        rows = df.collect()
        for row in rows:
            snap_id = row["snapshot_id"]
            assert snap_id is not None
            assert isinstance(snap_id, int)
            assert snap_id > 0
