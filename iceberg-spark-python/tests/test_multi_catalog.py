"""Tests for multi-catalog support.

Covers:
- Builder with multiple .catalog() calls
- session.catalog.currentCatalog() / setCurrentCatalog() / listCatalogs()
- USE CATALOG / SHOW CATALOGS SQL commands
- 3-part table names (catalog.namespace.table) in SQL
- Cross-catalog operations
- Backward compatibility: single-catalog session unchanged
"""

from __future__ import annotations

import os
import tempfile

import pyarrow as pa
import pytest

try:
    import pyiceberg.catalog.sql  # noqa: F401
    from iceberg_spark import IcebergSession
    HAS_SQL_CATALOG = True
except ImportError:
    HAS_SQL_CATALOG = False

pytestmark = pytest.mark.skipif(
    not HAS_SQL_CATALOG,
    reason="pyiceberg[sql-sqlite] / sqlalchemy not installed — skipping multi-catalog tests",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def warehouses():
    """Two separate warehouse directories for two catalogs."""
    with tempfile.TemporaryDirectory(prefix="iceberg_mc_wh1_") as wh1:
        with tempfile.TemporaryDirectory(prefix="iceberg_mc_wh2_") as wh2:
            yield wh1, wh2


@pytest.fixture(scope="module")
def multi_session(warehouses):
    """Session with two SQL catalogs: 'cat_a' and 'cat_b'."""
    wh1, wh2 = warehouses
    db1 = os.path.join(wh1, "catalog.db")
    db2 = os.path.join(wh2, "catalog.db")
    sess = (
        IcebergSession.builder()
        .catalog(
            "sql", name="cat_a",
            uri=f"sqlite:///{db1}",
            warehouse=f"file://{wh1}",
        )
        .catalog(
            "sql", name="cat_b",
            uri=f"sqlite:///{db2}",
            warehouse=f"file://{wh2}",
        )
        .defaultCatalog("cat_a")
        .build()
    )
    # Create namespaces in both catalogs
    sess.sql("CREATE DATABASE IF NOT EXISTS db")
    sess.sql("USE CATALOG cat_b")
    sess.sql("CREATE DATABASE IF NOT EXISTS db")
    sess.sql("USE CATALOG cat_a")
    yield sess
    sess.stop()


# ---------------------------------------------------------------------------
# Builder + basic API
# ---------------------------------------------------------------------------


class TestMultiCatalogBuilder:
    def test_two_catalogs_configured(self, multi_session):
        assert len(multi_session._catalogs) == 2
        assert "cat_a" in multi_session._catalogs
        assert "cat_b" in multi_session._catalogs

    def test_default_catalog_is_cat_a(self, multi_session):
        assert multi_session._current_catalog_name == "cat_a"

    def test_single_catalog_backward_compat(self):
        """Single-catalog builder works exactly as before."""
        with tempfile.TemporaryDirectory() as wh:
            db = os.path.join(wh, "catalog.db")
            sess = (
                IcebergSession.builder()
                .catalog("sql", name="only", uri=f"sqlite:///{db}",
                         warehouse=f"file://{wh}")
                .build()
            )
            assert len(sess._catalogs) == 1
            assert sess._catalog_name == "only"
            sess.stop()


# ---------------------------------------------------------------------------
# session.catalog API
# ---------------------------------------------------------------------------


class TestCatalogAPI:
    def test_current_catalog(self, multi_session):
        assert multi_session.catalog.currentCatalog() == "cat_a"

    def test_list_catalogs(self, multi_session):
        cats = multi_session.catalog.listCatalogs()
        assert sorted(cats) == ["cat_a", "cat_b"]

    def test_list_catalogs_with_pattern(self, multi_session):
        cats = multi_session.catalog.listCatalogs("cat_a*")
        assert cats == ["cat_a"]

    def test_set_current_catalog(self, multi_session):
        multi_session.catalog.setCurrentCatalog("cat_b")
        assert multi_session.catalog.currentCatalog() == "cat_b"
        # Switch back
        multi_session.catalog.setCurrentCatalog("cat_a")

    def test_set_current_catalog_resets_database(self, multi_session):
        multi_session.catalog.setCurrentDatabase("custom_ns")
        multi_session.catalog.setCurrentCatalog("cat_b")
        assert multi_session.catalog.currentDatabase() == "default"
        # Switch back
        multi_session.catalog.setCurrentCatalog("cat_a")

    def test_set_nonexistent_catalog_raises(self, multi_session):
        with pytest.raises(RuntimeError, match="not found"):
            multi_session.catalog.setCurrentCatalog("nonexistent")


# ---------------------------------------------------------------------------
# SQL: USE CATALOG / SHOW CATALOGS
# ---------------------------------------------------------------------------


class TestSQLCatalogCommands:
    def test_use_catalog(self, multi_session):
        multi_session.sql("USE CATALOG cat_b")
        assert multi_session._current_catalog_name == "cat_b"
        multi_session.sql("USE CATALOG cat_a")
        assert multi_session._current_catalog_name == "cat_a"

    def test_use_catalog_nonexistent_raises(self, multi_session):
        with pytest.raises(Exception, match="not found"):
            multi_session.sql("USE CATALOG does_not_exist")

    def test_show_catalogs(self, multi_session):
        result = multi_session.sql("SHOW CATALOGS").collect()
        catalogs = sorted(row[0] for row in result)
        assert catalogs == ["cat_a", "cat_b"]


# ---------------------------------------------------------------------------
# Cross-catalog operations via 3-part names
# ---------------------------------------------------------------------------


class TestCrossCatalogOperations:
    def test_create_tables_in_different_catalogs(self, multi_session):
        # Create in cat_a (current)
        multi_session.sql("DROP TABLE IF EXISTS db.tbl_a")
        multi_session.sql(
            "CREATE TABLE db.tbl_a (id INT, val STRING) USING iceberg"
        )
        multi_session.sql("INSERT INTO db.tbl_a VALUES (1, 'from_a')")

        # Create in cat_b via USE CATALOG
        multi_session.sql("USE CATALOG cat_b")
        multi_session.sql("DROP TABLE IF EXISTS db.tbl_b")
        multi_session.sql(
            "CREATE TABLE db.tbl_b (id INT, val STRING) USING iceberg"
        )
        multi_session.sql("INSERT INTO db.tbl_b VALUES (2, 'from_b')")

        # Switch back and verify cat_a table
        multi_session.sql("USE CATALOG cat_a")
        result_a = multi_session.sql("SELECT * FROM db.tbl_a").collect()
        assert len(result_a) == 1
        assert result_a[0][1] == "from_a"

        # Switch to cat_b and verify
        multi_session.sql("USE CATALOG cat_b")
        result_b = multi_session.sql("SELECT * FROM db.tbl_b").collect()
        assert len(result_b) == 1
        assert result_b[0][1] == "from_b"

        # Clean up
        multi_session.sql("DROP TABLE IF EXISTS db.tbl_b")
        multi_session.sql("USE CATALOG cat_a")
        multi_session.sql("DROP TABLE IF EXISTS db.tbl_a")

    def test_catalog_isolation(self, multi_session):
        """Tables in one catalog are not visible in the other."""
        multi_session.sql("DROP TABLE IF EXISTS db.isolated")
        multi_session.sql(
            "CREATE TABLE db.isolated (x INT) USING iceberg"
        )

        # Table should NOT exist in cat_b
        multi_session.sql("USE CATALOG cat_b")
        exists = multi_session.catalog.tableExists("db.isolated")
        assert not exists

        # Clean up
        multi_session.sql("USE CATALOG cat_a")
        multi_session.sql("DROP TABLE IF EXISTS db.isolated")

    def test_new_session_shares_catalogs(self, multi_session):
        """newSession() shares the catalogs dict."""
        new = multi_session.newSession()
        assert len(new._catalogs) == 2
        assert "cat_a" in new._catalogs
        assert "cat_b" in new._catalogs
