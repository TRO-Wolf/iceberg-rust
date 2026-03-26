"""Integration tests for REST catalog.

Requires Docker. Skipped automatically when Docker is not available.

Run with:
    uv run pytest tests/integration/test_rest_catalog.py -v --timeout=300
"""

from __future__ import annotations

import subprocess

import pytest

try:
    from tests.integration.conftest import _is_docker_available
except ImportError:
    def _is_docker_available():
        try:
            return subprocess.run(
                ["docker", "compose", "version"],
                capture_output=True, timeout=5,
            ).returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

pytestmark = [
    pytest.mark.skipif(not _is_docker_available(), reason="Docker not available"),
    pytest.mark.docker,
]


# ---------------------------------------------------------------------------
# Session creation
# ---------------------------------------------------------------------------


class TestRestCatalogSession:
    def test_session_creation(self, rest_session):
        assert rest_session is not None
        assert rest_session._catalog is not None


# ---------------------------------------------------------------------------
# Namespace operations
# ---------------------------------------------------------------------------


class TestRestCatalogNamespaces:
    def test_create_and_list_namespace(self, rest_session):
        rest_session.sql("CREATE DATABASE IF NOT EXISTS rest_ns_test")
        dbs = rest_session.sql("SHOW DATABASES").collect()
        db_names = [row[0] for row in dbs]
        assert "rest_ns_test" in db_names

    def test_drop_namespace(self, rest_session):
        rest_session.sql("CREATE DATABASE IF NOT EXISTS rest_ns_drop")
        rest_session.sql("DROP NAMESPACE rest_ns_drop")
        dbs = rest_session.sql("SHOW DATABASES").collect()
        db_names = [row[0] for row in dbs]
        assert "rest_ns_drop" not in db_names


# ---------------------------------------------------------------------------
# Table DDL
# ---------------------------------------------------------------------------


class TestRestCatalogDDL:
    @pytest.fixture(autouse=True)
    def setup_namespace(self, rest_session):
        rest_session.sql("CREATE DATABASE IF NOT EXISTS rest_ddl")

    def test_create_and_list_table(self, rest_session):
        rest_session.sql("DROP TABLE IF EXISTS rest_ddl.t1")
        rest_session.sql(
            "CREATE TABLE rest_ddl.t1 (id INT, name STRING) USING iceberg"
        )
        tables = rest_session.sql("SHOW TABLES IN rest_ddl").collect()
        table_names = [row[0] for row in tables]
        assert "t1" in table_names
        rest_session.sql("DROP TABLE IF EXISTS rest_ddl.t1")

    def test_describe_table(self, rest_session):
        rest_session.sql("DROP TABLE IF EXISTS rest_ddl.t_desc")
        rest_session.sql(
            "CREATE TABLE rest_ddl.t_desc (x INT, y DOUBLE) USING iceberg"
        )
        desc = rest_session.sql("DESCRIBE TABLE rest_ddl.t_desc").collect()
        col_names = [row[0] for row in desc]
        assert "x" in col_names
        assert "y" in col_names
        rest_session.sql("DROP TABLE IF EXISTS rest_ddl.t_desc")


# ---------------------------------------------------------------------------
# DML round-trip
# ---------------------------------------------------------------------------


class TestRestCatalogDML:
    @pytest.fixture(autouse=True)
    def setup_namespace(self, rest_session):
        rest_session.sql("CREATE DATABASE IF NOT EXISTS rest_dml")

    def test_insert_and_select(self, rest_session):
        rest_session.sql("DROP TABLE IF EXISTS rest_dml.t_rw")
        rest_session.sql(
            "CREATE TABLE rest_dml.t_rw (id INT, val STRING) USING iceberg"
        )
        rest_session.sql(
            "INSERT INTO rest_dml.t_rw VALUES (1, 'hello'), (2, 'world')"
        )
        result = rest_session.sql("SELECT * FROM rest_dml.t_rw").collect()
        assert len(result) == 2
        ids = sorted(row[0] for row in result)
        assert ids == [1, 2]
        rest_session.sql("DROP TABLE IF EXISTS rest_dml.t_rw")


# ---------------------------------------------------------------------------
# Full lifecycle
# ---------------------------------------------------------------------------


class TestRestCatalogLifecycle:
    def test_full_lifecycle(self, rest_session):
        ns = "rest_lifecycle"
        tbl = f"{ns}.lc_table"
        rest_session.sql(f"CREATE DATABASE IF NOT EXISTS {ns}")
        rest_session.sql(f"DROP TABLE IF EXISTS {tbl}")
        rest_session.sql(f"CREATE TABLE {tbl} (a INT, b STRING) USING iceberg")
        rest_session.sql(f"INSERT INTO {tbl} VALUES (10, 'x')")
        result = rest_session.sql(f"SELECT * FROM {tbl}").collect()
        assert len(result) == 1
        rest_session.sql(f"DROP TABLE IF EXISTS {tbl}")
        rest_session.sql(f"DROP NAMESPACE {ns}")
