"""Integration tests for Glue catalog (via moto mock).

Requires Docker and boto3. Skipped automatically when either is unavailable.

Run with:
    uv run pytest tests/integration/test_glue_catalog.py -v --timeout=300
"""

from __future__ import annotations

import subprocess

import pytest

try:
    import boto3  # noqa: F401
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False

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
    pytest.mark.skipif(not HAS_BOTO3, reason="boto3 not installed"),
    pytest.mark.docker,
]


# ---------------------------------------------------------------------------
# Session creation
# ---------------------------------------------------------------------------


class TestGlueCatalogSession:
    def test_session_creation(self, glue_session):
        assert glue_session is not None
        assert glue_session._catalog is not None


# ---------------------------------------------------------------------------
# Namespace operations
# ---------------------------------------------------------------------------


class TestGlueCatalogNamespaces:
    def test_create_and_list_namespace(self, glue_session):
        glue_session.sql("CREATE DATABASE IF NOT EXISTS glue_ns_test")
        dbs = glue_session.sql("SHOW DATABASES").collect()
        db_names = [row[0] for row in dbs]
        assert "glue_ns_test" in db_names

    def test_drop_namespace(self, glue_session):
        glue_session.sql("CREATE DATABASE IF NOT EXISTS glue_ns_drop")
        glue_session.sql("DROP NAMESPACE glue_ns_drop")
        dbs = glue_session.sql("SHOW DATABASES").collect()
        db_names = [row[0] for row in dbs]
        assert "glue_ns_drop" not in db_names


# ---------------------------------------------------------------------------
# Table DDL
# ---------------------------------------------------------------------------


class TestGlueCatalogDDL:
    @pytest.fixture(autouse=True)
    def setup_namespace(self, glue_session):
        glue_session.sql("CREATE DATABASE IF NOT EXISTS glue_ddl")

    def test_create_and_list_table(self, glue_session):
        glue_session.sql("DROP TABLE IF EXISTS glue_ddl.t1")
        glue_session.sql(
            "CREATE TABLE glue_ddl.t1 (id INT, name STRING) USING iceberg"
        )
        tables = glue_session.sql("SHOW TABLES IN glue_ddl").collect()
        table_names = [row[0] for row in tables]
        assert "t1" in table_names
        glue_session.sql("DROP TABLE IF EXISTS glue_ddl.t1")

    def test_describe_table(self, glue_session):
        glue_session.sql("DROP TABLE IF EXISTS glue_ddl.t_desc")
        glue_session.sql(
            "CREATE TABLE glue_ddl.t_desc (x INT, y DOUBLE) USING iceberg"
        )
        desc = glue_session.sql("DESCRIBE TABLE glue_ddl.t_desc").collect()
        col_names = [row[0] for row in desc]
        assert "x" in col_names
        assert "y" in col_names
        glue_session.sql("DROP TABLE IF EXISTS glue_ddl.t_desc")


# ---------------------------------------------------------------------------
# DML round-trip
# ---------------------------------------------------------------------------


class TestGlueCatalogDML:
    @pytest.fixture(autouse=True)
    def setup_namespace(self, glue_session):
        glue_session.sql("CREATE DATABASE IF NOT EXISTS glue_dml")

    def test_insert_and_select(self, glue_session):
        glue_session.sql("DROP TABLE IF EXISTS glue_dml.t_rw")
        glue_session.sql(
            "CREATE TABLE glue_dml.t_rw (id INT, val STRING) USING iceberg"
        )
        glue_session.sql(
            "INSERT INTO glue_dml.t_rw VALUES (1, 'hello'), (2, 'world')"
        )
        result = glue_session.sql("SELECT * FROM glue_dml.t_rw").collect()
        assert len(result) == 2
        ids = sorted(row[0] for row in result)
        assert ids == [1, 2]
        glue_session.sql("DROP TABLE IF EXISTS glue_dml.t_rw")


# ---------------------------------------------------------------------------
# Full lifecycle
# ---------------------------------------------------------------------------


class TestGlueCatalogLifecycle:
    def test_full_lifecycle(self, glue_session):
        ns = "glue_lifecycle"
        tbl = f"{ns}.lc_table"
        glue_session.sql(f"CREATE DATABASE IF NOT EXISTS {ns}")
        glue_session.sql(f"DROP TABLE IF EXISTS {tbl}")
        glue_session.sql(f"CREATE TABLE {tbl} (a INT, b STRING) USING iceberg")
        glue_session.sql(f"INSERT INTO {tbl} VALUES (10, 'x')")
        result = glue_session.sql(f"SELECT * FROM {tbl}").collect()
        assert len(result) == 1
        glue_session.sql(f"DROP TABLE IF EXISTS {tbl}")
        glue_session.sql(f"DROP NAMESPACE {ns}")
