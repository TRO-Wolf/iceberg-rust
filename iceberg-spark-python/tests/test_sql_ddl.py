"""Tests for SQL DDL gap handlers (Task 9A)."""

from __future__ import annotations

import pyarrow as pa
import pytest
from datafusion import SessionContext

from iceberg_spark.catalog_ops import (
    handle_create_view,
    handle_drop_view,
    handle_explain,
    handle_show_columns,
)


class MockSession:
    """Minimal session mock for testing handlers without a catalog."""

    def __init__(self):
        self._ctx = SessionContext()
        self._catalog = None
        self._registered_tables: dict[str, str] = {}

    def _ensure_tables_registered(self, sql):
        return sql

    def _ensure_table_registered(self, short_name, full_name=None):
        pass


@pytest.fixture
def session():
    s = MockSession()
    t = pa.table({"id": [1, 2, 3], "name": ["Alice", "Bob", "Carol"]})
    s._ctx.register_record_batches("t1", [t.to_batches()])
    return s


class TestCreateView:
    def test_create_view(self, session):
        result = handle_create_view(session, "v1", "SELECT * FROM t1")
        rows = result.collect()
        assert rows[0]["status"] == "View v1 created"
        # View should be queryable
        df = session._ctx.sql("SELECT * FROM v1").to_arrow_table()
        assert df.num_rows == 3

    def test_create_or_replace_view(self, session):
        handle_create_view(session, "v2", "SELECT id FROM t1")
        handle_create_view(session, "v2", "SELECT name FROM t1", or_replace=True)
        df = session._ctx.sql("SELECT * FROM v2").to_arrow_table()
        assert "name" in df.column_names

    def test_create_view_with_filter(self, session):
        handle_create_view(session, "v3", "SELECT * FROM t1 WHERE id > 1")
        df = session._ctx.sql("SELECT * FROM v3").to_arrow_table()
        assert df.num_rows == 2

    def test_create_view_duplicate_raises(self, session):
        handle_create_view(session, "v_dup", "SELECT * FROM t1")
        with pytest.raises(RuntimeError, match="already exists"):
            handle_create_view(session, "v_dup", "SELECT * FROM t1")

    def test_create_view_transforms_spark_syntax(self, session):
        """CREATE VIEW should apply Spark→DataFusion SQL transforms."""
        t = pa.table({"a": pa.array([1, None, 3], type=pa.int64())})
        session._ctx.register_record_batches("t_nvl", [t.to_batches()])
        # NVL is Spark syntax; should be transformed to COALESCE
        handle_create_view(session, "v_nvl", "SELECT NVL(a, 0) FROM t_nvl")
        df = session._ctx.sql("SELECT * FROM v_nvl").to_arrow_table()
        assert df.num_rows == 3


class TestDropView:
    def test_drop_view(self, session):
        handle_create_view(session, "v_drop", "SELECT * FROM t1")
        result = handle_drop_view(session, "v_drop")
        rows = result.collect()
        assert rows[0]["status"] == "View v_drop dropped"
        with pytest.raises(KeyError):
            session._ctx.table("v_drop")

    def test_drop_nonexistent_view(self, session):
        # Should not raise
        result = handle_drop_view(session, "nonexistent")
        rows = result.collect()
        assert "dropped" in rows[0]["status"]


class TestShowColumns:
    def test_show_columns_registered_table(self, session):
        result = handle_show_columns(session, "t1")
        rows = result.collect()
        col_names = [r["col_name"] for r in rows]
        assert "id" in col_names
        assert "name" in col_names
        assert len(col_names) == 2

    def test_show_columns_not_found(self, session):
        with pytest.raises(RuntimeError, match="not found"):
            handle_show_columns(session, "nonexistent_table")


class TestExplain:
    def test_explain_select(self, session):
        result = handle_explain(session, "SELECT * FROM t1")
        rows = result.collect()
        assert len(rows) > 0

    def test_explain_with_filter(self, session):
        result = handle_explain(session, "SELECT * FROM t1 WHERE id > 1")
        rows = result.collect()
        assert len(rows) > 0

    def test_explain_transforms_spark_syntax(self, session):
        """EXPLAIN should apply Spark→DataFusion SQL transforms to the inner query."""
        t = pa.table({"a": pa.array([1, None, 3], type=pa.int64()), "b": [10, 20, 30]})
        session._ctx.register_record_batches("t2", [t.to_batches()])
        # NVL is Spark syntax; should be transformed to COALESCE
        result = handle_explain(session, "SELECT NVL(a, 0) FROM t2")
        rows = result.collect()
        assert len(rows) > 0
