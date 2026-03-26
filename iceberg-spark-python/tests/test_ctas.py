"""Tests for CREATE TABLE AS SELECT (CTAS).

Covers:
- SQL preprocessor: CTAS detection and parsing
- Core DataFusion logic for CTAS handler (no Iceberg catalog needed)
"""

from __future__ import annotations

import pyarrow as pa
import pytest
from datafusion import SessionContext

from iceberg_spark.sql_preprocessor import CommandType, preprocess


# ---------------------------------------------------------------------------
# SQL Preprocessor — CTAS detection
# ---------------------------------------------------------------------------


class TestCTASPreprocessor:
    def test_basic_ctas(self):
        result = preprocess("CREATE TABLE db.new AS SELECT * FROM db.old")
        assert result.command_type == CommandType.CREATE_TABLE_AS_SELECT
        assert result.table_name == "db.new"
        assert result.extra["select_query"] == "SELECT * FROM db.old"
        assert result.extra["if_not_exists"] is False

    def test_ctas_if_not_exists(self):
        result = preprocess(
            "CREATE TABLE IF NOT EXISTS db.new AS SELECT * FROM db.old"
        )
        assert result.command_type == CommandType.CREATE_TABLE_AS_SELECT
        assert result.table_name == "db.new"
        assert result.extra["select_query"] == "SELECT * FROM db.old"
        assert result.extra["if_not_exists"] is True

    def test_ctas_with_where(self):
        result = preprocess(
            "CREATE TABLE db.filtered AS SELECT id, name FROM db.src WHERE id > 5"
        )
        assert result.command_type == CommandType.CREATE_TABLE_AS_SELECT
        assert result.table_name == "db.filtered"
        assert "WHERE id > 5" in result.extra["select_query"]

    def test_ctas_case_insensitive(self):
        result = preprocess("create table db.t2 as select * from db.t1")
        assert result.command_type == CommandType.CREATE_TABLE_AS_SELECT
        assert result.table_name == "db.t2"

    def test_ctas_with_join(self):
        sql = (
            "CREATE TABLE db.joined AS "
            "SELECT a.id, b.name FROM db.a JOIN db.b ON a.id = b.id"
        )
        result = preprocess(sql)
        assert result.command_type == CommandType.CREATE_TABLE_AS_SELECT
        assert result.table_name == "db.joined"
        assert "JOIN" in result.extra["select_query"]

    def test_ctas_with_aggregation(self):
        result = preprocess(
            "CREATE TABLE db.agg AS SELECT dept, COUNT(*) AS cnt FROM db.emp GROUP BY dept"
        )
        assert result.command_type == CommandType.CREATE_TABLE_AS_SELECT
        assert "GROUP BY" in result.extra["select_query"]

    def test_ctas_simple_table_name(self):
        result = preprocess("CREATE TABLE newtbl AS SELECT 1 AS id")
        assert result.command_type == CommandType.CREATE_TABLE_AS_SELECT
        assert result.table_name == "newtbl"

    def test_ctas_semicolon_stripped(self):
        result = preprocess("CREATE TABLE db.t AS SELECT * FROM db.src;")
        assert result.command_type == CommandType.CREATE_TABLE_AS_SELECT
        assert result.extra["select_query"] == "SELECT * FROM db.src"

    def test_regular_create_table_not_affected(self):
        """Ensure regular CREATE TABLE with column defs is not caught by CTAS regex."""
        result = preprocess("CREATE TABLE db.t1 (id INT, name STRING) USING iceberg")
        assert result.command_type == CommandType.CREATE_TABLE
        assert result.table_name == "db.t1"

    def test_create_table_if_not_exists_not_affected(self):
        """Ensure regular CREATE TABLE IF NOT EXISTS with column defs still works."""
        result = preprocess("CREATE TABLE IF NOT EXISTS db.t1 (id INT)")
        assert result.command_type == CommandType.CREATE_TABLE
        assert result.table_name == "db.t1"

    def test_ctas_with_subquery(self):
        result = preprocess(
            "CREATE TABLE db.sub AS SELECT * FROM db.t WHERE id IN (SELECT id FROM db.t2)"
        )
        assert result.command_type == CommandType.CREATE_TABLE_AS_SELECT
        assert "IN (SELECT" in result.extra["select_query"]


# ---------------------------------------------------------------------------
# CTAS DataFusion logic — verify SELECT execution + schema inference
# ---------------------------------------------------------------------------


class TestCTASDataFusionLogic:
    """Test the DataFusion query execution part of CTAS (no catalog needed)."""

    @pytest.fixture
    def ctx(self):
        return SessionContext()

    @pytest.fixture
    def source_table(self):
        return pa.table({
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Carol", "Dave", "Eve"],
            "salary": [50000.0, 60000.0, 70000.0, 55000.0, 65000.0],
            "dept": ["eng", "sales", "eng", "sales", "eng"],
        })

    @pytest.fixture
    def registered_ctx(self, ctx, source_table):
        ctx.register_record_batches("src", [source_table.to_batches()])
        return ctx

    def test_select_all(self, registered_ctx, source_table):
        """CTAS SELECT * — result has same schema and data."""
        result = registered_ctx.sql("SELECT * FROM src").to_arrow_table()
        assert result.schema == source_table.schema
        assert len(result) == 5

    def test_select_with_filter(self, registered_ctx):
        """CTAS SELECT ... WHERE — only matching rows."""
        result = registered_ctx.sql(
            "SELECT * FROM src WHERE salary > 55000"
        ).to_arrow_table()
        assert len(result) == 3
        assert all(r.as_py() > 55000 for r in result.column("salary"))

    def test_select_subset_columns(self, registered_ctx):
        """CTAS SELECT specific columns — schema reflects chosen columns."""
        result = registered_ctx.sql("SELECT id, name FROM src").to_arrow_table()
        assert result.schema.names == ["id", "name"]
        assert len(result) == 5

    def test_select_with_aggregation(self, registered_ctx):
        """CTAS SELECT with GROUP BY — schema inferred from aggregation result."""
        result = registered_ctx.sql(
            "SELECT dept, COUNT(*) AS cnt, AVG(salary) AS avg_sal FROM src GROUP BY dept"
        ).to_arrow_table()
        assert "dept" in result.schema.names
        assert "cnt" in result.schema.names
        assert "avg_sal" in result.schema.names
        assert len(result) == 2  # eng, sales

    def test_select_with_expression(self, registered_ctx):
        """CTAS SELECT with computed column — new column appears in schema."""
        result = registered_ctx.sql(
            "SELECT id, name, salary * 1.1 AS new_salary FROM src"
        ).to_arrow_table()
        assert "new_salary" in result.schema.names
        assert len(result) == 5

    def test_select_empty_result(self, registered_ctx):
        """CTAS SELECT that returns 0 rows — schema is still present."""
        result = registered_ctx.sql(
            "SELECT * FROM src WHERE id > 999"
        ).to_arrow_table()
        assert len(result) == 0
        assert result.schema.names == ["id", "name", "salary", "dept"]

    def test_select_with_join(self, ctx):
        """CTAS with JOIN — schema combines both sides."""
        t1 = pa.table({"id": [1, 2, 3], "name": ["A", "B", "C"]})
        t2 = pa.table({"id": [2, 3, 4], "score": [90, 85, 75]})
        ctx.register_record_batches("t1", [t1.to_batches()])
        ctx.register_record_batches("t2", [t2.to_batches()])
        result = ctx.sql(
            "SELECT t1.id, t1.name, t2.score FROM t1 "
            "JOIN t2 ON t1.id = t2.id"
        ).to_arrow_table()
        assert result.schema.names == ["id", "name", "score"]
        assert len(result) == 2  # ids 2, 3

    def test_schema_types_preserved(self, registered_ctx):
        """Arrow types from SELECT are preserved correctly."""
        result = registered_ctx.sql("SELECT id, salary FROM src").to_arrow_table()
        assert pa.types.is_int64(result.schema.field("id").type)
        assert pa.types.is_float64(result.schema.field("salary").type)
