"""Tests for Phase 2 + Phase 3 features.

Covers: SQL preprocessor additions (INSERT INTO, TRUNCATE), DataFrame enhancements
(toDF, sample, na, stat, range), window function API surface, and DataFrameWriter modes.

These tests only require datafusion + pyarrow (no Iceberg catalog needed).
"""

from __future__ import annotations

import pyarrow as pa
import pytest
from datafusion import SessionContext

from iceberg_spark.column import Column
from iceberg_spark.dataframe import DataFrame, DataFrameNaFunctions, DataFrameStatFunctions
from iceberg_spark.functions import (
    col,
    count,
    cume_dist,
    dense_rank,
    lag,
    lead,
    lit,
    ntile,
    percent_rank,
    rank,
    row_number,
    sum,
)
from iceberg_spark.sql_preprocessor import CommandType, preprocess
from iceberg_spark.window import Window, WindowSpec


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ctx():
    return SessionContext()


@pytest.fixture
def sample_table():
    return pa.table({
        "id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Carol", "Dave", "Eve"],
        "age": [30, 25, 35, 28, 32],
        "score": [85.5, 92.0, 78.3, 95.1, 88.7],
        "dept": ["eng", "sales", "eng", "sales", "eng"],
    })


@pytest.fixture
def df(ctx, sample_table):
    ctx.register_record_batches("t", [sample_table.to_batches()])
    return DataFrame(ctx.table("t"), session=None)


@pytest.fixture
def nullable_table():
    return pa.table({
        "a": pa.array([1, None, 3, None, 5], type=pa.int64()),
        "b": pa.array(["x", "y", None, None, "z"], type=pa.string()),
    })


@pytest.fixture
def df_with_nulls(ctx, nullable_table):
    ctx.register_record_batches("nulls", [nullable_table.to_batches()])
    return DataFrame(ctx.table("nulls"), session=None)


# ---------------------------------------------------------------------------
# SQL Preprocessor — new command types
# ---------------------------------------------------------------------------


class TestSqlPreprocessorPhase2:
    def test_insert_into_values(self):
        result = preprocess("INSERT INTO db.t1 VALUES (1, 'hello')")
        assert result.command_type == CommandType.INSERT_INTO
        assert result.table_name == "db.t1"
        assert result.extra.get("overwrite") is False

    def test_insert_into_select(self):
        result = preprocess("INSERT INTO t1 SELECT * FROM t2")
        assert result.command_type == CommandType.INSERT_INTO
        assert result.table_name == "t1"

    def test_insert_overwrite(self):
        result = preprocess("INSERT OVERWRITE t1 SELECT id FROM t2")
        assert result.command_type == CommandType.INSERT_INTO
        assert result.extra.get("overwrite") is True

    def test_insert_into_table_keyword(self):
        result = preprocess("INSERT INTO TABLE db.t1 VALUES (1)")
        assert result.command_type == CommandType.INSERT_INTO
        assert result.table_name == "db.t1"

    def test_truncate_table(self):
        result = preprocess("TRUNCATE TABLE db.t1")
        assert result.command_type == CommandType.TRUNCATE
        assert result.table_name == "db.t1"

    def test_truncate_case_insensitive(self):
        result = preprocess("truncate table my_table")
        assert result.command_type == CommandType.TRUNCATE
        assert result.table_name == "my_table"

    def test_alter_table_add_column(self):
        result = preprocess("ALTER TABLE t1 ADD COLUMN score DOUBLE")
        assert result.command_type == CommandType.ALTER_TABLE
        assert result.table_name == "t1"

    def test_alter_table_drop_column(self):
        result = preprocess("ALTER TABLE t1 DROP COLUMN old_col")
        assert result.command_type == CommandType.ALTER_TABLE

    def test_alter_table_rename_column(self):
        result = preprocess("ALTER TABLE t1 RENAME COLUMN old TO new_name")
        assert result.command_type == CommandType.ALTER_TABLE


# ---------------------------------------------------------------------------
# DataFrame.toDF
# ---------------------------------------------------------------------------


class TestToDF:
    def test_todf_renames_all_columns(self, df):
        result = df.select("id", "age").toDF("user_id", "user_age")
        assert result.columns == ["user_id", "user_age"]

    def test_todf_same_names(self, df):
        result = df.select("id", "name").toDF("id", "name")
        assert result.columns == ["id", "name"]

    def test_todf_wrong_count_raises(self, df):
        with pytest.raises(ValueError, match="Number of column names"):
            df.toDF("a", "b")  # df has 5 cols, only 2 provided


# ---------------------------------------------------------------------------
# DataFrame.sample
# ---------------------------------------------------------------------------


class TestSample:
    def test_sample_returns_dataframe(self, df):
        result = df.sample(fraction=1.0)
        assert isinstance(result, DataFrame)

    def test_sample_no_fraction_raises(self, df):
        with pytest.raises(ValueError, match="fraction is required"):
            df.sample()

    def test_sample_invalid_fraction_raises(self, df):
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            df.sample(fraction=1.5)

    def test_sample_zero_fraction(self, df):
        result = df.sample(fraction=0.0)
        assert result.count() == 0


# ---------------------------------------------------------------------------
# DataFrame.hint / transform
# ---------------------------------------------------------------------------


class TestHintAndTransform:
    def test_hint_is_noop(self, df):
        result = df.hint("broadcast")
        assert result.count() == df.count()

    def test_transform_applies_function(self, df):
        result = df.transform(lambda d: d.filter(col("age") > lit(30)))
        assert result.count() < df.count()
        assert all(r["age"] > 30 for r in result.collect())


# ---------------------------------------------------------------------------
# DataFrame.na
# ---------------------------------------------------------------------------


class TestNaFunctions:
    def test_na_drop_any(self, df_with_nulls):
        result = df_with_nulls.na.drop(how="any")
        rows = result.collect()
        assert all(r["a"] is not None and r["b"] is not None for r in rows)

    def test_na_drop_all(self, df_with_nulls):
        # Only rows where ALL cols are null should be dropped
        result = df_with_nulls.na.drop(how="all")
        # Row (None, None) should be gone, others kept
        assert result.count() < df_with_nulls.count()

    def test_na_drop_subset(self, df_with_nulls):
        result = df_with_nulls.na.drop(subset=["a"])
        rows = result.collect()
        assert all(r["a"] is not None for r in rows)

    def test_na_fill_scalar(self, df_with_nulls):
        result = df_with_nulls.na.fill(0, subset=["a"])
        rows = result.collect()
        assert all(r["a"] is not None for r in rows)

    def test_na_fill_dict(self, df_with_nulls):
        result = df_with_nulls.na.fill({"a": -1, "b": "unknown"})
        rows = result.collect()
        assert all(r["a"] is not None for r in rows)
        assert all(r["b"] is not None for r in rows)

    def test_na_replace(self, df):
        result = df.select("dept").na.replace("eng", "engineering")
        rows = result.collect()
        depts = {r["dept"] for r in rows}
        assert "eng" not in depts
        assert "engineering" in depts

    def test_na_is_accessor(self, df_with_nulls):
        assert isinstance(df_with_nulls.na, DataFrameNaFunctions)


# ---------------------------------------------------------------------------
# DataFrame.stat
# ---------------------------------------------------------------------------


class TestStatFunctions:
    def test_stat_is_accessor(self, df):
        assert isinstance(df.stat, DataFrameStatFunctions)

    def test_corr(self, df):
        # id and age are loosely correlated; result should be a float
        r = df.stat.corr("id", "age")
        assert isinstance(r, float)

    def test_cov(self, df):
        r = df.stat.cov("id", "score")
        assert isinstance(r, float)

    def test_crosstab(self, df):
        result = df.stat.crosstab("dept", "name")
        assert isinstance(result, DataFrame)
        assert "dept" in result.columns


# ---------------------------------------------------------------------------
# Window function API surface (object construction, not full execution)
# ---------------------------------------------------------------------------


class TestWindowFunctionAPI:
    def test_window_spec_creation(self):
        w = Window.partitionBy("dept").orderBy("age")
        assert isinstance(w, WindowSpec)
        assert len(w._partition_cols) == 1
        assert len(w._order_cols) == 1

    def test_rows_between(self):
        w = Window.orderBy("id").rowsBetween(Window.unboundedPreceding, Window.currentRow)
        assert isinstance(w, WindowSpec)
        assert w._frame is not None

    def test_range_between(self):
        w = Window.orderBy("id").rangeBetween(-1, 1)
        assert isinstance(w, WindowSpec)

    def test_row_number_returns_column(self):
        c = row_number()
        assert isinstance(c, Column)

    def test_rank_returns_column(self):
        assert isinstance(rank(), Column)

    def test_dense_rank_returns_column(self):
        assert isinstance(dense_rank(), Column)

    def test_percent_rank_returns_column(self):
        assert isinstance(percent_rank(), Column)

    def test_cume_dist_returns_column(self):
        assert isinstance(cume_dist(), Column)

    def test_ntile_returns_column(self):
        assert isinstance(ntile(4), Column)

    def test_lag_returns_column(self):
        assert isinstance(lag("age"), Column)
        assert isinstance(lag("age", 2), Column)
        assert isinstance(lag("age", 1, 0), Column)

    def test_lead_returns_column(self):
        assert isinstance(lead("age"), Column)
        assert isinstance(lead("age", 2, 0), Column)

    def test_over_returns_column(self, df):
        w = Window.partitionBy("dept").orderBy("age")
        # Just verify .over() doesn't raise and returns a Column
        result_col = row_number().over(w)
        assert isinstance(result_col, Column)

    def test_window_with_datafusion(self, ctx):
        """Smoke test: window function via SQL works end-to-end."""
        t = pa.table({
            "dept": ["eng", "eng", "sales", "sales"],
            "salary": [100, 120, 80, 90],
        })
        ctx.register_record_batches("emp", [t.to_batches()])
        df = DataFrame(ctx.table("emp"), session=None)
        # Use SQL for window since DataFusion SQL window is well-supported
        ctx.register_record_batches("emp_sql", [t.to_batches()])
        result = ctx.sql(
            "SELECT dept, salary, "
            "ROW_NUMBER() OVER (PARTITION BY dept ORDER BY salary) AS rn "
            "FROM emp_sql"
        )
        rows = result.collect()
        assert len(rows) > 0
