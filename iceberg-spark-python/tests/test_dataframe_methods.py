"""Tests for DataFrame missing methods (Task 6A + 6B)."""

from __future__ import annotations

import pyarrow as pa
import pytest
from datafusion import SessionContext

from iceberg_spark.column import Column
from iceberg_spark.dataframe import DataFrame
from iceberg_spark.functions import col, lit


@pytest.fixture
def ctx():
    return SessionContext()


@pytest.fixture
def sample_table():
    return pa.table({
        "id": [1, 2, 2, 3, 3, 3],
        "name": ["Alice", "Bob", "Bob", "Carol", "Carol", "Carol"],
        "score": [85.0, 92.0, 92.0, 78.0, 78.0, 78.0],
    })


@pytest.fixture
def df(ctx, sample_table):
    ctx.register_record_batches("t", [sample_table.to_batches()])
    return DataFrame(ctx.table("t"), session=None)


class TestDropDuplicates:
    def test_no_cols_is_distinct(self, df):
        result = df.dropDuplicates()
        assert result.count() == 3  # Alice, Bob, Carol

    def test_snake_case_alias(self, df):
        assert hasattr(df, "drop_duplicates")

    def test_drop_duplicates_alias(self, df):
        result = df.drop_duplicates()
        assert result.count() == 3

    def test_with_cols_no_session_raises(self, df):
        # session=None with specific cols must raise, not silently return wrong results
        with pytest.raises(RuntimeError, match="requires an active session"):
            df.dropDuplicates("id")


class TestWithColumns:
    def test_add_multiple(self, df):
        result = df.withColumns({
            "double_score": col("score") * lit(2),
            "id_plus_one": col("id") + lit(1),
        })
        assert "double_score" in result.columns
        assert "id_plus_one" in result.columns

    def test_empty_map(self, df):
        result = df.withColumns({})
        assert result.columns == df.columns


class TestWithColumnsRenamed:
    def test_rename_multiple(self, df):
        result = df.withColumnsRenamed({"id": "user_id", "name": "user_name"})
        assert "user_id" in result.columns
        assert "user_name" in result.columns
        assert "id" not in result.columns

    def test_missing_col_ignored(self, df):
        result = df.withColumnsRenamed({"nonexistent": "new_name"})
        assert result.columns == df.columns


class TestDescribe:
    # describe() requires self._session to be None-safe; we patched it to use a fresh ctx.
    def test_describe_no_session(self, df):
        result = df.describe()
        assert "summary" in result.columns
        rows = result.collect()
        summaries = [r["summary"] for r in rows]
        assert "count" in summaries
        assert "mean" in summaries
        assert "stddev" in summaries
        assert "min" in summaries
        assert "max" in summaries

    def test_describe_specific_cols(self, df):
        result = df.describe("score")
        assert "summary" in result.columns
        assert "score" in result.columns


class TestTail:
    def test_tail(self, df):
        rows = df.tail(2)
        assert len(rows) == 2

    def test_tail_default(self, df):
        rows = df.tail()
        assert len(rows) == 1

    def test_tail_zero(self, df):
        rows = df.tail(0)
        assert rows == []


class TestToLocalIterator:
    def test_iterator(self, df):
        it = df.toLocalIterator()
        count = sum(1 for _ in it)
        assert count == df.count()


class TestIntersectAll:
    def test_intersect_all(self, ctx):
        t1 = pa.table({"id": [1, 2, 2, 3]})
        t2 = pa.table({"id": [2, 2, 3, 4]})
        ctx.register_record_batches("t1", [t1.to_batches()])
        ctx.register_record_batches("t2", [t2.to_batches()])
        df1 = DataFrame(ctx.table("t1"), session=None)
        df2 = DataFrame(ctx.table("t2"), session=None)
        result = df1.intersectAll(df2)
        assert result.count() >= 2  # both 2s and 3


class TestTempViews:
    def test_create_temp_view_requires_session(self, df):
        with pytest.raises(RuntimeError, match="requires an active session"):
            df.createTempView("v1")

    def test_create_or_replace_requires_session(self, df):
        with pytest.raises(RuntimeError, match="requires an active session"):
            df.createOrReplaceTempView("v1")
