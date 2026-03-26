"""Tests for Sprint 1 Task 1.1: Core missing functions."""

from __future__ import annotations

import pyarrow as pa
import pytest
from datafusion import SessionContext

from iceberg_spark.dataframe import DataFrame
from iceberg_spark.functions import (
    bitwise_not,
    broadcast,
    col,
    expr,
    input_file_name,
    isnan,
    lit,
    monotonically_increasing_id,
    nanvl,
    spark_partition_id,
    typedLit,
)


@pytest.fixture
def ctx():
    return SessionContext()


@pytest.fixture
def df(ctx):
    t = pa.table({
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", "Carol"],
        "score": [85.5, float("nan"), 78.0],
        "flag": [1, 0, 1],
    })
    ctx.register_record_batches("t", [t.to_batches()])
    return DataFrame(ctx.table("t"), session=None)


class TestExpr:
    def test_expr_column_ref(self, df):
        result = df.select(expr("id"), expr("name"))
        assert result.count() == 3
        assert "id" in result.columns

    def test_expr_in_filter(self, df):
        result = df.filter(expr("id") > lit(1))
        assert result.count() == 2


class TestMonotonicallyIncreasingId:
    def test_returns_column(self):
        c = monotonically_increasing_id()
        assert c is not None


class TestSparkPartitionId:
    def test_returns_zero(self, df):
        result = df.select(spark_partition_id().alias("pid"))
        rows = result.collect()
        assert all(r["pid"] == 0 for r in rows)


class TestBroadcast:
    def test_returns_same_df(self, df):
        result = broadcast(df)
        assert result is df


class TestTypedLit:
    def test_typed_lit_int(self, df):
        result = df.select(typedLit(42).alias("val"))
        rows = result.collect()
        assert rows[0]["val"] == 42

    def test_typed_lit_string(self, df):
        result = df.select(typedLit("hello").alias("val"))
        rows = result.collect()
        assert rows[0]["val"] == "hello"


class TestInputFileName:
    def test_returns_empty_string(self, df):
        result = df.select(input_file_name().alias("fname"))
        rows = result.collect()
        assert all(r["fname"] == "" for r in rows)


class TestNanvl:
    def test_replaces_nan(self, df):
        result = df.select(
            col("id"),
            nanvl(col("score"), lit(0.0)).alias("score_clean"),
        )
        rows = result.collect()
        # Row with id=2 had NaN score, should be replaced with 0.0
        for r in rows:
            if r["id"] == 2:
                assert r["score_clean"] == 0.0
            else:
                assert r["score_clean"] > 0

    def test_keeps_non_nan(self, df):
        result = df.select(nanvl(col("score"), lit(-1.0)).alias("s"))
        rows = result.collect()
        non_nan_count = sum(1 for r in rows if r["s"] != -1.0)
        assert non_nan_count == 2


class TestBitwiseNot:
    def test_bitwise_not(self, df):
        result = df.select(bitwise_not(col("flag")).alias("notflag"))
        rows = result.collect()
        # ~1 = -2, ~0 = -1 in two's complement
        assert rows[0]["notflag"] == -2
        assert rows[1]["notflag"] == -1


class TestImports:
    """Verify all new functions are importable from iceberg_spark."""

    def test_import_from_package(self):
        from iceberg_spark import (
            bitwise_not,
            broadcast,
            expr,
            input_file_name,
            monotonically_increasing_id,
            nanvl,
            spark_partition_id,
            typedLit,
        )
        assert all(callable(f) for f in [
            bitwise_not, broadcast, expr, input_file_name,
            monotonically_increasing_id, nanvl, spark_partition_id, typedLit,
        ])
