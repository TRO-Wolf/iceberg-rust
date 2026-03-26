"""Tests for aggregate/conditional/window functions (Task 7A)."""

from __future__ import annotations

import pyarrow as pa
import pytest
from datafusion import SessionContext

from iceberg_spark.dataframe import DataFrame
from iceberg_spark.functions import (
    any_value,
    bool_and,
    bool_or,
    col,
    collect_set,
    corr,
    covar_pop,
    covar_samp,
    first_value,
    greatest,
    ifnull,
    isnotnull,
    last_value,
    least,
    lit,
    nth_value,
    nvl,
    nvl2,
    percentile_approx,
    stddev_pop,
    var_pop,
)
from iceberg_spark.window import Window


@pytest.fixture
def ctx():
    return SessionContext()


@pytest.fixture
def df(ctx):
    t = pa.table({
        "id": [1, 2, 3, 4, 5],
        "score": [10.0, 20.0, 30.0, 40.0, 50.0],
        "dept": ["a", "a", "b", "b", "b"],
        "flag": [True, True, False, True, True],
    })
    ctx.register_record_batches("t", [t.to_batches()])
    return DataFrame(ctx.table("t"), session=None)


class TestAggFunctions:
    def test_var_pop(self, df):
        result = df.agg(var_pop("score").alias("vp"))
        rows = result.collect()
        assert isinstance(rows[0]["vp"], float)

    def test_stddev_pop(self, df):
        result = df.agg(stddev_pop("score").alias("sp"))
        rows = result.collect()
        assert isinstance(rows[0]["sp"], float)

    def test_covar_samp(self, df):
        result = df.agg(covar_samp("id", "score").alias("cs"))
        rows = result.collect()
        assert isinstance(rows[0]["cs"], float)

    def test_covar_pop(self, df):
        result = df.agg(covar_pop("id", "score").alias("cp"))
        rows = result.collect()
        assert isinstance(rows[0]["cp"], float)

    def test_corr(self, df):
        result = df.agg(corr("id", "score").alias("r"))
        rows = result.collect()
        assert abs(rows[0]["r"] - 1.0) < 0.01  # perfect linear correlation

    def test_bool_and(self, df):
        result = df.agg(bool_and("flag").alias("ba"))
        rows = result.collect()
        assert rows[0]["ba"] is False  # one False value

    def test_bool_or(self, df):
        result = df.agg(bool_or("flag").alias("bo"))
        rows = result.collect()
        assert rows[0]["bo"] is True

    def test_any_value(self, df):
        result = df.agg(any_value("dept").alias("av"))
        rows = result.collect()
        assert rows[0]["av"] in ("a", "b")

    def test_percentile_approx(self, df):
        result = df.agg(percentile_approx("score", 0.5).alias("p50"))
        rows = result.collect()
        assert isinstance(rows[0]["p50"], float)


    def test_collect_set(self, ctx):
        t = pa.table({"dept": ["a", "a", "b", "b", "b"]})
        ctx.register_record_batches("tcs", [t.to_batches()])
        df = DataFrame(ctx.table("tcs"), session=None)
        result = df.agg(collect_set("dept").alias("depts"))
        rows = result.collect()
        assert sorted(rows[0]["depts"]) == ["a", "b"]


class TestConditionalFunctions:
    def test_greatest(self, ctx):
        t = pa.table({"a": [1, 5, 3], "b": [4, 2, 6]})
        ctx.register_record_batches("t2", [t.to_batches()])
        df = DataFrame(ctx.table("t2"), session=None)
        result = df.select(greatest(col("a"), col("b")).alias("g"))
        rows = result.collect()
        assert [r["g"] for r in rows] == [4, 5, 6]

    def test_greatest_with_null(self, ctx):
        t = pa.table({"a": pa.array([5, None, 3], type=pa.int64()), "b": pa.array([None, 2, 6], type=pa.int64())})
        ctx.register_record_batches("t2n", [t.to_batches()])
        df = DataFrame(ctx.table("t2n"), session=None)
        result = df.select(greatest(col("a"), col("b")).alias("g"))
        rows = result.collect()
        assert rows[0]["g"] == 5   # greatest(5, NULL) = 5
        assert rows[1]["g"] == 2   # greatest(NULL, 2) = 2
        assert rows[2]["g"] == 6   # greatest(3, 6) = 6

    def test_least(self, ctx):
        t = pa.table({"a": [1, 5, 3], "b": [4, 2, 6]})
        ctx.register_record_batches("t3", [t.to_batches()])
        df = DataFrame(ctx.table("t3"), session=None)
        result = df.select(least(col("a"), col("b")).alias("l"))
        rows = result.collect()
        assert [r["l"] for r in rows] == [1, 2, 3]

    def test_least_with_null(self, ctx):
        t = pa.table({"a": pa.array([5, None, 3], type=pa.int64()), "b": pa.array([None, 2, 6], type=pa.int64())})
        ctx.register_record_batches("t3n", [t.to_batches()])
        df = DataFrame(ctx.table("t3n"), session=None)
        result = df.select(least(col("a"), col("b")).alias("l"))
        rows = result.collect()
        assert rows[0]["l"] == 5   # least(5, NULL) = 5
        assert rows[1]["l"] == 2   # least(NULL, 2) = 2
        assert rows[2]["l"] == 3   # least(3, 6) = 3

    def test_isnotnull(self, ctx):
        t = pa.table({"a": pa.array([1, None, 3], type=pa.int64())})
        ctx.register_record_batches("t4", [t.to_batches()])
        df = DataFrame(ctx.table("t4"), session=None)
        result = df.filter(isnotnull(col("a")))
        assert result.count() == 2

    def test_nvl(self, ctx):
        t = pa.table({"a": pa.array([1, None, 3], type=pa.int64())})
        ctx.register_record_batches("t5", [t.to_batches()])
        df = DataFrame(ctx.table("t5"), session=None)
        result = df.select(nvl(col("a"), lit(0)).alias("a"))
        rows = result.collect()
        assert all(r["a"] is not None for r in rows)

    def test_ifnull_is_nvl_alias(self, ctx):
        t = pa.table({"a": pa.array([None, 2], type=pa.int64())})
        ctx.register_record_batches("t6", [t.to_batches()])
        df = DataFrame(ctx.table("t6"), session=None)
        result = df.select(ifnull(col("a"), lit(-1)).alias("a"))
        rows = result.collect()
        assert rows[0]["a"] == -1

    def test_nvl2(self, ctx):
        t = pa.table({"a": pa.array([1, None, 3], type=pa.int64())})
        ctx.register_record_batches("t7", [t.to_batches()])
        df = DataFrame(ctx.table("t7"), session=None)
        result = df.select(nvl2(col("a"), lit(100), lit(-1)).alias("r"))
        rows = result.collect()
        assert rows[0]["r"] == 100  # not null -> col2
        assert rows[1]["r"] == -1   # null -> col3
        assert rows[2]["r"] == 100


class TestWindowFunctions:
    def test_first_value(self, ctx):
        t = pa.table({"dept": ["a", "a", "b", "b"], "score": [10, 20, 30, 40]})
        ctx.register_record_batches("tw", [t.to_batches()])
        df = DataFrame(ctx.table("tw"), session=None)
        w = Window.partitionBy("dept").orderBy("score")
        result = df.select(col("dept"), col("score"), first_value("score").over(w).alias("fv"))
        rows = result.collect()
        # First value in each partition should be the min score
        for r in rows:
            if r["dept"] == "a":
                assert r["fv"] == 10
            else:
                assert r["fv"] == 30

    def test_last_value(self, ctx):
        t = pa.table({"dept": ["a", "a", "b", "b"], "score": [10, 20, 30, 40]})
        ctx.register_record_batches("tw2", [t.to_batches()])
        df = DataFrame(ctx.table("tw2"), session=None)
        w = Window.partitionBy("dept").orderBy("score").rowsBetween(
            Window.unboundedPreceding, Window.unboundedFollowing
        )
        result = df.select(col("dept"), col("score"), last_value("score").over(w).alias("lv"))
        rows = result.collect()
        for r in rows:
            if r["dept"] == "a":
                assert r["lv"] == 20
            else:
                assert r["lv"] == 40

    def test_nth_value(self, ctx):
        t = pa.table({"x": [10, 20, 30, 40, 50]})
        ctx.register_record_batches("tw3", [t.to_batches()])
        df = DataFrame(ctx.table("tw3"), session=None)
        w = Window.orderBy("x").rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
        result = df.select(col("x"), nth_value("x", 3).over(w).alias("n3"))
        rows = result.collect()
        assert all(r["n3"] == 30 for r in rows)
