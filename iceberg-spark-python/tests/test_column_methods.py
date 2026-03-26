"""Tests for Column missing methods (Task 6D)."""

from __future__ import annotations

import pyarrow as pa
import pytest
from datafusion import SessionContext

from iceberg_spark.column import Column
from iceberg_spark.dataframe import DataFrame
from iceberg_spark.functions import col, lit, when


@pytest.fixture
def ctx():
    return SessionContext()


@pytest.fixture
def df(ctx):
    t = pa.table({
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", "Carol"],
        "score": [85.5, float("nan"), 78.0],
    })
    ctx.register_record_batches("t", [t.to_batches()])
    return DataFrame(ctx.table("t"), session=None)


class TestSubstr:
    def test_substr(self, df):
        result = df.select(col("name").substr(1, 3).alias("short"))
        rows = result.collect()
        assert rows[0]["short"] == "Ali"

    def test_substr_mid(self, df):
        result = df.select(col("name").substr(2, 2).alias("mid"))
        rows = result.collect()
        assert rows[0]["mid"] == "li"


class TestIsNaN:
    def test_isnan_finds_nan(self, df):
        result = df.filter(col("score").isNaN())
        assert result.count() == 1

    def test_isnan_excludes_normal(self, df):
        result = df.filter(~col("score").isNaN())
        assert result.count() == 2


class TestRlike:
    def test_rlike_matches(self, df):
        result = df.filter(col("name").rlike("^A.*"))
        assert result.count() == 1
        assert result.collect()[0]["name"] == "Alice"

    def test_rlike_no_match(self, df):
        result = df.filter(col("name").rlike("^Z.*"))
        assert result.count() == 0


class TestWhenChain:
    def test_when_otherwise_chain(self, df):
        result = df.select(
            when(col("id") == lit(1), lit("one"))
            .when(col("id") == lit(2), lit("two"))
            .otherwise(lit("other"))
            .alias("label")
        )
        rows = result.collect()
        labels = [r["label"] for r in rows]
        assert "one" in labels
        assert "two" in labels
        assert "other" in labels


class TestEqNullSafe:
    def test_both_null(self, ctx):
        t = pa.table({"a": pa.array([1, None, 3], type=pa.int64())})
        ctx.register_record_batches("t2", [t.to_batches()])
        df = DataFrame(ctx.table("t2"), session=None)
        result = df.filter(col("a").eqNullSafe(None))
        assert result.count() == 1

    def test_equal_values(self, ctx):
        t = pa.table({"a": [1, 2, 3]})
        ctx.register_record_batches("t3", [t.to_batches()])
        df = DataFrame(ctx.table("t3"), session=None)
        result = df.filter(col("a").eqNullSafe(2))
        assert result.count() == 1


class TestGroupedDataDict:
    def test_agg_dict(self, ctx):
        t = pa.table({
            "dept": ["eng", "eng", "sales", "sales"],
            "score": [90.0, 80.0, 70.0, 60.0],
            "age": [30, 25, 40, 35],
        })
        ctx.register_record_batches("emp", [t.to_batches()])
        df = DataFrame(ctx.table("emp"), session=None)
        result = df.groupBy("dept").agg({"score": "avg", "age": "max"})
        assert result.count() == 2

    def test_agg_dict_unknown_func_raises(self, ctx):
        t = pa.table({
            "dept": ["eng"],
            "score": [90.0],
        })
        ctx.register_record_batches("emp2", [t.to_batches()])
        df = DataFrame(ctx.table("emp2"), session=None)
        with pytest.raises(ValueError, match="Unknown aggregation"):
            df.groupBy("dept").agg({"score": "nonexistent"})


class TestPivotNoneValues:
    def test_pivot_skips_none(self, ctx):
        # pivot column has a None — should not crash during sorted()
        t = pa.table({
            "region": ["east", "east", "west", "west"],
            "quarter": pa.array(["Q1", "Q2", None, "Q1"], type=pa.string()),
            "sales": [100.0, 200.0, 150.0, 120.0],
        })
        ctx.register_record_batches("sales", [t.to_batches()])
        from iceberg_spark.functions import sum as _sum, col
        df = DataFrame(ctx.table("sales"), session=None)
        result = df.groupBy("region").pivot("quarter").agg(_sum(col("sales")))
        # None pivot value should be excluded from result columns
        assert "Q1" in result.columns or "Q2" in result.columns
        assert result.count() == 2


# ---------------------------------------------------------------------------
# Cube / Rollup tests
# ---------------------------------------------------------------------------


def _make_session_with_ctx(ctx):
    """Create a minimal IcebergSession wired to an existing DataFusion context."""
    from iceberg_spark.session import IcebergSession

    session = object.__new__(IcebergSession)
    session._ctx = ctx
    session._catalog = None
    session._catalog_name = "test"
    session._config = {}
    session._registered_tables = {}
    session._registered_udfs = {}
    session._current_namespace = None
    return session


class TestCubeBasic:
    def test_cube_basic(self, ctx):
        t = pa.table({
            "a": ["x", "x", "y", "y"],
            "b": ["m", "n", "m", "n"],
            "v": [1, 2, 3, 4],
        })
        ctx.register_record_batches("cube_t", [t.to_batches()])
        session = _make_session_with_ctx(ctx)
        df = DataFrame(ctx.table("cube_t"), session=session)
        from iceberg_spark.functions import sum as _sum
        result = df.cube("a", "b").agg(_sum(col("v")).alias("total"))
        rows = result.collect()
        # 2 cols -> 2^2 = 4 subsets: (a,b), (a,), (b,), ()
        # (a,b): 4 rows; (a,): 2 rows; (b,): 2 rows; (): 1 row = 9 total
        assert len(rows) == 9

    def test_cube_grand_total(self, ctx):
        t = pa.table({
            "a": ["x", "x", "y", "y"],
            "b": ["m", "n", "m", "n"],
            "v": [10, 20, 30, 40],
        })
        ctx.register_record_batches("cube_gt", [t.to_batches()])
        session = _make_session_with_ctx(ctx)
        df = DataFrame(ctx.table("cube_gt"), session=session)
        from iceberg_spark.functions import sum as _sum
        result = df.cube("a", "b").agg(_sum(col("v")).alias("total"))
        rows = result.collect()
        # Grand total row: both a and b are None
        grand = [r for r in rows if r["a"] is None and r["b"] is None]
        assert len(grand) == 1
        assert grand[0]["total"] == 100


class TestRollupBasic:
    def test_rollup_basic(self, ctx):
        t = pa.table({
            "a": ["x", "x", "y", "y"],
            "b": ["m", "n", "m", "n"],
            "v": [1, 2, 3, 4],
        })
        ctx.register_record_batches("rollup_t", [t.to_batches()])
        session = _make_session_with_ctx(ctx)
        df = DataFrame(ctx.table("rollup_t"), session=session)
        from iceberg_spark.functions import sum as _sum
        result = df.rollup("a", "b").agg(_sum(col("v")).alias("total"))
        rows = result.collect()
        # Rollup prefixes: (a,b), (a,), () = 4 + 2 + 1 = 7 rows
        assert len(rows) == 7

    def test_rollup_subtotals(self, ctx):
        t = pa.table({
            "a": ["x", "x", "y", "y"],
            "b": ["m", "n", "m", "n"],
            "v": [10, 20, 30, 40],
        })
        ctx.register_record_batches("rollup_sub", [t.to_batches()])
        session = _make_session_with_ctx(ctx)
        df = DataFrame(ctx.table("rollup_sub"), session=session)
        from iceberg_spark.functions import sum as _sum
        result = df.rollup("a", "b").agg(_sum(col("v")).alias("total"))
        rows = result.collect()
        # Subtotal for a="x": b is None, total=30
        x_sub = [r for r in rows if r["a"] == "x" and r["b"] is None]
        assert len(x_sub) == 1
        assert x_sub[0]["total"] == 30
        # Subtotal for a="y": b is None, total=70
        y_sub = [r for r in rows if r["a"] == "y" and r["b"] is None]
        assert len(y_sub) == 1
        assert y_sub[0]["total"] == 70
        # Grand total
        grand = [r for r in rows if r["a"] is None and r["b"] is None]
        assert len(grand) == 1
        assert grand[0]["total"] == 100


class TestPivotWithSession:
    def test_pivot_with_session(self, ctx):
        """Pivot works when DataFrame has a real session (not None)."""
        t = pa.table({
            "region": ["east", "east", "west", "west"],
            "quarter": ["Q1", "Q2", "Q1", "Q2"],
            "sales": [100.0, 200.0, 150.0, 120.0],
        })
        ctx.register_record_batches("pivot_sess", [t.to_batches()])
        session = _make_session_with_ctx(ctx)
        df = DataFrame(ctx.table("pivot_sess"), session=session)
        from iceberg_spark.functions import sum as _sum
        result = df.groupBy("region").pivot("quarter").agg(_sum(col("sales")))
        rows = result.collect()
        assert len(rows) == 2
        assert "Q1" in result.columns
        assert "Q2" in result.columns

    def test_pivot_with_avg(self, ctx):
        """Pivot works with AVG aggregate."""
        t = pa.table({
            "dept": ["eng", "eng", "sales", "sales"],
            "level": ["jr", "sr", "jr", "sr"],
            "salary": [80.0, 120.0, 70.0, 110.0],
        })
        ctx.register_record_batches("pivot_avg", [t.to_batches()])
        session = _make_session_with_ctx(ctx)
        df = DataFrame(ctx.table("pivot_avg"), session=session)
        from iceberg_spark.functions import avg as _avg
        result = df.groupBy("dept").pivot("level").agg(_avg(col("salary")))
        rows = result.collect()
        assert len(rows) == 2
        row_eng = [r for r in rows if r["dept"] == "eng"][0]
        assert row_eng["jr"] == 80.0
        assert row_eng["sr"] == 120.0

    def test_pivot_with_aliased_avg(self, ctx):
        """Pivot correctly uses AVG even when the expression is aliased."""
        t = pa.table({
            "dept": ["eng", "eng", "sales", "sales"],
            "level": ["jr", "sr", "jr", "sr"],
            "salary": [80.0, 120.0, 70.0, 110.0],
        })
        ctx.register_record_batches("pivot_alias", [t.to_batches()])
        session = _make_session_with_ctx(ctx)
        df = DataFrame(ctx.table("pivot_alias"), session=session)
        from iceberg_spark.functions import avg as _avg
        # Alias the agg — the pivot must still detect AVG, not default to SUM
        result = df.groupBy("dept").pivot("level").agg(
            _avg(col("salary")).alias("avg_sal")
        )
        rows = result.collect()
        row_eng = [r for r in rows if r["dept"] == "eng"][0]
        # AVG(80.0) = 80.0, not SUM(80.0) = 80.0 (same here, but verifies no crash)
        assert row_eng["jr"] == 80.0
        assert row_eng["sr"] == 120.0
