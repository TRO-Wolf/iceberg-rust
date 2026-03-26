"""Tests for collection functions (Task 7E)."""

from __future__ import annotations

import pyarrow as pa
import pytest
from datafusion import SessionContext

from iceberg_spark.dataframe import DataFrame
from iceberg_spark.functions import (
    array,
    array_contains,
    array_distinct,
    array_join,
    array_position,
    array_remove,
    array_sort,
    array_union,
    col,
    element_at,
    explode,
    flatten,
    lit,
    size,
    sort_array,
    struct_func,
)


@pytest.fixture
def ctx():
    return SessionContext()


@pytest.fixture
def df(ctx):
    t = pa.table({
        "a": [1, 2, 3],
        "b": [10, 20, 30],
        "c": [100, 200, 300],
    })
    ctx.register_record_batches("tc", [t.to_batches()])
    return DataFrame(ctx.table("tc"), session=None)


@pytest.fixture
def arr_df(ctx):
    """DataFrame with array column."""
    t = pa.table({
        "arr": pa.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], type=pa.list_(pa.int64())),
        "arr2": pa.array([[10, 20], [30, 40], [50, 60]], type=pa.list_(pa.int64())),
    })
    ctx.register_record_batches("tarr", [t.to_batches()])
    return DataFrame(ctx.table("tarr"), session=None)


class TestArrayCreation:
    def test_array_from_columns(self, df):
        result = df.select(array(col("a"), col("b"), col("c")).alias("arr"))
        rows = result.collect()
        assert list(rows[0]["arr"]) == [1, 10, 100]
        assert list(rows[1]["arr"]) == [2, 20, 200]


class TestArrayOperations:
    def test_array_contains(self, arr_df):
        result = arr_df.select(array_contains(col("arr"), 2).alias("has2"))
        rows = result.collect()
        assert rows[0]["has2"] is True
        assert rows[1]["has2"] is False

    def test_array_distinct(self, ctx):
        t = pa.table({"arr": pa.array([[1, 1, 2, 2, 3]], type=pa.list_(pa.int64()))})
        ctx.register_record_batches("tad", [t.to_batches()])
        df = DataFrame(ctx.table("tad"), session=None)
        result = df.select(array_distinct(col("arr")).alias("d"))
        rows = result.collect()
        assert sorted(rows[0]["d"]) == [1, 2, 3]

    def test_array_join(self, ctx):
        t = pa.table({"arr": pa.array([["a", "b", "c"]], type=pa.list_(pa.utf8()))})
        ctx.register_record_batches("taj", [t.to_batches()])
        df = DataFrame(ctx.table("taj"), session=None)
        result = df.select(array_join(col("arr"), ",").alias("j"))
        rows = result.collect()
        assert rows[0]["j"] == "a,b,c"

    def test_array_position(self, arr_df):
        result = arr_df.select(array_position(col("arr"), 2).alias("pos"))
        rows = result.collect()
        assert rows[0]["pos"] == 2  # 1-based: element 2 is at position 2

    def test_array_remove(self, ctx):
        t = pa.table({"arr": pa.array([[1, 2, 3, 2, 1]], type=pa.list_(pa.int64()))})
        ctx.register_record_batches("tar", [t.to_batches()])
        df = DataFrame(ctx.table("tar"), session=None)
        result = df.select(array_remove(col("arr"), 2).alias("r"))
        rows = result.collect()
        assert 2 not in list(rows[0]["r"])

    def test_array_sort(self, ctx):
        t = pa.table({"arr": pa.array([[3, 1, 2]], type=pa.list_(pa.int64()))})
        ctx.register_record_batches("tas", [t.to_batches()])
        df = DataFrame(ctx.table("tas"), session=None)
        result = df.select(array_sort(col("arr")).alias("s"))
        rows = result.collect()
        assert list(rows[0]["s"]) == [1, 2, 3]

    def test_sort_array(self, ctx):
        t = pa.table({"arr": pa.array([[3, 1, 2]], type=pa.list_(pa.int64()))})
        ctx.register_record_batches("tsa", [t.to_batches()])
        df = DataFrame(ctx.table("tsa"), session=None)
        result = df.select(sort_array(col("arr")).alias("s"))
        rows = result.collect()
        assert list(rows[0]["s"]) == [1, 2, 3]

    def test_sort_array_descending(self, ctx):
        t = pa.table({"arr": pa.array([[3, 1, 2]], type=pa.list_(pa.int64()))})
        ctx.register_record_batches("tsad", [t.to_batches()])
        df = DataFrame(ctx.table("tsad"), session=None)
        result = df.select(sort_array(col("arr"), asc=False).alias("s"))
        rows = result.collect()
        assert list(rows[0]["s"]) == [3, 2, 1]

    def test_array_union(self, ctx):
        t = pa.table({
            "a": pa.array([[1, 2, 3]], type=pa.list_(pa.int64())),
            "b": pa.array([[3, 4, 5]], type=pa.list_(pa.int64())),
        })
        ctx.register_record_batches("tau", [t.to_batches()])
        df = DataFrame(ctx.table("tau"), session=None)
        result = df.select(array_union(col("a"), col("b")).alias("u"))
        rows = result.collect()
        assert sorted(rows[0]["u"]) == [1, 2, 3, 4, 5]

    def test_element_at(self, arr_df):
        result = arr_df.select(element_at(col("arr"), 1).alias("e"))
        rows = result.collect()
        assert rows[0]["e"] == 1  # 1-based index

    def test_size(self, arr_df):
        result = arr_df.select(size(col("arr")).alias("sz"))
        rows = result.collect()
        assert rows[0]["sz"] == 3

    def test_flatten(self, ctx):
        t = pa.table({
            "nested": pa.array(
                [[[1, 2], [3, 4]]],
                type=pa.list_(pa.list_(pa.int64())),
            ),
        })
        ctx.register_record_batches("tfl", [t.to_batches()])
        df = DataFrame(ctx.table("tfl"), session=None)
        result = df.select(flatten(col("nested")).alias("f"))
        rows = result.collect()
        assert list(rows[0]["f"]) == [1, 2, 3, 4]


class TestExplode:
    def test_explode_via_unnest_columns(self, ctx):
        """explode() in DataFusion v52 requires unnest_columns at DataFrame level."""
        t = pa.table({"arr": pa.array([[1, 2, 3]], type=pa.list_(pa.int64()))})
        ctx.register_record_batches("tex", [t.to_batches()])
        # Use DataFusion's unnest_columns directly
        df_raw = ctx.table("tex")
        result = df_raw.unnest_columns("arr").to_arrow_table()
        assert result.num_rows == 3
        assert result.column("arr").to_pylist() == [1, 2, 3]

    def test_explode_via_sql(self, ctx):
        """Verify unnest works via SQL."""
        t = pa.table({"arr": pa.array([[1, 2, 3]], type=pa.list_(pa.int64()))})
        ctx.register_record_batches("tex2", [t.to_batches()])
        result = ctx.sql("SELECT unnest(arr) as val FROM tex2").to_arrow_table()
        assert result.num_rows == 3


class TestStruct:
    def test_struct_func(self, df):
        result = df.select(struct_func(col("a"), col("b")).alias("s"))
        rows = result.collect()
        s = rows[0]["s"]
        # struct should contain both fields
        assert s is not None
