"""Tests for Sprint 2: format, collection, and JSON functions."""

from __future__ import annotations

import pyarrow as pa
import pytest
from datafusion import SessionContext

from iceberg_spark.dataframe import DataFrame
from iceberg_spark.functions import (
    array,
    array_except,
    array_intersect,
    arrays_overlap,
    col,
    format_number,
    format_string,
    from_json,
    lit,
    schema_of_json,
    to_json,
)


@pytest.fixture
def ctx():
    return SessionContext()


@pytest.fixture
def df(ctx):
    t = pa.table({
        "id": [1, 2, 3],
        "score": [85.5678, 92.1, 78.999],
        "name": ["Alice", "Bob", "Carol"],
    })
    ctx.register_record_batches("t", [t.to_batches()])
    return DataFrame(ctx.table("t"), session=None)


class TestFormatNumber:
    def test_two_decimals(self, df):
        result = df.select(format_number(col("score"), 2).alias("s"))
        rows = result.collect()
        assert rows[0]["s"] == "85.57"

    def test_zero_decimals(self, df):
        result = df.select(format_number(col("score"), 0).alias("s"))
        rows = result.collect()
        assert rows[0]["s"] in ("86", "86.0")  # rounded; float cast may keep .0


class TestFormatString:
    def test_simple(self, df):
        result = df.select(
            format_string("Hello %s, your score is %s", col("name"), col("score")).alias("msg")
        )
        rows = result.collect()
        assert "Hello Alice" in rows[0]["msg"]


class TestArrayExcept:
    def test_basic(self, ctx):
        t = pa.table({
            "a": [[1, 2, 3]],
            "b": [[2, 3, 4]],
        })
        ctx.register_record_batches("t2", [t.to_batches()])
        df = DataFrame(ctx.table("t2"), session=None)
        result = df.select(array_except(col("a"), col("b")).alias("diff"))
        rows = result.collect()
        assert 1 in rows[0]["diff"]
        assert 2 not in rows[0]["diff"]


class TestArrayIntersect:
    def test_basic(self, ctx):
        t = pa.table({
            "a": [[1, 2, 3]],
            "b": [[2, 3, 4]],
        })
        ctx.register_record_batches("t3", [t.to_batches()])
        df = DataFrame(ctx.table("t3"), session=None)
        result = df.select(array_intersect(col("a"), col("b")).alias("common"))
        rows = result.collect()
        assert set(rows[0]["common"]) == {2, 3}


class TestArraysOverlap:
    def test_overlap_true(self, ctx):
        t = pa.table({
            "a": [[1, 2]],
            "b": [[2, 3]],
        })
        ctx.register_record_batches("t4", [t.to_batches()])
        df = DataFrame(ctx.table("t4"), session=None)
        result = df.select(arrays_overlap(col("a"), col("b")).alias("overlap"))
        rows = result.collect()
        assert rows[0]["overlap"] is True

    def test_overlap_false(self, ctx):
        t = pa.table({
            "a": [[1, 2]],
            "b": [[3, 4]],
        })
        ctx.register_record_batches("t5", [t.to_batches()])
        df = DataFrame(ctx.table("t5"), session=None)
        result = df.select(arrays_overlap(col("a"), col("b")).alias("overlap"))
        rows = result.collect()
        assert rows[0]["overlap"] is False


class TestJSONFunctions:
    def test_from_json_basic(self, ctx):
        t = pa.table({"data": ['{"name": "Alice", "age": 30}', '{"name": "Bob", "age": 25}']})
        ctx.register_record_batches("json_t", [t.to_batches()])
        df = DataFrame(ctx.table("json_t"), session=None)
        result = df.select(from_json(col("data"), "struct<name:string,age:bigint>").alias("parsed"))
        rows = result.collect()
        assert rows[0]["parsed"]["name"] == "Alice"
        assert rows[0]["parsed"]["age"] == 30

    def test_schema_of_json_basic(self):
        result = schema_of_json('{"name": "Alice", "age": 30}')
        # schema_of_json returns a Column wrapping a literal DDL string
        # Collect it via a dummy DataFrame
        from datafusion import SessionContext as DFCtx
        _ctx = DFCtx()
        _ctx.register_record_batches("_dummy", [pa.table({"x": [1]}).to_batches()])
        rows = DataFrame(_ctx.table("_dummy"), session=None).select(result.alias("s")).collect()
        ddl = rows[0]["s"]
        assert "struct<" in ddl
        assert "name:string" in ddl

    def test_to_json_casts_to_string(self, df):
        result = df.select(to_json(col("id")).alias("j"))
        rows = result.collect()
        assert rows[0]["j"] == "1"


class TestImports:
    def test_all_importable(self):
        from iceberg_spark import (
            array_except,
            array_intersect,
            arrays_overlap,
            create_map,
            format_number,
            format_string,
            from_json,
            map_keys,
            map_values,
            schema_of_json,
            to_json,
        )
