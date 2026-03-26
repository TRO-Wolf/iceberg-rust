"""Tests for from_json(), to_json(), and schema_of_json() implementations."""

from __future__ import annotations

import json

import pyarrow as pa
import pytest
from datafusion import SessionContext

from iceberg_spark.dataframe import DataFrame
from iceberg_spark.functions import col, from_json, schema_of_json, to_json
from iceberg_spark.types import (
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)


@pytest.fixture
def ctx():
    return SessionContext()


def _make_df(ctx, col_name: str, values: list, arrow_type=pa.utf8()):
    """Create a single-column DataFrame from a list of values."""
    t = pa.table({col_name: pa.array(values, type=arrow_type)})
    ctx.register_record_batches("json_src", [t.to_batches()])
    return DataFrame(ctx.table("json_src"), session=None)


# ---------------------------------------------------------------------------
# schema_of_json
# ---------------------------------------------------------------------------


class TestSchemaOfJson:
    def test_simple_object(self):
        result = schema_of_json('{"name": "Alice", "age": 30}')
        # Collect via a helper DataFrame
        ctx = SessionContext()
        ctx.register_record_batches("d", [pa.table({"x": [1]}).to_batches()])
        rows = DataFrame(ctx.table("d"), session=None).select(result.alias("s")).collect()
        ddl = rows[0]["s"]
        assert ddl.startswith("struct<")
        assert "name:string" in ddl
        assert "age:bigint" in ddl

    def test_nested_types(self):
        result = schema_of_json('{"score": 3.14, "active": true}')
        ctx = SessionContext()
        ctx.register_record_batches("d", [pa.table({"x": [1]}).to_batches()])
        rows = DataFrame(ctx.table("d"), session=None).select(result.alias("s")).collect()
        ddl = rows[0]["s"]
        assert "score:double" in ddl
        assert "active:boolean" in ddl

    def test_returns_column(self):
        result = schema_of_json('{"a": 1}')
        from iceberg_spark.column import Column

        assert isinstance(result, Column)


# ---------------------------------------------------------------------------
# from_json — DDL string schema
# ---------------------------------------------------------------------------


class TestFromJsonDDL:
    def test_basic_struct(self, ctx):
        df = _make_df(ctx, "data", ['{"name":"Alice","age":30}', '{"name":"Bob","age":25}'])
        result = df.select(from_json(col("data"), "struct<name:string,age:bigint>").alias("parsed"))
        rows = result.collect()
        assert rows[0]["parsed"]["name"] == "Alice"
        assert rows[0]["parsed"]["age"] == 30
        assert rows[1]["parsed"]["name"] == "Bob"
        assert rows[1]["parsed"]["age"] == 25

    def test_bare_ddl_no_struct_wrapper(self, ctx):
        df = _make_df(ctx, "data", ['{"x": 42}'])
        result = df.select(from_json(col("data"), "x:bigint").alias("parsed"))
        rows = result.collect()
        assert rows[0]["parsed"]["x"] == 42

    def test_space_separated_types(self, ctx):
        df = _make_df(ctx, "data", ['{"val": "hello"}'])
        result = df.select(from_json(col("data"), "val string").alias("parsed"))
        rows = result.collect()
        assert rows[0]["parsed"]["val"] == "hello"


# ---------------------------------------------------------------------------
# from_json — StructType schema
# ---------------------------------------------------------------------------


class TestFromJsonStructType:
    def test_with_struct_type(self, ctx):
        schema = StructType([
            StructField("name", StringType()),
            StructField("age", LongType()),
        ])
        df = _make_df(ctx, "data", ['{"name":"Alice","age":30}'])
        result = df.select(from_json(col("data"), schema).alias("parsed"))
        rows = result.collect()
        assert rows[0]["parsed"]["name"] == "Alice"
        assert rows[0]["parsed"]["age"] == 30

    def test_with_arrow_struct(self, ctx):
        schema = pa.struct([pa.field("name", pa.utf8()), pa.field("age", pa.int64())])
        df = _make_df(ctx, "data", ['{"name":"Carol","age":40}'])
        result = df.select(from_json(col("data"), schema).alias("parsed"))
        rows = result.collect()
        assert rows[0]["parsed"]["name"] == "Carol"
        assert rows[0]["parsed"]["age"] == 40


# ---------------------------------------------------------------------------
# from_json — null and error handling
# ---------------------------------------------------------------------------


class TestFromJsonEdgeCases:
    def test_null_input(self, ctx):
        df = _make_df(
            ctx,
            "data",
            ['{"name":"Alice","age":30}', None],
        )
        result = df.select(from_json(col("data"), "struct<name:string,age:bigint>").alias("parsed"))
        rows = result.collect()
        assert rows[0]["parsed"]["name"] == "Alice"
        assert rows[1]["parsed"]["name"] is None
        assert rows[1]["parsed"]["age"] is None

    def test_invalid_json(self, ctx):
        df = _make_df(ctx, "data", ["not_json", '{"name":"Bob","age":25}'])
        result = df.select(from_json(col("data"), "struct<name:string,age:bigint>").alias("parsed"))
        rows = result.collect()
        # Invalid JSON produces null fields
        assert rows[0]["parsed"]["name"] is None
        assert rows[1]["parsed"]["name"] == "Bob"

    def test_missing_fields(self, ctx):
        df = _make_df(ctx, "data", ['{"name":"Carol"}'])
        result = df.select(from_json(col("data"), "struct<name:string,age:bigint>").alias("parsed"))
        rows = result.collect()
        assert rows[0]["parsed"]["name"] == "Carol"
        assert rows[0]["parsed"]["age"] is None

    def test_extra_fields_ignored(self, ctx):
        df = _make_df(ctx, "data", ['{"name":"Dan","age":20,"extra":"ignored"}'])
        result = df.select(from_json(col("data"), "struct<name:string,age:bigint>").alias("parsed"))
        rows = result.collect()
        assert rows[0]["parsed"]["name"] == "Dan"
        assert rows[0]["parsed"]["age"] == 20

    def test_schema_required(self):
        with pytest.raises(ValueError, match="schema is required"):
            from_json(col("data"))

    def test_column_name_string(self, ctx):
        df = _make_df(ctx, "data", ['{"v": 1}'])
        # Pass column as a string instead of Column object
        result = df.select(from_json("data", "struct<v:bigint>").alias("parsed"))
        rows = result.collect()
        assert rows[0]["parsed"]["v"] == 1


# ---------------------------------------------------------------------------
# to_json() tests
# ---------------------------------------------------------------------------


class TestToJson:
    """Tests for to_json() — converts columns to JSON strings."""

    def test_struct_to_json(self):
        ctx = SessionContext()
        batch = pa.record_batch(
            {
                "data": pa.StructArray.from_arrays(
                    [pa.array(["Alice", "Bob"]), pa.array([30, 25])],
                    names=["name", "age"],
                )
            }
        )
        ctx.register_record_batches("t", [[batch]])
        df = DataFrame(ctx.table("t"), session=None)
        result = df.select(to_json(col("data")).alias("j")).collect()
        parsed = [json.loads(r["j"]) for r in result]
        assert parsed[0] == {"name": "Alice", "age": 30}
        assert parsed[1] == {"name": "Bob", "age": 25}

    def test_array_to_json(self):
        ctx = SessionContext()
        batch = pa.record_batch(
            {"items": pa.array([[1, 2, 3], [4, 5]], type=pa.list_(pa.int64()))}
        )
        ctx.register_record_batches("t", [[batch]])
        df = DataFrame(ctx.table("t"), session=None)
        result = df.select(to_json(col("items")).alias("j")).collect()
        assert json.loads(result[0]["j"]) == [1, 2, 3]
        assert json.loads(result[1]["j"]) == [4, 5]

    def test_map_to_json(self):
        ctx = SessionContext()
        batch = pa.record_batch(
            {
                "m": pa.array(
                    [{"a": 1, "b": 2}],
                    type=pa.map_(pa.utf8(), pa.int64()),
                )
            }
        )
        ctx.register_record_batches("t", [[batch]])
        df = DataFrame(ctx.table("t"), session=None)
        result = df.select(to_json(col("m")).alias("j")).collect()
        parsed = json.loads(result[0]["j"])
        assert parsed == [["a", 1], ["b", 2]]  # PyArrow map → list of pairs

    def test_int_to_json(self):
        ctx = SessionContext()
        batch = pa.record_batch({"id": [1, 2, 3]})
        ctx.register_record_batches("t", [[batch]])
        df = DataFrame(ctx.table("t"), session=None)
        result = df.select(to_json(col("id")).alias("j")).collect()
        assert result[0]["j"] == "1"

    def test_string_to_json(self):
        ctx = SessionContext()
        batch = pa.record_batch({"name": ["Alice"]})
        ctx.register_record_batches("t", [[batch]])
        df = DataFrame(ctx.table("t"), session=None)
        result = df.select(to_json(col("name")).alias("j")).collect()
        assert result[0]["j"] == "Alice"

    def test_null_handling(self):
        ctx = SessionContext()
        batch = pa.record_batch(
            {
                "data": pa.StructArray.from_arrays(
                    [pa.array(["Alice", None]), pa.array([30, None])],
                    names=["name", "age"],
                )
            }
        )
        ctx.register_record_batches("t", [[batch]])
        df = DataFrame(ctx.table("t"), session=None)
        result = df.select(to_json(col("data")).alias("j")).collect()
        assert json.loads(result[0]["j"]) == {"name": "Alice", "age": 30}
        # Second row: struct with null fields → still produces JSON
        parsed = json.loads(result[1]["j"])
        assert parsed["name"] is None
        assert parsed["age"] is None
