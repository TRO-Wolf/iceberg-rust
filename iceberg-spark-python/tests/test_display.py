"""Tests for display formatting utilities."""

import pyarrow as pa

from iceberg_spark._internal.display import format_schema, format_table
from iceberg_spark.types import IntegerType, StringType, StructField, StructType


def test_format_table_basic():
    table = pa.table({"id": [1, 2, 3], "name": ["Alice", "Bob", "Carol"]})
    output = format_table(table, n=20, truncate=20)
    assert "+---+" in output or "+" in output
    assert "id" in output
    assert "name" in output
    assert "Alice" in output
    assert "Bob" in output
    assert "Carol" in output


def test_format_table_with_nulls():
    table = pa.table({"id": [1, None, 3], "val": ["a", "b", None]})
    output = format_table(table, n=20, truncate=20)
    assert "null" in output


def test_format_table_truncation():
    table = pa.table({"long_value": ["x" * 50]})
    output = format_table(table, n=20, truncate=10)
    assert "..." in output


def test_format_table_no_truncation():
    table = pa.table({"long_value": ["x" * 50]})
    output = format_table(table, n=20, truncate=False)
    assert "x" * 50 in output


def test_format_table_limit():
    table = pa.table({"id": list(range(100))})
    output = format_table(table, n=5, truncate=20)
    assert "only showing top 5 rows" in output


def test_format_schema():
    schema = StructType([
        StructField("id", IntegerType(), nullable=False),
        StructField("name", StringType(), nullable=True),
    ])
    output = format_schema(schema)
    assert "root" in output
    assert "|-- id: int (nullable = false)" in output
    assert "|-- name: string (nullable = true)" in output


def test_format_empty_table():
    table = pa.table({"id": pa.array([], type=pa.int32())})
    output = format_table(table, n=20, truncate=20)
    assert "id" in output
