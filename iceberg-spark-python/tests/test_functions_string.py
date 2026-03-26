"""Tests for string functions (Task 7B)."""

from __future__ import annotations

import pyarrow as pa
import pytest
from datafusion import SessionContext

from iceberg_spark.dataframe import DataFrame
from iceberg_spark.functions import (
    ascii_func,
    chr_func,
    col,
    initcap,
    instr,
    left,
    lit,
    locate,
    regexp_extract,
    repeat_func,
    right,
    translate,
)


@pytest.fixture
def ctx():
    return SessionContext()


@pytest.fixture
def df(ctx):
    t = pa.table({"name": ["alice", "bob", "charlie"], "val": ["hello world", "foo bar", "baz qux"]})
    ctx.register_record_batches("ts", [t.to_batches()])
    return DataFrame(ctx.table("ts"), session=None)


class TestStringFunctions:
    def test_initcap(self, df):
        result = df.select(initcap(col("name")).alias("ic"))
        rows = result.collect()
        assert rows[0]["ic"] == "Alice"
        assert rows[1]["ic"] == "Bob"
        assert rows[2]["ic"] == "Charlie"

    def test_locate(self, df):
        result = df.select(locate("ob", col("name")).alias("pos"))
        rows = result.collect()
        assert rows[1]["pos"] == 2  # "bob" -> "ob" at position 2

    def test_locate_with_pos(self, ctx):
        t = pa.table({"s": ["abcabc"]})
        ctx.register_record_batches("tlp", [t.to_batches()])
        df = DataFrame(ctx.table("tlp"), session=None)
        # Search for "abc" starting from position 2 — should find at position 4
        result = df.select(locate("abc", col("s"), 2).alias("pos"))
        rows = result.collect()
        assert rows[0]["pos"] == 4

    def test_instr(self, df):
        result = df.select(instr(col("name"), "li").alias("pos"))
        rows = result.collect()
        assert rows[0]["pos"] == 2  # "alice" -> "li" at position 2

    def test_left(self, df):
        result = df.select(left(col("name"), 3).alias("l"))
        rows = result.collect()
        assert rows[0]["l"] == "ali"
        assert rows[1]["l"] == "bob"
        assert rows[2]["l"] == "cha"

    def test_right(self, df):
        result = df.select(right(col("name"), 2).alias("r"))
        rows = result.collect()
        assert rows[0]["r"] == "ce"
        assert rows[1]["r"] == "ob"

    def test_translate(self, df):
        result = df.select(translate(col("name"), "aeiou", "AEIOU").alias("t"))
        rows = result.collect()
        assert rows[0]["t"] == "AlIcE"

    def test_repeat_func(self, df):
        result = df.select(repeat_func(col("name"), 2).alias("r"))
        rows = result.collect()
        assert rows[1]["r"] == "bobbob"

    def test_ascii_func(self, ctx):
        t = pa.table({"c": ["A", "a", "0"]})
        ctx.register_record_batches("ta", [t.to_batches()])
        df = DataFrame(ctx.table("ta"), session=None)
        result = df.select(ascii_func(col("c")).alias("a"))
        rows = result.collect()
        assert rows[0]["a"] == 65  # 'A'
        assert rows[1]["a"] == 97  # 'a'

    def test_chr_func(self, ctx):
        t = pa.table({"n": [65, 97, 48]})
        ctx.register_record_batches("tc", [t.to_batches()])
        df = DataFrame(ctx.table("tc"), session=None)
        result = df.select(chr_func(col("n")).alias("c"))
        rows = result.collect()
        assert rows[0]["c"] == "A"
        assert rows[1]["c"] == "a"

    def test_regexp_extract(self, ctx):
        t = pa.table({"s": ["abc123", "def456", "ghi"]})
        ctx.register_record_batches("tre", [t.to_batches()])
        df = DataFrame(ctx.table("tre"), session=None)
        # idx=0: extract the whole match
        result = df.select(regexp_extract(col("s"), r"\d+", 0).alias("m"))
        rows = result.collect()
        assert rows[0]["m"] == "123"
        assert rows[1]["m"] == "456"
        assert rows[2]["m"] is None

    def test_regexp_extract_group(self, ctx):
        t = pa.table({"s": ["John-42", "Jane-25"]})
        ctx.register_record_batches("treg", [t.to_batches()])
        df = DataFrame(ctx.table("treg"), session=None)
        # idx=1: extract first capture group
        result = df.select(regexp_extract(col("s"), r"(\w+)-(\d+)", 1).alias("name"))
        rows = result.collect()
        assert rows[0]["name"] == "John"
        assert rows[1]["name"] == "Jane"
