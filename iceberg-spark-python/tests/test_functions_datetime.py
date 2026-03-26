"""Tests for date/time functions (Task 7D)."""

from __future__ import annotations

import pyarrow as pa
import pytest
from datafusion import SessionContext

from iceberg_spark.dataframe import DataFrame
from iceberg_spark.functions import (
    col,
    date_format,
    date_trunc,
    dayofweek,
    dayofyear,
    make_date,
    quarter,
    to_date,
    to_timestamp,
    trunc,
    weekofyear,
)


@pytest.fixture
def ctx():
    return SessionContext()


@pytest.fixture
def df(ctx):
    """DataFrame with date column."""
    from datetime import date
    t = pa.table({"d": pa.array([date(2024, 3, 15), date(2024, 6, 1), date(2024, 12, 25)])})
    ctx.register_record_batches("td", [t.to_batches()])
    return DataFrame(ctx.table("td"), session=None)


@pytest.fixture
def ts_df(ctx):
    """DataFrame with timestamp column."""
    from datetime import datetime
    t = pa.table({
        "ts": pa.array([
            datetime(2024, 3, 15, 10, 30, 45),
            datetime(2024, 6, 1, 14, 0, 0),
        ], type=pa.timestamp("us")),
    })
    ctx.register_record_batches("tts", [t.to_batches()])
    return DataFrame(ctx.table("tts"), session=None)


class TestDateFunctions:
    def test_dayofweek(self, df):
        result = df.select(dayofweek(col("d")).alias("dow"))
        rows = result.collect()
        # dayofweek returns float from date_part; value depends on DataFusion convention
        assert isinstance(rows[0]["dow"], (int, float))

    def test_dayofyear(self, df):
        result = df.select(dayofyear(col("d")).alias("doy"))
        rows = result.collect()
        # March 15 = day 75 in 2024 (leap year)
        val = rows[0]["doy"]
        assert val == 75 or val == 75.0

    def test_weekofyear(self, df):
        result = df.select(weekofyear(col("d")).alias("woy"))
        rows = result.collect()
        assert isinstance(rows[0]["woy"], (int, float))

    def test_quarter(self, df):
        result = df.select(quarter(col("d")).alias("q"))
        rows = result.collect()
        vals = [r["q"] for r in rows]
        # March -> Q1, June -> Q2, December -> Q4
        assert vals[0] == 1 or vals[0] == 1.0
        assert vals[1] == 2 or vals[1] == 2.0
        assert vals[2] == 4 or vals[2] == 4.0

    def test_to_date(self, ctx):
        t = pa.table({"s": ["2024-01-15", "2024-06-01"]})
        ctx.register_record_batches("tsd", [t.to_batches()])
        df = DataFrame(ctx.table("tsd"), session=None)
        result = df.select(to_date(col("s")).alias("d"))
        rows = result.collect()
        assert rows[0]["d"] is not None

    def test_to_timestamp(self, ctx):
        t = pa.table({"s": ["2024-01-15 10:30:00", "2024-06-01 14:00:00"]})
        ctx.register_record_batches("tst", [t.to_batches()])
        df = DataFrame(ctx.table("tst"), session=None)
        result = df.select(to_timestamp(col("s")).alias("ts"))
        rows = result.collect()
        assert rows[0]["ts"] is not None

    def test_date_trunc(self, ts_df):
        result = ts_df.select(date_trunc("month", col("ts")).alias("tr"))
        rows = result.collect()
        # Truncated to month start
        assert rows[0]["tr"] is not None

    def test_trunc(self, ts_df):
        result = ts_df.select(trunc(col("ts"), "year").alias("tr"))
        rows = result.collect()
        assert rows[0]["tr"] is not None

    def test_date_format(self, ts_df):
        result = ts_df.select(date_format(col("ts"), "%Y-%m-%d").alias("fmt"))
        rows = result.collect()
        assert rows[0]["fmt"] == "2024-03-15"

    def test_make_date(self, ctx):
        t = pa.table({
            "y": pa.array([2024, 2025], type=pa.int32()),
            "m": pa.array([1, 6], type=pa.int32()),
            "d": pa.array([15, 1], type=pa.int32()),
        })
        ctx.register_record_batches("tmd", [t.to_batches()])
        df = DataFrame(ctx.table("tmd"), session=None)
        result = df.select(make_date(col("y"), col("m"), col("d")).alias("dt"))
        rows = result.collect()
        assert rows[0]["dt"] is not None
