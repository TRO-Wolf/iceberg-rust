"""Tests for Sprint 2 Task 2.1: Date/time function gaps."""

from __future__ import annotations

import datetime

import pyarrow as pa
import pytest
from datafusion import SessionContext

from iceberg_spark.dataframe import DataFrame
from iceberg_spark.functions import (
    add_months,
    col,
    date_sub,
    from_unixtime,
    last_day,
    lit,
    months_between,
    next_day,
    unix_timestamp,
)


@pytest.fixture
def ctx():
    return SessionContext()


@pytest.fixture
def df(ctx):
    t = pa.table({
        "d": [datetime.date(2024, 1, 15), datetime.date(2024, 6, 30), datetime.date(2024, 12, 31)],
        "ts_str": ["2024-01-15 10:30:00", "2024-06-30 00:00:00", "2024-12-31 23:59:59"],
    })
    ctx.register_record_batches("t", [t.to_batches()])
    return DataFrame(ctx.table("t"), session=None)


class TestUnixTimestamp:
    def test_from_date_column(self, df):
        result = df.select(unix_timestamp(col("d")).alias("epoch"))
        rows = result.collect()
        # 2024-01-15 => some epoch value
        assert isinstance(rows[0]["epoch"], (int, float))
        assert rows[0]["epoch"] > 0

    def test_from_string_column(self, df):
        result = df.select(unix_timestamp(col("ts_str")).alias("epoch"))
        rows = result.collect()
        assert rows[0]["epoch"] > 0


class TestFromUnixtime:
    def test_basic(self, ctx):
        t = pa.table({"epoch": [1705276800]})  # 2024-01-15 00:00:00 UTC
        ctx.register_record_batches("t2", [t.to_batches()])
        df = DataFrame(ctx.table("t2"), session=None)
        result = df.select(from_unixtime(col("epoch")).alias("ts"))
        rows = result.collect()
        ts = rows[0]["ts"]
        assert ts is not None


class TestDateSub:
    def test_subtract_days(self, df):
        result = df.select(date_sub(col("d"), 5).alias("d2"))
        rows = result.collect()
        assert rows[0]["d2"] == datetime.date(2024, 1, 10)

    def test_subtract_zero(self, df):
        result = df.select(date_sub(col("d"), 0).alias("d2"))
        rows = result.collect()
        assert rows[0]["d2"] == datetime.date(2024, 1, 15)


class TestAddMonths:
    def test_add_positive(self, df):
        result = df.select(add_months(col("d"), 3).alias("d2"))
        rows = result.collect()
        # 2024-01-15 + 3 months = 2024-04-15
        assert rows[0]["d2"] == datetime.date(2024, 4, 15)

    def test_add_negative(self, ctx):
        # Use a date where day fits in all months
        t = pa.table({"d": [datetime.date(2024, 3, 15)]})
        ctx.register_record_batches("t_neg", [t.to_batches()])
        df = DataFrame(ctx.table("t_neg"), session=None)
        result = df.select(add_months(col("d"), -1).alias("d2"))
        rows = result.collect()
        # 2024-03-15 - 1 month = 2024-02-15
        assert rows[0]["d2"] == datetime.date(2024, 2, 15)

    def test_year_wrap(self, df):
        result = df.select(add_months(col("d"), 12).alias("d2"))
        rows = result.collect()
        # 2024-01-15 + 12 months = 2025-01-15
        assert rows[0]["d2"] == datetime.date(2025, 1, 15)


class TestMonthsBetween:
    def test_same_day(self, ctx):
        t = pa.table({
            "d1": [datetime.date(2024, 6, 15)],
            "d2": [datetime.date(2024, 1, 15)],
        })
        ctx.register_record_batches("t3", [t.to_batches()])
        df = DataFrame(ctx.table("t3"), session=None)
        result = df.select(months_between(col("d1"), col("d2")).alias("m"))
        rows = result.collect()
        # Should be approximately 5.0
        assert abs(rows[0]["m"] - 5.0) < 0.1


class TestLastDay:
    def test_january(self, ctx):
        t = pa.table({"d": [datetime.date(2024, 1, 15)]})
        ctx.register_record_batches("t4", [t.to_batches()])
        df = DataFrame(ctx.table("t4"), session=None)
        result = df.select(last_day(col("d")).alias("ld"))
        rows = result.collect()
        assert rows[0]["ld"] == datetime.date(2024, 1, 31)

    def test_february_leap(self, ctx):
        t = pa.table({"d": [datetime.date(2024, 2, 1)]})
        ctx.register_record_batches("t5", [t.to_batches()])
        df = DataFrame(ctx.table("t5"), session=None)
        result = df.select(last_day(col("d")).alias("ld"))
        rows = result.collect()
        assert rows[0]["ld"] == datetime.date(2024, 2, 29)


class TestNextDay:
    def test_next_monday(self, ctx):
        # 2024-01-15 is a Monday
        t = pa.table({"d": [datetime.date(2024, 1, 15)]})
        ctx.register_record_batches("t6", [t.to_batches()])
        df = DataFrame(ctx.table("t6"), session=None)
        result = df.select(next_day(col("d"), "Monday").alias("nd"))
        rows = result.collect()
        # Next Monday after Monday 2024-01-15 = 2024-01-22
        assert rows[0]["nd"] == datetime.date(2024, 1, 22)

    def test_next_friday(self, ctx):
        t = pa.table({"d": [datetime.date(2024, 1, 15)]})  # Monday
        ctx.register_record_batches("t7", [t.to_batches()])
        df = DataFrame(ctx.table("t7"), session=None)
        result = df.select(next_day(col("d"), "Fri").alias("nd"))
        rows = result.collect()
        assert rows[0]["nd"] == datetime.date(2024, 1, 19)

    def test_invalid_day_raises(self):
        with pytest.raises(ValueError, match="Unknown day"):
            next_day(col("d"), "NotADay")


class TestImports:
    def test_import_from_package(self):
        from iceberg_spark import (
            add_months,
            date_add,
            date_sub,
            from_unixtime,
            last_day,
            months_between,
            next_day,
            unix_timestamp,
        )
        assert all(callable(f) for f in [
            add_months, date_add, date_sub, from_unixtime,
            last_day, months_between, next_day, unix_timestamp,
        ])
