"""Tests for math/trig/hash functions (Task 7C)."""

from __future__ import annotations

import math

import pyarrow as pa
import pytest
from datafusion import SessionContext

from iceberg_spark.dataframe import DataFrame
from iceberg_spark.functions import (
    acos,
    asin,
    atan,
    atan2,
    cbrt,
    col,
    cos,
    cosh,
    cot,
    degrees,
    e,
    factorial,
    hex_func,
    lit,
    log10,
    log1p,
    log2,
    md5,
    pi,
    pmod,
    pow,
    radians,
    sha1,
    sha2,
    sign,
    signum,
    sin,
    sinh,
    tan,
    tanh,
)


@pytest.fixture
def ctx():
    return SessionContext()


@pytest.fixture
def df(ctx):
    t = pa.table({"x": [1.0, 4.0, 8.0, 16.0, 27.0]})
    ctx.register_record_batches("tm", [t.to_batches()])
    return DataFrame(ctx.table("tm"), session=None)


class TestMathFunctions:
    def test_log2(self, df):
        result = df.select(log2(col("x")).alias("l2"))
        rows = result.collect()
        assert abs(rows[0]["l2"] - 0.0) < 0.01  # log2(1) = 0
        assert abs(rows[1]["l2"] - 2.0) < 0.01  # log2(4) = 2

    def test_log10(self, ctx):
        t = pa.table({"x": [1.0, 10.0, 100.0]})
        ctx.register_record_batches("tl10", [t.to_batches()])
        df = DataFrame(ctx.table("tl10"), session=None)
        result = df.select(log10(col("x")).alias("l10"))
        rows = result.collect()
        assert abs(rows[1]["l10"] - 1.0) < 0.01

    def test_log1p(self, ctx):
        t = pa.table({"x": [0.0, 1.0]})
        ctx.register_record_batches("tl1p", [t.to_batches()])
        df = DataFrame(ctx.table("tl1p"), session=None)
        result = df.select(log1p(col("x")).alias("lp"))
        rows = result.collect()
        assert abs(rows[0]["lp"] - 0.0) < 0.01  # ln(0+1) = 0
        assert abs(rows[1]["lp"] - math.log(2)) < 0.01

    def test_cbrt(self, df):
        result = df.select(cbrt(col("x")).alias("cb"))
        rows = result.collect()
        assert abs(rows[4]["cb"] - 3.0) < 0.01  # cbrt(27) = 3

    def test_pow(self, ctx):
        t = pa.table({"x": [2.0, 3.0]})
        ctx.register_record_batches("tpow", [t.to_batches()])
        df = DataFrame(ctx.table("tpow"), session=None)
        result = df.select(pow(col("x"), 3).alias("p"))
        rows = result.collect()
        assert abs(rows[0]["p"] - 8.0) < 0.01
        assert abs(rows[1]["p"] - 27.0) < 0.01

    def test_signum(self, ctx):
        t = pa.table({"x": [-5.0, 0.0, 3.0]})
        ctx.register_record_batches("tsig", [t.to_batches()])
        df = DataFrame(ctx.table("tsig"), session=None)
        result = df.select(signum(col("x")).alias("s"))
        rows = result.collect()
        assert rows[0]["s"] == -1.0
        assert rows[1]["s"] == 0.0
        assert rows[2]["s"] == 1.0

    def test_sign_alias(self, ctx):
        t = pa.table({"x": [-1.0, 1.0]})
        ctx.register_record_batches("tsign", [t.to_batches()])
        df = DataFrame(ctx.table("tsign"), session=None)
        result = df.select(sign(col("x")).alias("s"))
        rows = result.collect()
        assert rows[0]["s"] == -1.0
        assert rows[1]["s"] == 1.0

    def test_degrees_radians(self, ctx):
        t = pa.table({"x": [math.pi, math.pi / 2]})
        ctx.register_record_batches("tdr", [t.to_batches()])
        df = DataFrame(ctx.table("tdr"), session=None)
        result = df.select(degrees(col("x")).alias("d"))
        rows = result.collect()
        assert abs(rows[0]["d"] - 180.0) < 0.01

    def test_radians(self, ctx):
        t = pa.table({"x": [180.0, 90.0]})
        ctx.register_record_batches("trad", [t.to_batches()])
        df = DataFrame(ctx.table("trad"), session=None)
        result = df.select(radians(col("x")).alias("r"))
        rows = result.collect()
        assert abs(rows[0]["r"] - math.pi) < 0.01

    def test_factorial(self, ctx):
        t = pa.table({"x": pa.array([5], type=pa.int64())})
        ctx.register_record_batches("tfact", [t.to_batches()])
        df = DataFrame(ctx.table("tfact"), session=None)
        result = df.select(factorial(col("x")).alias("f"))
        rows = result.collect()
        assert rows[0]["f"] == 120

    def test_pi_and_e(self, ctx):
        t = pa.table({"x": [1]})
        ctx.register_record_batches("tpi", [t.to_batches()])
        df = DataFrame(ctx.table("tpi"), session=None)
        result = df.select(pi().alias("p"), e().alias("eu"))
        rows = result.collect()
        assert abs(rows[0]["p"] - math.pi) < 0.0001
        assert abs(rows[0]["eu"] - math.e) < 0.0001

    def test_pmod(self, ctx):
        t = pa.table({"a": [-7, 7], "b": [3, 3]})
        ctx.register_record_batches("tpmod", [t.to_batches()])
        df = DataFrame(ctx.table("tpmod"), session=None)
        result = df.select(pmod(col("a"), col("b")).alias("pm"))
        rows = result.collect()
        assert rows[0]["pm"] == 2  # ((-7 % 3) + 3) % 3 = 2
        assert rows[1]["pm"] == 1  # ((7 % 3) + 3) % 3 = 1


class TestTrigFunctions:
    def test_sin_cos(self, ctx):
        t = pa.table({"x": [0.0, math.pi / 2, math.pi]})
        ctx.register_record_batches("ttrig", [t.to_batches()])
        df = DataFrame(ctx.table("ttrig"), session=None)
        result = df.select(sin(col("x")).alias("s"), cos(col("x")).alias("c"))
        rows = result.collect()
        assert abs(rows[0]["s"] - 0.0) < 0.01
        assert abs(rows[0]["c"] - 1.0) < 0.01
        assert abs(rows[1]["s"] - 1.0) < 0.01

    def test_tan(self, ctx):
        t = pa.table({"x": [0.0]})
        ctx.register_record_batches("ttan", [t.to_batches()])
        df = DataFrame(ctx.table("ttan"), session=None)
        result = df.select(tan(col("x")).alias("t"))
        rows = result.collect()
        assert abs(rows[0]["t"] - 0.0) < 0.01

    def test_asin_acos_atan(self, ctx):
        t = pa.table({"x": [0.0, 1.0]})
        ctx.register_record_batches("tinv", [t.to_batches()])
        df = DataFrame(ctx.table("tinv"), session=None)
        result = df.select(
            asin(col("x")).alias("as_"),
            acos(col("x")).alias("ac"),
            atan(col("x")).alias("at"),
        )
        rows = result.collect()
        assert abs(rows[0]["as_"] - 0.0) < 0.01
        assert abs(rows[1]["ac"] - 0.0) < 0.01  # acos(1) = 0

    def test_atan2(self, ctx):
        t = pa.table({"y": [1.0], "x": [1.0]})
        ctx.register_record_batches("tat2", [t.to_batches()])
        df = DataFrame(ctx.table("tat2"), session=None)
        result = df.select(atan2(col("y"), col("x")).alias("a2"))
        rows = result.collect()
        assert abs(rows[0]["a2"] - math.pi / 4) < 0.01

    def test_sinh_cosh_tanh(self, ctx):
        t = pa.table({"x": [0.0]})
        ctx.register_record_batches("thyp", [t.to_batches()])
        df = DataFrame(ctx.table("thyp"), session=None)
        result = df.select(
            sinh(col("x")).alias("sh"),
            cosh(col("x")).alias("ch"),
            tanh(col("x")).alias("th"),
        )
        rows = result.collect()
        assert abs(rows[0]["sh"] - 0.0) < 0.01
        assert abs(rows[0]["ch"] - 1.0) < 0.01
        assert abs(rows[0]["th"] - 0.0) < 0.01


class TestHashFunctions:
    def test_md5(self, ctx):
        t = pa.table({"s": ["hello"]})
        ctx.register_record_batches("thash", [t.to_batches()])
        df = DataFrame(ctx.table("thash"), session=None)
        result = df.select(md5(col("s")).alias("h"))
        rows = result.collect()
        assert isinstance(rows[0]["h"], str)
        assert len(rows[0]["h"]) == 32  # MD5 hex is 32 chars

    def test_sha1(self, ctx):
        """sha1() uses SHA-224 since DataFusion doesn't support SHA-1."""
        t = pa.table({"s": ["hello"]})
        ctx.register_record_batches("tsha1", [t.to_batches()])
        df = DataFrame(ctx.table("tsha1"), session=None)
        result = df.select(sha1(col("s")).alias("h"))
        rows = result.collect()
        assert isinstance(rows[0]["h"], str)
        assert len(rows[0]["h"]) == 56  # SHA-224 hex is 56 chars

    def test_sha2(self, ctx):
        t = pa.table({"s": ["hello"]})
        ctx.register_record_batches("tsha2", [t.to_batches()])
        df = DataFrame(ctx.table("tsha2"), session=None)
        result = df.select(sha2(col("s")).alias("h"))
        rows = result.collect()
        assert isinstance(rows[0]["h"], str)
        assert len(rows[0]["h"]) == 64  # SHA-256 hex is 64 chars

    def test_hex_func(self, ctx):
        t = pa.table({"n": pa.array([255, 16, 0], type=pa.int64())})
        ctx.register_record_batches("thex", [t.to_batches()])
        df = DataFrame(ctx.table("thex"), session=None)
        result = df.select(hex_func(col("n")).alias("h"))
        rows = result.collect()
        assert rows[0]["h"].lower() in ("ff", "00000000000000ff")
        assert rows[2]["h"].lower() in ("0", "00", "0000000000000000")
