"""Tests for unpivot / melt, approxQuantile on DataFrame, and simple stubs."""

from __future__ import annotations

import pyarrow as pa
import pytest
from datafusion import SessionContext

from iceberg_spark.dataframe import DataFrame


class _MiniSession:
    """Minimal session stub that exposes a DataFusion SessionContext."""

    def __init__(self):
        self._ctx = SessionContext()


@pytest.fixture()
def session():
    return _MiniSession()


@pytest.fixture()
def df(session):
    t = pa.table({
        "year": [2020, 2020, 2021, 2021],
        "Q1": [100.0, 200.0, 300.0, 400.0],
        "Q2": [110.0, 210.0, 310.0, 410.0],
    })
    session._ctx.register_record_batches("unpivot_t", [t.to_batches()])
    return DataFrame(session._ctx.table("unpivot_t"), session=session)


# ---------------------------------------------------------------------------
# unpivot
# ---------------------------------------------------------------------------


class TestUnpivot:
    def test_basic_unpivot(self, df):
        result = df.unpivot(["year"], ["Q1", "Q2"], "quarter", "revenue")
        rows = result.collect()
        # 4 original rows * 2 value columns = 8 rows
        assert len(rows) == 8
        # Check columns
        assert set(result.columns) == {"year", "quarter", "revenue"}
        # Check variable values
        quarters = {r["quarter"] for r in rows}
        assert quarters == {"Q1", "Q2"}

    def test_single_value_column(self, df):
        result = df.unpivot(["year"], ["Q1"], "metric", "val")
        rows = result.collect()
        assert len(rows) == 4
        assert all(r["metric"] == "Q1" for r in rows)

    def test_melt_is_alias(self, df):
        r1 = sorted(
            [(r["year"], r["quarter"], r["revenue"])
             for r in df.unpivot(["year"], ["Q1", "Q2"], "quarter", "revenue").collect()]
        )
        r2 = sorted(
            [(r["year"], r["quarter"], r["revenue"])
             for r in df.melt(["year"], ["Q1", "Q2"], "quarter", "revenue").collect()]
        )
        assert r1 == r2

    def test_empty_values_raises(self, df):
        with pytest.raises(ValueError, match="at least one column"):
            df.unpivot(["year"], [], "v", "val")

    def test_no_session_raises(self):
        ctx = SessionContext()
        t = pa.table({"a": [1], "b": [2.0]})
        ctx.register_record_batches("tmp", [t.to_batches()])
        df = DataFrame(ctx.table("tmp"), session=None)
        with pytest.raises(RuntimeError, match="requires an active session"):
            df.unpivot(["a"], ["b"], "var", "val")


# ---------------------------------------------------------------------------
# approxQuantile on DataFrame (convenience wrapper)
# ---------------------------------------------------------------------------


class TestApproxQuantileOnDataFrame:
    def test_returns_list_of_floats(self, session):
        t = pa.table({"v": [10.0, 20.0, 30.0, 40.0, 50.0]})
        session._ctx.register_record_batches("aq_t", [t.to_batches()])
        df = DataFrame(session._ctx.table("aq_t"), session=session)
        result = df.approxQuantile("v", [0.0, 0.5, 1.0])
        assert isinstance(result, list)
        assert len(result) == 3
        # Median should be roughly 30
        assert 20.0 <= result[1] <= 40.0

    def test_empty_probabilities(self, session):
        t = pa.table({"v": [1.0, 2.0]})
        session._ctx.register_record_batches("aq_t2", [t.to_batches()])
        df = DataFrame(session._ctx.table("aq_t2"), session=session)
        assert df.approxQuantile("v", []) == []


# ---------------------------------------------------------------------------
# Simple stubs
# ---------------------------------------------------------------------------


class TestSimpleStubs:
    def test_isLocal(self, df):
        assert df.isLocal() is True

    def test_isStreaming(self, df):
        assert df.isStreaming() is False

    def test_checkpoint_returns_self(self, df):
        assert df.checkpoint() is df

    def test_localCheckpoint_returns_self(self, df):
        assert df.localCheckpoint() is df

    def test_withWatermark_returns_self(self, df):
        assert df.withWatermark("ts", "10 minutes") is df

    def test_colRegex_matches(self, df):
        result = df.colRegex("`Q.*`")
        assert set(result.columns) == {"Q1", "Q2"}

    def test_colRegex_no_match(self, df):
        result = df.colRegex("`zzz.*`")
        assert result.columns == []
