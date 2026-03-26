"""Tests for Task 12A (exports) and Task 12B (StatFunctions completeness)."""

from __future__ import annotations

import inspect

import pyarrow as pa
import pytest
from datafusion import SessionContext

from iceberg_spark.dataframe import DataFrame


# ---------------------------------------------------------------------------
# Task 12A — import / export smoke tests
# ---------------------------------------------------------------------------

class TestExports:
    def test_grouped_data_is_class(self):
        from iceberg_spark import GroupedData
        assert inspect.isclass(GroupedData)

    def test_dataframe_na_is_class(self):
        from iceberg_spark import DataFrameNaFunctions
        assert inspect.isclass(DataFrameNaFunctions)

    def test_dataframe_stat_is_class(self):
        from iceberg_spark import DataFrameStatFunctions
        assert inspect.isclass(DataFrameStatFunctions)

    def test_dataframe_reader_is_class(self):
        from iceberg_spark import DataFrameReader
        assert inspect.isclass(DataFrameReader)

    def test_dataframe_writer_is_class(self):
        from iceberg_spark import DataFrameWriter
        assert inspect.isclass(DataFrameWriter)

    def test_all_in___all__(self):
        import iceberg_spark
        for name in [
            "GroupedData",
            "DataFrameNaFunctions",
            "DataFrameStatFunctions",
            "DataFrameReader",
            "DataFrameWriter",
        ]:
            assert name in iceberg_spark.__all__, f"{name!r} missing from __all__"

    def test_stat_accessor_returns_stat_functions(self):
        """df.stat should return a DataFrameStatFunctions instance."""
        from iceberg_spark import DataFrameStatFunctions
        ctx = SessionContext()
        t = pa.table({"x": [1, 2, 3]})
        ctx.register_record_batches("tmp_stat_t", [t.to_batches()])
        df = DataFrame(ctx.table("tmp_stat_t"), session=None)
        assert isinstance(df.stat, DataFrameStatFunctions)

    def test_na_accessor_returns_na_functions(self):
        """df.na should return a DataFrameNaFunctions instance."""
        from iceberg_spark import DataFrameNaFunctions
        ctx = SessionContext()
        t = pa.table({"x": [1, 2, 3]})
        ctx.register_record_batches("tmp_na_t", [t.to_batches()])
        df = DataFrame(ctx.table("tmp_na_t"), session=None)
        assert isinstance(df.na, DataFrameNaFunctions)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ctx():
    return SessionContext()


@pytest.fixture
def df(ctx):
    t = pa.table({
        "id": [1, 2, 3, 4, 5],
        "score": [10.0, 20.0, 30.0, 40.0, 50.0],
        "dept": ["eng", "eng", "sales", "sales", "eng"],
    })
    ctx.register_record_batches("stat_t", [t.to_batches()])
    return DataFrame(ctx.table("stat_t"), session=None)


# ---------------------------------------------------------------------------
# Task 12B — approxQuantile
# ---------------------------------------------------------------------------

class TestApproxQuantile:
    def test_returns_list_of_floats(self, df):
        result = df.stat.approxQuantile("score", [0.25, 0.5, 0.75])
        assert isinstance(result, list)
        assert len(result) == 3
        for v in result:
            assert isinstance(v, float)

    def test_median(self, df):
        result = df.stat.approxQuantile("score", [0.5])
        # Median of [10, 20, 30, 40, 50] should be ~30
        assert 25.0 <= result[0] <= 35.0

    def test_monotone(self, df):
        """Lower quantile must be <= higher quantile."""
        result = df.stat.approxQuantile("score", [0.0, 0.25, 0.5, 0.75, 1.0])
        for a, b in zip(result, result[1:]):
            assert a <= b

    def test_single_probability(self, df):
        result = df.stat.approxQuantile("score", [0.5])
        assert len(result) == 1

    def test_empty_probabilities(self, df):
        result = df.stat.approxQuantile("score", [])
        assert result == []

    def test_empty_table_returns_none_not_crash(self, ctx):
        """approxQuantile on an empty table must return None, not crash."""
        # Use pa.record_batch (not table.to_batches()) to avoid the DataFusion
        # panic when register_record_batches receives an empty inner list.
        batch = pa.record_batch({"v": pa.array([], type=pa.float64())})
        ctx.register_record_batches("empty_t", [[batch]])
        df = DataFrame(ctx.table("empty_t"), session=None)
        result = df.stat.approxQuantile("v", [0.5])
        assert len(result) == 1
        assert result[0] is None  # not a TypeError


# ---------------------------------------------------------------------------
# Task 12B — freqItems (requires session for temp table registration)
# ---------------------------------------------------------------------------

def _make_session_df(ctx):
    """Create a DataFrame backed by a stub session with a real DataFusion ctx."""
    from unittest.mock import MagicMock
    session = MagicMock()
    session._ctx = ctx
    t = pa.table({
        "dept": ["eng", "eng", "eng", "sales", "sales", "hr"],
        "level": ["senior", "senior", "junior", "senior", "junior", "junior"],
    })
    ctx.register_record_batches("freq_t", [t.to_batches()])
    df = DataFrame(ctx.table("freq_t"), session=session)
    return df


class TestFreqItems:
    def test_requires_session(self, ctx):
        """freqItems must raise RuntimeError when session is None."""
        t = pa.table({"x": [1, 2, 3]})
        ctx.register_record_batches("no_sess_t", [t.to_batches()])
        df = DataFrame(ctx.table("no_sess_t"), session=None)
        with pytest.raises(RuntimeError, match="requires an active session"):
            df.stat.freqItems(["x"])

    def test_returns_dataframe(self, ctx):
        df = _make_session_df(ctx)
        result = df.stat.freqItems(["dept"])
        assert isinstance(result, DataFrame)

    def test_column_naming(self, ctx):
        df = _make_session_df(ctx)
        result = df.stat.freqItems(["dept"])
        assert "dept_freqItems" in result.columns

    def test_frequent_item_present(self, ctx):
        df = _make_session_df(ctx)
        # "eng" appears 3/6 = 50%; support=0.4 → min_count=ceil(2.4)=3 → eng qualifies
        result = df.stat.freqItems(["dept"], support=0.4)
        dept_vals = [r["dept_freqItems"] for r in result.collect()]
        assert "eng" in dept_vals

    def test_high_support_excludes_rare(self, ctx):
        df = _make_session_df(ctx)
        # "hr" appears 1/6 ≈ 17%; support=0.3 → min_count=ceil(1.8)=2 → hr excluded
        result = df.stat.freqItems(["dept"], support=0.3)
        dept_vals = [r["dept_freqItems"] for r in result.collect()]
        assert "hr" not in dept_vals

    def test_multiple_columns_schema(self, ctx):
        """Result DataFrame always has both freqItems columns regardless of row count."""
        df = _make_session_df(ctx)
        result = df.stat.freqItems(["dept", "level"], support=0.4)
        # Columns are determined by schema, not row count — check unconditionally
        assert "dept_freqItems" in result.columns
        assert "level_freqItems" in result.columns

    def test_very_high_support_returns_empty_rows(self, ctx):
        """support=1.0 means items must appear in 100% of rows — likely none qualify."""
        df = _make_session_df(ctx)
        result = df.stat.freqItems(["dept"], support=1.0)
        # Column still exists even with 0 rows
        assert "dept_freqItems" in result.columns
