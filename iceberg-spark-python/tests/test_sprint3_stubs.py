"""Tests for Sprint 3: DataFrameWriterV2, DataFrame stubs, reader/writer additions."""

from __future__ import annotations

import pyarrow as pa
import pytest
from datafusion import SessionContext

from iceberg_spark.dataframe import DataFrame, StorageLevel
from iceberg_spark.writer_v2 import DataFrameWriterV2


@pytest.fixture
def ctx():
    return SessionContext()


@pytest.fixture
def df(ctx):
    t = pa.table({"id": [1, 2, 3], "name": ["a", "b", "c"]})
    ctx.register_record_batches("t", [t.to_batches()])
    return DataFrame(ctx.table("t"), session=None)


class TestDataFrameWriterV2:
    def test_writeto_returns_writer(self, df):
        writer = df.writeTo("db.table")
        assert isinstance(writer, DataFrameWriterV2)

    def test_using_chain(self, df):
        writer = df.writeTo("db.table").using("iceberg")
        assert writer._provider == "iceberg"

    def test_table_property_chain(self, df):
        writer = df.writeTo("db.table").tableProperty("k", "v")
        assert writer._properties == {"k": "v"}

    def test_append_requires_session(self, df):
        with pytest.raises(RuntimeError, match="requires an active session"):
            df.writeTo("db.table").append()

    def test_create_requires_session(self, df):
        with pytest.raises(RuntimeError, match="requires an active session"):
            df.writeTo("db.table").create()

    def test_replace_requires_session(self, df):
        with pytest.raises(RuntimeError, match="requires an active session"):
            df.writeTo("db.table").replace()


class TestDataFrameStubs:
    def test_input_files(self, df):
        assert df.inputFiles() == []

    def test_storage_level(self, df):
        assert str(df.storageLevel) == "MEMORY_AND_DISK"

    def test_observe_noop(self, df):
        result = df.observe("metrics", "count")
        assert result is df

    def test_same_semantics(self, df):
        assert df.sameSemantics(df) is True

    def test_semantic_hash(self, df):
        h = df.semanticHash()
        assert isinstance(h, int)

    def test_map_in_pandas_works(self, df):
        """mapInPandas is now implemented (no longer raises)."""
        def identity(iterator):
            for pdf in iterator:
                yield pdf
        result = df.mapInPandas(identity, schema=df._df.to_arrow_table().schema)
        assert result.count() == df.count()

    def test_map_in_arrow_works(self, df):
        """mapInArrow is now implemented (no longer raises)."""
        def identity(iterator):
            for batch in iterator:
                yield batch
        result = df.mapInArrow(identity, schema=df._df.to_arrow_table().schema)
        assert result.count() == df.count()


class TestStorageLevel:
    def test_repr(self):
        sl = StorageLevel("MEMORY_ONLY")
        assert repr(sl) == "StorageLevel(MEMORY_ONLY)"

    def test_equality(self):
        sl1 = StorageLevel("MEMORY_ONLY")
        sl2 = StorageLevel("MEMORY_ONLY")
        assert sl1 == sl2

    def test_inequality(self):
        sl1 = StorageLevel("MEMORY_ONLY")
        sl2 = StorageLevel("DISK_ONLY")
        assert sl1 != sl2


class TestWriterAdditions:
    def test_bucket_by_noop(self, df):
        from iceberg_spark.writer import DataFrameWriter
        w = DataFrameWriter(df)
        result = w.bucketBy(10, "id")
        assert result is w

    def test_sort_by(self, df):
        from iceberg_spark.writer import DataFrameWriter
        w = DataFrameWriter(df)
        result = w.sortBy("id")
        assert result is w


class TestImports:
    def test_writer_v2_importable(self):
        from iceberg_spark import DataFrameWriterV2
        assert DataFrameWriterV2 is not None
