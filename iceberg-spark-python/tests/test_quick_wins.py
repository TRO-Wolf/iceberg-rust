"""Tests for quick-win improvements: JSON reader, text routing, error messages, writer_v2 fixes."""

from __future__ import annotations

import json
import os
import tempfile
import warnings
from unittest.mock import patch

import pyarrow as pa
import pytest
from datafusion import SessionContext

from iceberg_spark.dataframe import DataFrame
from iceberg_spark.reader import DataFrameReader
from iceberg_spark.writer import DataFrameWriter
from iceberg_spark.writer_v2 import DataFrameWriterV2


class _MockSession:
    """Lightweight mock session with a real DataFusion SessionContext."""

    def __init__(self):
        self._ctx = SessionContext()


@pytest.fixture
def mock_session():
    return _MockSession()


@pytest.fixture
def ctx():
    return SessionContext()


@pytest.fixture
def df(ctx):
    t = pa.table({"id": [1, 2, 3], "name": ["a", "b", "c"]})
    ctx.register_record_batches("t", [t.to_batches()])
    return DataFrame(ctx.table("t"), session=None)


# ---------- A) JSON reader via format("json").load(path) ----------


class TestJsonReader:
    def test_json_load(self, mock_session):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write('{"id": 1, "name": "a"}\n')
            f.write('{"id": 2, "name": "b"}\n')
            path = f.name

        try:
            reader = DataFrameReader(mock_session)
            result = reader.format("json").load(path)
            assert isinstance(result, DataFrame)
            rows = result._df.to_arrow_table()
            assert rows.num_rows == 2
            assert set(rows.column_names) >= {"id", "name"}
            ids = sorted(rows.column("id").to_pylist())
            assert ids == [1, 2]
            names = sorted(rows.column("name").to_pylist())
            assert names == ["a", "b"]
        finally:
            os.unlink(path)

    def test_json_shortcut(self, mock_session):
        """Test the .json(path) shortcut routes through format('json').load()."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write('{"x": 42}\n')
            path = f.name

        try:
            reader = DataFrameReader(mock_session)
            result = reader.json(path)
            assert isinstance(result, DataFrame)
            rows = result._df.to_arrow_table()
            assert rows.num_rows == 1
        finally:
            os.unlink(path)


# ---------- B) Text format routing via format("text").load(path) ----------


class TestTextReaderRouting:
    def test_text_load(self, mock_session):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write("hello\n")
            f.write("world\n")
            path = f.name

        try:
            reader = DataFrameReader(mock_session)
            result = reader.format("text").load(path)
            assert isinstance(result, DataFrame)
            rows = result._df.to_arrow_table()
            assert rows.column_names == ["value"]
            values = rows.column("value").to_pylist()
            assert values == ["hello", "world"]
        finally:
            os.unlink(path)


# ---------- C) Unsupported format error message ----------


class TestUnsupportedFormatError:
    def test_error_mentions_supported_formats(self, mock_session):
        reader = DataFrameReader(mock_session)
        with pytest.raises(NotImplementedError, match="Supported formats"):
            reader.format("avro").load("/some/path")

    def test_error_mentions_orc_advice(self, mock_session):
        reader = DataFrameReader(mock_session)
        with pytest.raises(NotImplementedError, match="ORC files"):
            reader.format("avro").load("/some/path")


# ---------- D) overwritePartitions requires active session ----------


class TestOverwritePartitionsRequiresSession:
    def test_raises_without_session(self, df):
        writer = DataFrameWriterV2(df, "db.table")
        with pytest.raises(RuntimeError, match="requires an active session"):
            writer.overwritePartitions()


# ---------- E) using() validation ----------


class TestUsingValidation:
    def test_using_parquet_raises_value_error(self, df):
        writer = DataFrameWriterV2(df, "db.table")
        with pytest.raises(ValueError, match="Unsupported provider"):
            writer.using("parquet")

    def test_using_iceberg_succeeds(self, df):
        writer = DataFrameWriterV2(df, "db.table")
        result = writer.using("iceberg")
        assert result._provider == "iceberg"
        assert result is writer

    def test_using_iceberg_case_insensitive(self, df):
        writer = DataFrameWriterV2(df, "db.table")
        result = writer.using("Iceberg")
        assert result._provider == "Iceberg"


# ---------- F) ORC error messages ----------


class TestOrcErrorMessages:
    def test_orc_reader_mentions_parquet(self, mock_session):
        reader = DataFrameReader(mock_session)
        with pytest.raises(NotImplementedError, match="Parquet"):
            reader.orc("/some/data.orc")

    def test_orc_writer_mentions_parquet(self, df):
        writer = DataFrameWriter(df)
        with pytest.raises(NotImplementedError, match="parquet"):
            writer.orc("/some/output.orc")
