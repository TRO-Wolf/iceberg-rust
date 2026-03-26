"""Tests for Phase 3 remaining features.

Covers:
- SQL preprocessor: METADATA_TABLE detection
- DataFrameWriter: parquet/csv/json path-based saves
- Spark Catalog API: IcebergCatalogAPI (unit tests using a stub catalog)

These tests only require datafusion + pyarrow (no Iceberg catalog needed).
"""

from __future__ import annotations

import os
import pathlib
import tempfile

import pyarrow as pa
import pytest
from datafusion import SessionContext

from iceberg_spark.catalog_api import IcebergCatalogAPI
from iceberg_spark.dataframe import DataFrame
from iceberg_spark.sql_preprocessor import CommandType, preprocess
from iceberg_spark.writer import DataFrameWriter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ctx():
    return SessionContext()


@pytest.fixture
def sample_arrow():
    return pa.table({
        "id": pa.array([1, 2, 3], type=pa.int64()),
        "name": pa.array(["a", "b", "c"], type=pa.string()),
    })


@pytest.fixture
def df(ctx, sample_arrow):
    ctx.register_record_batches("t", [sample_arrow.to_batches()])
    return DataFrame(ctx.table("t"), session=None)


# ---------------------------------------------------------------------------
# SQL Preprocessor — METADATA_TABLE detection
# ---------------------------------------------------------------------------


class TestMetadataTablePreprocessor:
    def test_snapshots(self):
        result = preprocess("SELECT * FROM db.t1.snapshots")
        assert result.command_type == CommandType.METADATA_TABLE
        assert result.table_name == "db.t1"
        assert result.extra["metadata_type"] == "snapshots"
        assert "__meta_db_t1_snapshots" in result.sql

    def test_manifests(self):
        result = preprocess("SELECT * FROM myns.mytbl.manifests")
        assert result.command_type == CommandType.METADATA_TABLE
        assert result.table_name == "myns.mytbl"
        assert result.extra["metadata_type"] == "manifests"

    def test_history(self):
        result = preprocess("SELECT * FROM t1.history")
        assert result.command_type == CommandType.METADATA_TABLE
        assert result.table_name == "t1"
        assert result.extra["metadata_type"] == "history"

    def test_entries(self):
        result = preprocess("SELECT * FROM ns.tbl.entries")
        assert result.command_type == CommandType.METADATA_TABLE
        assert result.extra["metadata_type"] == "entries"

    def test_files(self):
        result = preprocess("SELECT * FROM ns.tbl.files")
        assert result.command_type == CommandType.METADATA_TABLE
        assert result.extra["metadata_type"] == "files"

    def test_refs(self):
        result = preprocess("SELECT * FROM ns.tbl.refs")
        assert result.command_type == CommandType.METADATA_TABLE
        assert result.extra["metadata_type"] == "refs"

    def test_schemas(self):
        result = preprocess("SELECT * FROM ns.tbl.schemas")
        assert result.command_type == CommandType.METADATA_TABLE
        assert result.extra["metadata_type"] == "schemas"

    def test_partition_specs(self):
        result = preprocess("SELECT * FROM ns.tbl.partition_specs")
        assert result.command_type == CommandType.METADATA_TABLE
        assert result.extra["metadata_type"] == "partition_specs"

    def test_case_insensitive(self):
        result = preprocess("SELECT * FROM db.tbl.SNAPSHOTS")
        assert result.command_type == CommandType.METADATA_TABLE
        assert result.extra["metadata_type"] == "snapshots"

    def test_non_metadata_suffix_passes_through(self):
        result = preprocess("SELECT * FROM db.tbl.regular_col")
        # 'regular_col' is not a metadata keyword, should pass through
        assert result.command_type == CommandType.SQL

    def test_rewritten_sql_uses_temp_name(self):
        result = preprocess("SELECT snapshot_id FROM db.t1.snapshots WHERE snapshot_id > 0")
        assert "db.t1.snapshots" not in result.sql
        assert result.extra["temp_name"] in result.sql


# ---------------------------------------------------------------------------
# DataFrameWriter — path-based saves
# ---------------------------------------------------------------------------


class TestDataFrameWriterPathSaves:
    def test_parquet_save(self, df):
        with tempfile.TemporaryDirectory() as d:
            df.write.parquet(d)
            parquet_files = list(pathlib.Path(d).rglob("*.parquet"))
            assert len(parquet_files) > 0

    def test_csv_save(self, df):
        with tempfile.TemporaryDirectory() as d:
            df.write.csv(d)
            csv_files = list(pathlib.Path(d).rglob("*.csv"))
            assert len(csv_files) > 0

    def test_json_save(self, df):
        with tempfile.TemporaryDirectory() as d:
            df.write.json(d)
            json_files = list(pathlib.Path(d).rglob("*.json"))
            assert len(json_files) > 0

    def test_save_with_format_parquet(self, df):
        with tempfile.TemporaryDirectory() as d:
            df.write.format("parquet").save(d)
            parquet_files = list(pathlib.Path(d).rglob("*.parquet"))
            assert len(parquet_files) > 0

    def test_save_with_format_csv(self, df):
        with tempfile.TemporaryDirectory() as d:
            df.write.format("csv").save(d)
            csv_files = list(pathlib.Path(d).rglob("*.csv"))
            assert len(csv_files) > 0

    def test_save_no_path_raises(self, df):
        with pytest.raises(ValueError, match="path is required"):
            df.write.save()

    def test_save_unsupported_format_raises(self, df):
        with tempfile.TemporaryDirectory() as d:
            with pytest.raises(NotImplementedError, match="Format"):
                df.write.format("avro").save(d)

    def test_parquet_round_trip(self, ctx, sample_arrow):
        """Parquet written by DataFusion is readable by PyArrow."""
        import pyarrow.parquet as pq

        ctx.register_record_batches("src", [sample_arrow.to_batches()])
        src_df = DataFrame(ctx.table("src"), session=None)
        with tempfile.TemporaryDirectory() as d:
            src_df.write.parquet(d)
            parquet_files = list(pathlib.Path(d).rglob("*.parquet"))
            read_back = pq.read_table(parquet_files[0])
            assert set(read_back.column_names) == {"id", "name"}
            assert len(read_back) == 3


# ---------------------------------------------------------------------------
# Spark Catalog API — unit tests with stub catalog
# ---------------------------------------------------------------------------


class _StubNamespace:
    """Minimal namespace tuple wrapper."""
    def __init__(self, parts):
        self._parts = parts

    def __iter__(self):
        return iter(self._parts)


class _StubTable:
    """Minimal stub table."""
    def __init__(self, name):
        self.name = name


class _StubCatalog:
    """Stub PyIceberg catalog for unit-testing IcebergCatalogAPI."""

    def __init__(self):
        self._namespaces = [("default",), ("staging",)]
        self._tables = {
            ("default",): [_StubTable("orders"), _StubTable("items")],
            ("staging",): [_StubTable("temp")],
        }
        self._loaded: dict[str, object] = {}
        self._created_tables: list = []

    def list_namespaces(self):
        return self._namespaces

    def list_tables(self, namespace):
        return self._tables.get(tuple(namespace), [])

    def load_table(self, name: str):
        if name in self._loaded:
            return self._loaded[name]
        parts = name.split(".")
        tbl_name = parts[-1]
        ns = tuple(parts[:-1])
        for t in self._tables.get(ns, []):
            if (t.name if hasattr(t, "name") else t[-1]) == tbl_name:
                return object()
        raise Exception(f"Table not found: {name}")

    def load_namespace_properties(self, ns_tuple):
        if ns_tuple == ("default",):
            return {"comment": "Default namespace", "location": "s3://bucket/default"}
        return {}

    def create_table(self, name, schema=None):
        self._created_tables.append(name)


class _StubSession:
    def __init__(self):
        from unittest.mock import MagicMock
        self._catalog = _StubCatalog()
        self._registered_tables: dict[str, str] = {}
        self._ctx = MagicMock()

    def table(self, name):
        from unittest.mock import MagicMock
        return MagicMock()


@pytest.fixture
def catalog_api():
    return IcebergCatalogAPI(_StubSession())


class TestIcebergCatalogAPI:
    def test_current_database_default(self, catalog_api):
        assert catalog_api.currentDatabase() == "default"

    def test_set_current_database(self, catalog_api):
        catalog_api.setCurrentDatabase("staging")
        assert catalog_api.currentDatabase() == "staging"

    def test_list_databases(self, catalog_api):
        dbs = catalog_api.listDatabases()
        assert "default" in dbs
        assert "staging" in dbs

    def test_database_exists_true(self, catalog_api):
        assert catalog_api.databaseExists("default") is True

    def test_database_exists_false(self, catalog_api):
        assert catalog_api.databaseExists("nonexistent") is False

    def test_list_tables(self, catalog_api):
        tables = catalog_api.listTables("default")
        assert "orders" in tables
        assert "items" in tables

    def test_list_tables_uses_current_db(self, catalog_api):
        catalog_api.setCurrentDatabase("staging")
        tables = catalog_api.listTables()
        assert "temp" in tables

    def test_table_exists_true(self, catalog_api):
        assert catalog_api.tableExists("orders", "default") is True

    def test_table_exists_fully_qualified(self, catalog_api):
        assert catalog_api.tableExists("default.orders") is True

    def test_table_exists_false(self, catalog_api):
        assert catalog_api.tableExists("ghost", "default") is False

    def test_refresh_table_removes_from_registered(self, catalog_api):
        catalog_api._session._registered_tables["orders"] = "default.orders"
        catalog_api.refreshTable("default.orders")
        assert "orders" not in catalog_api._session._registered_tables

    def test_is_cached_always_false(self, catalog_api):
        assert catalog_api.isCached("anything") is False

    def test_function_exists_always_false(self, catalog_api):
        assert catalog_api.functionExists("my_udf") is False

    def test_list_functions_empty(self, catalog_api):
        assert catalog_api.listFunctions() == []

    def test_cache_operations_noop(self, catalog_api):
        # These should not raise
        catalog_api.cacheTable("t")
        catalog_api.uncacheTable("t")
        catalog_api.clearCache()
        catalog_api.refreshByPath("/some/path")


# ---------------------------------------------------------------------------
# Task 10A: Catalog API Completeness
# ---------------------------------------------------------------------------


class TestCatalogAPITask10A:
    """Tests for Task 10A additions: pattern filtering, getDatabase, dropTempView, createTable."""

    def test_list_databases_pattern_match(self, catalog_api):
        result = catalog_api.listDatabases("def*")
        assert "default" in result
        assert "staging" not in result

    def test_list_databases_pattern_no_match(self, catalog_api):
        result = catalog_api.listDatabases("xyz*")
        assert result == []

    def test_list_databases_no_pattern_returns_all(self, catalog_api):
        result = catalog_api.listDatabases()
        assert "default" in result
        assert "staging" in result

    def test_list_tables_pattern_match(self, catalog_api):
        result = catalog_api.listTables("default", pattern="ord*")
        assert "orders" in result
        assert "items" not in result

    def test_list_tables_pattern_no_match(self, catalog_api):
        result = catalog_api.listTables("default", pattern="xyz*")
        assert result == []

    def test_list_tables_no_pattern_returns_all(self, catalog_api):
        result = catalog_api.listTables("default")
        assert "orders" in result
        assert "items" in result

    def test_get_database_known(self, catalog_api):
        info = catalog_api.getDatabase("default")
        assert info["name"] == "default"
        assert info["description"] == "Default namespace"
        assert info["locationUri"] == "s3://bucket/default"

    def test_get_database_unknown_returns_empty_strings(self, catalog_api):
        info = catalog_api.getDatabase("nonexistent")
        assert info["name"] == "nonexistent"
        assert info["description"] == ""
        assert info["locationUri"] == ""

    def test_drop_temp_view_returns_true_on_success(self, catalog_api):
        result = catalog_api.dropTempView("my_view")
        assert result is True

    def test_drop_temp_view_returns_false_when_not_registered(self, catalog_api):
        # ctx.table() raises KeyError for unregistered views — simulate that
        catalog_api._session._ctx.table.side_effect = KeyError("no such table")
        result = catalog_api.dropTempView("missing_view")
        assert result is False

    def test_drop_temp_view_calls_deregister(self, catalog_api):
        """Verify deregister_table is actually called when the view exists."""
        result = catalog_api.dropTempView("my_view")
        assert result is True
        catalog_api._session._ctx.deregister_table.assert_called_once_with("my_view")

    def test_drop_global_temp_view_delegates(self, catalog_api):
        result = catalog_api.dropGlobalTempView("my_view")
        assert result is True

    def test_create_table_raises_without_schema(self, catalog_api):
        with pytest.raises(ValueError, match="schema is required"):
            catalog_api.createTable("default.t1")

    def test_create_table_with_arrow_schema(self, catalog_api):
        import pyarrow as pa
        schema = pa.schema([pa.field("id", pa.int64()), pa.field("name", pa.string())])
        catalog_api.createTable("default.new_table", schema=schema)
        assert "default.new_table" in catalog_api._catalog._created_tables

    def test_create_table_with_struct_type(self, catalog_api):
        from iceberg_spark.types import StructType, StructField, IntegerType, StringType
        schema = StructType([
            StructField("id", IntegerType(), nullable=False),
            StructField("val", StringType(), nullable=True),
        ])
        catalog_api.createTable("default.typed_table", schema=schema)
        assert "default.typed_table" in catalog_api._catalog._created_tables
