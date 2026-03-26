"""Tests for custom exception hierarchy and error handling.

Verifies that:
- Custom exceptions exist and have the correct hierarchy
- All custom exceptions subclass RuntimeError for backward compatibility
- Exception messages contain useful context
"""

from __future__ import annotations

import pytest

from iceberg_spark.catalog_ops import (
    DDLError,
    DMLError,
    IcebergSparkError,
    SchemaError,
    TableNotFoundError,
)


# ---------------------------------------------------------------------------
# Hierarchy tests — every custom exception is a RuntimeError
# ---------------------------------------------------------------------------


class TestExceptionHierarchy:
    def test_base_is_runtime_error(self):
        """IcebergSparkError subclasses RuntimeError for backward compatibility."""
        assert issubclass(IcebergSparkError, RuntimeError)

    def test_table_not_found_is_runtime_error(self):
        assert issubclass(TableNotFoundError, RuntimeError)

    def test_table_not_found_is_iceberg_spark_error(self):
        assert issubclass(TableNotFoundError, IcebergSparkError)

    def test_ddl_error_is_runtime_error(self):
        """DDLError subclasses RuntimeError for backward compatibility."""
        assert issubclass(DDLError, RuntimeError)

    def test_ddl_error_is_iceberg_spark_error(self):
        assert issubclass(DDLError, IcebergSparkError)

    def test_dml_error_is_runtime_error(self):
        """DMLError subclasses RuntimeError for backward compatibility."""
        assert issubclass(DMLError, RuntimeError)

    def test_dml_error_is_iceberg_spark_error(self):
        assert issubclass(DMLError, IcebergSparkError)

    def test_schema_error_is_runtime_error(self):
        assert issubclass(SchemaError, RuntimeError)

    def test_schema_error_is_iceberg_spark_error(self):
        assert issubclass(SchemaError, IcebergSparkError)


# ---------------------------------------------------------------------------
# Catching tests — existing `except RuntimeError` still catches our errors
# ---------------------------------------------------------------------------


class TestBackwardCompatibleCatch:
    def test_catch_ddl_as_runtime_error(self):
        with pytest.raises(RuntimeError):
            raise DDLError("CREATE TABLE failed")

    def test_catch_dml_as_runtime_error(self):
        with pytest.raises(RuntimeError):
            raise DMLError("INSERT INTO failed")

    def test_catch_table_not_found_as_runtime_error(self):
        with pytest.raises(RuntimeError):
            raise TableNotFoundError("Table x.y not found")

    def test_catch_schema_as_runtime_error(self):
        with pytest.raises(RuntimeError):
            raise SchemaError("Type mismatch")

    def test_catch_base_as_runtime_error(self):
        with pytest.raises(RuntimeError):
            raise IcebergSparkError("something went wrong")


# ---------------------------------------------------------------------------
# Message tests — exceptions carry context
# ---------------------------------------------------------------------------


class TestExceptionMessages:
    def test_ddl_error_message(self):
        err = DDLError("CREATE TABLE db.t1: already exists")
        assert "CREATE TABLE" in str(err)
        assert "db.t1" in str(err)

    def test_dml_error_message(self):
        err = DMLError("INSERT INTO db.t1: schema mismatch")
        assert "INSERT INTO" in str(err)

    def test_table_not_found_message(self):
        err = TableNotFoundError("DELETE FROM db.t1: NoSuchTableException")
        assert "db.t1" in str(err)

    def test_chained_exception(self):
        """Custom exceptions preserve the original cause via 'from e'."""
        original = ValueError("underlying issue")
        err = DDLError("CREATE TABLE failed")
        err.__cause__ = original
        assert err.__cause__ is original


# ---------------------------------------------------------------------------
# Import from package top-level
# ---------------------------------------------------------------------------


class TestPackageExports:
    def test_import_from_iceberg_spark(self):
        """All custom exceptions are importable from the top-level package."""
        from iceberg_spark import (
            DDLError,
            DMLError,
            IcebergSparkError,
            SchemaError,
            TableNotFoundError,
        )
        # Verify they are the same classes
        from iceberg_spark.catalog_ops import DDLError as DDLError2
        assert DDLError is DDLError2
