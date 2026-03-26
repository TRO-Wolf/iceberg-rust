"""Tests for Sprint 4: PySpark compatibility shim."""

from __future__ import annotations

import sys
import os

import pytest


# Add the project root to sys.path so the pyspark shim is importable
@pytest.fixture(autouse=True)
def add_pyspark_shim_to_path():
    project_root = os.path.dirname(os.path.dirname(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    yield


class TestPySparkSQLImports:
    def test_spark_session(self):
        from pyspark.sql import SparkSession
        from iceberg_spark.session import IcebergSession
        assert SparkSession is IcebergSession

    def test_dataframe(self):
        from pyspark.sql import DataFrame
        from iceberg_spark.dataframe import DataFrame as IDF
        assert DataFrame is IDF

    def test_column(self):
        from pyspark.sql import Column
        from iceberg_spark.column import Column as IC
        assert Column is IC

    def test_row(self):
        from pyspark.sql import Row
        from iceberg_spark.row import Row as IR
        assert Row is IR

    def test_window(self):
        from pyspark.sql import Window
        from iceberg_spark.window import Window as IW
        assert Window is IW


class TestPySparkFunctionsImports:
    def test_col(self):
        from pyspark.sql.functions import col
        assert callable(col)

    def test_lit(self):
        from pyspark.sql.functions import lit
        assert callable(lit)

    def test_when(self):
        from pyspark.sql.functions import when
        assert callable(when)

    def test_sum(self):
        from pyspark.sql.functions import sum
        assert callable(sum)

    def test_count(self):
        from pyspark.sql.functions import count
        assert callable(count)

    def test_expr(self):
        from pyspark.sql.functions import expr
        assert callable(expr)

    def test_broadcast(self):
        from pyspark.sql.functions import broadcast
        assert callable(broadcast)


class TestPySparkTypesImports:
    def test_struct_type(self):
        from pyspark.sql.types import StructType
        from iceberg_spark.types import StructType as IST
        assert StructType is IST

    def test_integer_type(self):
        from pyspark.sql.types import IntegerType
        from iceberg_spark.types import IntegerType as IIT
        assert IntegerType is IIT

    def test_string_type(self):
        from pyspark.sql.types import StringType
        from iceberg_spark.types import StringType as IST
        assert StringType is IST

    def test_struct_field(self):
        from pyspark.sql.types import StructField
        from iceberg_spark.types import StructField as ISF
        assert StructField is ISF


class TestPySparkWindowImports:
    def test_window(self):
        from pyspark.sql.window import Window
        from iceberg_spark.window import Window as IW
        assert Window is IW

    def test_window_spec(self):
        from pyspark.sql.window import WindowSpec
        from iceberg_spark.window import WindowSpec as IWS
        assert WindowSpec is IWS


class TestPySparkColumnImports:
    def test_column(self):
        from pyspark.sql.column import Column
        from iceberg_spark.column import Column as IC
        assert Column is IC


class TestPySparkRowImports:
    def test_row(self):
        from pyspark.sql.row import Row
        from iceberg_spark.row import Row as IR
        assert Row is IR
