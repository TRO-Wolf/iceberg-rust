"""PySpark SQL compatibility shim — routes to iceberg_spark."""

from iceberg_spark.session import IcebergSession as SparkSession
from iceberg_spark.dataframe import DataFrame, DataFrameNaFunctions, DataFrameStatFunctions
from iceberg_spark.column import Column
from iceberg_spark.row import Row
from iceberg_spark.grouped_data import GroupedData
from iceberg_spark.reader import DataFrameReader
from iceberg_spark.writer import DataFrameWriter
from iceberg_spark.writer_v2 import DataFrameWriterV2
from iceberg_spark.window import Window, WindowSpec
from iceberg_spark.catalog_api import IcebergCatalogAPI as Catalog

__all__ = [
    "SparkSession",
    "DataFrame",
    "Column",
    "Row",
    "GroupedData",
    "DataFrameNaFunctions",
    "DataFrameStatFunctions",
    "DataFrameReader",
    "DataFrameWriter",
    "DataFrameWriterV2",
    "Window",
    "WindowSpec",
    "Catalog",
]
