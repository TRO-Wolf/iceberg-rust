"""PySpark column compatibility shim — re-exports iceberg_spark.column."""

from iceberg_spark.column import Column

__all__ = ["Column"]
