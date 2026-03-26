"""PySpark window compatibility shim — re-exports iceberg_spark.window."""

from iceberg_spark.window import Window, WindowSpec

__all__ = ["Window", "WindowSpec"]
