"""PySpark compatibility shim — routes to iceberg_spark.

This package provides `from pyspark.sql import SparkSession` compatibility.
It CANNOT coexist with the real PySpark package.
"""
import importlib.metadata

# Check for real PySpark via pip metadata
try:
    importlib.metadata.version("pyspark")
    raise ImportError(
        "Real PySpark is installed (detected via pip metadata). "
        "The iceberg_spark PySpark shim cannot coexist with real PySpark. "
        "Uninstall one: pip uninstall pyspark OR pip uninstall iceberg-spark-pyspark"
    )
except importlib.metadata.PackageNotFoundError:
    pass

# Check for real PySpark via py4j (runtime check)
try:
    import py4j  # noqa: F401
    raise ImportError(
        "Real PySpark (py4j) is installed. "
        "The iceberg_spark PySpark shim cannot coexist. "
        "Uninstall one: pip uninstall pyspark OR pip uninstall iceberg-spark-pyspark"
    )
except ModuleNotFoundError:
    pass
