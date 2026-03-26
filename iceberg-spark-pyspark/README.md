# iceberg-spark-pyspark

PySpark compatibility shim for [iceberg-spark](../iceberg-spark-python/).

Allows existing PySpark code to run with iceberg_spark:

```python
from pyspark.sql import SparkSession  # → routes to IcebergSession
```

## Installation

```bash
pip install iceberg-spark-pyspark
# or
pip install iceberg-spark[pyspark-compat]
```

**Cannot coexist with real PySpark.** The shim detects and raises an error if both are installed.
