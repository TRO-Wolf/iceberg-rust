# Iceberg Spark Python (`iceberg_spark`)

`iceberg_spark` is a **pure Python**, zero-JVM drop-in replacement for PySpark's SQL and DataFrame interfaces. It allows you to run `spark.sql()` and PySpark DataFrame commands using **DataFusion** (a blazing-fast Rust query engine) and **PyIceberg** (a native Python Iceberg catalog/table API) instead of a traditional Spark cluster.

**Target Use Case:** Single-node production analytics, local testing, and data pipelines where you want the familiar PySpark API but without the overhead, memory footprint, or operational complexity of a JVM/Spark cluster environment.

**Status:** 904 tests passing. 100% DataFrame API coverage, 171 built-in functions, full UDF/UDAF support, partitioned tables, time travel, MERGE INTO, and PySpark compatibility shim.

---

## Installation

### Prerequisites
- Python 3.10+

*Note: `iceberg_spark` is currently unreleased on PyPI. It requires a custom build of `pyiceberg-core` from this repository's Rust bindings before installation.*

1. **Build the `pyiceberg-core` Rust bindings:**
   ```bash
   cd ../bindings/python
   pip install maturin
   maturin develop --release
   ```

2. **Install `iceberg_spark` locally:**
   ```bash
   cd ../../iceberg-spark-python
   pip install -e .
   ```

**Dependencies:**
- `datafusion >= 50.0.0`
- `pyiceberg[pyarrow] >= 0.10.0`
- `pyiceberg-core >= 0.7.0`
- `pyarrow >= 15.0.0`

**Optional Dependencies:**
- `sqlalchemy >= 2.0` (for SQLite/SQL catalog support)
- `pandas >= 1.5` (for Pandas integration)

---

## Quickstart

### 1. Initialize an Iceberg Session

The `IcebergSession` mirrors `SparkSession`. You configure your catalog during the builder phase.

```python
from iceberg_spark import IcebergSession

# Create a session with a local SQLite catalog and local warehouse
session = IcebergSession.builder() \
    .catalog("sql", uri="sqlite:///catalog.db", warehouse="file:///tmp/wh") \
    .build()
```

### 2. SQL DDL and DML

You can run familiar PySpark SQL commands. Under the hood, DDL/DML operations are intercepted and handled directly by PyIceberg, while SELECT queries are routed to DataFusion.

```python
# Create Database and Table
session.sql("CREATE DATABASE IF NOT EXISTS db")
session.sql("CREATE TABLE db.t1 (id INT, name STRING) USING iceberg")

# Insert Data
session.sql("INSERT INTO db.t1 VALUES (1, 'Alice'), (2, 'Bob')")

# Query Data
session.sql("SELECT * FROM db.t1 WHERE id > 1").show()

# Update and Delete (Copy-on-write supported natively)
session.sql("UPDATE db.t1 SET name = 'Bobby' WHERE id = 2")
session.sql("DELETE FROM db.t1 WHERE id = 1")
```

---

## Core Features

### DataFrame API 

`iceberg_spark` wraps `datafusion.DataFrame` to provide a full PySpark API surface.

```python
from iceberg_spark.functions import col, lit, sum, desc

df = session.sql("SELECT * FROM db.sales")

# PySpark syntax
df.filter(col("amount") > 100) \
  .groupBy("category") \
  .agg(sum("amount").alias("total")) \
  .orderBy(desc("total")) \
  .show()
```

171 PySpark functions are supported in `iceberg_spark.functions`, covering string manipulation, math, date/time, collections, logical operators, UDFs/UDAFs, and more.

### Reading and Writing Files

Read direct from Parquet or CSV easily:
```python
df = session.read.parquet("path/to/data.parquet")
df = session.read.csv("path/to/data.csv")
```

Write DataFrame back into Iceberg tables with Spark-like modes:
```python
# Modes: "append", "overwrite", "error", "ignore"
df.write.mode("overwrite").saveAsTable("db.my_table")
```

### Advanced Analytics & Advanced SQL

- **Window Functions**: Supports `.over(WindowSpec)`, `row_number()`, `rank()`, `lag()`, `lead()`, and frame bounds.
- **NA & Stat Functions**: `df.na.drop()`, `df.na.fill()`, `df.stat.corr()`, etc.
- **Time Travel**: Native Iceberg time travel.
  ```sql
  SELECT * FROM test_table TIMESTAMP AS OF '2024-01-01'
  SELECT * FROM test_table VERSION AS OF 123456789
  ```
- **Metadata Tables**: Query Iceberg metadata exactly like Spark.
  ```sql
  SELECT * FROM db.test_table.snapshots
  SELECT * FROM db.test_table.history
  ```
- **Merge Into**: Powerful copy-on-write `MERGE INTO` natively translated to update/insert data flows.

---

## Architecture Overview

1. **Lazy Table Registration**: Tables are not loaded entirely into memory. Instead, `iceberg_spark` registers PyIceberg tables with DataFusion via an FFI bridge on-the-fly when referenced by SQL.
2. **SQL Preprocessor**: Intercepts DDL/DML (like `MERGE INTO` or `UPDATE`) and manages state changes via PyIceberg, preventing the need for complex internal engine rewrites. 
3. **Copy-on-Write**: All deletes, updates, and merges read the table into Arrow, evaluate standard expressions via DataFusion seamlessly, and overwrite the relevant target data files safely using PyIceberg's transaction boundaries.

---

## Documentation

- **[User Guide](docs/user-guide.md)** — Comprehensive guide covering all features in depth
- **[Migration Guide](docs/migration-guide.md)** — Step-by-step guide for migrating from PySpark
- **[Cheat Sheet](docs/cheat-sheet.md)** — Quick reference for common operations
- **[Examples](docs/examples/)** — Runnable example scripts:
  - [`01_quickstart.py`](docs/examples/01_quickstart.py) — Session setup, basic SQL and DataFrame ops
  - [`02_dataframe_ops.py`](docs/examples/02_dataframe_ops.py) — Filters, joins, groupBy, window functions
  - [`03_dml_operations.py`](docs/examples/03_dml_operations.py) — INSERT, UPDATE, DELETE, MERGE INTO
  - [`04_partitioned_tables.py`](docs/examples/04_partitioned_tables.py) — Partition transforms, partitioned writes
  - [`05_time_travel.py`](docs/examples/05_time_travel.py) — Snapshot queries and metadata tables
- **[Changelog](CHANGELOG.md)** — Version history and release notes

---

## Limitations / Differences from PySpark

- **Single Node Only**: Executes via DataFusion and Arrow on your local machine/container. No cluster distribution.
- **Ecosystem Connectors**: Connects to Iceberg Catalogs (REST, SQL, Hive, Glue, etc. via PyIceberg) and local/cloud files (S3/GCS), not arbitrary Spark data sources (JDBC, Cassandra, etc.).
- **Copy-on-Write Scalability**: DML operations (DELETE, UPDATE, MERGE INTO) read the entire table into memory, transform via DataFusion, and overwrite. This works well for tables up to ~10 GB on typical hardware but does not scale to very large tables.
- **No Streaming**: `readStream` / `writeStream` are stubs. DataFusion does not support continuous streaming execution.
- **No Java UDFs**: Java/Scala UDFs are not supported. Python UDFs are fully supported via `session.udf.register()` and the `@udf` decorator, powered by DataFusion's native UDF engine.

### Supported Features

- **SQL**: SELECT, INSERT INTO/OVERWRITE, DELETE, UPDATE, MERGE INTO, TRUNCATE, CREATE/DROP TABLE, CREATE/DROP NAMESPACE, ALTER TABLE (ADD/DROP/RENAME COLUMN), CTAS, time travel (TIMESTAMP/VERSION AS OF), metadata tables (.snapshots, .manifests, .files, .entries, .history, .refs)
- **Partitioned Tables**: `PARTITIONED BY (col)`, `bucket(N, col)`, `year/month/day/hour(col)`, `truncate(N, col)` — full DDL, DML, and DataFrame API support
- **DataFrame API**: Full PySpark compatibility including `.na`, `.stat`, `.groupBy()`, window functions, `sample()`, `transform()`, `mapInPandas()`, `mapInArrow()`
- **Subqueries in DML**: `DELETE FROM t WHERE id IN (SELECT ...)` works via automatic table registration
- **UDFs**: Scalar Python UDFs via `session.udf.register()` and `@udf` decorator, usable in both SQL and DataFrame expressions
- **Complex Types**: Full support for ARRAY, MAP, and STRUCT types including nested structures, with array/map functions (`array_contains`, `explode`, `map_keys`, `map_values`, etc.)