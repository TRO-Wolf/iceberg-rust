# Changelog

All notable changes to the `iceberg-spark` project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-03-24

Initial release of `iceberg-spark` -- a pure-Python, zero-JVM drop-in replacement for PySpark's SQL and DataFrame interfaces, powered by DataFusion and PyIceberg.

### SQL

- SELECT with full expression support, JOINs, subqueries, CTEs, and aggregations
- INSERT INTO / INSERT OVERWRITE with VALUES and SELECT sources
- UPDATE with copy-on-write semantics (scan, transform via DataFusion, overwrite)
- DELETE FROM with copy-on-write semantics and subquery support in WHERE
- MERGE INTO with WHEN MATCHED / WHEN NOT MATCHED clauses (copy-on-write)
- CREATE TABLE ... AS SELECT (CTAS) with schema inference from Arrow
- DDL: CREATE/DROP TABLE, CREATE/DROP NAMESPACE, ALTER TABLE (ADD/DROP/RENAME COLUMN), TRUNCATE
- Time travel: `TIMESTAMP AS OF` and `VERSION AS OF` syntax
- Metadata tables: `.snapshots`, `.manifests`, `.history`, `.files`, `.entries`, `.refs`
- SET, USE, SHOW CREATE TABLE, SHOW TBLPROPERTIES, CACHE/UNCACHE TABLE, ADD JAR commands

### DataFrame API

- 130+ PySpark-compatible functions in `iceberg_spark.functions` (string, math, date/time, collection, logical, and more)
- Full Column expression system with operators, aliases, casts, and sorting
- GroupedData with `agg()`, `count()`, `sum()`, `avg()`, `min()`, `max()`, `pivot()`
- Window functions: `row_number`, `rank`, `dense_rank`, `percent_rank`, `cume_dist`, `ntile`, `lag`, `lead`
- NA functions: `df.na.drop()`, `df.na.fill()`, `df.na.replace()`
- Stat functions: `df.stat.corr()`, `df.stat.cov()`, `df.stat.freqItems()`, `df.stat.approxQuantile()`
- `sample()`, `transform()`, `toDF()`, `toJSON()`, `toPandas()`
- `foreach()`, `foreachPartition()`, `randomSplit()`
- `mapInPandas()`, `mapInArrow()` stubs
- Row type with field access by name and index
- `session.range(start, end, step)` for generating range DataFrames
- `session.catalog` API: `listTables`, `listDatabases`, `tableExists`, `setCurrentDatabase`, etc.
- `expr()` for parsing SQL expressions into Column objects

### Write

- DataFrameWriter with modes: `append`, `overwrite`, `error`, `ignore`
- `df.write.saveAsTable("db.table")` and `df.write.insertInto("db.table")`
- `df.write.partitionBy()` with auto-creation of partition specs
- Path-based saves: `.parquet(path)`, `.csv(path)`, `.json(path)`
- DataFrameWriterV2: `df.writeTo("db.table").append()`, `.overwritePartitions()`, `.create()`, `.partitionedBy()`
- Partition-scoped overwrites via WriterV2

### Read

- `session.read.parquet(path)` -- Parquet files
- `session.read.csv(path)` -- CSV files with header/schema inference
- `session.read.json(path)` -- JSON files
- `session.read.text(path)` -- text files (one row per line)
- `session.read.table("db.table")` -- Iceberg tables
- `session.read.orc(path)` -- ORC reader stub

### UDFs

- Scalar Python UDFs via `session.udf.register(name, func, returnType)`
- `@udf` decorator for inline UDF definition
- UDFs usable in both SQL queries and DataFrame expressions

### Types

- Full PySpark type system: `StringType`, `IntegerType`, `LongType`, `FloatType`, `DoubleType`, `BooleanType`, `DateType`, `TimestampType`, `DecimalType`, `BinaryType`, `ByteType`, `ShortType`
- Complex types: `ArrayType`, `MapType`, `StructType`, `StructField`
- `StructType` schema construction and field access

### Partitioning

- Identity partitions: `PARTITIONED BY (col)`
- Bucket partitions: `PARTITIONED BY (bucket(N, col))`
- Time-based partitions: `year(col)`, `month(col)`, `day(col)`, `hour(col)`
- Truncate partitions: `truncate(N, col)`
- Full DDL, DML, and DataFrame API support for all partition types

### PySpark Compatibility Shim

- Separate `iceberg-spark-pyspark` package provides `from pyspark.sql import SparkSession` compatibility
- Drop-in replacement: existing PySpark scripts work with `pip install iceberg-spark-pyspark`
- Routes all `pyspark.sql` imports to `iceberg_spark` equivalents
