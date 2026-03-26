# Migration Guide: PySpark to iceberg_spark

This guide covers how to migrate existing PySpark code to `iceberg_spark`, a pure Python, zero-JVM drop-in replacement that uses DataFusion for query execution and PyIceberg for Iceberg table management.

---

## Import Changes

### Direct Usage (iceberg_spark)

```python
# PySpark
from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local").getOrCreate()

# iceberg_spark (direct)
from iceberg_spark import IcebergSession
session = IcebergSession.builder() \
    .catalog("sql", uri="sqlite:///catalog.db", warehouse="file:///tmp/wh") \
    .build()
```

### PySpark Shim (Drop-in Replacement)

If your codebase has `from pyspark.sql import SparkSession` everywhere, the included `pyspark` shim package can route those imports to `iceberg_spark` without code changes:

```python
# This still works -- routes to IcebergSession under the hood
from pyspark.sql import SparkSession
```

**Important:** The shim cannot coexist with real PySpark. If `pyspark` is installed via pip, the shim raises an `ImportError` at import time. Uninstall one or the other:

```bash
pip uninstall pyspark   # remove real PySpark, keep the shim
# OR
pip uninstall iceberg-spark-pyspark  # remove the shim, keep real PySpark
```

### Functions

```python
# PySpark
from pyspark.sql.functions import col, lit, sum, when, window

# iceberg_spark
from iceberg_spark.functions import col, lit, sum, when
from iceberg_spark import Window
```

All 130+ functions from PySpark are available in `iceberg_spark.functions`. Some are renamed to avoid shadowing Python builtins, with aliases provided:

| PySpark name | iceberg_spark name | Alias available? |
|---|---|---|
| `ascii` | `ascii_func` | Yes -- `from iceberg_spark import ascii` |
| `chr` | `chr_func` | Yes -- `from iceberg_spark import chr` |
| `hex` | `hex_func` | Yes -- `from iceberg_spark import hex` |
| `repeat` | `repeat_func` | Yes -- `from iceberg_spark import repeat` |
| `struct` | `struct_func` | Yes -- `from iceberg_spark import struct` |

### Types

```python
# PySpark
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

# iceberg_spark
from iceberg_spark.types import StructType, StructField, StringType, IntegerType
# OR via the pyspark shim:
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
```

---

## Session Builder Differences

### PySpark

```python
spark = SparkSession.builder \
    .master("local[*]") \
    .appName("my-app") \
    .config("spark.sql.catalog.my_catalog", "org.apache.iceberg.spark.SparkCatalog") \
    .config("spark.sql.catalog.my_catalog.type", "rest") \
    .config("spark.sql.catalog.my_catalog.uri", "http://localhost:8181") \
    .getOrCreate()
```

### iceberg_spark

```python
session = IcebergSession.builder() \
    .appName("my-app") \
    .catalog("rest", name="my_catalog", uri="http://localhost:8181") \
    .build()
```

Key differences:
- `.catalog(type, name=..., **properties)` replaces multiple `.config()` calls
- `.master()` is accepted but is a no-op (DataFusion is always local)
- `.enableHiveSupport()` is accepted but is a no-op
- `.getOrCreate()` and `.build()` are interchangeable
- Supported catalog types: `sql`, `rest`, `hive`, `glue`, `dynamodb`

---

## What Works Identically

### SQL Commands

All of the following SQL statements work with the same syntax as PySpark:

**DDL:**
- `CREATE DATABASE / NAMESPACE`
- `DROP DATABASE / NAMESPACE`
- `CREATE TABLE ... USING iceberg`
- `CREATE TABLE ... AS SELECT` (CTAS)
- `DROP TABLE`
- `ALTER TABLE ADD/DROP/RENAME COLUMN`
- `TRUNCATE TABLE`
- `SHOW TABLES / DATABASES`
- `DESCRIBE TABLE`
- `SHOW CREATE TABLE`
- `SHOW TBLPROPERTIES`

**DML:**
- `INSERT INTO`
- `INSERT OVERWRITE`
- `UPDATE ... SET ... WHERE`
- `DELETE FROM ... WHERE`
- `MERGE INTO ... USING ... ON ... WHEN MATCHED/NOT MATCHED`

**Queries:**
- `SELECT` with full expression support
- `JOIN` (inner, left, right, full, cross, semi, anti)
- `GROUP BY` / `HAVING` / `ORDER BY`
- Subqueries: `WHERE col IN (SELECT ...)`
- Window functions in SQL
- `EXPLAIN`

**Time Travel:**
```sql
SELECT * FROM db.my_table TIMESTAMP AS OF '2024-01-01T00:00:00'
SELECT * FROM db.my_table VERSION AS OF 12345678901234
```

**Metadata Tables:**
```sql
SELECT * FROM db.my_table.snapshots
SELECT * FROM db.my_table.history
SELECT * FROM db.my_table.manifests
SELECT * FROM db.my_table.files
SELECT * FROM db.my_table.entries
SELECT * FROM db.my_table.refs
```

**Partitioned Tables:**
```sql
CREATE TABLE db.events (ts TIMESTAMP, data STRING) USING iceberg PARTITIONED BY (day(ts))
-- Supported transforms: identity, bucket(N, col), year/month/day/hour(col), truncate(N, col)
```

**Complex Types:**
```sql
-- ARRAY, MAP, and STRUCT columns are fully supported in CREATE TABLE and queries
CREATE TABLE db.records (
    tags ARRAY<STRING>,
    metadata MAP<STRING, STRING>,
    address STRUCT<street:STRING, city:STRING, zip:STRING>
) USING iceberg
```

### DataFrame API

The full PySpark DataFrame API is supported:

```python
# All of these work identically to PySpark
df.select("col1", "col2")
df.filter(col("x") > 10)
df.where("x > 10")
df.withColumn("new_col", col("x") + 1)
df.withColumnRenamed("old", "new")
df.drop("col")
df.distinct()
df.dropDuplicates("col1")
df.groupBy("col").agg(sum("val"), count("*"))
df.orderBy(col("x").desc())
df.sort("x", ascending=False)
df.join(other_df, on="key", how="left")
df.crossJoin(other_df)
df.union(other_df)
df.intersect(other_df)
df.subtract(other_df)
df.limit(10)
df.sample(fraction=0.1)
df.describe()
df.show()
df.collect()
df.toPandas()
df.toArrow()
df.count()
df.first()
df.head(5)
df.tail(5)
df.take(10)
df.isEmpty()
df.columns
df.schema
df.dtypes
df.printSchema()
df.explain()
df.createTempView("v")
df.createOrReplaceTempView("v")
df.toDF("a", "b", "c")
df.transform(lambda df: df.filter(...))
df.selectExpr("col1 + 1 AS col2")
df.toJSON()
df.foreach(func)
df.foreachPartition(func)
df.randomSplit([0.7, 0.3])
df.hint("broadcast")  # no-op
df.cache()  # no-op
df.persist()  # no-op
df.repartition(4)  # no-op
df.coalesce(1)  # no-op

# NA functions
df.na.drop(how="any")
df.na.fill(0)
df.na.fill({"col1": 0, "col2": "unknown"})
df.na.replace({"old_val": "new_val"})

# Stat functions
df.stat.corr("col1", "col2")
df.stat.cov("col1", "col2")
df.stat.crosstab("col1", "col2")
df.stat.freqItems(["col1"])
df.stat.approxQuantile("col1", [0.25, 0.5, 0.75], 0.01)

# Write API
df.write.mode("append").saveAsTable("db.table")
df.write.mode("overwrite").saveAsTable("db.table")
df.write.partitionBy("col").saveAsTable("db.table")
df.write.parquet("/path/to/output")
df.write.csv("/path/to/output")
df.write.json("/path/to/output")

# WriterV2 API
df.writeTo("db.table").append()
df.writeTo("db.table").overwrite(lit(True))
df.writeTo("db.table").create()
df.writeTo("db.table").createOrReplace()
df.writeTo("db.table").partitionedBy("col").create()
df.writeTo("db.table").overwritePartitions()

# Read API
session.read.parquet("/path/to/data.parquet")
session.read.csv("/path/to/data.csv")
session.read.json("/path/to/data.json")
session.read.text("/path/to/file.txt")
session.read.table("db.table")
```

### Window Functions

```python
from iceberg_spark import Window
from iceberg_spark.functions import row_number, rank, dense_rank, lag, lead, sum

w = Window.partitionBy("department").orderBy("salary")

df.withColumn("row_num", row_number().over(w))
df.withColumn("rank", rank().over(w))
df.withColumn("dense_rank", dense_rank().over(w))
df.withColumn("prev_salary", lag("salary", 1).over(w))
df.withColumn("next_salary", lead("salary", 1).over(w))

# Frame specifications
w_rows = Window.partitionBy("dept").orderBy("date").rowsBetween(-2, Window.currentRow)
w_range = Window.partitionBy("dept").orderBy("date").rangeBetween(Window.unboundedPreceding, 0)
```

### Row Type

```python
from iceberg_spark import Row
row = Row(name="Alice", age=30)
row["name"]  # "Alice"
row.name     # "Alice"
row.asDict() # {"name": "Alice", "age": 30}
```

### UDFs

Python UDFs are fully supported via DataFusion's ScalarUDF. They work in both SQL queries and the DataFrame API.

```python
# Register and use a Python UDF
session.udf.register("double_it", lambda x: x * 2, IntegerType())
session.sql("SELECT double_it(id) FROM db.table").show()

# Type-safe registration with inputTypes
session.udf.register("add_ints", lambda a, b: a + b,
                      returnType=IntegerType(),
                      inputTypes=[IntegerType(), IntegerType()])

# UDFs also work with DataFrame API
from iceberg_spark.functions import udf
double_udf = udf(lambda x: x * 2, IntegerType())
df.withColumn("doubled", double_udf(col("id")))
```

### Catalog API

```python
session.catalog.listDatabases()
session.catalog.listTables("db")
session.catalog.tableExists("db.table")
session.catalog.setCurrentDatabase("db")
session.catalog.currentDatabase()
```

### Parameterized SQL

```python
df = session.sql(
    "SELECT * FROM db.users WHERE name = :name AND age > :age",
    args={"name": "Alice", "age": 25}
)
```

---

## Behavioral Differences

These features work but behave slightly differently than PySpark:

### 1. DML is Copy-on-Write

`DELETE`, `UPDATE`, and `MERGE INTO` read the entire table into memory as Arrow, apply transformations via DataFusion, and write the result back via PyIceberg's `overwrite()`. This works well for tables up to ~10 GB but does not scale to very large tables like Spark's distributed execution.

For identity-partitioned tables, overwrites are partition-scoped: only the partitions containing affected rows are rewritten. Unaffected partitions are preserved. This also applies to `INSERT OVERWRITE`, `df.write.mode("overwrite")`, and `df.writeTo().overwritePartitions()`.

### 2. `df.intersect()` Behaves as INTERSECT ALL

DataFusion's `intersect()` preserves duplicates (INTERSECT ALL), unlike PySpark which performs INTERSECT DISTINCT. Use `df.intersect(other).distinct()` if you need PySpark's behavior.

### 3. `IN (SELECT ...)` Inside CASE WHEN

DataFusion does not support `IN (subquery)` inside `CASE WHEN` expressions. This is a DataFusion SQL engine limitation.

### 4. Single-Node Execution Only

There is no cluster distribution. `repartition()`, `coalesce()`, `cache()`, `persist()`, and `hint()` are all no-ops. `spark_partition_id()` always returns 0.

### 5. `mapInPandas` / `mapInArrow` Run on a Single Partition

These functions work but treat the entire DataFrame as one partition. There is no parallel partition processing.

### 6. `sha1()` Uses SHA-224

DataFusion does not have SHA-1. The `sha1()` function returns a SHA-224 hash instead (the closest available algorithm).

---

## What Does Not Work (with Workarounds)

### Structured Streaming

`readStream` and `writeStream` raise `NotImplementedError`. Use batch reads and writes instead:

```python
# Instead of streaming:
# df = session.readStream.format("iceberg").table("db.events")

# Use batch reads:
df = session.sql("SELECT * FROM db.events")
```

### ORC Format

ORC is not supported. Convert to Parquet:

```python
# session.read.orc("path")  # raises NotImplementedError
df = session.read.parquet("path/to/data.parquet")  # use Parquet instead
```

### JDBC / External Data Sources

There are no JDBC, Cassandra, or other Spark connector data sources. Only Iceberg tables and local file formats (Parquet, CSV, JSON, text) are supported.

### Java UDFs

`registerJavaFunction()` and `registerJavaUDAF()` raise `NotImplementedError` because there is no JVM. Python UDFs work fully via `session.udf.register()` -- see the UDF section above.

### SparkContext Operations

`sparkContext` is a stub with basic properties (`appName`, `version`, `master`). Methods like `addPyFile()` and `setCheckpointDir()` are no-ops.

---

## Function Compatibility Table

All functions listed below are available in `iceberg_spark.functions`.

### Aggregation Functions

| Function | Status | Notes |
|---|---|---|
| `count` | Working | |
| `sum` | Working | |
| `avg` / `mean` | Working | |
| `min` | Working | |
| `max` | Working | |
| `first` / `last` | Working | `ignorenulls` parameter accepted but not enforced |
| `count_distinct` | Working | |
| `approx_count_distinct` | Working | |
| `collect_list` | Working | |
| `collect_set` | Working | Returns distinct values |
| `stddev` | Working | Sample standard deviation |
| `variance` | Working | Sample variance |
| `stddev_pop` | Working | Population standard deviation |
| `var_pop` | Working | Population variance |
| `covar_samp` | Working | |
| `covar_pop` | Working | |
| `corr` | Working | Pearson correlation |
| `percentile_approx` | Working | Uses DataFusion `approx_percentile_cont` |
| `any_value` | Working | Returns first value (non-deterministic) |
| `bool_and` / `bool_or` | Working | |

### String Functions

| Function | Status | Notes |
|---|---|---|
| `upper` / `lower` | Working | |
| `trim` / `ltrim` / `rtrim` | Working | |
| `length` | Working | |
| `concat` | Working | |
| `concat_ws` | Working | |
| `substring` | Working | |
| `regexp_replace` | Working | |
| `regexp_extract` | Working | |
| `split` | Working | |
| `reverse` | Working | |
| `lpad` / `rpad` | Working | |
| `translate` | Working | |
| `locate` / `instr` | Working | |
| `initcap` | Working | |
| `ascii` / `ascii_func` | Working | |
| `chr` / `chr_func` | Working | |
| `repeat` / `repeat_func` | Working | |
| `left` / `right` | Working | |
| `format_number` | Working | Returns rounded number cast to string |
| `format_string` | Working | Supports `%s` and `%d` specifiers |

### Math Functions

| Function | Status | Notes |
|---|---|---|
| `abs` | Working | |
| `ceil` / `floor` | Working | |
| `round` | Working | |
| `sqrt` | Working | |
| `log` | Working | Natural logarithm |
| `log2` / `log10` / `log1p` | Working | |
| `exp` | Working | |
| `power` / `pow` | Working | |
| `cbrt` | Working | |
| `sign` / `signum` | Working | |
| `factorial` | Working | |
| `degrees` / `radians` | Working | |
| `pi` / `e` | Working | |
| `pmod` | Working | Positive modulo |
| `hex` / `hex_func` | Working | |

### Trigonometric Functions

| Function | Status | Notes |
|---|---|---|
| `sin` / `cos` / `tan` | Working | |
| `asin` / `acos` / `atan` | Working | |
| `atan2` | Working | |
| `sinh` / `cosh` / `tanh` | Working | |
| `cot` | Working | |

### Hash Functions

| Function | Status | Notes |
|---|---|---|
| `md5` | Working | |
| `sha1` | Working | Uses SHA-224 (DataFusion has no SHA-1) |
| `sha2` | Working | Supports 224, 256, 384, 512 bits |

### Date/Time Functions

| Function | Status | Notes |
|---|---|---|
| `current_date` | Working | |
| `current_timestamp` | Working | |
| `year` / `month` / `dayofmonth` | Working | |
| `hour` / `minute` / `second` | Working | |
| `dayofweek` / `dayofyear` | Working | |
| `weekofyear` / `quarter` | Working | |
| `date_format` | Working | Supports `%Y`, `%m`, `%d`, `%H`, `%M`, `%S` |
| `to_date` | Working | Uses cast to Date32 |
| `to_timestamp` | Working | |
| `trunc` / `date_trunc` | Working | |
| `make_date` | Working | |
| `unix_timestamp` | Working | |
| `from_unixtime` | Working | |
| `date_add` / `date_add_func` | Working | |
| `date_sub` | Working | |
| `add_months` | Working | |
| `months_between` | Working | |
| `last_day` | Working | |
| `next_day` | Working | |

### Conditional Functions

| Function | Status | Notes |
|---|---|---|
| `when` | Working | Chainable with `.when()` and `.otherwise()` |
| `coalesce` | Working | |
| `isnull` / `isnan` | Working | |
| `isnotnull` | Working | |
| `nullif` | Working | |
| `greatest` / `least` | Working | Implemented via CASE WHEN (no native DataFusion support) |
| `ifnull` / `nvl` | Working | Alias for coalesce |
| `nvl2` | Working | |
| `nanvl` | Working | |

### Window Functions

| Function | Status | Notes |
|---|---|---|
| `row_number` | Working | |
| `rank` / `dense_rank` | Working | |
| `percent_rank` | Working | |
| `cume_dist` | Working | |
| `ntile` | Working | |
| `lag` / `lead` | Working | |
| `first_value` / `last_value` | Working | |
| `nth_value` | Working | |

### Collection Functions

| Function | Status | Notes |
|---|---|---|
| `array` | Working | |
| `array_contains` | Working | |
| `array_distinct` | Working | |
| `array_join` | Working | `null_replacement` parameter not yet wired |
| `array_position` | Working | |
| `array_remove` | Working | |
| `array_sort` / `sort_array` | Working | |
| `array_union` | Working | |
| `array_except` | Working | |
| `array_intersect` | Working | |
| `arrays_overlap` | Working | |
| `element_at` | Working | |
| `explode` | Working | Via DataFusion `unnest_columns` |
| `flatten` | Working | |
| `size` | Working | |
| `struct` / `struct_func` | Working | |
| `create_map` | Working | Via SQL `make_map()` fallback |
| `map_keys` / `map_values` | Working | Via SQL fallback |

### JSON Functions

| Function | Status | Notes |
|---|---|---|
| `from_json` | Working | Schema required; supports nested complex types (arrays, maps, structs) |
| `to_json` | Working | Handles structs, arrays, maps, and primitives via deferred UDF pattern |
| `schema_of_json` | Working | Infers schema from JSON string via PyArrow JSON reader |

### Misc Functions

| Function | Status | Notes |
|---|---|---|
| `col` / `column` | Working | |
| `lit` / `typedLit` | Working | |
| `expr` | Working | Parses SQL expression strings |
| `broadcast` | No-op | Returns DataFrame unchanged |
| `monotonically_increasing_id` | Working | Equivalent to `row_number()` |
| `spark_partition_id` | Working | Always returns 0 |
| `input_file_name` | Stub | Always returns empty string |
| `bitwise_not` | Working | Implemented as `0 - col - 1` |
| `udf` | Working | Python UDFs via DataFusion ScalarUDF |

---

## Quick Reference: API Mapping

| PySpark | iceberg_spark |
|---|---|
| `SparkSession.builder` | `IcebergSession.builder()` |
| `.config("spark.sql.catalog.X", ...)` | `.catalog(type, name=..., **props)` |
| `.getOrCreate()` | `.build()` or `.getOrCreate()` |
| `spark.sql(...)` | `session.sql(...)` |
| `spark.table(...)` | `session.table(...)` |
| `spark.createDataFrame(...)` | `session.createDataFrame(...)` |
| `spark.range(...)` | `session.range(...)` |
| `spark.read.parquet(...)` | `session.read.parquet(...)` |
| `spark.catalog.listTables()` | `session.catalog.listTables()` |
| `spark.udf.register(...)` | `session.udf.register(...)` |
| `spark.conf.get(...)` | `session.conf.get(...)` |
| `spark.stop()` | `session.stop()` |
