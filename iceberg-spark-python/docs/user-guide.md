# iceberg-spark-python User Guide

A PySpark-compatible SQL and DataFrame interface for Apache Iceberg tables, powered by DataFusion and PyIceberg. Single-node, no JVM required.

---

## 1. Getting Started

### Prerequisites

- Python 3.10+
- Dependencies: `pyiceberg`, `datafusion>=52`, `pyarrow`

### Installation

```bash
cd iceberg-rust/iceberg-spark-python
uv venv .venv && source .venv/bin/activate
uv pip install -e .
```

### Hello World

```python
from iceberg_spark.session import IcebergSession

session = IcebergSession.builder().catalog("sql", warehouse="/tmp/warehouse").build()
session.sql("CREATE NAMESPACE default")
session.sql("CREATE TABLE default.greetings (msg STRING)")
session.sql("INSERT INTO default.greetings VALUES ('Hello, Iceberg!')")
session.sql("SELECT * FROM default.greetings").show()
```

---

## 2. Session Configuration

### Builder Pattern

```python
session = (
    IcebergSession.builder()
    .appName("my-app")
    .catalog("sql", name="local", warehouse="/tmp/wh", uri="sqlite:////tmp/catalog.db")
    .config("spark.sql.shuffle.partitions", "1")
    .build()
)
```

### Supported Catalogs

| Type | Key Properties |
|------|---------------|
| `sql` (SQLite) | `warehouse`, `uri` (e.g. `sqlite:///path/to/db`) |
| `rest` | `uri` (REST catalog endpoint), `token` or `credential` for auth |
| `hive` | `uri` (Hive Metastore Thrift URI) |
| `glue` | `warehouse` (S3 location), AWS credentials via env or explicit props |
| `dynamodb` | AWS credentials via environment |

Multiple catalogs can be configured in a single session — see the
[Catalog Authentication Guide](catalog-auth-guide.md) for detailed configuration
examples including REST OAuth2, Glue IAM, and S3-compatible storage.

### Multi-Catalog Support

```python
session = (
    IcebergSession.builder()
    .catalog("rest", name="prod", uri="http://rest:8181")
    .catalog("sql", name="local", uri="sqlite:///catalog.db", warehouse="/data")
    .defaultCatalog("prod")
    .build()
)

session.sql("USE CATALOG local")               # Switch catalogs
session.sql("SHOW CATALOGS")                   # List all catalogs
session.catalog.currentCatalog()               # "local"
session.catalog.setCurrentCatalog("prod")      # Switch via API
session.catalog.listCatalogs()                 # ["prod", "local"]
```

### Session Management

```python
session.stop()                          # Deactivate session
IcebergSession.getActiveSession()       # Get current active session
new_session = session.newSession()      # Share catalog, new DataFusion context
session.conf.set("key", "value")        # Runtime config
session.conf.get("key", "default")
```

---

## 3. SQL Operations

### DDL

```python
session.sql("CREATE NAMESPACE analytics")
session.sql("CREATE TABLE analytics.events (id BIGINT, name STRING, ts TIMESTAMP)")
session.sql("ALTER TABLE analytics.events ADD COLUMN score DOUBLE")
session.sql("ALTER TABLE analytics.events DROP COLUMN score")
session.sql("ALTER TABLE analytics.events RENAME COLUMN name TO event_name")
session.sql("DESCRIBE TABLE analytics.events").show()
session.sql("SHOW TABLES IN analytics").show()
session.sql("SHOW DATABASES").show()
session.sql("TRUNCATE TABLE analytics.events")
session.sql("DROP TABLE analytics.events")
session.sql("DROP NAMESPACE analytics")
```

Additional DDL: `ALTER TABLE SET TBLPROPERTIES`, `ALTER TABLE UNSET TBLPROPERTIES`, `SHOW CREATE TABLE`, `SHOW TBLPROPERTIES`, `SET key=value`, `USE database`, `CACHE TABLE`, `UNCACHE TABLE`, `ADD JAR` (no-op).

### DML

```python
session.sql("INSERT INTO db.t1 VALUES (1, 'a'), (2, 'b')")
session.sql("INSERT OVERWRITE db.t1 SELECT * FROM db.t2")
session.sql("UPDATE db.t1 SET name = 'updated' WHERE id = 1")
session.sql("DELETE FROM db.t1 WHERE id = 2")
session.sql("""
    MERGE INTO db.target t USING db.source s ON t.id = s.id
    WHEN MATCHED THEN UPDATE SET t.name = s.name
    WHEN NOT MATCHED THEN INSERT (id, name) VALUES (s.id, s.name)
""")
```

DML uses copy-on-write: scan, transform via DataFusion, overwrite via PyIceberg.

### CTAS

```python
session.sql("CREATE TABLE db.summary AS SELECT category, count(*) as cnt FROM db.events GROUP BY category")
```

### Time Travel

```python
session.sql("SELECT * FROM db.events VERSION AS OF 123456789").show()
session.sql("SELECT * FROM db.events TIMESTAMP AS OF '2026-01-15T10:00:00'").show()
```

### Metadata Tables

```python
session.sql("SELECT * FROM db.events.snapshots").show()
session.sql("SELECT * FROM db.events.history").show()
session.sql("SELECT * FROM db.events.files").show()
session.sql("SELECT * FROM db.events.manifests").show()
session.sql("SELECT * FROM db.events.entries").show()
session.sql("SELECT * FROM db.events.refs").show()
```

### Partitioned Tables

```python
session.sql("""
    CREATE TABLE db.logs (
        ts TIMESTAMP, level STRING, msg STRING
    ) PARTITIONED BY (day(ts), identity(level))
""")
```

Supported transforms: `identity`, `bucket(N, col)`, `year`, `month`, `day`, `hour`, `truncate(N, col)`.

### Parameterized SQL

```python
session.sql("SELECT * FROM db.t1 WHERE id = :id AND name = :name", args={"id": 42, "name": "Alice"}).show()
```

---

## 4. DataFrame API

### Creating DataFrames

```python
df = session.table("db.events")
df = session.sql("SELECT * FROM db.events WHERE id > 10")
df = session.createDataFrame([(1, "a"), (2, "b")], ["id", "name"])
df = session.createDataFrame([{"id": 1, "name": "a"}])
df = session.range(0, 100, step=2)       # Single 'id' column
```

### Selection & Projection

```python
df.select("id", "name")
df.select(col("id"), col("name").alias("n"))
df.selectExpr("id", "upper(name) as NAME")
df.drop("name")
df.withColumn("doubled", col("id") * 2)
df.withColumnRenamed("id", "event_id")
df.withColumns({"a": col("id") + 1, "b": col("id") * 2})
df.withColumnsRenamed({"id": "event_id", "name": "event_name"})
df.toDF("col1", "col2")                  # Rename all columns positionally
df.colRegex("`id|name`")                 # Select by regex
```

### Filtering

```python
df.filter(col("id") > 10)
df.where("id > 10 AND name IS NOT NULL")
```

### Sorting

```python
df.sort("id")
df.orderBy(col("id").desc(), col("name").asc_nulls_last())
df.sortWithinPartitions("id")            # Same as sort() in single-node
```

### Grouping & Aggregation

```python
from iceberg_spark.functions import count, sum, avg, max

df.groupBy("dept").agg(count("*").alias("n"), avg("salary").alias("avg_sal"))
df.groupBy("dept").count()
df.groupBy("dept").sum("salary")
df.groupBy("dept").agg({"salary": "avg", "age": "max"})  # Dict form
df.agg(count("*"), max("salary"))         # Aggregate without grouping
```

### Pivot / Cube / Rollup

```python
df.groupBy("dept").pivot("quarter", [1, 2, 3, 4]).agg(sum("revenue"))
df.cube("dept", "quarter").agg(sum("revenue"))     # All grouping combos + grand total
df.rollup("dept", "quarter").agg(sum("revenue"))   # Prefix groupings + grand total
```

### Unpivot / Melt

```python
df.unpivot(ids=["id"], values=["q1", "q2", "q3"], variableColumnName="quarter", valueColumnName="revenue")
df.melt(ids=["id"], values=["q1", "q2"])  # Alias for unpivot
```

### Joins

```python
df1.join(df2, on="id", how="inner")
df1.join(df2, on=["id", "name"], how="left")
df1.join(df2, on=col("df1.id") == col("df2.id"), how="full")
df1.crossJoin(df2)
```

Join types: `inner`, `left`, `right`, `full`, `cross`, `semi`, `anti` (plus aliases like `left_outer`, `leftsemi`).

### Set Operations

```python
df1.union(df2)                  # UNION ALL
df1.unionByName(df2)
df1.intersect(df2)
df1.intersectAll(df2)
df1.subtract(df2)               # EXCEPT
df1.exceptAll(df2)
```

### Null Handling

```python
df.na.drop(how="any")                    # Drop rows with any null
df.na.drop(how="all", subset=["a","b"])  # Drop if all specified cols null
df.na.drop(thresh=3)                     # Keep rows with >= 3 non-nulls
df.na.fill(0)                            # Fill all nulls with 0
df.na.fill({"age": 0, "name": "?"})      # Column-specific fills
df.na.replace({"old": "new"})            # Replace specific values
df.dropna()                              # Shortcut for na.drop()
df.fillna(0)                             # Shortcut for na.fill()
```

### Statistics

```python
df.describe("salary", "age").show()       # count, mean, stddev, min, max
df.summary().show()
df.stat.corr("salary", "age")            # Pearson correlation (returns float)
df.stat.cov("salary", "age")             # Sample covariance (returns float)
df.stat.crosstab("dept", "level").show()
df.stat.freqItems(["dept"]).show()
df.stat.approxQuantile("salary", [0.25, 0.5, 0.75], 0.0)
```

### Sampling

```python
df.sample(fraction=0.1, seed=42)
train, test = df.randomSplit([0.8, 0.2], seed=42)
```

### Deduplication

```python
df.distinct()
df.dropDuplicates(["id"])                 # Dedup by specific columns
```

### Schema Inspection

```python
df.columns                    # ['id', 'name', ...]
df.dtypes                     # [('id', 'bigint'), ('name', 'string')]
df.schema                     # StructType([StructField('id', LongType(), True), ...])
df.printSchema()              # Tree format output
```

### Actions (Triggers Execution)

```python
df.collect()                  # List of Row objects
df.show(20, truncate=True)    # Print formatted table
df.count()                    # int
df.first()                    # Single Row or None
df.head(5)                    # List of Rows (or single Row if n=1)
df.tail(5)                    # Last n rows
df.take(10)                   # Same as head() but always returns list
df.toPandas()                 # pandas.DataFrame
df.toArrow()                  # pyarrow.Table
df.toJSON().show()            # DataFrame with 'value' column of JSON strings
df.isEmpty()                  # bool
df.foreach(lambda row: print(row))
df.toLocalIterator()          # Iterator over Rows
```

### Temp Views

```python
df.createTempView("my_view")
df.createOrReplaceTempView("my_view")
df.createGlobalTempView("gv")             # Same as createTempView in single-node
session.sql("SELECT * FROM my_view").show()
```

### Other

```python
df.transform(lambda d: d.filter(col("id") > 0))
df.explain()                              # Print physical plan
df.cache()                                # No-op
df.repartition(4)                         # No-op
df.mapInPandas(func, schema)              # Apply pandas UDF
df.mapInArrow(func, schema)               # Apply Arrow UDF
```

---

## 5. Column Operations

### Creating Columns

```python
from iceberg_spark.functions import col, lit, expr

col("name")                   # Column reference
lit(42)                       # Literal value
expr("a + b")                 # SQL expression
```

### Operators

```python
col("a") + col("b")          # Arithmetic: +, -, *, /, %
col("a") == 1                # Comparison: ==, !=, <, <=, >, >=
(col("a") > 0) & (col("b") < 10)  # Boolean: &, |, ~
-col("a")                    # Negation
```

### Methods

```python
col("name").alias("n")
col("name").cast("int")               # Or cast(IntegerType())
col("name").cast(IntegerType())
col("id").asc()                        # Sort ascending
col("id").desc_nulls_last()            # asc_nulls_first, desc_nulls_first, asc_nulls_last
col("id").isin(1, 2, 3)
col("id").between(10, 20)
col("name").like("%alice%")
col("name").startswith("A")
col("name").endswith("z")
col("name").contains("ice")
col("name").rlike(r"^\d+$")
col("name").substr(1, 3)
col("val").isNull()
col("val").isNotNull()
col("val").isNaN()
col("a").eqNullSafe(col("b"))
col("a").bitwiseAND(col("b"))          # bitwiseOR, bitwiseXOR
col("val").over(window_spec)           # Apply window function
when(col("a") > 0, "pos").when(col("a") == 0, "zero").otherwise("neg")
```

---

## 6. Functions Reference

All functions are in `iceberg_spark.functions`. Import individually or as:
```python
from iceberg_spark import functions as F
```

### String Functions

`upper`, `lower`, `trim`, `ltrim`, `rtrim`, `length`, `concat`, `concat_ws`, `substring`, `regexp_replace`, `regexp_extract`, `split`, `reverse`, `lpad`, `rpad`, `initcap`, `translate`, `repeat`, `ascii` (as `ascii_func`), `chr` (as `chr_func`), `left`, `right`, `locate`, `instr`, `format_string`

```python
F.upper(col("name"))
F.concat_ws("-", col("a"), col("b"))
F.substring(col("name"), 1, 3)
F.regexp_replace(col("name"), r"\d+", "X")
F.regexp_extract(col("name"), r"(\d+)", 1)
```

### Math Functions

`abs`, `ceil`, `floor`, `round`, `sqrt`, `cbrt`, `log`, `log2`, `log10`, `log1p`, `exp`, `power`/`pow`, `signum`/`sign`, `factorial`, `degrees`, `radians`, `pi`, `e`, `pmod`, `hex` (as `hex_func`)

```python
F.round(col("price"), 2)
F.power(col("base"), 3)
```

### Trigonometric Functions

`sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`, `sinh`, `cosh`, `tanh`, `cot`

### Date/Time Functions

`current_date`, `current_timestamp`, `year`, `month`, `dayofmonth`, `dayofweek`, `dayofyear`, `hour`, `minute`, `second`, `weekofyear`, `quarter`, `date_format`, `to_date`, `to_timestamp`, `trunc`, `date_trunc`, `make_date`, `unix_timestamp`, `from_unixtime`, `date_add` (as `date_add_func`), `date_sub`, `add_months`, `months_between`, `last_day`, `next_day`

```python
F.year(col("ts"))
F.date_format(col("ts"), "%Y-%m-%d")
F.date_add_func(col("dt"), 7)
F.add_months(col("dt"), 3)
```

### Aggregate Functions

`count`, `sum`, `avg`/`mean`, `min`, `max`, `first`, `last`, `collect_list`, `collect_set`, `count_distinct`, `approx_count_distinct`, `stddev`, `stddev_pop`, `variance`, `var_pop`, `corr`, `covar_samp`, `covar_pop`, `percentile_approx`, `any_value`, `bool_and`, `bool_or`

```python
df.groupBy("dept").agg(F.count("*"), F.avg("salary"), F.collect_list("name"))
```

### Window Functions

`row_number`, `rank`, `dense_rank`, `percent_rank`, `cume_dist`, `ntile`, `lag`, `lead`, `nth_value`, `first_value`, `last_value`

```python
from iceberg_spark.window import Window
w = Window.partitionBy("dept").orderBy(col("salary").desc())
df.select("name", F.rank().over(w).alias("rank"))
F.lag(col("value"), 1, 0).over(w)
```

### Conditional Functions

`when`/`otherwise`, `coalesce`, `nullif`, `ifnull`, `nvl`, `nvl2`, `greatest`, `least`, `isnull`, `isnan`, `isnotnull`, `nanvl`

```python
F.when(col("a") > 0, "positive").otherwise("non-positive")
F.coalesce(col("a"), col("b"), F.lit(0))
F.greatest(col("a"), col("b"), col("c"))
```

### Collection Functions

`array`, `array_contains`, `array_distinct`, `array_join`, `array_position`, `array_remove`, `array_sort`, `array_union`, `array_except`, `array_intersect`, `arrays_overlap`, `element_at`, `explode`, `flatten`, `size`, `sort_array`, `struct` (as `struct_func`), `map_keys`, `map_values`, `create_map`

```python
F.array(col("a"), col("b"), col("c"))
F.array_contains(col("tags"), "urgent")
F.explode(col("items"))                # Creates one row per array element
```

### JSON Functions

`from_json`, `to_json`, `schema_of_json`

```python
F.from_json(col("json_str"), "struct<name:string,age:bigint>")
F.to_json(F.struct_func(col("name"), col("age")))
F.schema_of_json('{"name": "Alice", "age": 30}')
```

### Hash Functions

`md5`, `sha1` (uses SHA-224), `sha2`

```python
F.md5(col("data"))
F.sha2(col("data"), 256)
```

### Other Functions

`expr`, `lit`, `col`, `typedLit`, `broadcast` (no-op), `spark_partition_id` (always 0), `monotonically_increasing_id`, `input_file_name` (stub), `bitwise_not`, `format_number`

---

## 7. Window Functions

```python
from iceberg_spark.window import Window, WindowSpec

# Define window specs
w = Window.partitionBy("dept").orderBy("salary")
w_rows = Window.partitionBy("dept").orderBy("salary").rowsBetween(-1, 1)
w_range = Window.orderBy("id").rangeBetween(Window.unboundedPreceding, Window.currentRow)

# Apply
df.withColumn("rank", F.rank().over(w))
df.withColumn("running_avg", F.avg("salary").over(w_range))
df.withColumn("prev_salary", F.lag("salary", 1).over(w))
```

Frame boundaries: `Window.unboundedPreceding`, `Window.unboundedFollowing`, `Window.currentRow`, or integer offsets.

---

## 8. Writing Data

### DataFrameWriter (v1)

```python
df.write.mode("append").saveAsTable("db.events")
df.write.mode("overwrite").saveAsTable("db.events")
df.write.mode("error").saveAsTable("db.events")      # Raise if data exists
df.write.mode("ignore").saveAsTable("db.events")      # Skip if data exists
df.write.partitionBy("year", "month").saveAsTable("db.events")
df.write.insertInto("db.events")

# Path-based writes (non-Iceberg)
df.write.parquet("/tmp/output.parquet")
df.write.csv("/tmp/output.csv")
df.write.json("/tmp/output.json")
df.write.text("/tmp/output.txt")                       # Single-column only
```

### DataFrameWriterV2

```python
df.writeTo("db.events").append()
df.writeTo("db.events").overwrite(lit(True))
df.writeTo("db.events").overwritePartitions()          # Partition-scoped overwrite
df.writeTo("db.events").using("iceberg").partitionedBy("year").create()
df.writeTo("db.events").replace()
df.writeTo("db.events").createOrReplace()
```

---

## 9. Reading Data

```python
df = session.read.table("db.events")
df = session.read.parquet("/path/to/data.parquet")
df = session.read.csv("/path/to/data.csv")
df = session.read.json("/path/to/data.json")
df = session.read.text("/path/to/data.txt")            # Returns 'value' column
df = session.read.format("parquet").load("/path/to/data")
df = session.read.format("csv").option("header", "true").load("/path/to/data.csv")
```

ORC is not supported. Use Parquet instead.

---

## 10. User-Defined Functions

### Scalar UDF

```python
from iceberg_spark.functions import udf
from iceberg_spark.types import StringType, LongType

# Register for SQL + DataFrame use
double_udf = session.udf.register("double_val", lambda x: x * 2, returnType=LongType(), inputTypes=[LongType()])
session.sql("SELECT double_val(id) FROM db.events").show()
df.select(double_udf(col("id"))).show()

# Decorator form (DataFrame only, not registered for SQL)
@udf(returnType=StringType())
def my_upper(s):
    return s.upper() if s else None
df.select(my_upper(col("name"))).show()
```

### Batch Mode (Vectorized)

```python
@udf(returnType=LongType(), inputTypes=[LongType()], batch_mode=True)
def fast_double(arr):
    import pyarrow.compute as pc
    return pc.multiply(arr, 2)
```

### Aggregate UDF (UDAF)

```python
from iceberg_spark.functions import udaf

# Simple callable form
my_sum = session.udf.register_udaf("my_sum", lambda values: sum(values),
                                    returnType=LongType(), inputTypes=[LongType()])

# Decorator form
@udaf(returnType=LongType(), inputTypes=[LongType()])
def custom_sum(values):
    return sum(values)

# Accumulator class form (for complex state)
from datafusion.user_defined import Accumulator
class MyAccumulator(Accumulator):
    def __init__(self):
        self._sum = 0
    def update(self, *arrays):
        for arr in arrays:
            for v in arr:
                if v.is_valid:
                    self._sum += v.as_py()
    def merge(self, states):
        for s in states[0]:
            self._sum += s.as_py()
    def state(self):
        import pyarrow as pa
        return [pa.scalar(self._sum, type=pa.int64())]
    def evaluate(self):
        import pyarrow as pa
        return pa.scalar(self._sum, type=pa.int64())

session.udf.register_udaf("my_acc_sum", MyAccumulator, returnType=LongType(), inputTypes=[LongType()])
```

---

## 11. Catalog API

```python
cat = session.catalog

cat.listDatabases()                     # ['default', 'analytics']
cat.listDatabases("ana*")               # Glob filtering
cat.databaseExists("analytics")         # bool
cat.getDatabase("analytics")            # {'name': ..., 'description': ..., 'locationUri': ...}
cat.currentDatabase()                   # 'default'
cat.setCurrentDatabase("analytics")

cat.listTables("analytics")             # ['events', 'users']
cat.listTables(pattern="ev*")
cat.tableExists("events", "analytics")  # bool
cat.getTable("analytics.events")        # dict with name, database, tableType, ...
cat.listColumns("analytics.events")     # [{'name': 'id', 'dataType': 'long', ...}]

cat.listFunctions()                     # Registered UDFs
cat.functionExists("my_udf")           # bool

cat.dropTempView("my_view")            # Returns True if existed
cat.refreshTable("db.events")          # Force re-load from catalog

cat.createTable("db.new_table", schema=StructType([StructField("id", LongType())]))
```

---

## 12. Type System

```python
from iceberg_spark.types import *

# Simple types
IntegerType()        # int (32-bit)
LongType()           # bigint (64-bit)
FloatType()          # float (32-bit)
DoubleType()         # double (64-bit)
StringType()         # string (UTF-8)
BooleanType()        # boolean
DateType()           # date
TimestampType()      # timestamp with timezone
TimestampNTZType()   # timestamp without timezone
BinaryType()         # binary
DecimalType(10, 2)   # decimal(precision, scale)
ByteType()           # tinyint
ShortType()          # smallint
NullType()           # null

# Complex types
ArrayType(StringType())
MapType(StringType(), IntegerType())
StructType([StructField("name", StringType()), StructField("age", IntegerType())])

# Chained construction
schema = StructType().add("name", StringType()).add("age", IntegerType())

# Conversions
schema.to_arrow()               # -> pyarrow.Schema
from_arrow_schema(pa_schema)    # -> StructType
from_arrow_type(pa.int64())     # -> LongType()
```

---

## 13. Error Handling

All custom exceptions subclass `RuntimeError` for backward compatibility.

```python
from iceberg_spark.catalog_ops import (
    IcebergSparkError,       # Base exception
    TableNotFoundError,      # Table does not exist
    DDLError,                # DDL operation failed
    DMLError,                # DML operation failed
    SchemaError,             # Schema mismatch or invalid schema
)

try:
    session.table("nonexistent")
except TableNotFoundError as e:
    print(f"Table not found: {e}")
```

---

## 14. PySpark Compatibility Shim

For migrating existing PySpark code with minimal changes:

```python
from pyspark.sql import SparkSession  # Resolves to IcebergSession via shim
```

The `pyspark/` directory provides import-compatible modules that redirect to `iceberg_spark`.

---

## 15. Limitations & Differences from PySpark

| Area | Limitation |
|------|-----------|
| **Execution** | Single-node only (DataFusion, no Spark cluster) |
| **DML** | Copy-on-write (scan + filter + overwrite). Partition-scoped for identity partitions. |
| **Streaming** | Not supported (`readStream`/`writeStream` raise `NotImplementedError`) |
| **Java UDFs** | Not supported (no JVM) |
| **ORC** | Not supported (read or write) |
| **Caching** | `cache()`/`persist()` are no-ops |
| **Partitioning** | `repartition()`/`coalesce()` are no-ops |
| **Bucketing** | `bucketBy()` is a no-op |
| **Subqueries** | `IN (SELECT ...)` inside `CASE WHEN` not supported by DataFusion |
| **intersect()** | Behaves as INTERSECT ALL (preserves duplicates) |
| **SHA-1** | `sha1()` uses SHA-224 (DataFusion lacks SHA-1) |
| **input_file_name()** | Returns empty string (stub) |
| **monotonically_increasing_id()** | Returns `row_number()` -- must use with `.over()` |
