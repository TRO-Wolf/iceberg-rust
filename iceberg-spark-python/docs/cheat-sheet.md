# iceberg_spark Cheat Sheet

## Session

```python
from iceberg_spark import IcebergSession
session = IcebergSession.builder().catalog("sql", uri="sqlite:///catalog.db", warehouse="file:///tmp/wh").build()
```

## SQL

| Operation | Code |
|-----------|------|
| Create table | `session.sql("CREATE TABLE db.t (id INT, name STRING) USING iceberg")` |
| Insert | `session.sql("INSERT INTO db.t VALUES (1, 'Alice'), (2, 'Bob')")` |
| Select | `session.sql("SELECT * FROM db.t WHERE id > 1").show()` |
| Update | `session.sql("UPDATE db.t SET name = 'Bobby' WHERE id = 2")` |
| Delete | `session.sql("DELETE FROM db.t WHERE id = 1")` |
| Merge | `session.sql("MERGE INTO db.t USING db.src ON t.id = src.id WHEN MATCHED THEN UPDATE SET * WHEN NOT MATCHED THEN INSERT *")` |
| Time travel | `session.sql("SELECT * FROM db.t VERSION AS OF 123456789")` |
| Metadata | `session.sql("SELECT * FROM db.t.snapshots")` |
| CTAS | `session.sql("CREATE TABLE db.t2 AS SELECT * FROM db.t WHERE id > 0")` |
| Partitioned | `session.sql("CREATE TABLE db.p (id INT, ts TIMESTAMP) USING iceberg PARTITIONED BY (day(ts))")` |

## DataFrame

| Operation | Code |
|-----------|------|
| Select | `df.select("id", "name")` or `df.select(col("id"), col("name"))` |
| Filter | `df.filter(col("age") > 30)` or `df.where("age > 30")` |
| Group/Agg | `df.groupBy("dept").agg(sum("salary").alias("total"))` |
| Join | `df.join(df2, df.id == df2.id, "inner")` |
| Order | `df.orderBy(desc("amount"))` |
| Show | `df.show()` or `df.show(20, truncate=False)` |
| Collect | `rows = df.collect()` |
| Distinct | `df.distinct()` or `df.dropDuplicates(["col"])` |
| Limit | `df.limit(10)` |
| Union | `df.union(df2)` or `df.unionByName(df2)` |
| Sample | `df.sample(fraction=0.1)` |
| NA | `df.na.drop()` / `df.na.fill(0)` / `df.na.replace({"old": "new"})` |
| Pivot | `df.groupBy("year").pivot("quarter").sum("sales")` |
| Cube | `df.cube("dept", "year").agg(sum("salary"))` |
| Rollup | `df.rollup("dept", "year").agg(sum("salary"))` |
| Unpivot | `df.unpivot(["id"], ["q1", "q2"], "quarter", "sales")` |

## Functions

```python
from iceberg_spark.functions import col, lit, when, sum, count, avg, concat, upper, lower
```

| Function | Code |
|----------|------|
| col / lit | `col("name")` / `lit(42)` |
| when | `when(col("x") > 0, "pos").otherwise("neg")` |
| sum / count / avg | `sum("amount")` / `count("*")` / `avg("score")` |
| concat | `concat(col("first"), lit(" "), col("last"))` |
| upper / lower | `upper(col("name"))` / `lower(col("name"))` |
| to_json | `to_json(col("struct_col"))` |
| from_json | `from_json(col("json_str"), schema)` |
| coalesce | `coalesce(col("a"), col("b"), lit(0))` |
| date_add | `date_add(col("dt"), 7)` |
| explode | `explode(col("arr"))` |
| array | `array(col("a"), col("b"), col("c"))` |

## Window Functions

```python
from iceberg_spark.functions import row_number, rank, dense_rank, lag, lead
from iceberg_spark import Window

w = Window.partitionBy("dept").orderBy("salary")
df.select("name", row_number().over(w).alias("rn"), rank().over(w).alias("rnk"))
```

## Read

| Format | Code |
|--------|------|
| Parquet | `session.read.parquet("path/to/data.parquet")` |
| JSON | `session.read.json("path/to/data.json")` |
| CSV | `session.read.csv("path/to/data.csv")` |
| Table | `session.table("db.my_table")` |

## Write

| Operation | Code |
|-----------|------|
| Save as table | `df.write.mode("overwrite").saveAsTable("db.t")` |
| Append | `df.write.mode("append").saveAsTable("db.t")` |
| Partitioned | `df.write.partitionBy("year").saveAsTable("db.t")` |
| Parquet file | `df.write.parquet("/tmp/output")` |
| CSV file | `df.write.csv("/tmp/output")` |
| WriterV2 | `df.writeTo("db.t").overwritePartitions()` |

## UDF

```python
from iceberg_spark.functions import udf
from iceberg_spark.types import StringType

@udf(returnType=StringType())
def greet(name):
    return f"Hello, {name}!"

session.udf.register("greet", greet, StringType())
session.sql("SELECT greet(name) FROM db.t").show()
```

## Types

```python
from iceberg_spark.types import (
    IntegerType, LongType, FloatType, DoubleType, StringType,
    BooleanType, DateType, TimestampType, ArrayType, MapType,
    StructType, StructField, DecimalType
)
schema = StructType([StructField("id", IntegerType()), StructField("name", StringType())])
```

## Catalog

```python
session.catalog.listDatabases()
session.catalog.listTables("db")
session.catalog.tableExists("db.t")
```
