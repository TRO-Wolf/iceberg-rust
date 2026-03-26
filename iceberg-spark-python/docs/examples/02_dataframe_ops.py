#!/usr/bin/env python3
"""DataFrame operations example for iceberg_spark.

Demonstrates filter, groupBy, agg, join, window functions,
orderBy, withColumn, and sample.
"""

import os
import shutil
import tempfile

from iceberg_spark import IcebergSession, Window
from iceberg_spark.functions import (
    avg,
    col,
    count,
    dense_rank,
    lag,
    lead,
    lit,
    row_number,
    sum,
    when,
)

# -- Setup --
tmpdir = tempfile.mkdtemp(prefix="iceberg_df_ops_")
db_path = os.path.join(tmpdir, "catalog.db")
warehouse = os.path.join(tmpdir, "warehouse")

try:
    session = IcebergSession.builder() \
        .catalog("sql", name="demo", uri=f"sqlite:///{db_path}",
                 warehouse=f"file://{warehouse}") \
        .build()

    session.sql("CREATE NAMESPACE IF NOT EXISTS demo_db")

    # -- Create and populate a sales table --
    session.sql("""
        CREATE TABLE demo_db.sales (
            id INT,
            product STRING,
            region STRING,
            amount DOUBLE,
            quantity INT
        ) USING iceberg
    """)
    session.sql("""
        INSERT INTO demo_db.sales VALUES
        (1, 'Widget', 'East', 120.0, 3),
        (2, 'Gadget', 'West', 250.0, 1),
        (3, 'Widget', 'East', 80.0, 2),
        (4, 'Gadget', 'East', 250.0, 2),
        (5, 'Widget', 'West', 150.0, 4),
        (6, 'Doohickey', 'East', 300.0, 1),
        (7, 'Gadget', 'West', 250.0, 3),
        (8, 'Widget', 'East', 90.0, 1),
        (9, 'Doohickey', 'West', 350.0, 2),
        (10, 'Widget', 'West', 110.0, 2)
    """)

    df = session.sql("SELECT * FROM demo_db.sales")

    # -- 1. Filter --
    print("=== Filter: East region sales ===")
    df.filter(col("region") == "East").show()

    # -- 2. withColumn: add a total_value computed column --
    print("=== withColumn: total_value = amount * quantity ===")
    df_with_total = df.withColumn("total_value", col("amount") * col("quantity"))
    df_with_total.show()

    # -- 3. GroupBy + Agg --
    print("=== GroupBy product: count, total amount, avg amount ===")
    df.groupBy("product") \
      .agg(
          count("*").alias("num_sales"),
          sum("amount").alias("total_amount"),
          avg("amount").alias("avg_amount"),
      ) \
      .orderBy("product") \
      .show()

    # -- 4. GroupBy on multiple columns --
    print("=== GroupBy product + region ===")
    df.groupBy("product", "region") \
      .agg(sum("amount").alias("total_amount")) \
      .orderBy("product", "region") \
      .show()

    # -- 5. Conditional column with when/otherwise --
    print("=== Conditional: label sales as High/Low ===")
    df.withColumn(
        "tier",
        when(col("amount") >= 200, lit("High"))
        .otherwise(lit("Low"))
    ).select("product", "amount", "tier").show()

    # -- 6. Join: create a products table and join --
    session.sql("""
        CREATE TABLE demo_db.products (
            product_name STRING,
            category STRING
        ) USING iceberg
    """)
    session.sql("""
        INSERT INTO demo_db.products VALUES
        ('Widget', 'Hardware'),
        ('Gadget', 'Electronics'),
        ('Doohickey', 'Accessories')
    """)

    products_df = session.sql("SELECT * FROM demo_db.products")

    print("=== Inner join: sales with product category (via SQL) ===")
    session.sql("""
        SELECT s.id, s.product, p.category, s.amount
        FROM demo_db.sales s
        JOIN demo_db.products p ON s.product = p.product_name
        ORDER BY s.id
    """).show()

    # -- 7. Window functions --
    print("=== Window: row_number and dense_rank by product, ordered by amount desc ===")
    w = Window.partitionBy("product").orderBy(col("amount").desc())

    df.withColumn("row_num", row_number().over(w)) \
      .withColumn("d_rank", dense_rank().over(w)) \
      .select("product", "amount", "row_num", "d_rank") \
      .orderBy("product", col("amount").desc()) \
      .show()

    # -- 8. Lag / Lead --
    print("=== Window: lag and lead on amount ordered by id ===")
    w_ordered = Window.orderBy("id")

    df.withColumn("prev_amount", lag("amount", 1).over(w_ordered)) \
      .withColumn("next_amount", lead("amount", 1).over(w_ordered)) \
      .select("id", "product", "amount", "prev_amount", "next_amount") \
      .show()

    # -- 9. OrderBy descending --
    print("=== OrderBy amount descending ===")
    df.orderBy(col("amount").desc()).show()

    # -- 10. Sample --
    print("=== Sample ~50% of rows (approximate) ===")
    sampled = df.sample(fraction=0.5, seed=42)
    print(f"Sampled {sampled.count()} rows out of {df.count()}")
    sampled.show()

    # -- 11. Describe --
    print("=== Describe: basic statistics ===")
    df.describe("amount", "quantity").show()

    # -- 12. NA functions --
    print("=== NA functions: fill nulls ===")
    data_with_nulls = [
        (1, "A", 10.0),
        (2, "B", None),
        (3, None, 30.0),
        (4, None, None),
    ]
    df_nulls = session.createDataFrame(data_with_nulls, ["id", "name", "value"])
    df_nulls.show()

    print("After na.fill({'name': 'Unknown', 'value': 0.0}):")
    df_nulls.na.fill({"name": "Unknown", "value": 0.0}).show()

    print("After na.drop(how='any'):")
    df_nulls.na.drop(how="any").show()

    # -- Cleanup --
    session.sql("DROP TABLE demo_db.sales")
    session.sql("DROP TABLE demo_db.products")
    session.sql("DROP NAMESPACE demo_db")
    session.stop()
    print("=== Done ===")

finally:
    shutil.rmtree(tmpdir, ignore_errors=True)
