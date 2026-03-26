#!/usr/bin/env python3
"""Partitioned tables example for iceberg_spark.

Demonstrates creating partitioned tables, inserting data,
querying by partition, and INSERT OVERWRITE.
"""

import os
import shutil
import tempfile

from iceberg_spark import IcebergSession
from iceberg_spark.functions import col, count, sum

# -- Setup --
tmpdir = tempfile.mkdtemp(prefix="iceberg_partitioned_")
db_path = os.path.join(tmpdir, "catalog.db")
warehouse = os.path.join(tmpdir, "warehouse")

try:
    session = IcebergSession.builder() \
        .catalog("sql", name="demo", uri=f"sqlite:///{db_path}",
                 warehouse=f"file://{warehouse}") \
        .build()

    session.sql("CREATE NAMESPACE IF NOT EXISTS part_db")

    # =========================================================================
    # 1. Identity partitioning
    # =========================================================================
    print("=== 1. Identity partition by region ===")
    session.sql("""
        CREATE TABLE part_db.sales (
            id INT,
            region STRING,
            product STRING,
            amount DOUBLE
        ) USING iceberg
        PARTITIONED BY (region)
    """)

    session.sql("""
        INSERT INTO part_db.sales VALUES
        (1, 'East', 'Widget', 100.0),
        (2, 'East', 'Gadget', 200.0),
        (3, 'West', 'Widget', 150.0),
        (4, 'West', 'Gadget', 250.0),
        (5, 'East', 'Widget', 120.0)
    """)

    print("All sales:")
    session.sql("SELECT * FROM part_db.sales ORDER BY id").show()

    # Query a specific partition -- DataFusion can push down the partition filter
    print("East region only:")
    session.sql("SELECT * FROM part_db.sales WHERE region = 'East' ORDER BY id").show()

    # Aggregate by partition column
    print("Revenue by region:")
    session.sql("SELECT * FROM part_db.sales") \
        .groupBy("region") \
        .agg(sum("amount").alias("total"), count("*").alias("num_sales")) \
        .orderBy("region") \
        .show()

    # =========================================================================
    # 2. Bucket partitioning
    # =========================================================================
    print("=== 2. Bucket partition ===")
    session.sql("""
        CREATE TABLE part_db.users (
            user_id INT,
            name STRING,
            email STRING
        ) USING iceberg
        PARTITIONED BY (bucket(4, user_id))
    """)

    session.sql("""
        INSERT INTO part_db.users VALUES
        (1, 'Alice', 'alice@example.com'),
        (2, 'Bob', 'bob@example.com'),
        (3, 'Carol', 'carol@example.com'),
        (4, 'Dave', 'dave@example.com'),
        (5, 'Eve', 'eve@example.com')
    """)
    print("Users (bucket partitioned by user_id):")
    session.sql("SELECT * FROM part_db.users ORDER BY user_id").show()

    # =========================================================================
    # 3. Time-based partitioning (year transform)
    # =========================================================================
    print("=== 3. Year partition on date column ===")
    session.sql("""
        CREATE TABLE part_db.events (
            event_id INT,
            event_date STRING,
            event_type STRING
        ) USING iceberg
        PARTITIONED BY (event_date)
    """)

    session.sql("""
        INSERT INTO part_db.events VALUES
        (1, '2024-01-15', 'click'),
        (2, '2024-01-15', 'view'),
        (3, '2024-01-16', 'click'),
        (4, '2024-01-16', 'purchase'),
        (5, '2024-01-17', 'view')
    """)

    print("All events:")
    session.sql("SELECT * FROM part_db.events ORDER BY event_id").show()

    print("Events on 2024-01-16:")
    session.sql("""
        SELECT * FROM part_db.events
        WHERE event_date = '2024-01-16'
        ORDER BY event_id
    """).show()

    # =========================================================================
    # 4. DataFrame API: partitionBy + saveAsTable
    # =========================================================================
    print("=== 4. DataFrame write with partitionBy ===")

    data = [
        (1, "US", "Product A", 50.0),
        (2, "US", "Product B", 75.0),
        (3, "EU", "Product A", 60.0),
        (4, "EU", "Product C", 90.0),
    ]
    df = session.createDataFrame(data, ["id", "country", "product", "revenue"])

    # Write with partitioning -- auto-creates the table with partition spec
    df.write.partitionBy("country").mode("append").saveAsTable("part_db.regional_sales")

    print("Regional sales (partitioned by country):")
    session.sql("SELECT * FROM part_db.regional_sales ORDER BY id").show()

    # =========================================================================
    # 5. INSERT OVERWRITE into a partitioned table
    # =========================================================================
    print("=== 5. INSERT OVERWRITE ===")

    print("Before overwrite:")
    session.sql("SELECT * FROM part_db.sales ORDER BY id").show()

    # Overwrite replaces all rows
    session.sql("""
        INSERT OVERWRITE part_db.sales VALUES
        (10, 'East', 'NewProduct', 999.0),
        (11, 'West', 'NewProduct', 888.0)
    """)

    print("After overwrite:")
    session.sql("SELECT * FROM part_db.sales ORDER BY id").show()

    # =========================================================================
    # 6. CTAS with partitioning
    # =========================================================================
    print("=== 6. CREATE TABLE AS SELECT with partitioning ===")
    session.sql("""
        CREATE TABLE part_db.sales_summary
        USING iceberg
        PARTITIONED BY (region)
        AS SELECT region, product, SUM(amount) as total_amount
        FROM part_db.sales
        GROUP BY region, product
    """)

    print("Sales summary (CTAS, partitioned by region):")
    session.sql("SELECT * FROM part_db.sales_summary ORDER BY region, product").show()

    # -- Cleanup --
    session.sql("DROP TABLE part_db.sales")
    session.sql("DROP TABLE part_db.users")
    session.sql("DROP TABLE part_db.events")
    session.sql("DROP TABLE part_db.regional_sales")
    session.sql("DROP TABLE part_db.sales_summary")
    session.sql("DROP NAMESPACE part_db")
    session.stop()
    print("=== Done ===")

finally:
    shutil.rmtree(tmpdir, ignore_errors=True)
