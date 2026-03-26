#!/usr/bin/env python3
"""Quickstart example for iceberg_spark.

Creates a session with a temporary SQLite catalog, creates a table,
inserts data, queries it, and cleans up.
"""

import os
import shutil
import tempfile

from iceberg_spark import IcebergSession

# -- Setup: create a temporary directory for the warehouse and catalog --
tmpdir = tempfile.mkdtemp(prefix="iceberg_quickstart_")
db_path = os.path.join(tmpdir, "catalog.db")
warehouse = os.path.join(tmpdir, "warehouse")

try:
    # -- 1. Create an IcebergSession with a local SQLite catalog --
    session = IcebergSession.builder() \
        .appName("quickstart") \
        .catalog(
            "sql",
            name="demo",
            uri=f"sqlite:///{db_path}",
            warehouse=f"file://{warehouse}",
        ) \
        .build()

    print("=== Session created ===")
    print(f"Version: {session.version}")
    print()

    # -- 2. Create a namespace (database) --
    session.sql("CREATE NAMESPACE IF NOT EXISTS quickstart_db")
    print("=== Namespace created ===")
    print()

    # -- 3. Create a table --
    session.sql("""
        CREATE TABLE quickstart_db.employees (
            id INT,
            name STRING,
            department STRING,
            salary DOUBLE
        ) USING iceberg
    """)
    print("=== Table created ===")
    print()

    # -- 4. Insert data --
    session.sql("""
        INSERT INTO quickstart_db.employees VALUES
        (1, 'Alice', 'Engineering', 95000.0),
        (2, 'Bob', 'Marketing', 72000.0),
        (3, 'Carol', 'Engineering', 105000.0),
        (4, 'Dave', 'Marketing', 68000.0),
        (5, 'Eve', 'Engineering', 110000.0)
    """)
    print("=== Data inserted ===")
    print()

    # -- 5. Query the data --
    print("All employees:")
    session.sql("SELECT * FROM quickstart_db.employees ORDER BY id").show()
    print()

    # -- 6. Filter and aggregate --
    print("Engineering department average salary:")
    session.sql("""
        SELECT department, COUNT(*) as headcount, ROUND(AVG(salary), 2) as avg_salary
        FROM quickstart_db.employees
        WHERE department = 'Engineering'
        GROUP BY department
    """).show()
    print()

    # -- 7. Use DataFrame API --
    from iceberg_spark.functions import col, avg, count

    df = session.sql("SELECT * FROM quickstart_db.employees")
    print("DataFrame columns:", df.columns)
    print("DataFrame schema:")
    df.printSchema()
    print()

    print("Employees earning over 80k (DataFrame API):")
    df.filter(col("salary") > 80000) \
      .select("name", "department", "salary") \
      .orderBy(col("salary").desc()) \
      .show()

    # -- 8. Create a DataFrame from Python data --
    data = [
        (6, "Frank", "Sales", 62000.0),
        (7, "Grace", "Sales", 71000.0),
    ]
    new_df = session.createDataFrame(data, schema=["id", "name", "department", "salary"])
    print("New DataFrame from Python data:")
    new_df.show()

    # -- 9. Clean up --
    session.sql("DROP TABLE quickstart_db.employees")
    session.sql("DROP NAMESPACE quickstart_db")
    session.stop()
    print("=== Cleanup complete ===")

finally:
    shutil.rmtree(tmpdir, ignore_errors=True)
