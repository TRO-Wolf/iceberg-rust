#!/usr/bin/env python3
"""DML operations example for iceberg_spark.

Demonstrates INSERT, UPDATE, DELETE, and MERGE INTO.
All DML uses copy-on-write under the hood.
"""

import os
import shutil
import tempfile

from iceberg_spark import IcebergSession

# -- Setup --
tmpdir = tempfile.mkdtemp(prefix="iceberg_dml_")
db_path = os.path.join(tmpdir, "catalog.db")
warehouse = os.path.join(tmpdir, "warehouse")

try:
    session = IcebergSession.builder() \
        .catalog("sql", name="demo", uri=f"sqlite:///{db_path}",
                 warehouse=f"file://{warehouse}") \
        .build()

    session.sql("CREATE NAMESPACE IF NOT EXISTS dml_db")

    # -- Create table --
    session.sql("""
        CREATE TABLE dml_db.inventory (
            product_id INT,
            product_name STRING,
            quantity INT,
            price DOUBLE
        ) USING iceberg
    """)

    # =========================================================================
    # 1. INSERT INTO
    # =========================================================================
    print("=== 1. INSERT INTO ===")
    session.sql("""
        INSERT INTO dml_db.inventory VALUES
        (1, 'Laptop', 50, 999.99),
        (2, 'Mouse', 200, 29.99),
        (3, 'Keyboard', 150, 79.99),
        (4, 'Monitor', 75, 349.99),
        (5, 'Headset', 100, 59.99)
    """)
    session.sql("SELECT * FROM dml_db.inventory ORDER BY product_id").show()

    # =========================================================================
    # 2. INSERT INTO with SELECT
    # =========================================================================
    print("=== 2. INSERT INTO with SELECT (duplicate high-value items) ===")

    # Create a temp source table for the SELECT
    session.sql("""
        CREATE TABLE dml_db.new_stock (
            product_id INT,
            product_name STRING,
            quantity INT,
            price DOUBLE
        ) USING iceberg
    """)
    session.sql("""
        INSERT INTO dml_db.new_stock VALUES
        (6, 'Webcam', 80, 49.99),
        (7, 'USB Hub', 120, 24.99)
    """)
    session.sql("""
        INSERT INTO dml_db.inventory
        SELECT * FROM dml_db.new_stock
    """)
    print("After inserting from new_stock:")
    session.sql("SELECT * FROM dml_db.inventory ORDER BY product_id").show()

    # =========================================================================
    # 3. UPDATE
    # =========================================================================
    print("=== 3. UPDATE: increase price of Laptop by 10% ===")
    session.sql("""
        UPDATE dml_db.inventory
        SET price = price * 1.10
        WHERE product_name = 'Laptop'
    """)
    session.sql("SELECT * FROM dml_db.inventory WHERE product_name = 'Laptop'").show()

    print("=== 3b. UPDATE multiple columns ===")
    session.sql("""
        UPDATE dml_db.inventory
        SET quantity = quantity + 50, price = price - 5.0
        WHERE product_name = 'Mouse'
    """)
    session.sql("SELECT * FROM dml_db.inventory WHERE product_name = 'Mouse'").show()

    # =========================================================================
    # 4. DELETE
    # =========================================================================
    print("=== 4. DELETE: remove items with quantity < 100 ===")
    print("Before delete:")
    session.sql("SELECT * FROM dml_db.inventory ORDER BY product_id").show()

    session.sql("""
        DELETE FROM dml_db.inventory
        WHERE quantity < 100
    """)
    print("After delete:")
    session.sql("SELECT * FROM dml_db.inventory ORDER BY product_id").show()

    # =========================================================================
    # 5. MERGE INTO
    # =========================================================================
    print("=== 5. MERGE INTO ===")

    # Create a source table with updates and new rows
    session.sql("""
        CREATE TABLE dml_db.updates (
            product_id INT,
            product_name STRING,
            quantity INT,
            price DOUBLE
        ) USING iceberg
    """)
    session.sql("""
        INSERT INTO dml_db.updates VALUES
        (2, 'Mouse', 300, 19.99),
        (3, 'Keyboard', 200, 69.99),
        (8, 'Speakers', 60, 89.99),
        (9, 'Mousepad', 500, 14.99)
    """)

    print("Source (updates) table:")
    session.sql("SELECT * FROM dml_db.updates ORDER BY product_id").show()

    print("Target (inventory) before merge:")
    session.sql("SELECT * FROM dml_db.inventory ORDER BY product_id").show()

    session.sql("""
        MERGE INTO dml_db.inventory AS target
        USING dml_db.updates AS source
        ON target.product_id = source.product_id
        WHEN MATCHED THEN
            UPDATE SET quantity = source.quantity, price = source.price
        WHEN NOT MATCHED THEN
            INSERT (product_id, product_name, quantity, price)
            VALUES (source.product_id, source.product_name, source.quantity, source.price)
    """)
    print("After merge:")
    session.sql("SELECT * FROM dml_db.inventory ORDER BY product_id").show()

    # =========================================================================
    # 6. INSERT OVERWRITE
    # =========================================================================
    print("=== 6. INSERT OVERWRITE: replace all data ===")
    session.sql("""
        INSERT OVERWRITE dml_db.inventory VALUES
        (1, 'Laptop Pro', 25, 1499.99),
        (2, 'Mouse Wireless', 500, 39.99)
    """)
    print("After overwrite:")
    session.sql("SELECT * FROM dml_db.inventory ORDER BY product_id").show()

    # =========================================================================
    # 7. TRUNCATE
    # =========================================================================
    print("=== 7. TRUNCATE TABLE ===")
    session.sql("TRUNCATE TABLE dml_db.inventory")
    print(f"Row count after truncate: {session.sql('SELECT * FROM dml_db.inventory').count()}")

    # -- Cleanup --
    session.sql("DROP TABLE dml_db.inventory")
    session.sql("DROP TABLE dml_db.new_stock")
    session.sql("DROP TABLE dml_db.updates")
    session.sql("DROP NAMESPACE dml_db")
    session.stop()
    print("\n=== Done ===")

finally:
    shutil.rmtree(tmpdir, ignore_errors=True)
