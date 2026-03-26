#!/usr/bin/env python3
"""Time travel example for iceberg_spark.

Demonstrates inserting data in multiple steps, querying previous
versions, and querying Iceberg metadata tables.
"""

import os
import shutil
import tempfile
import time

from iceberg_spark import IcebergSession

# -- Setup --
tmpdir = tempfile.mkdtemp(prefix="iceberg_timetravel_")
db_path = os.path.join(tmpdir, "catalog.db")
warehouse = os.path.join(tmpdir, "warehouse")

try:
    session = IcebergSession.builder() \
        .catalog("sql", name="demo", uri=f"sqlite:///{db_path}",
                 warehouse=f"file://{warehouse}") \
        .build()

    session.sql("CREATE NAMESPACE IF NOT EXISTS tt_db")

    # =========================================================================
    # 1. Create table and insert initial data (Snapshot 1)
    # =========================================================================
    session.sql("""
        CREATE TABLE tt_db.accounts (
            id INT,
            name STRING,
            balance DOUBLE
        ) USING iceberg
    """)

    session.sql("""
        INSERT INTO tt_db.accounts VALUES
        (1, 'Alice', 1000.0),
        (2, 'Bob', 2000.0),
        (3, 'Carol', 3000.0)
    """)

    print("=== Snapshot 1: Initial data ===")
    session.sql("SELECT * FROM tt_db.accounts ORDER BY id").show()

    # Small pause so snapshot timestamps differ
    time.sleep(1)

    # =========================================================================
    # 2. Insert more data (Snapshot 2)
    # =========================================================================
    session.sql("""
        INSERT INTO tt_db.accounts VALUES
        (4, 'Dave', 4000.0),
        (5, 'Eve', 5000.0)
    """)

    print("=== Snapshot 2: After adding Dave and Eve ===")
    session.sql("SELECT * FROM tt_db.accounts ORDER BY id").show()

    time.sleep(1)

    # =========================================================================
    # 3. Update some data (Snapshot 3)
    # =========================================================================
    session.sql("""
        UPDATE tt_db.accounts
        SET balance = balance + 500.0
        WHERE name = 'Alice'
    """)

    print("=== Snapshot 3: After updating Alice's balance ===")
    session.sql("SELECT * FROM tt_db.accounts ORDER BY id").show()

    # =========================================================================
    # 4. Query metadata tables
    # =========================================================================
    print("=== Metadata: snapshots ===")
    session.sql("SELECT * FROM tt_db.accounts.snapshots").show()

    print("=== Metadata: history ===")
    session.sql("SELECT * FROM tt_db.accounts.history").show()

    # =========================================================================
    # 5. Time travel: query by snapshot ID
    # =========================================================================

    # Get the first snapshot ID from the history table
    snapshots = session.sql("SELECT * FROM tt_db.accounts.snapshots").collect()
    if len(snapshots) >= 2:
        first_snapshot_id = snapshots[0]["snapshot_id"]
        second_snapshot_id = snapshots[1]["snapshot_id"]

        print(f"=== Time Travel: VERSION AS OF {first_snapshot_id} (Snapshot 1) ===")
        session.sql(
            f"SELECT * FROM tt_db.accounts VERSION AS OF {first_snapshot_id} ORDER BY id"
        ).show()

        print(f"=== Time Travel: VERSION AS OF {second_snapshot_id} (Snapshot 2) ===")
        session.sql(
            f"SELECT * FROM tt_db.accounts VERSION AS OF {second_snapshot_id} ORDER BY id"
        ).show()

        print("=== Current state (Snapshot 3) ===")
        session.sql("SELECT * FROM tt_db.accounts ORDER BY id").show()
    else:
        print("(Not enough snapshots for time travel demo)")

    # =========================================================================
    # 6. Metadata: files and manifests
    # =========================================================================
    print("=== Metadata: data files ===")
    session.sql("SELECT * FROM tt_db.accounts.files").show()

    print("=== Metadata: manifest list ===")
    session.sql("SELECT * FROM tt_db.accounts.manifests").show()

    print("=== Metadata: refs (branches/tags) ===")
    session.sql("SELECT * FROM tt_db.accounts.refs").show()

    # -- Cleanup --
    session.sql("DROP TABLE tt_db.accounts")
    session.sql("DROP NAMESPACE tt_db")
    session.stop()
    print("=== Done ===")

finally:
    shutil.rmtree(tmpdir, ignore_errors=True)
