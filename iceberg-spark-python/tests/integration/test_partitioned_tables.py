"""Integration tests for partitioned table support.

Tests cover:
  - CREATE TABLE with identity, temporal, and bucket partition transforms
  - INSERT INTO partitioned tables (VALUES and SELECT)
  - CTAS with PARTITIONED BY
  - DataFrame API writes (partitionBy, partitionedBy)
  - DML (DELETE, UPDATE, MERGE INTO) on partitioned tables
  - Write modes (append, overwrite) on partitioned tables

Run with:
    uv run pytest tests/integration/test_partitioned_tables.py -v
"""

from __future__ import annotations

import os
import tempfile

import pyarrow as pa
import pytest

try:
    import pyiceberg.catalog.sql  # noqa: F401
    from iceberg_spark import IcebergSession
    HAS_SQL_CATALOG = True
except ImportError:
    HAS_SQL_CATALOG = False

pytestmark = pytest.mark.skipif(
    not HAS_SQL_CATALOG,
    reason="pyiceberg[sql-sqlite] / sqlalchemy not installed — skipping integration tests",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def warehouse():
    with tempfile.TemporaryDirectory(prefix="iceberg_partition_test_") as d:
        yield d


@pytest.fixture(scope="module")
def session(warehouse):
    db_path = os.path.join(warehouse, "catalog.db")
    sess = (
        IcebergSession.builder()
        .catalog(
            "sql",
            name="partition_catalog",
            uri=f"sqlite:///{db_path}",
            warehouse=f"file://{warehouse}",
        )
        .build()
    )
    # Create default namespace
    sess.sql("CREATE DATABASE IF NOT EXISTS default")
    return sess


# ---------------------------------------------------------------------------
# CREATE TABLE with partition specs
# ---------------------------------------------------------------------------

class TestCreatePartitionedTable:
    """Test CREATE TABLE with various partition transforms."""

    def test_identity_partition(self, session):
        session.sql("""
            CREATE TABLE default.part_identity (
                id INT,
                region STRING,
                value DOUBLE
            ) PARTITIONED BY (region)
        """)
        # Verify table exists and data round-trips
        session.sql("INSERT INTO default.part_identity VALUES (1, 'US', 10.0), (2, 'EU', 20.0)")
        result = session.sql("SELECT * FROM default.part_identity").toPandas()
        assert len(result) == 2
        assert set(result["region"]) == {"US", "EU"}

    def test_multi_column_identity_partition(self, session):
        session.sql("""
            CREATE TABLE default.part_multi (
                id INT,
                region STRING,
                category STRING,
                value DOUBLE
            ) PARTITIONED BY (region, category)
        """)
        session.sql("""
            INSERT INTO default.part_multi VALUES
                (1, 'US', 'A', 10.0),
                (2, 'EU', 'B', 20.0),
                (3, 'US', 'B', 30.0)
        """)
        result = session.sql("SELECT * FROM default.part_multi").toPandas()
        assert len(result) == 3

    def test_bucket_partition(self, session):
        session.sql("""
            CREATE TABLE default.part_bucket (
                id INT,
                name STRING
            ) PARTITIONED BY (bucket(4, id))
        """)
        session.sql("INSERT INTO default.part_bucket VALUES (1, 'a'), (2, 'b'), (3, 'c')")
        result = session.sql("SELECT * FROM default.part_bucket").toPandas()
        assert len(result) == 3

    def test_year_partition(self, session):
        session.sql("""
            CREATE TABLE default.part_year (
                id INT,
                ts TIMESTAMP,
                value DOUBLE
            ) PARTITIONED BY (year(ts))
        """)
        session.sql("""
            INSERT INTO default.part_year VALUES
                (1, TIMESTAMP '2024-01-15 10:00:00', 10.0),
                (2, TIMESTAMP '2025-06-20 12:00:00', 20.0)
        """)
        result = session.sql("SELECT * FROM default.part_year").toPandas()
        assert len(result) == 2

    def test_month_partition(self, session):
        session.sql("""
            CREATE TABLE default.part_month (
                id INT,
                ts TIMESTAMP
            ) PARTITIONED BY (month(ts))
        """)
        session.sql("""
            INSERT INTO default.part_month VALUES
                (1, TIMESTAMP '2024-01-15 10:00:00'),
                (2, TIMESTAMP '2024-03-20 12:00:00')
        """)
        result = session.sql("SELECT * FROM default.part_month").toPandas()
        assert len(result) == 2

    def test_day_partition(self, session):
        session.sql("""
            CREATE TABLE default.part_day (
                id INT,
                ts TIMESTAMP
            ) PARTITIONED BY (day(ts))
        """)
        session.sql("""
            INSERT INTO default.part_day VALUES
                (1, TIMESTAMP '2024-01-15 10:00:00'),
                (2, TIMESTAMP '2024-01-16 12:00:00')
        """)
        result = session.sql("SELECT * FROM default.part_day").toPandas()
        assert len(result) == 2

    def test_truncate_partition(self, session):
        session.sql("""
            CREATE TABLE default.part_trunc (
                id INT,
                name STRING
            ) PARTITIONED BY (truncate(3, name))
        """)
        session.sql("INSERT INTO default.part_trunc VALUES (1, 'abcdef'), (2, 'abcxyz')")
        result = session.sql("SELECT * FROM default.part_trunc").toPandas()
        assert len(result) == 2


# ---------------------------------------------------------------------------
# INSERT operations on partitioned tables
# ---------------------------------------------------------------------------

class TestInsertPartitioned:
    """Test INSERT operations on partitioned tables."""

    def test_insert_values(self, session):
        session.sql("""
            CREATE TABLE default.ins_part (
                id INT,
                region STRING,
                amount DOUBLE
            ) PARTITIONED BY (region)
        """)
        session.sql("INSERT INTO default.ins_part VALUES (1, 'US', 100.0)")
        session.sql("INSERT INTO default.ins_part VALUES (2, 'EU', 200.0)")
        result = session.sql("SELECT * FROM default.ins_part ORDER BY id").toPandas()
        assert len(result) == 2
        assert list(result["id"]) == [1, 2]

    def test_insert_select(self, session):
        # Create source table
        session.sql("""
            CREATE TABLE default.ins_src (id INT, region STRING, amount DOUBLE)
        """)
        session.sql("INSERT INTO default.ins_src VALUES (10, 'JP', 1000.0), (20, 'KR', 2000.0)")

        # Create partitioned target
        session.sql("""
            CREATE TABLE default.ins_part_sel (
                id INT, region STRING, amount DOUBLE
            ) PARTITIONED BY (region)
        """)
        session.sql("INSERT INTO default.ins_part_sel SELECT * FROM default.ins_src")
        result = session.sql("SELECT * FROM default.ins_part_sel").toPandas()
        assert len(result) == 2
        assert set(result["region"]) == {"JP", "KR"}

    def test_append_multiple_batches(self, session):
        session.sql("""
            CREATE TABLE default.ins_append (
                id INT, region STRING
            ) PARTITIONED BY (region)
        """)
        session.sql("INSERT INTO default.ins_append VALUES (1, 'US')")
        session.sql("INSERT INTO default.ins_append VALUES (2, 'US')")
        session.sql("INSERT INTO default.ins_append VALUES (3, 'EU')")
        result = session.sql("SELECT * FROM default.ins_append").toPandas()
        assert len(result) == 3

    def test_insert_overwrite(self, session):
        session.sql("""
            CREATE TABLE default.ins_overwrite (
                id INT, region STRING
            ) PARTITIONED BY (region)
        """)
        session.sql("INSERT INTO default.ins_overwrite VALUES (1, 'US'), (2, 'EU')")
        # Overwrite with JP data — only the JP partition is affected;
        # US and EU partitions are preserved (partition-scoped overwrite).
        session.sql("INSERT OVERWRITE default.ins_overwrite VALUES (3, 'JP')")
        result = session.sql("SELECT * FROM default.ins_overwrite ORDER BY id").toPandas()
        assert len(result) == 3
        assert set(result["region"]) == {"US", "EU", "JP"}


# ---------------------------------------------------------------------------
# CTAS with PARTITIONED BY
# ---------------------------------------------------------------------------

class TestCTASPartitioned:
    """Test CREATE TABLE AS SELECT with partition specs."""

    def test_ctas_identity_partition(self, session):
        # Create source data
        session.sql("CREATE TABLE default.ctas_src (id INT, region STRING, val DOUBLE)")
        session.sql("""
            INSERT INTO default.ctas_src VALUES
                (1, 'US', 10.0), (2, 'EU', 20.0), (3, 'US', 30.0)
        """)
        # CTAS with partition
        session.sql("""
            CREATE TABLE default.ctas_part
            PARTITIONED BY (region)
            AS SELECT * FROM default.ctas_src
        """)
        result = session.sql("SELECT * FROM default.ctas_part").toPandas()
        assert len(result) == 3
        assert set(result["region"]) == {"US", "EU"}

    def test_ctas_bucket_partition(self, session):
        session.sql("""
            CREATE TABLE default.ctas_bucket
            PARTITIONED BY (bucket(4, id))
            AS SELECT * FROM default.ctas_src
        """)
        result = session.sql("SELECT * FROM default.ctas_bucket").toPandas()
        assert len(result) == 3


# ---------------------------------------------------------------------------
# DataFrame API writes to partitioned tables
# ---------------------------------------------------------------------------

class TestDataFrameWriterPartitioned:
    """Test DataFrame writer APIs with partition specs."""

    def test_partition_by_save_as_table(self, session):
        df = session.sql("SELECT 1 AS id, 'US' AS region, 10.0 AS value")
        df.write.partitionBy("region").saveAsTable("default.df_part")
        result = session.sql("SELECT * FROM default.df_part").toPandas()
        assert len(result) == 1
        assert result["region"].iloc[0] == "US"

    def test_partition_by_append(self, session):
        """Append to the table created in the previous test."""
        df = session.sql("SELECT 2 AS id, 'EU' AS region, 20.0 AS value")
        df.write.mode("append").saveAsTable("default.df_part")
        result = session.sql("SELECT * FROM default.df_part").toPandas()
        assert len(result) == 2

    def test_writer_v2_partitioned_by_create(self, session):
        df = session.sql(
            "SELECT 1 AS id, 'US' AS region, 100.0 AS amount"
        )
        df.writeTo("default.v2_part").partitionedBy("region").create()
        result = session.sql("SELECT * FROM default.v2_part").toPandas()
        assert len(result) == 1
        assert result["region"].iloc[0] == "US"

    def test_writer_v2_append_to_partitioned(self, session):
        df = session.sql("SELECT 2 AS id, 'EU' AS region, 200.0 AS amount")
        df.writeTo("default.v2_part").append()
        result = session.sql("SELECT * FROM default.v2_part").toPandas()
        assert len(result) == 2


# ---------------------------------------------------------------------------
# DML on partitioned tables
# ---------------------------------------------------------------------------

class TestDMLPartitioned:
    """Test DML operations on partitioned tables."""

    @pytest.fixture(autouse=True)
    def setup_dml_table(self, session):
        """Create a fresh partitioned table for DML tests."""
        try:
            session.sql("DROP TABLE default.dml_part")
        except Exception:
            pass
        session.sql("""
            CREATE TABLE default.dml_part (
                id INT, region STRING, amount DOUBLE
            ) PARTITIONED BY (region)
        """)
        session.sql("""
            INSERT INTO default.dml_part VALUES
                (1, 'US', 100.0),
                (2, 'US', 200.0),
                (3, 'EU', 300.0),
                (4, 'EU', 400.0),
                (5, 'JP', 500.0)
        """)

    def test_delete_by_partition(self, session):
        session.sql("DELETE FROM default.dml_part WHERE region = 'US'")
        result = session.sql("SELECT * FROM default.dml_part").toPandas()
        assert len(result) == 3
        assert "US" not in set(result["region"])

    def test_delete_within_partition(self, session):
        session.sql("DELETE FROM default.dml_part WHERE id = 3")
        result = session.sql("SELECT * FROM default.dml_part").toPandas()
        assert len(result) == 4
        assert 3 not in list(result["id"])

    def test_update_by_partition(self, session):
        session.sql("UPDATE default.dml_part SET amount = 999.0 WHERE region = 'EU'")
        result = session.sql(
            "SELECT amount FROM default.dml_part WHERE region = 'EU'"
        ).toPandas()
        assert all(result["amount"] == 999.0)

    def test_update_within_partition(self, session):
        session.sql("UPDATE default.dml_part SET amount = 0.0 WHERE id = 1")
        result = session.sql(
            "SELECT amount FROM default.dml_part WHERE id = 1"
        ).toPandas()
        assert result["amount"].iloc[0] == 0.0

    def test_merge_into_partitioned(self, session):
        # Create source table
        session.sql("CREATE TABLE default.merge_src (id INT, region STRING, amount DOUBLE)")
        session.sql("""
            INSERT INTO default.merge_src VALUES
                (1, 'US', 150.0),
                (6, 'AU', 600.0)
        """)
        session.sql("""
            MERGE INTO default.dml_part AS t
            USING default.merge_src AS s
            ON t.id = s.id
            WHEN MATCHED THEN UPDATE SET amount = s.amount
            WHEN NOT MATCHED THEN INSERT (id, region, amount) VALUES (s.id, s.region, s.amount)
        """)
        result = session.sql("SELECT * FROM default.dml_part ORDER BY id").toPandas()
        # id=1 should be updated to 150.0, id=6 should be inserted
        assert len(result) == 6
        row_1 = result[result["id"] == 1]
        assert row_1["amount"].iloc[0] == 150.0
        row_6 = result[result["id"] == 6]
        assert row_6["region"].iloc[0] == "AU"


# ---------------------------------------------------------------------------
# Write modes on partitioned tables
# ---------------------------------------------------------------------------

class TestWriteModesPartitioned:
    """Test write modes (append, overwrite) on partitioned tables."""

    def test_overwrite_mode(self, session):
        session.sql("""
            CREATE TABLE default.mode_part (
                id BIGINT, region STRING
            ) PARTITIONED BY (region)
        """)
        session.sql("INSERT INTO default.mode_part VALUES (1, 'US'), (2, 'EU')")

        # Overwrite with JP data — partition-scoped: only JP partition affected
        df = session.sql("SELECT 3 AS id, 'JP' AS region")
        df.write.mode("overwrite").saveAsTable("default.mode_part")
        result = session.sql("SELECT * FROM default.mode_part ORDER BY id").toPandas()
        assert len(result) == 3
        assert set(result["region"]) == {"US", "EU", "JP"}

    def test_append_mode(self, session):
        session.sql("""
            CREATE TABLE default.mode_append (
                id BIGINT, region STRING
            ) PARTITIONED BY (region)
        """)
        session.sql("INSERT INTO default.mode_append VALUES (1, 'US')")

        df = session.sql("SELECT 2 AS id, 'EU' AS region")
        df.write.mode("append").saveAsTable("default.mode_append")
        result = session.sql("SELECT * FROM default.mode_append").toPandas()
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Partition-scoped overwrite tests
# ---------------------------------------------------------------------------

class TestPartitionScopedOverwrite:
    """Test that INSERT OVERWRITE only affects matching partitions."""

    def test_overwrite_preserves_other_partitions(self, session):
        """INSERT OVERWRITE with data for one partition preserves others."""
        session.sql("""
            CREATE TABLE default.pso_basic (
                id INT, name STRING, region STRING
            ) PARTITIONED BY (region)
        """)
        session.sql("""
            INSERT INTO default.pso_basic VALUES
                (1, 'Alice', 'US'),
                (2, 'Bob', 'EU'),
                (3, 'Carol', 'JP')
        """)
        # Overwrite with new US data only
        session.sql("INSERT OVERWRITE default.pso_basic SELECT 10, 'Dave', 'US'")
        rows = session.sql("SELECT * FROM default.pso_basic ORDER BY id").toPandas()
        # US data replaced (Alice -> Dave), EU and JP preserved
        assert len(rows) == 3
        assert set(rows["region"]) == {"US", "EU", "JP"}
        us_row = rows[rows["region"] == "US"]
        assert us_row["id"].iloc[0] == 10
        assert us_row["name"].iloc[0] == "Dave"
        # EU and JP unchanged
        eu_row = rows[rows["region"] == "EU"]
        assert eu_row["name"].iloc[0] == "Bob"
        jp_row = rows[rows["region"] == "JP"]
        assert jp_row["name"].iloc[0] == "Carol"

    def test_overwrite_replaces_matching_partition(self, session):
        """Overwriting a partition replaces all rows in that partition."""
        session.sql("""
            CREATE TABLE default.pso_replace (
                id INT, region STRING
            ) PARTITIONED BY (region)
        """)
        session.sql("""
            INSERT INTO default.pso_replace VALUES
                (1, 'US'), (2, 'US'), (3, 'EU')
        """)
        # Replace US partition with single row
        session.sql("INSERT OVERWRITE default.pso_replace SELECT 10, 'US'")
        rows = session.sql("SELECT * FROM default.pso_replace ORDER BY id").toPandas()
        assert len(rows) == 2  # 1 US (replaced) + 1 EU (preserved)
        us_rows = rows[rows["region"] == "US"]
        assert len(us_rows) == 1
        assert us_rows["id"].iloc[0] == 10

    def test_overwrite_multiple_partitions(self, session):
        """Overwriting with data spanning two partitions replaces both."""
        session.sql("""
            CREATE TABLE default.pso_multi (
                id INT, region STRING
            ) PARTITIONED BY (region)
        """)
        session.sql("""
            INSERT INTO default.pso_multi VALUES
                (1, 'US'), (2, 'EU'), (3, 'JP')
        """)
        # Replace US and EU partitions, preserve JP
        session.sql("""
            INSERT OVERWRITE default.pso_multi
            SELECT 10, 'US' UNION ALL SELECT 20, 'EU'
        """)
        rows = session.sql("SELECT * FROM default.pso_multi ORDER BY id").toPandas()
        assert len(rows) == 3
        assert set(rows["region"]) == {"US", "EU", "JP"}
        assert rows[rows["region"] == "US"]["id"].iloc[0] == 10
        assert rows[rows["region"] == "EU"]["id"].iloc[0] == 20
        assert rows[rows["region"] == "JP"]["id"].iloc[0] == 3  # preserved

    def test_overwrite_unpartitioned_table_full_replace(self, session):
        """Unpartitioned table: overwrite replaces all data (AlwaysTrue)."""
        session.sql("""
            CREATE TABLE default.pso_unpart (id INT, name STRING)
        """)
        session.sql("INSERT INTO default.pso_unpart VALUES (1, 'Alice'), (2, 'Bob')")
        session.sql("INSERT OVERWRITE default.pso_unpart SELECT 10, 'Dave'")
        rows = session.sql("SELECT * FROM default.pso_unpart").toPandas()
        assert len(rows) == 1
        assert rows["name"].iloc[0] == "Dave"

    def test_overwrite_df_api_preserves_partitions(self, session):
        """DataFrame write.mode('overwrite') with partition-scoped filter."""
        session.sql("""
            CREATE TABLE default.pso_df (
                id BIGINT, region STRING
            ) PARTITIONED BY (region)
        """)
        session.sql("INSERT INTO default.pso_df VALUES (1, 'US'), (2, 'EU')")
        df = session.sql("SELECT 10 AS id, 'US' AS region")
        df.write.mode("overwrite").saveAsTable("default.pso_df")
        rows = session.sql("SELECT * FROM default.pso_df ORDER BY id").toPandas()
        # US replaced, EU preserved
        assert len(rows) == 2
        assert set(rows["region"]) == {"US", "EU"}
        assert rows[rows["region"] == "US"]["id"].iloc[0] == 10


# ---------------------------------------------------------------------------
# WriterV2.overwritePartitions() tests
# ---------------------------------------------------------------------------

class TestWriterV2OverwritePartitions:
    """Test DataFrameWriterV2.overwritePartitions() partition-scoped overwrite."""

    def test_overwrite_partitions_preserves_other_partitions(self, session):
        """WriterV2.overwritePartitions() should only overwrite affected partitions."""
        session.sql("""
            CREATE TABLE default.owp_basic (
                id BIGINT, region STRING
            ) PARTITIONED BY (region)
        """)
        session.sql("INSERT INTO default.owp_basic VALUES (1, 'US'), (2, 'EU'), (3, 'JP')")

        # Create a DataFrame with US data only
        new_df = session.createDataFrame([(10, "US")], ["id", "region"])

        # overwritePartitions should only replace US partition
        new_df.writeTo("default.owp_basic").overwritePartitions()

        rows = session.sql("SELECT * FROM default.owp_basic ORDER BY id").collect()
        ids = sorted([r["id"] for r in rows])
        assert ids == [2, 3, 10]
        regions = {r["region"] for r in rows}
        assert regions == {"US", "EU", "JP"}

    def test_overwrite_partitions_replaces_all_rows_in_partition(self, session):
        """overwritePartitions() replaces all rows within the affected partition."""
        session.sql("""
            CREATE TABLE default.owp_replace (
                id BIGINT, region STRING
            ) PARTITIONED BY (region)
        """)
        session.sql("""
            INSERT INTO default.owp_replace VALUES
                (1, 'US'), (2, 'US'), (3, 'EU')
        """)

        new_df = session.createDataFrame([(10, "US")], ["id", "region"])
        new_df.writeTo("default.owp_replace").overwritePartitions()

        rows = session.sql("SELECT * FROM default.owp_replace ORDER BY id").collect()
        ids = sorted([r["id"] for r in rows])
        # Both US rows (1, 2) replaced by single row (10); EU row (3) preserved
        assert ids == [3, 10]

    def test_overwrite_partitions_multiple_partitions(self, session):
        """overwritePartitions() with data spanning two partitions replaces both."""
        session.sql("""
            CREATE TABLE default.owp_multi (
                id BIGINT, region STRING
            ) PARTITIONED BY (region)
        """)
        session.sql("""
            INSERT INTO default.owp_multi VALUES
                (1, 'US'), (2, 'EU'), (3, 'JP')
        """)

        new_df = session.createDataFrame([(10, "US"), (20, "EU")], ["id", "region"])
        new_df.writeTo("default.owp_multi").overwritePartitions()

        rows = session.sql("SELECT * FROM default.owp_multi ORDER BY id").collect()
        ids = sorted([r["id"] for r in rows])
        # US and EU replaced, JP preserved
        assert ids == [3, 10, 20]

    def test_overwrite_partitions_unpartitioned_table(self, session):
        """overwritePartitions() on unpartitioned table does full overwrite."""
        session.sql("CREATE TABLE default.owp_unpart (id BIGINT, name STRING)")
        session.sql("INSERT INTO default.owp_unpart VALUES (1, 'Alice'), (2, 'Bob')")

        new_df = session.createDataFrame([(10, "Dave")], ["id", "name"])
        new_df.writeTo("default.owp_unpart").overwritePartitions()

        rows = session.sql("SELECT * FROM default.owp_unpart").collect()
        assert len(rows) == 1
        assert rows[0]["name"] == "Dave"
