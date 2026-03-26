"""Integration tests for temp views used in DML operations.

Verifies that temp views (created via createOrReplaceTempView or CREATE VIEW)
can be referenced in INSERT INTO, CTAS, DELETE, UPDATE, and MERGE INTO.

Requires PyIceberg + SQLite catalog.  Skipped automatically when unavailable.

Run with:
    uv run pytest tests/integration/test_tempview_dml.py -v
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
    reason="pyiceberg[sql-sqlite] / sqlalchemy not installed — skipping temp-view DML tests",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def warehouse():
    with tempfile.TemporaryDirectory(prefix="iceberg_tvdml_test_") as d:
        yield d


@pytest.fixture(scope="module")
def session(warehouse):
    db_path = os.path.join(warehouse, "catalog.db")
    sess = (
        IcebergSession.builder()
        .catalog(
            "sql",
            name="tvdml_catalog",
            uri=f"sqlite:///{db_path}",
            warehouse=f"file://{warehouse}",
        )
        .build()
    )
    yield sess
    sess.stop()


@pytest.fixture(scope="module", autouse=True)
def setup_namespace(session):
    try:
        session.sql("CREATE DATABASE IF NOT EXISTS tv")
    except Exception:
        pass


@pytest.fixture
def fresh_table(session):
    """Creates a fresh table with 5 rows; drops it after the test."""
    tbl = "tv.employees"
    session.sql(f"DROP TABLE IF EXISTS {tbl}")
    session.sql(
        f"CREATE TABLE {tbl} (id INT, name STRING, salary DOUBLE) USING iceberg"
    )
    session.sql(
        f"INSERT INTO {tbl} VALUES "
        "(1, 'Alice', 90000.0), (2, 'Bob', 80000.0), "
        "(3, 'Carol', 70000.0), (4, 'Dave', 60000.0), (5, 'Eve', 50000.0)"
    )
    yield tbl
    session.sql(f"DROP TABLE IF EXISTS {tbl}")


# ---------------------------------------------------------------------------
# INSERT INTO ... SELECT FROM temp_view (baseline — should already work)
# ---------------------------------------------------------------------------


class TestTempViewInsert:
    def test_insert_from_temp_view(self, session, fresh_table):
        # Create temp view with new rows
        new_data = pa.table({
            "id": pa.array([6, 7], type=pa.int32()),
            "name": pa.array(["Frank", "Grace"], type=pa.string()),
            "salary": pa.array([55000.0, 65000.0], type=pa.float64()),
        })
        df = session.createDataFrame(new_data)
        df.createOrReplaceTempView("new_employees")

        session.sql(f"INSERT INTO {fresh_table} SELECT * FROM new_employees")

        result = session.sql(f"SELECT * FROM {fresh_table}").collect()
        assert len(result) == 7
        ids = sorted(row[0] for row in result)
        assert ids == [1, 2, 3, 4, 5, 6, 7]

    def test_ctas_from_temp_view(self, session):
        ctas_tbl = "tv.ctas_from_view"
        session.sql(f"DROP TABLE IF EXISTS {ctas_tbl}")

        view_data = pa.table({
            "x": pa.array([10, 20, 30], type=pa.int32()),
            "y": pa.array(["a", "b", "c"], type=pa.string()),
        })
        df = session.createDataFrame(view_data)
        df.createOrReplaceTempView("ctas_source")

        session.sql(f"CREATE TABLE {ctas_tbl} AS SELECT * FROM ctas_source")

        result = session.sql(f"SELECT * FROM {ctas_tbl}").collect()
        assert len(result) == 3
        session.sql(f"DROP TABLE IF EXISTS {ctas_tbl}")


# ---------------------------------------------------------------------------
# DELETE FROM ... WHERE ... (subquery from temp_view)
# ---------------------------------------------------------------------------


class TestTempViewDelete:
    def test_delete_with_subquery_from_temp_view(self, session, fresh_table):
        # Temp view contains IDs to delete
        ids_to_delete = pa.table({"id": pa.array([2, 4], type=pa.int32())})
        df = session.createDataFrame(ids_to_delete)
        df.createOrReplaceTempView("delete_ids")

        session.sql(
            f"DELETE FROM {fresh_table} WHERE id IN (SELECT id FROM delete_ids)"
        )

        result = session.sql(f"SELECT * FROM {fresh_table}").collect()
        remaining_ids = sorted(row[0] for row in result)
        assert remaining_ids == [1, 3, 5]

    def test_delete_with_temp_view_no_match(self, session, fresh_table):
        # Temp view contains IDs not in the target table
        ids_no_match = pa.table({"id": pa.array([99, 100], type=pa.int32())})
        df = session.createDataFrame(ids_no_match)
        df.createOrReplaceTempView("no_match_ids")

        session.sql(
            f"DELETE FROM {fresh_table} WHERE id IN (SELECT id FROM no_match_ids)"
        )

        result = session.sql(f"SELECT * FROM {fresh_table}").collect()
        assert len(result) == 5  # Nothing deleted


# ---------------------------------------------------------------------------
# UPDATE ... WHERE ... (subquery from temp_view)
# ---------------------------------------------------------------------------


class TestTempViewUpdate:
    def test_update_set_from_temp_view_scalar(self, session, fresh_table):
        # Temp view with a scalar value to use in SET
        bonus_data = pa.table({"bonus": pa.array([5000.0], type=pa.float64())})
        df = session.createDataFrame(bonus_data)
        df.createOrReplaceTempView("bonus_view")

        # Use scalar subquery in SET clause (DataFusion supports scalar subqueries)
        session.sql(
            f"UPDATE {fresh_table} SET salary = salary + (SELECT bonus FROM bonus_view) "
            "WHERE id <= 2"
        )

        result = session.sql(
            f"SELECT id, salary FROM {fresh_table} ORDER BY id"
        ).collect()
        salaries = {row[0]: row[1] for row in result}
        assert salaries[1] == 95000.0  # 90000 + 5000
        assert salaries[2] == 85000.0  # 80000 + 5000
        assert salaries[3] == 70000.0  # unchanged


# ---------------------------------------------------------------------------
# MERGE INTO ... USING temp_view
# ---------------------------------------------------------------------------


class TestTempViewMerge:
    def test_merge_using_temp_view(self, session, fresh_table):
        # Source temp view: update existing + insert new
        source_data = pa.table({
            "id": pa.array([1, 6], type=pa.int32()),
            "name": pa.array(["Alice Updated", "Frank"], type=pa.string()),
            "salary": pa.array([95000.0, 55000.0], type=pa.float64()),
        })
        df = session.createDataFrame(source_data)
        df.createOrReplaceTempView("merge_source")

        session.sql(f"""
            MERGE INTO {fresh_table} t
            USING merge_source s
            ON t.id = s.id
            WHEN MATCHED THEN UPDATE SET name = s.name, salary = s.salary
            WHEN NOT MATCHED THEN INSERT (id, name, salary) VALUES (s.id, s.name, s.salary)
        """)

        result = session.sql(
            f"SELECT id, name, salary FROM {fresh_table} ORDER BY id"
        ).collect()
        rows = {row[0]: (row[1], row[2]) for row in result}
        assert len(rows) == 6
        assert rows[1] == ("Alice Updated", 95000.0)
        assert rows[6] == ("Frank", 55000.0)
        # Unmatched rows should be unchanged
        assert rows[2] == ("Bob", 80000.0)

    def test_merge_using_temp_view_delete(self, session, fresh_table):
        # Source temp view: delete matched rows
        source_data = pa.table({
            "id": pa.array([2, 4], type=pa.int32()),
        })
        df = session.createDataFrame(source_data)
        df.createOrReplaceTempView("merge_delete_source")

        session.sql(f"""
            MERGE INTO {fresh_table} t
            USING merge_delete_source s
            ON t.id = s.id
            WHEN MATCHED THEN DELETE
        """)

        result = session.sql(f"SELECT id FROM {fresh_table} ORDER BY id").collect()
        ids = [row[0] for row in result]
        assert ids == [1, 3, 5]


# ---------------------------------------------------------------------------
# SQL CREATE VIEW ... then DML
# ---------------------------------------------------------------------------


class TestSqlViewDml:
    def test_sql_create_view_then_delete(self, session, fresh_table):
        # Create a SQL view (not a DataFrame temp view)
        session.sql("CREATE OR REPLACE VIEW high_earners AS "
                     f"SELECT id FROM {fresh_table} WHERE salary > 75000")

        session.sql(
            f"DELETE FROM {fresh_table} WHERE id IN (SELECT id FROM high_earners)"
        )

        result = session.sql(f"SELECT * FROM {fresh_table}").collect()
        remaining_ids = sorted(row[0] for row in result)
        # Alice (90k) and Bob (80k) should be deleted; Carol, Dave, Eve remain
        assert remaining_ids == [3, 4, 5]


# ---------------------------------------------------------------------------
# Mixed: temp view + Iceberg table in same query
# ---------------------------------------------------------------------------


class TestMixedSources:
    def test_delete_mixed_temp_view_and_iceberg_table(self, session, fresh_table):
        # Create a second Iceberg table as a reference
        ref_tbl = "tv.ref_ids"
        session.sql(f"DROP TABLE IF EXISTS {ref_tbl}")
        session.sql(f"CREATE TABLE {ref_tbl} (id INT) USING iceberg")
        session.sql(f"INSERT INTO {ref_tbl} VALUES (1)")

        # Create a temp view
        tv_data = pa.table({"id": pa.array([5], type=pa.int32())})
        df = session.createDataFrame(tv_data)
        df.createOrReplaceTempView("extra_ids")

        # Delete rows where id is in either source
        session.sql(
            f"DELETE FROM {fresh_table} WHERE "
            f"id IN (SELECT id FROM ref_ids) OR id IN (SELECT id FROM extra_ids)"
        )

        result = session.sql(f"SELECT id FROM {fresh_table} ORDER BY id").collect()
        ids = [row[0] for row in result]
        assert ids == [2, 3, 4]

        session.sql(f"DROP TABLE IF EXISTS {ref_tbl}")
