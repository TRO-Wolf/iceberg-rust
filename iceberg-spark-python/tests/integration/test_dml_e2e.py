"""Integration tests for Phase 4 DML: DELETE FROM and UPDATE.

Requires PyIceberg + SQLite catalog.  Skipped automatically when unavailable.

Run with:
    uv run pytest tests/integration/test_dml_e2e.py -v
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
    reason="pyiceberg[sql-sqlite] / sqlalchemy not installed — skipping DML integration tests",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def warehouse():
    with tempfile.TemporaryDirectory(prefix="iceberg_dml_test_") as d:
        yield d


@pytest.fixture(scope="module")
def session(warehouse):
    db_path = os.path.join(warehouse, "catalog.db")
    sess = (
        IcebergSession.builder()
        .catalog(
            "sql",
            name="dml_catalog",
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
        session.sql("CREATE DATABASE IF NOT EXISTS dml")
    except Exception:
        pass


@pytest.fixture
def fresh_table(session):
    """Creates a fresh table with 5 rows; drops it after the test."""
    tbl = "dml.employees"
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
# DELETE FROM tests
# ---------------------------------------------------------------------------


class TestDeleteFrom:
    def test_delete_with_where(self, session, fresh_table):
        session.sql(f"DELETE FROM {fresh_table} WHERE id > 3")
        result = session.sql(f"SELECT * FROM {fresh_table}").collect()
        ids = [row[0] for row in result]
        assert sorted(ids) == [1, 2, 3]

    def test_delete_all(self, session, fresh_table):
        session.sql(f"DELETE FROM {fresh_table}")
        result = session.sql(f"SELECT * FROM {fresh_table}").collect()
        assert len(result) == 0

    def test_delete_no_matching_rows(self, session, fresh_table):
        """DELETE with WHERE that matches nothing should leave table unchanged."""
        session.sql(f"DELETE FROM {fresh_table} WHERE id > 1000")
        result = session.sql(f"SELECT * FROM {fresh_table}").collect()
        assert len(result) == 5

    def test_delete_returns_dataframe(self, session, fresh_table):
        df = session.sql(f"DELETE FROM {fresh_table} WHERE id = 1")
        assert df is not None

    def test_delete_table_not_found(self, session):
        with pytest.raises(RuntimeError, match="nonexistent_table"):
            session.sql("DELETE FROM dml.nonexistent_table WHERE id = 1")


# ---------------------------------------------------------------------------
# UPDATE tests
# ---------------------------------------------------------------------------


class TestUpdate:
    def test_update_single_column_with_where(self, session, fresh_table):
        session.sql(f"UPDATE {fresh_table} SET salary = 99999.0 WHERE id = 1")
        result = session.sql(
            f"SELECT salary FROM {fresh_table} WHERE id = 1"
        ).collect()
        assert result[0][0] == pytest.approx(99999.0)
        # Other rows unchanged
        result2 = session.sql(
            f"SELECT salary FROM {fresh_table} WHERE id = 2"
        ).collect()
        assert result2[0][0] == pytest.approx(80000.0)

    def test_update_multiple_columns(self, session, fresh_table):
        session.sql(
            f"UPDATE {fresh_table} SET name = 'Updated', salary = 1.0 WHERE id = 3"
        )
        result = session.sql(
            f"SELECT name, salary FROM {fresh_table} WHERE id = 3"
        ).collect()
        assert result[0][0] == "Updated"
        assert result[0][1] == pytest.approx(1.0)

    def test_update_without_where(self, session, fresh_table):
        """UPDATE with no WHERE updates all rows."""
        session.sql(f"UPDATE {fresh_table} SET salary = 0.0")
        result = session.sql(f"SELECT salary FROM {fresh_table}").collect()
        assert all(row[0] == pytest.approx(0.0) for row in result)

    def test_update_arithmetic(self, session, fresh_table):
        """SET salary = salary * 2 WHERE id <= 2."""
        session.sql(f"UPDATE {fresh_table} SET salary = salary * 2 WHERE id <= 2")
        result = session.sql(
            f"SELECT id, salary FROM {fresh_table} ORDER BY id"
        ).collect()
        salaries = {row[0]: row[1] for row in result}
        assert salaries[1] == pytest.approx(180000.0)   # 90000 * 2
        assert salaries[2] == pytest.approx(160000.0)   # 80000 * 2
        assert salaries[3] == pytest.approx(70000.0)    # unchanged

    def test_update_returns_dataframe(self, session, fresh_table):
        df = session.sql(f"UPDATE {fresh_table} SET salary = 50000.0 WHERE id = 5")
        assert df is not None

    def test_update_table_not_found(self, session):
        with pytest.raises(RuntimeError, match="ghost_table"):
            session.sql("UPDATE dml.ghost_table SET col = 1")


# ---------------------------------------------------------------------------
# MERGE INTO tests
# ---------------------------------------------------------------------------


class TestMergeInto:
    @pytest.fixture(autouse=True)
    def setup_tables(self, session):
        """Creates target and source tables; drops after test."""
        target = "dml.merge_target"
        source = "dml.merge_source"
        session.sql(f"DROP TABLE IF EXISTS {target}")
        session.sql(f"DROP TABLE IF EXISTS {source}")
        session.sql(
            f"CREATE TABLE {target} (id INT, name STRING, salary DOUBLE) USING iceberg"
        )
        session.sql(
            f"CREATE TABLE {source} (id INT, name STRING, salary DOUBLE) USING iceberg"
        )
        session.sql(
            f"INSERT INTO {target} VALUES "
            "(1, 'Alice', 90000.0), (2, 'Bob', 80000.0), (3, 'Carol', 70000.0)"
        )
        session.sql(
            f"INSERT INTO {source} VALUES "
            "(2, 'Bobby', 85000.0), (4, 'Dave', 60000.0)"
        )
        self.target = target
        self.source = source
        yield
        session.sql(f"DROP TABLE IF EXISTS {target}")
        session.sql(f"DROP TABLE IF EXISTS {source}")

    def test_merge_update_and_insert(self, session):
        session.sql(
            f"MERGE INTO {self.target} t USING {self.source} s ON t.id = s.id "
            "WHEN MATCHED THEN UPDATE SET t.name = s.name, t.salary = s.salary "
            "WHEN NOT MATCHED THEN INSERT (id, name, salary) VALUES (s.id, s.name, s.salary)"
        )
        result = session.sql(
            f"SELECT * FROM {self.target} ORDER BY id"
        ).collect()
        ids = [r[0] for r in result]
        names = [r[1] for r in result]
        assert ids == [1, 2, 3, 4]
        assert names[0] == "Alice"   # unchanged
        assert names[1] == "Bobby"   # updated
        assert names[2] == "Carol"   # unchanged
        assert names[3] == "Dave"    # inserted

    def test_merge_delete(self, session):
        session.sql(
            f"MERGE INTO {self.target} t USING {self.source} s ON t.id = s.id "
            "WHEN MATCHED THEN DELETE"
        )
        result = session.sql(
            f"SELECT id FROM {self.target} ORDER BY id"
        ).collect()
        ids = [r[0] for r in result]
        assert ids == [1, 3]  # id=2 matched and deleted

    def test_merge_returns_dataframe(self, session):
        df = session.sql(
            f"MERGE INTO {self.target} t USING {self.source} s ON t.id = s.id "
            "WHEN MATCHED THEN UPDATE SET t.name = s.name"
        )
        assert df is not None

    def test_merge_source_not_found(self, session):
        with pytest.raises(RuntimeError, match="ghost"):
            session.sql(
                f"MERGE INTO {self.target} t USING dml.ghost s ON t.id = s.id "
                "WHEN MATCHED THEN DELETE"
            )


# ---------------------------------------------------------------------------
# Advanced MERGE INTO tests
# ---------------------------------------------------------------------------


class TestAdvancedMerge:
    """Advanced MERGE INTO scenarios: conditional clauses, delete-only, insert-only."""

    @pytest.fixture(autouse=True)
    def setup_tables(self, session):
        """Set up target and source tables; clean up after each test."""
        self.target = "dml.adv_target"
        self.source = "dml.adv_source"
        session.sql(f"DROP TABLE IF EXISTS {self.target}")
        session.sql(f"DROP TABLE IF EXISTS {self.source}")
        session.sql(
            f"CREATE TABLE {self.target} "
            "(id INT, name STRING, salary DOUBLE) USING iceberg"
        )
        session.sql(
            f"CREATE TABLE {self.source} "
            "(id INT, name STRING, salary DOUBLE) USING iceberg"
        )
        yield
        session.sql(f"DROP TABLE IF EXISTS {self.target}")
        session.sql(f"DROP TABLE IF EXISTS {self.source}")

    def test_merge_conditional_update_and_delete(self, session):
        """MERGE with conditional WHEN MATCHED clauses (both have conditions).

        Target: (1, Alice, 50000), (2, Bob, 70000), (3, Carol, 90000)
        Source: (2, Bob, 85000), (3, Carol, 95000), (4, Dave, 60000)

        Rules:
        - WHEN MATCHED AND t.salary < 80000 THEN UPDATE SET t.salary = s.salary
        - WHEN MATCHED AND t.salary >= 80000 THEN DELETE
        - WHEN NOT MATCHED THEN INSERT

        Expected results:
        - id=1: unchanged (not matched)
        - id=2: updated to 85000 (matched, salary 70000 < 80000 -> update)
        - id=3: deleted (matched, salary 90000 >= 80000 -> delete)
        - id=4: inserted (not matched in target)
        """
        session.sql(
            f"INSERT INTO {self.target} VALUES "
            "(1, 'Alice', 50000.0), (2, 'Bob', 70000.0), (3, 'Carol', 90000.0)"
        )
        session.sql(
            f"INSERT INTO {self.source} VALUES "
            "(2, 'Bob', 85000.0), (3, 'Carol', 95000.0), (4, 'Dave', 60000.0)"
        )

        session.sql(
            f"MERGE INTO {self.target} t USING {self.source} s ON t.id = s.id "
            "WHEN MATCHED AND t.salary < 80000 THEN UPDATE SET t.salary = s.salary "
            "WHEN MATCHED AND t.salary >= 80000 THEN DELETE "
            "WHEN NOT MATCHED THEN INSERT (id, name, salary) VALUES (s.id, s.name, s.salary)"
        )

        result = session.sql(
            f"SELECT * FROM {self.target} ORDER BY id"
        ).collect()
        ids = [r[0] for r in result]
        assert 1 in ids, "id=1 should remain (unmatched)"
        assert 2 in ids, "id=2 should remain (updated)"
        assert 3 not in ids, "id=3 should be deleted (matched, salary >= 80000)"
        assert 4 in ids, "id=4 should be inserted"

        # Verify id=2 was updated
        salaries = {r[0]: r[2] for r in result}
        assert salaries[2] == pytest.approx(85000.0), (
            f"id=2 salary should be 85000, got {salaries[2]}"
        )
        # Verify id=1 unchanged
        assert salaries[1] == pytest.approx(50000.0)

    def test_merge_delete_only(self, session):
        """MERGE with only WHEN MATCHED THEN DELETE.

        Target: (1, Alice, 50000), (2, Bob, 70000), (3, Carol, 90000)
        Source: (2, Bob, 85000), (3, Carol, 95000)

        Expected: id=2 and id=3 deleted, id=1 remains.
        """
        session.sql(
            f"INSERT INTO {self.target} VALUES "
            "(1, 'Alice', 50000.0), (2, 'Bob', 70000.0), (3, 'Carol', 90000.0)"
        )
        session.sql(
            f"INSERT INTO {self.source} VALUES "
            "(2, 'Bob', 85000.0), (3, 'Carol', 95000.0)"
        )

        session.sql(
            f"MERGE INTO {self.target} t USING {self.source} s ON t.id = s.id "
            "WHEN MATCHED THEN DELETE"
        )

        result = session.sql(
            f"SELECT * FROM {self.target} ORDER BY id"
        ).collect()
        ids = [r[0] for r in result]
        assert ids == [1], f"Expected only id=1 remaining, got {ids}"

    def test_merge_no_matches_insert_only(self, session):
        """MERGE where target and source share no IDs — only INSERT fires.

        Target: (1, Alice, 50000), (2, Bob, 70000)
        Source: (10, Xander, 100000), (11, Yara, 110000)

        Expected: all 4 rows present (originals + inserts).
        """
        session.sql(
            f"INSERT INTO {self.target} VALUES "
            "(1, 'Alice', 50000.0), (2, 'Bob', 70000.0)"
        )
        session.sql(
            f"INSERT INTO {self.source} VALUES "
            "(10, 'Xander', 100000.0), (11, 'Yara', 110000.0)"
        )

        session.sql(
            f"MERGE INTO {self.target} t USING {self.source} s ON t.id = s.id "
            "WHEN MATCHED THEN UPDATE SET t.salary = s.salary "
            "WHEN NOT MATCHED THEN INSERT (id, name, salary) VALUES (s.id, s.name, s.salary)"
        )

        result = session.sql(
            f"SELECT * FROM {self.target} ORDER BY id"
        ).collect()
        ids = [r[0] for r in result]
        assert ids == [1, 2, 10, 11], f"Expected [1, 2, 10, 11], got {ids}"
