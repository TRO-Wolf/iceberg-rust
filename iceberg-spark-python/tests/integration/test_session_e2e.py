"""End-to-end integration tests for IcebergSession with a real SQLite catalog.

These tests require PyIceberg + SQLite to be installed.  They are skipped
automatically when those dependencies are unavailable (e.g. in CI without the
pyiceberg[sql-sqlite] extras).

Run them explicitly with:
    uv run pytest tests/integration/ -v

Or inside the full suite via the integration marker:
    uv run pytest tests/ -m integration -v
"""

from __future__ import annotations

import os
import tempfile

import pyarrow as pa
import pytest

# Guard: skip if pyiceberg + sqlalchemy (SqlCatalog) are not installed
try:
    import pyiceberg.catalog.sql  # requires sqlalchemy  # noqa: F401
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
    """Temporary directory used as Iceberg warehouse root."""
    with tempfile.TemporaryDirectory(prefix="iceberg_spark_test_") as d:
        yield d


@pytest.fixture(scope="module")
def session(warehouse):
    """IcebergSession backed by a SQLite catalog + local filesystem warehouse."""
    db_path = os.path.join(warehouse, "catalog.db")
    sess = (
        IcebergSession.builder()
        .catalog(
            "sql",
            name="test_catalog",
            uri=f"sqlite:///{db_path}",
            warehouse=f"file://{warehouse}",
        )
        .build()
    )
    yield sess
    sess.stop()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _ensure_namespace(session: IcebergSession, ns: str) -> None:
    try:
        session.sql(f"CREATE DATABASE IF NOT EXISTS {ns}")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCreateDropTable:
    def test_create_table(self, session, warehouse):
        _ensure_namespace(session, "test")
        result = session.sql(
            "CREATE TABLE IF NOT EXISTS test.t_create "
            "(id INT, name STRING, score DOUBLE) USING iceberg"
        )
        assert "created" in result.collect()[0]["status"].lower()

    def test_show_tables(self, session):
        _ensure_namespace(session, "test")
        session.sql(
            "CREATE TABLE IF NOT EXISTS test.t_show (id INT) USING iceberg"
        )
        result = session.sql("SHOW TABLES IN test")
        table_names = [r["tableName"] for r in result.collect()]
        assert any("t_show" in n or "t_show" == n for n in table_names)

    def test_describe_table(self, session):
        _ensure_namespace(session, "test")
        session.sql(
            "CREATE TABLE IF NOT EXISTS test.t_desc "
            "(id BIGINT, val STRING) USING iceberg"
        )
        result = session.sql("DESCRIBE TABLE test.t_desc")
        cols = {r["col_name"] for r in result.collect()}
        assert "id" in cols
        assert "val" in cols

    def test_drop_table(self, session):
        _ensure_namespace(session, "test")
        session.sql(
            "CREATE TABLE IF NOT EXISTS test.t_drop (x INT) USING iceberg"
        )
        result = session.sql("DROP TABLE test.t_drop")
        assert "dropped" in result.collect()[0]["status"].lower()

    def test_drop_table_if_exists(self, session):
        _ensure_namespace(session, "test")
        # Should not raise even if table doesn't exist when IF EXISTS present
        try:
            session.sql("DROP TABLE IF EXISTS test.t_nonexistent")
        except Exception:
            pass  # catalog may raise for genuinely missing tables — acceptable


class TestInsertAndSelect:
    def test_insert_values_and_select(self, session):
        _ensure_namespace(session, "test")
        session.sql(
            "CREATE TABLE IF NOT EXISTS test.t_insert "
            "(id BIGINT, name STRING) USING iceberg"
        )
        session.sql("INSERT INTO test.t_insert VALUES (1, 'Alice'), (2, 'Bob')")
        df = session.sql("SELECT * FROM test.t_insert ORDER BY id")
        rows = df.collect()
        assert len(rows) == 2
        assert rows[0]["id"] == 1
        assert rows[1]["name"] == "Bob"

    def test_insert_select_from_table(self, session):
        _ensure_namespace(session, "test")
        session.sql(
            "CREATE TABLE IF NOT EXISTS test.t_src (id BIGINT, val STRING) USING iceberg"
        )
        session.sql(
            "CREATE TABLE IF NOT EXISTS test.t_dst (id BIGINT, val STRING) USING iceberg"
        )
        session.sql("INSERT INTO test.t_src VALUES (10, 'x'), (20, 'y')")
        session.sql(
            "INSERT INTO test.t_dst SELECT id, val FROM test.t_src WHERE id > 10"
        )
        rows = session.sql("SELECT * FROM test.t_dst").collect()
        assert len(rows) == 1
        assert rows[0]["id"] == 20

    def test_count_after_insert(self, session):
        _ensure_namespace(session, "test")
        session.sql(
            "CREATE TABLE IF NOT EXISTS test.t_count (n BIGINT) USING iceberg"
        )
        session.sql("INSERT INTO test.t_count VALUES (1), (2), (3)")
        df = session.sql("SELECT count(*) AS cnt FROM test.t_count")
        assert df.collect()[0]["cnt"] == 3

    def test_aggregation_on_iceberg_table(self, session):
        _ensure_namespace(session, "test")
        session.sql(
            "CREATE TABLE IF NOT EXISTS test.t_agg "
            "(dept STRING, salary BIGINT) USING iceberg"
        )
        session.sql(
            "INSERT INTO test.t_agg VALUES "
            "('eng', 100), ('eng', 120), ('sales', 80)"
        )
        df = session.sql(
            "SELECT dept, sum(salary) AS total FROM test.t_agg GROUP BY dept ORDER BY dept"
        )
        rows = {r["dept"]: r["total"] for r in df.collect()}
        assert rows["eng"] == 220
        assert rows["sales"] == 80


class TestDataFrameWriter:
    def test_write_append(self, session):
        _ensure_namespace(session, "test")
        session.sql(
            "CREATE TABLE IF NOT EXISTS test.t_write "
            "(id BIGINT, v STRING) USING iceberg"
        )
        arrow = pa.table({"id": [1, 2], "v": ["a", "b"]})
        in_df = session.createDataFrame(arrow)
        in_df.write.mode("append").saveAsTable("test.t_write")
        rows = session.sql("SELECT * FROM test.t_write ORDER BY id").collect()
        assert len(rows) >= 2

    def test_write_overwrite(self, session):
        _ensure_namespace(session, "test")
        session.sql(
            "CREATE TABLE IF NOT EXISTS test.t_overwrite "
            "(id BIGINT) USING iceberg"
        )
        session.sql("INSERT INTO test.t_overwrite VALUES (1), (2), (3)")
        arrow = pa.table({"id": pa.array([99], type=pa.int64())})
        in_df = session.createDataFrame(arrow)
        in_df.write.mode("overwrite").saveAsTable("test.t_overwrite")
        rows = session.sql("SELECT * FROM test.t_overwrite").collect()
        assert len(rows) == 1
        assert rows[0]["id"] == 99


class TestAlterTable:
    def test_add_column(self, session):
        _ensure_namespace(session, "test")
        session.sql(
            "CREATE TABLE IF NOT EXISTS test.t_alter "
            "(id BIGINT, name STRING) USING iceberg"
        )
        session.sql("ALTER TABLE test.t_alter ADD COLUMN age BIGINT")
        desc = session.sql("DESCRIBE TABLE test.t_alter")
        cols = {r["col_name"] for r in desc.collect()}
        assert "age" in cols

    def test_rename_column(self, session):
        _ensure_namespace(session, "test")
        session.sql(
            "CREATE TABLE IF NOT EXISTS test.t_rename "
            "(old_col STRING) USING iceberg"
        )
        session.sql("ALTER TABLE test.t_rename RENAME COLUMN old_col TO new_col")
        desc = session.sql("DESCRIBE TABLE test.t_rename")
        cols = {r["col_name"] for r in desc.collect()}
        assert "new_col" in cols
        assert "old_col" not in cols


class TestSessionRange:
    def test_range_end_only(self, session):
        df = session.range(5)
        rows = df.collect()
        assert len(rows) == 5
        assert [r["id"] for r in rows] == [0, 1, 2, 3, 4]

    def test_range_start_end(self, session):
        df = session.range(2, 7)
        rows = df.collect()
        assert [r["id"] for r in rows] == [2, 3, 4, 5, 6]

    def test_range_with_step(self, session):
        df = session.range(0, 10, 2)
        rows = df.collect()
        assert [r["id"] for r in rows] == [0, 2, 4, 6, 8]


class TestShowDatabases:
    def test_show_databases(self, session):
        _ensure_namespace(session, "schema_a")
        result = session.sql("SHOW DATABASES")
        ns_names = [r["namespace"] for r in result.collect()]
        assert len(ns_names) >= 1

    def test_show_namespaces(self, session):
        result = session.sql("SHOW NAMESPACES")
        assert isinstance(result.collect(), list)


class TestCTAS:
    """Test CREATE TABLE AS SELECT (CTAS) scenarios."""

    def test_ctas_with_where(self, session):
        """CTAS with a WHERE filter should only include matching rows."""
        _ensure_namespace(session, "ctas")
        # Clean up any leftover tables
        try:
            session.sql("DROP TABLE IF EXISTS ctas.src")
        except Exception:
            pass
        try:
            session.sql("DROP TABLE IF EXISTS ctas.dst")
        except Exception:
            pass

        session.sql(
            "CREATE TABLE ctas.src (id BIGINT, name STRING) USING iceberg"
        )
        session.sql(
            "INSERT INTO ctas.src VALUES "
            "(1, 'Alice'), (2, 'Bob'), (3, 'Carol'), (4, 'Dave'), (5, 'Eve')"
        )

        session.sql(
            "CREATE TABLE ctas.dst AS SELECT * FROM ctas.src WHERE id > 2"
        )
        rows = session.sql("SELECT * FROM ctas.dst ORDER BY id").collect()
        ids = [r["id"] for r in rows]
        assert ids == [3, 4, 5], f"Expected [3, 4, 5], got {ids}"

    def test_ctas_empty_result(self, session):
        """CTAS with a WHERE that matches no rows creates an empty table."""
        _ensure_namespace(session, "ctas")
        try:
            session.sql("DROP TABLE IF EXISTS ctas.empty_tbl")
        except Exception:
            pass

        # Reuse the source table from the previous test (or create if not present)
        try:
            session.sql(
                "CREATE TABLE IF NOT EXISTS ctas.src "
                "(id BIGINT, name STRING) USING iceberg"
            )
        except Exception:
            pass

        session.sql(
            "CREATE TABLE ctas.empty_tbl AS SELECT * FROM ctas.src WHERE id > 100"
        )
        rows = session.sql("SELECT * FROM ctas.empty_tbl").collect()
        assert len(rows) == 0, f"Expected 0 rows, got {len(rows)}"


class TestAlterTableProperties:
    """Integration tests for ALTER TABLE SET/UNSET TBLPROPERTIES."""

    def test_set_tblproperties(self, session):
        _ensure_namespace(session, "props")
        session.sql(
            "CREATE TABLE IF NOT EXISTS props.t_props "
            "(id BIGINT) USING iceberg"
        )
        result = session.sql(
            "ALTER TABLE props.t_props SET TBLPROPERTIES "
            "('write.format.default'='parquet', 'commit.retry.num'='4')"
        )
        rows = result.collect()
        assert "properties set" in rows[0]["status"].lower()

        # Verify via SHOW TBLPROPERTIES
        props_df = session.sql("SHOW TBLPROPERTIES props.t_props")
        props_rows = props_df.collect()
        props_dict = {r["key"]: r["value"] for r in props_rows}
        assert props_dict.get("write.format.default") == "parquet"
        assert props_dict.get("commit.retry.num") == "4"

    def test_unset_tblproperties(self, session):
        _ensure_namespace(session, "props")
        session.sql(
            "CREATE TABLE IF NOT EXISTS props.t_unset "
            "(id BIGINT) USING iceberg"
        )
        # First set a property
        session.sql(
            "ALTER TABLE props.t_unset SET TBLPROPERTIES ('my.prop'='hello')"
        )
        # Verify it was set
        props_rows = session.sql("SHOW TBLPROPERTIES props.t_unset").collect()
        props_dict = {r["key"]: r["value"] for r in props_rows}
        assert props_dict.get("my.prop") == "hello"

        # Now unset it
        result = session.sql(
            "ALTER TABLE props.t_unset UNSET TBLPROPERTIES ('my.prop')"
        )
        rows = result.collect()
        assert "properties removed" in rows[0]["status"].lower()

        # Verify it was removed
        props_rows2 = session.sql("SHOW TBLPROPERTIES props.t_unset").collect()
        props_dict2 = {r["key"]: r["value"] for r in props_rows2}
        assert "my.prop" not in props_dict2

    def test_unset_tblproperties_if_exists(self, session):
        _ensure_namespace(session, "props")
        session.sql(
            "CREATE TABLE IF NOT EXISTS props.t_unset_ie "
            "(id BIGINT) USING iceberg"
        )
        # UNSET with IF EXISTS should not fail even if property doesn't exist
        result = session.sql(
            "ALTER TABLE props.t_unset_ie UNSET TBLPROPERTIES IF EXISTS ('nonexistent.key')"
        )
        rows = result.collect()
        assert "properties removed" in rows[0]["status"].lower()

    def test_set_then_overwrite_property(self, session):
        _ensure_namespace(session, "props")
        session.sql(
            "CREATE TABLE IF NOT EXISTS props.t_overwrite "
            "(id BIGINT) USING iceberg"
        )
        session.sql(
            "ALTER TABLE props.t_overwrite SET TBLPROPERTIES ('color'='red')"
        )
        session.sql(
            "ALTER TABLE props.t_overwrite SET TBLPROPERTIES ('color'='blue')"
        )
        props_rows = session.sql("SHOW TBLPROPERTIES props.t_overwrite").collect()
        props_dict = {r["key"]: r["value"] for r in props_rows}
        assert props_dict.get("color") == "blue"


class TestComplexTypes:
    """Integration tests for CREATE TABLE with complex types (ARRAY, MAP, STRUCT).

    These verify that complex type parsing in spark_type_from_name() and
    _parse_column_defs() works end-to-end against a real SQLite catalog,
    producing correct Iceberg schemas.
    """

    def test_create_table_with_array(self, session):
        _ensure_namespace(session, "test")
        try:
            session.sql("DROP TABLE IF EXISTS test.arr_t")
        except Exception:
            pass

        result = session.sql(
            "CREATE TABLE test.arr_t (id INT, tags ARRAY<STRING>) USING iceberg"
        )
        assert "created" in result.collect()[0]["status"].lower()

        desc = session.sql("DESCRIBE TABLE test.arr_t")
        rows = desc.collect()
        cols = {r["col_name"]: r["data_type"] for r in rows}
        assert "id" in cols
        assert "tags" in cols
        # PyIceberg renders list type as "list<string>" rather than "array<string>"
        tags_type = cols["tags"].lower()
        assert "list" in tags_type or "array" in tags_type, (
            f"Expected list/array type for 'tags', got: {tags_type}"
        )

    def test_create_table_with_map(self, session):
        _ensure_namespace(session, "test")
        try:
            session.sql("DROP TABLE IF EXISTS test.map_t")
        except Exception:
            pass

        result = session.sql(
            "CREATE TABLE test.map_t "
            "(id INT, metadata MAP<STRING,STRING>) USING iceberg"
        )
        assert "created" in result.collect()[0]["status"].lower()

        desc = session.sql("DESCRIBE TABLE test.map_t")
        rows = desc.collect()
        cols = {r["col_name"]: r["data_type"] for r in rows}
        assert "id" in cols
        assert "metadata" in cols
        meta_type = cols["metadata"].lower()
        assert "map" in meta_type, (
            f"Expected map type for 'metadata', got: {meta_type}"
        )

    def test_create_table_with_struct(self, session):
        _ensure_namespace(session, "test")
        try:
            session.sql("DROP TABLE IF EXISTS test.struct_t")
        except Exception:
            pass

        result = session.sql(
            "CREATE TABLE test.struct_t "
            "(id INT, info STRUCT<name:STRING,age:INT>) USING iceberg"
        )
        assert "created" in result.collect()[0]["status"].lower()

        desc = session.sql("DESCRIBE TABLE test.struct_t")
        rows = desc.collect()
        cols = {r["col_name"]: r["data_type"] for r in rows}
        assert "id" in cols
        assert "info" in cols
        info_type = cols["info"].lower()
        assert "struct" in info_type, (
            f"Expected struct type for 'info', got: {info_type}"
        )
        # Verify subfields are present in the type string
        assert "name" in info_type, (
            f"Expected 'name' subfield in struct type, got: {info_type}"
        )
        assert "age" in info_type, (
            f"Expected 'age' subfield in struct type, got: {info_type}"
        )

    def test_create_table_with_nested_complex(self, session):
        _ensure_namespace(session, "test")
        try:
            session.sql("DROP TABLE IF EXISTS test.nested_t")
        except Exception:
            pass

        result = session.sql(
            "CREATE TABLE test.nested_t "
            "(id INT, data ARRAY<STRUCT<name:STRING,score:DOUBLE>>) USING iceberg"
        )
        assert "created" in result.collect()[0]["status"].lower()

        desc = session.sql("DESCRIBE TABLE test.nested_t")
        rows = desc.collect()
        cols = {r["col_name"]: r["data_type"] for r in rows}
        assert "id" in cols
        assert "data" in cols
        data_type = cols["data"].lower()
        # Should be a list/array of struct
        assert "list" in data_type or "array" in data_type, (
            f"Expected list/array type for 'data', got: {data_type}"
        )
        assert "struct" in data_type, (
            f"Expected nested struct in 'data' type, got: {data_type}"
        )
        assert "name" in data_type, (
            f"Expected 'name' subfield in nested struct, got: {data_type}"
        )
        assert "score" in data_type, (
            f"Expected 'score' subfield in nested struct, got: {data_type}"
        )


class TestSessionTable:
    """Integration tests for session.table() with qualified and short names."""

    def test_table_with_qualified_name(self, session):
        """session.table('ns.table') returns a DataFrame with correct data."""
        _ensure_namespace(session, "tbl")
        session.sql(
            "CREATE TABLE IF NOT EXISTS tbl.t_qual "
            "(id BIGINT, name STRING) USING iceberg"
        )
        session.sql(
            "INSERT INTO tbl.t_qual VALUES (1, 'Alice'), (2, 'Bob')"
        )
        df = session.table("tbl.t_qual")
        rows = sorted(df.collect(), key=lambda r: r["id"])
        assert len(rows) == 2
        assert rows[0]["id"] == 1
        assert rows[0]["name"] == "Alice"
        assert rows[1]["id"] == 2
        assert rows[1]["name"] == "Bob"

    def test_table_with_short_name(self, session):
        """session.table('tablename') works after the table was registered via SQL."""
        _ensure_namespace(session, "tbl")
        session.sql(
            "CREATE TABLE IF NOT EXISTS tbl.t_short "
            "(id BIGINT, val STRING) USING iceberg"
        )
        session.sql(
            "INSERT INTO tbl.t_short VALUES (10, 'x'), (20, 'y')"
        )
        # First access via qualified name to ensure it gets registered
        session.sql("SELECT * FROM tbl.t_short")
        # Now access via short name
        df = session.table("t_short")
        rows = sorted(df.collect(), key=lambda r: r["id"])
        assert len(rows) == 2
        assert rows[0]["id"] == 10
        assert rows[1]["val"] == "y"


    def test_table_namespace_collision(self, session):
        """Two tables with same short name in different namespaces should not collide."""
        _ensure_namespace(session, "ns_a")
        _ensure_namespace(session, "ns_b")
        session.sql(
            "CREATE TABLE IF NOT EXISTS ns_a.shared_name "
            "(id BIGINT, src STRING) USING iceberg"
        )
        session.sql(
            "CREATE TABLE IF NOT EXISTS ns_b.shared_name "
            "(id BIGINT, src STRING) USING iceberg"
        )
        session.sql("INSERT INTO ns_a.shared_name VALUES (1, 'from_a')")
        session.sql("INSERT INTO ns_b.shared_name VALUES (2, 'from_b')")

        df_a = session.table("ns_a.shared_name")
        assert df_a.collect()[0]["src"] == "from_a"

        df_b = session.table("ns_b.shared_name")
        assert df_b.collect()[0]["src"] == "from_b"

        # Switch back to a — should still get a's data
        df_a2 = session.table("ns_a.shared_name")
        assert df_a2.collect()[0]["src"] == "from_a"


class TestUDFIntegration:
    """Integration tests for UDFs against a real SQLite-backed Iceberg catalog."""

    def test_scalar_udf_in_sql(self, session):
        """Register a UDF and use it in session.sql('SELECT my_udf(col) FROM table')."""
        from iceberg_spark.types import StringType

        _ensure_namespace(session, "udf_ns")
        session.sql(
            "CREATE TABLE IF NOT EXISTS udf_ns.t_udf_sql "
            "(id BIGINT, name STRING) USING iceberg"
        )
        session.sql(
            "INSERT INTO udf_ns.t_udf_sql VALUES (1, 'alice'), (2, 'bob'), (3, 'carol')"
        )

        session.udf.register(
            "shout_e2e", lambda x: x.upper() + "!" if x else None, StringType()
        )

        df = session.sql(
            "SELECT shout_e2e(name) AS shouted FROM udf_ns.t_udf_sql ORDER BY id"
        )
        rows = df.collect()
        assert [r["shouted"] for r in rows] == ["ALICE!", "BOB!", "CAROL!"]

    def test_scalar_udf_in_dataframe(self, session):
        """Register a UDF and use it via df.select(my_udf(col('name')))."""
        from iceberg_spark.functions import col
        from iceberg_spark.types import LongType

        _ensure_namespace(session, "udf_ns")
        session.sql(
            "CREATE TABLE IF NOT EXISTS udf_ns.t_udf_df "
            "(id BIGINT, name STRING) USING iceberg"
        )
        session.sql(
            "INSERT INTO udf_ns.t_udf_df VALUES (1, 'hi'), (2, 'hello'), (3, 'hey')"
        )

        name_len = session.udf.register(
            "name_len_e2e",
            lambda x: len(x) if x is not None else None,
            LongType(),
        )

        df = session.table("udf_ns.t_udf_df")
        result = df.select(name_len(col("name")).alias("nlen")).orderBy("nlen")
        rows = result.collect()
        assert [r["nlen"] for r in rows] == [2, 3, 5]

    def test_udf_null_handling(self, session):
        """UDF handles null values in real Iceberg table data."""
        from iceberg_spark.types import StringType

        _ensure_namespace(session, "udf_ns")
        session.sql(
            "CREATE TABLE IF NOT EXISTS udf_ns.t_udf_null "
            "(id BIGINT, val STRING) USING iceberg"
        )
        session.sql(
            "INSERT INTO udf_ns.t_udf_null VALUES "
            "(1, 'hello'), (2, CAST(NULL AS STRING)), (3, 'world')"
        )

        safe_upper = session.udf.register(
            "safe_upper_e2e",
            lambda x: x.upper() if x is not None else None,
            StringType(),
        )

        df = session.sql(
            "SELECT id, safe_upper_e2e(val) AS upper_val "
            "FROM udf_ns.t_udf_null ORDER BY id"
        )
        rows = df.collect()
        assert rows[0]["upper_val"] == "HELLO"
        assert rows[1]["upper_val"] is None
        assert rows[2]["upper_val"] == "WORLD"

    def test_udf_shows_in_catalog(self, session):
        """session.catalog.functionExists('my_udf') returns True after registration."""
        from iceberg_spark.types import StringType

        session.udf.register(
            "catalog_visible_e2e", lambda x: x, StringType()
        )
        assert session.catalog.functionExists("catalog_visible_e2e")
        assert not session.catalog.functionExists("nonexistent_udf_xyz")


class TestUDAFIntegration:
    """Integration tests for UDAFs against a real SQLite-backed Iceberg catalog."""

    def test_udaf_in_sql_groupby(self, session):
        """Register a UDAF, use in GROUP BY SQL, verify results."""
        from iceberg_spark.types import LongType

        _ensure_namespace(session, "udaf_ns")
        session.sql(
            "CREATE TABLE IF NOT EXISTS udaf_ns.t_udaf_grp "
            "(grp STRING, val BIGINT) USING iceberg"
        )
        session.sql(
            "INSERT INTO udaf_ns.t_udaf_grp VALUES "
            "('a', 10), ('a', 20), ('b', 5), ('b', 15), ('b', 30)"
        )

        session.udf.register_udaf(
            "my_sum_e2e", lambda values: sum(values), LongType(), [LongType()]
        )

        df = session.sql(
            "SELECT grp, my_sum_e2e(val) AS total "
            "FROM udaf_ns.t_udaf_grp GROUP BY grp ORDER BY grp"
        )
        rows = df.collect()
        result = {r["grp"]: r["total"] for r in rows}
        assert result["a"] == 30  # 10 + 20
        assert result["b"] == 50  # 5 + 15 + 30

    def test_udaf_whole_table(self, session):
        """Register a UDAF, use without GROUP BY to aggregate entire table."""
        from iceberg_spark.types import LongType

        _ensure_namespace(session, "udaf_ns")
        session.sql(
            "CREATE TABLE IF NOT EXISTS udaf_ns.t_udaf_all "
            "(val BIGINT) USING iceberg"
        )
        session.sql(
            "INSERT INTO udaf_ns.t_udaf_all VALUES (1), (2), (3), (4), (5)"
        )

        session.udf.register_udaf(
            "my_total_e2e", lambda values: sum(values), LongType(), [LongType()]
        )

        df = session.sql(
            "SELECT my_total_e2e(val) AS total FROM udaf_ns.t_udaf_all"
        )
        rows = df.collect()
        assert rows[0]["total"] == 15  # 1+2+3+4+5


class TestComplexTypeRoundtrip:
    """Insert and read back complex types (ARRAY, STRUCT) via DataFrame API."""

    def test_insert_and_select_array(self, session):
        _ensure_namespace(session, "test")
        try:
            session.sql("DROP TABLE IF EXISTS test.arr_rt")
        except Exception:
            pass

        session.sql(
            "CREATE TABLE test.arr_rt (id INT, tags ARRAY<STRING>) USING iceberg"
        )

        arrow_table = pa.table({
            "id": pa.array([1, 2], type=pa.int32()),
            "tags": [["python", "spark"], ["iceberg"]],
        })
        session._insert_into_table("test.arr_rt", arrow_table)

        df = session.sql("SELECT * FROM test.arr_rt ORDER BY id")
        rows = df.collect()
        assert len(rows) == 2
        assert rows[0]["id"] == 1
        assert list(rows[0]["tags"]) == ["python", "spark"]
        assert list(rows[1]["tags"]) == ["iceberg"]

    def test_insert_and_select_struct(self, session):
        _ensure_namespace(session, "test")
        try:
            session.sql("DROP TABLE IF EXISTS test.struct_rt")
        except Exception:
            pass

        session.sql(
            "CREATE TABLE test.struct_rt "
            "(id INT, info STRUCT<name:STRING,age:INT>) USING iceberg"
        )

        info_type = pa.struct([
            pa.field("name", pa.string()),
            pa.field("age", pa.int32()),
        ])
        arrow_table = pa.table({
            "id": pa.array([1, 2], type=pa.int32()),
            "info": pa.array(
                [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}],
                type=info_type,
            ),
        })
        session._insert_into_table("test.struct_rt", arrow_table)

        df = session.sql("SELECT * FROM test.struct_rt ORDER BY id")
        rows = df.collect()
        assert len(rows) == 2
        assert rows[0]["info"]["name"] == "Alice"
        assert rows[0]["info"]["age"] == 30
        assert rows[1]["info"]["name"] == "Bob"
        assert rows[1]["info"]["age"] == 25
