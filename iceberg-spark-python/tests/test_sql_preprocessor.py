"""Tests for the Spark SQL dialect preprocessor."""

from iceberg_spark.sql_preprocessor import CommandType, preprocess


def test_passthrough_select():
    result = preprocess("SELECT * FROM my_table WHERE id > 5")
    assert result.command_type == CommandType.SQL
    assert "SELECT * FROM my_table WHERE id > 5" in result.sql


def test_strip_using_iceberg():
    result = preprocess("CREATE TABLE t1 (id INT) USING iceberg")
    # CREATE TABLE is intercepted as DDL
    assert result.command_type == CommandType.CREATE_TABLE


def test_nvl_to_coalesce():
    result = preprocess("SELECT NVL(a, b) FROM t")
    assert result.command_type == CommandType.SQL
    assert "COALESCE(a, b)" in result.sql
    assert "NVL" not in result.sql


def test_rlike_to_regex():
    result = preprocess("SELECT * FROM t WHERE name RLIKE '^A.*'")
    assert result.command_type == CommandType.SQL
    assert "~" in result.sql
    assert "RLIKE" not in result.sql


def test_date_add_transform():
    result = preprocess("SELECT date_add(created, 7) FROM t")
    assert result.command_type == CommandType.SQL
    assert "interval '7 days'" in result.sql


def test_datediff_transform():
    result = preprocess("SELECT datediff(end_date, start_date) FROM t")
    assert result.command_type == CommandType.SQL
    assert "date_diff" in result.sql


def test_show_tables():
    result = preprocess("SHOW TABLES")
    assert result.command_type == CommandType.SHOW_TABLES
    assert result.namespace is None


def test_show_tables_in_namespace():
    result = preprocess("SHOW TABLES IN my_db")
    assert result.command_type == CommandType.SHOW_TABLES
    assert result.namespace == "my_db"


def test_show_databases():
    result = preprocess("SHOW DATABASES")
    assert result.command_type == CommandType.SHOW_DATABASES


def test_show_namespaces():
    result = preprocess("SHOW NAMESPACES")
    assert result.command_type == CommandType.SHOW_DATABASES


def test_describe_table():
    result = preprocess("DESCRIBE TABLE my_table")
    assert result.command_type == CommandType.DESCRIBE
    assert result.table_name == "my_table"


def test_describe_short():
    result = preprocess("DESC my_table")
    assert result.command_type == CommandType.DESCRIBE
    assert result.table_name == "my_table"


def test_create_table():
    result = preprocess("CREATE TABLE db.t1 (id INT, name STRING) USING iceberg")
    assert result.command_type == CommandType.CREATE_TABLE
    assert result.table_name == "db.t1"


def test_create_table_if_not_exists():
    result = preprocess("CREATE TABLE IF NOT EXISTS db.t1 (id INT)")
    assert result.command_type == CommandType.CREATE_TABLE
    assert result.table_name == "db.t1"


def test_drop_table():
    result = preprocess("DROP TABLE db.t1")
    assert result.command_type == CommandType.DROP_TABLE
    assert result.table_name == "db.t1"


def test_drop_table_if_exists():
    result = preprocess("DROP TABLE IF EXISTS db.t1")
    assert result.command_type == CommandType.DROP_TABLE
    assert result.table_name == "db.t1"


def test_create_namespace():
    result = preprocess("CREATE DATABASE my_db")
    assert result.command_type == CommandType.CREATE_NAMESPACE
    assert result.namespace == "my_db"


def test_create_schema():
    result = preprocess("CREATE SCHEMA my_schema")
    assert result.command_type == CommandType.CREATE_NAMESPACE
    assert result.namespace == "my_schema"


def test_drop_namespace():
    result = preprocess("DROP DATABASE my_db")
    assert result.command_type == CommandType.DROP_NAMESPACE
    assert result.namespace == "my_db"


def test_alter_table():
    result = preprocess("ALTER TABLE t1 ADD COLUMN new_col STRING")
    assert result.command_type == CommandType.ALTER_TABLE
    assert result.table_name == "t1"


def test_timestamp_as_of():
    result = preprocess("SELECT * FROM my_table TIMESTAMP AS OF '2024-01-01T00:00:00'")
    assert result.command_type == CommandType.TIME_TRAVEL
    assert result.table_name == "my_table"
    assert result.timestamp == "2024-01-01T00:00:00"
    assert "my_table__time_travel" in result.sql


def test_version_as_of():
    result = preprocess("SELECT * FROM my_table VERSION AS OF 12345")
    assert result.command_type == CommandType.TIME_TRAVEL
    assert result.table_name == "my_table"
    assert result.snapshot_id == 12345
    assert "my_table__time_travel" in result.sql


def test_tblproperties_to_options():
    result = preprocess("SELECT * FROM t TBLPROPERTIES ('a'='b')")
    assert result.command_type == CommandType.SQL
    assert "OPTIONS" in result.sql
    assert "TBLPROPERTIES" not in result.sql


def test_semicolon_stripped():
    result = preprocess("SELECT * FROM t;")
    assert result.sql == "SELECT * FROM t"


def test_case_insensitive():
    result = preprocess("show tables")
    assert result.command_type == CommandType.SHOW_TABLES

    result = preprocess("SHOW TABLES")
    assert result.command_type == CommandType.SHOW_TABLES

    result = preprocess("Show Tables")
    assert result.command_type == CommandType.SHOW_TABLES


# --- Task 9A: CREATE VIEW, DROP VIEW, SHOW COLUMNS, EXPLAIN ---


def test_create_view():
    result = preprocess("CREATE VIEW v1 AS SELECT * FROM t1")
    assert result.command_type == CommandType.CREATE_VIEW
    assert result.table_name == "v1"
    assert result.extra["view_query"] == "SELECT * FROM t1"
    assert result.extra["or_replace"] is False


def test_create_or_replace_view():
    result = preprocess("CREATE OR REPLACE VIEW v1 AS SELECT id FROM t1")
    assert result.command_type == CommandType.CREATE_VIEW
    assert result.table_name == "v1"
    assert result.extra["view_query"] == "SELECT id FROM t1"
    assert result.extra["or_replace"] is True


def test_create_temp_view():
    result = preprocess("CREATE TEMPORARY VIEW tmp_v AS SELECT 1")
    assert result.command_type == CommandType.CREATE_VIEW
    assert result.table_name == "tmp_v"


def test_create_temp_view_short():
    result = preprocess("CREATE TEMP VIEW tmp_v AS SELECT 1")
    assert result.command_type == CommandType.CREATE_VIEW
    assert result.table_name == "tmp_v"


def test_drop_view():
    result = preprocess("DROP VIEW v1")
    assert result.command_type == CommandType.DROP_VIEW
    assert result.table_name == "v1"


def test_drop_view_if_exists():
    result = preprocess("DROP VIEW IF EXISTS v1")
    assert result.command_type == CommandType.DROP_VIEW
    assert result.table_name == "v1"


def test_show_columns_from():
    result = preprocess("SHOW COLUMNS FROM db.t1")
    assert result.command_type == CommandType.SHOW_COLUMNS
    assert result.table_name == "db.t1"


def test_show_columns_in():
    result = preprocess("SHOW COLUMNS IN my_table")
    assert result.command_type == CommandType.SHOW_COLUMNS
    assert result.table_name == "my_table"


def test_explain():
    result = preprocess("EXPLAIN SELECT * FROM t1")
    assert result.command_type == CommandType.EXPLAIN
    assert result.sql == "SELECT * FROM t1"


def test_explain_with_where():
    result = preprocess("EXPLAIN SELECT * FROM t1 WHERE id > 5")
    assert result.command_type == CommandType.EXPLAIN
    assert result.sql == "SELECT * FROM t1 WHERE id > 5"


def test_explain_case_insensitive():
    result = preprocess("explain select * from t1")
    assert result.command_type == CommandType.EXPLAIN
