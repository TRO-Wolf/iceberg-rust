"""Tests for ALTER TABLE SET/UNSET TBLPROPERTIES support."""

from __future__ import annotations

import pytest

from iceberg_spark.catalog_ops import _parse_tblproperties, _parse_tblproperties_keys
from iceberg_spark.sql_preprocessor import CommandType, preprocess


# ---------------------------------------------------------------------------
# Unit tests: parsing helpers
# ---------------------------------------------------------------------------


class TestParseTblproperties:
    def test_single_quoted_pair(self):
        result = _parse_tblproperties("'key1'='value1'")
        assert result == {"key1": "value1"}

    def test_multiple_pairs(self):
        result = _parse_tblproperties("'key1'='value1', 'key2'='value2'")
        assert result == {"key1": "value1", "key2": "value2"}

    def test_dotted_key(self):
        result = _parse_tblproperties("'write.format.default'='parquet'")
        assert result == {"write.format.default": "parquet"}

    def test_unquoted_key(self):
        result = _parse_tblproperties("mykey='myvalue'")
        assert result == {"mykey": "myvalue"}

    def test_double_quoted_key(self):
        result = _parse_tblproperties('"write.format"=\'avro\'')
        assert result == {"write.format": "avro"}

    def test_empty_value(self):
        result = _parse_tblproperties("'key1'=''")
        assert result == {"key1": ""}

    def test_multiple_dotted_keys(self):
        result = _parse_tblproperties(
            "'write.format.default'='parquet', 'commit.retry.num'='4'"
        )
        assert result == {"write.format.default": "parquet", "commit.retry.num": "4"}

    def test_value_with_embedded_double_quotes(self):
        result = _parse_tblproperties("'key'='value with \"embedded\" quotes'")
        assert result == {"key": 'value with "embedded" quotes'}

    def test_double_quoted_value(self):
        result = _parse_tblproperties('"key"="value"')
        assert result == {"key": "value"}

    def test_double_quoted_value_with_embedded_single_quotes(self):
        result = _parse_tblproperties("""'key'="it's a value" """)
        assert result == {"key": "it's a value"}


class TestParseTblpropertiesKeys:
    def test_single_quoted_key(self):
        result = _parse_tblproperties_keys("'key1'")
        assert result == ["key1"]

    def test_multiple_keys(self):
        result = _parse_tblproperties_keys("'key1', 'key2'")
        assert result == ["key1", "key2"]

    def test_dotted_keys(self):
        result = _parse_tblproperties_keys("'write.format.default', 'commit.retry.num'")
        assert result == ["write.format.default", "commit.retry.num"]

    def test_unquoted_keys(self):
        result = _parse_tblproperties_keys("key1, key2")
        assert result == ["key1", "key2"]

    def test_keys_with_dashes(self):
        result = _parse_tblproperties_keys("'key-with-dash', 'another-key'")
        assert result == ["key-with-dash", "another-key"]

    def test_keys_quoted_with_dots_and_dashes(self):
        result = _parse_tblproperties_keys("'write.format-version'")
        assert result == ["write.format-version"]


# ---------------------------------------------------------------------------
# Unit tests: SQL preprocessor routing
# ---------------------------------------------------------------------------


class TestPreprocessAlterTableProperties:
    def test_set_tblproperties_routes_to_alter_table(self):
        result = preprocess(
            "ALTER TABLE db.t1 SET TBLPROPERTIES ('key1'='value1')"
        )
        assert result.command_type == CommandType.ALTER_TABLE
        assert result.table_name == "db.t1"

    def test_unset_tblproperties_routes_to_alter_table(self):
        result = preprocess(
            "ALTER TABLE db.t1 UNSET TBLPROPERTIES ('key1')"
        )
        assert result.command_type == CommandType.ALTER_TABLE
        assert result.table_name == "db.t1"

    def test_unset_tblproperties_if_exists_routes(self):
        result = preprocess(
            "ALTER TABLE db.t1 UNSET TBLPROPERTIES IF EXISTS ('key1')"
        )
        assert result.command_type == CommandType.ALTER_TABLE
        assert result.table_name == "db.t1"

    def test_set_tblproperties_case_insensitive(self):
        result = preprocess(
            "alter table db.t1 set tblproperties ('a'='b')"
        )
        assert result.command_type == CommandType.ALTER_TABLE

    def test_set_multiple_properties(self):
        result = preprocess(
            "ALTER TABLE t1 SET TBLPROPERTIES ('k1'='v1', 'k2'='v2')"
        )
        assert result.command_type == CommandType.ALTER_TABLE
        assert result.table_name == "t1"
