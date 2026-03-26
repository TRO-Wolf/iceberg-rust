"""Tests for Sprint 3 Task 3.1: SQL command stubs."""

from __future__ import annotations

import pytest

from iceberg_spark.sql_preprocessor import CommandType, preprocess


class TestSetConfig:
    def test_set_key_value(self):
        result = preprocess("SET spark.sql.shuffle.partitions=200")
        assert result.command_type == CommandType.SET_CONFIG
        assert result.extra["key"] == "spark.sql.shuffle.partitions"
        assert result.extra["value"] == "200"

    def test_set_string_value(self):
        result = preprocess("SET spark.app.name = my_app")
        assert result.command_type == CommandType.SET_CONFIG
        assert result.extra["key"] == "spark.app.name"
        assert result.extra["value"] == "my_app"

    def test_set_all(self):
        result = preprocess("SET")
        assert result.command_type == CommandType.SET_CONFIG
        assert result.extra["key"] is None


class TestUseDatabase:
    def test_use_db(self):
        result = preprocess("USE mydb")
        assert result.command_type == CommandType.USE_DATABASE
        assert result.namespace == "mydb"

    def test_use_database_keyword(self):
        result = preprocess("USE DATABASE mydb")
        assert result.command_type == CommandType.USE_DATABASE
        assert result.namespace == "mydb"

    def test_use_namespace(self):
        result = preprocess("USE NAMESPACE myns")
        assert result.command_type == CommandType.USE_DATABASE
        assert result.namespace == "myns"


class TestShowCreateTable:
    def test_basic(self):
        result = preprocess("SHOW CREATE TABLE db.table1")
        assert result.command_type == CommandType.SHOW_CREATE_TABLE
        assert result.table_name == "db.table1"


class TestShowTblProperties:
    def test_basic(self):
        result = preprocess("SHOW TBLPROPERTIES db.table1")
        assert result.command_type == CommandType.SHOW_TBLPROPERTIES
        assert result.table_name == "db.table1"


class TestCacheTable:
    def test_cache(self):
        result = preprocess("CACHE TABLE t1")
        assert result.command_type == CommandType.CACHE_TABLE
        assert result.table_name == "t1"

    def test_cache_lazy(self):
        result = preprocess("CACHE LAZY TABLE t1")
        assert result.command_type == CommandType.CACHE_TABLE


class TestUncacheTable:
    def test_uncache(self):
        result = preprocess("UNCACHE TABLE t1")
        assert result.command_type == CommandType.UNCACHE_TABLE
        assert result.table_name == "t1"

    def test_uncache_if_exists(self):
        result = preprocess("UNCACHE TABLE IF EXISTS t1")
        assert result.command_type == CommandType.UNCACHE_TABLE


class TestAddJar:
    def test_add_jar(self):
        result = preprocess("ADD JAR /path/to/my.jar")
        assert result.command_type == CommandType.ADD_JAR

    def test_add_file(self):
        result = preprocess("ADD FILE /path/to/file.txt")
        assert result.command_type == CommandType.ADD_JAR  # Same command type
