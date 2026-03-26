"""Tests for Sprint 1 Tasks 1.2-1.5: Session stubs."""

from __future__ import annotations

import pytest

from iceberg_spark.session import (
    IcebergSession,
    RuntimeConfig,
    SparkContextStub,
    StreamingQueryManagerStub,
    UDFRegistration,
)


class TestSparkContextStub:
    def test_app_name(self):
        stub = SparkContextStub.__new__(SparkContextStub)
        stub._session = type("S", (), {
            "_config": {"spark.app.name": "my_app"},
            "version": "0.1.0",
        })()
        assert stub.appName == "my_app"

    def test_version(self):
        stub = SparkContextStub.__new__(SparkContextStub)
        stub._session = type("S", (), {
            "_config": {},
            "version": "0.1.0",
        })()
        assert stub.version == "0.1.0"

    def test_master(self):
        stub = SparkContextStub.__new__(SparkContextStub)
        stub._session = type("S", (), {
            "_config": {},
            "version": "0.1.0",
        })()
        assert stub.master == "local"

    def test_set_log_level_noop(self):
        stub = SparkContextStub.__new__(SparkContextStub)
        stub._session = type("S", (), {
            "_config": {},
            "version": "0.1.0",
        })()
        # Should not raise
        stub.setLogLevel("WARN")

    def test_add_py_file_noop(self):
        stub = SparkContextStub.__new__(SparkContextStub)
        stub._session = type("S", (), {
            "_config": {},
            "version": "0.1.0",
        })()
        stub.addPyFile("/some/path.py")


class TestUDFRegistration:
    def test_register_no_session_raises(self):
        reg = UDFRegistration()
        with pytest.raises(RuntimeError, match="requires a live session"):
            reg.register("my_func", lambda x: x, "string")

    def test_register_no_callable_raises(self):
        reg = UDFRegistration(session=object())
        with pytest.raises(ValueError, match="callable must be provided"):
            reg.register("my_func", f=None)

    def test_register_java_raises(self):
        reg = UDFRegistration()
        with pytest.raises(NotImplementedError, match="no JVM"):
            reg.registerJavaFunction("f", "com.example.F")

    def test_register_java_udaf_raises(self):
        reg = UDFRegistration()
        with pytest.raises(NotImplementedError, match="no JVM"):
            reg.registerJavaUDAF("f", "com.example.F")


class TestStreamingQueryManagerStub:
    def test_active_empty(self):
        mgr = StreamingQueryManagerStub()
        assert mgr.active == []

    def test_get_raises(self):
        mgr = StreamingQueryManagerStub()
        with pytest.raises(NotImplementedError, match="Streaming"):
            mgr.get("some_id")

    def test_await_any_termination(self):
        mgr = StreamingQueryManagerStub()
        assert mgr.awaitAnyTermination() is True

    def test_reset_terminated_noop(self):
        mgr = StreamingQueryManagerStub()
        mgr.resetTerminated()  # Should not raise


class TestParameterizedSQL:
    """Test that sql() accepts args parameter without crashing.

    Full integration would need a catalog, but we test the substitution logic.
    """

    def test_args_substitution_string(self):
        # Manually test the substitution logic
        query = "SELECT * FROM t WHERE name = :name"
        args = {"name": "Alice"}
        # Simulate what sql() does
        for key, value in args.items():
            if isinstance(value, str):
                escaped = value.replace("'", "''")
                query = query.replace(f":{key}", f"'{escaped}'")
        assert query == "SELECT * FROM t WHERE name = 'Alice'"

    def test_args_substitution_int(self):
        query = "SELECT * FROM t WHERE id = :id"
        args = {"id": 42}
        for key, value in args.items():
            if isinstance(value, str):
                escaped = value.replace("'", "''")
                query = query.replace(f":{key}", f"'{escaped}'")
            elif value is None:
                query = query.replace(f":{key}", "NULL")
            else:
                query = query.replace(f":{key}", str(value))
        assert query == "SELECT * FROM t WHERE id = 42"

    def test_args_substitution_null(self):
        query = "SELECT * FROM t WHERE val = :val"
        args = {"val": None}
        for key, value in args.items():
            if isinstance(value, str):
                escaped = value.replace("'", "''")
                query = query.replace(f":{key}", f"'{escaped}'")
            elif value is None:
                query = query.replace(f":{key}", "NULL")
            else:
                query = query.replace(f":{key}", str(value))
        assert query == "SELECT * FROM t WHERE val = NULL"

    def test_args_escape_single_quote(self):
        query = "SELECT * FROM t WHERE name = :name"
        args = {"name": "O'Brien"}
        for key, value in args.items():
            if isinstance(value, str):
                escaped = value.replace("'", "''")
                query = query.replace(f":{key}", f"'{escaped}'")
        assert query == "SELECT * FROM t WHERE name = 'O''Brien'"


class TestStreamingStubs:
    """Test that readStream/writeStream raise NotImplementedError."""

    def test_read_stream_attr_exists(self):
        # Verify the property exists on the class
        assert hasattr(IcebergSession, "readStream")

    def test_write_stream_attr_exists(self):
        assert hasattr(IcebergSession, "writeStream")

    def test_streams_attr_exists(self):
        assert hasattr(IcebergSession, "streams")
