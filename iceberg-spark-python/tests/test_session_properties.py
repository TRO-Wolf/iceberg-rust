"""Tests for Session properties (Task 8A): version, conf, getActiveSession, newSession."""

from __future__ import annotations

import pytest

from iceberg_spark.session import IcebergSession, RuntimeConfig


# ---------------------------------------------------------------------------
# Helpers — build a minimal session without a real catalog
# ---------------------------------------------------------------------------

def _make_session(**config) -> IcebergSession:
    """Create an IcebergSession backed by a no-op in-memory catalog."""
    from unittest.mock import MagicMock
    from datafusion import SessionContext

    catalog = MagicMock()
    catalog.list_namespaces.return_value = []
    return IcebergSession(
        ctx=SessionContext(),
        catalog=catalog,
        catalog_name="test",
        config=config,
    )


# ---------------------------------------------------------------------------
# RuntimeConfig
# ---------------------------------------------------------------------------

class TestRuntimeConfig:
    def test_get_missing_returns_default(self):
        cfg = RuntimeConfig({})
        assert cfg.get("missing") is None
        assert cfg.get("missing", "fallback") == "fallback"

    def test_set_and_get(self):
        cfg = RuntimeConfig({})
        cfg.set("spark.sql.shuffle.partitions", "200")
        assert cfg.get("spark.sql.shuffle.partitions") == "200"

    def test_unset_removes_key(self):
        cfg = RuntimeConfig({"key": "val"})
        cfg.unset("key")
        assert cfg.get("key") is None

    def test_unset_missing_key_is_noop(self):
        cfg = RuntimeConfig({})
        cfg.unset("nonexistent")  # should not raise

    def test_get_all(self):
        cfg = RuntimeConfig({"a": "1", "b": "2"})
        result = cfg.getAll()
        assert result == {"a": "1", "b": "2"}

    def test_get_all_returns_copy(self):
        data = {"a": "1"}
        cfg = RuntimeConfig(data)
        cfg.getAll()["a"] = "mutated"
        assert cfg.get("a") == "1"

    def test_is_modifiable(self):
        cfg = RuntimeConfig({})
        assert cfg.isModifiable("anything") is True


# ---------------------------------------------------------------------------
# IcebergSession.conf
# ---------------------------------------------------------------------------

class TestSessionConf:
    def test_conf_returns_runtime_config(self):
        session = _make_session()
        assert isinstance(session.conf, RuntimeConfig)

    def test_conf_set_persists(self):
        session = _make_session()
        session.conf.set("my.key", "hello")
        assert session.conf.get("my.key") == "hello"

    def test_conf_shares_underlying_dict(self):
        """Changes via session.conf are visible in subsequent .conf accesses."""
        session = _make_session()
        session.conf.set("x", "1")
        # A second access to session.conf should see the same value
        assert session.conf.get("x") == "1"

    def test_conf_initial_values(self):
        session = _make_session(foo="bar")
        assert session.conf.get("foo") == "bar"


# ---------------------------------------------------------------------------
# IcebergSession.version
# ---------------------------------------------------------------------------

class TestSessionVersion:
    def test_version_is_string(self):
        session = _make_session()
        assert isinstance(session.version, str)
        assert len(session.version) > 0

    def test_version_matches_package(self):
        from iceberg_spark import __version__
        session = _make_session()
        assert session.version == __version__

    def test_version_no_circular_import(self):
        """version must not import from iceberg_spark (circular)."""
        import ast, inspect, textwrap
        from iceberg_spark import session as session_mod
        src = textwrap.dedent(inspect.getsource(session_mod.IcebergSession.version.fget))
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert node.module != "iceberg_spark", \
                    "version property must not import from iceberg_spark (circular)"


# ---------------------------------------------------------------------------
# IcebergSession.getActiveSession
# ---------------------------------------------------------------------------

class TestGetActiveSession:
    def setup_method(self):
        # Reset between tests to avoid cross-test leakage
        IcebergSession._active_session = None

    def test_returns_none_when_no_session_built(self):
        assert IcebergSession.getActiveSession() is None

    def test_returns_session_after_manual_assignment(self):
        session = _make_session()
        IcebergSession._active_session = session
        assert IcebergSession.getActiveSession() is session

    def test_build_sets_active_session(self):
        """build() must set _active_session."""
        from unittest.mock import MagicMock, patch

        mock_catalog = MagicMock()
        mock_catalog.list_namespaces.return_value = []
        with patch("iceberg_spark.session.create_catalog", return_value=mock_catalog):
            session = IcebergSession.builder().catalog("sql").build()
        assert IcebergSession.getActiveSession() is session

    def test_stop_clears_active_session(self):
        """stop() must clear _active_session when it is this session."""
        session = _make_session()
        IcebergSession._active_session = session
        session.stop()
        assert IcebergSession.getActiveSession() is None

    def test_stop_leaves_other_session_active(self):
        """stop() on a non-active session must not clear a different active session."""
        session_a = _make_session()
        session_b = _make_session()
        IcebergSession._active_session = session_a
        session_b.stop()  # session_b is not active; should not clear session_a
        assert IcebergSession.getActiveSession() is session_a

    def teardown_method(self):
        IcebergSession._active_session = None


# ---------------------------------------------------------------------------
# IcebergSession.newSession
# ---------------------------------------------------------------------------

class TestNewSession:
    def test_new_session_is_different_object(self):
        session = _make_session()
        new = session.newSession()
        assert new is not session

    def test_new_session_shares_catalog(self):
        session = _make_session()
        new = session.newSession()
        assert new._catalog is session._catalog

    def test_new_session_shares_catalog_name(self):
        session = _make_session()
        new = session.newSession()
        assert new._catalog_name == session._catalog_name

    def test_new_session_has_fresh_context(self):
        session = _make_session()
        new = session.newSession()
        assert new._ctx is not session._ctx

    def test_new_session_copies_config(self):
        session = _make_session(my_key="my_val")
        new = session.newSession()
        assert new._config.get("my_key") == "my_val"

    def test_new_session_config_is_independent(self):
        """Mutations to new session's config don't affect the original."""
        session = _make_session(key="original")
        new = session.newSession()
        new._config["key"] = "changed"
        assert session._config.get("key") == "original"
