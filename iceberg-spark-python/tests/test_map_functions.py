"""Tests for map functions (map_keys, map_values, create_map) via SQL fallback."""

import pyarrow as pa
import pytest
from datafusion import SessionContext

from iceberg_spark.column import Column, _SqlFuncColumn
from iceberg_spark.dataframe import DataFrame
from iceberg_spark.functions import col, create_map, lit, map_keys, map_values


class _MiniSession:
    """Minimal session stub that exposes a DataFusion SessionContext."""

    def __init__(self):
        self._ctx = SessionContext()


@pytest.fixture()
def session():
    return _MiniSession()


@pytest.fixture()
def map_df(session):
    """DataFrame with a map column created via SQL."""
    raw = session._ctx.sql("SELECT make_map('a', 1, 'b', 2) AS m, 'hello' AS name")
    return DataFrame(raw, session=session)


class TestSqlFuncColumn:
    def test_is_sql_func_flag(self):
        c = _SqlFuncColumn("map_keys(m)")
        assert c._is_sql_func is True

    def test_alias_returns_sql_func_column(self):
        c = _SqlFuncColumn("map_keys(m)")
        aliased = c.alias("keys")
        assert isinstance(aliased, _SqlFuncColumn)
        assert aliased._sql_fragment == 'map_keys(m) AS "keys"'

    def test_repr(self):
        c = _SqlFuncColumn("map_keys(m)")
        assert repr(c) == "Column<'map_keys(m)'>"

    def test_internal_expr_is_none(self):
        c = _SqlFuncColumn("map_keys(m)")
        assert c._expr is None

    def test_expr_property_raises(self):
        c = _SqlFuncColumn("map_keys(m)")
        with pytest.raises(TypeError, match="has no DataFusion Expr"):
            _ = c.expr

    def test_eq_raises(self):
        c = _SqlFuncColumn("map_keys(m)")
        with pytest.raises(TypeError, match="Cannot call __eq__"):
            c == "foo"

    def test_ne_raises(self):
        c = _SqlFuncColumn("map_keys(m)")
        with pytest.raises(TypeError, match="Cannot call __ne__"):
            c != "foo"

    def test_lt_raises(self):
        c = _SqlFuncColumn("map_keys(m)")
        with pytest.raises(TypeError, match="Cannot call __lt__"):
            c < "foo"

    def test_add_raises(self):
        c = _SqlFuncColumn("map_keys(m)")
        with pytest.raises(TypeError, match="Cannot call __add__"):
            c + 1

    def test_sub_raises(self):
        c = _SqlFuncColumn("map_keys(m)")
        with pytest.raises(TypeError, match="Cannot call __sub__"):
            c - 1

    def test_mul_raises(self):
        c = _SqlFuncColumn("map_keys(m)")
        with pytest.raises(TypeError, match="Cannot call __mul__"):
            c * 2

    def test_truediv_raises(self):
        c = _SqlFuncColumn("map_keys(m)")
        with pytest.raises(TypeError, match="Cannot call __truediv__"):
            c / 2

    def test_and_raises(self):
        c = _SqlFuncColumn("map_keys(m)")
        with pytest.raises(TypeError, match="Cannot call __and__"):
            c & True

    def test_or_raises(self):
        c = _SqlFuncColumn("map_keys(m)")
        with pytest.raises(TypeError, match="Cannot call __or__"):
            c | True

    def test_invert_raises(self):
        c = _SqlFuncColumn("map_keys(m)")
        with pytest.raises(TypeError, match="Cannot call __invert__"):
            ~c

    def test_cast_raises(self):
        c = _SqlFuncColumn("map_keys(m)")
        with pytest.raises(TypeError, match="Cannot call cast"):
            c.cast("string")

    def test_isNull_raises(self):
        c = _SqlFuncColumn("map_keys(m)")
        with pytest.raises(TypeError, match="Cannot call isNull"):
            c.isNull()

    def test_isNotNull_raises(self):
        c = _SqlFuncColumn("map_keys(m)")
        with pytest.raises(TypeError, match="Cannot call isNotNull"):
            c.isNotNull()

    def test_isin_raises(self):
        c = _SqlFuncColumn("map_keys(m)")
        with pytest.raises(TypeError, match="Cannot call isin"):
            c.isin(1, 2, 3)

    def test_between_raises(self):
        c = _SqlFuncColumn("map_keys(m)")
        with pytest.raises(TypeError, match="Cannot call between"):
            c.between(1, 10)

    def test_like_raises(self):
        c = _SqlFuncColumn("map_keys(m)")
        with pytest.raises(TypeError, match="Cannot call like"):
            c.like("%foo%")


class TestMapKeys:
    def test_returns_sql_func_column(self):
        result = map_keys(col("m"))
        assert isinstance(result, _SqlFuncColumn)
        assert "map_keys" in result._sql_fragment

    def test_with_string_arg(self):
        result = map_keys("m")
        assert isinstance(result, _SqlFuncColumn)
        assert result._sql_fragment == "map_keys(m)"

    def test_map_keys_select(self, map_df):
        result = map_df.select(map_keys(col("m")).alias("keys"))
        rows = result.collect()
        assert len(rows) == 1
        keys = sorted(rows[0]["keys"])
        assert keys == ["a", "b"]


class TestMapValues:
    def test_returns_sql_func_column(self):
        result = map_values(col("m"))
        assert isinstance(result, _SqlFuncColumn)
        assert "map_values" in result._sql_fragment

    def test_with_string_arg(self):
        result = map_values("m")
        assert isinstance(result, _SqlFuncColumn)
        assert result._sql_fragment == "map_values(m)"

    def test_map_values_select(self, map_df):
        result = map_df.select(map_values(col("m")).alias("vals"))
        rows = result.collect()
        assert len(rows) == 1
        vals = sorted(rows[0]["vals"])
        assert vals == [1, 2]


class TestCreateMap:
    def test_returns_sql_func_column(self):
        result = create_map(col("k"), col("v"))
        assert isinstance(result, _SqlFuncColumn)
        assert "make_map" in result._sql_fragment

    def test_with_string_args(self):
        result = create_map("k", "v")
        assert isinstance(result, _SqlFuncColumn)
        assert result._sql_fragment == 'make_map("k", "v")'

    def test_create_map_select(self, session):
        raw = session._ctx.sql("SELECT 'x' AS k, 10 AS v")
        df = DataFrame(raw, session=session)
        result = df.select(create_map(col("k"), col("v")).alias("m"))
        rows = result.collect()
        assert len(rows) == 1
        # Map entries come back as list of tuples (key, value)
        m = rows[0]["m"]
        assert len(m) == 1
        assert m[0][0] == "x"
        assert m[0][1] == 10


class TestSelectMixedColumns:
    def test_normal_and_sql_func_columns(self, map_df):
        """Verify _SqlFuncColumn fallback in select() works alongside normal columns."""
        result = map_df.select(col("name"), map_keys(col("m")).alias("keys"))
        rows = result.collect()
        assert len(rows) == 1
        assert rows[0]["name"] == "hello"
        assert sorted(rows[0]["keys"]) == ["a", "b"]

    def test_string_and_sql_func_columns(self, session):
        """String column references work alongside _SqlFuncColumn."""
        raw = session._ctx.sql("SELECT make_map('x', 99) AS m, 42 AS val")
        df = DataFrame(raw, session=session)
        result = df.select("val", map_values(col("m")).alias("vals"))
        rows = result.collect()
        assert len(rows) == 1
        assert rows[0]["val"] == 42
        assert rows[0]["vals"] == [99]

    def test_chained_alias(self, map_df):
        """Alias chaining on map_keys result."""
        result = map_df.select(map_keys(col("m")).alias("my_keys"))
        rows = result.collect()
        assert len(rows) == 1
        assert sorted(rows[0]["my_keys"]) == ["a", "b"]

    def test_complex_expr_with_sql_func_raises(self, map_df):
        """Mixing complex Column expressions with SQL function columns raises TypeError."""
        complex_col = col("name") + lit(" world")
        with pytest.raises(TypeError, match="Cannot mix complex"):
            map_df.select(complex_col, map_keys(col("m")))
