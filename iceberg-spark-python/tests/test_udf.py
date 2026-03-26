"""Tests for Python UDF support via DataFusion ScalarUDF and AggregateUDF."""

from __future__ import annotations

import pyarrow as pa
import pytest
from datafusion import SessionContext

from iceberg_spark.column import Column
from iceberg_spark.dataframe import DataFrame
from iceberg_spark.functions import col, udf, udaf
from iceberg_spark.session import (
    IcebergSession,
    UDFRegistration,
    _UserDefinedFunction,
    _UserDefinedAggregateFunction,
)
from iceberg_spark.types import (
    BooleanType,
    DoubleType,
    IntegerType,
    LongType,
    StringType,
)


# ---------------------------------------------------------------------------
# Helpers — lightweight session backed by a real DataFusion SessionContext
# ---------------------------------------------------------------------------


def _make_session() -> IcebergSession:
    """Create a minimal IcebergSession with an in-memory catalog."""
    ctx = SessionContext()
    session = IcebergSession.__new__(IcebergSession)
    session._ctx = ctx
    session._catalog = None
    session._catalog_name = "test"
    session._config = {}
    session._registered_tables = {}
    session._registered_udfs = {}
    session._current_namespace = None
    return session


def _register_people(session: IcebergSession) -> DataFrame:
    """Register a 'people' table and return a DataFrame over it."""
    batch = pa.record_batch(
        {
            "name": ["alice", "bob", "charlie"],
            "age": [30, 25, 35],
            "score": [85.5, 92.0, 78.3],
        },
        schema=pa.schema(
            [
                ("name", pa.utf8()),
                ("age", pa.int64()),
                ("score", pa.float64()),
            ]
        ),
    )
    session._ctx.register_record_batches("people", [[batch]])
    return DataFrame(session._ctx.table("people"), session)


# ---------------------------------------------------------------------------
# Tests — UDFRegistration.register()
# ---------------------------------------------------------------------------


class TestUDFRegisterStringUDF:
    """Register a string UDF and use in df.select()."""

    def test_string_upper(self):
        session = _make_session()
        df = _register_people(session)

        my_upper = session.udf.register("my_upper", lambda x: x.upper() if x else None, StringType())

        result = df.select(my_upper(col("name")))
        rows = result.collect()
        values = [r[0] for r in rows]
        assert values == ["ALICE", "BOB", "CHARLIE"]

    def test_returns_user_defined_function(self):
        session = _make_session()
        _register_people(session)
        fn = session.udf.register("fn", lambda x: x, StringType())
        assert isinstance(fn, _UserDefinedFunction)


class TestUDFRegisterNumericUDF:
    """Register a multi-arg numeric UDF."""

    def test_add_two_columns(self):
        session = _make_session()
        batch = pa.record_batch(
            {"a": [1, 2, 3], "b": [10, 20, 30]},
            schema=pa.schema([("a", pa.int64()), ("b", pa.int64())]),
        )
        session._ctx.register_record_batches("nums", [[batch]])
        df = DataFrame(session._ctx.table("nums"), session)

        my_add = session.udf.register(
            "my_add",
            lambda x, y: int(x) + int(y) if x is not None and y is not None else None,
            LongType(),
        )
        result = df.select(my_add(col("a"), col("b")))
        values = [r[0] for r in result.collect()]
        assert values == [11, 22, 33]


class TestUDFInSQL:
    """Register a UDF and use it in SQL queries."""

    def test_sql_select(self):
        session = _make_session()
        _register_people(session)

        session.udf.register("shout", lambda x: x.upper() + "!" if x else None, StringType())

        result = session._ctx.sql("SELECT shout(name) FROM people").collect()
        values = [batch.column(0).to_pylist() for batch in result]
        flat = [v for sublist in values for v in sublist]
        assert flat == ["ALICE!", "BOB!", "CHARLIE!"]


class TestUDFDecorator:
    """Test @udf(returnType=...) decorator pattern."""

    def test_decorator_usage(self):
        session = _make_session()
        df = _register_people(session)

        @udf(returnType=StringType())
        def reverse_name(s):
            return s[::-1] if s else None

        result = df.select(reverse_name(col("name")))
        values = [r[0] for r in result.collect()]
        assert values == ["ecila", "bob", "eilrahc"]

    def test_function_call_usage(self):
        my_len = udf(lambda s: len(s) if s else 0, IntegerType())
        assert isinstance(my_len, _UserDefinedFunction)


class TestUDFNullHandling:
    """UDF with null values in input data."""

    def test_nulls_propagate(self):
        session = _make_session()
        batch = pa.record_batch(
            {"val": ["hello", None, "world"]},
            schema=pa.schema([("val", pa.utf8())]),
        )
        session._ctx.register_record_batches("nullable", [[batch]])
        df = DataFrame(session._ctx.table("nullable"), session)

        safe_upper = session.udf.register(
            "safe_upper",
            lambda x: x.upper() if x is not None else None,
            StringType(),
        )
        result = df.select(safe_upper(col("val")))
        values = [r[0] for r in result.collect()]
        assert values == ["HELLO", None, "WORLD"]


class TestUDFReturnTypes:
    """Various return types: IntegerType, DoubleType, BooleanType, StringType."""

    def test_integer_return(self):
        session = _make_session()
        df = _register_people(session)

        str_len = session.udf.register(
            "str_len",
            lambda x: len(x) if x else 0,
            IntegerType(),
        )
        result = df.select(str_len(col("name")))
        values = [r[0] for r in result.collect()]
        assert values == [5, 3, 7]

    def test_double_return(self):
        session = _make_session()
        df = _register_people(session)

        half = session.udf.register(
            "half",
            lambda x: float(x) / 2.0 if x is not None else None,
            DoubleType(),
        )
        result = df.select(half(col("age")))
        values = [r[0] for r in result.collect()]
        assert values == [15.0, 12.5, 17.5]

    def test_boolean_return(self):
        session = _make_session()
        df = _register_people(session)

        is_long = session.udf.register(
            "is_long_name",
            lambda x: len(x) > 4 if x else False,
            BooleanType(),
        )
        result = df.select(is_long(col("name")))
        values = [r[0] for r in result.collect()]
        assert values == [True, False, True]

    def test_string_return_type_name(self):
        """Accept a string type name instead of DataType instance."""
        session = _make_session()
        df = _register_people(session)

        fn = session.udf.register("echo", lambda x: x, "string")
        result = df.select(fn(col("name")))
        values = [r[0] for r in result.collect()]
        assert values == ["alice", "bob", "charlie"]


class TestUDFCallable:
    """Test that _UserDefinedFunction is callable with Column args."""

    def test_repr(self):
        session = _make_session()
        _register_people(session)
        fn = session.udf.register("my_fn", lambda x: x, StringType())
        assert "my_fn" in repr(fn)

    def test_accepts_string_arg(self):
        """Passing a plain string column name should work."""
        session = _make_session()
        df = _register_people(session)

        fn = session.udf.register("echo2", lambda x: x, StringType())
        result = df.select(fn("name"))
        values = [r[0] for r in result.collect()]
        assert values == ["alice", "bob", "charlie"]


class TestUDFExceptionWarning:
    """UDF that raises should emit a warning, not silently return None."""

    def test_udf_exception_warns(self):
        session = _make_session()
        df = _register_people(session)

        def bad_udf(x):
            raise ValueError("intentional error")

        my_func = session.udf.register("bad_func", bad_udf, StringType())
        with pytest.warns(UserWarning, match="intentional error"):
            result = df.select(my_func(col("name"))).collect()
        # Results should be None for all rows
        assert all(row[0] is None for row in result)


class TestUDFInputTypes:
    """Test the inputTypes parameter on register()."""

    def test_explicit_input_types(self):
        session = _make_session()
        batch = pa.record_batch(
            {"a": [1, 2, 3], "b": [10, 20, 30]},
            schema=pa.schema([("a", pa.int64()), ("b", pa.int64())]),
        )
        session._ctx.register_record_batches("nums2", [[batch]])
        df = DataFrame(session._ctx.table("nums2"), session)

        my_add = session.udf.register(
            "my_add2",
            lambda x, y: x + y if x is not None and y is not None else None,
            LongType(),
            inputTypes=[LongType(), LongType()],
        )
        result = df.select(my_add(col("a"), col("b")))
        values = [r[0] for r in result.collect()]
        assert values == [11, 22, 33]


class TestStandaloneFunctions:
    """Test the standalone udf() from functions.py."""

    def test_udf_function_creates_callable(self):
        my_udf = udf(lambda x: x, StringType())
        assert callable(my_udf)
        assert isinstance(my_udf, _UserDefinedFunction)

    def test_udf_decorator_creates_callable(self):
        @udf(returnType=IntegerType())
        def my_func(x):
            return len(x) if x else 0

        assert callable(my_func)
        assert isinstance(my_func, _UserDefinedFunction)


class TestCatalogUDFTracking:
    """Registered UDFs should be discoverable through the catalog API."""

    def test_list_functions_after_register(self):
        """Registered UDFs should appear in catalog.listFunctions()."""
        session = _make_session()
        session.udf.register("list_test_func", lambda x: x, StringType())
        funcs = session.catalog.listFunctions()
        names = [f["name"] for f in funcs]
        assert "list_test_func" in names

    def test_list_functions_returns_rows(self):
        """listFunctions() returns Row objects with name, className, isTemporary."""
        session = _make_session()
        session.udf.register("row_test", lambda x: x, StringType())
        funcs = session.catalog.listFunctions()
        assert len(funcs) == 1
        assert funcs[0]["name"] == "row_test"
        assert funcs[0]["className"] == "python_udf"
        assert funcs[0]["isTemporary"] is True

    def test_function_exists_after_register(self):
        """functionExists() returns True for registered UDFs, False otherwise."""
        session = _make_session()
        session.udf.register("exists_test", lambda x: x, StringType())
        assert session.catalog.functionExists("exists_test")
        assert not session.catalog.functionExists("nonexistent")

    def test_list_functions_empty_initially(self):
        """listFunctions() returns empty list before any UDFs are registered."""
        session = _make_session()
        assert session.catalog.listFunctions() == []

    def test_multiple_udfs_listed(self):
        """Multiple registered UDFs all appear in listFunctions()."""
        session = _make_session()
        session.udf.register("func_a", lambda x: x, StringType())
        session.udf.register("func_b", lambda x: x, IntegerType())
        funcs = session.catalog.listFunctions()
        names = [f["name"] for f in funcs]
        assert "func_a" in names
        assert "func_b" in names
        assert len(funcs) == 2


# ---------------------------------------------------------------------------
# Tests — batch_mode UDFs
# ---------------------------------------------------------------------------


class TestBatchModeBasic:
    """Batch-mode UDFs receive pa.Array and return pa.Array."""

    def test_batch_mode_multiply(self):
        import pyarrow.compute as pc

        session = _make_session()
        batch = pa.record_batch(
            {"id": [1, 2, 3]},
            schema=pa.schema([("id", pa.int64())]),
        )
        session._ctx.register_record_batches("ids", [[batch]])
        df = DataFrame(session._ctx.table("ids"), session)

        def double_vals(arr):
            return pc.multiply(arr, 2)

        my_udf = session.udf.register(
            "double_it", double_vals, LongType(), inputTypes=[LongType()], batch_mode=True
        )
        result = df.select(my_udf(col("id"))).collect()
        values = [r[0] for r in result]
        assert values == [2, 4, 6]

    def test_batch_mode_multi_arg(self):
        import pyarrow.compute as pc

        session = _make_session()
        batch = pa.record_batch(
            {"a": [1, 2, 3], "b": [10, 20, 30]},
            schema=pa.schema([("a", pa.int64()), ("b", pa.int64())]),
        )
        session._ctx.register_record_batches("pairs", [[batch]])
        df = DataFrame(session._ctx.table("pairs"), session)

        def add_arrays(arr_a, arr_b):
            return pc.add(arr_a, arr_b)

        my_add = session.udf.register(
            "batch_add", add_arrays, LongType(), inputTypes=[LongType(), LongType()], batch_mode=True
        )
        result = df.select(my_add(col("a"), col("b"))).collect()
        values = [r[0] for r in result]
        assert values == [11, 22, 33]


class TestBatchModeRequiresInputTypes:
    """batch_mode=True requires inputTypes to be provided."""

    def test_raises_without_input_types(self):
        session = _make_session()
        with pytest.raises(ValueError, match="inputTypes"):
            session.udf.register("bad", lambda x: x, StringType(), batch_mode=True)


class TestBatchModeDecorator:
    """Test @udf(returnType=..., batch_mode=True) decorator pattern."""

    def test_decorator_batch_mode(self):
        import pyarrow.compute as pc

        session = _make_session()
        batch = pa.record_batch(
            {"val": [10, 20, 30]},
            schema=pa.schema([("val", pa.int64())]),
        )
        session._ctx.register_record_batches("tripled", [[batch]])
        df = DataFrame(session._ctx.table("tripled"), session)

        @udf(returnType=LongType(), inputTypes=[LongType()], batch_mode=True)
        def triple(arr):
            return pc.multiply(arr, 3)

        result = df.select(triple(col("val"))).collect()
        values = [r[0] for r in result]
        assert values == [30, 60, 90]


class TestBatchModeNullHandling:
    """Batch-mode UDFs receive nulls in the pa.Array and must handle them."""

    def test_nulls_in_batch_mode(self):
        import pyarrow.compute as pc

        session = _make_session()
        batch = pa.record_batch(
            {"val": pa.array([1, None, 3], type=pa.int64())},
            schema=pa.schema([("val", pa.int64())]),
        )
        session._ctx.register_record_batches("nullvals", [[batch]])
        df = DataFrame(session._ctx.table("nullvals"), session)

        def double_nullable(arr):
            # pc.multiply propagates nulls automatically
            return pc.multiply(arr, 2)

        my_udf = session.udf.register(
            "double_null", double_nullable, LongType(), inputTypes=[LongType()], batch_mode=True
        )
        result = df.select(my_udf(col("val"))).collect()
        values = [r[0] for r in result]
        assert values == [2, None, 6]


class TestBatchModeSQL:
    """Batch-mode UDFs work in SQL queries too."""

    def test_batch_mode_in_sql(self):
        import pyarrow.compute as pc

        session = _make_session()
        batch = pa.record_batch(
            {"x": [5, 10, 15]},
            schema=pa.schema([("x", pa.int64())]),
        )
        session._ctx.register_record_batches("sql_batch", [[batch]])

        def negate(arr):
            return pc.negate(arr)

        session.udf.register(
            "negate_it", negate, LongType(), inputTypes=[LongType()], batch_mode=True
        )
        result = session._ctx.sql("SELECT negate_it(x) FROM sql_batch").collect()
        flat = [v for batch in result for v in batch.column(0).to_pylist()]
        assert flat == [-5, -10, -15]


# ---------------------------------------------------------------------------
# Tests — UDAF (Aggregate UDF) support
# ---------------------------------------------------------------------------


class TestUDAFSimpleSum:
    """Register a sum aggregate UDAF and use in SQL."""

    def test_udaf_sum_sql(self):
        session = _make_session()
        my_sum = session.udf.register_udaf(
            "my_sum", lambda values: sum(values), LongType(), [LongType()]
        )

        batch = pa.record_batch(
            {"grp": ["a", "a", "b", "b"], "val": [1, 2, 3, 4]},
            schema=pa.schema([("grp", pa.utf8()), ("val", pa.int64())]),
        )
        session._ctx.register_record_batches("agg_t", [[batch]])

        result = session._ctx.sql(
            "SELECT grp, my_sum(val) as total FROM agg_t GROUP BY grp ORDER BY grp"
        ).collect()
        rows = {}
        for b in result:
            d = b.to_pydict()
            for grp, total in zip(d["grp"], d["total"]):
                rows[grp] = total

        assert rows["a"] == 3  # 1 + 2
        assert rows["b"] == 7  # 3 + 4

    def test_udaf_returns_aggregate_function(self):
        session = _make_session()
        fn = session.udf.register_udaf("fn_agg", lambda v: sum(v), LongType(), [LongType()])
        assert isinstance(fn, _UserDefinedAggregateFunction)


class TestUDAFInGroupByAgg:
    """UDAF used in df.groupBy().agg()."""

    def test_groupby_agg(self):
        session = _make_session()
        my_sum = session.udf.register_udaf(
            "my_sum2", lambda values: sum(values), LongType(), [LongType()]
        )

        batch = pa.record_batch(
            {"grp": ["a", "a", "b", "b"], "val": [1, 2, 3, 4]},
            schema=pa.schema([("grp", pa.utf8()), ("val", pa.int64())]),
        )
        session._ctx.register_record_batches("agg_t2", [[batch]])
        df = DataFrame(session._ctx.table("agg_t2"), session)

        result = df.groupBy("grp").agg(my_sum(col("val")).alias("total"))
        rows = {r["grp"]: r["total"] for r in result.collect()}
        assert rows["a"] == 3
        assert rows["b"] == 7


class TestUDAFCatalogTracking:
    """UDAF should appear in catalog.listFunctions()."""

    def test_udaf_in_list_functions(self):
        session = _make_session()
        session.udf.register_udaf("agg_func_1", lambda v: sum(v), LongType(), [LongType()])
        funcs = session.catalog.listFunctions()
        names = [f["name"] for f in funcs]
        assert "agg_func_1" in names

    def test_udaf_function_exists(self):
        session = _make_session()
        session.udf.register_udaf("agg_exists", lambda v: sum(v), LongType(), [LongType()])
        assert session.catalog.functionExists("agg_exists")

    def test_udaf_and_udf_both_listed(self):
        session = _make_session()
        session.udf.register("scalar_fn", lambda x: x, StringType())
        session.udf.register_udaf("agg_fn", lambda v: sum(v), LongType(), [LongType()])
        funcs = session.catalog.listFunctions()
        names = [f["name"] for f in funcs]
        assert "scalar_fn" in names
        assert "agg_fn" in names
        assert len(funcs) == 2


class TestUDAFNullHandling:
    """UDAF with null values in input data."""

    def test_nulls_skipped(self):
        session = _make_session()
        my_sum = session.udf.register_udaf(
            "null_sum", lambda values: sum(values) if values else 0, LongType(), [LongType()]
        )

        batch = pa.record_batch(
            {"grp": ["a", "a", "a"], "val": pa.array([1, None, 3], type=pa.int64())},
            schema=pa.schema([("grp", pa.utf8()), ("val", pa.int64())]),
        )
        session._ctx.register_record_batches("null_agg", [[batch]])

        result = session._ctx.sql(
            "SELECT grp, null_sum(val) as total FROM null_agg GROUP BY grp"
        ).collect()
        total = result[0].to_pydict()["total"][0]
        assert total == 4  # 1 + 3 (null skipped)


class TestUDAFAccumulatorClass:
    """UDAF with a custom Accumulator class."""

    def test_custom_accumulator(self):
        import pyarrow.compute as pc
        from datafusion.user_defined import Accumulator

        class MyMax(Accumulator):
            def __init__(self):
                self._max = None

            def state(self):
                return [pa.scalar(self._max, type=pa.int64())]

            def update(self, values):
                batch_max = pc.max(values).as_py()
                if batch_max is not None:
                    if self._max is None or batch_max > self._max:
                        self._max = batch_max

            def merge(self, states):
                for state_arr in states:
                    batch_max = pc.max(state_arr).as_py()
                    if batch_max is not None:
                        if self._max is None or batch_max > self._max:
                            self._max = batch_max

            def evaluate(self):
                return pa.scalar(self._max if self._max is not None else 0, type=pa.int64())

        session = _make_session()
        my_max = session.udf.register_udaf(
            "my_max", MyMax, LongType(), [LongType()]
        )

        batch = pa.record_batch(
            {"grp": ["a", "a", "b", "b"], "val": [10, 20, 5, 15]},
            schema=pa.schema([("grp", pa.utf8()), ("val", pa.int64())]),
        )
        session._ctx.register_record_batches("max_agg", [[batch]])

        result = session._ctx.sql(
            "SELECT grp, my_max(val) as mx FROM max_agg GROUP BY grp ORDER BY grp"
        ).collect()
        rows = {}
        for b in result:
            d = b.to_pydict()
            for grp, mx in zip(d["grp"], d["mx"]):
                rows[grp] = mx

        assert rows["a"] == 20
        assert rows["b"] == 15


class TestUDAFDecorator:
    """Test @udaf(returnType=...) decorator pattern from functions.py."""

    def test_udaf_decorator(self):
        @udaf(returnType=LongType(), inputTypes=[LongType()])
        def my_product(values):
            result = 1
            for v in values:
                result *= v
            return result

        assert isinstance(my_product, _UserDefinedAggregateFunction)

    def test_udaf_function_call(self):
        my_agg = udaf(lambda v: sum(v), LongType(), [LongType()])
        assert isinstance(my_agg, _UserDefinedAggregateFunction)


class TestUDAFWithDoubleType:
    """UDAF with DoubleType return and input."""

    def test_avg_udaf(self):
        session = _make_session()
        my_avg = session.udf.register_udaf(
            "my_avg",
            lambda values: sum(values) / len(values) if values else 0.0,
            DoubleType(),
            [DoubleType()],
        )

        batch = pa.record_batch(
            {"grp": ["a", "a", "b", "b"], "val": [10.0, 20.0, 5.0, 15.0]},
            schema=pa.schema([("grp", pa.utf8()), ("val", pa.float64())]),
        )
        session._ctx.register_record_batches("avg_agg", [[batch]])

        result = session._ctx.sql(
            "SELECT grp, my_avg(val) as average FROM avg_agg GROUP BY grp ORDER BY grp"
        ).collect()
        rows = {}
        for b in result:
            d = b.to_pydict()
            for grp, average in zip(d["grp"], d["average"]):
                rows[grp] = average

        assert rows["a"] == pytest.approx(15.0)
        assert rows["b"] == pytest.approx(10.0)


class TestUDAFRepr:
    """Test _UserDefinedAggregateFunction repr."""

    def test_repr(self):
        session = _make_session()
        fn = session.udf.register_udaf("repr_agg", lambda v: sum(v), LongType(), [LongType()])
        assert "repr_agg" in repr(fn)


class TestUDAFEdgeCases:
    """Edge case tests for UDAF aggregation."""

    def test_single_row_groups(self):
        """UDAF with one row per group."""
        session = _make_session()
        my_sum = session.udf.register_udaf(
            "sr_sum", lambda values: sum(values), LongType(), [LongType()]
        )
        batch = pa.record_batch(
            {"grp": ["a", "b", "c"], "val": [10, 20, 30]},
            schema=pa.schema([("grp", pa.utf8()), ("val", pa.int64())]),
        )
        session._ctx.register_record_batches("sr_t", [[batch]])
        result = session._ctx.sql(
            "SELECT grp, sr_sum(val) as total FROM sr_t GROUP BY grp ORDER BY grp"
        ).collect()
        rows = {}
        for b in result:
            d = b.to_pydict()
            for grp, total in zip(d["grp"], d["total"]):
                rows[grp] = total
        assert rows == {"a": 10, "b": 20, "c": 30}

    def test_whole_table_aggregate(self):
        """UDAF without GROUP BY — aggregate entire table."""
        session = _make_session()
        my_sum = session.udf.register_udaf(
            "wt_sum", lambda values: sum(values), LongType(), [LongType()]
        )
        batch = pa.record_batch(
            {"val": [1, 2, 3, 4, 5]},
            schema=pa.schema([("val", pa.int64())]),
        )
        session._ctx.register_record_batches("wt_t", [[batch]])
        result = session._ctx.sql("SELECT wt_sum(val) as total FROM wt_t").collect()
        total = result[0].to_pydict()["total"][0]
        assert total == 15

    def test_udaf_evaluate_exception_warns(self):
        """UDAF that raises in evaluate() should warn, not crash silently."""
        session = _make_session()

        def bad_agg(values):
            raise ValueError("intentional error")

        my_bad = session.udf.register_udaf(
            "bad_agg", bad_agg, LongType(), [LongType()]
        )
        batch = pa.record_batch(
            {"val": [1, 2, 3]},
            schema=pa.schema([("val", pa.int64())]),
        )
        session._ctx.register_record_batches("bad_t", [[batch]])
        with pytest.warns(UserWarning, match="UDAF evaluate.*failed"):
            result = session._ctx.sql("SELECT bad_agg(val) FROM bad_t").collect()
        # Should return None (not crash)
        total = result[0].to_pydict()[result[0].schema.names[0]][0]
        assert total is None
