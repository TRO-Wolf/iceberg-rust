"""Tests for DataFusion integration — DataFrame, Column, functions.

These tests only require the `datafusion` and `pyarrow` packages,
not PyIceberg or pyiceberg-core.
"""

import pyarrow as pa
import pytest
from datafusion import SessionContext

from iceberg_spark.column import Column
from iceberg_spark.dataframe import DataFrame
from iceberg_spark.functions import avg, col, count, lit, max, min, sum, upper, when
from iceberg_spark.row import Row


@pytest.fixture
def ctx():
    return SessionContext()


@pytest.fixture
def sample_table():
    return pa.table({
        "id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Carol", "Dave", "Eve"],
        "age": [30, 25, 35, 28, 32],
        "score": [85.5, 92.0, 78.3, 95.1, 88.7],
    })


@pytest.fixture
def df(ctx, sample_table):
    ctx.register_record_batches("test_table", [sample_table.to_batches()])
    return DataFrame(ctx.table("test_table"), session=None)


class TestDataFrame:
    def test_columns(self, df):
        assert df.columns == ["id", "name", "age", "score"]

    def test_dtypes(self, df):
        dtypes = df.dtypes
        assert ("id", "bigint") in dtypes
        assert ("name", "string") in dtypes

    def test_count(self, df):
        assert df.count() == 5

    def test_show(self, df, capsys):
        df.show()
        captured = capsys.readouterr()
        assert "Alice" in captured.out
        assert "Bob" in captured.out

    def test_collect(self, df):
        rows = df.collect()
        assert len(rows) == 5
        assert isinstance(rows[0], Row)
        assert rows[0]["id"] == 1
        assert rows[0]["name"] == "Alice"

    def test_first(self, df):
        row = df.first()
        assert isinstance(row, Row)
        assert row["id"] == 1

    def test_head(self, df):
        rows = df.head(3)
        assert len(rows) == 3

    def test_take(self, df):
        rows = df.take(2)
        assert len(rows) == 2

    def test_select_strings(self, df):
        result = df.select("id", "name")
        assert result.columns == ["id", "name"]

    def test_select_columns(self, df):
        result = df.select(col("id"), col("name"))
        assert result.columns == ["id", "name"]

    def test_filter_column(self, df):
        result = df.filter(col("age") > lit(30))
        rows = result.collect()
        assert all(r["age"] > 30 for r in rows)

    def test_where_alias(self, df):
        result = df.where(col("age") > lit(30))
        assert result.count() == df.filter(col("age") > lit(30)).count()

    def test_limit(self, df):
        result = df.limit(3)
        assert result.count() == 3

    def test_distinct(self, df):
        result = df.distinct()
        assert result.count() == 5  # all rows are unique

    def test_sort(self, df):
        result = df.sort("age")
        rows = result.collect()
        ages = [r["age"] for r in rows]
        assert ages == sorted(ages)

    def test_orderby_alias(self, df):
        result = df.orderBy("age")
        rows = result.collect()
        ages = [r["age"] for r in rows]
        assert ages == sorted(ages)

    def test_drop_column(self, df):
        result = df.drop("score")
        assert "score" not in result.columns
        assert "id" in result.columns

    def test_to_pandas(self, df):
        pdf = df.toPandas()
        assert len(pdf) == 5
        assert list(pdf.columns) == ["id", "name", "age", "score"]

    def test_to_arrow(self, df):
        table = df.toArrow()
        assert isinstance(table, pa.Table)
        assert table.num_rows == 5

    def test_print_schema(self, df, capsys):
        df.printSchema()
        captured = capsys.readouterr()
        assert "root" in captured.out
        assert "id" in captured.out

    def test_is_empty(self, df):
        assert not df.isEmpty()
        empty = df.filter(col("id") > lit(100))
        assert empty.isEmpty()

    def test_union(self, df):
        result = df.union(df)
        assert result.count() == 10

    def test_cache_noop(self, df):
        result = df.cache()
        assert result.count() == 5


class TestColumn:
    def test_comparison_operators(self):
        c = col("x")
        assert isinstance(c == lit(1), Column)
        assert isinstance(c != lit(1), Column)
        assert isinstance(c < lit(1), Column)
        assert isinstance(c <= lit(1), Column)
        assert isinstance(c > lit(1), Column)
        assert isinstance(c >= lit(1), Column)

    def test_boolean_operators(self):
        c1 = col("x") > lit(0)
        c2 = col("y") < lit(10)
        assert isinstance(c1 & c2, Column)
        assert isinstance(c1 | c2, Column)
        assert isinstance(~c1, Column)

    def test_arithmetic_operators(self):
        c = col("x")
        assert isinstance(c + lit(1), Column)
        assert isinstance(c - lit(1), Column)
        assert isinstance(c * lit(2), Column)
        assert isinstance(c / lit(2), Column)

    def test_alias(self):
        c = col("x").alias("new_x")
        assert isinstance(c, Column)

    def test_repr(self):
        c = col("my_column")
        assert "my_column" in repr(c)


class TestFunctions:
    def test_col_and_lit(self, df):
        result = df.select(col("id"), lit(42).alias("const"))
        assert result.columns == ["id", "const"]
        rows = result.collect()
        assert all(r["const"] == 42 for r in rows)

    def test_count(self, df):
        result = df.agg(count("*").alias("cnt"))
        rows = result.collect()
        assert rows[0]["cnt"] == 5

    def test_sum(self, df):
        result = df.agg(sum("age").alias("total_age"))
        rows = result.collect()
        assert rows[0]["total_age"] == 150  # 30+25+35+28+32

    def test_avg(self, df):
        result = df.agg(avg("age").alias("avg_age"))
        rows = result.collect()
        assert rows[0]["avg_age"] == 30.0

    def test_min_max(self, df):
        result = df.agg(min("age").alias("min_age"), max("age").alias("max_age"))
        rows = result.collect()
        assert rows[0]["min_age"] == 25
        assert rows[0]["max_age"] == 35

    def test_upper(self, df):
        result = df.select(upper("name").alias("upper_name"))
        rows = result.collect()
        assert rows[0]["upper_name"] == "ALICE"

    def test_when(self, df):
        result = df.select(
            col("name"),
            when(col("age") > lit(30), lit("senior"))
            .otherwise(lit("junior"))
            .alias("category"),
        )
        rows = result.collect()
        categories = {r["name"]: r["category"] for r in rows}
        assert categories["Carol"] == "senior"
        assert categories["Bob"] == "junior"


class TestGroupedData:
    def test_group_by_count(self, ctx):
        table = pa.table({
            "dept": ["eng", "eng", "sales", "sales", "sales"],
            "salary": [100, 120, 80, 90, 85],
        })
        ctx.register_record_batches("employees", [table.to_batches()])
        df = DataFrame(ctx.table("employees"), session=None)

        result = df.groupBy("dept").count()
        rows = result.collect()
        dept_counts = {r["dept"]: r["count"] for r in rows}
        assert dept_counts["eng"] == 2
        assert dept_counts["sales"] == 3

    def test_group_by_sum(self, ctx):
        table = pa.table({
            "dept": ["eng", "eng", "sales"],
            "salary": [100, 120, 80],
        })
        ctx.register_record_batches("employees2", [table.to_batches()])
        df = DataFrame(ctx.table("employees2"), session=None)

        result = df.groupBy("dept").sum("salary")
        rows = result.collect()
        dept_sums = {r["dept"]: r["sum(salary)"] for r in rows}
        assert dept_sums["eng"] == 220
        assert dept_sums["sales"] == 80
