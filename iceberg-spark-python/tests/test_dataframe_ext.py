"""Tests for Sprint 2: DataFrame method extensions."""

from __future__ import annotations

import json

import pyarrow as pa
import pytest
from datafusion import SessionContext

from iceberg_spark.dataframe import DataFrame


@pytest.fixture
def ctx():
    return SessionContext()


@pytest.fixture
def df(ctx):
    t = pa.table({
        "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "name": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
    })
    ctx.register_record_batches("t", [t.to_batches()])
    return DataFrame(ctx.table("t"), session=None)


class TestRandomSplit:
    def test_two_way_split(self, df):
        splits = df.randomSplit([0.5, 0.5], seed=42)
        assert len(splits) == 2
        total = splits[0].count() + splits[1].count()
        assert total == 10

    def test_three_way_split(self, df):
        splits = df.randomSplit([0.2, 0.3, 0.5], seed=42)
        assert len(splits) == 3
        total = sum(s.count() for s in splits)
        assert total == 10

    def test_deterministic_with_seed(self, df):
        s1 = df.randomSplit([0.5, 0.5], seed=123)
        s2 = df.randomSplit([0.5, 0.5], seed=123)
        assert s1[0].count() == s2[0].count()


class TestToJSON:
    def test_basic(self, df):
        result = df.toJSON()
        assert "value" in result.columns
        rows = result.collect()
        assert len(rows) == 10
        # Parse first row JSON
        parsed = json.loads(rows[0]["value"])
        assert "id" in parsed
        assert "name" in parsed


class TestForeach:
    def test_foreach(self, df):
        results = []
        df.foreach(lambda row: results.append(row["id"]))
        assert len(results) == 10
        assert 1 in results

    def test_foreach_partition(self, df):
        results = []
        def process(iterator):
            for row in iterator:
                results.append(row["id"])
        df.foreachPartition(process)
        assert len(results) == 10


class TestMapInPandas:
    def test_basic_identity(self, df):
        """mapInPandas with identity function returns same data."""
        def identity(iterator):
            for pdf in iterator:
                yield pdf

        result = df.mapInPandas(identity, schema=df._df.to_arrow_table().schema)
        assert result.count() == 10

    def test_transform(self, df):
        """mapInPandas can transform columns."""
        schema = pa.schema([pa.field("id", pa.int64()), pa.field("upper_name", pa.string())])

        def transform(iterator):
            for pdf in iterator:
                yield pdf.assign(upper_name=pdf["name"].str.upper())[["id", "upper_name"]]

        result = df.mapInPandas(transform, schema=schema)
        assert result.count() == 10
        rows = result.collect()
        assert rows[0]["upper_name"] == "A"

    def test_filter(self, df):
        """mapInPandas can filter rows."""
        def filter_even(iterator):
            for pdf in iterator:
                yield pdf[pdf["id"] % 2 == 0]

        result = df.mapInPandas(filter_even, schema=df._df.to_arrow_table().schema)
        assert result.count() == 5

    def test_with_struct_type_schema(self, df):
        """mapInPandas accepts StructType schema."""
        from iceberg_spark.types import IntegerType, StringType, StructField, StructType

        schema = StructType([
            StructField("id", IntegerType()),
            StructField("name", StringType()),
        ])

        def identity(iterator):
            for pdf in iterator:
                yield pdf

        result = df.mapInPandas(identity, schema=schema)
        assert result.count() == 10


class TestMapInArrow:
    def test_basic_identity(self, df):
        """mapInArrow with identity function returns same data."""
        def identity(iterator):
            for batch in iterator:
                yield batch

        result = df.mapInArrow(identity, schema=df._df.to_arrow_table().schema)
        assert result.count() == 10

    def test_transform(self, df):
        """mapInArrow can transform batches."""
        import pyarrow.compute as pc

        output_schema = pa.schema([pa.field("id_doubled", pa.int64())])

        def double_id(iterator):
            for batch in iterator:
                ids = batch.column("id")
                doubled = pc.multiply(ids, 2)
                yield pa.record_batch([doubled], schema=output_schema)

        result = df.mapInArrow(double_id, schema=output_schema)
        assert result.count() == 10
        rows = result.collect()
        assert rows[0]["id_doubled"] == 2

    def test_filter(self, df):
        """mapInArrow can filter rows."""
        import pyarrow.compute as pc

        def filter_gt_5(iterator):
            for batch in iterator:
                mask = pc.greater(batch.column("id"), 5)
                yield batch.filter(mask)

        result = df.mapInArrow(filter_gt_5, schema=df._df.to_arrow_table().schema)
        assert result.count() == 5


class TestMapEmptyResults:
    """Edge case: user function yields nothing."""

    def test_map_in_pandas_empty(self, df):
        """mapInPandas returns empty DataFrame when func yields nothing."""
        import pandas as pd

        def yield_nothing(iterator):
            for pdf in iterator:
                # Yield an empty DataFrame with same columns
                yield pdf.iloc[0:0]

        result = df.mapInPandas(yield_nothing, schema=df._df.to_arrow_table().schema)
        assert result.count() == 0

    def test_map_in_arrow_empty(self, df):
        """mapInArrow returns empty DataFrame when func yields nothing."""
        def yield_nothing(iterator):
            for batch in iterator:
                yield batch.slice(0, 0)

        result = df.mapInArrow(yield_nothing, schema=df._df.to_arrow_table().schema)
        assert result.count() == 0

    def test_map_in_pandas_multiple_yields(self, df):
        """mapInPandas handles func that yields multiple DataFrames."""
        def split_in_two(iterator):
            for pdf in iterator:
                mid = len(pdf) // 2
                yield pdf.iloc[:mid]
                yield pdf.iloc[mid:]

        result = df.mapInPandas(split_in_two, schema=df._df.to_arrow_table().schema)
        assert result.count() == 10
