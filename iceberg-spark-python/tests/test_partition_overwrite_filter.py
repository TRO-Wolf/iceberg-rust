"""Unit tests for _build_partition_overwrite_filter()."""

from __future__ import annotations

from unittest.mock import MagicMock

import pyarrow as pa
import pytest

from pyiceberg.expressions import AlwaysTrue, And, EqualTo, In
from pyiceberg.partitioning import PartitionField, PartitionSpec
from pyiceberg.schema import Schema
from pyiceberg.transforms import (
    BucketTransform,
    IdentityTransform,
    YearTransform,
)
from pyiceberg.types import IntegerType, NestedField, StringType

from iceberg_spark.catalog_ops import _build_partition_overwrite_filter


def _make_mock_table(schema_fields, partition_fields):
    """Create a mock PyIceberg table with the given schema and partition spec."""
    schema = Schema(*schema_fields)
    spec = PartitionSpec(*partition_fields) if partition_fields else PartitionSpec()

    table = MagicMock()
    table.spec.return_value = spec
    table.schema.return_value = schema
    return table


class TestBuildPartitionOverwriteFilter:
    """Unit tests for the _build_partition_overwrite_filter helper."""

    def test_unpartitioned_returns_always_true(self):
        table = _make_mock_table(
            schema_fields=[
                NestedField(field_id=1, name="id", field_type=IntegerType(), required=False),
                NestedField(field_id=2, name="name", field_type=StringType(), required=False),
            ],
            partition_fields=[],
        )
        data = pa.table({"id": [1, 2], "name": ["a", "b"]})
        result = _build_partition_overwrite_filter(table, data)
        assert isinstance(result, AlwaysTrue)

    def test_single_identity_partition_single_value(self):
        table = _make_mock_table(
            schema_fields=[
                NestedField(field_id=1, name="id", field_type=IntegerType(), required=False),
                NestedField(field_id=2, name="region", field_type=StringType(), required=False),
            ],
            partition_fields=[
                PartitionField(
                    source_id=2, field_id=1000, transform=IdentityTransform(), name="region"
                ),
            ],
        )
        data = pa.table({"id": [1, 2], "region": ["US", "US"]})
        result = _build_partition_overwrite_filter(table, data)
        assert isinstance(result, EqualTo)

    def test_single_identity_partition_multiple_values(self):
        table = _make_mock_table(
            schema_fields=[
                NestedField(field_id=1, name="id", field_type=IntegerType(), required=False),
                NestedField(field_id=2, name="region", field_type=StringType(), required=False),
            ],
            partition_fields=[
                PartitionField(
                    source_id=2, field_id=1000, transform=IdentityTransform(), name="region"
                ),
            ],
        )
        data = pa.table({"id": [1, 2, 3], "region": ["US", "EU", "US"]})
        result = _build_partition_overwrite_filter(table, data)
        assert isinstance(result, In)

    def test_multi_identity_partition(self):
        table = _make_mock_table(
            schema_fields=[
                NestedField(field_id=1, name="id", field_type=IntegerType(), required=False),
                NestedField(field_id=2, name="region", field_type=StringType(), required=False),
                NestedField(field_id=3, name="category", field_type=StringType(), required=False),
            ],
            partition_fields=[
                PartitionField(
                    source_id=2, field_id=1000, transform=IdentityTransform(), name="region"
                ),
                PartitionField(
                    source_id=3, field_id=1001, transform=IdentityTransform(), name="category"
                ),
            ],
        )
        data = pa.table({
            "id": [1],
            "region": ["US"],
            "category": ["A"],
        })
        result = _build_partition_overwrite_filter(table, data)
        assert isinstance(result, And)

    def test_non_identity_transform_falls_back(self):
        table = _make_mock_table(
            schema_fields=[
                NestedField(field_id=1, name="id", field_type=IntegerType(), required=False),
                NestedField(field_id=2, name="region", field_type=StringType(), required=False),
            ],
            partition_fields=[
                PartitionField(
                    source_id=1, field_id=1000, transform=BucketTransform(num_buckets=4), name="id_bucket"
                ),
            ],
        )
        data = pa.table({"id": [1, 2], "region": ["US", "EU"]})
        result = _build_partition_overwrite_filter(table, data)
        assert isinstance(result, AlwaysTrue)

    def test_year_transform_falls_back(self):
        table = _make_mock_table(
            schema_fields=[
                NestedField(field_id=1, name="id", field_type=IntegerType(), required=False),
                NestedField(field_id=2, name="region", field_type=StringType(), required=False),
            ],
            partition_fields=[
                PartitionField(
                    source_id=2, field_id=1000, transform=YearTransform(), name="region_year"
                ),
            ],
        )
        data = pa.table({"id": [1], "region": ["US"]})
        result = _build_partition_overwrite_filter(table, data)
        assert isinstance(result, AlwaysTrue)

    def test_missing_column_falls_back(self):
        table = _make_mock_table(
            schema_fields=[
                NestedField(field_id=1, name="id", field_type=IntegerType(), required=False),
                NestedField(field_id=2, name="region", field_type=StringType(), required=False),
            ],
            partition_fields=[
                PartitionField(
                    source_id=2, field_id=1000, transform=IdentityTransform(), name="region"
                ),
            ],
        )
        # Data missing the partition column 'region'
        data = pa.table({"id": [1, 2]})
        result = _build_partition_overwrite_filter(table, data)
        assert isinstance(result, AlwaysTrue)

    def test_all_none_values_falls_back(self):
        table = _make_mock_table(
            schema_fields=[
                NestedField(field_id=1, name="id", field_type=IntegerType(), required=False),
                NestedField(field_id=2, name="region", field_type=StringType(), required=False),
            ],
            partition_fields=[
                PartitionField(
                    source_id=2, field_id=1000, transform=IdentityTransform(), name="region"
                ),
            ],
        )
        data = pa.table({"id": [1, 2], "region": pa.array([None, None], type=pa.string())})
        result = _build_partition_overwrite_filter(table, data)
        assert isinstance(result, AlwaysTrue)

    def test_exception_falls_back(self):
        """Any error in filter building should fall back to AlwaysTrue."""
        table = MagicMock()
        table.spec.side_effect = RuntimeError("boom")
        data = pa.table({"id": [1]})
        result = _build_partition_overwrite_filter(table, data)
        assert isinstance(result, AlwaysTrue)
