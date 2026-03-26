"""DataFrameWriterV2 — newer Spark write API for Iceberg tables."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from iceberg_spark.dataframe import DataFrame
    from iceberg_spark.session import IcebergSession


class DataFrameWriterV2:
    """Spark DataFrameWriterV2 API for writing to Iceberg tables.

    Usage:
        df.writeTo("db.table").append()
        df.writeTo("db.table").overwrite(lit(True))
        df.writeTo("db.table").create()
    """

    def __init__(self, df: DataFrame, table_name: str):
        self._df = df
        self._table_name = table_name
        self._provider: str | None = None
        self._properties: dict[str, str] = {}
        self._partition_cols: list[str] = []

    def using(self, provider: str) -> DataFrameWriterV2:
        """Specifies the data source provider.

        Only 'iceberg' is supported. Raises ValueError for other providers.
        """
        if provider.lower() != "iceberg":
            raise ValueError(
                f"Unsupported provider '{provider}'. "
                "iceberg_spark only supports Iceberg tables."
            )
        self._provider = provider
        return self

    def tableProperty(self, key: str, value: str) -> DataFrameWriterV2:
        """Adds a table property."""
        self._properties[key] = value
        return self

    def partitionedBy(self, *cols) -> DataFrameWriterV2:
        """Specifies partition columns."""
        self._partition_cols = [str(c) for c in cols]
        return self

    def option(self, key: str, value: Any) -> DataFrameWriterV2:
        """Sets an option."""
        self._properties[key] = str(value)
        return self

    def append(self) -> None:
        """Appends data to the table."""
        session = self._df._session
        if session is None:
            raise RuntimeError("writeTo().append() requires an active session")
        arrow_table = self._df._df.to_arrow_table()
        session._insert_into_table(self._table_name, arrow_table, overwrite=False)

    def overwrite(self, condition=None) -> None:
        """Overwrites data in the table."""
        session = self._df._session
        if session is None:
            raise RuntimeError("writeTo().overwrite() requires an active session")
        arrow_table = self._df._df.to_arrow_table()
        session._insert_into_table(self._table_name, arrow_table, overwrite=True)

    def overwritePartitions(self) -> None:
        """Overwrites only the partitions present in the DataFrame's data.

        For identity-partitioned tables, builds a partition-scoped filter so
        that only the affected partitions are replaced.  Unaffected partitions
        are preserved.  Falls back to a full overwrite for unpartitioned tables
        or non-identity transforms.
        """
        session = self._df._session
        if session is None:
            raise RuntimeError("writeTo().overwritePartitions() requires an active session")

        arrow_table = self._df._df.to_arrow_table()

        parts = self._table_name.split(".")
        short_name = parts[-1]
        table_ident = self._table_name if len(parts) >= 2 else f"default.{short_name}"

        from iceberg_spark.catalog_ops import (
            _build_partition_overwrite_filter,
            _invalidate_table_cache,
            _resolve_catalog_for_table,
        )

        catalog = _resolve_catalog_for_table(session, self._table_name)
        try:
            table = catalog.load_table(table_ident)
            overwrite_filter = _build_partition_overwrite_filter(table, arrow_table)
            table.overwrite(arrow_table, overwrite_filter=overwrite_filter)
            _invalidate_table_cache(session, short_name)
        except Exception as e:
            raise RuntimeError(
                f"Failed to overwrite partitions in table {self._table_name}: {e}"
            ) from e

    def create(self) -> None:
        """Creates the table and inserts data.

        Infers schema from the DataFrame, creates the table via PyIceberg,
        then appends the data. If partitionedBy() was called, the table
        is created with the specified partition spec.
        """
        session = self._df._session
        if session is None:
            raise RuntimeError("writeTo().create() requires an active session")
        arrow_table = self._df._df.to_arrow_table()

        from iceberg_spark.catalog_ops import (
            _build_identity_partition_spec,
            _pa_schema_to_iceberg,
        )

        partition_spec = _build_identity_partition_spec(
            self._partition_cols, arrow_table.schema
        )
        create_schema = _pa_schema_to_iceberg(arrow_table.schema) if partition_spec else arrow_table.schema
        kwargs: dict[str, Any] = {"schema": create_schema}
        if partition_spec:
            kwargs["partition_spec"] = partition_spec

        from iceberg_spark.catalog_ops import _resolve_catalog_for_table
        catalog = _resolve_catalog_for_table(session, self._table_name)
        catalog.create_table(self._table_name, **kwargs)

        # Append the data
        if len(arrow_table) > 0:
            session._insert_into_table(self._table_name, arrow_table, overwrite=False)

    def replace(self) -> None:
        """Drops the existing table and recreates it with the DataFrame's data."""
        session = self._df._session
        if session is None:
            raise RuntimeError("writeTo().replace() requires an active session")
        # Drop existing table
        from iceberg_spark.catalog_ops import _resolve_catalog_for_table
        catalog = _resolve_catalog_for_table(session, self._table_name)
        try:
            catalog.drop_table(self._table_name)
        except Exception:
            pass  # Table may not exist
        # Deregister from session cache
        from iceberg_spark.catalog_ops import _invalidate_table_cache
        short_name = self._table_name.split(".")[-1]
        _invalidate_table_cache(session, short_name)
        # Create and populate
        self.create()

    def createOrReplace(self) -> None:
        """Creates the table if it doesn't exist, or replaces it if it does."""
        self.replace()
