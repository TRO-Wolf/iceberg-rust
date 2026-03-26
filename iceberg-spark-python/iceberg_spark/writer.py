"""PySpark-compatible DataFrameWriter for saving DataFrame contents."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from iceberg_spark.dataframe import DataFrame


class DataFrameWriter:
    """Interface for saving a DataFrame to external storage.

    Phase 1: Stub with API surface. Full write support comes in Phase 2.
    """

    def __init__(self, df: DataFrame):
        self._df = df
        self._format = "iceberg"
        self._mode = "append"
        self._options: dict[str, Any] = {}
        self._partition_by: list[str] = []

    def format(self, source: str) -> DataFrameWriter:
        """Specifies the output data source format."""
        self._format = source
        return self

    def mode(self, saveMode: str) -> DataFrameWriter:
        """Specifies the behavior when data already exists.

        Args:
            saveMode: 'append', 'overwrite', 'error', 'ignore'.
        """
        self._mode = saveMode
        return self

    def option(self, key: str, value: Any) -> DataFrameWriter:
        """Adds a write option as a key-value pair."""
        self._options[key] = value
        return self

    def options(self, **kwargs) -> DataFrameWriter:
        """Adds multiple write options as keyword arguments."""
        self._options.update(kwargs)
        return self

    def partitionBy(self, *cols: str) -> DataFrameWriter:
        """Partitions the output by the given columns."""
        self._partition_by = list(cols)
        return self

    def saveAsTable(self, name: str) -> None:
        """Saves the DataFrame as a table.

        Modes:
          append   — append rows to an existing table (or create)
          overwrite — replace all existing rows
          error    — raise if table already has data
          ignore   — skip write if table already has data
        """
        session = self._df._session
        if session is None:
            raise RuntimeError("saveAsTable requires an active IcebergSession")

        arrow_table = self._df._df.to_arrow_table()

        # Ensure the table exists — create it if needed (with partition spec)
        self._ensure_table_exists(session, name, arrow_table.schema)

        if self._mode == "append":
            session._insert_into_table(name, arrow_table, overwrite=False)

        elif self._mode == "overwrite":
            session._insert_into_table(name, arrow_table, overwrite=True)

        elif self._mode == "error":
            try:
                existing = session.table(name)
                if existing.count() > 0:
                    raise RuntimeError(
                        f"Table '{name}' already exists with data. "
                        "Use mode('append') or mode('overwrite')."
                    )
            except RuntimeError:
                raise
            except Exception:
                pass
            session._insert_into_table(name, arrow_table, overwrite=False)

        elif self._mode == "ignore":
            try:
                existing = session.table(name)
                if existing.count() > 0:
                    return
            except Exception:
                pass
            session._insert_into_table(name, arrow_table, overwrite=False)

        else:
            raise ValueError(f"Unknown save mode: {self._mode}")

    def _ensure_table_exists(self, session, name: str, arrow_schema) -> None:
        """Create the table if it doesn't exist, applying partition spec from partitionBy()."""
        from iceberg_spark.catalog_ops import (
            _build_identity_partition_spec,
            _pa_schema_to_iceberg,
            _resolve_catalog_for_table,
            _split_table_name,
        )

        catalog = _resolve_catalog_for_table(session, name)
        ns, tbl = _split_table_name(name)
        table_ident = f"{ns}.{tbl}"

        try:
            catalog.load_table(table_ident)
            return  # Table already exists
        except Exception:
            pass  # Table doesn't exist, create it

        import pyarrow as pa

        pa_schema = pa.schema([
            pa.field(f.name, f.type, nullable=f.nullable)
            for f in arrow_schema
        ])

        partition_spec = _build_identity_partition_spec(self._partition_by, pa_schema)

        # Use PyIceberg Schema when partition spec is present for proper field ID resolution
        create_schema = _pa_schema_to_iceberg(pa_schema) if partition_spec else pa_schema
        kwargs: dict[str, Any] = {"schema": create_schema}
        if partition_spec:
            kwargs["partition_spec"] = partition_spec

        catalog.create_table(table_ident, **kwargs)

    def insertInto(self, tableName: str) -> None:
        """Inserts the DataFrame into an existing table."""
        self.mode("append").saveAsTable(tableName)

    def save(self, path: str | None = None, **kwargs) -> None:
        """Save the DataFrame to a path or table.

        Format is controlled by self._format (set via .format()):
          - 'parquet', 'csv', 'json' → write to the given path
          - 'iceberg' → requires saveAsTable()
        """
        if path is None:
            raise ValueError(
                "path is required for path-based saves. "
                "Use saveAsTable() to write to an Iceberg table."
            )
        fmt = self._format.lower()
        df = self._df._df
        if fmt == "parquet":
            df.write_parquet(path)
        elif fmt == "csv":
            df.write_csv(path)
        elif fmt == "json":
            df.write_json(path)
        else:
            raise NotImplementedError(
                f"Format '{self._format}' does not support path-based saves. "
                "Use saveAsTable() for Iceberg format."
            )

    def parquet(self, path: str) -> None:
        """Save DataFrame as Parquet files at the given path."""
        self._df._df.write_parquet(path)

    def csv(self, path: str, header: bool = True, sep: str = ",") -> None:
        """Save DataFrame as CSV files at the given path."""
        self._df._df.write_csv(path)

    def json(self, path: str) -> None:
        """Save DataFrame as JSON files at the given path."""
        self._df._df.write_json(path)

    def bucketBy(self, numBuckets: int, *cols: str) -> DataFrameWriter:
        """No-op in single-node mode (bucketing is a distributed concept)."""
        return self

    def sortBy(self, *cols: str) -> DataFrameWriter:
        """Stores sort columns for ordered writes."""
        self._sort_cols = list(cols)
        return self

    def text(self, path: str) -> None:
        """Writes a single-column DataFrame as text file."""
        import os
        arrow_table = self._df._df.to_arrow_table()
        col_name = arrow_table.column_names[0]
        values = arrow_table.column(col_name).to_pylist()
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as f:
            for v in values:
                f.write(str(v) + "\n")

    def orc(self, path: str) -> None:
        """ORC format is not supported."""
        raise NotImplementedError(
            "ORC write is not supported in single-node mode. "
            "Use df.write.parquet('path/to/output') instead."
        )
