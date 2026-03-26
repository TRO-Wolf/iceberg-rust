"""PySpark-compatible DataFrameReader for loading data."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from iceberg_spark.dataframe import DataFrame
    from iceberg_spark.session import IcebergSession


class DataFrameReader:
    """Interface for loading data into a DataFrame.

    Supports loading Iceberg tables and common file formats.
    """

    def __init__(self, session: IcebergSession):
        self._session = session
        self._format = "iceberg"
        self._options: dict[str, Any] = {}
        self._schema = None

    def format(self, source: str) -> DataFrameReader:
        """Specifies the input data source format."""
        self._format = source
        return self

    def option(self, key: str, value: Any) -> DataFrameReader:
        """Adds a read option as a key-value pair."""
        self._options[key] = value
        return self

    def options(self, **kwargs) -> DataFrameReader:
        """Adds multiple read options as keyword arguments."""
        self._options.update(kwargs)
        return self

    def schema(self, schema) -> DataFrameReader:
        """Specifies the schema for the data source."""
        self._schema = schema
        return self

    def table(self, tableName: str) -> DataFrame:
        """Returns the specified Iceberg table as a DataFrame."""
        return self._session.table(tableName)

    def load(self, path: str | None = None) -> DataFrame:
        """Load data from a source."""
        if self._format == "iceberg" and path:
            return self._session.table(path)

        if self._format == "parquet" and path:
            from iceberg_spark.dataframe import DataFrame

            ctx = self._session._ctx
            ctx.register_parquet(f"_parquet_{id(path)}", path)
            return DataFrame(ctx.table(f"_parquet_{id(path)}"), self._session)

        if self._format == "csv" and path:
            from iceberg_spark.dataframe import DataFrame

            ctx = self._session._ctx
            ctx.register_csv(f"_csv_{id(path)}", path)
            return DataFrame(ctx.table(f"_csv_{id(path)}"), self._session)

        if self._format == "json" and path:
            from iceberg_spark.dataframe import DataFrame
            ctx = self._session._ctx
            ctx.register_json(f"_json_{id(path)}", path)
            return DataFrame(ctx.table(f"_json_{id(path)}"), self._session)

        if self._format == "text" and path:
            return self.text(path)

        raise NotImplementedError(
            f"Format '{self._format}' is not supported. "
            f"Supported formats: parquet, csv, json, text, iceberg. "
            f"For ORC files, convert to Parquet first."
        )

    def parquet(self, *paths: str) -> DataFrame:
        """Load a Parquet file."""
        return self.format("parquet").load(paths[0])

    def csv(self, *paths: str, **kwargs) -> DataFrame:
        """Load a CSV file."""
        return self.format("csv").load(paths[0])

    def json(self, *paths: str) -> DataFrame:
        """Load a JSON file."""
        return self.format("json").load(paths[0])

    def text(self, path: str) -> DataFrame:
        """Reads a text file and returns a DataFrame with a single 'value' column."""
        import pyarrow as pa
        with open(path, "r") as f:
            lines = f.readlines()
        arrow_table = pa.table({"value": [line.rstrip("\n") for line in lines]})
        temp_name = f"_text_{id(path)}"
        self._session._ctx.register_record_batches(temp_name, [arrow_table.to_batches()])
        from iceberg_spark.dataframe import DataFrame
        return DataFrame(self._session._ctx.table(temp_name), self._session)

    def orc(self, path: str) -> DataFrame:
        """ORC format is not supported."""
        raise NotImplementedError(
            "ORC format is not supported in single-node mode. "
            "Consider converting to Parquet: session.read.parquet('path/to/data.parquet')"
        )
