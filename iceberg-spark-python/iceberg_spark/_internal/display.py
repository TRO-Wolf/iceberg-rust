"""Display utilities for Spark-style output formatting."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pyarrow as pa

    from iceberg_spark.types import StructType


def format_table(table: pa.Table, n: int = 20, truncate: int | bool = 20) -> str:
    """Format an Arrow table as a Spark-style ASCII table.

    Args:
        table: Arrow table to format.
        n: Maximum number of rows to show.
        truncate: Max column width. True=20, False=no truncation, int=custom width.
    """
    if isinstance(truncate, bool):
        max_width = 20 if truncate else None
    else:
        max_width = max(3, truncate)

    sliced = table.slice(0, n)
    columns = sliced.column_names
    rows = sliced.to_pydict()

    # Build string columns
    str_cols: list[list[str]] = []
    for col_name in columns:
        values = rows[col_name]
        str_values = [_format_value(v) for v in values]
        str_cols.append([col_name] + str_values)

    # Apply truncation
    if max_width is not None:
        str_cols = [
            [_truncate(s, max_width) for s in col]
            for col in str_cols
        ]

    # Compute column widths
    col_widths = [max(len(s) for s in col) for col in str_cols]

    # Build output
    lines = []
    separator = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    lines.append(separator)

    # Header
    header = "|" + "|".join(
        f" {str_cols[i][0]:>{col_widths[i]}} " for i in range(len(columns))
    ) + "|"
    lines.append(header)
    lines.append(separator)

    # Data rows
    num_rows = len(str_cols[0]) - 1 if str_cols else 0
    for row_idx in range(num_rows):
        row = "|" + "|".join(
            f" {str_cols[i][row_idx + 1]:>{col_widths[i]}} "
            for i in range(len(columns))
        ) + "|"
        lines.append(row)
    lines.append(separator)

    total_rows = len(table)
    if total_rows > n:
        lines.append(f"only showing top {n} rows")
    lines.append("")

    return "\n".join(lines)


def format_schema(schema: StructType) -> str:
    """Format a StructType as a Spark-style printSchema tree."""
    lines = ["root"]
    for field in schema.fields:
        nullable = "true" if field.nullable else "false"
        lines.append(f" |-- {field.name}: {field.dataType} (nullable = {nullable})")
    lines.append("")
    return "\n".join(lines)


def _format_value(value) -> str:
    if value is None:
        return "null"
    return str(value)


def _truncate(s: str, max_width: int) -> str:
    if len(s) <= max_width:
        return s
    return s[: max_width - 3] + "..."
