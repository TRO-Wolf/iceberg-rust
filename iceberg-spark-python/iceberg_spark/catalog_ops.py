"""DDL operations dispatcher — handles CREATE/DROP/ALTER via PyIceberg catalog API."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pyiceberg.catalog import Catalog

    from iceberg_spark.dataframe import DataFrame
    from iceberg_spark.session import IcebergSession

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exceptions — subclass RuntimeError for backward compatibility
# ---------------------------------------------------------------------------


class IcebergSparkError(RuntimeError):
    """Base exception for iceberg_spark operations."""
    pass


class TableNotFoundError(IcebergSparkError):
    """Raised when a referenced table does not exist."""
    pass


class DDLError(IcebergSparkError):
    """Raised when a DDL operation (CREATE/DROP/ALTER) fails."""
    pass


class DMLError(IcebergSparkError):
    """Raised when a DML operation (INSERT/UPDATE/DELETE/MERGE) fails."""
    pass


class SchemaError(IcebergSparkError):
    """Raised when a type or schema operation fails."""
    pass


def _resolve_catalog_for_table(session: "IcebergSession", table_name: str):
    """Return the correct PyIceberg Catalog for *table_name*.

    For 3-part names (``catalog.ns.table``) the named catalog is returned.
    For 2-part or bare names the current catalog is returned.
    """
    parts = table_name.split(".")
    if len(parts) >= 3 and parts[0] in session._catalogs:
        return session._catalogs[parts[0]]
    return session._catalog


def handle_create_table(session: IcebergSession, sql: str, table_name: str) -> DataFrame:
    """Handle CREATE TABLE by parsing the SQL and delegating to PyIceberg."""
    import pyarrow as pa

    from iceberg_spark.dataframe import DataFrame

    catalog = _resolve_catalog_for_table(session, table_name)
    ns, tbl = _split_table_name(table_name)

    # Parse column definitions from the first (...) group after the table name.
    # Strip PARTITIONED BY and anything after from the SQL before extracting columns.
    sql_for_cols = re.split(r"\)\s*PARTITIONED\s+BY", sql, flags=re.IGNORECASE)[0]
    if ")" not in sql_for_cols:
        sql_for_cols += ")"  # re-close the paren we split on
    col_match = re.search(r"\((.*)\)", sql_for_cols, re.DOTALL)
    if col_match:
        col_defs = col_match.group(1).strip()
        schema = _parse_column_defs(col_defs)

        # Check for IF NOT EXISTS
        if_not_exists = bool(re.search(r"IF\s+NOT\s+EXISTS", sql, re.IGNORECASE))

        # Parse PARTITIONED BY if present
        partition_spec = _parse_partition_by(sql, schema)

        try:
            # When partition spec is present, convert to PyIceberg Schema with
            # proper field IDs so partition source_id resolution works correctly.
            create_schema = _pa_schema_to_iceberg(schema) if partition_spec else schema
            kwargs: dict[str, Any] = {"schema": create_schema}
            if partition_spec is not None:
                kwargs["partition_spec"] = partition_spec
            if if_not_exists:
                catalog.create_table_if_not_exists(f"{ns}.{tbl}", **kwargs)
            else:
                catalog.create_table(f"{ns}.{tbl}", **kwargs)
        except Exception as e:
            raise DDLError(f"CREATE TABLE {table_name}: {e}") from e

    # Return empty DataFrame
    return _empty_result_df(session, "status", "Table created")


def handle_create_table_as_select(
    session: IcebergSession,
    table_name: str,
    select_query: str,
    if_not_exists: bool = False,
    middle_clause: str | None = None,
) -> DataFrame:
    """Handle CREATE TABLE ... AS SELECT ... (CTAS).

    Executes the SELECT query via DataFusion, infers the table schema from the
    Arrow result, creates the table via PyIceberg, then appends the data.
    """
    import pyarrow as pa

    from iceberg_spark.dataframe import DataFrame

    catalog = _resolve_catalog_for_table(session, table_name)
    ns, tbl = _split_table_name(table_name)

    # Check if table already exists when IF NOT EXISTS is specified
    if if_not_exists:
        try:
            catalog.load_table(f"{ns}.{tbl}")
            # Table exists — return without doing anything
            return _empty_result_df(session, "status", "Table already exists")
        except Exception:
            logger.debug("Table %s.%s does not exist yet, proceeding with CTAS", ns, tbl)

    # Execute the SELECT query to get Arrow data
    select_query = session._ensure_tables_registered(select_query)
    try:
        arrow_table = session._ctx.sql(select_query).to_arrow_table()
    except Exception as e:
        raise DMLError(f"CTAS SELECT for {table_name}: {e}") from e

    # Convert Arrow schema to PyIceberg schema and create the table
    arrow_schema = arrow_table.schema
    pa_schema = pa.schema([
        pa.field(f.name, f.type, nullable=f.nullable)
        for f in arrow_schema
    ])

    # Parse PARTITIONED BY from the middle clause if present
    partition_spec = None
    if middle_clause:
        partition_spec = _parse_partition_by(middle_clause, pa_schema)

    try:
        create_schema = _pa_schema_to_iceberg(pa_schema) if partition_spec else pa_schema
        kwargs: dict[str, Any] = {"schema": create_schema}
        if partition_spec is not None:
            kwargs["partition_spec"] = partition_spec
        if if_not_exists:
            ice_table = catalog.create_table_if_not_exists(f"{ns}.{tbl}", **kwargs)
        else:
            ice_table = catalog.create_table(f"{ns}.{tbl}", **kwargs)
    except Exception as e:
        raise DDLError(f"CREATE TABLE {table_name}: {e}") from e

    # Append data if the SELECT produced rows
    if len(arrow_table) > 0:
        try:
            ice_table.append(arrow_table)
        except Exception as e:
            raise DMLError(
                f"INSERT into newly created {table_name}: {e}"
            ) from e

    _invalidate_table_cache(session, tbl)
    return _empty_result_df(
        session, "status", f"Table created with {len(arrow_table)} row(s)"
    )


def _parse_tblproperties(props_str: str) -> dict[str, str]:
    """Parse TBLPROPERTIES key-value pairs from SQL.

    Handles: 'key1'='value1', 'key2'='value2'
    Also: "key"="value", key='value', dotted.key='value'
    """
    props: dict[str, str] = {}
    # Match: key = 'single-quoted value' OR key = "double-quoted value"
    pattern = re.compile(
        r"""['"]?([\w][\w.]*)['"]?\s*=\s*"""
        r"""(?:'([^']*)'|"([^"]*)")""",
    )
    for m in pattern.finditer(props_str):
        key = m.group(1)
        value = m.group(2) if m.group(2) is not None else m.group(3)
        props[key] = value
    return props


def _parse_tblproperties_keys(keys_str: str) -> list[str]:
    """Parse TBLPROPERTIES key names from UNSET clause."""
    keys: list[str] = []
    # Match: 'quoted-key' or "quoted-key" or unquoted_key
    pattern = re.compile(r"""'([^']+)'|"([^"]+)"|([\w][\w.]*)""")
    for m in pattern.finditer(keys_str):
        key = m.group(1) or m.group(2) or m.group(3)
        if key:
            keys.append(key)
    return keys


def handle_alter_table(session: IcebergSession, sql: str, table_name: str) -> DataFrame:
    """Handle ALTER TABLE — ADD/DROP/RENAME COLUMN, SET/UNSET TBLPROPERTIES via PyIceberg."""
    import re

    catalog = _resolve_catalog_for_table(session, table_name)
    ns, tbl = _split_table_name(table_name)

    try:
        table = catalog.load_table(f"{ns}.{tbl}")
    except Exception as e:
        raise TableNotFoundError(f"ALTER TABLE {table_name}: {e}") from e

    # ADD COLUMN
    add_m = re.search(
        r"ADD\s+COLUMNS?\s+([`\"\w]+)\s+(\w+(?:\([^)]+\))?)",
        sql,
        re.IGNORECASE,
    )
    if add_m:
        col_name = add_m.group(1).strip("`\"'")
        type_str = add_m.group(2)
        from iceberg_spark._internal.type_mapping import spark_type_from_name
        arrow_type = spark_type_from_name(type_str).to_arrow()
        ice_type = _arrow_to_iceberg_type(arrow_type)
        with table.update_schema() as update:
            update.add_column(col_name, ice_type)
        _invalidate_table_cache(session, tbl)
        return _empty_result_df(session, "status", f"Column {col_name} added")

    # DROP COLUMN
    drop_m = re.search(r"DROP\s+COLUMNS?\s+([`\"\w]+)", sql, re.IGNORECASE)
    if drop_m:
        col_name = drop_m.group(1).strip("`\"'")
        with table.update_schema() as update:
            update.delete_column(col_name)
        _invalidate_table_cache(session, tbl)
        return _empty_result_df(session, "status", f"Column {col_name} dropped")

    # RENAME COLUMN
    rename_m = re.search(
        r"RENAME\s+COLUMN\s+([`\"\w]+)\s+TO\s+([`\"\w]+)",
        sql,
        re.IGNORECASE,
    )
    if rename_m:
        old_name = rename_m.group(1).strip("`\"'")
        new_name = rename_m.group(2).strip("`\"'")
        with table.update_schema() as update:
            update.rename_column(old_name, new_name)
        _invalidate_table_cache(session, tbl)
        return _empty_result_df(session, "status", f"Column {old_name} renamed to {new_name}")

    # UNSET TBLPROPERTIES (must be checked before SET to avoid false match)
    unset_props_m = re.search(
        r"UNSET\s+TBLPROPERTIES\s*(IF\s+EXISTS\s*)?\((.+)\)",
        sql,
        re.IGNORECASE | re.DOTALL,
    )
    if unset_props_m:
        if_exists = unset_props_m.group(1) is not None
        keys_str = unset_props_m.group(2)
        keys = _parse_tblproperties_keys(keys_str)
        existing_props = table.metadata.properties if hasattr(table, "metadata") else {}
        if if_exists:
            keys = [k for k in keys if k in existing_props]
        if keys:
            with table.transaction() as txn:
                txn.remove_properties(*keys)
        _invalidate_table_cache(session, tbl)
        return _empty_result_df(session, "status", f"Properties removed: {', '.join(keys)}")

    # SET TBLPROPERTIES
    set_props_m = re.search(
        r"SET\s+TBLPROPERTIES\s*\((.+)\)",
        sql,
        re.IGNORECASE | re.DOTALL,
    )
    if set_props_m:
        props_str = set_props_m.group(1)
        props = _parse_tblproperties(props_str)
        with table.transaction() as txn:
            txn.set_properties(props)
        _invalidate_table_cache(session, tbl)
        return _empty_result_df(session, "status", f"Properties set: {', '.join(props.keys())}")

    raise NotImplementedError(
        "Unsupported ALTER TABLE operation. "
        "Supported: ADD COLUMN, DROP COLUMN, RENAME COLUMN, "
        "SET TBLPROPERTIES, UNSET TBLPROPERTIES."
    )


def handle_truncate_table(session: IcebergSession, table_name: str) -> DataFrame:
    """Handle TRUNCATE TABLE — delete all rows via PyIceberg overwrite."""
    import pyarrow as pa

    catalog = _resolve_catalog_for_table(session, table_name)
    ns, tbl = _split_table_name(table_name)

    try:
        table = catalog.load_table(f"{ns}.{tbl}")
    except Exception as e:
        raise TableNotFoundError(f"TRUNCATE TABLE {table_name}: {e}") from e

    arrow_schema = table.schema().as_arrow()
    empty = pa.table({name: pa.array([], type=t) for name, t in zip(
        arrow_schema.names, [arrow_schema.field(n).type for n in arrow_schema.names]
    )})

    try:
        from pyiceberg.expressions import AlwaysTrue
        table.overwrite(empty, overwrite_filter=AlwaysTrue())
    except Exception as e:
        raise DMLError(f"TRUNCATE TABLE {table_name}: {e}") from e

    _invalidate_table_cache(session, tbl)
    return _empty_result_df(session, "status", "Table truncated")


def handle_insert_into(
    session: IcebergSession,
    sql: str,
    table_name: str,
    overwrite: bool = False,
) -> DataFrame:
    """Handle INSERT INTO table VALUES (...) or INSERT INTO table SELECT ..."""
    import re

    import pyarrow as pa

    catalog = _resolve_catalog_for_table(session, table_name)
    ns, tbl = _split_table_name(table_name)

    try:
        table = catalog.load_table(f"{ns}.{tbl}")
    except Exception as e:
        raise TableNotFoundError(f"INSERT INTO {table_name}: {e}") from e

    # INSERT INTO table SELECT ...
    select_m = re.search(
        r"INSERT\s+(?:INTO|OVERWRITE)\s+(?:TABLE\s+)?\S+\s+(SELECT\b.*)",
        sql,
        re.IGNORECASE | re.DOTALL,
    )
    # INSERT INTO table VALUES (...)
    values_m = re.search(
        r"INSERT\s+(?:INTO|OVERWRITE)\s+(?:TABLE\s+)?\S+\s+VALUES\s*(\(.*)",
        sql,
        re.IGNORECASE | re.DOTALL,
    )

    if select_m:
        select_sql = select_m.group(1)
        select_sql = session._ensure_tables_registered(select_sql)
        arrow_table = session._ctx.sql(select_sql).to_arrow_table()
    elif values_m:
        values_clause = values_m.group(1)
        temp_sql = f"SELECT * FROM (VALUES {values_clause}) AS __t"
        arrow_table = session._ctx.sql(temp_sql).to_arrow_table()
    else:
        raise ValueError(f"Cannot parse INSERT statement: {sql}")

    # Align columns with the target table schema:
    # 1. Rename columns (VALUES produce column1, column2, etc.)
    # 2. Cast types to match (DataFusion may infer int64 where table expects int32)
    import pyarrow as pa
    table_schema = table.schema()
    table_col_names = [f.name for f in table_schema.fields]
    if (
        len(arrow_table.column_names) == len(table_col_names)
        and arrow_table.column_names != table_col_names
    ):
        arrow_table = arrow_table.rename_columns(table_col_names)
    # Cast to match target schema types (e.g., int64 → int32 for VALUES literals)
    target_arrow_schema = table.schema().as_arrow()
    if arrow_table.schema != target_arrow_schema:
        try:
            arrow_table = arrow_table.cast(target_arrow_schema)
        except Exception as cast_err:
            logger.warning(
                "Schema cast failed for INSERT into %s: %s. "
                "Proceeding with original types.", table_name, cast_err,
            )

    try:
        if overwrite:
            overwrite_filter = _build_partition_overwrite_filter(table, arrow_table)
            table.overwrite(arrow_table, overwrite_filter=overwrite_filter)
        else:
            table.append(arrow_table)
    except Exception as e:
        raise DMLError(f"INSERT INTO {table_name}: {e}") from e

    _invalidate_table_cache(session, tbl)
    return _empty_result_df(session, "status", f"{len(arrow_table)} row(s) inserted")


def _parse_partition_by(sql: str, schema) -> "PartitionSpec | None":
    """Parse PARTITIONED BY clause from SQL, supporting transform expressions.

    Supports: identity (bare column), bucket(N, col), year(col), month(col),
    day(col), hour(col), truncate(N, col).

    The schema can be a PyArrow Schema or a PyIceberg Schema. Field IDs are
    resolved from the schema's field positions (1-based) for PyArrow schemas,
    or from the actual field_id for PyIceberg schemas.
    """
    # Match PARTITIONED BY with balanced parentheses (handles nested parens)
    m = re.search(r"PARTITIONED\s+BY\s*\(", sql, re.IGNORECASE)
    if not m:
        return None

    # Extract the content inside PARTITIONED BY(...) handling nested parens
    start = m.end()
    depth = 1
    i = start
    while i < len(sql) and depth > 0:
        if sql[i] == "(":
            depth += 1
        elif sql[i] == ")":
            depth -= 1
        i += 1
    if depth != 0:
        return None
    partition_content = sql[start : i - 1].strip()
    if not partition_content:
        return None

    # Split on top-level commas (not inside nested parens)
    parts = _split_at_commas(partition_content)

    from pyiceberg.partitioning import PartitionField, PartitionSpec

    # Build a name→source_id map from the schema
    field_map = _schema_field_map(schema)

    fields = []
    for i, part in enumerate(parts):
        expr = part.strip()
        transform, col_name, partition_name = _parse_partition_transform(expr)
        key = col_name.lower()
        if key not in field_map:
            raise ValueError(
                f"Partition column '{col_name}' not found in table schema"
            )
        source_id = field_map[key]
        fields.append(
            PartitionField(
                source_id=source_id,
                field_id=1000 + i,
                transform=transform,
                name=partition_name,
            )
        )

    return PartitionSpec(*fields) if fields else None


def _schema_field_map(schema) -> dict[str, int]:
    """Build a lowercase name → field_id map from a schema.

    Works with both PyArrow schemas (1-based positional IDs) and
    PyIceberg schemas (actual field_id from the spec).
    """
    import pyarrow as pa

    field_map: dict[str, int] = {}
    if isinstance(schema, pa.Schema):
        for idx, field in enumerate(schema):
            field_map[field.name.lower()] = idx + 1
    else:
        # PyIceberg Schema — use actual field_id
        for field in schema.fields:
            field_map[field.name.lower()] = field.field_id
    return field_map


def _pa_schema_to_iceberg(pa_schema) -> "Schema":
    """Convert a PyArrow schema to a PyIceberg Schema with proper field IDs.

    PyIceberg's create_table with a PyArrow schema creates an intermediate schema
    with field_id=-1, which breaks partition spec assignment. This function creates
    a proper Schema with sequential field IDs (1, 2, 3, ...).
    """
    from pyiceberg.schema import Schema
    from pyiceberg.types import NestedField

    fields = []
    for idx, arrow_field in enumerate(pa_schema):
        ice_type = _arrow_to_iceberg_type(arrow_field.type)
        fields.append(
            NestedField(
                field_id=idx + 1,
                name=arrow_field.name,
                field_type=ice_type,
                required=not arrow_field.nullable,
            )
        )
    return Schema(*fields)


def _build_identity_partition_spec(column_names: list[str], pa_schema) -> "PartitionSpec | None":
    """Build a PartitionSpec with IdentityTransform for the given column names.

    Used by DataFrameWriter.partitionBy() and DataFrameWriterV2.partitionedBy()
    to create partition specs from plain column name lists.

    Args:
        column_names: List of column names to partition by.
        pa_schema: A PyArrow schema to resolve column positions from.

    Returns:
        PartitionSpec or None if column_names is empty.

    Raises:
        ValueError: If a column name is not found in the schema.
    """
    if not column_names:
        return None

    from pyiceberg.partitioning import PartitionField, PartitionSpec
    from pyiceberg.transforms import IdentityTransform

    # Build name→position map
    schema_cols = {field.name: idx + 1 for idx, field in enumerate(pa_schema)}

    fields = []
    for i, col_name in enumerate(column_names):
        if col_name not in schema_cols:
            raise ValueError(
                f"Partition column '{col_name}' not found in DataFrame schema. "
                f"Available columns: {list(schema_cols.keys())}"
            )
        fields.append(
            PartitionField(
                source_id=schema_cols[col_name],
                field_id=1000 + i,
                transform=IdentityTransform(),
                name=col_name,
            )
        )
    return PartitionSpec(*fields)


def _parse_partition_transform(expr: str):
    """Parse a single partition expression and return (transform, source_column, partition_name).

    Examples:
        "region"           → (IdentityTransform(), "region", "region")
        "bucket(16, id)"   → (BucketTransform(16), "id", "id_bucket")
        "year(ts)"         → (YearTransform(), "ts", "ts_year")
        "truncate(10, s)"  → (TruncateTransform(10), "s", "s_truncate")
    """
    from pyiceberg.transforms import (
        BucketTransform,
        DayTransform,
        HourTransform,
        IdentityTransform,
        MonthTransform,
        TruncateTransform,
        YearTransform,
    )

    expr = expr.strip()

    # Match transform(args) pattern
    func_match = re.match(r"(\w+)\s*\(([^)]+)\)", expr)
    if func_match:
        func_name = func_match.group(1).lower()
        args = [a.strip() for a in func_match.group(2).split(",")]

        if func_name == "bucket" and len(args) == 2:
            n = int(args[0])
            col = args[1]
            return BucketTransform(n), col, f"{col}_bucket"
        elif func_name == "truncate" and len(args) == 2:
            n = int(args[0])
            col = args[1]
            return TruncateTransform(n), col, f"{col}_truncate"
        elif func_name == "year" and len(args) == 1:
            return YearTransform(), args[0], f"{args[0]}_year"
        elif func_name == "month" and len(args) == 1:
            return MonthTransform(), args[0], f"{args[0]}_month"
        elif func_name == "day" and len(args) == 1:
            return DayTransform(), args[0], f"{args[0]}_day"
        elif func_name == "hour" and len(args) == 1:
            return HourTransform(), args[0], f"{args[0]}_hour"
        else:
            raise ValueError(f"Unsupported partition transform: {expr}")

    # Bare column name → identity transform
    return IdentityTransform(), expr, expr


def _arrow_to_iceberg_type(arrow_type):
    """Convert an Arrow type to the corresponding PyIceberg type.

    Supports primitives, decimals, and nested types (list, map, struct).
    Falls back to StringType with a warning for unmapped types.
    """
    import pyarrow as pa
    from pyiceberg.types import (
        BinaryType,
        BooleanType,
        DateType,
        DecimalType,
        DoubleType,
        FloatType,
        IntegerType,
        ListType,
        LongType,
        MapType,
        NestedField,
        StringType,
        StructType,
        TimeType,
        TimestampType,
        TimestamptzType,
    )

    mapping = {
        pa.bool_(): BooleanType(),
        pa.int8(): IntegerType(),
        pa.int16(): IntegerType(),
        pa.int32(): IntegerType(),
        pa.int64(): LongType(),
        pa.float32(): FloatType(),
        pa.float64(): DoubleType(),
        pa.string(): StringType(),
        pa.large_string(): StringType(),
        pa.binary(): BinaryType(),
        pa.large_binary(): BinaryType(),
        pa.date32(): DateType(),
        pa.date64(): DateType(),
        pa.time64("us"): TimeType(),
        pa.time64("ns"): TimeType(),
        pa.time32("ms"): TimeType(),
        pa.timestamp("us", tz="UTC"): TimestamptzType(),
        pa.timestamp("us"): TimestampType(),
        pa.timestamp("ns", tz="UTC"): TimestamptzType(),
        pa.timestamp("ns"): TimestampType(),
    }
    if arrow_type in mapping:
        return mapping[arrow_type]

    # Decimal types
    if hasattr(arrow_type, "precision") and hasattr(arrow_type, "scale"):
        return DecimalType(arrow_type.precision, arrow_type.scale)

    # Timestamp with non-UTC timezone → TimestamptzType
    if pa.types.is_timestamp(arrow_type):
        if arrow_type.tz is not None:
            return TimestamptzType()
        return TimestampType()

    # List / LargeList
    if pa.types.is_list(arrow_type) or pa.types.is_large_list(arrow_type):
        element_type = _arrow_to_iceberg_type(arrow_type.value_type)
        return ListType(
            element_id=1,
            element_type=element_type,
            element_required=not arrow_type.value_field.nullable,
        )

    # Map
    if pa.types.is_map(arrow_type):
        key_type = _arrow_to_iceberg_type(arrow_type.key_type)
        value_type = _arrow_to_iceberg_type(arrow_type.item_type)
        return MapType(
            key_id=1,
            key_type=key_type,
            value_id=2,
            value_type=value_type,
            value_required=not arrow_type.item_field.nullable,
        )

    # Struct
    if pa.types.is_struct(arrow_type):
        fields = []
        for i, field in enumerate(arrow_type):
            ice_field_type = _arrow_to_iceberg_type(field.type)
            fields.append(
                NestedField(
                    field_id=i + 1,
                    name=field.name,
                    field_type=ice_field_type,
                    required=not field.nullable,
                )
            )
        return StructType(*fields)

    # Fallback with warning
    logger.warning(
        "Unmapped Arrow type %s -- falling back to StringType", arrow_type
    )
    return StringType()


def handle_drop_table(session: IcebergSession, table_name: str, if_exists: bool = False) -> DataFrame:
    """Handle DROP TABLE by delegating to PyIceberg."""
    catalog = _resolve_catalog_for_table(session, table_name)
    ns, tbl = _split_table_name(table_name)

    try:
        catalog.drop_table(f"{ns}.{tbl}")
    except Exception as e:
        # Only silently ignore "table not found" for IF EXISTS.
        # Propagate real errors (network, permission, corruption).
        is_not_found = "NoSuchTable" in type(e).__name__ or "not found" in str(e).lower()
        if if_exists and is_not_found:
            logger.debug("DROP TABLE IF EXISTS %s: table not found, ignoring", table_name)
        else:
            raise DDLError(f"DROP TABLE {table_name}: {e}") from e

    _invalidate_table_cache(session, tbl)
    return _empty_result_df(session, "status", "Table dropped")


def handle_create_namespace(session: IcebergSession, namespace: str) -> DataFrame:
    """Handle CREATE DATABASE/NAMESPACE."""
    catalog = session._catalog
    try:
        catalog.create_namespace_if_not_exists(namespace)
    except Exception as e:
        raise DDLError(f"CREATE NAMESPACE {namespace}: {e}") from e
    return _empty_result_df(session, "status", "Namespace created")


def handle_drop_namespace(session: IcebergSession, namespace: str) -> DataFrame:
    """Handle DROP DATABASE/NAMESPACE."""
    catalog = session._catalog
    try:
        catalog.drop_namespace(namespace)
    except Exception as e:
        raise DDLError(f"DROP NAMESPACE {namespace}: {e}") from e
    return _empty_result_df(session, "status", "Namespace dropped")


def handle_show_tables(session: IcebergSession, namespace: str | None) -> DataFrame:
    """Handle SHOW TABLES."""
    import pyarrow as pa
    from datafusion import SessionContext

    catalog = session._catalog
    tables = []

    namespaces = (
        [tuple(namespace.split("."))] if namespace else catalog.list_namespaces()
    )
    for ns in namespaces:
        ns_name = ".".join(ns)
        try:
            for table_ident in catalog.list_tables(ns):
                tbl_name = table_ident.name if hasattr(table_ident, "name") else table_ident[-1]
                tables.append({"namespace": ns_name, "tableName": tbl_name})
        except Exception:
            logger.debug("Failed to list tables in namespace %s", ns_name)
            continue

    arrow_table = pa.table({
        "namespace": [t["namespace"] for t in tables],
        "tableName": [t["tableName"] for t in tables],
    })

    from iceberg_spark.dataframe import DataFrame

    ctx = SessionContext()
    ctx.register_record_batches("_show_tables", [arrow_table.to_batches()])
    return DataFrame(ctx.table("_show_tables"), session)


def handle_show_databases(session: IcebergSession) -> DataFrame:
    """Handle SHOW DATABASES/NAMESPACES."""
    import pyarrow as pa
    from datafusion import SessionContext

    catalog = session._catalog
    namespaces = catalog.list_namespaces()
    ns_names = [".".join(ns) for ns in namespaces]

    arrow_table = pa.table({"namespace": ns_names})

    from iceberg_spark.dataframe import DataFrame

    ctx = SessionContext()
    ctx.register_record_batches("_show_dbs", [arrow_table.to_batches()])
    return DataFrame(ctx.table("_show_dbs"), session)


def handle_describe_table(session: IcebergSession, table_name: str) -> DataFrame:
    """Handle DESCRIBE TABLE."""
    import pyarrow as pa
    from datafusion import SessionContext

    catalog = _resolve_catalog_for_table(session, table_name)
    ns, tbl = _split_table_name(table_name)

    try:
        table = catalog.load_table(f"{ns}.{tbl}")
    except Exception as e:
        raise TableNotFoundError(f"DESCRIBE TABLE {table_name}: {e}") from e

    schema = table.schema()
    col_names = []
    data_types = []
    comments = []

    for field in schema.fields:
        col_names.append(field.name)
        data_types.append(str(field.field_type))
        comments.append("")

    arrow_table = pa.table({
        "col_name": col_names,
        "data_type": data_types,
        "comment": comments,
    })

    from iceberg_spark.dataframe import DataFrame

    ctx = SessionContext()
    ctx.register_record_batches("_describe", [arrow_table.to_batches()])
    return DataFrame(ctx.table("_describe"), session)


# --- Helpers ---


def _build_partition_overwrite_filter(table, arrow_data):
    """Build a partition-scoped overwrite filter for identity-partitioned tables.

    For identity-partitioned tables, constructs an In/EqualTo filter targeting
    only the partitions present in the data.  Falls back to AlwaysTrue() for
    non-identity transforms, unpartitioned tables, or any error.

    Args:
        table: PyIceberg Table object.
        arrow_data: PyArrow Table with the data being written.

    Returns:
        PyIceberg BooleanExpression (In, EqualTo, And, or AlwaysTrue).
    """
    from pyiceberg.expressions import AlwaysTrue, And, EqualTo, In

    try:
        spec = table.spec()
        if spec.is_unpartitioned():
            return AlwaysTrue()

        filters = []
        for field in spec.fields:
            # Only optimise identity transforms
            from pyiceberg.transforms import IdentityTransform

            if not isinstance(field.transform, IdentityTransform):
                return AlwaysTrue()

            # Resolve the source column name from the table schema
            source_field = table.schema().find_field(field.source_id)
            col_name = source_field.name

            if col_name not in arrow_data.column_names:
                return AlwaysTrue()

            # Collect distinct non-None values for this partition column
            values = arrow_data.column(col_name).unique().to_pylist()
            values = [v for v in values if v is not None]

            if not values:
                return AlwaysTrue()
            elif len(values) == 1:
                filters.append(EqualTo(col_name, values[0]))
            else:
                filters.append(In(col_name, set(values)))

        if not filters:
            return AlwaysTrue()

        # Combine with And
        result = filters[0]
        for f in filters[1:]:
            result = And(result, f)

        return result
    except Exception:
        logger.debug("Partition filter build failed, using full overwrite")
        return AlwaysTrue()


def _invalidate_table_cache(session: IcebergSession, short_name: str) -> None:
    """Remove a table from both the Python tracking set and DataFusion's context.

    Must be called after any write operation (INSERT, UPDATE, DELETE, MERGE,
    TRUNCATE, ALTER) so that subsequent queries see the updated data.
    """
    session._registered_tables.pop(short_name, None)
    try:
        session._ctx.deregister_table(short_name)
    except Exception:
        logger.debug("Table %s not found in DataFusion cache during invalidation", short_name)


def _split_table_name(name: str) -> tuple[str, str]:
    """Split 'namespace.table' into (namespace, table). Default namespace is 'default'."""
    parts = name.split(".")
    if len(parts) >= 2:
        return ".".join(parts[:-1]), parts[-1]
    return "default", parts[0]


def _parse_column_defs(col_defs: str):
    """Parse SQL column definitions into a PyArrow schema."""
    import pyarrow as pa

    from iceberg_spark._internal.type_mapping import (
        _split_top_level,
        spark_type_from_name,
    )

    fields = []
    # Split by comma respecting <> nesting so complex types are kept intact
    for part in _split_top_level(col_defs, ","):
        part = part.strip()
        if not part:
            continue
        # Skip constraints like PRIMARY KEY, etc.
        if re.match(r"^\s*(PRIMARY|FOREIGN|UNIQUE|CHECK|CONSTRAINT)", part, re.IGNORECASE):
            continue
        tokens = part.split()
        if len(tokens) >= 2:
            name = tokens[0].strip("`\"'")
            # Rejoin remaining tokens to capture complex type strings like
            # "array<string>" or "map<string, int>" that may have been split
            type_str = " ".join(tokens[1:])
            # Strip trailing NOT NULL / NULL qualifiers for type parsing
            nullable = True
            upper_type = type_str.upper()
            if upper_type.endswith(" NOT NULL"):
                nullable = False
                type_str = type_str[: -len(" NOT NULL")].strip()
            elif upper_type.endswith(" NULL"):
                type_str = type_str[: -len(" NULL")].strip()
            # Handle types with parentheses like decimal(10,2)
            if "(" in type_str and "<" not in type_str:
                paren_match = re.search(r"(\w+\([^)]+\))", type_str)
                if paren_match:
                    type_str = paren_match.group(1)
            spark_type = spark_type_from_name(type_str)
            fields.append(pa.field(name, spark_type.to_arrow(), nullable=nullable))

    return pa.schema(fields)


def _empty_result_df(session: IcebergSession, col_name: str, value: str) -> DataFrame:
    """Create a single-row DataFrame with a status message."""
    import pyarrow as pa
    from datafusion import SessionContext

    from iceberg_spark.dataframe import DataFrame

    arrow_table = pa.table({col_name: [value]})
    ctx = SessionContext()
    ctx.register_record_batches("_result", [arrow_table.to_batches()])
    return DataFrame(ctx.table("_result"), session)


def handle_create_view(
    session: "IcebergSession",
    view_name: str,
    view_query: str,
    or_replace: bool = False,
) -> "DataFrame":
    """Handle CREATE VIEW — materialize the query result as a registered table."""
    from iceberg_spark.sql_preprocessor import preprocess

    # Check for existing view when OR REPLACE is not specified
    if not or_replace:
        try:
            session._ctx.table(view_name)
            raise DDLError(
                f"CREATE VIEW {view_name}: view already exists. Use CREATE OR REPLACE VIEW."
            )
        except KeyError:
            pass  # View doesn't exist — proceed
    else:
        try:
            session._ctx.deregister_table(view_name)
        except Exception:
            logger.debug("View %s not registered, nothing to deregister for OR REPLACE", view_name)

    # Apply Spark→DataFusion SQL transforms (NVL→COALESCE, RLIKE→~, etc.)
    inner_result = preprocess(view_query)
    transformed = inner_result.sql

    transformed = session._ensure_tables_registered(transformed)
    arrow_table = session._ctx.sql(transformed).to_arrow_table()
    session._ctx.register_record_batches(view_name, [arrow_table.to_batches()])
    return _empty_result_df(session, "status", f"View {view_name} created")


def handle_drop_view(session: "IcebergSession", view_name: str) -> "DataFrame":
    """Handle DROP VIEW — deregister from DataFusion context."""
    try:
        session._ctx.deregister_table(view_name)
    except Exception:
        logger.debug("View %s not registered, nothing to drop", view_name)
    return _empty_result_df(session, "status", f"View {view_name} dropped")


def handle_show_columns(session: "IcebergSession", table_name: str) -> "DataFrame":
    """Handle SHOW COLUMNS — return column names and types."""
    import pyarrow as pa
    from datafusion import SessionContext

    from iceberg_spark.dataframe import DataFrame

    short_name = table_name.split(".")[-1]
    names = None
    types = None

    # Try DataFusion context first (works for registered/temp tables)
    try:
        session._ensure_table_registered(short_name, table_name)
        df = session._ctx.table(short_name)
        schema = df.schema()
        names = [f.name for f in schema]
        types = [str(f.type) for f in schema]
    except Exception:
        logger.debug("SHOW COLUMNS: table %s not in DataFusion context, trying catalog", table_name)

    # Fall back to PyIceberg catalog
    if names is None and session._catalog is not None:
        try:
            ns, tbl = _split_table_name(table_name)
            iceberg_table = session._catalog.load_table(f"{ns}.{tbl}")
            iceberg_schema = iceberg_table.schema()
            names = [f.name for f in iceberg_schema.fields]
            types = [str(f.field_type) for f in iceberg_schema.fields]
        except Exception:
            logger.debug("SHOW COLUMNS: table %s not found in catalog either", table_name)

    if names is None:
        raise TableNotFoundError(f"SHOW COLUMNS: table '{table_name}' not found")

    arrow = pa.table({"col_name": names, "data_type": types})
    ctx = SessionContext()
    ctx.register_record_batches("_show_cols", [arrow.to_batches()])
    return DataFrame(ctx.table("_show_cols"), session)


def handle_explain(session: "IcebergSession", query: str) -> "DataFrame":
    """Handle EXPLAIN — return DataFusion's query plan."""
    from iceberg_spark.dataframe import DataFrame
    from iceberg_spark.sql_preprocessor import preprocess

    # Apply Spark→DataFusion SQL transformations to the inner query
    inner_result = preprocess(query)
    transformed = inner_result.sql

    transformed = session._ensure_tables_registered(transformed)
    df = session._ctx.sql(f"EXPLAIN {transformed}")
    return DataFrame(df, session)


def handle_metadata_table(
    session,
    table_name: str,
    metadata_type: str,
    temp_name: str,
    rewritten_sql: str,
) -> "DataFrame":
    """Handle SELECT * FROM table.snapshots / .manifests / .history / etc."""
    import pyarrow as pa
    from datafusion import SessionContext

    from iceberg_spark.dataframe import DataFrame

    catalog = _resolve_catalog_for_table(session, table_name)
    ns, tbl = _split_table_name(table_name)

    try:
        table = catalog.load_table(f"{ns}.{tbl}")
    except Exception as e:
        raise TableNotFoundError(f"Metadata query on {table_name}: {e}") from e

    try:
        inspect = table.inspect
    except AttributeError:
        inspect = None

    arrow_table = _get_metadata_arrow(table, inspect, metadata_type)

    # Register as in-memory DataFusion table and execute the rewritten SQL
    ctx = SessionContext()
    ctx.register_record_batches(temp_name, [arrow_table.to_batches()])
    df = ctx.sql(rewritten_sql)
    return DataFrame(df, session)


def _get_metadata_arrow(table, inspect, metadata_type: str):
    """Retrieve metadata as an Arrow table using PyIceberg inspect API."""
    import pyarrow as pa

    def _to_arrow(result):
        """Convert an inspect result to Arrow, handling both PyArrow tables and scan results."""
        if isinstance(result, pa.Table):
            return result
        return result.to_arrow()

    try:
        if metadata_type == "snapshots":
            return _to_arrow(inspect.snapshots()) if inspect else _snapshots_arrow(table)
        elif metadata_type == "manifests":
            return _to_arrow(inspect.manifests()) if inspect else _empty_arrow()
        elif metadata_type == "entries":
            return _to_arrow(inspect.entries()) if inspect else _empty_arrow()
        elif metadata_type == "files":
            return _to_arrow(inspect.files()) if inspect else _empty_arrow()
        elif metadata_type == "history":
            return _to_arrow(inspect.history()) if inspect else _history_arrow(table)
        elif metadata_type == "refs":
            return _to_arrow(inspect.refs()) if inspect else _empty_arrow()
        elif metadata_type == "schemas":
            return _schemas_arrow(table)
        elif metadata_type == "partition_specs":
            return _partition_specs_arrow(table)
        else:
            return _empty_arrow()
    except Exception:
        logger.debug("Inspect API failed for %s, falling back to metadata construction", metadata_type)
        if metadata_type == "snapshots":
            return _snapshots_arrow(table)
        elif metadata_type == "history":
            return _history_arrow(table)
        elif metadata_type == "schemas":
            return _schemas_arrow(table)
        return _empty_arrow()


def _snapshots_arrow(table):
    """Build snapshots Arrow table from PyIceberg metadata."""
    import pyarrow as pa

    rows = []
    for snap in table.metadata.snapshots:
        op_val = snap.summary.get("operation", "") if snap.summary else ""
        rows.append({
            "committed_at": snap.timestamp_ms,
            "snapshot_id": snap.snapshot_id,
            "parent_id": snap.parent_snapshot_id,
            "operation": str(op_val) if op_val else "",
            "manifest_list": snap.manifest_list or "",
            "summary": str(snap.summary) if snap.summary else "",
        })
    if not rows:
        return pa.table({
            "committed_at": pa.array([], type=pa.int64()),
            "snapshot_id": pa.array([], type=pa.int64()),
            "parent_id": pa.array([], type=pa.int64()),
            "operation": pa.array([], type=pa.string()),
            "manifest_list": pa.array([], type=pa.string()),
            "summary": pa.array([], type=pa.string()),
        })
    return pa.table({
        "committed_at": [r["committed_at"] for r in rows],
        "snapshot_id": [r["snapshot_id"] for r in rows],
        "parent_id": [r["parent_id"] for r in rows],
        "operation": [r["operation"] for r in rows],
        "manifest_list": [r["manifest_list"] for r in rows],
        "summary": [r["summary"] for r in rows],
    })


def _history_arrow(table):
    """Build history Arrow table from snapshots."""
    import pyarrow as pa

    rows = []
    for snap in table.metadata.snapshots:
        rows.append({
            "made_current_at": snap.timestamp_ms,
            "snapshot_id": snap.snapshot_id,
            "parent_id": snap.parent_snapshot_id,
            "is_current_ancestor": True,
        })
    if not rows:
        return pa.table({
            "made_current_at": pa.array([], type=pa.int64()),
            "snapshot_id": pa.array([], type=pa.int64()),
            "parent_id": pa.array([], type=pa.int64()),
            "is_current_ancestor": pa.array([], type=pa.bool_()),
        })
    return pa.table({
        "made_current_at": [r["made_current_at"] for r in rows],
        "snapshot_id": [r["snapshot_id"] for r in rows],
        "parent_id": [r["parent_id"] for r in rows],
        "is_current_ancestor": [r["is_current_ancestor"] for r in rows],
    })


def _schemas_arrow(table):
    """Build schemas Arrow table from PyIceberg metadata."""
    import pyarrow as pa

    rows = []
    for schema in table.metadata.schemas:
        rows.append({
            "schema_id": schema.schema_id,
            "schema": str(schema),
        })
    if not rows:
        return pa.table({
            "schema_id": pa.array([], type=pa.int32()),
            "schema": pa.array([], type=pa.string()),
        })
    return pa.table({
        "schema_id": [r["schema_id"] for r in rows],
        "schema": [r["schema"] for r in rows],
    })


def _partition_specs_arrow(table):
    """Build partition_specs Arrow table."""
    import pyarrow as pa

    rows = []
    for spec in table.metadata.partition_specs:
        rows.append({
            "spec_id": spec.spec_id,
            "partition_spec": str(spec),
        })
    if not rows:
        return pa.table({
            "spec_id": pa.array([], type=pa.int32()),
            "partition_spec": pa.array([], type=pa.string()),
        })
    return pa.table({
        "spec_id": [r["spec_id"] for r in rows],
        "partition_spec": [r["partition_spec"] for r in rows],
    })


def _empty_arrow():
    """Return an empty Arrow table with a message column."""
    import pyarrow as pa
    return pa.table({"message": pa.array([], type=pa.string())})


# ---------------------------------------------------------------------------
# Phase 4 — DML: DELETE FROM / UPDATE (copy-on-write via PyIceberg)
# ---------------------------------------------------------------------------


def _resolve_temp_view_arrow(session: "IcebergSession", name: str):
    """Try to read a table/view from session._ctx as Arrow.

    Returns a ``pa.Table`` if *name* exists in the session's DataFusion context
    (covers temp views created via ``createOrReplaceTempView`` and SQL
    ``CREATE VIEW``), or ``None`` if the name is not registered.
    """
    try:
        df = session._ctx.table(name)
        return df.to_arrow_table()
    except (KeyError, Exception):
        return None


def _register_referenced_tables(session: "IcebergSession", ctx, *sql_fragments: str | None) -> None:
    """Register tables/views referenced in SQL fragments into a DML context.

    DML handlers use a fresh ``SessionContext`` with only the target table
    registered under a temp name.  Subqueries like
    ``WHERE id IN (SELECT id FROM t2)`` need ``t2`` available in that fresh
    context.  This helper scans for FROM/JOIN references, resolves each from
    the session context (temp views) or the catalog, and registers the data
    as record batches in the provided *ctx*.
    """
    import re

    import pyarrow as pa

    pattern = r"(?:FROM|JOIN)\s+([a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)*)"
    seen: set[str] = set()
    for fragment in sql_fragments:
        if not fragment:
            continue
        for name in re.findall(pattern, fragment, re.IGNORECASE):
            if name.startswith("__"):  # Skip internal temp tables
                continue
            if name in seen:
                continue
            seen.add(name)
            parts = name.split(".")
            short_name = parts[-1]

            # 1. Check session context first (temp views + already-registered tables)
            arrow = _resolve_temp_view_arrow(session, short_name)
            if arrow is None:
                # 2. Try catalog via session's namespace-aware registration,
                #    then read back from session._ctx
                try:
                    session._ensure_table_registered(short_name, full_name=name)
                    arrow = _resolve_temp_view_arrow(session, short_name)
                except Exception:
                    pass

            if arrow is not None:
                batches = arrow.to_batches()
                if not batches:
                    schema = arrow.schema
                    empty_arrays = [pa.array([], type=f.type) for f in schema]
                    batches = [pa.record_batch(empty_arrays, schema=schema)]
                ctx.register_record_batches(short_name, [batches])
            else:
                logger.debug("Could not register referenced table %s (may be CTE/subquery alias)", name)


def handle_delete_from(
    session: "IcebergSession",
    table_name: str,
    where_clause: str | None,
) -> "DataFrame":
    """Handle DELETE FROM table [WHERE condition] using copy-on-write.

    Reads the full table as Arrow, filters out matching rows, and overwrites
    via PyIceberg table.overwrite().
    """
    import pyarrow as pa
    from datafusion import SessionContext

    catalog = _resolve_catalog_for_table(session, table_name)
    ns, tbl = _split_table_name(table_name)

    try:
        table = catalog.load_table(f"{ns}.{tbl}")
    except Exception as e:
        raise TableNotFoundError(f"DELETE FROM {table_name}: {e}") from e

    # Read current snapshot as Arrow
    try:
        arrow_data = table.scan().to_arrow()
    except Exception as e:
        raise DMLError(f"DELETE FROM {table_name} scan failed: {e}") from e

    if where_clause is None:
        # DELETE all rows — overwrite with empty table of same schema
        empty = pa.table(
            {name: pa.array([], type=arrow_data.schema.field(name).type)
             for name in arrow_data.schema.names}
        )
        try:
            from pyiceberg.expressions import AlwaysTrue
            table.overwrite(empty, overwrite_filter=AlwaysTrue())
        except Exception as e:
            raise DMLError(f"DELETE FROM {table_name} overwrite failed: {e}") from e
        _invalidate_table_cache(session, tbl)
        return _empty_result_df(session, "num_affected_rows", str(len(arrow_data)))

    if len(arrow_data) == 0:
        return _empty_result_df(session, "num_affected_rows", "0")

    # Use DataFusion to filter: keep rows that do NOT match the WHERE condition
    ctx = SessionContext()
    ctx.register_record_batches("__delete_tmp", [arrow_data.to_batches()])
    _register_referenced_tables(session, ctx, where_clause)
    keep_sql = f"SELECT * FROM __delete_tmp WHERE NOT ({where_clause})"
    try:
        kept_arrow = ctx.sql(keep_sql).collect()
        kept_table = pa.Table.from_batches(kept_arrow, schema=arrow_data.schema) if kept_arrow else arrow_data.slice(0, 0)
    except Exception as e:
        raise DMLError(f"DELETE FROM {table_name} WHERE {where_clause}: {e}") from e

    try:
        # Use the original (pre-delete) data to determine affected partitions
        overwrite_filter = _build_partition_overwrite_filter(table, arrow_data)
        table.overwrite(kept_table, overwrite_filter=overwrite_filter)
    except Exception as e:
        raise DMLError(f"DELETE FROM {table_name} overwrite failed: {e}") from e

    _invalidate_table_cache(session, tbl)
    num_deleted = len(arrow_data) - len(kept_table)
    return _empty_result_df(session, "num_affected_rows", str(num_deleted))


def handle_update(
    session: "IcebergSession",
    table_name: str,
    set_clause: str,
    where_clause: str | None,
) -> "DataFrame":
    """Handle UPDATE table SET col=expr [WHERE condition] using copy-on-write.

    Reads the full table as Arrow, applies SET expressions via a DataFusion
    CASE WHEN query, and overwrites via PyIceberg table.overwrite().
    """
    import pyarrow as pa
    from datafusion import SessionContext

    catalog = _resolve_catalog_for_table(session, table_name)
    ns, tbl = _split_table_name(table_name)

    try:
        table = catalog.load_table(f"{ns}.{tbl}")
    except Exception as e:
        raise TableNotFoundError(f"UPDATE {table_name}: {e}") from e

    try:
        arrow_data = table.scan().to_arrow()
    except Exception as e:
        raise DMLError(f"UPDATE {table_name} scan failed: {e}") from e

    if len(arrow_data) == 0:
        return _empty_result_df(session, "num_affected_rows", "0")

    # Parse SET clause: "col1 = expr1, col2 = expr2"
    set_pairs = _parse_set_clause(set_clause)
    set_dict = {col.split(".")[-1].strip().lower(): expr.strip() for col, expr in set_pairs}

    # Build SELECT with CASE WHEN for updated columns
    col_names = arrow_data.schema.names
    select_parts = []
    for col_name in col_names:
        if col_name.lower() in set_dict:
            expr = set_dict[col_name.lower()]
            if where_clause:
                select_parts.append(
                    f"CASE WHEN ({where_clause}) THEN ({expr}) ELSE {col_name} END AS {col_name}"
                )
            else:
                select_parts.append(f"({expr}) AS {col_name}")
        else:
            select_parts.append(col_name)

    update_sql = f"SELECT {', '.join(select_parts)} FROM __update_tmp"

    ctx = SessionContext()
    ctx.register_record_batches("__update_tmp", [arrow_data.to_batches()])
    _register_referenced_tables(session, ctx, where_clause, set_clause)
    try:
        result_batches = ctx.sql(update_sql).collect()
        if result_batches:
            # Use original schema to prevent nullable/type drift from expressions
            try:
                updated_table = pa.Table.from_batches(result_batches, schema=arrow_data.schema)
            except pa.lib.ArrowInvalid:
                # Schema mismatch (e.g., nullable vs non-null) — cast instead
                updated_table = pa.Table.from_batches(result_batches)
                updated_table = updated_table.cast(arrow_data.schema)
        else:
            updated_table = arrow_data.slice(0, 0)
    except Exception as e:
        raise DMLError(f"UPDATE {table_name} SET failed: {e}") from e

    try:
        # Use the original data to determine affected partitions
        overwrite_filter = _build_partition_overwrite_filter(table, arrow_data)
        table.overwrite(updated_table, overwrite_filter=overwrite_filter)
    except Exception as e:
        raise DMLError(f"UPDATE {table_name} overwrite failed: {e}") from e

    _invalidate_table_cache(session, tbl)
    if where_clause:
        count_result = ctx.sql(
            f"SELECT COUNT(*) AS c FROM __update_tmp WHERE {where_clause}"
        ).collect()
        num_affected = pa.Table.from_batches(count_result).column(0)[0].as_py()
    else:
        num_affected = len(arrow_data)
    return _empty_result_df(session, "num_affected_rows", str(num_affected))


def _split_at_commas(text: str) -> list[str]:
    """Split text by commas, respecting parentheses and string quotes."""
    parts: list[str] = []
    depth = 0
    in_quote: str | None = None
    current = ""
    for ch in text:
        if in_quote:
            current += ch
            if ch == in_quote:
                in_quote = None
            continue
        if ch in ("'", '"'):
            in_quote = ch
            current += ch
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(current.strip())
            current = ""
        else:
            current += ch
    if current.strip():
        parts.append(current.strip())
    return parts


def _parse_set_clause(set_str: str) -> list[tuple[str, str]]:
    """Parse 'col1 = expr1, col2 = expr2' into [(col1, expr1), (col2, expr2)].

    Handles commas inside parentheses (function calls) and string literals.
    """
    result = []
    for pair in _split_at_commas(set_str):
        eq_idx = pair.index("=")
        col = pair[:eq_idx].strip()
        expr = pair[eq_idx + 1:].strip()
        result.append((col, expr))
    return result


# ---------------------------------------------------------------------------
# Phase 4 — DML: MERGE INTO (copy-on-write via PyIceberg)
# ---------------------------------------------------------------------------

# Regex for splitting MERGE INTO on WHEN MATCHED / WHEN NOT MATCHED boundaries.
_WHEN_SPLIT = re.compile(r"\bWHEN\s+(?=(?:NOT\s+)?MATCHED\b)", re.IGNORECASE)

# Header: MERGE INTO target [AS] alias USING source [AS] alias ON condition
# Negative lookaheads prevent capturing keywords as aliases.
_MERGE_HEADER = re.compile(
    r"MERGE\s+INTO\s+(\S+)"
    r"(?:\s+(?:AS\s+)?(?!USING\b)(\w+))?"
    r"\s+USING\s+(\S+)"
    r"(?:\s+(?:AS\s+)?(?!ON\b)(\w+))?"
    r"\s+ON\s+(.+)$",
    re.IGNORECASE,
)


def _parse_merge_into(sql: str) -> dict:
    """Parse a MERGE INTO statement into its components.

    Returns dict with keys:
        target_table, target_alias, source_table, source_alias,
        on_condition, when_clauses (list of dicts).
    """
    normalized = " ".join(sql.split())
    segments = _WHEN_SPLIT.split(normalized)

    header_m = _MERGE_HEADER.match(segments[0].strip())
    if not header_m:
        raise DMLError(f"MERGE INTO: failed to parse statement: {sql[:120]}")

    result = {
        "target_table": header_m.group(1),
        "target_alias": header_m.group(2),
        "source_table": header_m.group(3),
        "source_alias": header_m.group(4),
        "on_condition": header_m.group(5).strip(),
        "when_clauses": [],
    }

    for segment in segments[1:]:
        clause = _parse_when_clause(segment.strip())
        if clause:
            result["when_clauses"].append(clause)

    if not result["when_clauses"]:
        raise DMLError("MERGE INTO: requires at least one WHEN clause")

    return result


def _parse_when_clause(text: str) -> dict | None:
    """Parse a single WHEN [NOT] MATCHED ... THEN ... clause."""
    m = re.match(
        r"(NOT\s+)?MATCHED(?:\s+AND\s+(.+?))?\s+THEN\s+(.+)$",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if not m:
        return None

    clause: dict = {
        "type": "not_matched" if m.group(1) else "matched",
        "condition": m.group(2).strip() if m.group(2) else None,
    }
    action_text = m.group(3).strip()

    if re.match(r"DELETE\b", action_text, re.IGNORECASE):
        clause["action"] = "delete"
    elif re.match(r"UPDATE\s+SET\b", action_text, re.IGNORECASE):
        clause["action"] = "update"
        set_m = re.match(
            r"UPDATE\s+SET\s+(.+)$", action_text, re.IGNORECASE | re.DOTALL
        )
        clause["set_clause"] = set_m.group(1).strip()
    elif re.match(r"INSERT\b", action_text, re.IGNORECASE):
        clause["action"] = "insert"
        insert_m = re.match(
            r"INSERT\s*(?:\((.+?)\))?\s*VALUES\s*\((.+)\)\s*$",
            action_text,
            re.IGNORECASE | re.DOTALL,
        )
        if insert_m:
            cols_str = insert_m.group(1)
            vals_str = insert_m.group(2)
            clause["columns"] = (
                [c.strip() for c in _split_at_commas(cols_str)] if cols_str else None
            )
            clause["values"] = [v.strip() for v in _split_at_commas(vals_str)]
        else:
            clause["columns"] = None
            clause["values"] = None
    else:
        return None

    return clause


def handle_merge_into(
    session: "IcebergSession",
    table_name: str,
    sql: str,
) -> "DataFrame":
    """Handle MERGE INTO using copy-on-write.

    Parses the MERGE statement, loads target and source as Arrow, builds a
    UNION ALL query in DataFusion that applies matched updates/deletes and
    not-matched inserts, then overwrites the target via PyIceberg.
    """
    import pyarrow as pa
    from datafusion import SessionContext

    merge = _parse_merge_into(sql)

    catalog = _resolve_catalog_for_table(session, table_name)
    ns, tbl = _split_table_name(table_name)

    try:
        target_iceberg = catalog.load_table(f"{ns}.{tbl}")
    except Exception as e:
        raise TableNotFoundError(f"MERGE INTO target {table_name}: {e}") from e

    try:
        target_arrow = target_iceberg.scan().to_arrow()
    except Exception as e:
        raise DMLError(f"MERGE INTO {table_name} target scan failed: {e}") from e

    source_name = merge["source_table"]
    source_tbl_short = source_name.split(".")[-1]
    # Try temp view / session context first
    source_arrow = _resolve_temp_view_arrow(session, source_tbl_short)
    if source_arrow is None:
        source_ns, source_tbl = _split_table_name(source_name)
        try:
            source_iceberg = catalog.load_table(f"{source_ns}.{source_tbl}")
            source_arrow = source_iceberg.scan().to_arrow()
        except Exception as e:
            raise TableNotFoundError(f"MERGE INTO source {source_name}: {e}") from e

    # Determine aliases (fall back to short table name)
    t_alias = merge["target_alias"] or tbl
    s_alias = merge["source_alias"] or source_tbl_short

    ctx = SessionContext()
    ctx.register_record_batches(t_alias, [target_arrow.to_batches()])
    ctx.register_record_batches(s_alias, [source_arrow.to_batches()])

    # Register any additional tables referenced in WHEN clause subqueries
    when_fragments = [merge["on_condition"]]
    for wc in merge["when_clauses"]:
        if wc.get("condition"):
            when_fragments.append(wc["condition"])
        if wc.get("set_clause"):
            when_fragments.append(wc["set_clause"])
    _register_referenced_tables(session, ctx, *when_fragments)

    on_cond = merge["on_condition"]
    target_cols = target_arrow.schema.names

    # Classify WHEN clauses
    matched_updates = [
        c for c in merge["when_clauses"]
        if c["type"] == "matched" and c["action"] == "update"
    ]
    matched_deletes = [
        c for c in merge["when_clauses"]
        if c["type"] == "matched" and c["action"] == "delete"
    ]
    not_matched_inserts = [
        c for c in merge["when_clauses"]
        if c["type"] == "not_matched" and c["action"] == "insert"
    ]

    query_parts: list[str] = []

    # --- Part 1: Matched target rows (update/keep, exclude deleted) ---
    has_any_matched = bool(matched_updates or matched_deletes)
    if has_any_matched:
        # Pre-parse SET clauses once per update clause (not per column)
        parsed_updates = []
        for upd in matched_updates:
            set_pairs = _parse_set_clause(upd["set_clause"])
            set_dict = {
                c.split(".")[-1].strip().lower(): e.strip()
                for c, e in set_pairs
            }
            parsed_updates.append((upd, set_dict))

        select_exprs: list[str] = []
        for col_name in target_cols:
            applied = False
            for upd, set_dict in parsed_updates:
                if col_name.lower() in set_dict:
                    expr = set_dict[col_name.lower()]
                    cond = upd.get("condition")
                    if cond:
                        select_exprs.append(
                            f"CASE WHEN ({cond}) THEN ({expr}) "
                            f"ELSE {t_alias}.{col_name} END AS {col_name}"
                        )
                    else:
                        select_exprs.append(f"({expr}) AS {col_name}")
                    applied = True
                    break
            if not applied:
                select_exprs.append(f"{t_alias}.{col_name}")

        matched_sql: str | None = (
            f"SELECT {', '.join(select_exprs)} "
            f"FROM {t_alias} INNER JOIN {s_alias} ON {on_cond}"
        )

        # Exclude rows targeted by DELETE clauses
        delete_conds: list[str] = []
        for d in matched_deletes:
            cond = d.get("condition")
            if cond:
                delete_conds.append(f"({cond})")
            else:
                # Unconditional delete — remove all matched rows
                delete_conds = []
                matched_sql = None
                break

        if matched_sql and delete_conds:
            matched_sql += f" WHERE NOT ({' OR '.join(delete_conds)})"

        if matched_sql:
            query_parts.append(matched_sql)

    # --- Part 2: Unmatched target rows (always kept) ---
    if has_any_matched:
        # Only unmatched rows — matched ones handled in Part 1
        unmatched_sql = (
            f"SELECT {', '.join(f'{t_alias}.{c}' for c in target_cols)} "
            f"FROM {t_alias} WHERE NOT EXISTS ("
            f"SELECT 1 FROM {s_alias} WHERE {on_cond})"
        )
    else:
        # No matched clauses — keep ALL target rows as-is
        unmatched_sql = (
            f"SELECT {', '.join(f'{t_alias}.{c}' for c in target_cols)} "
            f"FROM {t_alias}"
        )
    query_parts.append(unmatched_sql)

    # --- Part 3: Not-matched source rows (INSERT) ---
    for ins in not_matched_inserts:
        insert_select: list[str] = []
        if ins.get("columns") and ins.get("values"):
            col_map = {
                c.split(".")[-1].strip().lower(): v.strip()
                for c, v in zip(ins["columns"], ins["values"])
            }
            for col_name in target_cols:
                if col_name.lower() in col_map:
                    insert_select.append(
                        f"({col_map[col_name.lower()]}) AS {col_name}"
                    )
                else:
                    insert_select.append(f"NULL AS {col_name}")
        elif ins.get("values"):
            for i, col_name in enumerate(target_cols):
                if i < len(ins["values"]):
                    insert_select.append(
                        f"({ins['values'][i]}) AS {col_name}"
                    )
                else:
                    insert_select.append(f"NULL AS {col_name}")
        else:
            for col_name in target_cols:
                insert_select.append(f"{s_alias}.{col_name}")

        nm_sql = (
            f"SELECT {', '.join(insert_select)} FROM {s_alias} "
            f"WHERE NOT EXISTS (SELECT 1 FROM {t_alias} WHERE {on_cond})"
        )
        ins_cond = ins.get("condition")
        if ins_cond:
            nm_sql += f" AND ({ins_cond})"
        query_parts.append(nm_sql)

    final_sql = " UNION ALL ".join(f"({p})" for p in query_parts)

    try:
        result_batches = ctx.sql(final_sql).collect()
        result_table = (
            pa.Table.from_batches(result_batches, schema=target_arrow.schema)
            if result_batches
            else target_arrow.slice(0, 0)
        )
    except Exception as e:
        raise DMLError(f"MERGE INTO {table_name} query failed: {e}") from e

    try:
        from pyiceberg.expressions import AlwaysTrue

        target_iceberg.overwrite(result_table, overwrite_filter=AlwaysTrue())
    except Exception as e:
        raise DMLError(f"MERGE INTO {table_name} overwrite failed: {e}") from e

    _invalidate_table_cache(session, tbl)
    return _empty_result_df(
        session, "num_affected_rows", str(len(result_table))
    )


def handle_set_config(session: IcebergSession, key, value):
    """Handle SET key=value — update session config."""
    if key is None:
        # SET with no args: return all config
        import pyarrow as pa
        from iceberg_spark.dataframe import DataFrame
        config = session._config
        keys = list(config.keys()) if config else ["(empty)"]
        vals = [str(config[k]) for k in config.keys()] if config else [""]
        arrow = pa.table({"key": keys, "value": vals})
        temp_name = f"_set_{id(config)}"
        session._ctx.register_record_batches(temp_name, [arrow.to_batches()])
        return DataFrame(session._ctx.table(temp_name), session)
    session._config[key] = value
    return _empty_result_df(session, "status", f"SET {key}={value}")


def handle_use_database(session: IcebergSession, namespace):
    """Handle USE database — set current namespace."""
    session._current_namespace = namespace
    return _empty_result_df(session, "status", f"Using database {namespace}")


def handle_use_catalog(session: "IcebergSession", catalog_name: str) -> "DataFrame":
    """Handle USE CATALOG — switch the active catalog."""
    if catalog_name not in session._catalogs:
        raise IcebergSparkError(
            f"Catalog '{catalog_name}' not found. "
            f"Available: {list(session._catalogs.keys())}"
        )
    session._current_catalog_name = catalog_name
    session._current_namespace = None  # reset namespace on catalog switch
    return _empty_result_df(session, "status", f"Using catalog {catalog_name}")


def handle_show_catalogs(session: "IcebergSession") -> "DataFrame":
    """Handle SHOW CATALOGS — list all configured catalogs."""
    import pyarrow as pa

    from iceberg_spark.dataframe import DataFrame

    names = list(session._catalogs.keys())
    arrow = pa.table({"catalog": pa.array(names, type=pa.string())})
    temp_name = f"_catalogs_{id(session)}"
    session._ctx.register_record_batches(temp_name, [arrow.to_batches()])
    return DataFrame(session._ctx.table(temp_name), session)


def handle_show_create_table(session: IcebergSession, table_name):
    """Handle SHOW CREATE TABLE — reconstruct DDL."""
    import pyarrow as pa
    from iceberg_spark.dataframe import DataFrame

    ns, tbl = _split_table_name(table_name)
    catalog = _resolve_catalog_for_table(session, table_name)
    table = catalog.load_table(f"{ns}.{tbl}")
    schema = table.schema()

    # Reconstruct CREATE TABLE DDL
    columns = []
    for field in schema.fields:
        col_type = str(field.field_type).upper()
        nullable = "" if field.optional else " NOT NULL"
        columns.append(f"  {field.name} {col_type}{nullable}")
    col_defs = ",\n".join(columns)
    ddl = f"CREATE TABLE {table_name} (\n{col_defs}\n) USING iceberg"

    arrow = pa.table({"createtab_stmt": [ddl]})
    temp_name = f"_show_create_{id(table)}"
    session._ctx.register_record_batches(temp_name, [arrow.to_batches()])
    return DataFrame(session._ctx.table(temp_name), session)


def handle_show_tblproperties(session: IcebergSession, table_name):
    """Handle SHOW TBLPROPERTIES — return table properties."""
    import pyarrow as pa
    from iceberg_spark.dataframe import DataFrame

    ns, tbl = _split_table_name(table_name)
    catalog = _resolve_catalog_for_table(session, table_name)
    table = catalog.load_table(f"{ns}.{tbl}")
    props = table.metadata.properties if hasattr(table, 'metadata') else {}

    keys = list(props.keys()) if props else ["(none)"]
    vals = [str(props[k]) for k in props.keys()] if props else [""]

    arrow = pa.table({"key": keys, "value": vals})
    temp_name = f"_tblprops_{id(table)}"
    session._ctx.register_record_batches(temp_name, [arrow.to_batches()])
    return DataFrame(session._ctx.table(temp_name), session)


def handle_cache_table(session: IcebergSession, table_name):
    """Handle CACHE TABLE — no-op in single-node mode."""
    return _empty_result_df(session, "status", f"Table {table_name} cached (no-op)")


def handle_uncache_table(session: IcebergSession, table_name):
    """Handle UNCACHE TABLE — no-op in single-node mode."""
    return _empty_result_df(session, "status", f"Table {table_name} uncached (no-op)")


def handle_add_jar(session: IcebergSession):
    """Handle ADD JAR/FILE — no-op in single-node mode."""
    return _empty_result_df(session, "status", "Added (no-op)")
