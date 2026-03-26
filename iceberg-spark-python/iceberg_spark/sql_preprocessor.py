"""Spark SQL dialect preprocessor for DataFusion compatibility.

Transforms Spark-specific SQL syntax into DataFusion-compatible SQL,
and intercepts commands that need to be handled outside the SQL engine
(DDL, metadata queries, time travel).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto


class CommandType(Enum):
    """Classification of SQL command types."""
    SQL = auto()           # Pass through to DataFusion (possibly transformed)
    SHOW_TABLES = auto()   # SHOW TABLES
    SHOW_DATABASES = auto()  # SHOW DATABASES / SHOW NAMESPACES
    DESCRIBE = auto()      # DESCRIBE TABLE
    CREATE_TABLE = auto()  # CREATE TABLE
    DROP_TABLE = auto()    # DROP TABLE
    CREATE_NAMESPACE = auto()
    DROP_NAMESPACE = auto()
    ALTER_TABLE = auto()
    TIME_TRAVEL = auto()   # Query with TIMESTAMP AS OF / VERSION AS OF
    INSERT_INTO = auto()   # INSERT INTO table ...
    TRUNCATE = auto()      # TRUNCATE TABLE
    METADATA_TABLE = auto()  # SELECT * FROM table.snapshots / .manifests / etc.
    DELETE_FROM = auto()     # DELETE FROM table WHERE ...
    UPDATE = auto()          # UPDATE table SET col=val WHERE ...
    MERGE_INTO = auto()      # MERGE INTO target USING source ON ... WHEN ...
    CREATE_TABLE_AS_SELECT = auto()  # CREATE TABLE ... AS SELECT ...
    CREATE_VIEW = auto()     # CREATE [OR REPLACE] [TEMP] VIEW ... AS SELECT ...
    DROP_VIEW = auto()       # DROP VIEW [IF EXISTS] ...
    SHOW_COLUMNS = auto()    # SHOW COLUMNS FROM/IN table
    EXPLAIN = auto()         # EXPLAIN <query>
    SET_CONFIG = auto()      # SET key=value or SET (show all)
    USE_DATABASE = auto()    # USE [DATABASE|NAMESPACE] db
    USE_CATALOG = auto()     # USE CATALOG <name>
    SHOW_CATALOGS = auto()   # SHOW CATALOGS
    SHOW_CREATE_TABLE = auto()  # SHOW CREATE TABLE t
    SHOW_TBLPROPERTIES = auto()  # SHOW TBLPROPERTIES t
    CACHE_TABLE = auto()     # CACHE [LAZY] TABLE t
    UNCACHE_TABLE = auto()   # UNCACHE TABLE [IF EXISTS] t
    ADD_JAR = auto()         # ADD JAR/FILE/ARCHIVE ...


@dataclass
class PreprocessResult:
    """Result of preprocessing a SQL statement."""
    command_type: CommandType
    sql: str                         # Transformed SQL (for SQL type)
    table_name: str | None = None    # Relevant table name
    namespace: str | None = None     # Relevant namespace
    snapshot_id: int | None = None   # For time travel
    timestamp: str | None = None     # For time travel
    extra: dict = field(default_factory=dict)


# Compiled regex patterns
_USING_ICEBERG = re.compile(r"\s+USING\s+iceberg\b", re.IGNORECASE)
_TBLPROPERTIES = re.compile(r"\bTBLPROPERTIES\b", re.IGNORECASE)
_SHOW_TABLES = re.compile(r"^\s*SHOW\s+TABLES(?:\s+IN\s+(\S+))?\s*$", re.IGNORECASE)
_SHOW_DATABASES = re.compile(r"^\s*SHOW\s+(DATABASES|NAMESPACES|SCHEMAS)\s*$", re.IGNORECASE)
_DESCRIBE_TABLE = re.compile(r"^\s*DESC(?:RIBE)?\s+(?:TABLE\s+)?(?:EXTENDED\s+|FORMATTED\s+)?(\S+)\s*$", re.IGNORECASE)
_CREATE_TABLE_AS_SELECT = re.compile(
    r"^\s*CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(\S+)\s+(.+?\s+)?AS\s+(SELECT\b.+)$",
    re.IGNORECASE | re.DOTALL,
)
_CREATE_TABLE = re.compile(
    r"^\s*CREATE\s+(?:OR\s+REPLACE\s+)?(?:EXTERNAL\s+)?TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(\S+)",
    re.IGNORECASE,
)
_DROP_TABLE = re.compile(
    r"^\s*DROP\s+TABLE\s+(?:IF\s+EXISTS\s+)?(\S+)\s*$",
    re.IGNORECASE,
)
_CREATE_NAMESPACE = re.compile(
    r"^\s*CREATE\s+(?:DATABASE|NAMESPACE|SCHEMA)\s+(?:IF\s+NOT\s+EXISTS\s+)?(\S+)",
    re.IGNORECASE,
)
_DROP_NAMESPACE = re.compile(
    r"^\s*DROP\s+(?:DATABASE|NAMESPACE|SCHEMA)\s+(?:IF\s+EXISTS\s+)?(\S+)\s*$",
    re.IGNORECASE,
)
_TIMESTAMP_AS_OF = re.compile(
    r"FROM\s+(\S+)\s+TIMESTAMP\s+AS\s+OF\s+'([^']+)'",
    re.IGNORECASE,
)
_VERSION_AS_OF = re.compile(
    r"FROM\s+(\S+)\s+VERSION\s+AS\s+OF\s+(\d+)",
    re.IGNORECASE,
)
_NVL = re.compile(r"\bNVL\s*\(", re.IGNORECASE)
_RLIKE = re.compile(r"\bRLIKE\b", re.IGNORECASE)
_DATE_ADD = re.compile(r"\bdate_add\s*\(\s*([^,]+),\s*(\d+)\s*\)", re.IGNORECASE)
_DATEDIFF = re.compile(r"\bdatediff\s*\(\s*([^,]+),\s*([^)]+)\s*\)", re.IGNORECASE)
_INSERT_INTO = re.compile(
    r"^\s*INSERT\s+(?:INTO|OVERWRITE)\s+(?:TABLE\s+)?(\S+)\s*",
    re.IGNORECASE,
)
_TRUNCATE_TABLE = re.compile(
    r"^\s*TRUNCATE\s+TABLE\s+(\S+)\s*$",
    re.IGNORECASE,
)
_DELETE_FROM = re.compile(
    r"^\s*DELETE\s+FROM\s+(\S+)(?:\s+WHERE\s+(.+))?$",
    re.IGNORECASE | re.DOTALL,
)
_UPDATE = re.compile(
    r"^\s*UPDATE\s+(\S+)\s+SET\s+(.+?)(?:\s+WHERE\s+(.+))?$",
    re.IGNORECASE | re.DOTALL,
)
_MERGE_INTO = re.compile(
    r"^\s*MERGE\s+INTO\s+(\S+)\s+",
    re.IGNORECASE,
)
_CREATE_VIEW = re.compile(
    r"^\s*CREATE\s+(OR\s+REPLACE\s+)?(?:TEMP(?:ORARY)?\s+)?VIEW\s+(\S+)\s+AS\s+(.+)$",
    re.IGNORECASE | re.DOTALL,
)
_DROP_VIEW = re.compile(
    r"^\s*DROP\s+VIEW\s+(?:IF\s+EXISTS\s+)?(\S+)\s*$",
    re.IGNORECASE,
)
_SHOW_COLUMNS = re.compile(
    r"^\s*SHOW\s+COLUMNS\s+(?:FROM|IN)\s+(\S+)\s*$",
    re.IGNORECASE,
)
_EXPLAIN = re.compile(
    r"^\s*EXPLAIN\s+(.+)$",
    re.IGNORECASE | re.DOTALL,
)
_SET_CONFIG = re.compile(
    r"^\s*SET\s+(\S+)\s*=\s*(.+?)\s*$",
    re.IGNORECASE,
)
_SET_ALL = re.compile(r"^\s*SET\s*$", re.IGNORECASE)
_USE_CATALOG = re.compile(
    r"^\s*USE\s+CATALOG\s+(\S+)\s*$",
    re.IGNORECASE,
)
_SHOW_CATALOGS = re.compile(r"^\s*SHOW\s+CATALOGS\s*$", re.IGNORECASE)
_USE_DATABASE = re.compile(
    r"^\s*USE\s+(?:DATABASE\s+|NAMESPACE\s+)?(\S+)\s*$",
    re.IGNORECASE,
)
_SHOW_CREATE_TABLE = re.compile(
    r"^\s*SHOW\s+CREATE\s+TABLE\s+(\S+)\s*$",
    re.IGNORECASE,
)
_SHOW_TBLPROPERTIES = re.compile(
    r"^\s*SHOW\s+TBLPROPERTIES\s+(\S+)\s*$",
    re.IGNORECASE,
)
_CACHE_TABLE = re.compile(
    r"^\s*CACHE\s+(?:LAZY\s+)?TABLE\s+(\S+)\s*$",
    re.IGNORECASE,
)
_UNCACHE_TABLE = re.compile(
    r"^\s*UNCACHE\s+TABLE\s+(?:IF\s+EXISTS\s+)?(\S+)\s*$",
    re.IGNORECASE,
)
_ADD_JAR = re.compile(
    r"^\s*ADD\s+(?:JAR|FILE|ARCHIVE)\s+",
    re.IGNORECASE,
)

# Metadata table references: <table>.<metadata_type>
# Matches e.g. "t1.snapshots", "db.t1.snapshots" inside a FROM/JOIN clause
_METADATA_TYPES = (
    "snapshots", "manifests", "entries", "files",
    "partition_specs", "schemas", "history", "refs",
)
_METADATA_TABLE = re.compile(
    r"\bFROM\s+([\w.]+)\.(snapshots|manifests|entries|files|"
    r"partition_specs|schemas|history|refs)\b",
    re.IGNORECASE,
)


def preprocess(sql: str) -> PreprocessResult:
    """Preprocess a Spark SQL statement for DataFusion execution.

    Returns a PreprocessResult indicating the command type and any transformations.
    """
    stripped = sql.strip().rstrip(";")

    # --- Intercept metadata commands ---

    m = _SHOW_TABLES.match(stripped)
    if m:
        ns = m.group(1)
        return PreprocessResult(
            command_type=CommandType.SHOW_TABLES,
            sql=stripped,
            namespace=ns,
        )

    m = _SHOW_DATABASES.match(stripped)
    if m:
        return PreprocessResult(
            command_type=CommandType.SHOW_DATABASES,
            sql=stripped,
        )

    m = _SHOW_COLUMNS.match(stripped)
    if m:
        return PreprocessResult(
            command_type=CommandType.SHOW_COLUMNS,
            sql=stripped,
            table_name=m.group(1),
        )

    m = _DESCRIBE_TABLE.match(stripped)
    if m:
        return PreprocessResult(
            command_type=CommandType.DESCRIBE,
            sql=stripped,
            table_name=m.group(1),
        )

    # --- Intercept DDL ---

    m = _CREATE_VIEW.match(stripped)
    if m:
        or_replace = m.group(1) is not None
        return PreprocessResult(
            command_type=CommandType.CREATE_VIEW,
            sql=stripped,
            table_name=m.group(2),
            extra={"view_query": m.group(3).strip(), "or_replace": or_replace},
        )

    m = _CREATE_TABLE_AS_SELECT.match(stripped)
    if m:
        if_not_exists = bool(re.search(r"IF\s+NOT\s+EXISTS", stripped, re.IGNORECASE))
        middle_clause = (m.group(2) or "").strip()
        extra = {"select_query": m.group(3).strip(), "if_not_exists": if_not_exists}
        if middle_clause:
            extra["middle_clause"] = middle_clause
        return PreprocessResult(
            command_type=CommandType.CREATE_TABLE_AS_SELECT,
            sql=stripped,
            table_name=m.group(1),
            extra=extra,
        )

    m = _CREATE_TABLE.match(stripped)
    if m:
        return PreprocessResult(
            command_type=CommandType.CREATE_TABLE,
            sql=stripped,
            table_name=m.group(1),
        )

    m = _DROP_TABLE.match(stripped)
    if m:
        if_exists = bool(re.search(r"IF\s+EXISTS", stripped, re.IGNORECASE))
        return PreprocessResult(
            command_type=CommandType.DROP_TABLE,
            sql=stripped,
            table_name=m.group(1),
            extra={"if_exists": if_exists},
        )

    m = _DROP_VIEW.match(stripped)
    if m:
        return PreprocessResult(
            command_type=CommandType.DROP_VIEW,
            sql=stripped,
            table_name=m.group(1),
        )

    m = _CREATE_NAMESPACE.match(stripped)
    if m:
        return PreprocessResult(
            command_type=CommandType.CREATE_NAMESPACE,
            sql=stripped,
            namespace=m.group(1),
        )

    m = _DROP_NAMESPACE.match(stripped)
    if m:
        return PreprocessResult(
            command_type=CommandType.DROP_NAMESPACE,
            sql=stripped,
            namespace=m.group(1),
        )

    if re.match(r"^\s*ALTER\s+TABLE\b", stripped, re.IGNORECASE):
        table_m = re.match(r"^\s*ALTER\s+TABLE\s+(\S+)", stripped, re.IGNORECASE)
        return PreprocessResult(
            command_type=CommandType.ALTER_TABLE,
            sql=stripped,
            table_name=table_m.group(1) if table_m else None,
        )

    m = _TRUNCATE_TABLE.match(stripped)
    if m:
        return PreprocessResult(
            command_type=CommandType.TRUNCATE,
            sql=stripped,
            table_name=m.group(1),
        )

    m = _INSERT_INTO.match(stripped)
    if m:
        table_name = m.group(1)
        is_overwrite = bool(re.match(r"^\s*INSERT\s+OVERWRITE\b", stripped, re.IGNORECASE))
        return PreprocessResult(
            command_type=CommandType.INSERT_INTO,
            sql=stripped,
            table_name=table_name,
            extra={"overwrite": is_overwrite},
        )

    # --- DML: DELETE FROM / UPDATE ---

    m = _DELETE_FROM.match(stripped)
    if m:
        table_name = m.group(1)
        where_clause = m.group(2)
        return PreprocessResult(
            command_type=CommandType.DELETE_FROM,
            sql=stripped,
            table_name=table_name,
            extra={"where_clause": where_clause.strip() if where_clause else None},
        )

    m = _UPDATE.match(stripped)
    if m:
        table_name = m.group(1)
        set_clause = m.group(2).strip()
        where_clause = m.group(3)
        return PreprocessResult(
            command_type=CommandType.UPDATE,
            sql=stripped,
            table_name=table_name,
            extra={
                "set_clause": set_clause,
                "where_clause": where_clause.strip() if where_clause else None,
            },
        )

    m = _MERGE_INTO.match(stripped)
    if m:
        return PreprocessResult(
            command_type=CommandType.MERGE_INTO,
            sql=stripped,
            table_name=m.group(1),
        )

    m = _EXPLAIN.match(stripped)
    if m:
        return PreprocessResult(
            command_type=CommandType.EXPLAIN,
            sql=m.group(1).strip(),
        )

    # --- Session / utility commands ---

    m = _SET_CONFIG.match(stripped)
    if m:
        return PreprocessResult(
            command_type=CommandType.SET_CONFIG,
            sql=stripped,
            extra={"key": m.group(1), "value": m.group(2).strip()},
        )

    if _SET_ALL.match(stripped):
        return PreprocessResult(
            command_type=CommandType.SET_CONFIG,
            sql=stripped,
            extra={"key": None, "value": None},
        )

    # USE CATALOG must be matched before USE DATABASE (which would match "USE catalog_name")
    m = _USE_CATALOG.match(stripped)
    if m:
        return PreprocessResult(
            command_type=CommandType.USE_CATALOG,
            sql=stripped,
            extra={"catalog_name": m.group(1)},
        )

    if _SHOW_CATALOGS.match(stripped):
        return PreprocessResult(
            command_type=CommandType.SHOW_CATALOGS,
            sql=stripped,
        )

    m = _USE_DATABASE.match(stripped)
    if m:
        return PreprocessResult(
            command_type=CommandType.USE_DATABASE,
            sql=stripped,
            namespace=m.group(1),
        )

    m = _SHOW_CREATE_TABLE.match(stripped)
    if m:
        return PreprocessResult(
            command_type=CommandType.SHOW_CREATE_TABLE,
            sql=stripped,
            table_name=m.group(1),
        )

    m = _SHOW_TBLPROPERTIES.match(stripped)
    if m:
        return PreprocessResult(
            command_type=CommandType.SHOW_TBLPROPERTIES,
            sql=stripped,
            table_name=m.group(1),
        )

    m = _CACHE_TABLE.match(stripped)
    if m:
        return PreprocessResult(
            command_type=CommandType.CACHE_TABLE,
            sql=stripped,
            table_name=m.group(1),
        )

    m = _UNCACHE_TABLE.match(stripped)
    if m:
        return PreprocessResult(
            command_type=CommandType.UNCACHE_TABLE,
            sql=stripped,
            table_name=m.group(1),
        )

    if _ADD_JAR.match(stripped):
        return PreprocessResult(
            command_type=CommandType.ADD_JAR,
            sql=stripped,
        )

    # --- Metadata tables (table.snapshots, table.manifests, etc.) ---

    m = _METADATA_TABLE.search(stripped)
    if m:
        table_name = m.group(1)
        metadata_type = m.group(2).lower()
        # Rewrite FROM table.metadata_type → FROM __meta_<safe>_<type>
        safe = table_name.replace(".", "_")
        temp_name = f"__meta_{safe}_{metadata_type}"
        rewritten = _METADATA_TABLE.sub(f"FROM {temp_name}", stripped)
        return PreprocessResult(
            command_type=CommandType.METADATA_TABLE,
            sql=rewritten,
            table_name=table_name,
            extra={"metadata_type": metadata_type, "temp_name": temp_name},
        )

    # --- Time travel ---

    m = _TIMESTAMP_AS_OF.search(stripped)
    if m:
        table_name = m.group(1)
        timestamp = m.group(2)
        # Use the short (unqualified) table name for the time-travel alias,
        # since _register_time_travel_table registers with the short name.
        short_name = table_name.split(".")[-1]
        cleaned = _TIMESTAMP_AS_OF.sub(f"FROM {short_name}__time_travel", stripped)
        return PreprocessResult(
            command_type=CommandType.TIME_TRAVEL,
            sql=cleaned,
            table_name=table_name,
            timestamp=timestamp,
        )

    m = _VERSION_AS_OF.search(stripped)
    if m:
        table_name = m.group(1)
        snapshot_id = int(m.group(2))
        short_name = table_name.split(".")[-1]
        cleaned = _VERSION_AS_OF.sub(f"FROM {short_name}__time_travel", stripped)
        return PreprocessResult(
            command_type=CommandType.TIME_TRAVEL,
            sql=cleaned,
            table_name=table_name,
            snapshot_id=snapshot_id,
        )

    # --- SQL transformations (pass through to DataFusion) ---

    transformed = stripped

    # Strip USING iceberg
    transformed = _USING_ICEBERG.sub("", transformed)

    # TBLPROPERTIES -> OPTIONS
    transformed = _TBLPROPERTIES.sub("OPTIONS", transformed)

    # NVL(a, b) -> COALESCE(a, b)
    transformed = _NVL.sub("COALESCE(", transformed)

    # RLIKE -> ~
    transformed = _RLIKE.sub("~", transformed)

    # date_add(col, n) -> col + interval 'n days'
    transformed = _DATE_ADD.sub(r"(\1 + interval '\2 days')", transformed)

    # datediff(a, b) -> date_diff('day', b, a)  (Spark arg order is end, start)
    transformed = _DATEDIFF.sub(r"date_diff('day', \2, \1)", transformed)

    return PreprocessResult(
        command_type=CommandType.SQL,
        sql=transformed,
    )
