"""PySpark-compatible IcebergSession — the main entry point.

Provides a builder pattern and sql() interface that mirrors SparkSession,
backed by DataFusion for query execution and PyIceberg/iceberg-rust for
Iceberg table management.
"""

from __future__ import annotations

import logging
from typing import Any

import pyarrow as pa
from datafusion import SessionContext

logger = logging.getLogger(__name__)

from iceberg_spark._internal.catalog_factory import create_catalog
from iceberg_spark._internal.table_registration import register_iceberg_table
from iceberg_spark.dataframe import DataFrame
from iceberg_spark.sql_preprocessor import CommandType, preprocess


class IcebergSession:
    """PySpark-compatible session for querying Iceberg tables via DataFusion.

    Use IcebergSession.builder() to create a session.
    """

    _active_session: IcebergSession | None = None

    def __init__(
        self,
        ctx: SessionContext,
        catalog=None,
        catalog_name: str = "default",
        config: dict[str, Any] | None = None,
        *,
        catalogs: dict | None = None,
    ):
        self._ctx = ctx
        # Multi-catalog support: store all catalogs in a dict.
        # The _catalog / _catalog_name properties delegate to the current catalog.
        if catalogs is not None:
            self._catalogs: dict = dict(catalogs)
        elif catalog is not None:
            self._catalogs = {catalog_name: catalog}
        else:
            self._catalogs = {}
        self._current_catalog_name: str = catalog_name
        self._config = config or {}
        self._registered_tables: dict[str, str] = {}  # short_name -> full_name
        self._registered_udfs: dict[str, str] = {}  # name -> return_type_str
        self._current_namespace: str | None = None

    @property
    def _catalog(self):
        """Returns the currently active PyIceberg Catalog."""
        return self._catalogs.get(self._current_catalog_name)

    @_catalog.setter
    def _catalog(self, value):
        """Set the catalog for the current catalog name (backward compat)."""
        if not hasattr(self, "_catalogs"):
            self._catalogs = {}
        cat_name = getattr(self, "_current_catalog_name", "default")
        self._catalogs[cat_name] = value

    @property
    def _catalog_name(self):
        """Returns the current catalog name."""
        return self._current_catalog_name

    @_catalog_name.setter
    def _catalog_name(self, value: str):
        """Set the current catalog name (backward compat)."""
        # If renaming, migrate the catalog entry in the dict
        old_name = getattr(self, "_current_catalog_name", None)
        catalogs = getattr(self, "_catalogs", None)
        if catalogs and old_name and old_name in catalogs and old_name != value:
            catalogs[value] = catalogs.pop(old_name)
        self._current_catalog_name = value

    @staticmethod
    def builder() -> IcebergSessionBuilder:
        """Creates a builder for IcebergSession."""
        return IcebergSessionBuilder()

    def sql(self, query: str, args: dict[str, Any] | None = None, **kwargs) -> DataFrame:
        """Execute a SQL query and return a DataFrame.

        Supports Spark SQL syntax — statements are preprocessed to handle
        Spark-specific extensions (USING iceberg, time travel, DDL, etc.)
        before being dispatched to DataFusion or the catalog API.

        Args:
            query: SQL query string.
            args: Optional dict of named parameters.  ``:key`` placeholders
                  in the query are replaced with escaped literal values.
        """
        if args:
            for key, value in args.items():
                if isinstance(value, str):
                    escaped = value.replace("'", "''")
                    query = query.replace(f":{key}", f"'{escaped}'")
                elif value is None:
                    query = query.replace(f":{key}", "NULL")
                else:
                    query = query.replace(f":{key}", str(value))

        result = preprocess(query)

        # --- Metadata commands ---
        if result.command_type == CommandType.SHOW_TABLES:
            from iceberg_spark.catalog_ops import handle_show_tables

            return handle_show_tables(self, result.namespace)

        if result.command_type == CommandType.SHOW_DATABASES:
            from iceberg_spark.catalog_ops import handle_show_databases

            return handle_show_databases(self)

        if result.command_type == CommandType.DESCRIBE:
            from iceberg_spark.catalog_ops import handle_describe_table

            return handle_describe_table(self, result.table_name)

        if result.command_type == CommandType.SHOW_COLUMNS:
            from iceberg_spark.catalog_ops import handle_show_columns

            return handle_show_columns(self, result.table_name)

        # --- DDL commands ---
        if result.command_type == CommandType.CREATE_VIEW:
            from iceberg_spark.catalog_ops import handle_create_view

            return handle_create_view(
                self,
                result.table_name,
                result.extra["view_query"],
                or_replace=result.extra.get("or_replace", False),
            )

        if result.command_type == CommandType.CREATE_TABLE_AS_SELECT:
            from iceberg_spark.catalog_ops import handle_create_table_as_select

            return handle_create_table_as_select(
                self,
                table_name=result.table_name,
                select_query=result.extra["select_query"],
                if_not_exists=result.extra.get("if_not_exists", False),
                middle_clause=result.extra.get("middle_clause"),
            )

        if result.command_type == CommandType.CREATE_TABLE:
            from iceberg_spark.catalog_ops import handle_create_table

            return handle_create_table(self, result.sql, result.table_name)

        if result.command_type == CommandType.DROP_TABLE:
            from iceberg_spark.catalog_ops import handle_drop_table

            return handle_drop_table(
                self, result.table_name,
                if_exists=result.extra.get("if_exists", False) if result.extra else False,
            )

        if result.command_type == CommandType.DROP_VIEW:
            from iceberg_spark.catalog_ops import handle_drop_view

            return handle_drop_view(self, result.table_name)

        if result.command_type == CommandType.CREATE_NAMESPACE:
            from iceberg_spark.catalog_ops import handle_create_namespace

            return handle_create_namespace(self, result.namespace)

        if result.command_type == CommandType.DROP_NAMESPACE:
            from iceberg_spark.catalog_ops import handle_drop_namespace

            return handle_drop_namespace(self, result.namespace)

        if result.command_type == CommandType.ALTER_TABLE:
            from iceberg_spark.catalog_ops import handle_alter_table

            return handle_alter_table(self, result.sql, result.table_name)

        if result.command_type == CommandType.TRUNCATE:
            from iceberg_spark.catalog_ops import handle_truncate_table

            return handle_truncate_table(self, result.table_name)

        if result.command_type == CommandType.METADATA_TABLE:
            from iceberg_spark.catalog_ops import handle_metadata_table
            return handle_metadata_table(
                self,
                table_name=result.table_name,
                metadata_type=result.extra['metadata_type'],
                temp_name=result.extra['temp_name'],
                rewritten_sql=result.sql,
            )

        if result.command_type == CommandType.DELETE_FROM:
            from iceberg_spark.catalog_ops import handle_delete_from

            return handle_delete_from(
                self,
                table_name=result.table_name,
                where_clause=result.extra.get("where_clause"),
            )

        if result.command_type == CommandType.UPDATE:
            from iceberg_spark.catalog_ops import handle_update

            return handle_update(
                self,
                table_name=result.table_name,
                set_clause=result.extra["set_clause"],
                where_clause=result.extra.get("where_clause"),
            )

        if result.command_type == CommandType.MERGE_INTO:
            from iceberg_spark.catalog_ops import handle_merge_into

            return handle_merge_into(
                self,
                table_name=result.table_name,
                sql=result.sql,
            )

        if result.command_type == CommandType.INSERT_INTO:
            from iceberg_spark.catalog_ops import handle_insert_into

            return handle_insert_into(
                self,
                result.sql,
                result.table_name,
                overwrite=result.extra.get("overwrite", False),
            )

        if result.command_type == CommandType.EXPLAIN:
            from iceberg_spark.catalog_ops import handle_explain

            return handle_explain(self, result.sql)

        # --- Session / utility commands ---
        if result.command_type == CommandType.SET_CONFIG:
            from iceberg_spark.catalog_ops import handle_set_config
            return handle_set_config(
                self,
                result.extra.get("key"),
                result.extra.get("value"),
            )

        if result.command_type == CommandType.USE_CATALOG:
            from iceberg_spark.catalog_ops import handle_use_catalog
            return handle_use_catalog(self, result.extra["catalog_name"])

        if result.command_type == CommandType.SHOW_CATALOGS:
            from iceberg_spark.catalog_ops import handle_show_catalogs
            return handle_show_catalogs(self)

        if result.command_type == CommandType.USE_DATABASE:
            from iceberg_spark.catalog_ops import handle_use_database
            return handle_use_database(self, result.namespace)

        if result.command_type == CommandType.SHOW_CREATE_TABLE:
            from iceberg_spark.catalog_ops import handle_show_create_table
            return handle_show_create_table(self, result.table_name)

        if result.command_type == CommandType.SHOW_TBLPROPERTIES:
            from iceberg_spark.catalog_ops import handle_show_tblproperties
            return handle_show_tblproperties(self, result.table_name)

        if result.command_type == CommandType.CACHE_TABLE:
            from iceberg_spark.catalog_ops import handle_cache_table
            return handle_cache_table(self, result.table_name)

        if result.command_type == CommandType.UNCACHE_TABLE:
            from iceberg_spark.catalog_ops import handle_uncache_table
            return handle_uncache_table(self, result.table_name)

        if result.command_type == CommandType.ADD_JAR:
            from iceberg_spark.catalog_ops import handle_add_jar
            return handle_add_jar(self)

        # --- Time travel ---
        if result.command_type == CommandType.TIME_TRAVEL:
            self._register_time_travel_table(
                result.table_name,
                snapshot_id=result.snapshot_id,
                timestamp=result.timestamp,
            )
            df = self._ctx.sql(result.sql)
            return DataFrame(df, self)

        # --- Regular SQL ---
        # Ensure referenced tables are registered and rewrite qualified names
        rewritten_sql = self._ensure_tables_registered(result.sql)

        df = self._ctx.sql(rewritten_sql)
        return DataFrame(df, self)

    def table(self, tableName: str) -> DataFrame:
        """Returns the specified table as a DataFrame."""
        short_name = tableName.split(".")[-1]
        self._ensure_table_registered(short_name, full_name=tableName)
        df = self._ctx.table(short_name)
        return DataFrame(df, self)

    def createDataFrame(self, data, schema=None) -> DataFrame:
        """Creates a DataFrame from a list of rows or a Pandas DataFrame.

        Args:
            data: List of tuples/dicts, Pandas DataFrame, or Arrow Table.
            schema: Optional schema (StructType, list of column names, or Arrow schema).
        """
        if isinstance(data, pa.Table):
            arrow_table = data
        elif hasattr(data, "to_arrow"):
            # Pandas DataFrame
            arrow_table = pa.Table.from_pandas(data)
        else:
            # List of tuples/dicts
            if schema is not None:
                from iceberg_spark.types import StructType

                if isinstance(schema, StructType):
                    arrow_schema = pa.schema([f.to_arrow() for f in schema.fields])
                    col_names = [f.name for f in schema.fields]
                elif isinstance(schema, list) and all(isinstance(s, str) for s in schema):
                    col_names = schema
                    arrow_schema = None
                elif isinstance(schema, pa.Schema):
                    arrow_schema = schema
                    col_names = [f.name for f in schema]
                else:
                    raise TypeError(f"Unsupported schema type: {type(schema)}")
            else:
                col_names = None
                arrow_schema = None

            if data and isinstance(data[0], dict):
                if col_names is None:
                    col_names = list(data[0].keys())
                columns = {k: [row.get(k) for row in data] for k in col_names}
                arrow_table = pa.table(columns, schema=arrow_schema)
            elif data and isinstance(data[0], (list, tuple)):
                if col_names is None:
                    col_names = [f"_{i}" for i in range(len(data[0]))]
                columns = {
                    col_names[i]: [row[i] for row in data]
                    for i in range(len(col_names))
                }
                arrow_table = pa.table(columns, schema=arrow_schema)
            else:
                raise TypeError(f"Cannot create DataFrame from {type(data)}")

        ctx = self._ctx
        temp_name = f"_temp_{id(arrow_table)}"
        ctx.register_record_batches(temp_name, [arrow_table.to_batches()])
        return DataFrame(ctx.table(temp_name), self)

    def range(
        self,
        start: int,
        end: int | None = None,
        step: int = 1,
        numPartitions: int | None = None,
    ) -> DataFrame:
        """Creates a DataFrame with a single 'id' Long column.

        Args:
            start: Start (inclusive). If *end* is omitted this is the exclusive end.
            end: Exclusive end.
            step: Step between values.
        """
        if end is None:
            start, end = 0, start
        values = list(range(start, end, step))
        arrow_table = pa.table({"id": pa.array(values, type=pa.int64())})
        temp_name = f"_range_{id(values)}"
        self._ctx.register_record_batches(temp_name, [arrow_table.to_batches()])
        return DataFrame(self._ctx.table(temp_name), self)

    def read(self):
        """Returns a DataFrameReader for loading data."""
        from iceberg_spark.reader import DataFrameReader

        return DataFrameReader(self)

    @property
    def catalog(self):
        """Access the Spark-compatible catalog API (session.catalog.listTables etc.)."""
        from iceberg_spark.catalog_api import IcebergCatalogAPI
        return IcebergCatalogAPI(self)

    def stop(self) -> None:
        """Stop the session and clear it as the active session."""
        if IcebergSession._active_session is self:
            IcebergSession._active_session = None

    @property
    def version(self) -> str:
        """Returns the version of iceberg_spark."""
        try:
            from importlib.metadata import version
            return version("iceberg-spark")
        except Exception:
            logger.debug("Could not read package version from metadata, using default")
            return "0.1.0"

    @classmethod
    def getActiveSession(cls) -> IcebergSession | None:
        """Returns the most recently created session, or None."""
        return cls._active_session

    def newSession(self) -> IcebergSession:
        """Returns a new session sharing the same underlying catalog(s)."""
        return IcebergSession(
            ctx=SessionContext(),
            catalog_name=self._current_catalog_name,
            config=dict(self._config),
            catalogs=self._catalogs,
        )

    @property
    def conf(self) -> RuntimeConfig:
        """Access the runtime configuration."""
        return RuntimeConfig(self._config)

    @property
    def sparkContext(self) -> SparkContextStub:
        """Returns a stub SparkContext for compatibility."""
        return SparkContextStub(self)

    @property
    def udf(self) -> UDFRegistration:
        """Returns UDF registration backed by DataFusion ScalarUDF."""
        return UDFRegistration(self)

    @property
    def readStream(self):
        """Structured Streaming is not supported in single-node mode."""
        raise NotImplementedError(
            "Structured Streaming is not supported in single-node mode. "
            "Use session.read for batch reads."
        )

    @property
    def writeStream(self):
        """Structured Streaming is not supported in single-node mode."""
        raise NotImplementedError(
            "Structured Streaming is not supported in single-node mode. "
            "Use df.write for batch writes."
        )

    @property
    def streams(self) -> StreamingQueryManagerStub:
        """Returns a stub StreamingQueryManager."""
        return StreamingQueryManagerStub()

    # --- Internal methods ---

    def _resolve_table_name(self, name: str):
        """Resolve a (possibly 3-part) table name to (catalog, namespace, table).

        Supports the following forms:

        * ``catalog.namespace.table`` → specific catalog
        * ``namespace.table`` → current catalog
        * ``table`` → current catalog + current/default namespace

        Returns a ``(catalog_name, namespace, table_name, catalog_obj)`` tuple.
        Raises ``IcebergSparkError`` if the resolved catalog doesn't exist.
        """
        from iceberg_spark.catalog_ops import IcebergSparkError

        parts = name.split(".")
        if len(parts) >= 3:
            cat_name = parts[0]
            namespace = ".".join(parts[1:-1])
            table = parts[-1]
        elif len(parts) == 2:
            cat_name = self._current_catalog_name
            namespace = parts[0]
            table = parts[1]
        else:
            cat_name = self._current_catalog_name
            namespace = self._current_namespace or "default"
            table = parts[0]

        if cat_name not in self._catalogs:
            raise IcebergSparkError(
                f"Catalog '{cat_name}' not found. "
                f"Available: {list(self._catalogs.keys())}"
            )
        return (cat_name, namespace, table, self._catalogs[cat_name])

    def _ensure_tables_registered(self, sql: str) -> str:
        """Lazily register tables referenced in the SQL.

        Returns the SQL with fully-qualified table names replaced by short names
        so DataFusion can find the registered tables.  Uses word-boundary-aware
        replacement to avoid corrupting substrings (e.g., ``db.t`` must not
        match the ``t`` in ``count``).
        """
        import re

        pattern = r"(?:FROM|JOIN|INTO)\s+([a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)*)"
        matches = re.findall(pattern, sql, re.IGNORECASE)

        for name in matches:
            parts = name.split(".")
            table_name = parts[-1]
            self._ensure_table_registered(table_name, name)
            # Replace fully-qualified name with short name using word boundaries
            if len(parts) > 1 and table_name != name:
                sql = re.sub(
                    r"\b" + re.escape(name) + r"\b",
                    table_name,
                    sql,
                )
        return sql

    def _ensure_table_registered(self, short_name: str, full_name: str | None = None) -> None:
        """Register a table with DataFusion if not already registered."""
        # Skip internal/temp tables
        if short_name.startswith("_"):
            return

        full_name = full_name or short_name
        parts = full_name.split(".")

        # If already registered with the SAME full_name, skip
        if short_name in self._registered_tables:
            if self._registered_tables[short_name] == full_name:
                return
            # Different full_name for same short_name — deregister old one
            try:
                self._ctx.deregister_table(short_name)
            except Exception:
                pass
            del self._registered_tables[short_name]

        try:
            # Determine which catalog to use (supports 3-part names)
            if len(parts) >= 3 and parts[0] in self._catalogs:
                # catalog.namespace.table — use named catalog
                catalog = self._catalogs[parts[0]]
                table_ident = ".".join(parts[1:])  # namespace.table
            elif len(parts) >= 2:
                catalog = self._catalog
                table_ident = full_name
            else:
                catalog = self._catalog
                # Try to find the table in any namespace
                table_ident = None
                for ns in catalog.list_namespaces():
                    ns_name = ".".join(ns)
                    for tbl in catalog.list_tables(ns):
                        tbl_name = tbl.name if hasattr(tbl, "name") else tbl[-1]
                        if tbl_name == short_name:
                            table_ident = f"{ns_name}.{short_name}"
                            break
                    else:
                        continue
                    break
                if table_ident is None:
                    return  # Table not found in catalog

            table = catalog.load_table(table_ident)
            register_iceberg_table(self._ctx, table, short_name)
            self._registered_tables[short_name] = full_name
        except Exception as e:
            # Table might not exist in catalog or be a CTE/subquery alias
            logger.debug("Could not register table %s: %s", short_name, e)

    def _register_time_travel_table(
        self,
        table_name: str,
        snapshot_id: int | None = None,
        timestamp: str | None = None,
    ) -> None:
        """Register a time-travel version of a table."""
        parts = table_name.split(".")
        short_name = parts[-1]

        # Load the table
        try:
            if len(parts) >= 2:
                table = self._catalog.load_table(table_name)
            else:
                # Search for the table
                table = None
                for ns in self._catalog.list_namespaces():
                    ns_name = ".".join(ns)
                    try:
                        table = self._catalog.load_table(f"{ns_name}.{short_name}")
                        break
                    except Exception:
                        logger.debug("Table %s not in namespace %s", short_name, ns_name)
                        continue
                if table is None:
                    from iceberg_spark.catalog_ops import TableNotFoundError
                    raise TableNotFoundError(f"Time travel: table {table_name} not found")
        except Exception as e:
            from iceberg_spark.catalog_ops import TableNotFoundError
            if isinstance(e, TableNotFoundError):
                raise
            from iceberg_spark.catalog_ops import DMLError
            raise DMLError(f"Time travel: failed to load table {table_name}: {e}") from e

        if snapshot_id is not None:
            target_snapshot_id = snapshot_id
        elif timestamp is not None:
            # Find snapshot by timestamp
            from datetime import datetime

            ts = datetime.fromisoformat(timestamp)
            target_snapshot_id = None
            for snapshot in table.metadata.snapshots:
                snap_ts = datetime.fromtimestamp(snapshot.timestamp_ms / 1000)
                if snap_ts <= ts:
                    target_snapshot_id = snapshot.snapshot_id
            if target_snapshot_id is None:
                from iceberg_spark.catalog_ops import DMLError
                raise DMLError(
                    f"Time travel: no snapshot found at or before timestamp {timestamp}"
                )
        else:
            raise ValueError("Either snapshot_id or timestamp must be provided")

        # Register with a special name
        tt_name = f"{short_name}__time_travel"

        # Use PyIceberg scan with snapshot_id to get Arrow data at that point in time.
        # This is pure Python and works without Rust-side snapshot support.
        try:
            # Deregister any previously registered time-travel table with the same name
            try:
                self._ctx.deregister_table(tt_name)
            except Exception:
                logger.debug("Time-travel table %s not previously registered", tt_name)
            scan = table.scan(snapshot_id=target_snapshot_id)
            arrow_table = scan.to_arrow()
            self._ctx.register_record_batches(tt_name, [arrow_table.to_batches()])
        except Exception as e:
            from iceberg_spark.catalog_ops import DMLError
            raise DMLError(
                f"Time travel: failed to read snapshot {target_snapshot_id} for {table_name}: {e}"
            ) from e

    def _insert_into_table(
        self,
        table_name: str,
        arrow_table: pa.Table,
        overwrite: bool = False,
    ) -> None:
        """Insert (or overwrite) Arrow data into an Iceberg table via PyIceberg."""
        from iceberg_spark.catalog_ops import _resolve_catalog_for_table

        catalog = _resolve_catalog_for_table(self, table_name)
        parts = table_name.split(".")
        short_name = parts[-1]

        # Strip catalog prefix for the identifier passed to PyIceberg
        if len(parts) >= 3 and parts[0] in self._catalogs:
            table_ident = ".".join(parts[1:])
        elif len(parts) >= 2:
            table_ident = table_name
        else:
            table_ident = f"default.{short_name}"

        try:
            table = catalog.load_table(table_ident)
            if overwrite:
                from iceberg_spark.catalog_ops import _build_partition_overwrite_filter
                overwrite_filter = _build_partition_overwrite_filter(table, arrow_table)
                table.overwrite(arrow_table, overwrite_filter=overwrite_filter)
            else:
                table.append(arrow_table)

            # Deregister so next query picks up the updated snapshot
            from iceberg_spark.catalog_ops import _invalidate_table_cache
            _invalidate_table_cache(self, short_name)
        except Exception as e:
            from iceberg_spark.catalog_ops import DMLError
            raise DMLError(f"INSERT INTO {table_name}: {e}") from e


class IcebergSessionBuilder:
    """Builder for IcebergSession, mimicking SparkSession.builder.

    Supports multiple catalogs::

        session = (
            IcebergSession.builder()
            .catalog("rest", name="prod", uri="http://rest:8181")
            .catalog("sql", name="local", uri="sqlite:///catalog.db", warehouse="/data")
            .defaultCatalog("prod")
            .build()
        )
    """

    def __init__(self):
        self._catalog_configs: dict[str, dict[str, Any]] = {}
        self._default_catalog_name: str | None = None
        self._config: dict[str, Any] = {}
        self._app_name: str = "iceberg_spark"

    def appName(self, name: str) -> IcebergSessionBuilder:
        """Sets the application name for this session."""
        self._app_name = name
        return self

    def catalog(
        self,
        catalog_type: str,
        name: str = "default",
        **properties: Any,
    ) -> IcebergSessionBuilder:
        """Configure an Iceberg catalog.

        Can be called multiple times to register multiple catalogs.
        The first catalog registered becomes the default unless
        :meth:`defaultCatalog` is called explicitly.

        Args:
            catalog_type: Catalog type ('rest', 'sql', 'hive', 'glue', 'dynamodb').
            name: Catalog name for registration.
            **properties: Catalog properties (uri, warehouse, etc.).
        """
        self._catalog_configs[name] = {"type": catalog_type, **properties}
        if self._default_catalog_name is None:
            self._default_catalog_name = name
        return self

    def defaultCatalog(self, name: str) -> IcebergSessionBuilder:
        """Set the default (active) catalog name for the session."""
        self._default_catalog_name = name
        return self

    def config(self, key: str, value: Any) -> IcebergSessionBuilder:
        """Set a configuration option."""
        self._config[key] = value
        return self

    def master(self, master: str) -> IcebergSessionBuilder:
        """No-op for compatibility. DataFusion is always local."""
        return self

    def enableHiveSupport(self) -> IcebergSessionBuilder:
        """No-op for compatibility."""
        return self

    def getOrCreate(self) -> IcebergSession:
        """Alias for build()."""
        return self.build()

    def build(self) -> IcebergSession:
        """Build and return an IcebergSession."""
        # If no catalogs configured, create a default in-memory SQL catalog
        if not self._catalog_configs:
            self._catalog_configs["default"] = {"type": "sql"}
            self._default_catalog_name = "default"

        # Create all configured catalogs
        catalogs: dict = {}
        for cat_name, cat_cfg in self._catalog_configs.items():
            cat_type = cat_cfg.pop("type")
            catalogs[cat_name] = create_catalog(cat_name, cat_type, cat_cfg)

        default_name = self._default_catalog_name or next(iter(catalogs))

        # Create the DataFusion session context
        ctx = SessionContext()

        session = IcebergSession(
            ctx=ctx,
            catalog_name=default_name,
            config=self._config,
            catalogs=catalogs,
        )
        IcebergSession._active_session = session
        return session


class RuntimeConfig:
    """Runtime configuration for an IcebergSession (dict-backed).

    Mirrors SparkSession.conf — supports get/set/unset/getAll.
    """

    def __init__(self, config: dict[str, Any]):
        self._config = config

    def get(self, key: str, default: str | None = None) -> str | None:
        """Returns the value for key, or default if not set."""
        return self._config.get(key, default)

    def set(self, key: str, value: str) -> None:
        """Sets a configuration key-value pair."""
        self._config[key] = value

    def unset(self, key: str) -> None:
        """Removes a configuration key (no-op if missing)."""
        self._config.pop(key, None)

    def getAll(self) -> dict[str, str]:
        """Returns a copy of all configuration key-value pairs."""
        return dict(self._config)

    def isModifiable(self, key: str) -> bool:
        """Returns True — all keys are modifiable in single-node mode."""
        return True


class SparkContextStub:
    """Stub SparkContext for PySpark compatibility.

    Provides basic properties that PySpark scripts commonly access.
    """

    def __init__(self, session: IcebergSession):
        self._session = session

    @property
    def appName(self) -> str:
        """Returns the application name."""
        return self._session._config.get("spark.app.name", "iceberg_spark")

    @property
    def version(self) -> str:
        """Returns the version string."""
        return self._session.version

    @property
    def master(self) -> str:
        """Returns the master URL (always 'local' in single-node mode)."""
        return "local"

    def getConf(self) -> RuntimeConfig:
        """Returns the runtime configuration."""
        return self._session.conf

    def setLogLevel(self, logLevel: str) -> None:
        """No-op — DataFusion uses Rust logging."""
        pass

    def addPyFile(self, path: str) -> None:
        """No-op — single-node mode."""
        pass

    def setCheckpointDir(self, dirName: str) -> None:
        """No-op — single-node mode."""
        pass


class _UserDefinedFunction:
    """Wrapper returned by udf.register() -- callable on Columns."""

    def __init__(self, name: str, scalar_udf):
        self._name = name
        self._scalar_udf = scalar_udf

    def __call__(self, *cols):
        from iceberg_spark.column import Column, _unwrap

        args = [_unwrap(c if isinstance(c, Column) else Column(c)) for c in cols]
        return Column(self._scalar_udf(*args))

    def __repr__(self):
        return f"<function {self._name}>"


class _UserDefinedAggregateFunction:
    """Wrapper returned by udf.register_udaf() -- callable on Columns."""

    def __init__(self, name: str, aggregate_udf):
        self._name = name
        self._aggregate_udf = aggregate_udf

    def __call__(self, *cols):
        from iceberg_spark.column import Column, _unwrap

        args = [_unwrap(c if isinstance(c, Column) else Column(c)) for c in cols]
        return Column(self._aggregate_udf(*args))

    def __repr__(self):
        return f"<function {self._name}>"


class _SimpleAccumulator:
    """Wraps a plain Python function ``(list -> scalar)`` into the Accumulator interface.

    This allows users to register a simple callable as a UDAF without implementing
    the full Accumulator abstract class.  The state is serialized as a list scalar
    containing the accumulated buffer values.
    """

    def __init__(self, func, arrow_return_type, arrow_input_type):
        self._func = func
        self._buffer = []
        self._arrow_return_type = arrow_return_type
        self._arrow_input_type = arrow_input_type

    def state(self):
        import pyarrow as pa

        return [pa.scalar(self._buffer, type=pa.list_(self._arrow_input_type))]

    def update(self, *arrays):
        for arr in arrays:
            for val in arr:
                py_val = val.as_py()
                if py_val is not None:
                    self._buffer.append(py_val)

    def merge(self, states):
        # states is a list of pa.Array — one per state field.
        # states[0] is a ListArray where each element is a list.
        for list_scalar in states[0]:
            py_list = list_scalar.as_py()
            if py_list is not None:
                self._buffer.extend(py_list)

    def evaluate(self):
        import pyarrow as pa

        try:
            result = self._func(self._buffer)
        except Exception as e:
            import warnings

            warnings.warn(
                f"UDAF evaluate() failed: {e}",
                UserWarning,
                stacklevel=2,
            )
            result = None
        return pa.scalar(result, type=self._arrow_return_type)


def _resolve_return_type(returnType) -> "pa.DataType":
    """Resolve a PySpark DataType or type-name string to a PyArrow DataType."""
    import pyarrow as pa

    from iceberg_spark.types import DataType as SparkDataType

    if returnType is None:
        # Default to string, matching PySpark behaviour
        return pa.utf8()
    if isinstance(returnType, SparkDataType):
        return returnType.to_arrow()
    if isinstance(returnType, str):
        from iceberg_spark._internal.type_mapping import spark_type_from_name

        return spark_type_from_name(returnType).to_arrow()
    if isinstance(returnType, pa.DataType):
        return returnType
    raise TypeError(f"Unsupported returnType: {type(returnType)}")


def _make_vectorized_wrapper(f, n_args, arrow_return_type, name="udf"):
    """Create a vectorized wrapper that receives PyArrow arrays and calls *f* per row."""
    import warnings

    import pyarrow as pa

    def _wrapper(*arrays):
        if n_args == 0:
            # Zero-arg UDF (rare but valid) -- produce one value per row in first array
            length = len(arrays[0]) if arrays else 1
            results = [f() for _ in range(length)]
        elif n_args == 1:
            results = []
            for v in arrays[0]:
                py_val = v.as_py()
                try:
                    results.append(f(py_val))
                except Exception as e:
                    warnings.warn(
                        f"UDF '{name}' failed on input {py_val!r}: {e}",
                        UserWarning,
                        stacklevel=2,
                    )
                    results.append(None)
        else:
            # Multi-arg: zip across all arrays
            results = []
            for vals in zip(*(arr for arr in arrays)):
                py_vals = [v.as_py() for v in vals]
                try:
                    results.append(f(*py_vals))
                except Exception as e:
                    warnings.warn(
                        f"UDF '{name}' failed on input {py_vals!r}: {e}",
                        UserWarning,
                        stacklevel=2,
                    )
                    results.append(None)
        return pa.array(results, type=arrow_return_type)

    return _wrapper


class UDFRegistration:
    """UDF registration for PySpark compatibility, backed by DataFusion ScalarUDF."""

    def __init__(self, session=None):
        self._session = session

    def register(self, name: str, f=None, returnType=None, inputTypes=None, batch_mode=False):
        """Register a Python UDF for use in SQL and DataFrame expressions.

        Args:
            name: Name to register the UDF under.
            f: A Python callable that operates on individual Python values
                (default) or on ``pyarrow.Array`` objects when *batch_mode*
                is ``True``.
            returnType: PySpark DataType instance or type name string.
            inputTypes: Optional list of PySpark DataType / type-name / pa.DataType
                for the UDF input columns.  When provided, DataFusion will
                expect columns of those exact types.  When *None* (the
                default), all inputs are declared as ``pa.utf8()`` and
                DataFusion coerces automatically; user functions receive
                Python objects via ``.as_py()``.
                **Required** when *batch_mode* is ``True``.
            batch_mode: When ``True``, the user function receives
                ``pyarrow.Array`` arguments directly and must return a
                ``pyarrow.Array``.  The per-row wrapper is skipped,
                yielding much better performance for vectorized operations.
                Defaults to ``False``.

        Returns:
            A callable (_UserDefinedFunction) that accepts Column args and
            returns a Column.
        """
        import inspect

        import pyarrow as pa
        from datafusion.user_defined import ScalarUDF

        if f is None:
            raise ValueError("A callable must be provided to register().")
        if self._session is None:
            raise RuntimeError(
                "UDFRegistration requires a live session. "
                "Use session.udf.register() instead."
            )

        if batch_mode and inputTypes is None:
            raise ValueError(
                "inputTypes is required when batch_mode=True. "
                "Provide a list of DataType / type-name / pa.DataType for each input column."
            )

        arrow_return_type = _resolve_return_type(returnType)

        # Determine the number of arguments the user function expects
        try:
            sig = inspect.signature(f)
            n_args = len(sig.parameters)
        except (ValueError, TypeError):
            n_args = 1  # fallback

        # Build input types
        if inputTypes is not None:
            input_types = [_resolve_return_type(t) for t in inputTypes]
        else:
            # Default: accept string inputs; DataFusion coerces automatically.
            # User functions receive Python objects via .as_py().
            input_types = [pa.utf8()] * n_args

        if batch_mode:
            # In batch mode the user function already operates on pa.Array
            # objects and returns a pa.Array — pass it directly.
            func = f
        else:
            func = _make_vectorized_wrapper(f, n_args, arrow_return_type, name=name)

        scalar_udf = ScalarUDF.udf(
            func,
            input_types,
            arrow_return_type,
            "volatile",
            name,
        )

        self._session._ctx.register_udf(scalar_udf)
        self._session._registered_udfs[name] = str(arrow_return_type)

        return _UserDefinedFunction(name, scalar_udf)

    def register_udaf(self, name: str, f=None, returnType=None, inputTypes=None):
        """Register a Python aggregate UDF (UDAF) for use in SQL and DataFrame expressions.

        Args:
            name: Name to register the UDAF under.
            f: Either a plain Python callable ``(list -> scalar)`` that receives
               a list of values and returns a single aggregate result, or a class
               that implements the Accumulator interface (with ``update``,
               ``merge``, ``state``, ``evaluate`` methods).
            returnType: PySpark DataType instance, type-name string, or
                ``pyarrow.DataType`` for the aggregate result.
            inputTypes: List of PySpark DataType / type-name / pa.DataType for
                the input columns.  Defaults to ``[LongType()]`` if not provided.

        Returns:
            A callable (_UserDefinedAggregateFunction) that accepts Column args
            and returns a Column.
        """
        import pyarrow as pa
        from datafusion.user_defined import Accumulator, AggregateUDF

        if f is None:
            raise ValueError("A callable or Accumulator class must be provided to register_udaf().")
        if self._session is None:
            raise RuntimeError(
                "UDFRegistration requires a live session. "
                "Use session.udf.register_udaf() instead."
            )

        arrow_return_type = _resolve_return_type(returnType)

        # Build input types
        if inputTypes is not None:
            input_types = [_resolve_return_type(t) for t in inputTypes]
        else:
            input_types = [pa.int64()]

        # Determine if f is a class/factory that produces Accumulator instances,
        # or a plain callable to wrap in _SimpleAccumulator.
        is_accumulator_class = False
        try:
            if isinstance(f, type) and issubclass(f, Accumulator):
                is_accumulator_class = True
            elif callable(f):
                # Check if calling it produces an Accumulator instance
                test_instance = f() if isinstance(f, type) else None
                if test_instance is not None and isinstance(test_instance, Accumulator):
                    is_accumulator_class = True
                elif hasattr(f, "update") and hasattr(f, "merge") and hasattr(f, "state") and hasattr(f, "evaluate"):
                    # f itself is an Accumulator-like class (without inheriting)
                    is_accumulator_class = True
        except (TypeError, Exception):
            pass

        if is_accumulator_class:
            # f is already an Accumulator factory/class
            accumulator_factory = f
            # State types: use the return type as state type
            state_types = [arrow_return_type]
        else:
            # Wrap a plain callable in _SimpleAccumulator
            captured_func = f
            captured_return_type = arrow_return_type
            captured_input_type = input_types[0]
            # State is a list of accumulated input values
            state_types = [pa.list_(captured_input_type)]

            def accumulator_factory():
                return _SimpleAccumulator(captured_func, captured_return_type, captured_input_type)

        aggregate_udf = AggregateUDF(
            name=name,
            accumulator=accumulator_factory,
            input_types=input_types,
            return_type=arrow_return_type,
            state_type=state_types,
            volatility="volatile",
        )

        self._session._ctx.register_udaf(aggregate_udf)
        self._session._registered_udfs[name] = str(arrow_return_type)

        return _UserDefinedAggregateFunction(name, aggregate_udf)

    def registerJavaFunction(self, name: str, javaClassName: str, returnType=None):
        """Not supported — no JVM."""
        raise NotImplementedError("Java UDFs are not supported (no JVM).")

    def registerJavaUDAF(self, name: str, javaClassName: str):
        """Not supported — no JVM."""
        raise NotImplementedError("Java UDAFs are not supported (no JVM).")


class StreamingQueryManagerStub:
    """Stub StreamingQueryManager for PySpark compatibility."""

    @property
    def active(self) -> list:
        """Returns empty list — no streaming queries in single-node mode."""
        return []

    def get(self, id: str):
        """Not supported."""
        raise NotImplementedError("Streaming is not supported in single-node mode.")

    def awaitAnyTermination(self, timeout: float | None = None) -> bool:
        """No-op — no streaming queries."""
        return True

    def resetTerminated(self) -> None:
        """No-op."""
        pass
