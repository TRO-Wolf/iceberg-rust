"""PySpark-compatible Catalog API — session.catalog.listTables() etc."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from iceberg_spark.session import IcebergSession

logger = logging.getLogger(__name__)


class IcebergCatalogAPI:
    """Spark-compatible catalog API exposed as session.catalog.

    Wraps the underlying PyIceberg catalog to provide the Spark Catalog API
    surface: listTables, listDatabases, tableExists, etc.
    """

    def __init__(self, session: IcebergSession):
        self._session = session
        self._current_database: str = "default"

    @property
    def _catalog(self):
        """Returns the currently active PyIceberg Catalog (follows session)."""
        return self._session._catalog

    # ------------------------------------------------------------------
    # Catalog operations (multi-catalog support)
    # ------------------------------------------------------------------

    def currentCatalog(self) -> str:
        """Returns the name of the current catalog."""
        return self._session._current_catalog_name

    def setCurrentCatalog(self, catalogName: str) -> None:
        """Sets the current catalog by name.

        Resets the current database to ``"default"`` on switch.
        Raises RuntimeError if the catalog was not configured.
        """
        if catalogName not in self._session._catalogs:
            raise RuntimeError(
                f"Catalog '{catalogName}' not found. "
                f"Available: {list(self._session._catalogs.keys())}"
            )
        self._session._current_catalog_name = catalogName
        self._current_database = "default"

    def listCatalogs(self, pattern: str | None = None) -> list[str]:
        """Returns a list of configured catalog names, optionally filtered."""
        result = list(self._session._catalogs.keys())
        if pattern:
            import fnmatch
            result = [r for r in result if fnmatch.fnmatch(r, pattern)]
        return result

    # ------------------------------------------------------------------
    # Database / namespace operations
    # ------------------------------------------------------------------

    def currentDatabase(self) -> str:
        """Returns the current default database."""
        return self._current_database

    def setCurrentDatabase(self, dbName: str) -> None:
        """Sets the current default database."""
        self._current_database = dbName

    def listDatabases(self, pattern: str | None = None) -> list[str]:
        """Returns a list of database names, optionally filtered by a glob pattern."""
        try:
            namespaces = self._catalog.list_namespaces()
            result = [".".join(ns) for ns in namespaces]
        except Exception:
            logger.debug("Failed to list namespaces from catalog")
            result = []
        if pattern:
            import fnmatch
            result = [r for r in result if fnmatch.fnmatch(r, pattern)]
        return result

    def getDatabase(self, dbName: str) -> dict:
        """Returns database metadata as a dict with name, description, locationUri."""
        ns_tuple = tuple(dbName.split("."))
        try:
            props = self._catalog.load_namespace_properties(ns_tuple)
        except Exception:
            logger.debug("Could not load namespace properties for %s", dbName)
            props = {}
        return {
            "name": dbName,
            "description": props.get("comment", ""),
            "locationUri": props.get("location", ""),
        }

    def databaseExists(self, dbName: str) -> bool:
        """Returns True if the database exists."""
        try:
            ns_tuple = tuple(dbName.split("."))
            namespaces = self._catalog.list_namespaces()
            return any(tuple(ns) == ns_tuple for ns in namespaces)
        except Exception:
            logger.debug("Error checking database existence for %s", dbName)
            return False

    # ------------------------------------------------------------------
    # Table operations
    # ------------------------------------------------------------------

    def listTables(self, dbName: str | None = None, pattern: str | None = None) -> list[str]:
        """Returns a list of table names in the given database.

        Args:
            dbName: Database name. Uses currentDatabase() if not specified.
            pattern: Optional glob-style filter applied to table names.
        """
        db = dbName or self._current_database
        try:
            ns_tuple = tuple(db.split("."))
            tables = self._catalog.list_tables(ns_tuple)
            result = []
            for tbl in tables:
                name = tbl.name if hasattr(tbl, "name") else tbl[-1]
                result.append(name)
        except Exception:
            logger.debug("Failed to list tables in database %s", db)
            result = []
        if pattern:
            import fnmatch
            result = [r for r in result if fnmatch.fnmatch(r, pattern)]
        return result

    def tableExists(self, tableName: str, dbName: str | None = None) -> bool:
        """Returns True if the table exists.

        Args:
            tableName: Table name (may be fully qualified as db.table).
            dbName: Optional database name to qualify the table.
        """
        if "." in tableName:
            full_name = tableName
        else:
            db = dbName or self._current_database
            full_name = f"{db}.{tableName}"
        try:
            self._catalog.load_table(full_name)
            return True
        except Exception:
            logger.debug("Table %s not found in catalog", full_name)
            return False

    def getTable(self, tableName: str):
        """Returns a TableInfo-like dict for the named table."""
        if "." not in tableName:
            tableName = f"{self._current_database}.{tableName}"
        try:
            table = self._catalog.load_table(tableName)
            parts = tableName.split(".")
            return {
                "name": parts[-1],
                "database": ".".join(parts[:-1]),
                "description": "",
                "tableType": "MANAGED",
                "isTemporary": False,
            }
        except Exception as e:
            from iceberg_spark.catalog_ops import TableNotFoundError
            raise TableNotFoundError(f"getTable: {tableName}") from e

    # ------------------------------------------------------------------
    # Column / function operations
    # ------------------------------------------------------------------

    def listColumns(self, tableName: str, dbName: str | None = None) -> list[dict]:
        """Returns a list of column info dicts for the given table.

        Each dict has keys: name, description, dataType, nullable, isPartition, isBucket.
        """
        if "." in tableName:
            full_name = tableName
        else:
            db = dbName or self._current_database
            full_name = f"{db}.{tableName}"
        try:
            table = self._catalog.load_table(full_name)
            schema = table.schema()
            return [
                {
                    "name": field.name,
                    "description": "",
                    "dataType": str(field.field_type),
                    "nullable": field.optional,
                    "isPartition": False,
                    "isBucket": False,
                }
                for field in schema.fields
            ]
        except Exception as e:
            from iceberg_spark.catalog_ops import TableNotFoundError
            raise TableNotFoundError(f"listColumns: {tableName}: {e}") from e

    def listFunctions(self, dbName: str | None = None) -> list:
        """Returns a list of registered UDF descriptions.

        Each entry is a dict with ``name``, ``className``, ``isTemporary``
        keys, approximating PySpark's ``Function`` Row type.
        """
        from iceberg_spark.row import Row

        udfs = getattr(self._session, "_registered_udfs", {})
        return [
            Row(name=name, className="python_udf", isTemporary=True)
            for name in udfs
        ]

    def functionExists(self, functionName: str, dbName: str | None = None) -> bool:
        """Returns True if a UDF with the given name has been registered."""
        udfs = getattr(self._session, "_registered_udfs", {})
        return functionName in udfs

    # ------------------------------------------------------------------
    # Cache operations (no-ops — DataFusion manages its own caching)
    # ------------------------------------------------------------------

    def cacheTable(self, tableName: str) -> None:
        """No-op: DataFusion does not expose a caching API."""
        pass

    def uncacheTable(self, tableName: str) -> None:
        """No-op."""
        pass

    def isCached(self, tableName: str) -> bool:
        """Always returns False (no explicit caching layer)."""
        return False

    def clearCache(self) -> None:
        """No-op."""
        pass

    def refreshTable(self, tableName: str) -> None:
        """Deregisters the table so it is re-loaded from the catalog on next access."""
        parts = tableName.split(".")
        short_name = parts[-1]
        self._session._registered_tables.pop(short_name, None)

    def refreshByPath(self, path: str) -> None:
        """No-op."""
        pass

    # ------------------------------------------------------------------
    # Temp view operations
    # ------------------------------------------------------------------

    def dropTempView(self, viewName: str) -> bool:
        """Drops a temporary view. Returns True if it existed, False otherwise."""
        try:
            # ctx.table() raises KeyError when the table doesn't exist;
            # ctx.deregister_table() silently returns None in both cases, so
            # we must check existence first.
            self._session._ctx.table(viewName)
        except Exception:
            logger.debug("Temp view %s does not exist, nothing to drop", viewName)
            return False
        self._session._ctx.deregister_table(viewName)
        return True

    def dropGlobalTempView(self, viewName: str) -> bool:
        """Drops a global temporary view (same as dropTempView for single-node)."""
        return self.dropTempView(viewName)

    # ------------------------------------------------------------------
    # Programmatic table creation
    # ------------------------------------------------------------------

    def createTable(self, tableName: str, schema=None, **kwargs):
        """Creates a table programmatically from a schema.

        Args:
            tableName: Fully qualified (db.table) or unqualified table name.
            schema: A StructType or PyArrow schema. Required.
        """
        if schema is None:
            raise ValueError("schema is required for createTable")
        import pyarrow as pa
        from iceberg_spark.types import StructType
        if isinstance(schema, StructType):
            arrow_schema = pa.schema([f.to_arrow() for f in schema.fields])
        else:
            arrow_schema = schema
        parts = tableName.split(".")
        ns = parts[0] if len(parts) > 1 else "default"
        tbl = parts[-1]
        self._catalog.create_table(f"{ns}.{tbl}", schema=arrow_schema)
        return self._session.table(tableName)
