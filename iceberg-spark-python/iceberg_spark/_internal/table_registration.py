"""Table registration with DataFusion via Iceberg Rust FFI or Arrow fallback."""

from __future__ import annotations

import logging
from types import MethodType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datafusion import SessionContext
    from pyiceberg.catalog import Catalog
    from pyiceberg.table import Table as PyIcebergTable

logger = logging.getLogger(__name__)


def register_iceberg_table(
    ctx: SessionContext,
    table: PyIcebergTable,
    table_name: str,
) -> None:
    """Register a PyIceberg table with a DataFusion SessionContext.

    Tries the pyiceberg-core Rust FFI bridge first for zero-copy performance.
    Falls back to Arrow materialization via PyIceberg scan if FFI is unavailable
    or incompatible (e.g., DataFusion v52 changed the FFI signature from v50).
    """
    if _try_ffi_registration(ctx, table, table_name):
        return
    _arrow_fallback_registration(ctx, table, table_name)


def _try_ffi_registration(
    ctx: SessionContext,
    table: PyIcebergTable,
    table_name: str,
) -> bool:
    """Attempt FFI-based registration. Returns True on success."""
    try:
        from pyiceberg_core.datafusion import IcebergDataFusionTable
    except ImportError:
        logger.debug("pyiceberg-core not available, skipping FFI registration")
        return False

    try:
        def __datafusion_table_provider__(self):
            return IcebergDataFusionTable(
                identifier=self.name(),
                metadata_location=self.metadata_location,
                file_io_properties=self.io.properties,
            ).__datafusion_table_provider__()

        table.__datafusion_table_provider__ = MethodType(
            __datafusion_table_provider__, table
        )
        ctx.register_table(table_name, table)
        return True
    except Exception as e:
        # Clean up the monkey-patch so the table object is not polluted
        try:
            delattr(table, "__datafusion_table_provider__")
        except AttributeError:
            pass
        logger.debug("FFI registration failed for %s: %s", table_name, e)
        return False


def _arrow_fallback_registration(
    ctx: SessionContext,
    table: PyIcebergTable,
    table_name: str,
) -> None:
    """Register table by materializing it as Arrow record batches.

    This is the same approach used by time-travel queries and DML handlers.
    Slightly slower than FFI (full table scan) but works regardless of
    DataFusion/pyiceberg-core version compatibility.
    """
    import pyarrow as pa

    arrow_table = table.scan().to_arrow()
    batches = arrow_table.to_batches()
    if not batches:
        # Empty table — create a single empty batch with proper empty arrays
        schema = arrow_table.schema
        empty_arrays = [pa.array([], type=f.type) for f in schema]
        batches = [pa.record_batch(empty_arrays, schema=schema)]
    ctx.register_record_batches(table_name, [batches])


def register_catalog_tables(
    ctx: SessionContext,
    catalog: Catalog,
    namespace: str | None = None,
) -> list[str]:
    """Register all tables from a catalog (or specific namespace) with DataFusion.

    Returns the list of registered table names.
    """
    registered = []
    namespaces = (
        [tuple(namespace.split("."))]
        if namespace
        else catalog.list_namespaces()
    )

    for ns in namespaces:
        ns_name = ".".join(ns)
        try:
            tables = catalog.list_tables(ns)
        except Exception:
            continue

        for table_ident in tables:
            try:
                table = catalog.load_table(table_ident)
                full_name = f"{ns_name}.{table_ident.name if hasattr(table_ident, 'name') else table_ident[-1]}"
                short_name = table_ident.name if hasattr(table_ident, "name") else table_ident[-1]
                register_iceberg_table(ctx, table, short_name)
                registered.append(full_name)
            except Exception as e:
                logger.debug("Failed to register table %s: %s", table_ident, e)
                continue

    return registered
