"""Catalog creation via PyIceberg."""

from __future__ import annotations

from typing import Any


def create_catalog(name: str, catalog_type: str, properties: dict[str, Any]):
    """Create a PyIceberg catalog.

    Args:
        name: Catalog name for registration.
        catalog_type: Catalog type (rest, sql, hive, glue, dynamodb).
        properties: Catalog configuration properties (uri, warehouse, etc.).
    """
    from pyiceberg.catalog import load_catalog

    config = dict(properties)
    config["type"] = catalog_type

    return load_catalog(name, **config)
