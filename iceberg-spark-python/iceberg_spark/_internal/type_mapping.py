"""Type mappings between Spark, Arrow, and Iceberg type systems."""

from __future__ import annotations

import pyarrow as pa

from iceberg_spark.types import (
    ArrayType,
    BinaryType,
    BooleanType,
    ByteType,
    DataType,
    DateType,
    DecimalType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    MapType,
    NullType,
    ShortType,
    StringType,
    StructField,
    StructType,
    TimestampNTZType,
    TimestampType,
)

# Spark SQL type name -> DataType constructor
_SPARK_TYPE_NAMES: dict[str, type[DataType] | DataType] = {
    "null": NullType,
    "boolean": BooleanType,
    "tinyint": ByteType,
    "byte": ByteType,
    "smallint": ShortType,
    "short": ShortType,
    "int": IntegerType,
    "integer": IntegerType,
    "bigint": LongType,
    "long": LongType,
    "float": FloatType,
    "real": FloatType,
    "double": DoubleType,
    "string": StringType,
    "varchar": StringType,
    "char": StringType,
    "binary": BinaryType,
    "date": DateType,
    "timestamp": TimestampType,
    "timestamp_ntz": TimestampNTZType,
}


def _split_top_level(s: str, delimiter: str = ",") -> list[str]:
    """Split string on *delimiter*, respecting ``<>`` and ``()`` nesting."""
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in s:
        if ch in "<(":
            depth += 1
            current.append(ch)
        elif ch in ">)":
            depth -= 1
            current.append(ch)
        elif ch == delimiter and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current))
    if depth != 0:
        raise ValueError(f"Unbalanced brackets in type expression: {s!r}")
    return parts


def spark_type_from_name(name: str) -> DataType:
    """Parse a Spark SQL type name into a DataType.

    Handles simple types (int, string, etc.), decimal(p,s),
    and complex types: array<T>, map<K,V>, struct<field:type,...>.
    """
    name = name.strip()
    lower = name.lower()

    # Handle decimal(p, s)
    if lower.startswith("decimal"):
        if "(" in name:
            params = name[name.index("(") + 1 : name.index(")")].split(",")
            precision = int(params[0].strip())
            scale = int(params[1].strip()) if len(params) > 1 else 0
            return DecimalType(precision, scale)
        return DecimalType()

    # Handle varchar(n) and char(n) — map to StringType (length is informational)
    if lower.startswith("varchar") or lower.startswith("char"):
        return StringType()

    # Handle array<elementType>
    if lower.startswith("array<") and name.endswith(">"):
        inner = name[6:-1]  # strip "array<" and ">"
        element_type = spark_type_from_name(inner)
        return ArrayType(element_type)

    # Handle map<keyType, valueType>
    if lower.startswith("map<") and name.endswith(">"):
        inner = name[4:-1]  # strip "map<" and ">"
        parts = _split_top_level(inner, ",")
        if len(parts) != 2:
            raise ValueError(
                f"MAP type requires exactly 2 type arguments, got {len(parts)}: {name}"
            )
        key_type = spark_type_from_name(parts[0])
        value_type = spark_type_from_name(parts[1])
        return MapType(key_type, value_type)

    # Handle struct<field1:type1, field2:type2, ...>
    if lower.startswith("struct<") and name.endswith(">"):
        inner = name[7:-1]  # strip "struct<" and ">"
        fields = []
        for part in _split_top_level(inner, ","):
            part = part.strip()
            if not part:
                continue
            if ":" not in part:
                raise ValueError(f"Struct field must be 'name:type', got: {part!r}")
            field_name, field_type_str = part.split(":", 1)
            field_name = field_name.strip()
            field_type = spark_type_from_name(field_type_str.strip())
            fields.append(StructField(field_name, field_type))
        return StructType(fields)

    # Simple type lookup
    cls = _SPARK_TYPE_NAMES.get(lower)
    if cls is None:
        raise ValueError(f"Unknown Spark SQL type: {name}")
    if isinstance(cls, type):
        return cls()
    return cls


def arrow_type_to_spark_string(arrow_type: pa.DataType) -> str:
    """Convert an Arrow data type to its Spark SQL string name."""
    if pa.types.is_null(arrow_type):
        return "void"
    if pa.types.is_boolean(arrow_type):
        return "boolean"
    if pa.types.is_int8(arrow_type):
        return "tinyint"
    if pa.types.is_int16(arrow_type):
        return "smallint"
    if pa.types.is_int32(arrow_type):
        return "int"
    if pa.types.is_int64(arrow_type):
        return "bigint"
    if pa.types.is_float32(arrow_type):
        return "float"
    if pa.types.is_float64(arrow_type):
        return "double"
    if pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
        return "string"
    if pa.types.is_binary(arrow_type) or pa.types.is_large_binary(arrow_type) or pa.types.is_fixed_size_binary(arrow_type):
        return "binary"
    if pa.types.is_date(arrow_type):
        return "date"
    if pa.types.is_timestamp(arrow_type):
        return "timestamp" if arrow_type.tz else "timestamp_ntz"
    if pa.types.is_decimal(arrow_type):
        return f"decimal({arrow_type.precision},{arrow_type.scale})"
    if pa.types.is_list(arrow_type) or pa.types.is_large_list(arrow_type):
        return f"array<{arrow_type_to_spark_string(arrow_type.value_type)}>"
    if pa.types.is_map(arrow_type):
        return f"map<{arrow_type_to_spark_string(arrow_type.key_type)},{arrow_type_to_spark_string(arrow_type.item_type)}>"
    if pa.types.is_struct(arrow_type):
        fields = ",".join(
            f"{f.name}:{arrow_type_to_spark_string(f.type)}"
            for f in arrow_type
        )
        return f"struct<{fields}>"
    return "string"
