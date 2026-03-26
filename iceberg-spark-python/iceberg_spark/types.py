"""PySpark-compatible type system mapped to Arrow types."""

from __future__ import annotations

import pyarrow as pa


class DataType:
    """Base class for all data types."""

    def simpleString(self) -> str:
        """Returns a readable string representation of the type."""
        return str(self)

    def jsonValue(self) -> str:
        """Returns the JSON-compatible string representation of the type."""
        return str(self)


class NullType(DataType):
    """Null data type."""

    def __repr__(self):
        return "NullType()"

    def __str__(self):
        return "null"

    def to_arrow(self) -> pa.DataType:
        """Converts to the corresponding Arrow data type."""
        return pa.null()


class BooleanType(DataType):
    def __repr__(self):
        return "BooleanType()"

    def __str__(self):
        return "boolean"

    def to_arrow(self) -> pa.DataType:
        return pa.bool_()


class ByteType(DataType):
    def __repr__(self):
        return "ByteType()"

    def __str__(self):
        return "tinyint"

    def to_arrow(self) -> pa.DataType:
        return pa.int8()


class ShortType(DataType):
    def __repr__(self):
        return "ShortType()"

    def __str__(self):
        return "smallint"

    def to_arrow(self) -> pa.DataType:
        return pa.int16()


class IntegerType(DataType):
    def __repr__(self):
        return "IntegerType()"

    def __str__(self):
        return "int"

    def to_arrow(self) -> pa.DataType:
        return pa.int32()


class LongType(DataType):
    def __repr__(self):
        return "LongType()"

    def __str__(self):
        return "bigint"

    def to_arrow(self) -> pa.DataType:
        return pa.int64()


class FloatType(DataType):
    def __repr__(self):
        return "FloatType()"

    def __str__(self):
        return "float"

    def to_arrow(self) -> pa.DataType:
        return pa.float32()


class DoubleType(DataType):
    def __repr__(self):
        return "DoubleType()"

    def __str__(self):
        return "double"

    def to_arrow(self) -> pa.DataType:
        return pa.float64()


class StringType(DataType):
    def __repr__(self):
        return "StringType()"

    def __str__(self):
        return "string"

    def to_arrow(self) -> pa.DataType:
        return pa.string()


class BinaryType(DataType):
    def __repr__(self):
        return "BinaryType()"

    def __str__(self):
        return "binary"

    def to_arrow(self) -> pa.DataType:
        return pa.binary()


class DateType(DataType):
    def __repr__(self):
        return "DateType()"

    def __str__(self):
        return "date"

    def to_arrow(self) -> pa.DataType:
        return pa.date32()


class TimestampType(DataType):
    def __repr__(self):
        return "TimestampType()"

    def __str__(self):
        return "timestamp"

    def to_arrow(self) -> pa.DataType:
        return pa.timestamp("us")


class TimestampNTZType(DataType):
    def __repr__(self):
        return "TimestampNTZType()"

    def __str__(self):
        return "timestamp_ntz"

    def to_arrow(self) -> pa.DataType:
        return pa.timestamp("us")


class DecimalType(DataType):
    """Decimal data type with fixed precision and scale."""

    def __init__(self, precision: int = 10, scale: int = 0):
        self.precision = precision
        self.scale = scale

    def __repr__(self):
        return f"DecimalType({self.precision}, {self.scale})"

    def __str__(self):
        return f"decimal({self.precision},{self.scale})"

    def to_arrow(self) -> pa.DataType:
        return pa.decimal128(self.precision, self.scale)


class StructField:
    """A field in a StructType, with a name, data type, and nullable flag."""

    def __init__(self, name: str, dataType: DataType, nullable: bool = True, metadata: dict | None = None):
        self.name = name
        self.dataType = dataType
        self.nullable = nullable
        self.metadata = metadata or {}

    def __repr__(self):
        return f"StructField('{self.name}', {self.dataType}, {self.nullable})"

    def simpleString(self) -> str:
        """Returns a readable string in the form ``name:type``."""
        return f"{self.name}:{self.dataType.simpleString()}"

    def to_arrow(self) -> pa.Field:
        """Converts to a PyArrow Field."""
        return pa.field(self.name, self.dataType.to_arrow(), nullable=self.nullable)


class StructType(DataType):
    def __init__(self, fields: list[StructField] | None = None):
        self.fields = fields or []

    def add(self, name: str, dataType: DataType, nullable: bool = True, metadata: dict | None = None) -> StructType:
        """Adds a new field and returns this StructType for chaining."""
        self.fields.append(StructField(name, dataType, nullable, metadata))
        return self

    def __repr__(self):
        return f"StructType({self.fields!r})"

    def __str__(self):
        fields_str = ",".join(f.simpleString() for f in self.fields)
        return f"struct<{fields_str}>"

    def __iter__(self):
        return iter(self.fields)

    def __len__(self):
        return len(self.fields)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.fields[key]
        if isinstance(key, str):
            for f in self.fields:
                if f.name == key:
                    return f
            raise KeyError(f"No field named '{key}'")
        raise TypeError(f"StructType indices must be int or str, not {type(key).__name__}")

    @property
    def fieldNames(self) -> list[str]:
        """Returns a list of field names."""
        return [f.name for f in self.fields]

    def to_arrow(self) -> pa.DataType:
        """Converts to a PyArrow struct type."""
        return pa.struct([f.to_arrow() for f in self.fields])


class ArrayType(DataType):
    """Array (list) data type containing elements of a single type."""

    def __init__(self, elementType: DataType, containsNull: bool = True):
        self.elementType = elementType
        self.containsNull = containsNull

    def __repr__(self):
        return f"ArrayType({self.elementType}, {self.containsNull})"

    def __str__(self):
        return f"array<{self.elementType}>"

    def to_arrow(self) -> pa.DataType:
        """Converts to a PyArrow list type."""
        return pa.list_(self.elementType.to_arrow())


class MapType(DataType):
    """Map (dictionary) data type with key and value types."""

    def __init__(self, keyType: DataType, valueType: DataType, valueContainsNull: bool = True):
        self.keyType = keyType
        self.valueType = valueType
        self.valueContainsNull = valueContainsNull

    def __repr__(self):
        return f"MapType({self.keyType}, {self.valueType}, {self.valueContainsNull})"

    def __str__(self):
        return f"map<{self.keyType},{self.valueType}>"

    def to_arrow(self) -> pa.DataType:
        """Converts to a PyArrow map type."""
        return pa.map_(self.keyType.to_arrow(), self.valueType.to_arrow())


class CharType(DataType):
    """Fixed-length character type."""

    def __init__(self, length: int):
        self.length = length

    def __repr__(self):
        return f"CharType({self.length})"

    def __str__(self):
        return f"char({self.length})"

    def to_arrow(self) -> pa.DataType:
        return pa.string()


class VarcharType(DataType):
    """Variable-length character type."""

    def __init__(self, length: int):
        self.length = length

    def __repr__(self):
        return f"VarcharType({self.length})"

    def __str__(self):
        return f"varchar({self.length})"

    def to_arrow(self) -> pa.DataType:
        return pa.string()


class DayTimeIntervalType(DataType):
    """Day-time interval type."""

    def __repr__(self):
        return "DayTimeIntervalType()"

    def __str__(self):
        return "interval day to second"

    def to_arrow(self) -> pa.DataType:
        return pa.duration("us")


# Arrow type -> Spark DataType mapping
_ARROW_TO_SPARK: dict[str, type[DataType]] = {}


def from_arrow_type(arrow_type: pa.DataType) -> DataType:
    """Convert an Arrow data type to a Spark-compatible DataType."""
    if pa.types.is_null(arrow_type):
        return NullType()
    if pa.types.is_boolean(arrow_type):
        return BooleanType()
    if pa.types.is_int8(arrow_type):
        return ByteType()
    if pa.types.is_int16(arrow_type):
        return ShortType()
    if pa.types.is_int32(arrow_type):
        return IntegerType()
    if pa.types.is_int64(arrow_type):
        return LongType()
    if pa.types.is_float32(arrow_type):
        return FloatType()
    if pa.types.is_float64(arrow_type):
        return DoubleType()
    if pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
        return StringType()
    if pa.types.is_binary(arrow_type) or pa.types.is_large_binary(arrow_type) or pa.types.is_fixed_size_binary(arrow_type):
        return BinaryType()
    if pa.types.is_date(arrow_type):
        return DateType()
    if pa.types.is_timestamp(arrow_type):
        if arrow_type.tz:
            return TimestampType()
        return TimestampNTZType()
    if pa.types.is_duration(arrow_type):
        return DayTimeIntervalType()
    if pa.types.is_decimal(arrow_type):
        return DecimalType(arrow_type.precision, arrow_type.scale)
    if pa.types.is_list(arrow_type) or pa.types.is_large_list(arrow_type):
        return ArrayType(from_arrow_type(arrow_type.value_type))
    if pa.types.is_map(arrow_type):
        return MapType(
            from_arrow_type(arrow_type.key_type),
            from_arrow_type(arrow_type.item_type),
        )
    if pa.types.is_struct(arrow_type):
        fields = [
            StructField(f.name, from_arrow_type(f.type), f.nullable)
            for f in arrow_type
        ]
        return StructType(fields)
    # Fallback
    return StringType()


def from_arrow_schema(schema: pa.Schema) -> StructType:
    """Convert an Arrow schema to a Spark-compatible StructType."""
    fields = [
        StructField(field.name, from_arrow_type(field.type), field.nullable)
        for field in schema
    ]
    return StructType(fields)
