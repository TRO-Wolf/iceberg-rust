"""Tests for the Spark-compatible type system."""

import pyarrow as pa

from iceberg_spark.types import (
    ArrayType,
    BinaryType,
    BooleanType,
    ByteType,
    CharType,
    DateType,
    DayTimeIntervalType,
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
    VarcharType,
    from_arrow_schema,
    from_arrow_type,
)


def test_primitive_types_to_arrow():
    assert NullType().to_arrow() == pa.null()
    assert BooleanType().to_arrow() == pa.bool_()
    assert ByteType().to_arrow() == pa.int8()
    assert ShortType().to_arrow() == pa.int16()
    assert IntegerType().to_arrow() == pa.int32()
    assert LongType().to_arrow() == pa.int64()
    assert FloatType().to_arrow() == pa.float32()
    assert DoubleType().to_arrow() == pa.float64()
    assert StringType().to_arrow() == pa.string()
    assert BinaryType().to_arrow() == pa.binary()
    assert DateType().to_arrow() == pa.date32()
    assert TimestampType().to_arrow() == pa.timestamp("us")


def test_decimal_type():
    dt = DecimalType(18, 6)
    assert dt.precision == 18
    assert dt.scale == 6
    assert dt.to_arrow() == pa.decimal128(18, 6)
    assert str(dt) == "decimal(18,6)"


def test_struct_type():
    st = StructType([
        StructField("id", IntegerType()),
        StructField("name", StringType()),
    ])
    assert len(st) == 2
    assert st.fieldNames == ["id", "name"]
    assert st["id"].dataType.to_arrow() == pa.int32()
    assert st[0].name == "id"


def test_struct_type_add():
    st = StructType()
    st.add("id", IntegerType()).add("name", StringType(), nullable=False)
    assert len(st) == 2
    assert not st["name"].nullable


def test_array_type():
    at = ArrayType(IntegerType())
    assert at.to_arrow() == pa.list_(pa.int32())
    assert str(at) == "array<int>"


def test_map_type():
    mt = MapType(StringType(), IntegerType())
    assert mt.to_arrow() == pa.map_(pa.string(), pa.int32())
    assert str(mt) == "map<string,int>"


def test_from_arrow_type_primitives():
    assert isinstance(from_arrow_type(pa.null()), NullType)
    assert isinstance(from_arrow_type(pa.bool_()), BooleanType)
    assert isinstance(from_arrow_type(pa.int32()), IntegerType)
    assert isinstance(from_arrow_type(pa.int64()), LongType)
    assert isinstance(from_arrow_type(pa.float64()), DoubleType)
    assert isinstance(from_arrow_type(pa.string()), StringType)
    assert isinstance(from_arrow_type(pa.date32()), DateType)


def test_from_arrow_type_timestamp():
    assert isinstance(from_arrow_type(pa.timestamp("us", tz="UTC")), TimestampType)
    assert isinstance(from_arrow_type(pa.timestamp("us")), TimestampNTZType)


def test_from_arrow_type_decimal():
    dt = from_arrow_type(pa.decimal128(10, 2))
    assert isinstance(dt, DecimalType)
    assert dt.precision == 10
    assert dt.scale == 2


def test_from_arrow_type_nested():
    # List
    dt = from_arrow_type(pa.list_(pa.int32()))
    assert isinstance(dt, ArrayType)
    assert isinstance(dt.elementType, IntegerType)

    # Map
    dt = from_arrow_type(pa.map_(pa.string(), pa.int64()))
    assert isinstance(dt, MapType)
    assert isinstance(dt.keyType, StringType)
    assert isinstance(dt.valueType, LongType)

    # Struct
    dt = from_arrow_type(pa.struct([pa.field("x", pa.int32())]))
    assert isinstance(dt, StructType)
    assert len(dt) == 1
    assert dt[0].name == "x"


def test_from_arrow_schema():
    schema = pa.schema([
        pa.field("id", pa.int64()),
        pa.field("name", pa.string()),
        pa.field("score", pa.float64(), nullable=True),
    ])
    st = from_arrow_schema(schema)
    assert isinstance(st, StructType)
    assert len(st) == 3
    assert st.fieldNames == ["id", "name", "score"]
    assert isinstance(st["id"].dataType, LongType)
    assert isinstance(st["name"].dataType, StringType)
    assert isinstance(st["score"].dataType, DoubleType)
    assert st["score"].nullable is True


def test_type_string_repr():
    assert str(IntegerType()) == "int"
    assert str(LongType()) == "bigint"
    assert str(StringType()) == "string"
    assert str(BooleanType()) == "boolean"
    assert str(DoubleType()) == "double"
    assert str(DateType()) == "date"
    assert str(TimestampType()) == "timestamp"


# --- Task 11A: Type System Gaps ---

def test_char_type_to_arrow():
    assert CharType(10).to_arrow() == pa.string()


def test_char_type_simple_string():
    assert CharType(10).simpleString() == "char(10)"
    assert CharType(1).simpleString() == "char(1)"


def test_char_type_str_repr():
    assert str(CharType(10)) == "char(10)"
    assert repr(CharType(10)) == "CharType(10)"
    assert CharType(255).length == 255


def test_varchar_type_to_arrow():
    assert VarcharType(255).to_arrow() == pa.string()


def test_varchar_type_simple_string():
    assert VarcharType(255).simpleString() == "varchar(255)"
    assert VarcharType(1).simpleString() == "varchar(1)"


def test_varchar_type_str_repr():
    assert str(VarcharType(100)) == "varchar(100)"
    assert repr(VarcharType(100)) == "VarcharType(100)"
    assert VarcharType(64).length == 64


def test_day_time_interval_type_to_arrow():
    assert DayTimeIntervalType().to_arrow() == pa.duration("us")


def test_day_time_interval_type_simple_string():
    assert DayTimeIntervalType().simpleString() == "interval day to second"


def test_day_time_interval_type_str_repr():
    assert str(DayTimeIntervalType()) == "interval day to second"
    assert repr(DayTimeIntervalType()) == "DayTimeIntervalType()"


def test_day_time_interval_from_arrow_roundtrip():
    dt = from_arrow_type(pa.duration("us"))
    assert isinstance(dt, DayTimeIntervalType)
    assert dt.to_arrow() == pa.duration("us")


def test_new_types_importable_from_package():
    from iceberg_spark import CharType, DayTimeIntervalType, VarcharType  # noqa: F401

    import iceberg_spark
    assert "CharType" in iceberg_spark.__all__
    assert "VarcharType" in iceberg_spark.__all__
    assert "DayTimeIntervalType" in iceberg_spark.__all__
