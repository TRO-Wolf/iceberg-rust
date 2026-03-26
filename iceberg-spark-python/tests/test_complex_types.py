"""Tests for recursive complex type parsing in spark_type_from_name."""

import pyarrow as pa
import pytest

from iceberg_spark._internal.type_mapping import (
    _split_top_level,
    spark_type_from_name,
)
from iceberg_spark.functions import _parse_ddl_struct
from iceberg_spark.types import (
    ArrayType,
    DecimalType,
    DoubleType,
    IntegerType,
    LongType,
    MapType,
    StringType,
    StructField,
    StructType,
)


# ---- _split_top_level tests ------------------------------------------------


def test_split_top_level_simple():
    assert _split_top_level("a,b,c") == ["a", "b", "c"]


def test_split_top_level_no_delimiter():
    assert _split_top_level("abc") == ["abc"]


def test_split_top_level_nested_angle_brackets():
    assert _split_top_level("map<string,int>,array<string>") == [
        "map<string,int>",
        "array<string>",
    ]


def test_split_top_level_nested_struct():
    assert _split_top_level("struct<x:int,y:int>,string") == [
        "struct<x:int,y:int>",
        "string",
    ]


def test_split_top_level_deeply_nested():
    assert _split_top_level(
        "array<map<string,int>>,struct<a:int,b:string>"
    ) == ["array<map<string,int>>", "struct<a:int,b:string>"]


def test_split_top_level_parentheses():
    assert _split_top_level("decimal(10,2),int") == ["decimal(10,2)", "int"]


def test_split_top_level_empty():
    assert _split_top_level("") == []


# ---- Simple types -----------------------------------------------------------


def test_simple_int():
    t = spark_type_from_name("int")
    assert isinstance(t, IntegerType)


def test_simple_string():
    t = spark_type_from_name("string")
    assert isinstance(t, StringType)


def test_simple_bigint():
    t = spark_type_from_name("bigint")
    assert isinstance(t, LongType)


def test_decimal_with_params():
    t = spark_type_from_name("decimal(10,2)")
    assert isinstance(t, DecimalType)
    assert t.precision == 10
    assert t.scale == 2


def test_decimal_no_params():
    t = spark_type_from_name("decimal")
    assert isinstance(t, DecimalType)
    assert t.precision == 10
    assert t.scale == 0


def test_simple_with_whitespace():
    t = spark_type_from_name("  string  ")
    assert isinstance(t, StringType)


# ---- Array types ------------------------------------------------------------


def test_array_string():
    t = spark_type_from_name("array<string>")
    assert isinstance(t, ArrayType)
    assert isinstance(t.elementType, StringType)


def test_array_int():
    t = spark_type_from_name("array<int>")
    assert isinstance(t, ArrayType)
    assert isinstance(t.elementType, IntegerType)


def test_array_case_insensitive():
    t = spark_type_from_name("ARRAY<STRING>")
    assert isinstance(t, ArrayType)
    assert isinstance(t.elementType, StringType)


def test_array_to_arrow():
    t = spark_type_from_name("array<string>")
    assert t.to_arrow() == pa.list_(pa.utf8())


# ---- Map types --------------------------------------------------------------


def test_map_string_int():
    t = spark_type_from_name("map<string,int>")
    assert isinstance(t, MapType)
    assert isinstance(t.keyType, StringType)
    assert isinstance(t.valueType, IntegerType)


def test_map_with_spaces():
    t = spark_type_from_name("map<string, double>")
    assert isinstance(t, MapType)
    assert isinstance(t.keyType, StringType)
    assert isinstance(t.valueType, DoubleType)


def test_map_to_arrow():
    t = spark_type_from_name("map<string,int>")
    assert t.to_arrow() == pa.map_(pa.utf8(), pa.int32())


def test_map_wrong_arity():
    with pytest.raises(ValueError, match="MAP type requires exactly 2"):
        spark_type_from_name("map<string>")


# ---- Struct types -----------------------------------------------------------


def test_struct_basic():
    t = spark_type_from_name("struct<name:string,age:int>")
    assert isinstance(t, StructType)
    assert len(t) == 2
    assert t[0].name == "name"
    assert isinstance(t[0].dataType, StringType)
    assert t[1].name == "age"
    assert isinstance(t[1].dataType, IntegerType)


def test_struct_to_arrow():
    t = spark_type_from_name("struct<name:string,age:int>")
    expected = pa.struct([pa.field("name", pa.utf8()), pa.field("age", pa.int32())])
    assert t.to_arrow() == expected


def test_struct_missing_colon():
    with pytest.raises(ValueError, match="Struct field must be"):
        spark_type_from_name("struct<badfield>")


# ---- Nested types -----------------------------------------------------------


def test_array_of_map():
    t = spark_type_from_name("array<map<string,int>>")
    assert isinstance(t, ArrayType)
    inner = t.elementType
    assert isinstance(inner, MapType)
    assert isinstance(inner.keyType, StringType)
    assert isinstance(inner.valueType, IntegerType)


def test_map_with_struct_value():
    t = spark_type_from_name("map<string,struct<x:int,y:int>>")
    assert isinstance(t, MapType)
    assert isinstance(t.keyType, StringType)
    assert isinstance(t.valueType, StructType)
    assert len(t.valueType) == 2
    assert t.valueType[0].name == "x"
    assert t.valueType[1].name == "y"


def test_deeply_nested():
    t = spark_type_from_name("array<struct<name:string,scores:array<double>>>")
    assert isinstance(t, ArrayType)
    inner = t.elementType
    assert isinstance(inner, StructType)
    assert len(inner) == 2
    assert inner[0].name == "name"
    assert isinstance(inner[0].dataType, StringType)
    assert inner[1].name == "scores"
    assert isinstance(inner[1].dataType, ArrayType)
    assert isinstance(inner[1].dataType.elementType, DoubleType)


def test_struct_with_complex_fields():
    t = spark_type_from_name("struct<data:array<string>,meta:map<string,int>>")
    assert isinstance(t, StructType)
    assert len(t) == 2
    assert t[0].name == "data"
    assert isinstance(t[0].dataType, ArrayType)
    assert isinstance(t[0].dataType.elementType, StringType)
    assert t[1].name == "meta"
    assert isinstance(t[1].dataType, MapType)
    assert isinstance(t[1].dataType.keyType, StringType)
    assert isinstance(t[1].dataType.valueType, IntegerType)


def test_nested_to_arrow_roundtrip():
    t = spark_type_from_name("map<string,struct<x:int,y:int>>")
    arrow_t = t.to_arrow()
    assert pa.types.is_map(arrow_t)
    assert pa.types.is_struct(arrow_t.item_type)


# ---- _parse_ddl_struct with nested types ------------------------------------


def test_parse_ddl_struct_nested_array():
    result = _parse_ddl_struct("struct<items:array<string>,count:int>")
    assert isinstance(result, pa.DataType)
    assert pa.types.is_struct(result)
    assert len(result) == 2
    assert result.field("items").type == pa.list_(pa.utf8())
    assert result.field("count").type == pa.int32()


def test_parse_ddl_struct_nested_map():
    result = _parse_ddl_struct("struct<data:map<string,int>,name:string>")
    assert pa.types.is_struct(result)
    assert result.field("data").type == pa.map_(pa.utf8(), pa.int32())
    assert result.field("name").type == pa.utf8()


def test_parse_ddl_struct_bare_nested():
    """Bare field list (no struct<...> wrapper) with complex types."""
    result = _parse_ddl_struct("items:array<string>,data:map<string,int>")
    assert pa.types.is_struct(result)
    assert result.field("items").type == pa.list_(pa.utf8())
    assert result.field("data").type == pa.map_(pa.utf8(), pa.int32())


# ---- Error cases ------------------------------------------------------------


def test_unknown_type():
    with pytest.raises(ValueError, match="Unknown Spark SQL type"):
        spark_type_from_name("unknown_type")


# ---- varchar/char with length -----------------------------------------------


def test_varchar_plain():
    assert isinstance(spark_type_from_name("varchar"), StringType)


def test_varchar_with_length():
    t = spark_type_from_name("varchar(100)")
    assert isinstance(t, StringType)


def test_char_with_length():
    t = spark_type_from_name("char(10)")
    assert isinstance(t, StringType)


def test_varchar_in_struct():
    t = spark_type_from_name("struct<name:varchar(255),age:int>")
    assert isinstance(t, StructType)
    assert isinstance(t[0].dataType, StringType)
    assert isinstance(t[1].dataType, IntegerType)


# ---- Unbalanced brackets ---------------------------------------------------


def test_unbalanced_missing_close():
    with pytest.raises(ValueError, match="Unbalanced brackets"):
        _split_top_level("map<string,int")


def test_unbalanced_extra_close():
    with pytest.raises(ValueError, match="Unbalanced brackets"):
        _split_top_level("string>,int")


def test_unbalanced_paren():
    with pytest.raises(ValueError, match="Unbalanced brackets"):
        _split_top_level("decimal(10,2")


# ---- Decimal inside complex types ------------------------------------------


def test_decimal_in_array():
    t = spark_type_from_name("array<decimal(10,2)>")
    assert isinstance(t, ArrayType)
    assert isinstance(t.elementType, DecimalType)
    assert t.elementType.precision == 10
    assert t.elementType.scale == 2


def test_decimal_in_map_value():
    t = spark_type_from_name("map<string,decimal(18,4)>")
    assert isinstance(t, MapType)
    assert isinstance(t.valueType, DecimalType)
    assert t.valueType.precision == 18
    assert t.valueType.scale == 4


def test_decimal_in_struct():
    t = spark_type_from_name("struct<amount:decimal(10,2),rate:decimal(5,3)>")
    assert isinstance(t, StructType)
    assert t[0].dataType.precision == 10
    assert t[1].dataType.precision == 5


# ---- _parse_ddl_struct space-separated form ---------------------------------


def test_parse_ddl_struct_space_separated_complex():
    """Space-separated form with complex types."""
    result = _parse_ddl_struct("items array<string>, count int")
    assert pa.types.is_struct(result)
    assert result.field("items").type == pa.list_(pa.utf8())
    assert result.field("count").type == pa.int32()
