// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

// Tests for the ORC data-file reader. The headline `test_read_java_iceberg_orc_fixture` reads a
// REAL Java-Iceberg 1.10.0 ORC file (written by `GenericAppenderFactory` over a primitive + logical
// + optional/null fixture, committed under `testdata/orc/`) and asserts field-id-correct rows and
// canonical Iceberg Arrow types — the true 1:1 interop oracle. The remaining tests cover projection
// (subset, reorder, missing optional/required), promotion, and per-type conversion against that
// same golden file.

use std::collections::HashMap;

use arrow_array::cast::AsArray;
use arrow_array::types::{
    Date32Type, Decimal128Type, Float32Type, Float64Type, Int32Type, Int64Type,
    Time64MicrosecondType, TimestampMicrosecondType,
};
use arrow_array::{Array, BooleanArray, FixedSizeBinaryArray, LargeBinaryArray, StringArray};
use parquet::arrow::PARQUET_FIELD_ID_META_KEY;

use super::*;
use crate::arrow::UTC_TIME_ZONE;
use crate::spec::NestedField;

// -------------------------------------------------------------------------------------------------
// Fixture + helpers
// -------------------------------------------------------------------------------------------------

/// The Java-Iceberg 1.10.0 golden ORC file (ZLIB-compressed, 14 columns, 3 rows).
const FIXTURE: &[u8] = include_bytes!("../../testdata/orc/iceberg_primitives.orc");

fn schema_of(fields: Vec<NestedField>) -> Schema {
    Schema::builder()
        .with_fields(fields.into_iter().map(|f| f.into()))
        .build()
        .expect("build iceberg schema")
}

fn prim(p: PrimitiveType) -> Type {
    Type::Primitive(p)
}

/// The full 14-column schema matching the fixture exactly (field ids 1..=14).
fn full_schema() -> Schema {
    schema_of(vec![
        NestedField::required(1, "id", prim(PrimitiveType::Long)),
        NestedField::optional(2, "int_col", prim(PrimitiveType::Int)),
        NestedField::optional(3, "float_col", prim(PrimitiveType::Float)),
        NestedField::optional(4, "double_col", prim(PrimitiveType::Double)),
        NestedField::optional(5, "bool_col", prim(PrimitiveType::Boolean)),
        NestedField::optional(6, "string_col", prim(PrimitiveType::String)),
        NestedField::optional(7, "date_col", prim(PrimitiveType::Date)),
        NestedField::optional(8, "time_col", prim(PrimitiveType::Time)),
        NestedField::optional(9, "ts_col", prim(PrimitiveType::Timestamp)),
        NestedField::optional(10, "tstz_col", prim(PrimitiveType::Timestamptz)),
        NestedField::optional(11, "binary_col", prim(PrimitiveType::Binary)),
        NestedField::optional(12, "decimal_col", prim(PrimitiveType::Decimal {
            precision: 10,
            scale: 2,
        })),
        NestedField::optional(13, "uuid_col", prim(PrimitiveType::Uuid)),
        NestedField::optional(14, "fixed_col", prim(PrimitiveType::Fixed(4))),
    ])
}

/// Decode the fixture against `expected`, asserting a single batch and returning it.
fn read_fixture(expected: &Schema) -> RecordBatch {
    let mut batches = read_orc_data_bytes(FIXTURE, expected, 1024).expect("decode ORC fixture");
    assert_eq!(batches.len(), 1, "fixture has one stripe → one batch");
    batches.pop().expect("one batch")
}

// -------------------------------------------------------------------------------------------------
// Headline interop test: read the REAL Java-Iceberg ORC file, by field-id
// -------------------------------------------------------------------------------------------------

#[test]
fn test_read_java_iceberg_orc_fixture() {
    let schema = full_schema();
    let batch = read_fixture(&schema);

    assert_eq!(batch.num_rows(), 3);
    assert_eq!(batch.num_columns(), 14);

    // Output schema equals the canonical Iceberg→Arrow conversion, in expected field order.
    let expected_arrow = schema_to_arrow_schema(&schema).expect("arrow schema");
    assert_eq!(
        batch.schema().fields(),
        expected_arrow.fields(),
        "output schema must equal schema_to_arrow_schema(expected)"
    );

    // Every output field carries its Iceberg field-id (the load-bearing U2 contract).
    for (pos, field) in batch.schema().fields().iter().enumerate() {
        let id = field
            .metadata()
            .get(PARQUET_FIELD_ID_META_KEY)
            .unwrap_or_else(|| panic!("field {pos} '{}' missing field-id", field.name()));
        assert_eq!(id, &(pos as i32 + 1).to_string());
    }

    // --- id (required LONG) ---
    let id = batch.column(0).as_primitive::<Int64Type>();
    assert_eq!(id.values(), &[1, 2, 3]);
    assert_eq!(id.null_count(), 0);

    // --- int_col (optional INT): row1 is null ---
    let int_col = batch.column(1).as_primitive::<Int32Type>();
    assert_eq!(int_col.value(0), 42);
    assert!(int_col.is_null(1));
    assert_eq!(int_col.value(2), -7);

    // --- float / double ---
    assert_eq!(batch.column(2).as_primitive::<Float32Type>().value(0), 1.5);
    assert_eq!(batch.column(3).as_primitive::<Float64Type>().value(0), 2.5);
    assert!(batch.column(2).is_null(1));

    // --- bool ---
    let bool_col = batch.column(4).as_any().downcast_ref::<BooleanArray>().unwrap();
    assert!(bool_col.value(0));
    assert!(bool_col.is_null(1));
    assert!(!bool_col.value(2));

    // --- string (row2 is empty string, not null) ---
    let string_col = batch.column(5).as_any().downcast_ref::<StringArray>().unwrap();
    assert_eq!(string_col.value(0), "hello");
    assert!(string_col.is_null(1));
    assert_eq!(string_col.value(2), "");

    // --- date (Date32, days): 2021-02-03 = 18661, 1969-12-31 = -1 ---
    let date_col = batch.column(6).as_primitive::<Date32Type>();
    assert_eq!(date_col.value(0), 18661);
    assert!(date_col.is_null(1));
    assert_eq!(date_col.value(2), -1);

    // --- time (Time64 µs): 12:34:56.789 = 45_296_789_000 µs ---
    assert_eq!(batch.column(7).data_type(), &DataType::Time64(TimeUnit::Microsecond));
    let time_col = batch.column(7).as_primitive::<Time64MicrosecondType>();
    assert_eq!(time_col.value(0), 45_296_789_000);
    assert!(time_col.is_null(1));
    assert_eq!(time_col.value(2), 0);

    // --- timestamp (µs, no tz): 2021-02-03 04:05:06.123456 = 1_612_325_106_123_456 ---
    assert_eq!(
        batch.column(8).data_type(),
        &DataType::Timestamp(TimeUnit::Microsecond, None)
    );
    let ts_col = batch.column(8).as_primitive::<TimestampMicrosecondType>();
    assert_eq!(ts_col.value(0), 1_612_325_106_123_456);
    assert!(ts_col.is_null(1));
    assert_eq!(ts_col.value(2), 915_148_800_000_000);

    // --- timestamptz (µs, UTC) ---
    assert_eq!(
        batch.column(9).data_type(),
        &DataType::Timestamp(TimeUnit::Microsecond, Some(UTC_TIME_ZONE.into()))
    );
    let tstz_col = batch.column(9).as_primitive::<TimestampMicrosecondType>();
    assert_eq!(tstz_col.value(0), 1_612_325_106_123_456);

    // --- binary (LargeBinary): row0 = [1,2,3,4,5], row2 = [] ---
    assert_eq!(batch.column(10).data_type(), &DataType::LargeBinary);
    let bin_col = batch.column(10).as_any().downcast_ref::<LargeBinaryArray>().unwrap();
    assert_eq!(bin_col.value(0), &[1, 2, 3, 4, 5]);
    assert!(bin_col.is_null(1));
    assert_eq!(bin_col.value(2), &[] as &[u8]);

    // --- decimal(10,2): 123.45 = 12345 unscaled, -99.99 = -9999 ---
    assert_eq!(batch.column(11).data_type(), &DataType::Decimal128(10, 2));
    let dec_col = batch.column(11).as_primitive::<Decimal128Type>();
    assert_eq!(dec_col.value(0), 12345);
    assert!(dec_col.is_null(1));
    assert_eq!(dec_col.value(2), -9999);

    // --- uuid (FixedSizeBinary(16)) ---
    assert_eq!(batch.column(12).data_type(), &DataType::FixedSizeBinary(16));
    let uuid_col = batch.column(12).as_any().downcast_ref::<FixedSizeBinaryArray>().unwrap();
    let expected_uuid: [u8; 16] = [
        0x12, 0x34, 0x56, 0x78, 0x12, 0x34, 0x56, 0x78, 0x12, 0x34, 0x56, 0x78, 0x12, 0x34, 0x56,
        0x78,
    ];
    assert_eq!(uuid_col.value(0), &expected_uuid);
    assert!(uuid_col.is_null(1));

    // --- fixed[4] (FixedSizeBinary(4)): row0 = 0xDEADBEEF ---
    assert_eq!(batch.column(13).data_type(), &DataType::FixedSizeBinary(4));
    let fixed_col = batch.column(13).as_any().downcast_ref::<FixedSizeBinaryArray>().unwrap();
    assert_eq!(fixed_col.value(0), &[0xDE, 0xAD, 0xBE, 0xEF]);
    assert!(fixed_col.is_null(1));
    assert_eq!(fixed_col.value(2), &[0, 0, 0, 0]);
}

// -------------------------------------------------------------------------------------------------
// Projection: subset, reorder (by field-id, NOT position)
// -------------------------------------------------------------------------------------------------

#[test]
fn test_projection_subset_and_reorder() {
    // Ask for string (id 6), then id (id 1) — reversed vs file order — plus decimal (id 12).
    let schema = schema_of(vec![
        NestedField::optional(6, "string_col", prim(PrimitiveType::String)),
        NestedField::required(1, "id", prim(PrimitiveType::Long)),
        NestedField::optional(12, "decimal_col", prim(PrimitiveType::Decimal {
            precision: 10,
            scale: 2,
        })),
    ]);
    let batch = read_fixture(&schema);

    assert_eq!(batch.num_columns(), 3);
    // Output is in EXPECTED order: string, id, decimal.
    assert_eq!(batch.schema().field(0).name(), "string_col");
    assert_eq!(batch.schema().field(1).name(), "id");
    assert_eq!(batch.schema().field(2).name(), "decimal_col");

    assert_eq!(
        batch.column(0).as_any().downcast_ref::<StringArray>().unwrap().value(0),
        "hello"
    );
    assert_eq!(batch.column(1).as_primitive::<Int64Type>().values(), &[1, 2, 3]);
    assert_eq!(batch.column(2).as_primitive::<Decimal128Type>().value(0), 12345);
}

// -------------------------------------------------------------------------------------------------
// Missing columns: optional → all-null, required-no-default → error
// -------------------------------------------------------------------------------------------------

#[test]
fn test_missing_optional_column_is_all_null() {
    // id 999 is absent from the file; optional → all-null column synthesized.
    let schema = schema_of(vec![
        NestedField::required(1, "id", prim(PrimitiveType::Long)),
        NestedField::optional(999, "ghost", prim(PrimitiveType::String)),
    ]);
    let batch = read_fixture(&schema);

    assert_eq!(batch.num_columns(), 2);
    let ghost = batch.column(1);
    assert_eq!(ghost.len(), 3);
    assert_eq!(ghost.null_count(), 3, "absent optional column is all-null");
}

#[test]
fn test_missing_required_column_errors() {
    // id 999 required + no initial-default → Java "Missing required field".
    let schema = schema_of(vec![
        NestedField::required(1, "id", prim(PrimitiveType::Long)),
        NestedField::required(999, "must_exist", prim(PrimitiveType::String)),
    ]);
    let err = read_orc_data_bytes(FIXTURE, &schema, 1024)
        .expect_err("required-missing must error");
    assert_eq!(err.kind(), ErrorKind::DataInvalid);
    assert!(
        err.message().contains("Missing required field"),
        "Java-parity message: {err}"
    );
}

// -------------------------------------------------------------------------------------------------
// Promotion + type-mismatch (parity with ORCSchemaUtil.getPromotedType / isSameType)
// -------------------------------------------------------------------------------------------------

#[test]
fn test_int_to_long_promotion() {
    // File col 2 is ORC INT; read it as Iceberg LONG → widened to Int64.
    let schema = schema_of(vec![NestedField::optional(
        2,
        "int_col",
        prim(PrimitiveType::Long),
    )]);
    let batch = read_fixture(&schema);
    assert_eq!(batch.column(0).data_type(), &DataType::Int64);
    let col = batch.column(0).as_primitive::<Int64Type>();
    assert_eq!(col.value(0), 42);
    assert!(col.is_null(1));
    assert_eq!(col.value(2), -7);
}

#[test]
fn test_float_to_double_promotion() {
    // File col 3 is ORC FLOAT; read as Iceberg DOUBLE → widened to Float64.
    let schema = schema_of(vec![NestedField::optional(
        3,
        "float_col",
        prim(PrimitiveType::Double),
    )]);
    let batch = read_fixture(&schema);
    assert_eq!(batch.column(0).data_type(), &DataType::Float64);
    assert_eq!(batch.column(0).as_primitive::<Float64Type>().value(0), 1.5);
}

#[test]
fn test_decimal_precision_promotion() {
    // File decimal is (10,2); read it as (12,2) — same scale, wider precision → Java promotion.
    let schema = schema_of(vec![NestedField::optional(
        12,
        "decimal_col",
        prim(PrimitiveType::Decimal {
            precision: 12,
            scale: 2,
        }),
    )]);
    let batch = read_fixture(&schema);
    assert_eq!(batch.column(0).data_type(), &DataType::Decimal128(12, 2));
    assert_eq!(batch.column(0).as_primitive::<Decimal128Type>().value(0), 12345);
}

#[test]
fn test_incompatible_type_errors() {
    // File col 2 is ORC INT; asking for STRING is not a sanctioned promotion → error.
    let schema = schema_of(vec![NestedField::optional(
        2,
        "int_col",
        prim(PrimitiveType::String),
    )]);
    let err = read_orc_data_bytes(FIXTURE, &schema, 1024).expect_err("int→string must error");
    assert_eq!(err.kind(), ErrorKind::DataInvalid);
    assert!(err.message().contains("Can not promote"), "Java-parity message: {err}");
}

#[test]
fn test_decimal_scale_mismatch_is_stricter_than_java() {
    // Reading file decimal(10,2) as an expected decimal(10,3) — a SCALE mismatch.
    //
    // This is a DELIBERATE divergence from Java in the SAFE direction, not a Java-parity claim.
    // Java `ORCSchemaUtil.buildOrcProjection` does NOT reject a scale mismatch: `getPromotedType`
    // returns empty when scales differ, control falls to `checkArgument(isSameType(...))`, and
    // `isSameType` for a non-timestamp type is `TYPE_MAPPING.containsEntry(typeId, category)`,
    // which for DECIMAL→DECIMAL is unconditionally true (no precision/scale check). So Java ACCEPTS
    // the projection and `GenericOrcReaders$DecimalReader.nonNullRead` then `setScale()`s to the
    // FILE's scale — a latent Java mismatch where the requested scale is silently ignored.
    //
    // The Rust reader instead rejects a scale mismatch up front (a stricter, safer guard). This is
    // unreachable on real Iceberg-written ORC — the writer always emits the table's declared scale,
    // so a read schema's scale always equals the file's — so the stricter guard never bites in
    // practice and never diverges observably from Java on a real file.
    let schema = schema_of(vec![NestedField::optional(
        12,
        "decimal_col",
        prim(PrimitiveType::Decimal {
            precision: 10,
            scale: 3,
        }),
    )]);
    let err = read_orc_data_bytes(FIXTURE, &schema, 1024)
        .expect_err("Rust rejects a decimal scale mismatch (stricter than Java)");
    assert_eq!(err.kind(), ErrorKind::DataInvalid);
    assert!(
        err.message().contains("Can not promote"),
        "stricter-than-Java guard reuses the promotion-failure message: {err}"
    );
}

// -------------------------------------------------------------------------------------------------
// Batching + edge cases
// -------------------------------------------------------------------------------------------------

#[test]
fn test_batch_size_splits_rows() {
    // batch_size 2 over 3 rows → batches of 2 + 1.
    let schema = schema_of(vec![NestedField::required(1, "id", prim(PrimitiveType::Long))]);
    let batches = read_orc_data_bytes(FIXTURE, &schema, 2).expect("decode in batches of 2");
    let total: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total, 3);
    assert!(batches.len() >= 2, "expected the 3 rows split across ≥2 batches");
}

#[test]
fn test_zero_batch_size_errors() {
    let schema = schema_of(vec![NestedField::required(1, "id", prim(PrimitiveType::Long))]);
    let err = read_orc_data_bytes(FIXTURE, &schema, 0).expect_err("batch_size 0 must error");
    assert_eq!(err.kind(), ErrorKind::DataInvalid);
}

#[test]
fn test_nested_type_unsupported() {
    // A struct field is rejected (nested ORC read is deferred).
    use crate::spec::StructType;
    let inner = StructType::new(vec![
        NestedField::required(100, "x", prim(PrimitiveType::Long)).into(),
    ]);
    let schema = schema_of(vec![NestedField::optional(
        50,
        "nested",
        Type::Struct(inner),
    )]);
    let err = read_orc_data_bytes(FIXTURE, &schema, 1024).expect_err("nested must error");
    assert_eq!(err.kind(), ErrorKind::FeatureUnsupported);
}

// -------------------------------------------------------------------------------------------------
// Async entry point over a local InputFile (mirrors read_avro_data_file)
// -------------------------------------------------------------------------------------------------

#[tokio::test]
async fn test_read_orc_data_file_via_input_file() {
    use std::sync::Arc;

    use tempfile::TempDir;

    use crate::io::FileIO;

    let dir = TempDir::new().expect("tempdir");
    let path = dir.path().join("fixture.orc");
    std::fs::write(&path, FIXTURE).expect("write fixture to temp");

    let file_io = FileIO::new_with_fs();
    let input = file_io
        .new_input(path.to_str().expect("utf-8 path"))
        .expect("new input file");

    let schema = Arc::new(full_schema());
    let batches = read_orc_data_file(&input, schema, 1024)
        .await
        .expect("read via InputFile");
    let total: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total, 3);
}

// Silence the unused-import lint for HashMap when only some tests reference it.
#[allow(dead_code)]
fn _touch() -> HashMap<i32, i32> {
    HashMap::new()
}
