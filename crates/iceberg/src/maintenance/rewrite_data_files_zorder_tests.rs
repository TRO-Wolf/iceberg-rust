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

use std::collections::HashMap;
use std::sync::Arc;

use crate::maintenance::rewrite_data_files::RewriteDataFiles;
use crate::maintenance::rewrite_data_files::tests::{append_files, local_fs_catalog};
use crate::maintenance::rewrite_data_files_sort_harness::{
    OracleRow, concatenated_rows, live_output_paths, oracle_table, output_files,
};
use crate::maintenance::rewrite_data_files_sort_vectors::{
    SPARK_ZORDER_1, SPARK_ZORDER_2, SPARK_ZORDER_3_TYPES, SPARK_ZORDER_MAX_OUTPUT,
    SPARK_ZORDER_PARTITION_COL, SPARK_ZORDER_VAR_LEN, Z_ID, Z_ID_V, Z_S_ID_MAX4, Z_S_ID_VAR2,
    Z_S_TS_ID,
};
use crate::maintenance::rewrite_data_files_zorder::{
    byte_truncate_or_fill, floating_point_ordered_bytes, interleave_bits, string_to_ordered_bytes,
    whole_number_ordered_bytes,
};
use crate::maintenance::{RewriteStrategy, ZOrderSpec};
use crate::spec::{
    DataContentType, DataFile, FormatVersion, NestedField, PrimitiveType, Schema, StructType, Type,
};
use crate::table::Table;
use crate::{Catalog, NamespaceIdent, TableCreation};

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[test]
fn whole_numbers_match_javas_ordered_bytes() {
    for (value, expected) in [
        (-5i64, "7ffffffffffffffb"),
        (-1, "7fffffffffffffff"),
        (0, "8000000000000000"),
        (7, "8000000000000007"),
    ] {
        assert_eq!(hex(&whole_number_ordered_bytes(value)), expected, "{value}");
    }
}

#[test]
fn floating_point_matches_javas_ordered_bytes_including_its_shift_quirk() {
    for (value, expected) in [
        (-2.5f64, "3ffbffff80080000"),
        (-0.0, "7fffffff00000000"),
        (0.0, "8000000000000000"),
        (1.5, "bff800007ff00000"),
        (f64::INFINITY, "fff00000ffe00000"),
        (f64::NEG_INFINITY, "000fffffffe00000"),
        (f64::NAN, "fff80000fff00000"),
        (f64::from_bits(0xfff8_0000_0000_0000), "fff80000fff00000"),
        (f64::from_bits(0x7ff8_0000_0000_0001), "fff80000fff00000"),
        (f64::from_bits(0xfff8_0000_0000_abcd), "fff80000fff00000"),
        (f64::from_bits(0x7ff0_0000_0000_0001), "fff80000fff00000"),
    ] {
        assert_eq!(
            hex(&floating_point_ordered_bytes(value)),
            expected,
            "{value}"
        );
    }
}

#[test]
fn every_nan_collapses_to_javas_canonical_nan_in_the_z_value() {
    let nans = [
        f64::NAN,
        f64::from_bits(0xfff8_0000_0000_0000),
        f64::from_bits(0x7ff8_0000_0000_0001),
        f64::from_bits(0xfff8_0000_0000_abcd),
        f64::from_bits(0x7ff0_0000_0000_0001),
    ];
    let canonical = floating_point_ordered_bytes(f64::NAN);
    for value in nans {
        assert_eq!(
            floating_point_ordered_bytes(value),
            canonical,
            "NaN bits {:016x} must encode as Java's canonical NaN",
            value.to_bits()
        );
    }
    let smallest = floating_point_ordered_bytes(f64::NEG_INFINITY);
    let largest = floating_point_ordered_bytes(f64::INFINITY);
    assert!(
        canonical > largest && canonical > smallest,
        "a NaN must sort above every number, not below them"
    );
}

#[tokio::test]
async fn a_float_column_canonicalises_its_nans_like_javas_widening_encoder() {
    use arrow_array::{Float32Array, RecordBatch};

    let schema = Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![Arc::new(NestedField::optional(
            1,
            "f",
            Type::Primitive(PrimitiveType::Float),
        ))])
        .build()
        .expect("build the float schema");
    let arrow_schema =
        Arc::new(crate::arrow::schema_to_arrow_schema(&schema).expect("arrow schema"));
    let batch = RecordBatch::try_new(arrow_schema.clone(), vec![
        Arc::new(Float32Array::from(vec![
            f32::from_bits(0x7fc0_0000),
            f32::from_bits(0xffc0_0000),
            f32::from_bits(0xffc0_abcd),
            f32::from_bits(0x7f80_0001),
            1.5f32,
        ])) as arrow_array::ArrayRef,
    ])
    .expect("float batch");

    let encoder = crate::maintenance::rewrite_data_files_zorder::ZOrderEncoder::build(
        &["f".to_string()],
        &schema,
        &arrow_schema,
        8,
        i32::MAX as usize,
    )
    .expect("z encoder");
    let mut keys = vec![Vec::new(); batch.num_rows()];
    encoder.encode(&batch, &mut keys).expect("encode");

    for (row, key) in keys.iter().take(4).enumerate() {
        assert_eq!(
            hex(key),
            "fff80000fff00000",
            "float NaN row {row} must widen to Java's canonical NaN"
        );
    }
    assert_eq!(hex(&keys[4]), "bff800007ff00000");
    assert!(keys[4] < keys[0], "1.5 must sort below a NaN, as in Java");
}

#[test]
fn strings_truncate_on_a_character_boundary_like_javas_encoder() {
    for (value, width, expected) in [
        ("a\u{e9}", 2, "6100"),
        ("\u{e9}a", 2, "c3a9"),
        ("ab", 2, "6162"),
        ("", 2, "0000"),
        ("\u{20ac}", 2, "0000"),
        ("a\u{e9}", 4, "61c3a900"),
        ("\u{20ac}x", 4, "e282ac78"),
        ("abcdef", 4, "61626364"),
    ] {
        let mut buffer = vec![0u8; width];
        string_to_ordered_bytes(value, &mut buffer);
        assert_eq!(hex(&buffer), expected, "{value:?} in {width} bytes");
    }
}

#[test]
fn bytes_truncate_or_fill_to_the_column_width() {
    let mut buffer = vec![0u8; 4];
    byte_truncate_or_fill(&[0xAB, 0xCD], &mut buffer);
    assert_eq!(hex(&buffer), "abcd0000");
    byte_truncate_or_fill(&[1, 2, 3, 4, 5, 6], &mut buffer);
    assert_eq!(hex(&buffer), "01020304");
    byte_truncate_or_fill(&[], &mut buffer);
    assert_eq!(hex(&buffer), "00000000");
}

#[test]
fn interleave_bits_matches_java_over_uneven_columns_and_its_output_cap() {
    let columns = vec![vec![0xFF, 0x00], vec![0x0F], vec![0xAA, 0x55, 0x33]];
    let mut out = Vec::new();
    interleave_bits(&columns, 6, &mut out);
    assert_eq!(hex(&out), "b2cfbe111133");

    let mut capped = Vec::new();
    interleave_bits(&columns, 3, &mut capped);
    assert_eq!(hex(&capped), "b2cfbe");
}

fn java_z_values(vectors: &[(i64, &str)], rows: &[OracleRow]) -> Vec<String> {
    rows.iter()
        .map(|row| {
            vectors
                .iter()
                .find(|(index, _)| *index == row.index)
                .map(|(_, value)| (*value).to_string())
                .unwrap_or_else(|| panic!("no Java z value recorded for row {}", row.index))
        })
        .collect()
}

fn assert_same_z_order(
    cell: &str,
    vectors: &[(i64, &str)],
    spark_order: &[i64],
    rewritten: &[OracleRow],
) {
    let spark_rows: Vec<OracleRow> = spark_order
        .iter()
        .map(|index| crate::maintenance::rewrite_data_files_sort_harness::oracle_row(*index))
        .collect();
    let expected = java_z_values(vectors, &spark_rows);
    let actual = java_z_values(vectors, rewritten);
    assert_eq!(
        expected, actual,
        "{cell}: the rewritten rows are not in Spark's z-value order"
    );
    for window in actual.windows(2) {
        assert!(
            window[0] <= window[1],
            "{cell}: z value {} precedes {} but sorts after it",
            window[0],
            window[1]
        );
    }
}

async fn zorder_cell(
    cell: &str,
    spec: ZOrderSpec,
    partitioned: bool,
) -> (
    Vec<OracleRow>,
    Vec<Option<i32>>,
    Vec<(Option<String>, Vec<OracleRow>)>,
) {
    let (catalog, _guard) = local_fs_catalog().await;
    let table = oracle_table(&catalog, FormatVersion::V2, partitioned, None).await;
    RewriteDataFiles::new(table.clone())
        .strategy(RewriteStrategy::ZOrder(spec))
        .rewrite_all(true)
        .execute(&catalog)
        .await
        .unwrap_or_else(|error| panic!("{cell}: z-order rewrite failed: {error}"));
    let table = catalog
        .load_table(table.identifier())
        .await
        .expect("reload");
    let files = output_files(&table).await;
    let stamps = files.iter().map(|file| file.sort_order_id).collect();
    let per_partition = files
        .iter()
        .map(|file| {
            (
                file.rows.first().and_then(|row| row.cat.clone()),
                file.rows.clone(),
            )
        })
        .collect();
    (concatenated_rows(&files), stamps, per_partition)
}

#[tokio::test]
async fn zorder_over_one_column_matches_the_spark_row_order_and_stamps_zero() {
    let (rows, stamps, _) = zorder_cell("ZORDER-1", ZOrderSpec::new(["id"]), false).await;
    assert_eq!(stamps, vec![Some(0); stamps.len()]);
    assert_same_z_order("ZORDER-1", Z_ID, SPARK_ZORDER_1, &rows);
}

#[tokio::test]
async fn zorder_over_two_columns_matches_the_spark_row_order() {
    let (rows, stamps, _) = zorder_cell("ZORDER-2", ZOrderSpec::new(["id", "v"]), false).await;
    assert_eq!(stamps, vec![Some(0); stamps.len()]);
    assert_same_z_order("ZORDER-2", Z_ID_V, SPARK_ZORDER_2, &rows);
}

#[tokio::test]
async fn zorder_over_a_string_a_timestamp_and_a_long_matches_the_spark_row_order() {
    let (rows, _, _) =
        zorder_cell("ZORDER-3-TYPES", ZOrderSpec::new(["s", "ts", "id"]), false).await;
    assert_same_z_order("ZORDER-3-TYPES", Z_S_TS_ID, SPARK_ZORDER_3_TYPES, &rows);
}

#[tokio::test]
async fn zorder_var_length_contribution_changes_the_row_order_like_spark() {
    let (rows, _, _) = zorder_cell(
        "ZORDER-VAR-LEN",
        ZOrderSpec::new(["s", "id"]).var_length_contribution(2),
        false,
    )
    .await;
    assert_same_z_order("ZORDER-VAR-LEN", Z_S_ID_VAR2, SPARK_ZORDER_VAR_LEN, &rows);
}

#[tokio::test]
async fn zorder_max_output_size_truncates_the_interleaved_value_like_spark() {
    let (rows, _, _) = zorder_cell(
        "ZORDER-MAX-OUTPUT",
        ZOrderSpec::new(["s", "id"]).max_output_size(4),
        false,
    )
    .await;
    assert_same_z_order(
        "ZORDER-MAX-OUTPUT",
        Z_S_ID_MAX4,
        SPARK_ZORDER_MAX_OUTPUT,
        &rows,
    );
}

#[tokio::test]
async fn zorder_drops_an_identity_partition_column_from_the_tuple() {
    let (_, stamps, per_partition) =
        zorder_cell("ZORDER-PARTITION-COL", ZOrderSpec::new(["cat", "id"]), true).await;
    assert_eq!(stamps, vec![Some(0); stamps.len()]);
    assert_eq!(per_partition.len(), 4);
    for (partition, spark_rows) in SPARK_ZORDER_PARTITION_COL {
        let (_, rows) = per_partition
            .iter()
            .find(|(cat, _)| cat.as_deref().unwrap_or("") == *partition)
            .unwrap_or_else(|| panic!("no output file for partition '{partition}'"));
        assert_same_z_order("ZORDER-PARTITION-COL", Z_ID, spark_rows, rows);
    }
}

#[tokio::test]
async fn zorder_preconditions_are_refused_with_javas_messages() {
    let (catalog, _guard) = local_fs_catalog().await;
    let table = oracle_table(&catalog, FormatVersion::V2, true, None).await;

    let cases: Vec<(ZOrderSpec, &str)> = vec![
        (
            ZOrderSpec::new(Vec::<String>::new()),
            "Cannot ZOrder when no columns are specified",
        ),
        (
            ZOrderSpec::new(["cat"]),
            "Cannot ZOrder, all columns provided were identity partition columns and cannot be used",
        ),
        (
            ZOrderSpec::new(["id", "nope"]),
            "Cannot find column 'nope' in table schema (case sensitive = false): struct<1: id: optional long, 2: cat: optional string, 3: ts: optional timestamptz, 4: v: optional double, 5: s: optional string>",
        ),
        (
            ZOrderSpec::new(["id"]).var_length_contribution(0),
            "Cannot use less than 1 byte for variable length types with ZOrder, 'var-length-contribution' was set to 0",
        ),
        (
            ZOrderSpec::new(["id"]).max_output_size(0),
            "Cannot have the interleaved ZOrder value use less than 1 byte, 'max-output-size' was set to 0",
        ),
    ];
    for (spec, expected) in cases {
        let error = RewriteDataFiles::new(table.clone())
            .strategy(RewriteStrategy::ZOrder(spec))
            .rewrite_all(true)
            .execute(&catalog)
            .await
            .expect_err("the z-order tuple must be refused");
        assert_eq!(error.message(), expected);
    }
}

#[tokio::test]
async fn zorder_over_an_unsupported_type_is_refused_like_spark() {
    let (catalog, _guard) = local_fs_catalog().await;
    let schema = Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            Arc::new(NestedField::optional(
                1,
                "id",
                Type::Primitive(PrimitiveType::Long),
            )),
            Arc::new(NestedField::optional(
                2,
                "d",
                Type::Primitive(PrimitiveType::Decimal {
                    precision: 10,
                    scale: 2,
                }),
            )),
        ])
        .build()
        .expect("build the decimal schema");
    let namespace = NamespaceIdent::new(format!("ns-{}", uuid::Uuid::new_v4()));
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("create namespace");
    let table = catalog
        .create_table(&namespace, TableCreation {
            name: "t".to_string(),
            location: None,
            schema,
            partition_spec: None,
            sort_order: None,
            properties: HashMap::new(),
            format_version: FormatVersion::V2,
        })
        .await
        .expect("create the decimal table");

    let error = RewriteDataFiles::new(table)
        .strategy(RewriteStrategy::ZOrder(ZOrderSpec::new(["d", "id"])))
        .rewrite_all(true)
        .execute(&catalog)
        .await
        .expect_err("a decimal column cannot be z-ordered");
    assert_eq!(
        error.message(),
        "Cannot use column d of type decimal(10,2) in ZOrdering, the type is unsupported"
    );
}

fn nested_namesake_schema() -> Schema {
    Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            Arc::new(NestedField::optional(
                1,
                "id",
                Type::Primitive(PrimitiveType::Long),
            )),
            Arc::new(NestedField::optional(
                2,
                "st",
                Type::Struct(StructType::new(vec![
                    Arc::new(NestedField::optional(
                        4,
                        "id",
                        Type::Primitive(PrimitiveType::Long),
                    )),
                    Arc::new(NestedField::optional(
                        5,
                        "d",
                        Type::Primitive(PrimitiveType::Decimal {
                            precision: 10,
                            scale: 2,
                        }),
                    )),
                ])),
            )),
            Arc::new(NestedField::optional(
                3,
                "v",
                Type::Primitive(PrimitiveType::Double),
            )),
        ])
        .build()
        .expect("build the nested schema")
}

async fn write_nested_namesake_file(
    table: &Table,
    file_name: &str,
    rows: &[(Option<i64>, i64)],
) -> DataFile {
    use arrow_array::{Decimal128Array, Float64Array, Int64Array, RecordBatch, StructArray};
    use arrow_schema::DataType;

    use crate::spec::Struct;
    use crate::writer::file_writer::{FileWriter, FileWriterBuilder, ParquetWriterBuilder};

    let schema = table.metadata().current_schema();
    let arrow_schema =
        Arc::new(crate::arrow::schema_to_arrow_schema(schema).expect("arrow schema"));
    let DataType::Struct(nested_fields) = arrow_schema
        .field_with_name("st")
        .expect("the st column")
        .data_type()
        .clone()
    else {
        panic!("st must be an arrow struct");
    };
    let nested = StructArray::new(
        nested_fields,
        vec![
            Arc::new(Int64Array::from(
                rows.iter().map(|(_, nested)| *nested).collect::<Vec<i64>>(),
            )) as arrow_array::ArrayRef,
            Arc::new(
                Decimal128Array::from(
                    rows.iter()
                        .map(|(_, nested)| i128::from(*nested) * 100)
                        .collect::<Vec<i128>>(),
                )
                .with_precision_and_scale(10, 2)
                .expect("decimal precision"),
            ) as arrow_array::ArrayRef,
        ],
        None,
    );
    let batch = RecordBatch::try_new(arrow_schema.clone(), vec![
        Arc::new(Int64Array::from(
            rows.iter().map(|(id, _)| *id).collect::<Vec<Option<i64>>>(),
        )) as arrow_array::ArrayRef,
        Arc::new(nested) as arrow_array::ArrayRef,
        Arc::new(Float64Array::from(
            rows.iter()
                .map(|(_, nested)| *nested as f64)
                .collect::<Vec<f64>>(),
        )) as arrow_array::ArrayRef,
    ])
    .expect("nested batch");

    let file_path = format!("{}/data/{file_name}", table.metadata().location());
    let output = table.file_io().new_output(file_path).expect("output file");
    let parquet_builder = ParquetWriterBuilder::new(
        parquet::file::properties::WriterProperties::builder().build(),
        schema.clone(),
    );
    let mut writer = parquet_builder.build(output).await.expect("parquet writer");
    writer.write(&batch).await.expect("write nested rows");
    let mut builder = writer
        .close()
        .await
        .expect("close nested writer")
        .into_iter()
        .next()
        .expect("one data file builder");
    builder.content(DataContentType::Data);
    builder.partition_spec_id(0).partition(Struct::empty());
    builder.build().expect("build the nested data file")
}

async fn nested_namesake_output(table: &Table) -> Vec<(Option<i64>, i64)> {
    use arrow_array::cast::AsArray;
    use arrow_array::types::Int64Type;
    use futures::TryStreamExt;
    use parquet::arrow::async_reader::ParquetRecordBatchStreamBuilder;

    use crate::arrow::ArrowFileReader;
    use crate::io::FileMetadata;

    let mut rows = Vec::new();
    for (path, _) in live_output_paths(table).await {
        let input = table.file_io().new_input(&path).expect("input file");
        let size = input.metadata().await.expect("file metadata").size;
        let reader = ArrowFileReader::new(
            FileMetadata { size },
            input.reader().await.expect("file reader"),
        );
        let stream = ParquetRecordBatchStreamBuilder::new(reader)
            .await
            .expect("parquet stream builder")
            .build()
            .expect("parquet stream");
        let batches: Vec<arrow_array::RecordBatch> =
            stream.try_collect().await.expect("read output batches");
        for batch in batches {
            let ids = batch.column_by_name("id").expect("id column").clone();
            let nested = batch
                .column_by_name("st")
                .expect("st column")
                .as_struct()
                .column_by_name("id")
                .expect("st.id column")
                .clone();
            for row in 0..batch.num_rows() {
                rows.push((
                    (!ids.is_null(row)).then(|| ids.as_primitive::<Int64Type>().value(row)),
                    nested.as_primitive::<Int64Type>().value(row),
                ));
            }
        }
    }
    rows
}

#[tokio::test]
async fn a_nested_z_order_column_is_refused_and_never_binds_its_top_level_namesake() {
    let (catalog, _guard) = local_fs_catalog().await;
    let schema = nested_namesake_schema();
    let namespace = NamespaceIdent::new(format!("ns-{}", uuid::Uuid::new_v4()));
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("create namespace");
    let mut table = catalog
        .create_table(&namespace, TableCreation {
            name: "t".to_string(),
            location: None,
            schema,
            partition_spec: None,
            sort_order: None,
            properties: HashMap::new(),
            format_version: FormatVersion::V2,
        })
        .await
        .expect("create the nested table");

    let cases = [
        ("st.id", "No such struct field `st`.`id` in `id`, `st`, `v`"),
        ("st.d", "No such struct field `st`.`d` in `id`, `st`, `v`"),
        ("ID", "No such struct field `ID` in `id`, `st`, `v`"),
    ];
    for (column, expected) in cases {
        let error = RewriteDataFiles::new(table.clone())
            .strategy(RewriteStrategy::ZOrder(ZOrderSpec::new([column])))
            .rewrite_all(true)
            .execute(&catalog)
            .await
            .expect_err("a z-order column that is not a top-level column must be refused");
        assert_eq!(error.message(), expected, "zorder({column})");
    }

    let first = write_nested_namesake_file(&table, "in-0.parquet", &[
        (Some(30), 5),
        (Some(10), 7),
        (None, 6),
        (Some(20), 8),
    ])
    .await;
    let second = write_nested_namesake_file(&table, "in-1.parquet", &[
        (Some(70), 1),
        (Some(50), 3),
        (Some(60), 2),
        (Some(40), 4),
    ])
    .await;
    table = append_files(&catalog, &table, vec![first]).await;
    table = append_files(&catalog, &table, vec![second]).await;

    RewriteDataFiles::new(table.clone())
        .strategy(RewriteStrategy::ZOrder(ZOrderSpec::new(["id"])))
        .rewrite_all(true)
        .execute(&catalog)
        .await
        .expect("zorder(id) must bind the top-level column and rewrite the table");
    let table = catalog
        .load_table(table.identifier())
        .await
        .expect("reload");

    let rewritten = nested_namesake_output(&table).await;
    assert_eq!(
        rewritten,
        vec![
            (None, 6),
            (Some(10), 7),
            (Some(20), 8),
            (Some(30), 5),
            (Some(40), 4),
            (Some(50), 3),
            (Some(60), 2),
            (Some(70), 1)
        ],
        "zorder(id) must z-order the TOP-LEVEL id; the nested st.id order is its reverse and must not appear"
    );
}

#[test]
fn zorder_encodes_a_boolean_like_javas_udf_and_a_null_boolean_as_zero_bytes() {
    use arrow_array::{BooleanArray, Int64Array, RecordBatch};

    let schema = Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            Arc::new(NestedField::optional(
                1,
                "b",
                Type::Primitive(PrimitiveType::Boolean),
            )),
            Arc::new(NestedField::optional(
                2,
                "id",
                Type::Primitive(PrimitiveType::Long),
            )),
        ])
        .build()
        .expect("build the boolean schema");
    let arrow_schema =
        Arc::new(crate::arrow::schema_to_arrow_schema(&schema).expect("arrow schema"));
    let batch = RecordBatch::try_new(arrow_schema.clone(), vec![
        Arc::new(BooleanArray::from(vec![Some(true), Some(false), None])) as arrow_array::ArrayRef,
        Arc::new(Int64Array::from(vec![1i64, 2, 3])) as arrow_array::ArrayRef,
    ])
    .expect("boolean batch");

    let encoder = crate::maintenance::rewrite_data_files_zorder::ZOrderEncoder::build(
        &["b".to_string()],
        &schema,
        &arrow_schema,
        8,
        i32::MAX as usize,
    )
    .expect("z encoder");
    let mut keys = vec![Vec::new(); 3];
    encoder.encode(&batch, &mut keys).expect("encode");

    let mut expected_true = vec![0u8; 8];
    expected_true[0] = 0x81;
    assert_eq!(keys[0], expected_true);
    assert_eq!(keys[1], vec![0u8; 8]);
    assert_eq!(
        keys[2],
        vec![0u8; 8],
        "a NULL boolean encodes as zero bytes; Java instead fails the job on an unboxing NPE"
    );
    assert!(
        keys[1] < keys[0],
        "false must sort before true, as in Spark"
    );
}
