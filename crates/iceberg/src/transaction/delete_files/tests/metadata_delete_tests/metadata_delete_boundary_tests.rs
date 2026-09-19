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

use super::super::append_files;
use super::{
    assert_decision, counted_file, make_oracle_table_in_catalog, make_table_in_catalog,
    oracle_file1, oracle_file2,
};
use crate::expr::Reference;
use crate::memory::tests::new_memory_catalog;
use crate::spec::{
    DataContentType, DataFile, Datum, FormatVersion, NestedField, PrimitiveType, Schema, Struct,
    StructType, Type,
};
use crate::table::Table;

fn nested_oracle_schema() -> Schema {
    Schema::builder()
        .with_fields(vec![
            NestedField::optional(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::optional(
                2,
                "s",
                Type::Struct(StructType::new(vec![
                    NestedField::optional(3, "x", Type::Primitive(PrimitiveType::Int)).into(),
                ])),
            )
            .into(),
        ])
        .build()
        .expect("the nested oracle schema builds")
}

async fn write_oracle_data_file(
    table: &Table,
    file_name: &str,
    rows: &[(i32, &str, &str)],
) -> DataFile {
    use std::sync::Arc;

    use arrow_array::{ArrayRef, Int32Array, RecordBatch, StringArray};

    use crate::arrow::schema_to_arrow_schema;
    use crate::writer::file_writer::{FileWriter, FileWriterBuilder, ParquetWriterBuilder};

    let schema = table.metadata().current_schema();
    let arrow_schema = Arc::new(schema_to_arrow_schema(schema).unwrap());
    let batch = RecordBatch::try_new(arrow_schema, vec![
        Arc::new(Int32Array::from(
            rows.iter().map(|(id, _, _)| *id).collect::<Vec<_>>(),
        )) as ArrayRef,
        Arc::new(StringArray::from(
            rows.iter().map(|(_, cat, _)| *cat).collect::<Vec<_>>(),
        )) as ArrayRef,
        Arc::new(StringArray::from(
            rows.iter().map(|(_, _, v)| *v).collect::<Vec<_>>(),
        )) as ArrayRef,
    ])
    .expect("the record batch builds");

    let file_path = format!("{}/data/{}", table.metadata().location(), file_name);
    let output = table.file_io().new_output(file_path).unwrap();
    let parquet_builder = ParquetWriterBuilder::new(
        parquet::file::properties::WriterProperties::builder().build(),
        schema.clone(),
    );
    let mut writer = parquet_builder.build(output).await.unwrap();
    writer.write(&batch).await.unwrap();
    let mut builder = writer.close().await.unwrap().into_iter().next().unwrap();
    builder
        .content(DataContentType::Data)
        .partition_spec_id(table.metadata().default_partition_spec_id())
        .partition(Struct::empty())
        .build()
        .expect("written data file builds")
}

#[tokio::test]
async fn can_delete_using_metadata_string_lt_bmp_upper_below_supplementary() {
    let catalog = new_memory_catalog().await;
    let table = make_oracle_table_in_catalog(&catalog, FormatVersion::V2, None).await;
    let file =
        write_oracle_data_file(&table, "test/unicode1.parquet", &[(1, "x", "\u{FFFF}")]).await;
    let table = append_files(&catalog, &table, vec![file]).await;

    assert_decision(
        &table,
        &Reference::new("v").less_than(Datum::string("\u{10000}")),
        None,
        true,
        "v < U+10000 on bounds [U+FFFF,U+FFFF]: the measured utf16_delete_truth cell — Java's CharSeqComparator is code-point order (U+FFFF < U+10000), every row is provably below the literal; UTF-16 order would answer false",
    )
    .await;
}

#[tokio::test]
async fn can_delete_using_metadata_string_lt_mixed_bounds_below_supplementary() {
    let catalog = new_memory_catalog().await;
    let table = make_oracle_table_in_catalog(&catalog, FormatVersion::V2, None).await;
    let file = write_oracle_data_file(&table, "test/unicode2.parquet", &[
        (1, "x", "\u{FFFF}"),
        (2, "x", "\u{10000}"),
    ])
    .await;
    let table = append_files(&catalog, &table, vec![file]).await;

    assert_decision(
        &table,
        &Reference::new("v").less_than(Datum::string("\u{10001}")),
        None,
        true,
        "v < U+10001 on bounds [U+FFFF,U+10000]: the measured cell — upper U+10000 < U+10001 in code-point order, every row provably below the literal",
    )
    .await;
}

#[tokio::test]
async fn can_delete_using_metadata_string_gt_bmp_upper_above_pua() {
    let catalog = new_memory_catalog().await;
    let table = make_oracle_table_in_catalog(&catalog, FormatVersion::V2, None).await;
    let file =
        write_oracle_data_file(&table, "test/unicode3.parquet", &[(1, "x", "\u{FFFF}")]).await;
    let table = append_files(&catalog, &table, vec![file]).await;

    assert_decision(
        &table,
        &Reference::new("v").greater_than(Datum::string("\u{E000}")),
        None,
        true,
        "v > U+E000 on bounds [U+FFFF,U+FFFF]: the measured cell — lower U+FFFF > U+E000, every row provably above the literal",
    )
    .await;
}

#[tokio::test]
async fn can_delete_using_metadata_string_gt_supplementary_above_bmp_max() {
    let catalog = new_memory_catalog().await;
    let table = make_oracle_table_in_catalog(&catalog, FormatVersion::V2, None).await;
    let file =
        write_oracle_data_file(&table, "test/unicode4.parquet", &[(1, "x", "\u{10000}")]).await;
    let table = append_files(&catalog, &table, vec![file]).await;

    assert_decision(
        &table,
        &Reference::new("v").greater_than(Datum::string("\u{FFFF}")),
        None,
        true,
        "v > U+FFFF on bounds [U+10000,U+10000]: the measured cell — lower U+10000 > U+FFFF in code-point order; UTF-16 order would answer false",
    )
    .await;
}

#[tokio::test]
async fn can_delete_using_metadata_string_not_eq_pua_below_range() {
    let catalog = new_memory_catalog().await;
    let table = make_oracle_table_in_catalog(&catalog, FormatVersion::V2, None).await;
    let file = write_oracle_data_file(&table, "test/unicode5.parquet", &[
        (1, "x", "\u{FFFF}"),
        (2, "x", "\u{10000}"),
    ])
    .await;
    let table = append_files(&catalog, &table, vec![file]).await;

    assert_decision(
        &table,
        &Reference::new("v").not_equal_to(Datum::string("\u{E000}")),
        None,
        true,
        "v <> U+E000 on bounds [U+FFFF,U+10000]: the measured cell — lower U+FFFF > U+E000, no row can equal the literal",
    )
    .await;
}

#[tokio::test]
async fn can_delete_using_metadata_starts_with_supplementary_prefix_is_vacuous() {
    let catalog = new_memory_catalog().await;
    let table = make_oracle_table_in_catalog(&catalog, FormatVersion::V2, None).await;
    let file = write_oracle_data_file(&table, "test/unicode6.parquet", &[
        (1, "x", "\u{E000}"),
        (2, "x", "\u{FFFF}"),
    ])
    .await;
    let table = append_files(&catalog, &table, vec![file]).await;

    assert_decision(
        &table,
        &Reference::new("v").starts_with(Datum::string("\u{10000}")),
        None,
        true,
        "v STARTS WITH U+10000 on bounds [U+E000,U+FFFF]: the verified comparator puts every bound below the supplementary prefix, so the inclusive arm prunes the file and the decision is vacuously true",
    )
    .await;
}

#[tokio::test]
async fn can_delete_using_metadata_not_starts_with_supplementary_prefix() {
    let catalog = new_memory_catalog().await;
    let table = make_oracle_table_in_catalog(&catalog, FormatVersion::V2, None).await;
    let file = write_oracle_data_file(&table, "test/unicode7.parquet", &[
        (1, "x", "\u{E000}"),
        (2, "x", "\u{FFFF}"),
    ])
    .await;
    let table = append_files(&catalog, &table, vec![file]).await;

    assert_decision(
        &table,
        &Reference::new("v").not_starts_with(Datum::string("\u{10000}")),
        None,
        true,
        "v NOT STARTS WITH U+10000 on bounds [U+E000,U+FFFF]: strict notStartsWith must-match — the truncated upper U+FFFF compares below the supplementary prefix in code-point order",
    )
    .await;
}

#[tokio::test]
async fn can_delete_using_metadata_not_starts_with_order_boundary() {
    let catalog = new_memory_catalog().await;
    let table = make_oracle_table_in_catalog(&catalog, FormatVersion::V2, None).await;
    let file = write_oracle_data_file(&table, "test/unicode8.parquet", &[
        (1, "x", "\u{E000}"),
        (2, "x", "\u{10000}z"),
    ])
    .await;
    let table = append_files(&catalog, &table, vec![file]).await;

    assert_decision(
        &table,
        &Reference::new("v").not_starts_with(Datum::string("\u{FFFF}")),
        None,
        false,
        "v NOT STARTS WITH U+FFFF on bounds [U+E000,U+10000z]: the truncated upper compares as the supplementary code point U+10000, which Java's comparator places ABOVE U+FFFF, so no arm proves the file — a UTF-16 unit order would compare the leading surrogate below U+FFFF and wrongly answer true",
    )
    .await;
}

#[tokio::test]
async fn can_delete_using_metadata_nested_field_metrics_are_not_provable() {
    let catalog = new_memory_catalog().await;
    let table =
        make_table_in_catalog(&catalog, FormatVersion::V2, None, nested_oracle_schema()).await;
    let file = counted_file("test/nested.parquet", 0, Struct::empty(), 1, &[(3, 0)], &[
        (3, Datum::int(7), Datum::int(7)),
    ]);
    let table = append_files(&catalog, &table, vec![file]).await;

    assert_decision(
        &table,
        &Reference::new("s.x").equal_to(Datum::int(7)),
        None,
        false,
        "s.x = 7 on nested bounds [7,7]: Java isNestedColumn (struct.field(id) == null) makes every strict-metrics arm MIGHT_NOT_MATCH for non-top-level field ids — the fork must not prove a nested field from column metrics",
    )
    .await;
}

#[tokio::test]
async fn can_delete_using_metadata_short_circuits_before_unreadable_manifest() {
    let catalog = new_memory_catalog().await;
    let table = make_oracle_table_in_catalog(&catalog, FormatVersion::V2, None).await;
    let table = append_files(&catalog, &table, vec![oracle_file2()]).await;
    let table = append_files(&catalog, &table, vec![oracle_file1()]).await;

    let snapshot = table
        .metadata()
        .current_snapshot()
        .expect("two appends leave a current snapshot");
    let manifest_list = snapshot
        .load_manifest_list(table.file_io(), table.metadata())
        .await
        .expect("the manifest list loads");
    assert_eq!(
        manifest_list.entries().len(),
        2,
        "each append produced one manifest"
    );
    let last_manifest = manifest_list.entries().last().expect("two manifests");
    table
        .file_io()
        .delete(&last_manifest.manifest_path)
        .await
        .expect("the second manifest deletes");

    let decision = table
        .can_delete_using_metadata(
            &Reference::new("id").greater_than_or_equal_to(Datum::int(2)),
            None,
            true,
        )
        .await;
    assert!(
        matches!(decision, Ok(false)),
        "id >= 2 on file1 [1,3] is an unproven candidate in the FIRST manifest of the list (newest first): the walk must answer Ok(false) without opening the deleted second manifest — a collect-then-decide walk errors instead, got {decision:?}"
    );
}
