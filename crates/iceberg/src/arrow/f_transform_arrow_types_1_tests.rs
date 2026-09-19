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

use arrow_array::{
    ArrayRef, BinaryArray, BinaryViewArray, Int32Array, LargeBinaryArray, LargeStringArray,
    RecordBatch, StringArray, StringViewArray, StructArray,
};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use futures::TryStreamExt;
use parquet::arrow::PARQUET_FIELD_ID_META_KEY;
use tempfile::TempDir;

use crate::arrow::{
    PartitionValueCalculator, RecordBatchPartitionSplitter, arrow_struct_to_literal,
};
use crate::expr::Reference;
use crate::io::LocalFsStorageFactory;
use crate::memory::MemoryCatalogBuilder;
use crate::spec::{
    DataFileFormat, Datum, FormatVersion, Literal, NestedField, PartitionSpec, PrimitiveLiteral,
    PrimitiveType, Schema, Struct, Transform, Type,
};
use crate::table::Table;
use crate::transaction::{ApplyTransactionAction, Transaction};
use crate::writer::base_writer::data_file_writer::DataFileWriterBuilder;
use crate::writer::file_writer::ParquetWriterBuilder;
use crate::writer::file_writer::location_generator::{
    DefaultFileNameGenerator, DefaultLocationGenerator,
};
use crate::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
use crate::writer::partitioning::PartitioningWriter;
use crate::writer::partitioning::fanout_writer::FanoutWriter;
use crate::{Catalog, CatalogBuilder, NamespaceIdent, Result, TableCreation};

const IDS: [i32; 7] = [1, 2, 3, 4, 5, 6, 7];

const BINARY_ROWS: [Option<&[u8]>; 7] = [
    Some(&[]),
    Some(&[0x01]),
    Some(&[0x01, 0x02]),
    Some(&[0x01, 0x02, 0x03]),
    Some(&[0xff, 0x00, 0xff]),
    None,
    Some(&[0xe4, 0xb8, 0xad]),
];

const STRING_ROWS: [Option<&str>; 7] = [
    Some(""),
    Some("iceberg"),
    Some("中文字"),
    Some("a中b"),
    Some("🚀"),
    None,
    Some("abcdefg"),
];

fn binary_columns() -> Vec<ArrayRef> {
    vec![
        Arc::new(BinaryArray::from(BINARY_ROWS.to_vec())),
        Arc::new(LargeBinaryArray::from(BINARY_ROWS.to_vec())),
        Arc::new(BinaryViewArray::from(BINARY_ROWS.to_vec())),
    ]
}

fn string_columns() -> Vec<ArrayRef> {
    vec![
        Arc::new(StringArray::from(STRING_ROWS.to_vec())),
        Arc::new(LargeStringArray::from(STRING_ROWS.to_vec())),
        Arc::new(StringViewArray::from(STRING_ROWS.to_vec())),
    ]
}

fn id_binary_schema() -> Schema {
    Schema::builder()
        .with_fields(vec![
            Arc::new(NestedField::required(
                1,
                "id",
                Type::Primitive(PrimitiveType::Int),
            )),
            Arc::new(NestedField::optional(
                2,
                "b",
                Type::Primitive(PrimitiveType::Binary),
            )),
        ])
        .build()
        .expect("build id/b schema")
}

fn id_string_schema() -> Schema {
    Schema::builder()
        .with_fields(vec![
            Arc::new(NestedField::required(
                1,
                "id",
                Type::Primitive(PrimitiveType::Int),
            )),
            Arc::new(NestedField::optional(
                2,
                "s",
                Type::Primitive(PrimitiveType::String),
            )),
        ])
        .build()
        .expect("build id/s schema")
}

fn batch_for(column_name: &str, column: ArrayRef) -> RecordBatch {
    let arrow_schema = Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new(column_name, column.data_type().clone(), true),
    ]));
    RecordBatch::try_new(arrow_schema, vec![
        Arc::new(Int32Array::from(IDS.to_vec())) as ArrayRef,
        column,
    ])
    .expect("build record batch")
}

fn partition_field_bytes(struct_array: &StructArray, field_name: &str) -> Vec<Option<Vec<u8>>> {
    let column = struct_array
        .column_by_name(field_name)
        .unwrap_or_else(|| panic!("partition field {field_name} missing"));
    assert_eq!(
        column.data_type(),
        &DataType::LargeBinary,
        "canonical partition type for a binary source must be LargeBinary"
    );
    column
        .as_any()
        .downcast_ref::<LargeBinaryArray>()
        .expect("LargeBinary partition column")
        .iter()
        .map(|v| v.map(|v| v.to_vec()))
        .collect()
}

fn partition_field_strings(struct_array: &StructArray, field_name: &str) -> Vec<Option<String>> {
    let column = struct_array
        .column_by_name(field_name)
        .unwrap_or_else(|| panic!("partition field {field_name} missing"));
    assert_eq!(
        column.data_type(),
        &DataType::Utf8,
        "canonical partition type for a string source must be Utf8"
    );
    column
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("Utf8 partition column")
        .iter()
        .map(|v| v.map(|v| v.to_string()))
        .collect()
}

fn partition_field_ints(struct_array: &StructArray, field_name: &str) -> Vec<Option<i32>> {
    let column = struct_array
        .column_by_name(field_name)
        .unwrap_or_else(|| panic!("partition field {field_name} missing"));
    assert_eq!(column.data_type(), &DataType::Int32);
    column
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("Int32 partition column")
        .iter()
        .collect()
}

fn binary_spec(transform: Transform) -> PartitionSpec {
    PartitionSpec::builder(Arc::new(id_binary_schema()))
        .with_spec_id(0)
        .add_partition_field("b", "b_part", transform)
        .expect("add partition field")
        .build()
        .expect("build spec")
}

fn string_spec(transform: Transform) -> PartitionSpec {
    PartitionSpec::builder(Arc::new(id_string_schema()))
        .with_spec_id(0)
        .add_partition_field("s", "s_part", transform)
        .expect("add partition field")
        .build()
        .expect("build spec")
}

fn partition_struct(spec: &PartitionSpec, schema: &Schema, batch: &RecordBatch) -> StructArray {
    let calculator = PartitionValueCalculator::try_new(spec, schema).expect("build calculator");
    let out = calculator.calculate(batch).expect("calculate partitions");
    out.as_any()
        .downcast_ref::<StructArray>()
        .expect("partition output must be a StructArray")
        .clone()
}

#[test]
fn test_calculator_truncate_binary_every_layout_canonicalizes() {
    let schema = id_binary_schema();
    let spec = binary_spec(Transform::Truncate(1));
    let expected: Vec<Option<Vec<u8>>> = vec![
        Some(vec![]),
        Some(vec![0x01]),
        Some(vec![0x01]),
        Some(vec![0x01]),
        Some(vec![0xff]),
        None,
        Some(vec![0xe4]),
    ];
    for column in binary_columns() {
        let layout = column.data_type().clone();
        let struct_array = partition_struct(&spec, &schema, &batch_for("b", column));
        assert_eq!(
            partition_field_bytes(&struct_array, "b_part"),
            expected,
            "truncate[1] partition values for {layout:?} input"
        );
    }
}

#[test]
fn test_calculator_identity_binary_every_layout_canonicalizes() {
    let schema = id_binary_schema();
    let spec = binary_spec(Transform::Identity);
    let expected: Vec<Option<Vec<u8>>> =
        BINARY_ROWS.iter().map(|v| v.map(|v| v.to_vec())).collect();
    for column in binary_columns() {
        let layout = column.data_type().clone();
        let struct_array = partition_struct(&spec, &schema, &batch_for("b", column));
        assert_eq!(
            partition_field_bytes(&struct_array, "b_part"),
            expected,
            "identity partition values for {layout:?} input"
        );
    }
}

#[test]
fn test_calculator_void_binary_every_layout_canonicalizes() {
    let schema = id_binary_schema();
    let spec = PartitionSpec::builder(Arc::new(id_binary_schema()))
        .with_spec_id(0)
        .add_partition_field("b", "b_part", Transform::Truncate(1))
        .expect("add partition field")
        .add_partition_field("b", "b_void", Transform::Void)
        .expect("add void field")
        .build()
        .expect("build spec");
    for column in binary_columns() {
        let layout = column.data_type().clone();
        let struct_array = partition_struct(&spec, &schema, &batch_for("b", column));
        assert_eq!(
            partition_field_bytes(&struct_array, "b_void"),
            vec![None; 7],
            "void partition values for {layout:?} input"
        );
    }
}

#[test]
fn test_calculator_bucket_binary_every_layout() {
    let schema = id_binary_schema();
    let spec = binary_spec(Transform::Bucket(4));
    for column in binary_columns() {
        let layout = column.data_type().clone();
        let struct_array = partition_struct(&spec, &schema, &batch_for("b", column));
        assert_eq!(
            partition_field_ints(&struct_array, "b_part"),
            vec![Some(0), Some(3), Some(2), Some(0), Some(2), None, Some(2)],
            "bucket[4] partition values for {layout:?} input"
        );
    }
}

#[test]
fn test_calculator_truncate_string_every_layout_canonicalizes() {
    let schema = id_string_schema();
    let spec = string_spec(Transform::Truncate(2));
    let expected: Vec<Option<String>> = ["", "ic", "中文", "a中", "🚀", "", "ab"]
        .into_iter()
        .enumerate()
        .map(|(i, s)| if i == 5 { None } else { Some(s.to_string()) })
        .collect();
    for column in string_columns() {
        let layout = column.data_type().clone();
        let struct_array = partition_struct(&spec, &schema, &batch_for("s", column));
        assert_eq!(
            partition_field_strings(&struct_array, "s_part"),
            expected,
            "truncate[2] partition values for {layout:?} input"
        );
    }
}

#[test]
fn test_calculator_identity_string_every_layout_canonicalizes() {
    let schema = id_string_schema();
    let spec = string_spec(Transform::Identity);
    let expected: Vec<Option<String>> = STRING_ROWS
        .iter()
        .map(|v| v.map(|v| v.to_string()))
        .collect();
    for column in string_columns() {
        let layout = column.data_type().clone();
        let struct_array = partition_struct(&spec, &schema, &batch_for("s", column));
        assert_eq!(
            partition_field_strings(&struct_array, "s_part"),
            expected,
            "identity partition values for {layout:?} input"
        );
    }
}

#[test]
fn test_calculator_bucket_string_every_layout() {
    let schema = id_string_schema();
    let spec = string_spec(Transform::Bucket(16));
    for column in string_columns() {
        let layout = column.data_type().clone();
        let struct_array = partition_struct(&spec, &schema, &batch_for("s", column));
        assert_eq!(
            partition_field_ints(&struct_array, "s_part"),
            vec![Some(0), Some(9), Some(10), Some(11), Some(5), None, Some(6)],
            "bucket[16] partition values for {layout:?} input"
        );
    }
}

fn key_partition_literal(key: &crate::spec::PartitionKey) -> Option<Vec<u8>> {
    match key.data().fields().first() {
        Some(Some(Literal::Primitive(PrimitiveLiteral::Binary(bytes)))) => Some(bytes.clone()),
        Some(None) => None,
        other => panic!("unexpected partition literal: {other:?}"),
    }
}

#[test]
fn test_splitter_truncate_binary_every_layout() {
    let schema = Arc::new(id_binary_schema());
    let spec = Arc::new(binary_spec(Transform::Truncate(1)));
    let expected_groups: [(Option<Vec<u8>>, Vec<i32>); 5] = [
        (Some(vec![]), vec![1]),
        (Some(vec![0x01]), vec![2, 3, 4]),
        (Some(vec![0xff]), vec![5]),
        (None, vec![6]),
        (Some(vec![0xe4]), vec![7]),
    ];
    for column in binary_columns() {
        let layout = column.data_type().clone();
        let splitter = RecordBatchPartitionSplitter::try_new_with_computed_values(
            schema.clone(),
            spec.clone(),
        )
        .expect("build splitter");
        let groups = splitter
            .split(&batch_for("b", column))
            .unwrap_or_else(|e| panic!("split must accept {layout:?}: {e}"));
        let mut actual: Vec<(Option<Vec<u8>>, Vec<i32>)> = groups
            .iter()
            .map(|(key, batch)| {
                let ids: Vec<i32> = batch
                    .column_by_name("id")
                    .expect("id column")
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .expect("Int32 id column")
                    .iter()
                    .map(|v| v.expect("id is required"))
                    .collect();
                (key_partition_literal(key), ids)
            })
            .collect();
        actual.sort_by(|a, b| a.1[0].cmp(&b.1[0]));
        assert_eq!(actual, expected_groups.to_vec(), "groups for {layout:?}");
    }
}

#[test]
fn test_arrow_struct_to_literal_view_layouts() {
    let binary_expected: Vec<Option<Literal>> = BINARY_ROWS
        .iter()
        .map(|v| {
            Some(Literal::Struct(Struct::from_iter([v.map(|v| {
                Literal::Primitive(PrimitiveLiteral::Binary(v.to_vec()))
            })])))
        })
        .collect();
    let string_expected: Vec<Option<Literal>> = STRING_ROWS
        .iter()
        .map(|v| {
            Some(Literal::Struct(Struct::from_iter([v.map(|v| {
                Literal::Primitive(PrimitiveLiteral::String(v.to_string()))
            })])))
        })
        .collect();
    for (schema, spec, leaf, field_name, expected) in [
        (
            id_binary_schema(),
            binary_spec(Transform::Identity),
            Arc::new(BinaryViewArray::from(BINARY_ROWS.to_vec())) as ArrayRef,
            "b_part",
            binary_expected,
        ),
        (
            id_string_schema(),
            string_spec(Transform::Identity),
            Arc::new(StringViewArray::from(STRING_ROWS.to_vec())) as ArrayRef,
            "s_part",
            string_expected,
        ),
    ] {
        let partition_type = spec.partition_type(&schema).expect("partition type");
        let field_id = partition_type.fields()[0].id;
        let field = Arc::new(
            Field::new(field_name, leaf.data_type().clone(), true).with_metadata(HashMap::from([
                (PARQUET_FIELD_ID_META_KEY.to_string(), field_id.to_string()),
            ])),
        );
        let layout = leaf.data_type().clone();
        let struct_array: ArrayRef = Arc::new(
            StructArray::try_new(vec![field].into(), vec![leaf], None).expect("struct array"),
        );
        let literals = arrow_struct_to_literal(&struct_array, &partition_type)
            .unwrap_or_else(|e| panic!("arrow_struct_to_literal must read a {layout:?} leaf: {e}"));
        assert_eq!(
            literals, expected,
            "decoded values for {layout:?}: empty vs NULL vs bytes must round-trip"
        );
    }
}

async fn local_fs_catalog() -> (impl Catalog, TempDir) {
    let temp_dir = TempDir::new().expect("temp dir");
    let warehouse = temp_dir
        .path()
        .to_str()
        .expect("utf8 temp path")
        .to_string();
    let catalog = MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(LocalFsStorageFactory))
        .load(
            "memory",
            HashMap::from([("warehouse".to_string(), warehouse)]),
        )
        .await
        .expect("load local-fs memory catalog");
    (catalog, temp_dir)
}

async fn create_table(catalog: &impl Catalog, format_version: FormatVersion) -> Table {
    let namespace = NamespaceIdent::new(format!("ns-{}", uuid::Uuid::new_v4()));
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("create namespace");
    let schema = id_binary_schema();
    let spec = binary_spec(Transform::Truncate(1));
    let creation = TableCreation::builder()
        .name("t".to_string())
        .schema(schema)
        .format_version(format_version)
        .partition_spec(spec)
        .build();
    catalog
        .create_table(&namespace, creation)
        .await
        .expect("create table")
}

async fn write_computed_files(table: &Table, batch: &RecordBatch) -> Vec<crate::spec::DataFile> {
    let schema = table.metadata().current_schema().clone();
    let spec = table.metadata().default_partition_spec().clone();
    let splitter = RecordBatchPartitionSplitter::try_new_with_computed_values(schema.clone(), spec)
        .expect("build splitter");
    let location_generator =
        DefaultLocationGenerator::new(table.metadata().clone()).expect("location generator");
    let file_name_generator = DefaultFileNameGenerator::new(
        "data".to_string(),
        Some(uuid::Uuid::now_v7().to_string()),
        DataFileFormat::Parquet,
    );
    let parquet_builder = ParquetWriterBuilder::new(
        parquet::file::properties::WriterProperties::builder().build(),
        schema.clone(),
    );
    let rolling = RollingFileWriterBuilder::new_with_default_file_size(
        parquet_builder,
        table.file_io().clone(),
        location_generator,
        file_name_generator,
    );
    let mut writer = FanoutWriter::new(DataFileWriterBuilder::new(rolling));
    for (partition_key, partition_batch) in splitter.split(batch).expect("split batch") {
        writer
            .write(partition_key, partition_batch)
            .await
            .expect("write partition batch");
    }
    writer.close().await.expect("close fanout writer")
}

async fn append_files(
    catalog: &impl Catalog,
    table: &Table,
    files: Vec<crate::spec::DataFile>,
) -> Table {
    let tx = Transaction::new(table);
    let action = tx.fast_append().add_data_files(files);
    let tx = action.apply(tx).expect("apply fast append");
    tx.commit(catalog).await.expect("commit fast append")
}

async fn large_binary_partitioned_write_scan(format_version: FormatVersion) -> Result<()> {
    let (catalog, _tmp) = local_fs_catalog().await;
    let table = create_table(&catalog, format_version).await;

    let batch = batch_for("b", Arc::new(LargeBinaryArray::from(BINARY_ROWS.to_vec())));
    let files = write_computed_files(&table, &batch).await;
    assert_eq!(
        files.len(),
        5,
        "truncate[1] over the oracle rows must produce 5 partitions"
    );
    let mut recorded: Vec<Option<Vec<u8>>> = files
        .iter()
        .map(|file| match file.partition().fields().first() {
            Some(Some(Literal::Primitive(PrimitiveLiteral::Binary(bytes)))) => Some(bytes.clone()),
            Some(None) => None,
            other => panic!("unexpected partition tuple: {other:?}"),
        })
        .collect();
    recorded.sort();
    assert_eq!(recorded, vec![
        None,
        Some(vec![]),
        Some(vec![0x01]),
        Some(vec![0xe4]),
        Some(vec![0xff]),
    ]);
    let table = append_files(&catalog, &table, files).await;

    let mut all_ids: Vec<i32> = Vec::new();
    for batch in table
        .scan()
        .select(["id"])
        .build()
        .expect("build scan")
        .to_arrow()
        .await
        .expect("scan to arrow")
        .try_collect::<Vec<RecordBatch>>()
        .await
        .expect("collect batches")
    {
        all_ids.extend(
            batch
                .column_by_name("id")
                .expect("id column")
                .as_any()
                .downcast_ref::<Int32Array>()
                .expect("Int32 id")
                .iter()
                .map(|v| v.expect("id required")),
        );
    }
    all_ids.sort();
    assert_eq!(all_ids, IDS.to_vec(), "every row must scan back");

    let predicate = Reference::new("b").equal_to(Datum::binary(vec![0x01, 0x02]));
    let task_paths: Vec<String> = table
        .scan()
        .with_filter(predicate.clone())
        .build()
        .expect("build filtered scan")
        .plan_files()
        .await
        .expect("plan files")
        .try_collect::<Vec<_>>()
        .await
        .expect("collect tasks")
        .into_iter()
        .map(|task| task.data_file_path.to_string())
        .collect();
    assert_eq!(
        task_paths.len(),
        1,
        "b = X'0102' must prune down to the single b_part = X'01' file"
    );

    let mut filtered_ids: Vec<i32> = Vec::new();
    for batch in table
        .scan()
        .with_filter(predicate)
        .build()
        .expect("build filtered scan")
        .to_arrow()
        .await
        .expect("filtered scan to arrow")
        .try_collect::<Vec<RecordBatch>>()
        .await
        .expect("collect filtered batches")
    {
        filtered_ids.extend(
            batch
                .column_by_name("id")
                .expect("id column")
                .as_any()
                .downcast_ref::<Int32Array>()
                .expect("Int32 id")
                .iter()
                .map(|v| v.expect("id required")),
        );
    }
    filtered_ids.sort();
    assert_eq!(
        filtered_ids,
        vec![3],
        "b = X'0102' must return exactly the oracle row id 3"
    );
    Ok(())
}

#[tokio::test]
async fn test_large_binary_truncate_partitioned_write_scan_v2() -> Result<()> {
    large_binary_partitioned_write_scan(FormatVersion::V2).await
}

#[tokio::test]
async fn test_large_binary_truncate_partitioned_write_scan_v3() -> Result<()> {
    large_binary_partitioned_write_scan(FormatVersion::V3).await
}

async fn not_starts_with_longer_than_width_keeps_every_partition(
    format_version: FormatVersion,
) -> Result<()> {
    let (catalog, _tmp) = local_fs_catalog().await;
    let table = create_table(&catalog, format_version).await;

    let batch = batch_for("b", Arc::new(LargeBinaryArray::from(BINARY_ROWS.to_vec())));
    let files = write_computed_files(&table, &batch).await;
    assert_eq!(files.len(), 5);
    let table = append_files(&catalog, &table, files).await;

    let predicate = Reference::new("b").not_starts_with(Datum::binary(vec![0x01, 0x02]));
    let task_count = table
        .scan()
        .with_filter(predicate.clone())
        .build()
        .expect("build filtered scan")
        .plan_files()
        .await
        .expect("plan files")
        .try_collect::<Vec<_>>()
        .await
        .expect("collect tasks")
        .len();
    assert_eq!(
        task_count, 5,
        "NOT STARTS WITH on a 2-byte literal through truncate[1] cannot prune any partition"
    );

    let mut filtered_ids: Vec<i32> = Vec::new();
    for batch in table
        .scan()
        .with_filter(predicate)
        .build()
        .expect("build filtered scan")
        .to_arrow()
        .await
        .expect("filtered scan to arrow")
        .try_collect::<Vec<RecordBatch>>()
        .await
        .expect("collect filtered batches")
    {
        filtered_ids.extend(
            batch
                .column_by_name("id")
                .expect("id column")
                .as_any()
                .downcast_ref::<Int32Array>()
                .expect("Int32 id")
                .iter()
                .map(|v| v.expect("id required")),
        );
    }
    filtered_ids.sort();
    assert_eq!(
        filtered_ids,
        vec![1, 2, 5, 6, 7],
        "rows X'', X'01', X'FF00FF', NULL, X'E4B8AD' do not start with X'0102'; X'0102' and X'010203' drop"
    );
    Ok(())
}

#[tokio::test]
async fn test_not_starts_with_binary_partitioned_write_scan_v2() -> Result<()> {
    not_starts_with_longer_than_width_keeps_every_partition(FormatVersion::V2).await
}

#[tokio::test]
async fn test_not_starts_with_binary_partitioned_write_scan_v3() -> Result<()> {
    not_starts_with_longer_than_width_keeps_every_partition(FormatVersion::V3).await
}

async fn starts_with_empty_prefix_keeps_every_partition(
    format_version: FormatVersion,
) -> Result<()> {
    let (catalog, _tmp) = local_fs_catalog().await;
    let table = create_table(&catalog, format_version).await;

    let batch = batch_for("b", Arc::new(LargeBinaryArray::from(BINARY_ROWS.to_vec())));
    let files = write_computed_files(&table, &batch).await;
    assert_eq!(files.len(), 5);
    let table = append_files(&catalog, &table, files).await;

    let predicate = Reference::new("b").starts_with(Datum::binary(vec![]));
    let task_count = table
        .scan()
        .with_filter(predicate.clone())
        .build()
        .expect("build filtered scan")
        .plan_files()
        .await
        .expect("plan files")
        .try_collect::<Vec<_>>()
        .await
        .expect("collect tasks")
        .len();
    assert_eq!(
        task_count, 4,
        "STARTS WITH on an empty literal through truncate[1] keeps every non-null partition"
    );

    let mut filtered_ids: Vec<i32> = Vec::new();
    for batch in table
        .scan()
        .with_filter(predicate)
        .build()
        .expect("build filtered scan")
        .to_arrow()
        .await
        .expect("filtered scan to arrow")
        .try_collect::<Vec<RecordBatch>>()
        .await
        .expect("collect filtered batches")
    {
        filtered_ids.extend(
            batch
                .column_by_name("id")
                .expect("id column")
                .as_any()
                .downcast_ref::<Int32Array>()
                .expect("Int32 id")
                .iter()
                .map(|v| v.expect("id required")),
        );
    }
    filtered_ids.sort();
    assert_eq!(
        filtered_ids,
        vec![1, 2, 3, 4, 5, 7],
        "every non-null row starts with the empty prefix; the NULL row drops"
    );
    Ok(())
}

#[tokio::test]
async fn test_starts_with_binary_partitioned_write_scan_v2() -> Result<()> {
    starts_with_empty_prefix_keeps_every_partition(FormatVersion::V2).await
}

#[tokio::test]
async fn test_starts_with_binary_partitioned_write_scan_v3() -> Result<()> {
    starts_with_empty_prefix_keeps_every_partition(FormatVersion::V3).await
}
